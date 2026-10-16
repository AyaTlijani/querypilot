"""LLM clients: local open-weight models through Ollama (default, free),
the Mistral API (optional), and a scripted fake for tests and offline demos.

All expose the same `chat(messages, tools=None, model=...)` method and return
an `LLMReply`, so the agent never knows which one it's talking to.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("QP_PROVIDER", "ollama").lower()   # "ollama" or "mistral"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
NUM_CTX = int(os.getenv("QP_NUM_CTX", "16384"))  # BIRD schemas need a long context

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

# USD per 1M tokens (input, output). Verify against https://mistral.ai/pricing
# before publishing numbers; override with env vars if prices change.
PRICES = {
    "mistral-large-latest": (
        float(os.getenv("PRICE_LARGE_IN", "0.5")), float(os.getenv("PRICE_LARGE_OUT", "1.5"))),
    "mistral-small-latest": (
        float(os.getenv("PRICE_SMALL_IN", "0.1")), float(os.getenv("PRICE_SMALL_OUT", "0.3"))),
}

_DEFAULTS = {"ollama": ("qwen2.5-coder:7b", "qwen2.5-coder:3b"),
             "mistral": ("mistral-large-latest", "mistral-small-latest"),
             "gemini": ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite")}
if PROVIDER == "gemini":   # its own variables, so a local-model .env doesn't interfere
    LARGE = os.getenv("QP_GEMINI_LARGE", _DEFAULTS["gemini"][0])
    SMALL = os.getenv("QP_GEMINI_SMALL", _DEFAULTS["gemini"][1])
else:
    LARGE = os.getenv("QP_LARGE_MODEL", _DEFAULTS.get(PROVIDER, _DEFAULTS["ollama"])[0])
    SMALL = os.getenv("QP_SMALL_MODEL", _DEFAULTS.get(PROVIDER, _DEFAULTS["ollama"])[1])

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


class QuotaExhausted(RuntimeError):
    """The provider's daily quota is used up: stop the run, resume later."""


@dataclass
class LLMReply:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{id, name, arguments(dict)}]
    model: str = ""            # the exact model version that answered
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def as_message(self) -> dict:
        msg = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                for c in self.tool_calls
            ]
        return msg


_TOOL_NAMES = {"list_tables", "describe_table", "sample_rows", "run_sql", "submit_sql"}


def parse_text_tool_calls(text: str) -> list[dict]:
    """Small local models often write a tool call as JSON text, e.g.
    ```json {"name": "run_sql", "arguments": {"sql": "..."}} ``` or inside
    <tool_call> tags, instead of using the structured tool-call field.
    Recover those calls so the agent loop works the same for every model."""
    if not text or '"name"' not in text:
        return []
    decoder, calls, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and obj.get("name") in _TOOL_NAMES:
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"id": f"text_call_{len(calls)}", "name": obj["name"],
                          "arguments": args if isinstance(args, dict) else {}})
        i = end
    return calls


def cost_of(model_alias: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model_alias not in PRICES:
        return 0.0  # local models cost nothing per call
    p_in, p_out = PRICES[model_alias]
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000


class MistralClient:
    def __init__(self, api_key: str | None = None, max_retries: int = 4):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set (put it in a .env file).")
        self.max_retries = max_retries

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             model: str = LARGE, temperature: float = 0.0) -> LLMReply:
        body = {"model": model, "messages": messages, "temperature": temperature}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        start = time.monotonic()
        for attempt in range(self.max_retries + 1):
            resp = requests.post(MISTRAL_URL, json=body, timeout=120,
                                 headers={"Authorization": f"Bearer {self.api_key}"})
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                time.sleep(2 ** attempt * 2)
                continue
            resp.raise_for_status()
            break
        data = resp.json()
        choice = data["choices"][0]["message"]
        calls = []
        for c in choice.get("tool_calls") or []:
            args = c["function"].get("arguments") or "{}"
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                args = {"_raw": args}
            calls.append({"id": c.get("id") or f"call_{len(calls)}",
                          "name": c["function"]["name"], "arguments": args})
        usage = data.get("usage", {})
        pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        return LLMReply(
            content=choice.get("content") or "", tool_calls=calls,
            model=data.get("model", model), prompt_tokens=pt, completion_tokens=ct,
            cost_usd=cost_of(model, pt, ct), latency_s=time.monotonic() - start,
        )


class OllamaClient:
    """Local models through Ollama's native chat API (supports tool calling).
    Uses /api/chat rather than the OpenAI-compatible endpoint so the context
    window can be set: Ollama's default is too short for BIRD schemas and
    would silently cut the prompt."""

    def __init__(self, url: str = OLLAMA_URL, num_ctx: int = NUM_CTX):
        self.url, self.num_ctx = url.rstrip("/"), num_ctx
        try:
            requests.get(f"{self.url}/api/tags", timeout=5).raise_for_status()
        except requests.RequestException as err:
            raise RuntimeError("Ollama is not reachable. Is the Ollama app running?") from err

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        """OpenAI-style history -> Ollama format (arguments as objects, tool names on results)."""
        out, names = [], {}
        for m in messages:
            m = dict(m)
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    args = c["function"]["arguments"]
                    names[c.get("id")] = c["function"]["name"]
                    calls.append({"function": {"name": c["function"]["name"],
                                               "arguments": json.loads(args) if isinstance(args, str) else args}})
                m["tool_calls"] = calls
            if m["role"] == "tool":
                m = {"role": "tool", "content": m["content"],
                     "tool_name": m.get("name") or names.get(m.get("tool_call_id"), "")}
            out.append(m)
        return out

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             model: str = LARGE, temperature: float = 0.0) -> LLMReply:
        body = {"model": model, "messages": self._convert(messages), "stream": False,
                "options": {"temperature": temperature, "num_ctx": self.num_ctx, "seed": 0}}
        if tools:
            body["tools"] = tools
        start = time.monotonic()
        resp = requests.post(f"{self.url}/api/chat", json=body, timeout=600)
        if resp.status_code == 404:
            raise RuntimeError(f"Model '{model}' is not installed. Run: ollama pull {model}")
        if resp.status_code == 400 and "support tools" in resp.text:
            raise RuntimeError(f"Model '{model}' can't use tools in Ollama. "
                               "Pick a tool-capable model (e.g. qwen2.5-coder) in .env.")
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append({"id": f"call_{i}_{int(time.time() * 1000) % 100000}",
                          "name": fn.get("name", ""), "arguments": args})
        content = msg.get("content") or ""
        if not calls and tools:
            calls = parse_text_tool_calls(content)
            if calls:
                content = ""  # the text WAS the tool call, not a thought
        return LLMReply(
            content=content, tool_calls=calls, model=data.get("model", model),
            prompt_tokens=data.get("prompt_eval_count", 0), completion_tokens=data.get("eval_count", 0),
            cost_usd=0.0, latency_s=time.monotonic() - start,
        )


class GeminiClient:
    """Google Gemini through its OpenAI-compatible endpoint, paced for the free tier.

    Free-tier limits are per project and change over time; check yours at
    https://aistudio.google.com/rate-limit and set QP_GEMINI_RPM accordingly.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in your .env file).")
        self.min_gap = 60.0 / float(os.getenv("QP_GEMINI_RPM", "14"))
        self._last = 0.0

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             model: str = LARGE, temperature: float = 0.0) -> LLMReply:
        body = {"model": model, "messages": [{"role": m["role"], "content": m.get("content") or ""}
                                             for m in messages], "temperature": temperature}
        start = time.monotonic()
        for attempt in range(6):
            wait = self._last + self.min_gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)          # stay under the requests-per-minute limit
            self._last = time.monotonic()
            resp = requests.post(GEMINI_URL, json=body, timeout=120,
                                 headers={"Authorization": f"Bearer {self.api_key}"})
            if resp.status_code == 429:
                text = resp.text.lower()
                if "per day" in text or "perday" in text or "daily" in text:
                    raise QuotaExhausted("Gemini daily quota reached. Run the same command "
                                         "again after the reset (midnight Pacific time).")
                time.sleep(min(60, 10 * (attempt + 1)))
                continue
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            if resp.status_code == 404:
                raise RuntimeError(f"Gemini model '{model}' not found. Check the exact model name "
                                   "in Google AI Studio and set QP_GEMINI_LARGE in .env.")
            resp.raise_for_status()
            break
        else:
            raise RuntimeError("Gemini kept refusing requests (rate limit). Try again later.")
        data = resp.json()
        usage = data.get("usage", {})
        content = data["choices"][0]["message"].get("content") or ""
        return LLMReply(content=content, model=data.get("model", model),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        cost_usd=0.0, latency_s=time.monotonic() - start)


def make_client():
    """The configured model: local Ollama by default; QP_PROVIDER=gemini or mistral for APIs."""
    if PROVIDER == "gemini":
        return GeminiClient()
    return MistralClient() if PROVIDER == "mistral" else OllamaClient()


class FakeLLM:
    """Replays scripted replies. Used by tests, and by the demo when no key is set."""

    def __init__(self, script: list[LLMReply] | None = None, responder=None):
        self.script = list(script or [])
        self.responder = responder  # optional fn(messages, tools, model) -> LLMReply
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, model=LARGE, temperature=0.0) -> LLMReply:
        self.calls.append({"model": model, "messages": messages, "tools": tools})
        if self.responder:
            reply = self.responder(messages, tools, model)
        elif self.script:
            reply = self.script.pop(0)
        else:
            reply = LLMReply(content="I could not answer.")
        reply.model = reply.model or f"fake:{model}"
        return reply
