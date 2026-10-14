"""LLM clients: the real Mistral API, and a scripted fake for tests and offline demos.

Both expose the same `chat(messages, tools=None, model=...)` method and return
an `LLMReply`, so the agent never knows which one it's talking to.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import requests

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

# USD per 1M tokens (input, output). Verify against https://mistral.ai/pricing
# before publishing numbers; override with env vars if prices change.
PRICES = {
    "mistral-large-latest": (
        float(os.getenv("PRICE_LARGE_IN", "0.5")), float(os.getenv("PRICE_LARGE_OUT", "1.5"))),
    "mistral-small-latest": (
        float(os.getenv("PRICE_SMALL_IN", "0.1")), float(os.getenv("PRICE_SMALL_OUT", "0.3"))),
}

LARGE = os.getenv("QP_LARGE_MODEL", "mistral-large-latest")
SMALL = os.getenv("QP_SMALL_MODEL", "mistral-small-latest")


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


def cost_of(model_alias: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = PRICES.get(model_alias, PRICES[LARGE] if LARGE in PRICES else (0.5, 1.5))
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
