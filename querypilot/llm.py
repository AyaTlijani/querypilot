"""Model clients: local Ollama (default), Gemini and Mistral APIs, and a fake model for tests."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("QP_PROVIDER", "ollama").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
NUM_CTX = int(os.getenv("QP_NUM_CTX", "16384"))
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

PRICES = {
    "mistral-large-latest": (
        float(os.getenv("PRICE_LARGE_IN", "0.5")), float(os.getenv("PRICE_LARGE_OUT", "1.5"))),
    "mistral-small-latest": (
        float(os.getenv("PRICE_SMALL_IN", "0.1")), float(os.getenv("PRICE_SMALL_OUT", "0.3"))),
}

_DEFAULTS = {"ollama": ("qwen2.5-coder:7b", "qwen2.5-coder:1.5b"),
             "mistral": ("mistral-large-latest", "mistral-small-latest"),
             "gemini": ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite")}
if PROVIDER == "gemini":
    LARGE = os.getenv("QP_GEMINI_LARGE", _DEFAULTS["gemini"][0])
    SMALL = os.getenv("QP_GEMINI_SMALL", _DEFAULTS["gemini"][1])
else:
    LARGE = os.getenv("QP_LARGE_MODEL", _DEFAULTS.get(PROVIDER, _DEFAULTS["ollama"])[0])
    SMALL = os.getenv("QP_SMALL_MODEL", _DEFAULTS.get(PROVIDER, _DEFAULTS["ollama"])[1])


class StopRun(RuntimeError):
    """Raised when the model can't be reached, so a benchmark pauses instead of scoring wrong answers."""


class QuotaExhausted(StopRun):
    """Raised when an API's daily quota is used up."""


@dataclass
class LLMReply:
    """One model answer, with its token counts, cost and time."""
    content: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return the dollar cost of a call (zero for local and free models)."""
    if model not in PRICES:
        return 0.0
    price_in, price_out = PRICES[model]
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


class OllamaClient:
    """Talks to local models through Ollama."""

    def __init__(self, url: str = OLLAMA_URL, num_ctx: int = NUM_CTX):
        """Connect to Ollama and fail early if it isn't running."""
        self.url, self.num_ctx = url.rstrip("/"), num_ctx
        try:
            requests.get(f"{self.url}/api/tags", timeout=5).raise_for_status()
        except requests.RequestException as err:
            raise RuntimeError("Ollama is not reachable. Is the Ollama app running?") from err

    def chat(self, messages: list[dict], model: str = LARGE, temperature: float = 0.0) -> LLMReply:
        """Send a conversation to a local model and return its reply."""
        body = {"model": model, "messages": messages, "stream": False,
                "options": {"temperature": temperature, "num_ctx": self.num_ctx, "seed": 0}}
        start = time.monotonic()
        try:
            resp = requests.post(f"{self.url}/api/chat", json=body, timeout=600)
        except (requests.ConnectionError, requests.Timeout) as err:
            raise StopRun("Ollama stopped responding. Open the Ollama app, then run the "
                          "same command again to resume.") from err
        if resp.status_code == 404:
            raise RuntimeError(f"Model '{model}' is not installed. Run: ollama pull {model}")
        resp.raise_for_status()
        data = resp.json()
        return LLMReply(content=data.get("message", {}).get("content") or "",
                        model=data.get("model", model),
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        completion_tokens=data.get("eval_count", 0),
                        latency_s=time.monotonic() - start)


class GeminiClient:
    """Talks to Google Gemini on the free tier, staying under its rate limits."""

    def __init__(self, api_key: str | None = None):
        """Read the API key and the allowed requests per minute."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in your .env file).")
        self.min_gap = 60.0 / float(os.getenv("QP_GEMINI_RPM", "14"))
        self._last = 0.0

    def chat(self, messages: list[dict], model: str = LARGE, temperature: float = 0.0) -> LLMReply:
        """Send a conversation to Gemini, waiting and retrying when rate-limited."""
        body = {"model": model, "temperature": temperature,
                "messages": [{"role": m["role"], "content": m.get("content") or ""} for m in messages]}
        start = time.monotonic()
        for attempt in range(10):
            wait = self._last + self.min_gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = requests.post(GEMINI_URL, json=body, timeout=120,
                                     headers={"Authorization": f"Bearer {self.api_key}"})
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(min(60, 10 * (attempt + 1)))
                continue
            if resp.status_code == 429:
                text = resp.text.lower()
                if "per day" in text or "perday" in text or "daily" in text:
                    raise QuotaExhausted("Gemini daily quota reached. Run the same command "
                                         "again after the reset (midnight Pacific time).")
                hint = re.search(r'retrydelay"?\s*:\s*"?(\d+)', text)
                time.sleep(min(90, int(hint.group(1)) + 2 if hint else 15 * (attempt + 1)))
                continue
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(min(60, 10 * (attempt + 1)))
                continue
            if resp.status_code == 404:
                raise RuntimeError(f"Gemini model '{model}' not found. Check the exact model name "
                                   "in Google AI Studio and set QP_GEMINI_LARGE in .env.")
            resp.raise_for_status()
            break
        else:
            raise StopRun("Gemini kept refusing or failing requests. Run the same command "
                          "again later to resume.")
        data = resp.json()
        usage = data.get("usage", {})
        return LLMReply(content=data["choices"][0]["message"].get("content") or "",
                        model=data.get("model", model),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        latency_s=time.monotonic() - start)


class MistralClient:
    """Talks to the paid Mistral API."""

    def __init__(self, api_key: str | None = None, max_retries: int = 4):
        """Read the API key."""
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set (put it in a .env file).")
        self.max_retries = max_retries

    def chat(self, messages: list[dict], model: str = LARGE, temperature: float = 0.0) -> LLMReply:
        """Send a conversation to Mistral, retrying on temporary errors."""
        body = {"model": model, "messages": messages, "temperature": temperature}
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
        usage = data.get("usage", {})
        pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        return LLMReply(content=data["choices"][0]["message"].get("content") or "",
                        model=data.get("model", model), prompt_tokens=pt, completion_tokens=ct,
                        cost_usd=cost_of(model, pt, ct), latency_s=time.monotonic() - start)


def make_client():
    """Create the model client chosen in .env (Ollama unless QP_PROVIDER says otherwise)."""
    if PROVIDER == "gemini":
        return GeminiClient()
    if PROVIDER == "mistral":
        return MistralClient()
    return OllamaClient()


class FakeLLM:
    """Returns scripted answers, for tests and the offline demo."""

    def __init__(self, script: list[LLMReply] | None = None, responder=None):
        """Use a fixed list of replies, or a function that builds each reply."""
        self.script = list(script or [])
        self.responder = responder
        self.calls: list[dict] = []

    def chat(self, messages, model=LARGE, temperature=0.0) -> LLMReply:
        """Record the request and return the next scripted reply."""
        self.calls.append({"model": model, "messages": messages})
        if self.responder:
            reply = self.responder(messages, model)
        elif self.script:
            reply = self.script.pop(0)
        else:
            reply = LLMReply(content="I could not answer.")
        reply.model = reply.model or f"fake:{model}"
        return reply
