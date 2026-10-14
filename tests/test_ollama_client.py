"""The Ollama client can't be run here without a GPU, so its request/response
handling is tested against a mocked HTTP layer."""
import json

from querypilot import llm
from querypilot.agent import Agent
from querypilot.safe_db import SafeDB


class Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status
        self.text = str(data)

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_agent_runs_through_ollama_format(shop, monkeypatch):
    sent = []
    replies = [
        {"model": "qwen2.5-coder:7b", "prompt_eval_count": 900, "eval_count": 30,
         "message": {"content": "", "tool_calls": [
             {"function": {"name": "sample_rows", "arguments": {"table": "customers"}}}]}},
        {"model": "qwen2.5-coder:7b", "prompt_eval_count": 1000, "eval_count": 25,
         "message": {"content": "", "tool_calls": [
             {"function": {"name": "submit_sql",
                           "arguments": {"sql": "SELECT COUNT(*) FROM customers WHERE country='France'"}}}]}},
    ]
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: Resp({"models": []}))
    monkeypatch.setattr(llm.requests, "post",
                        lambda url, json=None, timeout=None: (sent.append(json), Resp(replies.pop(0)))[1])
    client = llm.OllamaClient()
    out = Agent(client, SafeDB(shop["db"])).run("How many customers live in France?")
    assert out["status"] == "ok" and out["rows"] == [[2]]
    assert out["tokens"] == 900 + 30 + 1000 + 25 and out["cost"] == 0
    second = sent[1]
    assert second["options"]["num_ctx"] >= 8192 and second["options"]["temperature"] == 0
    assistant = [m for m in second["messages"] if m.get("tool_calls")][0]
    assert isinstance(assistant["tool_calls"][0]["function"]["arguments"], dict)
    tool = [m for m in second["messages"] if m["role"] == "tool"][0]
    assert tool["tool_name"] == "sample_rows"
    json.dumps(second)  # serializable


def test_missing_model_gives_clear_message(monkeypatch):
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: Resp({}))
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: Resp({}, 404))
    try:
        llm.OllamaClient().chat([{"role": "user", "content": "hi"}], model="nope:1b")
    except RuntimeError as err:
        assert "ollama pull nope:1b" in str(err)
