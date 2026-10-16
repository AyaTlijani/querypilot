"""The Ollama client is tested against a mocked HTTP layer (no GPU here)."""
from querypilot import llm
from querypilot.agent import Agent
from querypilot.safe_db import SafeDB


class Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code, self.text = data, status, str(data)

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_agent_runs_through_ollama(shop, monkeypatch):
    sent = []
    replies = [
        {"model": "qwen2.5-coder:7b", "prompt_eval_count": 500, "eval_count": 20,
         "message": {"content": "```sql\nSELECT COUNT(*) FROM customer\n```"}},
        {"model": "qwen2.5-coder:7b", "prompt_eval_count": 900, "eval_count": 30,
         "message": {"content": "```sql\nSELECT COUNT(*) FROM customers WHERE country='France'\n```"}},
    ]
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: Resp({"models": []}))
    monkeypatch.setattr(llm.requests, "post",
                        lambda url, json=None, timeout=None: (sent.append(json), Resp(replies.pop(0)))[1])
    out = Agent(llm.OllamaClient(), SafeDB(shop["db"])).run("How many customers live in France?")
    assert out["status"] == "ok" and out["rows"] == [[2]]
    assert out["tokens"] == 500 + 20 + 900 + 30 and out["cost"] == 0
    assert sent[1]["options"]["num_ctx"] >= 8192 and sent[1]["options"]["temperature"] == 0
    assert "tools" not in sent[1]


def test_missing_model_gives_clear_message(monkeypatch):
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: Resp({}))
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: Resp({}, 404))
    try:
        llm.OllamaClient().chat([{"role": "user", "content": "hi"}], model="nope:1b")
        assert False
    except RuntimeError as err:
        assert "ollama pull nope:1b" in str(err)


def test_text_tool_call_parser():
    assert llm.parse_text_tool_calls('<tool_call>{"name": "list_tables", "arguments": {}}</tool_call>')[0]["name"] == "list_tables"
    assert llm.parse_text_tool_calls("I think the answer is 42.") == []
