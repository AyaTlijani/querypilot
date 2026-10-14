from querypilot.agent import Agent, one_shot, user_message
from querypilot.llm import LARGE, SMALL, FakeLLM, LLMReply
from querypilot.safe_db import SafeDB


def call(name, **args):
    return {"id": f"id_{name}", "name": name, "arguments": args}


def test_agent_explores_fails_repairs_and_succeeds(shop):
    llm = FakeLLM([
        LLMReply(content="Look at values", tool_calls=[call("sample_rows", table="customers")]),
        LLMReply(tool_calls=[call("submit_sql", sql="SELECT COUNT(*) FROM customer WHERE country='France'")]),
        LLMReply(content="Typo in table name", tool_calls=[call("submit_sql",
                 sql="SELECT COUNT(*) FROM customers WHERE country='France'")]),
    ])
    events = []
    out = Agent(llm, SafeDB(shop["db"]), emit=events.append).run("How many customers live in France?")
    assert out["status"] == "ok" and out["rows"] == [[2]]
    assert out["repairs"] == 1
    kinds = [e["type"] for e in events]
    assert "tool" in kinds and "repair" in kinds and kinds[-1] == "done"
    assert all(c["model"] == LARGE for c in llm.calls)


def test_routed_starts_small_and_escalates_on_failure(shop):
    llm = FakeLLM([
        LLMReply(tool_calls=[call("submit_sql", sql="SELECT bad FROM customers")]),
        LLMReply(tool_calls=[call("submit_sql", sql="SELECT COUNT(*) FROM customers")]),
    ])
    out = Agent(llm, SafeDB(shop["db"]), routed=True).run("How many customers?")
    assert out["status"] == "ok" and out["escalated"]
    assert [c["model"] for c in llm.calls] == [SMALL, LARGE]


def test_routed_stays_small_when_it_succeeds(shop):
    llm = FakeLLM([LLMReply(tool_calls=[call("submit_sql", sql="SELECT COUNT(*) FROM customers")])])
    out = Agent(llm, SafeDB(shop["db"]), routed=True).run("How many customers?")
    assert not out["escalated"] and [c["model"] for c in llm.calls] == [SMALL]


def test_agent_gives_up_after_step_budget(shop):
    looping = FakeLLM(responder=lambda m, t, model: LLMReply(tool_calls=[call("list_tables")]))
    out = Agent(looping, SafeDB(shop["db"])).run("anything")
    assert out["status"] == "gave_up"


def test_write_attempts_by_the_model_are_refused(shop):
    llm = FakeLLM([
        LLMReply(tool_calls=[call("run_sql", sql="DROP TABLE customers")]),
        LLMReply(tool_calls=[call("submit_sql", sql="SELECT COUNT(*) FROM customers")]),
    ])
    events = []
    Agent(llm, SafeDB(shop["db"]), emit=events.append).run("q")
    tool = [e for e in events if e["type"] == "tool"][0]
    assert not tool["ok"]
    assert SafeDB(shop["db"]).query("SELECT COUNT(*) FROM customers").rows == [(3,)]


def test_all_setups_receive_identical_context(shop):
    """Fair comparison: same question, evidence and schema text for every setup."""
    db = SafeDB(shop["db"])
    schema = db.schema_context()
    expected = user_message("Q?", "hint", schema)
    a = FakeLLM([LLMReply(content="```sql\nSELECT 1\n```")])
    one_shot(a, db, "Q?", "hint", schema)
    b = FakeLLM([LLMReply(tool_calls=[call("submit_sql", sql="SELECT 1")])])
    Agent(b, db).run("Q?", "hint", schema)
    assert a.calls[0]["messages"][1]["content"] == expected
    assert b.calls[0]["messages"][1]["content"] == expected
