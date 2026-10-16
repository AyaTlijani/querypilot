from querypilot.agent import ONE_SHOT_PROMPT, Agent, one_shot, user_message
from querypilot.llm import LARGE, SMALL, FakeLLM, LLMReply
from querypilot.safe_db import SafeDB


def call(name, **args):
    return {"id": f"id_{name}", "name": name, "arguments": args}


def draft(sql):
    return LLMReply(content=f"```sql\n{sql}\n```")


def test_working_draft_is_accepted_untouched(shop):
    """A query that runs and returns rows is never rewritten: 1 call, no tools."""
    llm = FakeLLM([draft("SELECT COUNT(*) FROM customers WHERE country='France'")])
    out = Agent(llm, SafeDB(shop["db"])).run("How many customers live in France?")
    assert out["status"] == "ok" and out["rows"] == [[2]]
    assert out["calls"] == 1 and out["repairs"] == 0
    assert llm.calls[0]["tools"] is None


def test_failed_draft_triggers_tool_repair(shop):
    llm = FakeLLM([
        draft("SELECT COUNT(*) FROM customer WHERE country='France'"),   # typo -> error
        LLMReply(content="Check the table name", tool_calls=[call("list_tables")]),
        LLMReply(tool_calls=[call("submit_sql", sql="SELECT COUNT(*) FROM customers WHERE country='France'")]),
    ])
    events = []
    out = Agent(llm, SafeDB(shop["db"]), emit=events.append).run("How many customers live in France?")
    assert out["status"] == "ok" and out["rows"] == [[2]] and out["repairs"] == 1
    kinds = [e["type"] for e in events]
    assert kinds[:4] == ["start", "draft", "verify", "repair"] and "tool" in kinds and kinds[-1] == "done"
    assert all(c["model"] == LARGE for c in llm.calls)


def test_empty_draft_is_rechecked_once(shop):
    llm = FakeLLM([
        draft("SELECT name FROM customers WHERE country='france'"),        # 0 rows
        LLMReply(tool_calls=[call("submit_sql", sql="SELECT name FROM customers WHERE country='france'")]),
    ])
    out = Agent(llm, SafeDB(shop["db"])).run("Customers in France?")
    assert out["status"] == "empty" and out["calls"] == 2   # resubmitted empty answer is accepted


def test_routed_drafts_small_and_escalates_on_failure(shop):
    llm = FakeLLM([
        draft("SELECT bad FROM customers"),
        LLMReply(tool_calls=[call("submit_sql", sql="SELECT COUNT(*) FROM customers")]),
    ])
    out = Agent(llm, SafeDB(shop["db"]), routed=True).run("How many customers?")
    assert out["status"] == "ok" and out["escalated"]
    assert [c["model"] for c in llm.calls] == [SMALL, LARGE]


def test_routed_stays_small_when_draft_works(shop):
    llm = FakeLLM([draft("SELECT COUNT(*) FROM customers")])
    out = Agent(llm, SafeDB(shop["db"]), routed=True).run("How many customers?")
    assert not out["escalated"] and [c["model"] for c in llm.calls] == [SMALL]


def test_agent_gives_up_after_step_budget_and_keeps_last_attempt(shop):
    def responder(messages, tools, model):
        return draft("SELECT nope FROM customers") if not tools else LLMReply(tool_calls=[call("list_tables")])
    out = Agent(FakeLLM(responder=responder), SafeDB(shop["db"])).run("anything")
    assert out["status"] == "gave_up" and out["final_sql"] == "SELECT nope FROM customers"


def test_write_attempts_by_the_model_are_refused(shop):
    llm = FakeLLM([
        draft("DROP TABLE customers"),
        LLMReply(tool_calls=[call("run_sql", sql="DELETE FROM customers")]),
        LLMReply(tool_calls=[call("submit_sql", sql="SELECT COUNT(*) FROM customers")]),
    ])
    events = []
    Agent(llm, SafeDB(shop["db"]), emit=events.append).run("q")
    tool = [e for e in events if e["type"] == "tool"][0]
    assert not tool["ok"]
    assert SafeDB(shop["db"]).query("SELECT COUNT(*) FROM customers").rows == [(3,)]


def test_agent_draft_is_exactly_the_one_shot_request(shop):
    """Fair comparison: the agent's first call is identical to the one-shot baseline."""
    db = SafeDB(shop["db"])
    schema = db.schema_context()
    a = FakeLLM([draft("SELECT 1")])
    one_shot(a, db, "Q?", "hint", schema)
    b = FakeLLM([draft("SELECT 1")])
    Agent(b, db).run("Q?", "hint", schema)
    assert a.calls[0]["messages"] == b.calls[0]["messages"]
    assert b.calls[0]["messages"][0]["content"] == ONE_SHOT_PROMPT
    assert b.calls[0]["messages"][1]["content"] == user_message("Q?", "hint", schema)
