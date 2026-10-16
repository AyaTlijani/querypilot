"""Tests for the agent's draft, repair and routing behaviour."""

from querypilot.agent import ONE_SHOT_PROMPT, REPAIR_PROMPT, Agent, diagnose, one_shot, user_message
from querypilot.llm import LARGE, SMALL, FakeLLM, LLMReply
from querypilot.safe_db import SafeDB


def sql(q):
    """Build a fake model reply containing a query."""
    return LLMReply(content=f"```sql\n{q}\n```")


def test_working_draft_is_kept_untouched(shop):
    """Check that working draft is kept untouched."""
    llm = FakeLLM([sql("SELECT COUNT(*) FROM customers WHERE country='France'")])
    out = Agent(llm, SafeDB(shop["db"])).run("How many customers live in France?")
    assert out["status"] == "ok" and out["rows"] == [[2]] and out["calls"] == 1 and out["repairs"] == 0


def test_error_is_diagnosed_and_repaired(shop):
    """Check that error is diagnosed and repaired."""
    llm = FakeLLM([sql("SELECT COUNT(*) FROM customers WHERE nation='France'"),
                   sql("SELECT COUNT(*) FROM customers WHERE country='France'")])
    events = []
    out = Agent(llm, SafeDB(shop["db"]), emit=events.append).run("How many customers live in France?")
    assert out["status"] == "ok" and out["rows"] == [[2]] and out["repairs"] == 1
    kinds = [e["type"] for e in events]
    assert kinds[:5] == ["start", "draft", "verify", "repair", "tool"] and kinds[-1] == "done"
    repair_msgs = llm.calls[1]["messages"]
    assert repair_msgs[0]["content"] == REPAIR_PROMPT
    assert "no such column" in repair_msgs[-1]["content"] and "customers" in repair_msgs[-1]["content"]


def test_empty_result_shows_real_values(shop):
    """Check that empty result shows real values."""
    llm = FakeLLM([sql("SELECT name FROM customers WHERE country='france'"),
                   sql("SELECT name FROM customers WHERE country='France'")])
    out = Agent(llm, SafeDB(shop["db"])).run("Customers in France?")
    assert out["status"] == "ok"
    assert "'France'" in llm.calls[1]["messages"][-1]["content"]


def test_empty_twice_is_accepted(shop):
    """Check that empty twice is accepted."""
    llm = FakeLLM([sql("SELECT name FROM customers WHERE country='Peru'"),
                   sql("SELECT name FROM customers WHERE country='Peru'")])
    out = Agent(llm, SafeDB(shop["db"])).run("Customers in Peru?")
    assert out["status"] == "empty" and out["calls"] == 2


def test_gives_up_after_three_repairs_and_keeps_last_attempt(shop):
    """Check that gives up after three repairs and keeps last attempt."""
    llm = FakeLLM(responder=lambda m, model: sql("SELECT nope FROM customers"))
    out = Agent(llm, SafeDB(shop["db"])).run("anything")
    assert out["status"] == "error" and out["calls"] == 4 and out["final_sql"] == "SELECT nope FROM customers"


def test_routed_drafts_small_and_repairs_large(shop):
    """Check that routed drafts small and repairs large."""
    llm = FakeLLM([sql("SELECT bad FROM customers"), sql("SELECT COUNT(*) FROM customers")])
    out = Agent(llm, SafeDB(shop["db"]), routed=True).run("How many customers?")
    assert out["status"] == "ok" and out["escalated"]
    assert [c["model"] for c in llm.calls] == [SMALL, LARGE]


def test_routed_stays_small_when_draft_works(shop):
    """Check that routed stays small when draft works."""
    llm = FakeLLM([sql("SELECT COUNT(*) FROM customers")])
    out = Agent(llm, SafeDB(shop["db"]), routed=True).run("How many customers?")
    assert not out["escalated"] and [c["model"] for c in llm.calls] == [SMALL]


def test_model_write_attempts_never_touch_the_db(shop):
    """Check that model write attempts never touch the db."""
    llm = FakeLLM([sql("DROP TABLE customers"), sql("SELECT COUNT(*) FROM customers")])
    out = Agent(llm, SafeDB(shop["db"])).run("q")
    assert out["status"] == "ok" and out["rows"] == [[3]]


def test_diagnose_unknown_table(shop):
    """Check that diagnose unknown table."""
    found = diagnose(SafeDB(shop["db"]), "SELECT * FROM clients", "error", "no such table: clients")
    assert found[0]["name"] == "list_tables" and "customers" in found[0]["output"]


def test_agent_draft_is_exactly_the_one_shot_request(shop):
    """Check that agent draft is exactly the one shot request."""
    db = SafeDB(shop["db"])
    schema = db.schema_context()
    a, b = FakeLLM([sql("SELECT 1")]), FakeLLM([sql("SELECT 1")])
    one_shot(a, db, "Q?", "hint", schema)
    Agent(b, db).run("Q?", "hint", schema)
    assert a.calls[0]["messages"] == b.calls[0]["messages"]
    assert b.calls[0]["messages"][0]["content"] == ONE_SHOT_PROMPT
    assert b.calls[0]["messages"][1]["content"] == user_message("Q?", "hint", schema)
    assert a.calls[0]["model"] == b.calls[0]["model"] == LARGE


def test_extract_sql_handles_fence_labels():
    """Check that extract sql handles fence labels."""
    from querypilot.agent import extract_sql
    assert extract_sql("```sqlite\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("```SQL\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("```\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("```SELECT a FROM t```") == "SELECT a FROM t"
    assert extract_sql("Here you go:\nSELECT a FROM t;") == "SELECT a FROM t"
    assert extract_sql("```sql\nWITH x AS (SELECT 1) SELECT * FROM x\n```").startswith("WITH")


def test_extract_sql_drops_label_lines_inside_block():
    """Check that extract sql drops label lines inside block."""
    from querypilot.agent import extract_sql
    assert extract_sql("```\nsql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("```sql\nsql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("```\r\nSQLite\r\nSELECT 1\r\n```") == "SELECT 1"
