"""The QueryPilot agent and the one-shot baseline.

Execution-guided agent:
  1. draft  : write the query exactly like the one-shot baseline
  2. run it : if it executes and returns rows, that's the answer (a working
              answer is never rewritten, so the agent can't talk itself out of it)
  3. repair : only on evidence of failure (error or 0 rows), enter the
              LangGraph tool loop: think -> act (tools) -> verify -> ...

Three setups share the exact same inputs (question, evidence, schema text):
  * one_shot : one call to the large model, no tools, no repair
  * agent    : tools + self-repair, large model throughout
  * routed   : same agent, but starts on the small model and escalates to the
               large model only when it fails verification or runs out of steps
"""

from __future__ import annotations

import json
import re
from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

from .llm import LARGE, SMALL, LLMReply
from .safe_db import QueryTimeoutError, SafeDB, UnsafeQueryError

MAX_STEPS = 12      # LLM calls per question
MAX_REPAIRS = 3     # verification failures before giving up
PREVIEW_ROWS = 20

RULES = """You translate questions into SQLite queries.
Rules:
- Use only tables and columns that exist in the schema.
- Use the "evidence" hint when given; it defines terms used in the question.
- Return exactly the columns the question asks for, nothing extra.
- Text values are case-sensitive: check real values before filtering on them.
- SQLite dialect only."""

AGENT_PROMPT = RULES + """
You have tools to explore the database. Look at real values when unsure,
test intermediate queries with run_sql, then call submit_sql with your final query.
Always finish by calling submit_sql."""

ONE_SHOT_PROMPT = RULES + """
Reply with the final query only, inside a ```sql code block."""

TOOLS = [
    {"type": "function", "function": {
        "name": "list_tables", "description": "List all tables.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "describe_table", "description": "Columns and types of a table.",
        "parameters": {"type": "object", "properties": {"table": {"type": "string"}},
                       "required": ["table"]}}},
    {"type": "function", "function": {
        "name": "sample_rows", "description": "A few real rows from a table.",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string"}, "n": {"type": "integer"}}, "required": ["table"]}}},
    {"type": "function", "function": {
        "name": "run_sql", "description": "Run a read-only SELECT to explore. Returns up to 20 rows.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}},
                       "required": ["sql"]}}},
    {"type": "function", "function": {
        "name": "submit_sql", "description": "Submit the final SQL query answering the question.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}},
                       "required": ["sql"]}}},
]


def user_message(question: str, evidence: str, schema: str) -> str:
    parts = [f"Database schema:\n{schema}"]
    if evidence:
        parts.append(f"Evidence: {evidence}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.+?)```", text or "", re.S | re.I)
    return (m.group(1) if m else text or "").strip().rstrip(";")


def _table(res, limit=PREVIEW_ROWS) -> str:
    rows = res.rows[:limit]
    out = [" | ".join(res.columns)] + [" | ".join(str(v) for v in r) for r in rows]
    more = len(res.rows) - len(rows)
    if more > 0 or res.truncated:
        out.append(f"... ({'more than ' if res.truncated else ''}{len(res.rows)} rows total)")
    return "\n".join(out) if res.columns else "(no rows)"


class Run(TypedDict, total=False):
    messages: list
    steps: int
    repairs: int
    model: str
    escalated: bool
    final_sql: str
    columns: list
    rows: list
    status: str          # ok | empty | error | gave_up
    error: str
    cost: float
    tokens: int
    calls: int
    models_used: list
    last_failed_sql: str
    _pending: list
    _next: str
    _empty_checked: bool


class Agent:
    def __init__(self, llm, db: SafeDB, routed: bool = False,
                 emit: Callable[[dict], None] | None = None):
        self.llm, self.db, self.routed = llm, db, routed
        self.emit = emit or (lambda e: None)
        self.graph = self._build()

    # -- nodes ---------------------------------------------------------------
    def _call(self, s: Run) -> LLMReply:
        reply = self.llm.chat(s["messages"], tools=TOOLS, model=s["model"])
        s["cost"] += reply.cost_usd
        s["tokens"] += reply.prompt_tokens + reply.completion_tokens
        s["calls"] += 1
        s["steps"] += 1
        if reply.model not in s["models_used"]:
            s["models_used"].append(reply.model)
        return reply

    def think(self, s: Run) -> Run:
        reply = self._call(s)
        s["messages"].append(reply.as_message())
        if reply.content:
            self.emit({"type": "thought", "text": reply.content, "model": reply.model})
        if not reply.tool_calls:
            s["messages"].append({"role": "user",
                                  "content": "Please call submit_sql with your final query."})
        s["_pending"] = reply.tool_calls
        s["_next"] = "act" if reply.tool_calls else self._budget(s)
        return s

    def act(self, s: Run) -> Run:
        for call in s.pop("_pending", []):
            name, args = call["name"], call["arguments"]
            if name == "submit_sql":
                s["final_sql"] = (args.get("sql") or "").strip().rstrip(";")
                self.emit({"type": "submit", "sql": s["final_sql"]})
                out = "Submitted. Verifying..."
            else:
                out = self._tool(name, args)
            s["messages"].append({"role": "tool", "tool_call_id": call["id"],
                                  "name": name, "content": out[:4000]})
        s["_next"] = "verify" if s.get("final_sql") else self._budget(s)
        return s

    def _tool(self, name: str, args: dict) -> str:
        try:
            if name == "list_tables":
                out = ", ".join(self.db.tables())
            elif name == "describe_table":
                out = "\n".join(f"{c} {t}" for c, t in self.db.describe(args.get("table", "")))
            elif name == "sample_rows":
                out = _table(self.db.sample(args.get("table", ""), min(int(args.get("n", 5)), 20)))
            elif name == "run_sql":
                out = _table(self.db.query(args.get("sql", ""), max_rows=200))
            else:
                out = f"Unknown tool {name}"
            ok = True
        except Exception as err:  # tool errors go back to the model, never crash
            out, ok = f"Error: {err}", False
        self.emit({"type": "tool", "name": name, "args": args, "ok": ok, "output": out[:1500]})
        return out

    def verify(self, s: Run) -> Run:
        sql = s["final_sql"]
        try:
            res = self.db.query(sql)
            s["columns"], s["rows"] = res.columns, [list(r) for r in res.rows]
            s["status"] = "ok" if res.rows else "empty"
            s["error"] = ""
        except (UnsafeQueryError, QueryTimeoutError, Exception) as err:
            s["status"], s["error"] = "error", str(err)
        self.emit({"type": "verify", "status": s["status"], "error": s.get("error", ""),
                   "rows": len(s.get("rows") or [])})
        s["_next"] = END
        if s["status"] == "ok":
            return s
        if s["status"] == "empty" and s.get("_empty_checked"):
            return s  # already asked once; an empty answer can be correct
        if s["repairs"] >= MAX_REPAIRS:
            return s
        s["repairs"] += 1
        if s["status"] == "empty":
            s["_empty_checked"] = True
            feedback = ("Your query returned 0 rows. Check the filter values against real data "
                        "(sample_rows / run_sql). If 0 rows is truly correct, submit it again.")
        else:
            feedback = f"Your query failed: {s['error']}. Fix it and submit again."
        self._maybe_escalate(s, "verification failed")
        self.emit({"type": "repair", "attempt": s["repairs"], "reason": feedback})
        s["messages"].append({"role": "user", "content": feedback})
        s["last_failed_sql"] = s["final_sql"]
        s["final_sql"] = ""
        s["_next"] = self._budget(s)
        return s

    def _maybe_escalate(self, s: Run, why: str):
        if self.routed and s["model"] != LARGE:
            s["model"], s["escalated"] = LARGE, True
            self.emit({"type": "escalate", "reason": why, "model": LARGE})

    # -- routing: decided inside nodes, edges just read `_next` -------------
    def _budget(self, s: Run) -> str:
        if s["steps"] < MAX_STEPS:
            return "think"
        if self.routed and s["model"] != LARGE:
            self._maybe_escalate(s, "step budget used")
            s["steps"] = MAX_STEPS // 2   # give the large model half a budget
            return "think"
        s["status"] = "gave_up"
        return END

    def _build(self):
        g = StateGraph(Run)
        g.add_node("think", self.think)
        g.add_node("act", self.act)
        g.add_node("verify", self.verify)
        g.set_entry_point("think")
        for node in ("think", "act", "verify"):
            g.add_conditional_edges(node, lambda s: s["_next"])
        return g.compile()

    # -- public --------------------------------------------------------------
    def run(self, question: str, evidence: str = "", schema: str | None = None) -> Run:
        schema = schema if schema is not None else self.db.schema_context()
        ask = user_message(question, evidence, schema)
        s: Run = {
            "messages": [], "steps": 0, "repairs": 0, "model": SMALL if self.routed else LARGE,
            "escalated": False, "final_sql": "", "status": "gave_up", "error": "",
            "cost": 0.0, "tokens": 0, "calls": 0, "models_used": [], "columns": [], "rows": [],
        }
        self.emit({"type": "start", "model": s["model"]})

        # 1) Draft: identical prompt and inputs to the one-shot baseline.
        draft = self.llm.chat([{"role": "system", "content": ONE_SHOT_PROMPT},
                               {"role": "user", "content": ask}], model=s["model"])
        s["cost"] += draft.cost_usd
        s["tokens"] += draft.prompt_tokens + draft.completion_tokens
        s["calls"] += 1
        s["models_used"].append(draft.model)
        sql = extract_sql(draft.content)
        self.emit({"type": "draft", "sql": sql, "model": draft.model})

        # 2) Run it. A query that works and returns rows is accepted as is.
        s["final_sql"] = sql
        try:
            res = self.db.query(sql)
            s["columns"], s["rows"] = res.columns, [list(r) for r in res.rows]
            s["status"] = "ok" if res.rows else "empty"
        except Exception as err:
            s["status"], s["error"] = "error", str(err)
        self.emit({"type": "verify", "status": s["status"], "error": s["error"],
                   "rows": len(s["rows"])})
        if s["status"] == "ok":
            return self._finish(s)

        # 3) Repair with tools, only because the draft failed.
        if s["status"] == "empty":
            feedback = ("Your query returned 0 rows. Check the filter values against real data "
                        "with sample_rows / run_sql. If 0 rows is truly correct, submit it again.")
            s["_empty_checked"] = True
        else:
            feedback = f"Your query failed: {s['error']}"
        s["repairs"] = 1
        s["last_failed_sql"] = sql
        s["final_sql"] = ""
        self._maybe_escalate(s, "draft failed")
        self.emit({"type": "repair", "attempt": 1, "reason": feedback})
        s["messages"] = [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "user", "content": ask},
            {"role": "assistant", "content": f"```sql\n{sql}\n```"},
            {"role": "user", "content": feedback + " Investigate with the tools, then call submit_sql."},
        ]
        out = self.graph.invoke(s, {"recursion_limit": 4 * MAX_STEPS + 20})
        return self._finish(out)

    def _finish(self, out) -> Run:
        out = dict(out)
        for k in ("_pending", "_empty_checked", "_next"):
            out.pop(k, None)
        if not out.get("final_sql") and out.get("last_failed_sql"):
            out["final_sql"] = out["last_failed_sql"]  # score the last attempt, not nothing
        self.emit({"type": "done", "status": out["status"], "sql": out.get("final_sql", ""),
                   "cost": out["cost"], "tokens": out["tokens"]})
        return out


def one_shot(llm, db: SafeDB, question: str, evidence: str = "",
             schema: str | None = None) -> Run:
    schema = schema if schema is not None else db.schema_context()
    messages = [{"role": "system", "content": ONE_SHOT_PROMPT},
                {"role": "user", "content": user_message(question, evidence, schema)}]
    reply = llm.chat(messages, model=LARGE)
    return {"final_sql": extract_sql(reply.content), "cost": reply.cost_usd, "calls": 1,
            "tokens": reply.prompt_tokens + reply.completion_tokens,
            "models_used": [reply.model], "escalated": False, "repairs": 0, "steps": 1,
            "status": "submitted"}


def summarize(llm, question: str, columns: list, rows: list) -> str:
    """One-sentence plain-language answer for the demo (not used in the benchmark)."""
    preview = json.dumps({"columns": columns, "rows": rows[:10]}, default=str)
    reply = llm.chat([
        {"role": "system", "content": "Answer the question in one or two plain sentences "
                                      "using only the query result. No SQL talk."},
        {"role": "user", "content": f"Question: {question}\nResult: {preview}"}], model=SMALL)
    return reply.content.strip()
