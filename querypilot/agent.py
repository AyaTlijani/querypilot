"""The QueryPilot agent and the one-shot baseline.

Execution-guided self-correction, as a LangGraph workflow:

    draft -> execute --(rows)-------------------------------------------> done
                 '--(error / 0 rows)--> diagnose (database tools) --> revise -> execute ...

  * draft    : the query is written exactly like the one-shot baseline
  * execute  : it runs in the read-only sandbox; a query that returns rows is kept,
               so a working answer is never rewritten
  * diagnose : the workflow itself calls database tools to gather evidence for the
               failure (similar column names, real table names, the actual values of
               a column the query filtered on, sample rows)
  * revise   : the model rewrites the query from that evidence (up to 3 times)

The tools are run by the workflow, not chosen by the model. Small local models are
unreliable at deciding which tool to call; given the evidence directly, they fix
far more errors.

Three setups share the exact same inputs (question, evidence, schema text):
  * one_shot : one call to the large model
  * agent    : the workflow above, large model throughout
  * routed   : drafts with the small model, escalates to the large one to repair
"""

from __future__ import annotations

import json
import re
from typing import Callable, TypedDict

import sqlglot
from langgraph.graph import END, StateGraph
from sqlglot import exp

from .llm import LARGE, SMALL
from .safe_db import SafeDB

MAX_REPAIRS = 3     # revisions after a failed query
PREVIEW_ROWS = 20

RULES = """You translate questions into SQLite queries.
Rules:
- Use only tables and columns that exist in the schema.
- Use the "evidence" hint when given; it defines terms used in the question.
- Return exactly the columns the question asks for, nothing extra.
- Text values are case-sensitive: check real values before filtering on them.
- SQLite dialect only."""

ONE_SHOT_PROMPT = RULES + """
Reply with the final query only, inside a ```sql code block."""

REPAIR_PROMPT = RULES + """
Your previous query did not work. Use the database diagnostics to correct it.
Reply with the corrected query only, inside a ```sql code block."""


def user_message(question: str, evidence: str, schema: str) -> str:
    parts = [f"Database schema:\n{schema}"]
    if evidence:
        parts.append(f"Evidence: {evidence}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


_LANG_TAGS = {"", "sql", "sqlite", "sqlite3", "postgresql", "mysql", "ansi"}
_SQL_START = re.compile(r"\b(WITH|SELECT)\b", re.I)


def extract_sql(text: str) -> str:
    """The query from a model reply. Handles ```sql, ```sqlite and other fence labels,
    and replies without a fence. Never fails a correct query because of formatting."""
    text = text or ""
    m = re.search(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n?(.*?)```", text, re.S)
    if m:
        tag, body = m.group(1), m.group(2)
        if tag.lower() not in _LANG_TAGS:          # e.g. ```SELECT ... on one line
            body = f"{tag} {body}"
    else:
        start = _SQL_START.search(text)
        body = text[start.start():] if start else text
    return body.strip().rstrip(";").strip()


def _table(res, limit=PREVIEW_ROWS) -> str:
    rows = res.rows[:limit]
    out = [" | ".join(res.columns)] + [" | ".join(str(v) for v in r) for r in rows]
    more = len(res.rows) - len(rows)
    if more > 0 or res.truncated:
        out.append(f"... ({'more than ' if res.truncated else ''}{len(res.rows)} rows total)")
    return "\n".join(out) if res.columns else "(no rows)"


# ---------------------------------------------------------------------------
# Database diagnostics (the tools the workflow runs after a failure)
# ---------------------------------------------------------------------------

def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _parse(sql: str):
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None


def diagnose(db: SafeDB, sql: str, status: str, error: str) -> list[dict]:
    """Gather evidence about why `sql` failed. Each item: {name, args, output}."""
    tables = db.tables()
    by_lower = {t.lower(): t for t in tables}
    tree = _parse(sql)
    used = []
    if tree is not None:
        for t in tree.find_all(exp.Table):
            real = by_lower.get(t.name.lower())
            if real and real not in used:
                used.append(real)
    found: list[dict] = []

    def add(name, args, output):
        found.append({"name": name, "args": args, "output": output[:1500]})

    if status == "error":
        if re.search(r"no such table", error, re.I):
            add("list_tables", {}, ", ".join(tables))
        m = re.search(r"no such column: ([\w.\"`\[\] ]+)", error, re.I)
        if m:
            wanted = m.group(1).split(".")[-1].strip(' "`[]').lower()
            hits = []
            for t in tables:
                for c, _ in db.describe(t):
                    cl = c.lower()
                    if wanted and (wanted in cl or cl in wanted or
                                   wanted.replace("_", "") == cl.replace("_", "").replace(" ", "")):
                        hits.append(f"{t}.{c}")
            add("find_column", {"name": wanted},
                "similar columns: " + (", ".join(hits[:20]) if hits else "none found"))

    if status == "empty" and tree is not None:
        for node in tree.find_all(exp.EQ, exp.Like, exp.In):
            col = node.this if isinstance(node.this, exp.Column) else None
            lits = [l for l in node.find_all(exp.Literal) if l.is_string]
            if col is None or not lits:
                continue
            for t in used:
                cols = {c.lower(): c for c, _ in db.describe(t)}
                real_col = cols.get(col.name.lower())
                if not real_col:
                    continue
                wanted = lits[0].this
                try:
                    vals = [r[0] for r in db.query(
                        f"SELECT DISTINCT {_q(real_col)} FROM {_q(t)} "
                        f"WHERE {_q(real_col)} IS NOT NULL LIMIT 500").rows]
                except Exception:
                    continue
                key = wanted.lower().strip("%")
                close = [v for v in vals if key and (key in str(v).lower() or str(v).lower() in key)]
                shown = (close or vals)[:20]
                add("distinct_values", {"column": f"{t}.{real_col}", "looking_for": wanted},
                    f"real values of {t}.{real_col}" + (" similar to the filter" if close else "")
                    + ": " + ", ".join(repr(v) for v in shown))
                break

    for t in used[:3]:
        try:
            add("sample_rows", {"table": t}, _table(db.sample(t, 3)))
        except Exception:
            pass
    if not found:
        add("list_tables", {}, ", ".join(tables))
    return found


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class Run(TypedDict, total=False):
    ask: str
    messages: list
    model: str
    sql: str
    final_sql: str
    columns: list
    rows: list
    status: str          # ok | empty | error
    error: str
    repairs: int
    escalated: bool
    cost: float
    tokens: int
    calls: int
    models_used: list
    empty_checked: bool


class Agent:
    def __init__(self, llm, db: SafeDB, routed: bool = False,
                 emit: Callable[[dict], None] | None = None):
        self.llm, self.db, self.routed = llm, db, routed
        self.emit = emit or (lambda e: None)
        self.graph = self._build()

    def _chat(self, s: Run, messages: list) -> str:
        reply = self.llm.chat(messages, model=s["model"])
        s["cost"] += reply.cost_usd
        s["tokens"] += reply.prompt_tokens + reply.completion_tokens
        s["calls"] += 1
        if reply.model not in s["models_used"]:
            s["models_used"].append(reply.model)
        return reply.content

    # -- nodes ---------------------------------------------------------------
    def draft(self, s: Run) -> Run:
        content = self._chat(s, [{"role": "system", "content": ONE_SHOT_PROMPT},
                                 {"role": "user", "content": s["ask"]}])
        s["sql"] = extract_sql(content)
        self.emit({"type": "draft", "sql": s["sql"], "model": s["models_used"][-1]})
        return s

    def execute(self, s: Run) -> Run:
        s["final_sql"] = s["sql"]
        try:
            res = self.db.query(s["sql"])
            s["columns"], s["rows"] = res.columns, [list(r) for r in res.rows]
            s["status"], s["error"] = ("ok" if res.rows else "empty"), ""
        except Exception as err:
            s["columns"], s["rows"] = [], []
            s["status"], s["error"] = "error", str(err)
        self.emit({"type": "verify", "status": s["status"], "error": s["error"],
                   "rows": len(s["rows"])})
        return s

    def diagnose(self, s: Run) -> Run:
        s["repairs"] += 1
        if s["status"] == "empty":
            s["empty_checked"] = True
            problem = "The query ran but returned 0 rows."
        else:
            problem = f"The query failed with: {s['error']}"
        if self.routed and s["model"] != LARGE:
            s["model"], s["escalated"] = LARGE, True
            self.emit({"type": "escalate", "reason": "first draft failed", "model": LARGE})
        self.emit({"type": "repair", "attempt": s["repairs"], "reason": problem})
        evidence = diagnose(self.db, s["sql"], s["status"], s["error"])
        for item in evidence:
            self.emit({"type": "tool", "ok": True, **item})
        notes = "\n\n".join(f"[{e['name']} {json.dumps(e['args'])}]\n{e['output']}" for e in evidence)
        if not s["messages"]:
            s["messages"] = [{"role": "system", "content": REPAIR_PROMPT},
                             {"role": "user", "content": s["ask"]}]
        s["messages"] += [
            {"role": "assistant", "content": f"```sql\n{s['sql']}\n```"},
            {"role": "user", "content": f"{problem}\n\nDatabase diagnostics:\n{notes}\n\n"
                                        "Write the corrected query."},
        ]
        return s

    def revise(self, s: Run) -> Run:
        s["sql"] = extract_sql(self._chat(s, s["messages"]))
        self.emit({"type": "submit", "sql": s["sql"]})
        return s

    def after_execute(self, s: Run) -> str:
        if s["status"] == "ok":
            return END
        if s["status"] == "empty" and s.get("empty_checked"):
            return END   # already re-checked once: an empty answer can be correct
        if s["repairs"] >= MAX_REPAIRS:
            return END
        return "diagnose"

    def _build(self):
        g = StateGraph(Run)
        g.add_node("draft", self.draft)
        g.add_node("execute", self.execute)
        g.add_node("diagnose", self.diagnose)
        g.add_node("revise", self.revise)
        g.set_entry_point("draft")
        g.add_edge("draft", "execute")
        g.add_conditional_edges("execute", self.after_execute)
        g.add_edge("diagnose", "revise")
        g.add_edge("revise", "execute")
        return g.compile()

    # -- public --------------------------------------------------------------
    def run(self, question: str, evidence: str = "", schema: str | None = None) -> Run:
        schema = schema if schema is not None else self.db.schema_context()
        s: Run = {
            "ask": user_message(question, evidence, schema), "messages": [],
            "model": SMALL if self.routed else LARGE, "sql": "", "final_sql": "",
            "columns": [], "rows": [], "status": "error", "error": "", "repairs": 0,
            "escalated": False, "cost": 0.0, "tokens": 0, "calls": 0, "models_used": [],
            "empty_checked": False,
        }
        self.emit({"type": "start", "model": s["model"]})
        out = dict(self.graph.invoke(s, {"recursion_limit": 50}))
        for k in ("ask", "messages", "sql", "empty_checked"):
            out.pop(k, None)
        self.emit({"type": "done", "status": out["status"], "sql": out["final_sql"],
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
            "models_used": [reply.model], "escalated": False, "repairs": 0,
            "status": "submitted"}


def summarize(llm, question: str, columns: list, rows: list) -> str:
    """One-sentence plain-language answer for the demo (not used in the benchmark)."""
    preview = json.dumps({"columns": columns, "rows": rows[:10]}, default=str)
    reply = llm.chat([
        {"role": "system", "content": "Answer the question in one or two plain sentences "
                                      "using only the query result. No SQL talk."},
        {"role": "user", "content": f"Question: {question}\nResult: {preview}"}], model=SMALL)
    return reply.content.strip()
