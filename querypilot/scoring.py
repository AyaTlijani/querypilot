"""Scores predicted SQL against BIRD's reference SQL, using BIRD's official rule."""

from __future__ import annotations

import itertools

import sqlglot
from sqlglot import exp

from .safe_db import QueryTimeoutError, SafeDB, UnsafeQueryError

TIMEOUT_S = 30.0
MAX_ROWS = 100_000


def execute(db: SafeDB, sql: str, trusted: bool = False):
    """Run a query and return its rows, or the kind of error it hit."""
    if not sql:
        return None, "no_sql", "No SQL produced."
    try:
        res = db.query(sql, timeout_s=TIMEOUT_S, max_rows=MAX_ROWS, checked=not trusted)
        return [tuple(r) for r in res.rows], None, ""
    except QueryTimeoutError as err:
        return None, "timeout", str(err)
    except UnsafeQueryError as err:
        return None, "blocked", str(err)
    except Exception as err:
        return None, "sql_error", str(err)


def _round(row):
    """Round floats in a row so tiny rounding differences can be detected."""
    return tuple(round(v, 2) if isinstance(v, float) else v for v in row)


def near_miss(pred: list[tuple], gold: list[tuple]) -> bool:
    """Check if a wrong answer is only off by rounding or extra columns (reported, never counted correct)."""
    if not pred or not gold:
        return False
    if set(map(_round, pred)) == set(map(_round, gold)):
        return True
    width = len(gold[0])
    if width < len(pred[0]) <= 12:
        gold_set = set(gold)
        cols = range(len(pred[0]))
        for combo in itertools.combinations(cols, width):
            if {tuple(r[i] for i in combo) for r in pred} == gold_set:
                return True
    return False


def tables_in(sql: str) -> set[str]:
    """List the tables a query uses."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        return {t.name.lower() for t in tree.find_all(exp.Table)}
    except Exception:
        return set()


def score(db: SafeDB, pred_sql: str, gold_sql: str) -> dict:
    """Compare a predicted query with the reference: correct or not, and why."""
    gold, g_kind, g_msg = execute(db, gold_sql, trusted=True)
    pred, p_kind, p_msg = execute(db, pred_sql)
    out = {"correct": False, "gold_empty": gold == [], "near_miss": False,
           "category": "", "error": p_msg}
    if gold is None:
        out["category"] = "gold_failed"
        out["error"] = f"reference query failed: {g_msg}"
        return out
    if pred is None:
        out["category"] = p_kind
        return out
    if set(pred) == set(gold):
        out["correct"] = True
        out["category"] = "correct"
        return out
    out["near_miss"] = near_miss(pred, gold)
    if not pred:
        out["category"] = "empty_result"
    elif tables_in(pred_sql) != tables_in(gold_sql):
        out["category"] = "wrong_tables"
    elif gold and len(pred[0]) != len(gold[0]):
        out["category"] = "wrong_columns"
    else:
        out["category"] = "wrong_logic"
    return out
