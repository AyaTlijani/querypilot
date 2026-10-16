"""Read-only, sandboxed access to SQLite databases.

Every query the agent (or anyone) runs goes through `SafeDB.query`.
Writes are blocked by several independent layers, so a trick that gets
past one layer is still stopped by the next:

1. The file is opened with SQLite's read-only URI mode (`mode=ro`).
2. `PRAGMA query_only = ON` makes the connection refuse any write.
3. An authorizer callback whitelists read operations only. Writes, schema
   changes, ATTACH, and PRAGMA statements are denied inside the engine itself.
4. The SQL text must parse as exactly ONE statement, and it must be a SELECT
   (WITH ... SELECT is fine).
5. A progress handler enforces a time limit, and results are capped in rows.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import sqlglot
from sqlglot import exp

_FALLBACK_OK = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_ROWS = 1000

# Operations a pure read needs. Everything else is denied.
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


class UnsafeQueryError(ValueError):
    """The query was rejected before or during execution for safety reasons."""


class QueryTimeoutError(TimeoutError):
    """The query ran longer than the allowed time."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool = False
    elapsed_s: float = 0.0
    meta: dict = field(default_factory=dict)


def check_sql(sql: str) -> str:
    """Return the cleaned SQL if it is a single read-only statement, else raise."""
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise UnsafeQueryError("Empty query.")
    try:
        statements = [s for s in sqlglot.parse(text, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError:
        # The parser doesn't know every SQLite construct. Rejecting valid SQL
        # would unfairly lower benchmark scores, so fall back to a strict text
        # rule. The engine-level layers (read-only file, query_only, authorizer)
        # still block any write.
        if _FALLBACK_OK.match(text) and ";" not in text:
            return text
        raise UnsafeQueryError("Only single SELECT queries are allowed.")
    if len(statements) != 1:
        raise UnsafeQueryError("Only one statement per query is allowed.")
    stmt = statements[0]
    if isinstance(stmt, exp.Command):
        # sqlglot couldn't model this statement: same strict text rule as above.
        if _FALLBACK_OK.match(text):
            return text
        raise UnsafeQueryError("Only SELECT queries are allowed.")
    if not isinstance(stmt, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        # WITH ... SELECT parses as a Select carrying a `with` arg.
        raise UnsafeQueryError("Only SELECT queries are allowed.")
    # A write can't hide inside a SELECT in SQLite, and the engine layers block
    # writes anyway; this scan only catches odd parses. (Unsupported functions
    # inside a valid SELECT are not writes and are not rejected.)
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Pragma)
    if any(True for _ in stmt.find_all(*forbidden)):
        raise UnsafeQueryError("Only SELECT queries are allowed.")
    return text


# Read-only pragmas, reachable only as table functions like pragma_table_info().
_READ_PRAGMAS = {"table_info", "table_xinfo", "foreign_key_list", "index_list", "index_info"}


def _authorizer(action, arg1, arg2, db_name, trigger):
    if action in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() in _READ_PRAGMAS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class SafeDB:
    """A read-only handle on one SQLite file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Database not found: {self.path}")

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_authorizer)
        return conn

    def query(self, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S,
              max_rows: int = DEFAULT_MAX_ROWS, checked: bool = True) -> QueryResult:
        """Run one read-only query. `checked=False` skips only the text check
        (used for trusted benchmark reference SQL); the engine-level
        protections always stay on."""
        text = check_sql(sql) if checked else sql
        conn = self._connect()
        deadline = time.monotonic() + timeout_s
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        start = time.monotonic()
        try:
            cur = conn.execute(text)
            rows = cur.fetchmany(max_rows + 1)
            columns = [d[0] for d in (cur.description or [])]
        except sqlite3.OperationalError as err:
            if time.monotonic() > deadline:
                raise QueryTimeoutError(f"Query exceeded {timeout_s:.0f}s.") from err
            if "not authorized" in str(err):
                raise UnsafeQueryError("Blocked by the read-only policy.") from err
            raise
        except sqlite3.DatabaseError as err:
            if "not authorized" in str(err) or "readonly" in str(err).lower():
                raise UnsafeQueryError("Blocked by the read-only policy.") from err
            raise
        finally:
            conn.close()
        truncated = len(rows) > max_rows
        return QueryResult(columns, rows[:max_rows], truncated, time.monotonic() - start)

    # ---- schema helpers (all read-only) -------------------------------------

    def tables(self) -> list[str]:
        res = self.query("SELECT name FROM sqlite_master WHERE type='table' "
                         "AND name NOT LIKE 'sqlite_%' ORDER BY name")
        return [r[0] for r in res.rows]

    def ddl(self) -> str:
        res = self.query("SELECT sql FROM sqlite_master WHERE type='table' "
                         "AND name NOT LIKE 'sqlite_%' ORDER BY name")
        return "\n\n".join(r[0] for r in res.rows if r[0])

    def describe(self, table: str) -> list[tuple]:
        if table not in self.tables():
            raise UnsafeQueryError(f"Unknown table: {table}")
        safe_name = table.replace("'", "''")
        return self.query(f"SELECT name, type FROM pragma_table_info('{safe_name}')").rows

    def sample(self, table: str, n: int = 3) -> QueryResult:
        if table not in self.tables():
            raise UnsafeQueryError(f"Unknown table: {table}")
        safe_name = table.replace('"', '""')
        return self.query(f'SELECT * FROM "{safe_name}" LIMIT {int(n)}')

    def schema_context(self, sample_rows: int = 2) -> str:
        """The schema text given to EVERY setup (same context for all)."""
        parts = []
        for t in self.tables():
            safe_name = t.replace("'", "''")
            create = self.query(
                f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{safe_name}'"
            ).rows[0][0]
            sample = self.sample(t, sample_rows)
            lines = [create, f"/* {sample_rows} example rows from {t}:"]
            lines.append(" | ".join(sample.columns))
            lines += [" | ".join(_short(v) for v in row) for row in sample.rows]
            lines.append("*/")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)


def _short(value, limit: int = 40) -> str:
    s = "NULL" if value is None else str(value)
    return s if len(s) <= limit else s[: limit - 1] + "…"
