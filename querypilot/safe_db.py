"""Read-only access to SQLite databases, with several layers that block any write."""

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

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


class UnsafeQueryError(ValueError):
    """Raised when a query is not a single read-only SELECT."""


class QueryTimeoutError(TimeoutError):
    """Raised when a query runs longer than the time limit."""


@dataclass
class QueryResult:
    """The columns and rows a query returned."""
    columns: list[str]
    rows: list[tuple]
    truncated: bool = False
    elapsed_s: float = 0.0
    meta: dict = field(default_factory=dict)


def check_sql(sql: str) -> str:
    """Accept only a single SELECT statement; reject everything else."""
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise UnsafeQueryError("Empty query.")
    try:
        statements = [s for s in sqlglot.parse(text, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError:
        if _FALLBACK_OK.match(text) and ";" not in text:
            return text
        raise UnsafeQueryError("Only single SELECT queries are allowed.")
    if len(statements) != 1:
        raise UnsafeQueryError("Only one statement per query is allowed.")
    stmt = statements[0]
    if isinstance(stmt, exp.Command):
        if _FALLBACK_OK.match(text):
            return text
        raise UnsafeQueryError("Only SELECT queries are allowed.")
    if not isinstance(stmt, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise UnsafeQueryError("Only SELECT queries are allowed.")
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Pragma)
    if any(True for _ in stmt.find_all(*forbidden)):
        raise UnsafeQueryError("Only SELECT queries are allowed.")
    return text


_READ_PRAGMAS = {"table_info", "table_xinfo", "foreign_key_list", "index_list", "index_info"}


def _authorizer(action, arg1, arg2, db_name, trigger):
    """Let SQLite allow read operations only."""
    if action in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() in _READ_PRAGMAS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class SafeDB:
    """A read-only connection to one SQLite file."""

    def __init__(self, path: str | Path):
        """Remember the database file, failing if it doesn't exist."""
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Database not found: {self.path}")

    def _connect(self) -> sqlite3.Connection:
        """Open the file read-only, with writes blocked inside SQLite."""
        uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_authorizer)
        return conn

    def query(self, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S,
              max_rows: int = DEFAULT_MAX_ROWS, checked: bool = True) -> QueryResult:
        """Run one read-only query with a time limit and a row limit."""
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

    def tables(self) -> list[str]:
        """List the table names."""
        res = self.query("SELECT name FROM sqlite_master WHERE type='table' "
                         "AND name NOT LIKE 'sqlite_%' ORDER BY name")
        return [r[0] for r in res.rows]

    def ddl(self) -> str:
        """Return the CREATE TABLE statements."""
        res = self.query("SELECT sql FROM sqlite_master WHERE type='table' "
                         "AND name NOT LIKE 'sqlite_%' ORDER BY name")
        return "\n\n".join(r[0] for r in res.rows if r[0])

    def describe(self, table: str) -> list[tuple]:
        """List a table's columns and types."""
        if table not in self.tables():
            raise UnsafeQueryError(f"Unknown table: {table}")
        safe_name = table.replace("'", "''")
        return self.query(f"SELECT name, type FROM pragma_table_info('{safe_name}')").rows

    def sample(self, table: str, n: int = 3) -> QueryResult:
        """Return a few rows from a table."""
        if table not in self.tables():
            raise UnsafeQueryError(f"Unknown table: {table}")
        safe_name = table.replace('"', '""')
        return self.query(f'SELECT * FROM "{safe_name}" LIMIT {int(n)}')

    def schema_context(self, sample_rows: int = 2) -> str:
        """Build the schema text given to the model: tables plus example rows."""
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
    """Shorten a long value for display."""
    s = "NULL" if value is None else str(value)
    return s if len(s) <= limit else s[: limit - 1] + "…"
