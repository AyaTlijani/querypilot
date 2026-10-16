"""Exposes the read-only database tools over the Model Context Protocol (MCP)."""

from __future__ import annotations

import os

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

from .agent import _table
from .safe_db import SafeDB

db = SafeDB(os.getenv("QP_DB", "data/demo/shop.sqlite"))
mcp = FastMCP("querypilot")


@mcp.tool()
def list_tables() -> str:
    """List all tables in the database."""
    return ", ".join(db.tables())


@mcp.tool()
def describe_table(table: str) -> str:
    """List the columns and types of one table."""
    return "\n".join(f"{c} {t}" for c, t in db.describe(table))


@mcp.tool()
def sample_rows(table: str, n: int = 5) -> str:
    """Return a few real rows from a table (max 20)."""
    return _table(db.sample(table, min(n, 20)))


@mcp.tool()
def run_sql(sql: str) -> str:
    """Run one read-only SELECT query (30 s limit, 200 rows max)."""
    return _table(db.query(sql, max_rows=200), limit=200)


if __name__ == "__main__":
    mcp.run()
