"""Tests that the read-only sandbox blocks every kind of write."""

import hashlib

import pytest

from querypilot.safe_db import SafeDB, UnsafeQueryError, QueryTimeoutError

ATTACKS = [
    "DELETE FROM customers",
    "DROP TABLE customers",
    "UPDATE customers SET name = 'x'",
    "INSERT INTO customers VALUES (9,'x','y')",
    "CREATE TABLE evil (a)",
    "ALTER TABLE customers ADD COLUMN x",
    "ATTACH DATABASE 'other.db' AS other",
    "PRAGMA writable_schema = 1",
    "PRAGMA query_only = OFF",
    "SELECT 1; DROP TABLE customers",
    "SELECT 1; DELETE FROM customers",
    "SELECT * FROM customers; -- harmless?\nDELETE FROM customers",
    "/* hi */ DELETE FROM customers",
    "WITH x AS (SELECT 1) DELETE FROM customers",
    "REPLACE INTO customers VALUES (1,'x','y')",
    "VACUUM",
    "",
]


def digest(path):
    """Return a hash of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("sql", ATTACKS)
def test_writes_are_blocked_and_file_unchanged(shop, sql):
    """Check that writes are blocked and file unchanged."""
    db = SafeDB(shop["db"])
    before = digest(shop["db"])
    with pytest.raises((UnsafeQueryError, Exception)):
        db.query(sql)
    assert digest(shop["db"]) == before
    assert db.query("SELECT COUNT(*) FROM customers").rows == [(3,)]


def test_engine_blocks_writes_even_without_text_check(shop):
    """Check that engine blocks writes even without text check."""
    db = SafeDB(shop["db"])
    before = digest(shop["db"])
    for sql in ("DELETE FROM customers", "DROP TABLE orders", "ATTACH 'x.db' AS y"):
        with pytest.raises(Exception):
            db.query(sql, checked=False)
    assert digest(shop["db"]) == before


def test_reads_work(shop):
    """Check that reads work."""
    db = SafeDB(shop["db"])
    assert db.tables() == ["customers", "orders"]
    assert ("country", "TEXT") in db.describe("customers")
    res = db.query("WITH f AS (SELECT * FROM customers WHERE country='France') SELECT COUNT(*) FROM f")
    assert res.rows == [(2,)]


def test_timeout(shop):
    """Check that timeout."""
    db = SafeDB(shop["db"])
    with pytest.raises(QueryTimeoutError):
        db.query("WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c) "
                 "SELECT COUNT(*) FROM c", timeout_s=0.5)


def test_row_cap(shop):
    """Check that row cap."""
    db = SafeDB(shop["db"])
    res = db.query("SELECT * FROM orders", max_rows=2)
    assert len(res.rows) == 2 and res.truncated


def test_valid_sqlite_selects_are_never_blocked(shop):
    """Check that valid sqlite selects are never blocked."""
    from querypilot.safe_db import check_sql
    for q in ["SELECT `country` FROM customers", "SELECT IIF(id > 1, 'a', 'b') FROM customers",
              "-- note\nSELECT 1", "SELECT name FROM customers WHERE name LIKE '%;%'",
              "WITH r AS (SELECT id, RANK() OVER (ORDER BY id) AS k FROM customers) SELECT id FROM r"]:
        check_sql(q)
