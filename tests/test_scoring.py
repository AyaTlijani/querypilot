"""Tests for the benchmark scoring rules."""

from querypilot.safe_db import SafeDB
from querypilot.scoring import score


def test_equivalent_sql_counts_as_correct(shop):
    """Check that equivalent sql counts as correct."""
    db = SafeDB(shop["db"])
    gold = "SELECT COUNT(*) FROM customers WHERE country = 'France'"
    assert score(db, "select count(id) from customers where country='France'", gold)["correct"]
    assert score(db, "SELECT COUNT(*) FROM (SELECT * FROM customers) WHERE country IN ('France')", gold)["correct"]


def test_order_and_duplicates_ignored_like_official_bird(shop):
    """Check that order and duplicates ignored like official bird."""
    db = SafeDB(shop["db"])
    gold = "SELECT name FROM customers ORDER BY name"
    assert score(db, "SELECT name FROM customers ORDER BY name DESC", gold)["correct"]


def test_wrong_and_failed_predictions(shop):
    """Check that wrong and failed predictions."""
    db = SafeDB(shop["db"])
    gold = "SELECT COUNT(*) FROM customers WHERE country = 'France'"
    assert score(db, "SELECT COUNT(*) FROM customers WHERE country = 'france'", gold)["category"] == "wrong_logic"
    assert score(db, "SELECT nope FROM customers", gold)["category"] == "sql_error"
    assert score(db, "DELETE FROM customers", gold)["category"] == "blocked"
    assert score(db, "", gold)["category"] == "no_sql"
    assert score(db, "SELECT COUNT(*) FROM orders", gold)["category"] == "wrong_tables"


def test_near_miss_is_reported_but_not_correct(shop):
    """Check that near miss is reported but not correct."""
    db = SafeDB(shop["db"])
    gold = "SELECT name FROM customers WHERE country = 'France'"
    s = score(db, "SELECT name, country FROM customers WHERE country = 'France'", gold)
    assert not s["correct"] and s["near_miss"]


def test_empty_gold_flagged(shop):
    """Check that empty gold flagged."""
    db = SafeDB(shop["db"])
    s = score(db, "SELECT name FROM customers WHERE 0", "SELECT name FROM customers WHERE country='Peru'")
    assert s["gold_empty"]
