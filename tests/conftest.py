"""Shared test fixtures: a tiny sample database and questions file."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def shop(tmp_path):
    """Create a tiny sample database and questions file for tests."""
    root = tmp_path / "dbs" / "shop"
    root.mkdir(parents=True)
    path = root / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, status TEXT);
        INSERT INTO customers VALUES (1,'Ana','France'),(2,'Ben','Germany'),(3,'Cleo','France');
        INSERT INTO orders VALUES (1,1,10.5,'delivered'),(2,1,20.0,'returned'),
                                  (3,2,5.25,'delivered'),(4,3,7.0,'delivered');
    """)
    con.commit()
    con.close()
    questions = [
        {"question_id": 0, "db_id": "shop", "question": "How many customers live in France?",
         "evidence": "", "SQL": "SELECT COUNT(*) FROM customers WHERE country = 'France'",
         "difficulty": "simple"},
        {"question_id": 1, "db_id": "shop", "question": "Total delivered amount?",
         "evidence": "delivered refers to status = 'delivered'",
         "SQL": "SELECT SUM(amount) FROM orders WHERE status = 'delivered'",
         "difficulty": "moderate"},
    ]
    qfile = tmp_path / "questions.json"
    qfile.write_text(json.dumps(questions))
    return {"db": path, "root": tmp_path / "dbs", "questions": qfile}
