"""Builds the demo online-shop database and the offline fake model."""

from __future__ import annotations

import random
import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from .llm import FakeLLM, LLMReply

DEMO_PATH = Path("data/demo/shop.sqlite")

EXAMPLES = [
    "Which product categories lost the most revenue to returns in 2025?",
    "What is the average delivery delay per carrier, worst first?",
    "Who are the top 5 customers by total spend in Germany?",
    "How did monthly revenue evolve over the last 6 months?",
    "Which products have an average rating below 3 with at least 10 reviews?",
]

CITIES = {"France": ["Paris", "Lyon", "Marseille", "Lille", "Toulouse"],
          "Germany": ["Berlin", "Munich", "Hamburg", "Cologne"],
          "Spain": ["Madrid", "Barcelona", "Valencia"],
          "Italy": ["Milan", "Rome", "Turin"],
          "Belgium": ["Brussels", "Antwerp"],
          "Netherlands": ["Amsterdam", "Rotterdam"]}
CATEGORIES = {"Electronics": (40, 900), "Home": (10, 300), "Fashion": (15, 220),
              "Beauty": (5, 90), "Sports": (12, 400), "Books": (6, 45)}
FIRST = ["Camille", "Lucas", "Léa", "Hugo", "Emma", "Jonas", "Mia", "Lukas", "Sofia",
         "Mateo", "Giulia", "Marco", "Lotte", "Daan", "Chloé", "Noah", "Elena", "Paul"]
LAST = ["Martin", "Bernard", "Müller", "Schmidt", "García", "López", "Rossi", "Bianchi",
        "Peeters", "De Jong", "Dubois", "Fischer", "Moreau", "Romano", "Janssens"]
CARRIERS = ["DHL", "UPS", "Colissimo", "GLS", "DPD"]


def build(path: Path = DEMO_PATH, seed: int = 7) -> Path:
    """Generate the demo shop database with a fixed random seed."""
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT,
        country TEXT, segment TEXT, signup_date TEXT);
    CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, category TEXT,
        price REAL, supplier TEXT);
    CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers,
        order_date TEXT, status TEXT, channel TEXT);
    CREATE TABLE order_items (order_id INTEGER REFERENCES orders, product_id INTEGER REFERENCES products,
        quantity INTEGER, unit_price REAL, discount REAL);
    CREATE TABLE deliveries (order_id INTEGER PRIMARY KEY REFERENCES orders, carrier TEXT,
        promised_date TEXT, delivered_date TEXT, delay_days INTEGER);
    CREATE TABLE reviews (review_id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders,
        product_id INTEGER REFERENCES products, rating INTEGER, review_date TEXT, comment TEXT);
    """)
    customers = []
    for cid in range(1, 1501):
        country = rng.choices(list(CITIES), weights=[30, 25, 15, 12, 9, 9])[0]
        customers.append((cid, f"{rng.choice(FIRST)} {rng.choice(LAST)}", rng.choice(CITIES[country]),
                          country, rng.choices(["consumer", "business"], [85, 15])[0],
                          str(date(2023, 1, 1) + timedelta(days=rng.randint(0, 900)))))
    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customers)

    products, pid = [], 1
    for cat, (lo, hi) in CATEGORIES.items():
        for i in range(20):
            products.append((pid, f"{cat} item {i + 1:02d}", cat, round(rng.uniform(lo, hi), 2),
                             f"Supplier {rng.choice('ABCDEFG')}"))
            pid += 1
    con.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)
    weak = set(rng.sample(range(1, pid), 6))
    late_carrier = "GLS"

    start = date(2025, 1, 1)
    order_rows, items, deliveries, reviews = [], [], [], []
    rid = 1
    for oid in range(1, 6001):
        cust = rng.choice(customers)
        d = start + timedelta(days=int(rng.triangular(0, 638, 520)))
        status = rng.choices(["delivered", "shipped", "cancelled", "returned"], [80, 6, 6, 8])[0]
        order_rows.append((oid, cust[0], str(d), status, rng.choice(["web", "app", "marketplace"])))
        for p in rng.sample(products, rng.choices([1, 2, 3], [65, 25, 10])[0]):
            items.append((oid, p[0], rng.choices([1, 2, 3], [80, 15, 5])[0], p[3],
                          rng.choice([0, 0, 0, 0.1, 0.15])))
            if status == "delivered" and rng.random() < 0.3:
                base = 2.4 if p[0] in weak else 4.2
                rating = max(1, min(5, round(rng.gauss(base, 0.9))))
                reviews.append((rid, oid, p[0], rating, str(d + timedelta(days=rng.randint(3, 20))),
                                rng.choice(["", "Great", "As described", "Arrived damaged",
                                            "Too small", "Would buy again", "Late delivery"])))
                rid += 1
        if status in ("delivered", "returned", "shipped"):
            carrier = rng.choice(CARRIERS)
            promised = d + timedelta(days=rng.randint(2, 5))
            delay = max(0, int(rng.gauss(2.5 if carrier == late_carrier else 0.6, 1.5)))
            delivered = None if status == "shipped" else str(promised + timedelta(days=delay))
            deliveries.append((oid, carrier, str(promised), delivered,
                               None if delivered is None else delay))
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", order_rows)
    con.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)
    con.executemany("INSERT INTO deliveries VALUES (?,?,?,?,?)", deliveries)
    con.executemany("INSERT INTO reviews VALUES (?,?,?,?,?,?)", reviews)
    con.commit()
    con.close()
    return path


def offline_llm() -> FakeLLM:

    """A fake model used when no real model is available, so the app still works."""
    def responder(messages, model):
        """Return a scripted reply that walks through a draft, a failure and a repair."""
        system = messages[0]["content"].lower()
        if "previous query did not work" in system:
            notes = messages[-1]["content"]
            m = re.search(r"\[list_tables \{\}\]\n([^\n]+)", notes)
            tables = [t.strip() for t in (m.group(1) if m else "orders").split(",") if t.strip()]
            table = "orders" if "orders" in tables else tables[0]
            sql = ("SELECT status, COUNT(*) AS orders FROM orders GROUP BY status" if table == "orders"
                   else f'SELECT COUNT(*) AS row_count FROM "{table}"')
            return LLMReply(content=f"```sql\n{sql}\n```")
        if "translate questions into sqlite" in system:
            return LLMReply(content="```sql\nSELECT * FROM my_table\n```")
        return LLMReply(content="Offline mode: start Ollama for real answers.")

    return FakeLLM(responder=responder)


if __name__ == "__main__":
    print(f"Built {build()}")
