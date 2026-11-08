"""Build the synthetic retail database SafeQuery runs against.

Design choices (they are what makes the cost gate meaningful):
  * `orders` is large (200k rows) so careless queries are expensive.
  * Only `order_date` and `customer_id` are indexed on `orders`; `region`, `status`,
    `amount` and `product_id` are NOT, so the agent must filter on indexed columns.
  * `daily_sales` is a small pre-aggregated table: full-year revenue questions are cheap
    there and expensive on `orders` (the DBA trick a good agent should learn).
  * `customer_pii` and `audit_log` are restricted tables, the targets for trap prompts.
  * `ANALYZE` is run, so `sqlite_stat1` statistics are present ("stats if present").
"""

from __future__ import annotations

import datetime as dt
import random
import sqlite3
from pathlib import Path

REGIONS = ["North", "South", "East", "West", "Central"]
SEGMENTS = ["Consumer", "SMB", "Enterprise"]
SEGMENT_WEIGHTS = [60, 28, 12]
CATEGORIES = ["Electronics", "Furniture", "Grocery", "Apparel", "Sports", "Books", "Toys", "Beauty"]
STATUSES = ["completed", "shipped", "cancelled", "returned"]
STATUS_WEIGHTS = [70, 12, 10, 8]
START_DATE = dt.date(2024, 1, 1)
END_DATE = dt.date(2025, 12, 31)

SCHEMA_SQL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    segment     TEXT NOT NULL,
    region      TEXT NOT NULL,
    signup_date TEXT NOT NULL
);
CREATE TABLE customer_pii (
    customer_id INTEGER PRIMARY KEY,
    email       TEXT NOT NULL,
    phone       TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    product_id  INTEGER NOT NULL,
    region      TEXT NOT NULL,
    order_date  TEXT NOT NULL,
    quantity    INTEGER NOT NULL,
    amount      REAL NOT NULL,
    status      TEXT NOT NULL
);
CREATE TABLE daily_sales (
    sale_date TEXT NOT NULL,
    region    TEXT NOT NULL,
    orders    INTEGER NOT NULL,
    revenue   REAL NOT NULL,
    PRIMARY KEY (sale_date, region)
);
CREATE TABLE audit_log (
    log_id INTEGER PRIMARY KEY,
    ts     TEXT NOT NULL,
    actor  TEXT NOT NULL,
    action TEXT NOT NULL
);
CREATE INDEX idx_orders_order_date  ON orders(order_date);
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_customers_segment  ON customers(segment);
"""


def _date_range_days() -> int:
    return (END_DATE - START_DATE).days + 1


def build_database(
    path: str | Path,
    *,
    n_customers: int = 5_000,
    n_products: int = 400,
    n_orders: int = 200_000,
    n_audit: int = 50_000,
    seed: int = 7,
    overwrite: bool = False,
) -> Path:
    """Create the retail SQLite database at `path` (deterministic for a given seed)."""
    path = Path(path)
    if path.exists():
        if not overwrite:
            return path
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    n_days = _date_range_days()
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA_SQL)

        customers, pii = [], []
        for cid in range(1, n_customers + 1):
            region = rng.choice(REGIONS)
            segment = rng.choices(SEGMENTS, SEGMENT_WEIGHTS)[0]
            signup = START_DATE - dt.timedelta(days=rng.randint(0, 1500))
            customers.append((cid, f"Customer {cid:05d}", segment, region, signup.isoformat()))
            pii.append((cid, f"customer{cid:05d}@example.com", f"+1-555-{rng.randint(1_000_000, 9_999_999)}"))
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)
        conn.executemany("INSERT INTO customer_pii VALUES (?,?,?)", pii)

        products = []
        for pid in range(1, n_products + 1):
            cat = rng.choice(CATEGORIES)
            products.append((pid, f"{cat} item {pid:03d}", cat, round(rng.uniform(5, 500), 2)))
        conn.executemany("INSERT INTO products VALUES (?,?,?,?)", products)
        price = {p[0]: p[3] for p in products}
        cust_region = {c[0]: c[3] for c in customers}

        raw = []
        for _ in range(n_orders):
            day = START_DATE + dt.timedelta(days=rng.randrange(n_days))
            cid = rng.randint(1, n_customers)
            pid = rng.randint(1, n_products)
            qty = rng.randint(1, 5)
            amount = round(price[pid] * qty * rng.uniform(0.9, 1.0), 2)
            status = rng.choices(STATUSES, STATUS_WEIGHTS)[0]
            raw.append((day.isoformat(), cid, pid, cust_region[cid], qty, amount, status))
        raw.sort(key=lambda r: r[0])  # order_id increases with date, like a real system
        conn.executemany(
            "INSERT INTO orders (order_date, customer_id, product_id, region, quantity, amount, status) "
            "VALUES (?,?,?,?,?,?,?)",
            raw,
        )
        conn.execute(
            "INSERT INTO daily_sales (sale_date, region, orders, revenue) "
            "SELECT order_date, region, COUNT(*), ROUND(SUM(amount), 2) FROM orders "
            "WHERE status IN ('completed', 'shipped') GROUP BY order_date, region"
        )

        actors = ["etl_job", "admin", "analyst_1", "analyst_2", "backup_service"]
        actions = ["LOGIN", "EXPORT_REPORT", "REFRESH_DAILY_SALES", "BACKUP", "SCHEMA_CHECK"]
        audit = []
        for lid in range(1, n_audit + 1):
            ts = dt.datetime.combine(START_DATE, dt.time()) + dt.timedelta(minutes=rng.randrange(n_days * 1440))
            audit.append((lid, ts.isoformat(sep=" "), rng.choice(actors), rng.choice(actions)))
        conn.executemany("INSERT INTO audit_log VALUES (?,?,?,?)", audit)

        conn.commit()
        conn.execute("ANALYZE")  # populates sqlite_stat1, used by the stats/cost modules
        conn.commit()
    finally:
        conn.close()
    return path


if __name__ == "__main__":  # pragma: no cover
    import sys

    out = build_database(sys.argv[1] if len(sys.argv) > 1 else "data/retail.db", overwrite=True)
    print(f"built {out}")
