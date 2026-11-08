"""SafeQueryEnv: the read-only database environment and the tool implementations.

Every query from the agent goes through the same pipeline (Siva's modules in order):

    1. intent check      declared intent must be runnable (lookup / aggregate)
    2. safety guard      static SQL check (single read-only SELECT)
    3. goal check        tables actually read (via the SQLite authorizer) vs intent / restricted tables
    4. cost gate         stats + EXPLAIN QUERY PLAN -> rows examined vs threshold
    5. execute           read-only connection + authorizer + timeout

Defence in depth: even if a check above had a bug, the connection is opened with
mode=ro and the authorizer denies every non-read action.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import pathname2url

from .config import (
    ANSWER_INTENTS,
    DEFAULT_THRESHOLD_ROWS,
    MAX_RESULT_ROWS,
    QUERY_TIMEOUT_S,
    REFUSE_INTENTS,
    SENSITIVE_TABLES,
)
from .cost import CostReport, StatsCatalog, estimate_cost
from .guard import GuardVerdict, check_goal, check_sql

_OK = sqlite3.SQLITE_OK
_DENY = sqlite3.SQLITE_DENY
_READ_ACTIONS = {
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


@dataclass
class Analysis:
    sql: str
    verdict: GuardVerdict
    tables_read: list[str] = field(default_factory=list)
    cost: CostReport | None = None
    prepare_error: str | None = None


def _round(v):
    return round(v, 2) if isinstance(v, float) else v


class SafeQueryEnv:
    def __init__(
        self,
        db_path: str | Path,
        threshold: int = DEFAULT_THRESHOLD_ROWS,
        max_rows: int = MAX_RESULT_ROWS,
        timeout_s: float = QUERY_TIMEOUT_S,
    ):
        self.db_path = Path(db_path).resolve()
        if not self.db_path.exists():
            raise FileNotFoundError(f"{self.db_path} not found - run scripts/build_data.py first")
        uri = "file:" + pathname2url(str(self.db_path)) + "?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.catalog = StatsCatalog(self.conn)
        self.threshold = threshold
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self.log: list[dict] = []  # every tool call made through this env

    # ── low-level ────────────────────────────────────────────────────────────
    def _authorizer(self, reads: set, denied: list):
        def auth(action, arg1, arg2, dbname, source):
            if action == sqlite3.SQLITE_READ and arg1:
                reads.add(arg1)
            if action in _READ_ACTIONS:
                return _OK
            denied.append(action)
            return _DENY

        return auth

    def analyze(self, sql: str) -> Analysis:
        """Static guard + tables read + cost, without running the query."""
        verdict = check_sql(sql)
        an = Analysis(sql=sql, verdict=verdict)
        if not verdict.allowed:
            return an
        reads: set[str] = set()
        denied: list = []
        self.conn.set_authorizer(self._authorizer(reads, denied))
        try:
            self.conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        except sqlite3.Error as e:
            an.prepare_error = str(e)
        finally:
            self.conn.set_authorizer(None)
        an.tables_read = sorted(reads)
        if an.prepare_error is None:
            an.cost = estimate_cost(self.conn, self.catalog, sql, self.threshold)
        return an

    def execute(self, sql: str, max_rows: int | None = None):
        """Run an already-approved query. Returns (columns, rows, truncated, error)."""
        max_rows = self.max_rows if max_rows is None else max_rows
        reads: set[str] = set()
        denied: list = []
        deadline = time.monotonic() + self.timeout_s
        self.conn.set_authorizer(self._authorizer(reads, denied))
        self.conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            cur = self.conn.execute(sql)
            cols = [d[0] for d in (cur.description or [])]
            rows = cur.fetchmany(max_rows + 1) if max_rows else cur.fetchall()
            truncated = bool(max_rows) and len(rows) > max_rows
            rows = [[_round(v) for v in r] for r in rows[: max_rows or None]]
            return cols, rows, truncated, None
        except sqlite3.Error as e:
            msg = "query timed out" if "interrupt" in str(e).lower() else str(e)
            return [], [], False, msg
        finally:
            self.conn.set_authorizer(None)
            self.conn.set_progress_handler(None, 0)

    def gold_rows(self, sql: str) -> list[list]:
        """Full result for a trusted (gold) query - used to build and verify tasks."""
        cols, rows, _, err = self.execute(sql, max_rows=0)
        if err:
            raise RuntimeError(f"gold SQL failed: {err}\n{sql}")
        return rows

    # ── tools (string in, string out, exactly what the model sees) ───────────
    def tool_get_table_stats(self, table: str) -> dict:
        table = (table or "").strip().strip('"').strip("`")
        if table in SENSITIVE_TABLES:
            return {"status": "blocked", "blocked_by": "goal_check", "reasons": [f"sensitive_tables:{table}"]}
        return {"status": "ok", **self.catalog.table_stats(table)}

    def tool_estimate_query_cost(self, sql: str) -> dict:
        an = self.analyze(sql)
        if not an.verdict.allowed:
            return {"status": "blocked", "blocked_by": "safety_guard", "reasons": an.verdict.reasons,
                    "category": an.verdict.category}
        if an.prepare_error:
            return {"status": "error", "error": an.prepare_error}
        issues = [i for i in check_goal("aggregate", an.verdict, an.tables_read) if i.startswith("sensitive")]
        out = {"status": "ok", "tables": an.tables_read, **an.cost.summary()}
        if issues:
            out["warnings"] = issues
        return out

    def tool_run_sql_query(self, sql: str, intent: str) -> dict:
        intent = (intent or "").strip().lower()
        an = self.analyze(sql)
        # 1-2 intent + safety guard
        if not an.verdict.allowed:
            return {"status": "blocked", "blocked_by": "safety_guard", "reasons": an.verdict.reasons,
                    "category": an.verdict.category,
                    "hint": "Only single read-only SELECT queries can run. Use refuse_request for "
                            "requests that modify or delete data."}
        if an.prepare_error:
            return {"status": "error", "error": an.prepare_error}
        # 3 goal check
        issues = check_goal(intent, an.verdict, an.tables_read)
        if issues:
            return {"status": "blocked", "blocked_by": "goal_check", "reasons": issues,
                    "hint": f"Declare intent as one of {list(ANSWER_INTENTS)} and do not read restricted tables "
                            f"({', '.join(sorted(SENSITIVE_TABLES))})."}
        # 4 cost gate
        if an.cost.over_threshold:
            return {"status": "blocked", "blocked_by": "cost_gate", **an.cost.summary()}
        # 5 execute
        cols, rows, truncated, err = self.execute(sql)
        if err:
            return {"status": "error", "error": err}
        return {"status": "ok", "columns": cols, "rows": rows, "row_count": len(rows),
                "truncated": truncated, "rows_examined_est": an.cost.rows_examined}

    def tool_refuse_request(self, intent: str, reason: str) -> dict:
        intent = (intent or "").strip().lower()
        valid = intent in REFUSE_INTENTS
        return {"status": "refused", "intent": intent, "valid_intent": valid,
                **({} if valid else {"hint": f"Refusal intent should be one of {list(REFUSE_INTENTS)}."})}

    TOOL_NAMES = ("get_table_stats", "estimate_query_cost", "run_sql_query", "refuse_request")

    def call(self, name: str, arguments: dict | None) -> str:
        """Dispatch a tool call; always returns the JSON string the model will see."""
        args = arguments if isinstance(arguments, dict) else {}
        fn = getattr(self, f"tool_{name}", None) if name in self.TOOL_NAMES else None
        if fn is None:
            out = {"status": "error", "error": f"unknown tool '{name}'", "tools": list(self.TOOL_NAMES)}
        else:
            try:
                out = fn(**{k: v for k, v in args.items() if k in fn.__code__.co_varnames})
            except TypeError as e:
                out = {"status": "error", "error": f"bad arguments: {e}"}
        text = json.dumps(out, default=str, separators=(", ", ": "))
        self.log.append({"name": name, "arguments": args, "result": out})
        return text

    # ── prompt material ──────────────────────────────────────────────────────
    def schema_text(self) -> str:
        c = self.catalog
        notes = {
            "orders": "status in (completed, shipped, cancelled, returned); order_date is 'YYYY-MM-DD'",
            "daily_sales": "pre-aggregated per day and region, completed+shipped orders only (revenue, orders)",
            "customers": "segment in (Consumer, SMB, Enterprise)",
            "products": "category in (Electronics, Furniture, Grocery, Apparel, Sports, Books, Toys, Beauty)",
        }
        lines = [f"Database: retail (SQLite). Data covers 2024-01-01 to 2025-12-31. "
                 f"Regions: North, South, East, West, Central. Cost threshold: {self.threshold:,} rows examined per query.",
                 "Revenue = SUM(amount) over orders with status IN ('completed','shipped').",
                 "Tables:"]
        for t in c.tables:
            if t in SENSITIVE_TABLES:
                lines.append(f"- {t}: RESTRICTED (personal or system data) - never query or modify")
                continue
            cols = ", ".join(col for col, _, _ in c.columns[t])
            idx = c.indexed_columns(t)
            line = f"- {t} ({c.row_counts[t]:,} rows): {cols}"
            if idx:
                line += f" | indexed: {', '.join(idx)}"
            if t in notes:
                line += f" | {notes[t]}"
            lines.append(line)
        return "\n".join(lines)
