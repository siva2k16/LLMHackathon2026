"""Stats, query-plan analysis and cost estimation (Siva's ideas #2-#6).

  stats if present -> StatsCatalog reads sqlite_stat1 (falls back to COUNT(*)) and
                      builds equi-depth histograms on indexed columns
  planner check    -> SQLite's own EXPLAIN QUERY PLAN
  full scan/filter/index -> each plan step is classified: full_scan, covering_index_scan,
                      index_search, pk_lookup
  data complexity  -> estimated rows examined, temp sorts, joins -> low / medium / high
  estimated cost   -> cost units = rows examined (weighted) + sort overhead
  threshold        -> over_threshold when estimated rows examined > threshold

The estimator is a heuristic, like every query optimiser's. It is deliberately simple
and fully deterministic so it can be used inside a training reward.
"""

from __future__ import annotations

import re
import sqlite3
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field

from .config import DEFAULT_THRESHOLD_ROWS, LARGE_TABLE_ROWS

_HIST_BUCKETS = 64
_DEFAULT_RANGE_SELECTIVITY = 0.25
_PLAN_RE = re.compile(r"^(SCAN|SEARCH)\s+(?:TABLE\s+)?(\w+)(?:\s+AS\s+(\w+))?(.*)$", re.I)
_ALIAS_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(\w+)(?:\s+(?:AS\s+)?(?!(?:ON|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|CROSS|"
    r"GROUP|ORDER|LIMIT|USING|NATURAL|HAVING|UNION)\b)(\w+))?",
    re.I,
)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class StatsCatalog:
    """Table statistics for the cost model. Built once per database connection."""

    def __init__(self, conn: sqlite3.Connection, buckets: int = _HIST_BUCKETS):
        self.tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        self.has_stat1 = bool(
            conn.execute("SELECT 1 FROM sqlite_master WHERE name='sqlite_stat1'").fetchone()
        )
        self.row_counts: dict[str, int] = {}
        self.stats_source: dict[str, str] = {}
        self.per_key: dict[str, float] = {}  # index -> avg rows per value of its first column
        self.columns: dict[str, list[tuple[str, str, bool]]] = {}  # table -> [(col, type, is_pk)]
        self.indexes: dict[str, dict[str, list[str]]] = {}  # table -> {index: [cols]}
        self.histograms: dict[tuple[str, str], list] = {}

        if self.has_stat1:
            for tbl, idx, stat in conn.execute("SELECT tbl, idx, stat FROM sqlite_stat1"):
                nums = [int(x) for x in str(stat).split() if x.isdigit()]
                if not nums:
                    continue
                self.row_counts[tbl] = nums[0]
                self.stats_source[tbl] = "sqlite_stat1"
                if idx and len(nums) > 1:
                    self.per_key[idx] = float(nums[1])

        for t in self.tables:
            if t not in self.row_counts:
                self.row_counts[t] = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                self.stats_source[t] = "count"
            self.columns[t] = [
                (r[1], (r[2] or "").upper(), bool(r[5])) for r in conn.execute(f'PRAGMA table_info("{t}")')
            ]
            idx_map: dict[str, list[str]] = {}
            for r in conn.execute(f'PRAGMA index_list("{t}")'):
                name = r[1]
                cols = [c[2] for c in conn.execute(f'PRAGMA index_info("{name}")')]
                idx_map[name] = cols
            self.indexes[t] = idx_map
            # histograms on the leading column of every index on large tables
            if self.row_counts[t] >= 1_000:
                for cols in idx_map.values():
                    col = cols[0]
                    if (t, col) in self.histograms:
                        continue
                    vals = [r[0] for r in conn.execute(f'SELECT "{col}" FROM "{t}" WHERE "{col}" IS NOT NULL ORDER BY 1')]
                    if vals:
                        step = max(1, len(vals) // buckets)
                        self.histograms[(t, col)] = vals[::step] + [vals[-1]]

    # ── helpers ──────────────────────────────────────────────────────────────
    def indexed_columns(self, table: str) -> list[str]:
        cols = {c[0] for c in self.columns.get(table, []) if c[2]}
        for idx_cols in self.indexes.get(table, {}).values():
            cols.add(idx_cols[0])
        return sorted(cols)

    def range_fraction(self, table: str, col: str, lo=None, hi=None) -> float | None:
        """Fraction of rows with lo <= col < hi, from the histogram (None if unknown)."""
        hist = self.histograms.get((table, col))
        if not hist or (lo is None and hi is None):
            return None
        numeric = _num(hist[0]) is not None
        key = (lambda v: _num(v)) if numeric else (lambda v: str(v))
        h = [key(v) for v in hist]
        lo_i = bisect_left(h, key(lo)) if lo is not None else 0
        hi_i = bisect_right(h, key(hi)) if hi is not None else len(h)
        return max(0.0, min(1.0, (hi_i - lo_i) / max(1, len(h) - 1)))

    def table_stats(self, table: str) -> dict:
        if table not in self.row_counts:
            return {"error": f"unknown table '{table}'", "tables": self.tables}
        return {
            "table": table,
            "rows": self.row_counts[table],
            "stats_source": self.stats_source[table],
            "columns": [f"{c} {t}{' PK' if pk else ''}".strip() for c, t, pk in self.columns[table]],
            "indexes": {k: v for k, v in self.indexes[table].items() if not k.startswith("sqlite_autoindex")},
            "indexed_columns": self.indexed_columns(table),
            "large_table": self.row_counts[table] >= LARGE_TABLE_ROWS,
        }


@dataclass
class PlanStep:
    detail: str
    op: str  # SCAN | SEARCH | SORT | OTHER
    table: str | None = None
    access: str = "other"  # full_scan | covering_index_scan | index_search | pk_lookup | other
    index: str | None = None
    est_rows: float = 0.0


@dataclass
class CostReport:
    ok: bool
    steps: list[PlanStep] = field(default_factory=list)
    rows_examined: int = 0
    cost_units: float = 0.0
    full_scans: list[str] = field(default_factory=list)  # large tables scanned in full
    uses_index: bool = False
    temp_sorts: int = 0
    n_tables: int = 0
    complexity: str = "low"
    threshold: int = DEFAULT_THRESHOLD_ROWS
    over_threshold: bool = False
    suggestion: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [{"detail": s.detail, "access": s.access, "est_rows": int(s.est_rows)} for s in self.steps]
        return d

    def summary(self) -> dict:
        """Compact view returned to the model."""
        return {
            "plan": [f"{s.access}:{s.table}" + (f"({s.index})" if s.index else "") for s in self.steps if s.table],
            "rows_examined_est": self.rows_examined,
            "complexity": self.complexity,
            "threshold": self.threshold,
            "over_threshold": self.over_threshold,
            **({"full_scans": self.full_scans} if self.full_scans else {}),
            **({"suggestion": self.suggestion} if self.suggestion else {}),
        }


def explain(conn: sqlite3.Connection, sql: str) -> list[str]:
    return [str(r[3]) for r in conn.execute("EXPLAIN QUERY PLAN " + sql)]


def _alias_map(sql: str, tables: list[str]) -> dict[str, str]:
    lower_tables = {t.lower(): t for t in tables}
    out = {t.lower(): t for t in tables}
    for m in _ALIAS_RE.finditer(sql):
        tbl, alias = m.group(1), m.group(2)
        if tbl and tbl.lower() in lower_tables and alias:
            out[alias.lower()] = lower_tables[tbl.lower()]
    return out


def _bounds_for(sql: str, col: str):
    """Pull literal bounds on `col` out of the SQL text (for histogram selectivity)."""
    lo = hi = None
    pat_col = rf"(?:\b\w+\.)?\b{re.escape(col)}\b"
    m = re.search(pat_col + r"\s+BETWEEN\s+'?([\w\-:. ]+?)'?\s+AND\s+'?([\w\-:. ]+?)'?(?:\s|$|\))", sql, re.I)
    if m:
        return m.group(1), m.group(2)
    for op, val in re.findall(pat_col + r"\s*(>=|<=|>|<|=)\s*'?([\w\-:.]+)'?", sql, re.I):
        if op in (">", ">="):
            lo = val
        elif op in ("<", "<="):
            hi = val
        elif op == "=":
            lo = hi = val
    return lo, hi


def estimate_cost(
    conn: sqlite3.Connection,
    catalog: StatsCatalog,
    sql: str,
    threshold: int = DEFAULT_THRESHOLD_ROWS,
) -> CostReport:
    """Estimate rows examined for `sql` using SQLite's plan plus catalog statistics."""
    try:
        details = explain(conn, sql)
    except sqlite3.Error as e:
        return CostReport(ok=False, threshold=threshold, error=str(e))

    aliases = _alias_map(sql, catalog.tables)
    rep = CostReport(ok=True, threshold=threshold)
    driving: float | None = None
    tables_seen: set[str] = set()

    for d in details:
        m = _PLAN_RE.match(d.strip())
        if not m:
            if "TEMP B-TREE" in d.upper():
                rep.temp_sorts += 1
                rep.steps.append(PlanStep(d, "SORT"))
            else:
                rep.steps.append(PlanStep(d, "OTHER"))
            continue
        op, name, _as, rest = m.group(1).upper(), m.group(2), m.group(3), m.group(4) or ""
        table = aliases.get(name.lower(), aliases.get((_as or "").lower(), name))
        n = float(catalog.row_counts.get(table, 1_000))
        rest_u = rest.upper()
        idx_m = re.search(r"INDEX (\w+)", rest, re.I)
        index = idx_m.group(1) if idx_m else None
        cons_m = re.search(r"\((.*)\)", rest)
        cons = cons_m.group(1) if cons_m else ""
        weight = 1.0

        if op == "SCAN":
            if "COVERING INDEX" in rest_u:
                access, est, weight = "covering_index_scan", n, 0.5
            else:
                access, est = "full_scan", n
                if n >= LARGE_TABLE_ROWS:
                    rep.full_scans.append(table)
        elif "PRIMARY KEY" in rest_u and "=" in cons and "<" not in cons and ">" not in cons:
            access, est = "pk_lookup", 1.0
        else:
            access = "index_search"
            first_col = None
            if index and "AUTOMATIC" not in rest_u:
                idx_cols = catalog.indexes.get(table, {}).get(index)
                first_col = idx_cols[0] if idx_cols else None
            elif "PRIMARY KEY" in rest_u:
                first_col = next((c[0] for c in catalog.columns.get(table, []) if c[2]), None)
            if first_col and re.search(rf"\b{re.escape(first_col)}=\?", cons) and "AUTOMATIC" not in rest_u:
                est = catalog.per_key.get(index, max(1.0, n / 100.0)) if index else 1.0
            elif first_col and re.search(rf"\b{re.escape(first_col)}[<>]", cons):
                lo, hi = _bounds_for(sql, first_col)
                frac = catalog.range_fraction(table, first_col, lo, hi)
                est = n * (frac if frac is not None else _DEFAULT_RANGE_SELECTIVITY)
            else:
                est = max(1.0, n / 100.0)
            if "AUTOMATIC" in rest_u:
                rep.cost_units += n  # building the automatic index reads the table once
                est = 1.0
            rep.uses_index = True

        if access in ("covering_index_scan",):
            rep.uses_index = True
        if driving is None:
            driving, total = est, est
        else:
            total = driving * est  # nested-loop: one probe per outer row
        tables_seen.add(table)
        rep.steps.append(PlanStep(d, op, table, access, index, total))
        rep.rows_examined += int(total)
        rep.cost_units += total * weight

    rep.n_tables = len(tables_seen)
    rep.cost_units += rep.temp_sorts * (driving or 0) * 0.1
    rep.cost_units = round(rep.cost_units, 1)
    rep.over_threshold = rep.rows_examined > threshold
    if rep.rows_examined < 5_000:
        rep.complexity = "low"
    elif rep.rows_examined <= threshold:
        rep.complexity = "medium"
    else:
        rep.complexity = "high"
    if rep.over_threshold or rep.full_scans:
        rep.suggestion = _suggest(catalog, rep, sql)
    return rep


def _suggest(catalog: StatsCatalog, rep: CostReport, sql: str) -> str:
    tips: list[str] = []
    for t in dict.fromkeys(rep.full_scans or [s.table for s in rep.steps if s.table]):
        cols = catalog.indexed_columns(t)
        if cols:
            tips.append(f"Filter {t} on an indexed column ({', '.join(cols)}) with a range or equality condition.")
    if re.search(r"\b(strftime|substr|date|LIKE)\b", sql, re.I):
        tips.append("Functions or LIKE on a column prevent index use; compare the raw column, "
                    "e.g. order_date >= '2025-01-01' AND order_date < '2026-01-01'.")
    if "orders" in {s.table for s in rep.steps} and "daily_sales" in catalog.tables:
        tips.append("For revenue or order totals by date/region use the pre-aggregated daily_sales table.")
    return " ".join(dict.fromkeys(tips))
