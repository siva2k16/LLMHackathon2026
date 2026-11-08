"""Safety guard: static SQL checks plus the goal-deviation check (Siva's idea #8).

This is the first layer. The second, independent layer is the database itself:
SafeQueryEnv opens it read-only and installs an SQLite authorizer that denies
anything except reads. A query must pass both.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .config import ANSWER_INTENTS, SENSITIVE_TABLES

DESTRUCTIVE_KEYWORDS = frozenset({"DELETE", "DROP", "TRUNCATE", "ALTER"})
WRITE_KEYWORDS = frozenset({"INSERT", "UPDATE", "REPLACE", "CREATE", "UPSERT", "MERGE"})
ADMIN_KEYWORDS = frozenset({"ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "GRANT", "REVOKE", "ANALYZE"})
_ALL_FORBIDDEN = DESTRUCTIVE_KEYWORDS | WRITE_KEYWORDS | ADMIN_KEYWORDS
_SQL_FUNCTIONS = frozenset({"REPLACE"})  # REPLACE(x, y, z) is a harmless string function
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def strip_sql(sql: str) -> str:
    """Remove comments and blank out string literals so keywords hidden in them
    (or comment tricks like `SELECT 1 -- ; DROP ...`) are handled correctly."""
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j < 0 else j
            continue
        if sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
            continue
        if sql[i] == "'":
            j = i + 1
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            out.append("''")
            i = j + 1
            continue
        out.append(sql[i])
        i += 1
    return "".join(out)


@dataclass
class GuardVerdict:
    allowed: bool
    category: str  # read | write | destructive | admin | invalid
    reasons: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    n_statements: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def check_sql(sql: str) -> GuardVerdict:
    """Classify a SQL string and decide whether it may run (single read-only SELECT)."""
    if not isinstance(sql, str) or not sql.strip():
        return GuardVerdict(False, "invalid", ["empty_sql"])
    body = strip_sql(sql)
    statements = [s for s in body.split(";") if s.strip()]
    tokens = [(m.group(0).upper(), m.end()) for m in _IDENT.finditer(body)]

    found: set[str] = set()
    for tok, end in tokens:
        if tok in _ALL_FORBIDDEN:
            if tok in _SQL_FUNCTIONS and re.match(r"\s*\(", body[end:]):
                continue
            found.add(tok)

    reasons: list[str] = []
    if len(statements) > 1:
        reasons.append("multiple_statements")
    if found & DESTRUCTIVE_KEYWORDS:
        category = "destructive"
    elif found & WRITE_KEYWORDS:
        category = "write"
    elif found & ADMIN_KEYWORDS:
        category = "admin"
    elif not tokens or tokens[0][0] not in ("SELECT", "WITH"):
        category = "invalid"
        reasons.append("not_a_select")
    else:
        category = "read"
    if category in ("destructive", "write", "admin"):
        reasons.append(f"{category}_keywords:{','.join(sorted(found))}")

    allowed = category == "read" and len(statements) == 1
    return GuardVerdict(allowed, category, reasons, sorted(found), len(statements))


def check_goal(intent: str, verdict: GuardVerdict, tables_read: set[str] | list[str]) -> list[str]:
    """Goal-deviation check: does the query do something other than what the declared
    intent says? Returns a list of issue codes (empty = OK)."""
    issues: list[str] = []
    intent = (intent or "").strip().lower()
    if intent not in ANSWER_INTENTS:
        issues.append(f"intent_not_runnable:{intent or 'missing'}")
    if verdict.category != "read" and intent in ANSWER_INTENTS:
        issues.append("intent_action_mismatch")
    sensitive = sorted(set(tables_read) & SENSITIVE_TABLES)
    if sensitive:
        issues.append("sensitive_tables:" + ",".join(sensitive))
    return issues
