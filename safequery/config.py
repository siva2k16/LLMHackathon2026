"""Central configuration for SafeQuery. Every tunable constant lives here."""

from __future__ import annotations

# ── Cost gate ────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD_ROWS = 60_000   # max estimated rows a single query may examine
LARGE_TABLE_ROWS = 10_000         # a full scan on a table this big is reported as a "full scan"
MAX_RESULT_ROWS = 20              # rows returned to the model per query
QUERY_TIMEOUT_S = 3.0             # hard wall-clock limit per query

# ── Safety ───────────────────────────────────────────────────────────────────
SENSITIVE_TABLES = frozenset({"customer_pii", "audit_log"})

# ── Intents ──────────────────────────────────────────────────────────────────
ANSWER_INTENTS = ("lookup", "aggregate")              # the agent may run a query
REFUSE_INTENTS = ("write", "destructive", "sensitive")  # the agent must refuse
ALL_INTENTS = ANSWER_INTENTS + REFUSE_INTENTS

# ── Business definitions ─────────────────────────────────────────────────────
REVENUE_STATUSES = ("completed", "shipped")

# ── Models (allowed Cohere models only, see hackathon rules) ─────────────────
DEFAULT_MODEL = "CohereLabs/tiny-aya-global"
