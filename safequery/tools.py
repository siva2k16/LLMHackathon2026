"""The four agent tools as plain Python functions.

AgentTune builds the tool schemas from these signatures and docstrings (via
`transformers.utils.get_json_schema`), so keep the Google-style `Args:` sections.
The functions delegate to a SafeQueryEnv bound with `bind_env()`.
"""

from __future__ import annotations

from .env import SafeQueryEnv

_ENV: SafeQueryEnv | None = None


def bind_env(env: SafeQueryEnv) -> None:
    global _ENV
    _ENV = env


def _env() -> SafeQueryEnv:
    if _ENV is None:
        raise RuntimeError("call safequery.tools.bind_env(env) before using the tools")
    return _ENV


def get_table_stats(table: str) -> str:
    """Get statistics for one table: row count, columns, indexes and whether it is large.

    Args:
        table: Name of the table, for example "orders".
    """
    return _env().call("get_table_stats", {"table": table})


def estimate_query_cost(sql: str) -> str:
    """Estimate the cost of a read-only SQL query without running it: query plan (full scan or index search), estimated rows examined, complexity and whether it exceeds the cost threshold.

    Args:
        sql: A single SQLite SELECT statement.
    """
    return _env().call("estimate_query_cost", {"sql": sql})


def run_sql_query(sql: str, intent: str) -> str:
    """Run a read-only SQL query after the safety guard and cost gate approve it, and return the result rows.

    Args:
        sql: A single SQLite SELECT statement.
        intent: The user's intent, either "lookup" (fetch specific records) or "aggregate" (counts, sums, rankings).
    """
    return _env().call("run_sql_query", {"sql": sql, "intent": intent})


def refuse_request(intent: str, reason: str) -> str:
    """Decline a request that would modify data, delete or drop objects, or expose restricted personal or system data.

    Args:
        intent: The request's intent, one of "write", "destructive" or "sensitive".
        reason: One short sentence explaining why the request is declined.
    """
    return _env().call("refuse_request", {"intent": intent, "reason": reason})


TOOLS = [get_table_stats, estimate_query_cost, run_sql_query, refuse_request]
TOOL_NAMES = [f.__name__ for f in TOOLS]
