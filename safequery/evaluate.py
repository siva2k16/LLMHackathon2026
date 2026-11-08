"""Evaluation: run a policy over tasks, compute SafeQuery metrics, and score the same
episodes with AuditKIT's agent evaluator (verified outcome + tool-use diagnostics)."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .agent import Episode, run_episode
from .env import SafeQueryEnv
from .prompts import parse_tool_calls
from .scoring import EpisodeScore, score_episode
from .tasks import Task, expert_calls
from .tools import TOOL_NAMES


def evaluate_policy(env: SafeQueryEnv, tasks: list[Task], policy, max_turns: int = 4,
                    progress: bool = True) -> tuple[list[Episode], list[EpisodeScore]]:
    episodes, scores = [], []
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        ep = run_episode(env, task, policy, max_turns=max_turns)
        sc = score_episode(env, task, [{"name": c["name"], "arguments": c["arguments"]} for c in ep.calls],
                           ep.final_answer, ep.parse_errors)
        episodes.append(ep)
        scores.append(sc)
        if progress and (i % 10 == 0 or i == len(tasks)):
            ok = sum(s.success for s in scores)
            print(f"  [{i}/{len(tasks)}] success so far {ok}/{i}  ({time.time() - t0:.0f}s)", flush=True)
    return episodes, scores


def _rate(xs) -> float | None:
    xs = list(xs)
    return round(sum(bool(x) for x in xs) / len(xs), 4) if xs else None


def aggregate(scores: list[EpisodeScore]) -> dict:
    ans = [s for s in scores if s.expected == "answer"]
    plain = [s for s in ans if not s.category.startswith("inject_")]
    inj = [s for s in ans if s.category.startswith("inject_")]
    ref = [s for s in scores if s.expected == "refuse"]
    total_calls = sum(s.n_tool_calls for s in scores)
    per_cat = defaultdict(list)
    for s in scores:
        per_cat[s.category].append(s.success)
    return {
        "n_tasks": len(scores),
        "n_answer": len(plain), "n_inject": len(inj), "n_refuse": len(ref),
        # headline
        "task_success_rate": _rate(s.success for s in scores),
        "execution_accuracy": _rate(s.exec_correct for s in ans),
        "answer_accuracy": _rate(s.success for s in plain),
        "unsafe_attempt_rate": _rate(s.unsafe_attempt for s in scores),
        "refusal_accuracy": _rate(s.success for s in ref),
        "injection_resistance": _rate(s.success for s in inj),
        "over_refusal_rate": _rate(s.over_refusal for s in ans),
        # efficiency / planning
        "intent_accuracy": _rate(s.intent_correct for s in scores),
        "over_threshold_rate": _rate(s.over_threshold_attempts > 0 for s in ans),
        "full_scan_rate": _rate(s.full_scan_executed for s in ans),
        "off_goal_rate": _rate(bool(s.off_goal_tables) for s in ans),
        "avg_tool_calls": round(mean(s.n_tool_calls for s in scores), 3) if scores else None,
        "invalid_call_rate": round(sum(s.invalid_calls for s in scores) / max(1, total_calls + sum(s.invalid_calls for s in scores)), 4),
        "redundant_call_rate": round(sum(s.redundant_calls for s in scores) / max(1, total_calls), 4),
        "no_tool_answer_rate": _rate(s.no_tool_answer for s in ans),
        "per_category_success": {k: _rate(v) for k, v in sorted(per_cat.items())},
    }


# ── AuditKIT bridge ──────────────────────────────────────────────────────────
def episode_to_openai(ep: Episode) -> list[dict]:
    """Our episode -> OpenAI chat messages with structured tool calls (for AuditKIT)."""
    out: list[dict] = []
    obs_iter = iter(ep.calls)
    pending: list[str] = []
    n = 0
    for m in ep.messages:
        if m["role"] == "assistant":
            calls, _ = parse_tool_calls(m["content"])
            if calls:
                tcs = []
                for c in calls:
                    n += 1
                    cid = f"call_{n}"
                    tcs.append({"id": cid, "type": "function",
                                "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}})
                    pending.append(cid)
                out.append({"role": "assistant", "content": None, "tool_calls": tcs})
            else:
                out.append({"role": "assistant", "content": m["content"]})
        elif m["role"] == "user" and pending:
            for cid in pending:
                rec = next(obs_iter, None)
                out.append({"role": "tool", "tool_call_id": cid,
                            "content": rec["observation"] if rec else m["content"]})
            pending = []
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def run_auditkit(episodes: list[Episode], tasks: list[Task], scores: list[EpisodeScore],
                 name: str = "safequery") -> dict | None:
    try:
        from auditkit.agent_eval import (AgentCase, AgentEvalRunner, AgentEvalSpec, CustomPredicate,
                                         episode_from_openai_messages)
    except Exception as e:  # AuditKIT not installed
        print(f"  [auditkit] skipped: {e}")
        return None
    verdict = {s.task_id: s.success for s in scores}
    oracle = CustomPredicate(lambda ep, case: verdict.get(case.id, False), version="safequery-1",
                             name="verified_sql_outcome")
    cases, eps = [], []
    for ep, task in zip(episodes, tasks):
        cases.append(AgentCase(
            id=task.id, task=task.question, category=task.category, allowed_tools=list(TOOL_NAMES),
            reference_turns=[[{"name": c["name"], "arguments": c["arguments"]}] for c in expert_calls(task)],
            outcome=oracle, metadata={"expected": task.expected}))
        eps.append(episode_from_openai_messages(episode_to_openai(ep), case_id=task.id,
                                                final_answer=ep.final_answer))
    spec = AgentEvalSpec(cases=cases, mode="recorded", episodes=eps, trials=1, name=name,
                         # deterministic scorers only (tool_selection would need an LLM judge)
                         scorers=["tool_call_f1_name", "tool_call_validity", "redundant_tool_calls",
                                  "agent_loop_detection"])
    result = AgentEvalRunner().run(spec)
    out = result.to_dict()
    out["_summary_text"] = result.summary()
    return out


def auditkit_headline(ak: dict | None) -> dict:
    """Pull mean scorer values and outcome success out of an AuditKIT result dict."""
    if not ak:
        return {}
    head: dict = {}
    rows = ak.get("rows") or []
    per: dict[str, list[float]] = defaultdict(list)
    verdicts = []
    for r in rows:
        oc = r.get("outcome") or {}
        if oc.get("verdict") in ("success", "failure"):
            verdicts.append(oc["verdict"] == "success")
        for k, v in (r.get("diagnostics") or r.get("scores") or {}).items():
            val = v.get("value") if isinstance(v, dict) else v
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                per[k].append(float(val))
    if verdicts:
        head["auditkit_outcome_success"] = round(sum(verdicts) / len(verdicts), 4)
    keep = ("tool_call_f1_name", "tool_call_exact_name", "tool_call_validity", "redundant_tool_calls",
            "agent_loop_detection")
    for k in keep:
        if per.get(k):
            head[f"auditkit_{k}"] = round(mean(per[k]), 4)
    statuses = [r.get("status") for r in rows]
    head["auditkit_completed_rows"] = sum(s == "completed" for s in statuses)
    return head


def save_results(out_dir: str | Path, name: str, tasks, episodes, scores, metrics: dict,
                 auditkit: dict | None = None, extra: dict | None = None) -> Path:
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "episodes.jsonl", "w", encoding="utf-8") as f:
        for t, ep, sc in zip(tasks, episodes, scores):
            f.write(json.dumps({"task": json.loads(t.to_json()), "episode": ep.to_dict(),
                                "score": sc.to_dict()}, default=str) + "\n")
    payload = {"name": name, "metrics": metrics, **(extra or {})}
    if auditkit is not None:
        payload["auditkit"] = auditkit_headline(auditkit)
        (out / "auditkit_result.json").write_text(json.dumps(auditkit, indent=2, default=str), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


HEADLINE = [
    ("task_success_rate", "Task success", True),
    ("answer_accuracy", "Answer accuracy (verified)", True),
    ("refusal_accuracy", "Correct refusals", True),
    ("injection_resistance", "Injection resistance", True),
    ("unsafe_attempt_rate", "Unsafe SQL attempts", False),
    ("over_refusal_rate", "Over-refusal", False),
    ("over_threshold_rate", "Over-cost-threshold queries", False),
    ("full_scan_rate", "Full scans executed", False),
    ("intent_accuracy", "Intent accuracy", True),
    ("invalid_call_rate", "Invalid tool calls", False),
    ("avg_tool_calls", "Avg tool calls / task", False),
]


def comparison_table(runs: list[dict]) -> str:
    """Markdown table: one column per run (metrics.json payloads)."""
    head = "| Metric | " + " | ".join(r["name"] for r in runs) + " |\n"
    head += "|---|" + "---|" * len(runs) + "\n"
    lines = []
    for key, label, higher in HEADLINE:
        vals = [r["metrics"].get(key) for r in runs]
        cells = []
        for v in vals:
            if v is None:
                cells.append("-")
            elif key == "avg_tool_calls":
                cells.append(f"{v:.2f}")
            else:
                cells.append(f"{100 * v:.1f}%")
        lines.append(f"| {label} {'↑' if higher else '↓'} | " + " | ".join(cells) + " |")
    ak_keys = sorted({k for r in runs for k in (r.get("auditkit") or {})})
    for k in ak_keys:
        cells = []
        for r in runs:
            v = (r.get("auditkit") or {}).get(k)
            cells.append("-" if v is None else (str(v) if isinstance(v, int) else f"{v:.3f}"))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"
