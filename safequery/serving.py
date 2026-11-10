"""Serving identity: the settings that turn model weights into a reported score.

A score belongs to model + adapter + decoding + post-processing, not to the weights alone.
This module keeps those settings with every results file and makes them comparable:

  serving block   grouped by what can change a score: adapter, decoding, prompt,
                  post_processing, and versions (environment, should not change a score)
  serving_id      a short hash of the groups that can change a score, plus one hash per group,
                  so two results files can be compared at a glance
  provenance      code commit, adapter weight hash, test-set hash
  wrapper_report  the trimmed (reported) score next to the raw behaviour before trimming:
                  raw overrun rate, successes that needed the wrapper, and the success rate
                  with no wrapper intervention at all
  diff_results    which group changed between two results files, next to the score change
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from .prompts import parse_tool_calls

# Groups that can change a score. "versions" is recorded but kept out of serving_id: a library
# upgrade should not change behaviour, and if it does, diff_results still shows it.
SCORE_GROUPS = ("adapter", "decoding", "prompt", "post_processing")
GROUP_LABEL = {"adapter": "model / adapter", "decoding": "decoding and stopping",
               "prompt": "prompt", "post_processing": "post-processing", "versions": "library versions"}


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def file_sha(path: str | Path | None, n: int = 12) -> str | None:
    """Hash of a file (e.g. adapter_model.safetensors), so a reader can check they have the same weights."""
    if not path:
        return None
    p = Path(path)
    if p.is_dir():
        p = next((p / f for f in ("adapter_model.safetensors", "adapter_model.bin") if (p / f).exists()), p)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def git_commit(root: str | Path = ".") -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    sha = run("rev-parse", "HEAD")
    return {"commit": sha or None, "dirty": bool(run("status", "--porcelain", "--untracked-files=no")) if sha else None}


def with_ids(serving: dict) -> dict:
    """Add serving_id (score-relevant groups) and one id per group."""
    ids = {g: _h(serving.get(g)) for g in (*SCORE_GROUPS, "versions")}
    return {"serving_id": _h({g: serving.get(g) for g in SCORE_GROUPS}), "group_ids": ids, **serving}


def provenance(adapter: str | None, adapter_source: str | None, data_dir: str | Path | None,
               code_root: str | Path) -> dict:
    test = Path(data_dir) / "tasks_test.jsonl" if data_dir else None
    return {"code": git_commit(code_root),
            "adapter_weights_sha256": file_sha(adapter),
            "adapter_source": adapter_source,
            "test_set_sha256": file_sha(test) if test else None}


# ── raw behaviour vs reported score ──────────────────────────────────────────
def turn_flags(raw: str, returned: str) -> dict:
    """Would the agent loop have read this turn differently without the wrapper's trimming?"""
    c_raw, e_raw = parse_tool_calls(raw)
    c_ret, e_ret = parse_tool_calls(returned)
    return {"parse_rescued": bool(e_raw) and not e_ret,        # malformed raw, valid after trimming
            "parse_changed": (c_raw, e_raw) != (c_ret, e_ret)}  # trimming changed what the loop sees


def wrapper_report(raw_log: list[dict], scores) -> dict:
    """The trimmed (reported) score next to the raw behaviour that produced it.

    A turn "needed the wrapper" when the wrapper stopped it early (an extra end-of-turn token or
    the stop string) or trim_turn() changed it. An episode needed the wrapper when any of its
    turns did. success_without_intervention counts only successes where no turn needed it: the
    score the model earns on its own output, a lower bound on what a bare generate() would get.
    """
    by_task: dict = {}
    for r in raw_log:
        by_task.setdefault(r.get("task_id"), []).append(r)
    n = len(scores)
    needed = [any(t["wrapper_intervened"] for t in by_task.get(s.task_id, [])) for s in scores]
    overran = [any(t["overrun"] for t in by_task.get(s.task_id, [])) for s in scores]
    succ = [bool(s.success) for s in scores]
    rate = (lambda k: round(k / n, 4) if n else None)
    turns = len(raw_log)
    trate = (lambda k: round(k / turns, 4) if turns else None)
    return {
        "reported_task_success_rate": rate(sum(succ)),                       # the trimmed score
        "success_without_intervention_rate": rate(sum(s and not w for s, w in zip(succ, needed))),
        "successes_needing_wrapper": sum(s and w for s, w in zip(succ, needed)),
        "episodes_needing_wrapper_rate": rate(sum(needed)),
        "episodes_with_raw_overrun_rate": rate(sum(overran)),
        "turns": turns,
        "raw_overrun_rate": trate(sum(t["overrun"] for t in raw_log)),
        "turns_needing_wrapper_rate": trate(sum(t["wrapper_intervened"] for t in raw_log)),
        "parse_rescued_turns": sum(t["parse_rescued"] for t in raw_log),
        "parse_changed_turns": sum(t["parse_changed"] for t in raw_log),
        "stop_reasons": dict(Counter(t["stop_reason"] for t in raw_log)),
    }


# ── comparing two results files ──────────────────────────────────────────────
def _flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def load_result(path: str | Path) -> dict:
    """metrics.json or benchmarks.json, or a results folder holding metrics.json."""
    p = Path(path)
    if p.is_dir():
        p = p / "metrics.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["_path"] = str(p)
    return d


SCORE_KEYS = ("task_success_rate", "answer_accuracy", "refusal_accuracy", "injection_resistance",
              "unsafe_attempt_rate", "invalid_call_rate")


def diff_results(a: dict, b: dict) -> dict:
    """Which serving groups differ between two results, and how the scores moved."""
    sa, sb = a.get("serving"), b.get("serving")
    out: dict = {"a": a.get("name"), "b": b.get("name"), "a_path": a.get("_path"), "b_path": b.get("_path")}
    if not sa or not sb or "group_ids" not in sa or "group_ids" not in sb:
        out["serving_recorded"] = False
        out["changed_groups"] = None
    else:
        out["serving_recorded"] = True
        out["a_serving_id"], out["b_serving_id"] = sa["serving_id"], sb["serving_id"]
        changed = {}
        for g in (*SCORE_GROUPS, "versions"):
            fa, fb = _flat(sa.get(g) or {}), _flat(sb.get(g) or {})
            fields = {k: [fa.get(k), fb.get(k)] for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)}
            if fields:
                changed[g] = fields
        pa, pb = a.get("provenance") or {}, b.get("provenance") or {}
        for k in ("adapter_weights_sha256",):   # the weights decide; where they were downloaded from does not
            if pa.get(k) != pb.get(k) and (pa.get(k) or pb.get(k)):
                changed.setdefault("adapter", {})[f"provenance.{k}"] = [pa.get(k), pb.get(k)]
        if (pa.get("test_set_sha256") or pb.get("test_set_sha256")) and pa.get("test_set_sha256") != pb.get("test_set_sha256"):
            changed["test_set"] = {"test_set_sha256": [pa.get("test_set_sha256"), pb.get("test_set_sha256")]}
        out["changed_groups"] = changed
    ma, mb = a.get("metrics") or {}, b.get("metrics") or {}
    out["score_deltas"] = {k: [ma.get(k), mb.get(k)] for k in SCORE_KEYS if k in ma or k in mb}
    wa, wb = a.get("wrapper_report") or {}, b.get("wrapper_report") or {}
    out["wrapper_deltas"] = {k: [wa.get(k), wb.get(k)] for k in
                             ("success_without_intervention_rate", "raw_overrun_rate", "episodes_needing_wrapper_rate")
                             if k in wa or k in wb}
    return out


def _pct(v):
    return "-" if v is None else f"{100 * v:.1f}%"


def diff_markdown(d: dict) -> str:
    lines = [f"### {d['a']} → {d['b']}", ""]
    if not d["serving_recorded"]:
        lines.append("- Serving settings: **not recorded** in at least one file (results from before Run 8), "
                     "so a score change cannot be attributed.")
    else:
        lines.append(f"- serving_id: `{d['a_serving_id']}` → `{d['b_serving_id']}`")
        ch = d["changed_groups"]
        if not ch:
            lines.append("- Serving settings: **identical**. A score change here is run-to-run noise, not the setup.")
        else:
            score_groups = [g for g in ch if g != "versions"]
            lines.append("- Changed: " + (", ".join(f"**{GROUP_LABEL.get(g, g)}**" for g in score_groups)
                                          if score_groups else "library versions only"))
            for g, fields in ch.items():
                for k, (va, vb) in fields.items():
                    lines.append(f"  - {GROUP_LABEL.get(g, g)} · `{k}`: `{va}` → `{vb}`")
    lines += ["", "| Metric | " + str(d["a"]) + " | " + str(d["b"]) + " | Δ (pts) |", "|---|---|---|---|"]
    for k, (va, vb) in {**d["score_deltas"], **d["wrapper_deltas"]}.items():
        delta = "-" if va is None or vb is None else f"{100 * (vb - va):+.1f}"
        lines.append(f"| {k} | {_pct(va)} | {_pct(vb)} | {delta} |")
    return "\n".join(lines) + "\n"


__all__ = ["SCORE_GROUPS", "with_ids", "provenance", "file_sha", "turn_flags", "wrapper_report",
           "load_result", "diff_results", "diff_markdown"]
