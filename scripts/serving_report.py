"""Run 9 report: the trimmed (reported) score next to the raw behaviour before trimming.

For each model it shows, under the safequery wrapper (the reported settings) and plain
(bare generate()):
  - reported task success (after trimming) and task success with no wrapper intervention
  - how many successes needed the wrapper, raw overrun rate, stop reasons
  - the serving_id each row was produced under
then, for each model, what changed between the two wrappers (explain_diff), and a check
that the safequery rows equal an earlier run with the same adapters.

    python scripts/serving_report.py --results results_run9 --models base sft grpo \
        --baseline results_run6 --out results_run9/WRAPPERS.md
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from safequery.serving import diff_markdown, diff_results, load_result

SCORE = [("task_success_rate", "Task success (reported, trimmed)"), ("answer_accuracy", "Answer acc."),
         ("refusal_accuracy", "Refusals"), ("injection_resistance", "Injection res."),
         ("unsafe_attempt_rate", "Unsafe attempts"), ("invalid_call_rate", "Invalid calls")]
RAW = [("success_without_intervention_rate", "Task success, no wrapper intervention"),
       ("successes_needing_wrapper", "Successes needing wrapper"),
       ("raw_overrun_rate", "Raw overrun (turns)"),
       ("episodes_with_raw_overrun_rate", "Episodes with raw overrun"),
       ("parse_rescued_turns", "Turns rescued by trimming")]
COMPARE_KEYS = [k for k, _ in SCORE]


def _fmt(v):
    if v is None:
        return "-"
    return str(v) if isinstance(v, int) else f"{100 * v:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--models", nargs="+", default=["base", "sft", "grpo"])
    ap.add_argument("--baseline", default=None, help="earlier results folder to check the safequery rows against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = Path(args.results)

    rows = ["| Model | Wrapper | " + " | ".join(c for _, c in SCORE + RAW) + " | Stop reasons | serving_id |",
            "|---|---|" + "---|" * (len(SCORE) + len(RAW) + 2)]
    found = {}
    for m in args.models:
        for w in ("safequery", "plain"):
            name = m if w == "safequery" else f"{m}_plain"
            p = res / name / "metrics.json"
            if not p.exists():
                continue
            d = load_result(p)
            found[(m, w)] = d
            M, R = d["metrics"], d.get("wrapper_report") or {}
            stops = ", ".join(f"{k} {v}" for k, v in sorted((R.get("stop_reasons") or {}).items()))
            rows.append(f"| {m} | {w} | " + " | ".join(_fmt(M.get(k)) for k, _ in SCORE) + " | "
                        + " | ".join(_fmt(R.get(k)) for k, _ in RAW)
                        + f" | {stops} | `{(d.get('serving') or {}).get('serving_id', '-')}` |")

    md = ["## Run 9: reported score next to the raw behaviour before trimming", "",
          "*Reported* = the score after the safequery wrapper (stop tokens + stop string + `trim_turn`). "
          "*No wrapper intervention* counts only successes where no turn was stopped early or trimmed by the "
          "wrapper: a lower bound on what the model earns on its own output. *Plain* = a bare `generate()`.", "",
          *rows, ""]

    md += ["## What changed between the two wrappers", ""]
    for m in args.models:
        if (m, "safequery") in found and (m, "plain") in found:
            md.append(diff_markdown(diff_results(found[(m, "safequery")], found[(m, "plain")])))

    bench = []
    for m in args.models:
        for w in ("safequery", "plain"):
            p = res / (m if w == "safequery" else f"{m}_plain") / "benchmarks.json"
            if p.exists():
                b = json.loads(p.read_text(encoding="utf-8"))
                ws = b.get("wrapper_stats") or {}
                cells = [f"{k}·{mn} {v:.3f}" for k in sorted(b) if k.startswith(("bfcl_", "arc_"))
                         for mn, v in sorted((b[k] or {}).items()) if isinstance(v, (int, float))]
                bench.append(f"| {m} | {w} | " + "<br>".join(cells) + f" | {_fmt(ws.get('raw_overrun_rate'))} | "
                             f"`{(b.get('serving') or {}).get('serving_id', '-')}` |")
    if bench:
        md += ["## Public benchmarks (BFCL / ARC) under each wrapper", "",
               "| Model | Wrapper | Scores | "
               "Raw overrun | serving_id |", "|---|---|---|---|---|", *bench, ""]

    if args.baseline:
        md += [f"## Check: safequery rows vs `{args.baseline}`", "",
               "Same adapters and settings, so the trimmed scores should match. "
               "The logging added in Run 9 must not change behaviour.", ""]
        for m in args.models:
            p0 = Path(args.baseline) / m / "metrics.json"
            if (m, "safequery") in found and p0.exists():
                m0, m1 = load_result(p0)["metrics"], found[(m, "safequery")]["metrics"]
                diff = [f"{k} {_fmt(m0.get(k))}→{_fmt(m1.get(k))}" for k in COMPARE_KEYS
                        if m0.get(k) is not None and abs(m0[k] - m1[k]) > 1e-9]
                md.append(f"- {m}: " + ("identical" if not diff else "DIFFERENT: " + ", ".join(diff)))
        md.append("")

    first = next(iter(found.values()), None)
    if first and first.get("serving"):
        md += ["## Serving settings recorded with each results file (example: "
               f"`{first['name']}`)", "", "```json",
               json.dumps({"serving": first["serving"], "provenance": first.get("provenance")}, indent=2), "```", ""]
    text = "\n".join(md)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
