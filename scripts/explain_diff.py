"""Why did a score change? Compare two results files and name what differs between them.

Each results file (metrics.json / benchmarks.json, Run 9 on) carries its serving settings,
grouped by what can change a score. This script shows which group changed (model / adapter,
decoding and stopping, prompt, post-processing, library versions) next to the score deltas.

    python scripts/explain_diff.py results_run9/sft results_run9/sft_plain
    python scripts/explain_diff.py results_run9/sft/metrics.json results_run6/sft/metrics.json --out DIFF.md
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from safequery.serving import diff_markdown, diff_results, load_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="results folder or metrics.json / benchmarks.json")
    ap.add_argument("b")
    ap.add_argument("--out", default=None, help="write the markdown here")
    ap.add_argument("--json", action="store_true", help="print the raw diff as JSON")
    args = ap.parse_args()

    d = diff_results(load_result(args.a), load_result(args.b))
    print(json.dumps(d, indent=2, default=str) if args.json else diff_markdown(d))
    if args.out:
        Path(args.out).write_text(diff_markdown(d), encoding="utf-8")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
