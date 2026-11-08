"""Before/after table from results/<run>/metrics.json (+ benchmarks.json if present).

    python scripts/compare.py --runs base sft grpo --out results/COMPARISON.md
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from safequery.evaluate import comparison_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs, bench = [], []
    for name in args.runs:
        d = Path(args.results) / name
        runs.append(json.loads((d / "metrics.json").read_text()))
        b = d / "benchmarks.json"
        bench.append(json.loads(b.read_text()) if b.exists() else {})
    md = "## SafeQuery benchmark (held-out test set)\n\n" + comparison_table(runs)

    keys = sorted({k for b in bench for k in b if k.startswith(("bfcl_", "arc_"))})
    if keys:
        md += "\n## Public benchmarks (AuditKIT)\n\n| Benchmark | " + " | ".join(args.runs) + " |\n"
        md += "|---|" + "---|" * len(args.runs) + "\n"
        for k in keys:
            metric_names = sorted({m for b in bench for m in (b.get(k) or {})})
            for m in metric_names:
                cells = []
                for b in bench:
                    v = (b.get(k) or {}).get(m)
                    cells.append("-" if v is None else f"{v:.3f}")
                md += f"| {k} · {m} | " + " | ".join(cells) + " |\n"
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
