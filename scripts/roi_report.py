"""ROI table. Token counts and throughput are read from an eval run; prices are flags.

    python scripts/roi_report.py --run results/sft --closed-in 3.0 --closed-out 15.0 \
        --gpu-name "NVIDIA L4" --gpu-price 0.80 --out results/ROI.md

Fill the prices in from the providers' current public pricing pages and state the
date in your write-up. Defaults are placeholders.
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from safequery.roi import RoiInputs, compute_roi, roi_markdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="results/<name> folder with metrics.json (for measured tokens)")
    ap.add_argument("--questions-per-month", type=int, default=100_000)
    ap.add_argument("--closed-in", type=float, default=3.0, help="USD per 1M input tokens (fill in)")
    ap.add_argument("--closed-out", type=float, default=15.0, help="USD per 1M output tokens (fill in)")
    ap.add_argument("--gpu-name", default="NVIDIA L4 (24GB)")
    ap.add_argument("--gpu-price", type=float, default=0.80, help="USD per GPU hour (fill in)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--tokens-per-sec", type=float, default=None, help="override measured throughput")
    ap.add_argument("--training-hours", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    x = RoiInputs(questions_per_month=args.questions_per_month, closed_price_in_per_m=args.closed_in,
                  closed_price_out_per_m=args.closed_out, gpu_name=args.gpu_name,
                  gpu_price_per_hour=args.gpu_price, concurrency=args.concurrency,
                  training_gpu_hours=args.training_hours, training_gpu_price_per_hour=args.gpu_price)
    if args.run:
        m = json.loads((Path(args.run) / "metrics.json").read_text())
        tp = m.get("throughput") or {}
        n_tasks = m["metrics"]["n_tasks"]
        if tp.get("model_calls"):
            x.llm_calls_per_question = round(tp["model_calls"] / n_tasks, 2)
            x.input_tokens_per_call = int(tp["prompt_tokens"] / tp["model_calls"])
            x.output_tokens_per_call = int(tp["generated_tokens"] / tp["model_calls"])
        if tp.get("tokens_per_sec"):
            x.tokens_per_sec = tp["tokens_per_sec"]
    if args.tokens_per_sec:
        x.tokens_per_sec = args.tokens_per_sec
    r = compute_roi(x)
    md = "## ROI: closed model vs self-hosted fine-tuned Tiny Aya\n\n" + roi_markdown(r)
    md += "\nAssumptions: prices are inputs (see flags); throughput and tokens per call are measured from the eval run.\n"
    print(md)
    print(json.dumps(r, indent=2))
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
