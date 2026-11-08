"""Evaluate a real model (base or + LoRA adapter) on the SafeQuery test set, with AuditKIT.

    python scripts/eval_model.py --name base     --model CohereLabs/tiny-aya-global
    python scripts/eval_model.py --name sft      --model CohereLabs/tiny-aya-global --adapter runs/sft
    python scripts/eval_model.py --name grpo     --model runs/sft_merged --adapter runs/grpo
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from safequery.config import DEFAULT_MODEL
from safequery.env import SafeQueryEnv
from safequery.evaluate import aggregate, auditkit_headline, evaluate_policy, run_auditkit, save_results
from safequery.policies import HFPolicy
from safequery.tasks import Task
from safequery.tools import bind_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name, e.g. base / sft / grpo")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="LoRA adapter folder (optional)")
    ap.add_argument("--load-4bit", action="store_true", help="4-bit weights (saves GPU memory)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N test tasks")
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args()

    data = Path(args.data)
    env = SafeQueryEnv(data / "retail.db")
    bind_env(env)
    tasks = [Task.from_json(line) for line in open(data / "tasks_test.jsonl", encoding="utf-8")]
    if args.limit:
        tasks = tasks[: args.limit]

    policy = HFPolicy(args.model, adapter=args.adapter, load_4bit=args.load_4bit,
                      max_new_tokens=args.max_new_tokens)
    print(f"[eval] {args.name}: {policy.name} on {len(tasks)} tasks")
    eps, scs = evaluate_policy(env, tasks, policy, max_turns=args.max_turns)
    metrics = aggregate(scs)
    ak = run_auditkit(eps, tasks, scs, name=args.name)
    extra = {"model": args.model, "adapter": args.adapter, "load_4bit": args.load_4bit,
             "throughput": policy.throughput()}
    out = save_results(args.results, args.name, tasks, eps, scs, metrics, ak, extra)
    print(json.dumps({**{k: metrics[k] for k in ("task_success_rate", "answer_accuracy", "refusal_accuracy",
                                                 "injection_resistance", "unsafe_attempt_rate",
                                                 "over_refusal_rate", "invalid_call_rate", "avg_tool_calls")},
                      **auditkit_headline(ak), "throughput": policy.throughput()}, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
