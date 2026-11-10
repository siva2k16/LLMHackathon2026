"""Evaluate a real model (base or + LoRA adapter) on the SafeQuery test set, with AuditKIT.

    python scripts/eval_model.py --name base     --model CohereLabs/tiny-aya-global
    python scripts/eval_model.py --name sft      --model CohereLabs/tiny-aya-global --adapter runs/sft
    python scripts/eval_model.py --name grpo     --model runs/sft_merged --adapter runs/grpo
    python scripts/eval_model.py --name sft_plain --adapter runs/sft --wrapper plain   # bare generate()

metrics.json holds, next to the scores:
  serving         grouped settings (adapter / decoding / prompt / post_processing / versions)
                  with a serving_id and one id per group
  provenance      code commit, adapter weight hash, where the adapter came from, test-set hash
  wrapper_report  the trimmed (reported) score next to the raw behaviour before trimming
raw_turns.jsonl holds every turn before and after trimming, with its stop reason.
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from safequery.config import DEFAULT_MODEL
from safequery.env import SafeQueryEnv
from safequery.evaluate import aggregate, auditkit_headline, evaluate_policy, run_auditkit, save_results
from safequery.policies import WRAPPERS, HFPolicy
from safequery.serving import provenance, with_ids, wrapper_report
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
    ap.add_argument("--wrapper", choices=WRAPPERS, default="safequery",
                    help="safequery = stop tokens + stop string + trim_turn (the reported numbers); "
                         "plain = bare generate()")
    ap.add_argument("--adapter-source", default=None,
                    help="where the adapter came from (e.g. 'drive:safequery_run6/runs/sft' or a Hub repo@commit); "
                         "recorded in provenance")
    args = ap.parse_args()

    data = Path(args.data)
    env = SafeQueryEnv(data / "retail.db")
    bind_env(env)
    tasks = [Task.from_json(line) for line in open(data / "tasks_test.jsonl", encoding="utf-8")]
    if args.limit:
        tasks = tasks[: args.limit]

    policy = HFPolicy(args.model, adapter=args.adapter, load_4bit=args.load_4bit,
                      max_new_tokens=args.max_new_tokens, wrapper=args.wrapper)
    print(f"[eval] {args.name}: {policy.name} on {len(tasks)} tasks")
    eps, scs = evaluate_policy(env, tasks, policy, max_turns=args.max_turns)
    metrics = aggregate(scs)
    ak = run_auditkit(eps, tasks, scs, name=args.name)
    serving = with_ids(policy.serving_config(max_turns=args.max_turns))
    report = wrapper_report(policy.raw_log, scs)
    extra = {"model": args.model, "adapter": args.adapter, "load_4bit": args.load_4bit,
             "throughput": policy.throughput(),
             "serving": serving,
             "provenance": provenance(args.adapter, args.adapter_source, data, _bootstrap.ROOT),
             "wrapper_report": report,
             "wrapper_stats": policy.wrapper_stats()}
    out = save_results(args.results, args.name, tasks, eps, scs, metrics, ak, extra)
    with open(out / "raw_turns.jsonl", "w", encoding="utf-8") as f:
        for r in policy.raw_log:
            f.write(json.dumps({"serving_id": serving["serving_id"], **r}, ensure_ascii=False) + "\n")
    print(json.dumps({**{k: metrics[k] for k in ("task_success_rate", "answer_accuracy", "refusal_accuracy",
                                                 "injection_resistance", "unsafe_attempt_rate",
                                                 "over_refusal_rate", "invalid_call_rate", "avg_tool_calls")},
                      **auditkit_headline(ak), "throughput": policy.throughput(),
                      "serving_id": serving["serving_id"], "wrapper_report": report}, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
