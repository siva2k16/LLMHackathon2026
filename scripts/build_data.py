"""Build the database, the train/test task sets and the SFT demonstrations.

    python scripts/build_data.py                    # full size (200k orders, 600 train / 150 test)
    python scripts/build_data.py --small            # quick local version

Outputs (data/):
    retail.db            the SQLite database
    tasks_train.jsonl    tasks with gold SQL + verified answers + traps
    tasks_test.jsonl     held-out tasks (includes unseen paraphrases)
    sft_train.jsonl      expert demonstrations {"messages": [...]} in AgentTune's Tiny Aya format
    grpo_train.jsonl     {"prompt": [user msg], "task": json} rows for create_agentic_trainer
"""

import _bootstrap  # noqa: F401

import argparse
import json
from collections import Counter
from pathlib import Path

from safequery.db import build_database
from safequery.env import SafeQueryEnv
from safequery.prompts import AGENTTUNE_PROMPTS, user_prompt
from safequery.tasks import expert_conversation, generate_tasks
from safequery.tools import bind_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--small", action="store_true", help="20k orders, 120/40 tasks - for quick local runs")
    ap.add_argument("--rebuild-db", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_orders = 20_000 if args.small else 200_000
    if args.small:
        args.n_train, args.n_test = 120, 40
    db = build_database(out / "retail.db", n_orders=n_orders, overwrite=args.rebuild_db)
    env = SafeQueryEnv(db)
    bind_env(env)
    print(f"[db] {db}  rows: " + ", ".join(f"{t}={n:,}" for t, n in env.catalog.row_counts.items()))

    train = generate_tasks(env, args.n_train, "train", seed=1)
    test = generate_tasks(env, args.n_test, "test", seed=2)
    test_q = {t.question for t in train}
    test = [t for t in test if t.question not in test_q]  # no exact duplicates across splits

    for name, ts in (("train", train), ("test", test)):
        with open(out / f"tasks_{name}.jsonl", "w", encoding="utf-8") as f:
            for t in ts:
                f.write(t.to_json() + "\n")
        print(f"[tasks] {name}: {len(ts)}  " + str(dict(Counter(t.category for t in ts))))

    with open(out / "sft_train.jsonl", "w", encoding="utf-8") as f:
        for t in train:
            f.write(json.dumps({"messages": expert_conversation(env, t), "task_id": t.id}) + "\n")
    schema = env.schema_text()
    with open(out / "grpo_train.jsonl", "w", encoding="utf-8") as f:
        for t in train:
            f.write(json.dumps({"prompt": [{"role": "user", "content": user_prompt(schema, t.question)}],
                                "task": t.to_json()}) + "\n")
    print(f"[sft] {len(train)} demonstrations -> {out / 'sft_train.jsonl'}")
    print(f"[prompts] using AgentTune's exact fallback prompt: {AGENTTUNE_PROMPTS}"
          + ("" if AGENTTUNE_PROMPTS else "  (local copy; rebuild on Colab after installing AgentTune)"))


if __name__ == "__main__":
    main()
