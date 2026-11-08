"""Stage 2 - agentic GRPO with AgentTune's `create_agentic_trainer` (multi-turn tool rollouts).

The model calls our real tools during rollouts; the reward re-verifies every call
(guard, goal check, cost gate, execution, grounded answer). Rollouts are written to
<out>/trajectories.jsonl, which AuditKIT reads with `episodes_from_agenttune`.

    # start from the SFT policy (recommended)
    python scripts/train_grpo.py --model runs/sft_merged --out runs/grpo --max-steps 60

    # reward-hacking ablation (naive "safety-only" reward)
    python scripts/train_grpo.py --model runs/sft_merged --out runs/grpo_naive --naive-reward --max-steps 30

GPU notes: needs ~20GB+ for comfort (L4 / A10G / A100). On a 16GB T4 keep
--num-generations 2 --max-completion-length 192.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import random
from pathlib import Path

from safequery.config import DEFAULT_MODEL
from safequery.env import SafeQueryEnv
from safequery.prompts import SYSTEM_PROMPT
from safequery.rewards import make_grpo_reward
from safequery.tools import TOOLS, bind_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="base model or merged SFT model folder")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/grpo")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--max-steps-per-turn", type=int, default=3)
    ap.add_argument("--n-prompts", type=int, default=None, help="subsample training prompts")
    ap.add_argument("--naive-reward", action="store_true", help="ablation: hackable safety-only reward")
    args = ap.parse_args()

    import torch
    from datasets import Dataset

    from agenttune.core.backend_factory import create_agentic_trainer

    env = SafeQueryEnv(Path(args.data) / "retail.db")
    bind_env(env)
    rows = [json.loads(line) for line in open(Path(args.data) / "grpo_train.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(rows)
    if args.n_prompts:
        rows = rows[: args.n_prompts]
    ds = Dataset.from_list(rows)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    trainer = create_agentic_trainer(
        algorithm="grpo",
        model=args.model,
        train_dataset=ds,
        tools=TOOLS,
        reward_funcs=[make_grpo_reward(env, naive=args.naive_reward)],
        system_prompt=SYSTEM_PROMPT,
        output_dir=args.out,
        max_steps=args.max_steps,
        max_steps_per_turn=args.max_steps_per_turn,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        use_vllm=False,
        peft_config=dict(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        model_init_kwargs={"dtype": "bfloat16" if bf16 else "float16"},
        logging_steps=1,
        save_steps=max(10, args.max_steps // 3),
        report_to="none",
    )
    result = trainer.train()
    hist = getattr(getattr(trainer, "trainer", None), "state", None)
    log = {"base_model": args.model, "naive_reward": args.naive_reward, "args": vars(args),
           "result": result, "log_history": hist.log_history if hist else None,
           "trainer": "agenttune.core.backend_factory.create_agentic_trainer(algorithm='grpo')"}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "safequery_train_log.json").write_text(json.dumps(log, indent=2, default=str))
    rewards = [h["reward"] for h in (log["log_history"] or []) if "reward" in h]
    if rewards:
        k = max(1, len(rewards) // 5)
        print(f"[grpo] mean reward first {k} steps {sum(rewards[:k]) / k:.3f} -> last {k} steps "
              f"{sum(rewards[-k:]) / k:.3f}")
    print(f"[grpo] adapter + trajectories.jsonl in {args.out}")


if __name__ == "__main__":
    main()
