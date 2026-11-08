"""Stage 1 - SFT (agentic distillation of expert trajectories) with AgentTune.

Uses AgentTune's built-in TRL SFT factory (the one behind `create_distill_trainer`),
which also repairs conversations for Tiny Aya's strict chat template. The expert
trajectories come from our environment (real tool outputs), in the same fallback
tool format AgentTune's GRPO rollout uses, so SFT and RL see identical prompts.

    python scripts/train_sft.py --model CohereLabs/tiny-aya-global --out runs/sft
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from safequery.config import DEFAULT_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="train on the first N demonstrations")
    args = ap.parse_args()

    import torch
    from agenttune.agentic.distill_trainer import _DefaultSFTTrainerFactory

    from safequery.prompts import AGENTTUNE_PROMPTS

    if not AGENTTUNE_PROMPTS:
        raise SystemExit("AgentTune not importable - install it first (see README)")
    rows = [json.loads(line) for line in open(Path(args.data) / "sft_train.jsonl", encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    rows = [{"messages": r["messages"]} for r in rows]

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = torch.bfloat16 if bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    factory = _DefaultSFTTrainerFactory(
        use_peft=True, lora_r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        lora_targets="all-linear", dtype=dtype, device_map="auto",
    )
    trainer = factory(
        model=args.model, train_dataset=rows,
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length, gradient_checkpointing=True, lr_scheduler_type="cosine",
        warmup_steps=0.05, logging_steps=5, save_strategy="no", report_to="none",
        bf16=bf16, fp16=torch.cuda.is_available() and not bf16,
    )
    print(f"[sft] {len(rows)} demonstrations | model {args.model} | dtype {dtype}")
    trainer.train()
    trainer.save_model(args.out)
    factory.tokenizer.save_pretrained(args.out)
    hist = [h for h in trainer.state.log_history if "loss" in h]
    log = {"base_model": args.model, "n_demonstrations": len(rows), "epochs": args.epochs, "lr": args.lr,
           "lora_r": args.lora_r, "first_loss": hist[0]["loss"] if hist else None,
           "last_loss": hist[-1]["loss"] if hist else None, "log_history": trainer.state.log_history,
           "trainer": "agenttune.agentic.distill_trainer._DefaultSFTTrainerFactory (TRL SFTTrainer + LoRA)"}
    Path(args.out, "safequery_train_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"[sft] loss {log['first_loss']} -> {log['last_loss']} | adapter saved to {args.out}")


if __name__ == "__main__":
    main()
