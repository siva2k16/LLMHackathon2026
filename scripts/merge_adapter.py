"""Merge a LoRA adapter into its base model (GRPO then starts from the SFT policy).

    python scripts/merge_adapter.py --model CohereLabs/tiny-aya-global --adapter runs/sft --out runs/sft_merged
"""

import _bootstrap  # noqa: F401

import argparse

from safequery.config import DEFAULT_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print(f"[merge] {args.adapter} + {args.model} -> {args.out}")


if __name__ == "__main__":
    main()
