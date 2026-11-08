"""Public benchmarks through AuditKIT: BFCL v3 (tool calling) and ARC-Challenge (no-regression).

BFCL shows whether tool-calling skill generalises beyond our SQL tools; ARC checks
that fine-tuning did not damage general reasoning.

    python scripts/eval_benchmarks.py --name base --model CohereLabs/tiny-aya-global
    python scripts/eval_benchmarks.py --name sft  --model CohereLabs/tiny-aya-global --adapter runs/sft
"""

import _bootstrap  # noqa: F401

import argparse
import json
import re
from pathlib import Path

from safequery.config import DEFAULT_MODEL

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"  # pinned, as in AuditKIT's docs


ARC_INSTRUCTION = "\n\nReply with only the number of the correct option."


def make_callable(model_id, adapter, load_4bit, max_new_tokens):
    """AuditKIT accepts any list[str] -> list[str] callable; we wrap our HF policy.

    The callable is wrapped in a named CallableModel: AuditKIT's disk cache is keyed on the
    model's name, and the default name of a closure is the same for base and +adapter runs,
    so without this the SFT run was served the base run's cached results.
    """
    from auditkit.model import CallableModel
    from safequery.policies import HFPolicy

    pol = HFPolicy(model_id, adapter=adapter, load_4bit=load_4bit, max_new_tokens=max_new_tokens)

    def generate(prompts):
        pol.max_new_tokens = max_new_tokens
        return [pol([{"role": "user", "content": p}], None) for p in prompts]

    def generate_choice(prompts):
        """ARC: AuditKIT's MCQ prompt lists options as 0., 1., 2. ... and scores with Acc, which
        needs the output to be an option number, letter or exact option text - so ask for the
        number and keep the first one in the reply."""
        pol.max_new_tokens = 16
        outs = []
        for p in prompts:
            text = pol([{"role": "user", "content": p + ARC_INSTRUCTION}], None)
            m = re.search(r"\b(\d{1,2})\b", text)
            outs.append(m.group(1) if m else text.strip())
        return outs

    return CallableModel(generate, name=pol.name), CallableModel(generate_choice, name=pol.name + "-mcq"), pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--bfcl-categories", default="simple,irrelevance")
    ap.add_argument("--bfcl-limit", type=int, default=100)
    ap.add_argument("--arc-limit", type=int, default=200)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    import auditkit as ak
    from huggingface_hub import snapshot_download

    gen, gen_choice, pol = make_callable(args.model, args.adapter, args.load_4bit, max_new_tokens=256)
    out: dict = {"name": args.name, "model": args.model, "adapter": args.adapter}

    root = snapshot_download(BFCL_REPO, repo_type="dataset", revision=BFCL_REVISION)
    for cat in args.bfcl_categories.split(","):
        samples = ak.load_bfcl(f"{root}/BFCL_v3_{cat}.json", revision=BFCL_REVISION[:7], limit=args.bfcl_limit)
        res = ak.evaluate(samples, model=gen, adapter=ak.ToolCallAdapter(mode="prompt"),
                          scorers=[ak.ToolCallF1(arg_match=ak.bfcl_arg_match)])
        out[f"bfcl_{cat}"] = res.headline
        print(f"[bfcl:{cat}] n={len(samples)} {res.headline}")

    if args.arc_limit:
        from auditkit.scenarios.arc import ARCScenario

        samples = ARCScenario("test").samples()[: args.arc_limit]
        from auditkit.metric import Acc

        res = ak.evaluate(samples, model=gen_choice, scorers=[Acc()])
        out["arc_challenge"] = res.headline
        print(f"[arc] n={len(samples)} {res.headline}")

    out["throughput"] = pol.throughput()
    path = Path(args.results) / args.name
    path.mkdir(parents=True, exist_ok=True)
    (path / "benchmarks.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[saved] {path / 'benchmarks.json'}")


if __name__ == "__main__":
    main()
