"""Publish all artifacts to the Hugging Face Hub and group them in two public collections
(the hackathon requires a public model collection and a public dataset collection).

    huggingface-cli login            # or set HF_TOKEN; never paste tokens into code
    python scripts/push_to_hub.py --user <hf-username> --adapters sft=runs/sft grpo=runs/grpo

Creates (all public):
  models   <user>/safequery-tiny-aya-global-<name>-lora      one per adapter
  datasets <user>/safequery-retail-tasks                     db build script output, tasks, SFT demos
           <user>/safequery-eval-traces                       eval episodes, metrics, AuditKIT results, train logs
  collections "SafeQuery models" and "SafeQuery datasets"
"""

import _bootstrap  # noqa: F401

import argparse
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def model_card(name, base, metrics_md, user):
    return f"""---
base_model: {base}
library_name: peft
license: cc-by-nc-4.0
tags: [agenttune, auditkit, text-to-sql, tool-calling, safety, lexsi-hackathon]
---
# SafeQuery - Tiny Aya Global ({name} LoRA)

A cost-aware, safety-gated SQL agent: LoRA adapter for `{base}`, post-trained with
**AgentTune** ({'agentic GRPO' if 'grpo' in name else 'SFT on expert tool trajectories'}) and evaluated with **AuditKIT**.

The agent declares intent, checks cost (stats + query plan + threshold), runs only
read-only SQL, refuses write / destructive / sensitive requests and resists prompt
injection. Tools and environment: see the GitHub repo and `{user}/safequery-retail-tasks`.

## Results (held-out test set)
{metrics_md}

## Licence
Base model: CC-BY-NC-4.0 (Cohere Labs). Trained with AgentTune (LSAL v1.1); evaluated with AuditKIT (LSAL v1.2).
Non-commercial use only. Safety-relevant behaviour was re-evaluated after training (see results).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--base-model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adapters", nargs="*", default=[], help="name=folder pairs, e.g. sft=runs/sft")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    cmp_md = (Path(args.results) / "COMPARISON.md")
    metrics_md = cmp_md.read_text(encoding="utf-8") if cmp_md.exists() else "_see eval-traces dataset_"
    model_ids, dataset_ids = [], []

    for pair in args.adapters:
        name, folder = pair.split("=", 1)
        repo = f"{args.user}/safequery-tiny-aya-global-{name}-lora"
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        Path(folder, "README.md").write_text(model_card(name, args.base_model, metrics_md, args.user), encoding="utf-8")
        api.upload_folder(repo_id=repo, folder_path=folder, repo_type="model",
                          ignore_patterns=["checkpoint-*", "*.pt", "optimizer*", "trajectories.jsonl"])
        model_ids.append(repo)
        print(f"[model] https://huggingface.co/{repo}")

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp, "tasks")
        d.mkdir()
        for f in ("tasks_train.jsonl", "tasks_test.jsonl", "sft_train.jsonl", "grpo_train.jsonl"):
            shutil.copy(Path(args.data) / f, d / f)
        shutil.copy(ROOT / "safequery" / "db.py", d / "build_retail_db.py")
        (d / "README.md").write_text(
            "---\nlicense: cc-by-nc-4.0\ntags: [text-to-sql, tool-calling, safety, agenttune, auditkit]\n---\n"
            "# SafeQuery retail tasks\n\nSynthetic, fully reproducible text-to-SQL agent tasks over a 200k-order "
            "retail SQLite database (rebuild it with `build_retail_db.py`, seed 7). Each task has gold SQL and a "
            "verified answer, or is a trap: destructive / write / sensitive requests to refuse, and prompt "
            "injections embedded in benign questions. `sft_train.jsonl` holds expert tool trajectories in "
            "AgentTune's Tiny Aya tool format; `grpo_train.jsonl` holds prompts for agentic GRPO.\n",
            encoding="utf-8")
        repo = f"{args.user}/safequery-retail-tasks"
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(d), repo_type="dataset")
        dataset_ids.append(repo)
        print(f"[dataset] https://huggingface.co/datasets/{repo}")

        t = Path(tmp, "traces")
        shutil.copytree(args.results, t / "results")
        runs = Path(args.runs_dir)
        if runs.exists():
            for f in list(runs.glob("*/safequery_train_log.json")) + list(runs.glob("*/trajectories.jsonl")):
                dest = t / "training" / f.parent.name / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dest)
        (t / "README.md").write_text(
            "---\nlicense: cc-by-nc-4.0\ntags: [evaluation, auditkit, agenttune, traces]\n---\n"
            "# SafeQuery evaluation traces and training logs\n\n`results/<run>/episodes.jsonl` - every evaluated "
            "episode (messages, tool calls, verified score); `metrics.json` - aggregate metrics; "
            "`auditkit_result.json` - AuditKIT agent-eval output; `benchmarks.json` - BFCL / ARC via AuditKIT; "
            "`training/` - AgentTune training logs and GRPO rollout trajectories.\n\n" + metrics_md,
            encoding="utf-8")
        repo = f"{args.user}/safequery-eval-traces"
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(t), repo_type="dataset")
        dataset_ids.append(repo)
        print(f"[dataset] https://huggingface.co/datasets/{repo}")

    for title, ids, kind in (("SafeQuery models", model_ids, "model"), ("SafeQuery datasets", dataset_ids, "dataset")):
        if not ids:
            continue
        col = api.create_collection(title=title, namespace=args.user, private=False, exists_ok=True,
                                    description="Lexsi Labs hackathon - Agentic Fine-tuning track (AgentTune + AuditKIT)")
        for i in ids:
            api.add_collection_item(col.slug, item_id=i, item_type=kind, exists_ok=True)
        print(f"[collection] https://huggingface.co/collections/{col.slug}")
    Path(args.results, "hub_links.json").write_text(json.dumps({"models": model_ids, "datasets": dataset_ids}, indent=2))


if __name__ == "__main__":
    main()
