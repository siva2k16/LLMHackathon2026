"""ROI: self-hosted fine-tuned Tiny Aya vs closed-model token billing.

All prices are INPUTS. Fill them in from current public price pages on the day you
publish, and state the date in the write-up. The defaults below are placeholders for
illustration, not quotes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RoiInputs:
    questions_per_month: int = 100_000
    llm_calls_per_question: float = 2.0      # measured: avg assistant turns per task
    input_tokens_per_call: int = 1_500       # measured from eval (prompt_tokens / model_calls)
    output_tokens_per_call: int = 80         # measured from eval (generated_tokens / model_calls)
    closed_price_in_per_m: float = 3.00      # USD per 1M input tokens (PLACEHOLDER - fill in)
    closed_price_out_per_m: float = 15.00    # USD per 1M output tokens (PLACEHOLDER - fill in)
    gpu_name: str = "NVIDIA L4 (24GB)"
    gpu_price_per_hour: float = 0.80         # USD/hour (PLACEHOLDER - fill in)
    tokens_per_sec: float = 25.0             # measured generation throughput (single stream)
    concurrency: int = 8                     # parallel streams a serving engine sustains
    prefill_speedup: float = 20.0            # prompt tokens are processed this much faster than generated
    training_gpu_hours: float = 3.0          # one-off fine-tuning cost
    training_gpu_price_per_hour: float = 0.80


def compute_roi(x: RoiInputs) -> dict:
    calls = x.questions_per_month * x.llm_calls_per_question
    tin, tout = calls * x.input_tokens_per_call, calls * x.output_tokens_per_call
    closed = tin / 1e6 * x.closed_price_in_per_m + tout / 1e6 * x.closed_price_out_per_m

    agg_tps = x.tokens_per_sec * x.concurrency
    gen_hours = tout / agg_tps / 3600
    prefill_hours = tin / (agg_tps * x.prefill_speedup) / 3600
    busy_hours = gen_hours + prefill_hours
    self_busy = busy_hours * x.gpu_price_per_hour
    self_always_on = 730 * x.gpu_price_per_hour
    training = x.training_gpu_hours * x.training_gpu_price_per_hour
    return {
        "inputs": asdict(x),
        "monthly_llm_calls": int(calls),
        "monthly_input_tokens": int(tin),
        "monthly_output_tokens": int(tout),
        "closed_monthly_usd": round(closed, 2),
        "gpu_busy_hours_per_month": round(busy_hours, 1),
        "self_hosted_usage_based_usd": round(self_busy, 2),
        "self_hosted_always_on_usd": round(self_always_on, 2),
        "one_off_training_usd": round(training, 2),
        "monthly_saving_vs_always_on_usd": round(closed - self_always_on, 2),
        "cost_per_1k_questions_closed": round(closed / x.questions_per_month * 1000, 3),
        "cost_per_1k_questions_self_always_on": round(self_always_on / x.questions_per_month * 1000, 3),
        "fits_on_one_gpu": busy_hours <= 730,
    }


def roi_markdown(r: dict) -> str:
    x = r["inputs"]
    return (
        "| | Closed model (token billing) | Fine-tuned Tiny Aya (self-hosted) |\n|---|---|---|\n"
        f"| Monthly volume | {x['questions_per_month']:,} questions, {r['monthly_llm_calls']:,} LLM calls | same |\n"
        f"| Tokens / month | {r['monthly_input_tokens']:,} in + {r['monthly_output_tokens']:,} out | same |\n"
        f"| Pricing used | ${x['closed_price_in_per_m']}/M in, ${x['closed_price_out_per_m']}/M out | "
        f"{x['gpu_name']} at ${x['gpu_price_per_hour']}/h, {x['tokens_per_sec']} tok/s x {x['concurrency']} streams |\n"
        f"| GPU busy hours / month | - | {r['gpu_busy_hours_per_month']} h |\n"
        f"| **Monthly cost** | **${r['closed_monthly_usd']:,}** | **${r['self_hosted_always_on_usd']:,}** always-on "
        f"(${r['self_hosted_usage_based_usd']:,} if billed per busy hour) |\n"
        f"| Cost per 1k questions | ${r['cost_per_1k_questions_closed']} | ${r['cost_per_1k_questions_self_always_on']} |\n"
        f"| One-off training | - | ${r['one_off_training_usd']} |\n"
    )
