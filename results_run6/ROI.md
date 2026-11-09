## ROI: closed model vs self-hosted fine-tuned Tiny Aya

| | Closed model (token billing) | Fine-tuned Tiny Aya (self-hosted) |
|---|---|---|
| Monthly volume | 100,000 questions, 200,000 LLM calls | same |
| Tokens / month | 332,600,000 in + 8,000,000 out | same |
| Pricing used | $3.0/M in, $15.0/M out | NVIDIA A100 at $0.35/h, 6.7 tok/s x 8 streams |
| GPU busy hours / month | - | 127.6 h |
| **Monthly cost** | **$1,117.8** | **$255.5** always-on ($44.68 if billed per busy hour) |
| Cost per 1k questions | $11.178 | $2.555 |
| One-off training | - | $1.05 |

Assumptions: prices are inputs (see flags); throughput and tokens per call are measured from the eval run.
