## SafeQuery benchmark (held-out test set)

| Metric | base | sft | grpo |
|---|---|---|---|
| Task success ↑ | 19.8% | 99.1% | 99.1% |
| Answer accuracy (verified) ↑ | 10.5% | 100.0% | 100.0% |
| Correct refusals ↑ | 64.0% | 96.0% | 96.0% |
| Injection resistance ↑ | 0.0% | 100.0% | 100.0% |
| Unsafe SQL attempts ↓ | 0.9% | 0.9% | 0.9% |
| Over-refusal ↓ | 0.0% | 0.0% | 0.0% |
| Over-cost-threshold queries ↓ | 1.2% | 0.0% | 0.0% |
| Full scans executed ↓ | 0.0% | 0.0% | 0.0% |
| Intent accuracy ↑ | 49.5% | 99.1% | 99.1% |
| Invalid tool calls ↓ | 0.0% | 0.0% | 0.0% |
| Avg tool calls / task ↓ | 1.06 | 1.00 | 1.00 |
| auditkit_agent_loop_detection | 1.000 | 1.000 | 1.000 |
| auditkit_completed_rows | 111 | 111 | 111 |
| auditkit_outcome_success | 0.198 | 0.991 | 0.991 |
| auditkit_redundant_tool_calls | 0.000 | 0.000 | 0.000 |
| auditkit_tool_call_exact_name | 0.171 | 0.991 | 0.991 |
| auditkit_tool_call_f1_name | 0.213 | 0.991 | 0.991 |
| auditkit_tool_call_validity | 1.000 | 1.000 | 1.000 |

## Public benchmarks (AuditKIT)

| Benchmark | base | sft | grpo |
|---|---|---|---|
| arc_challenge · acc | 0.156 | 0.126 | - |
| bfcl_irrelevance · tool_call_exact_custom_794bf7e6 | 0.160 | 0.220 | - |
| bfcl_irrelevance · tool_call_f1_custom_794bf7e6 | 0.160 | 0.220 | - |
| bfcl_irrelevance · tool_call_precision_custom_794bf7e6 | 0.160 | 0.220 | - |
| bfcl_irrelevance · tool_call_recall_custom_794bf7e6 | 0.160 | 0.220 | - |
| bfcl_simple · tool_call_exact_custom_794bf7e6 | 0.920 | 0.450 | - |
| bfcl_simple · tool_call_f1_custom_794bf7e6 | 0.920 | 0.450 | - |
| bfcl_simple · tool_call_precision_custom_794bf7e6 | 0.920 | 0.450 | - |
| bfcl_simple · tool_call_recall_custom_794bf7e6 | 0.920 | 0.450 | - |
