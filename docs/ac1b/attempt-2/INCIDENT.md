# AC1b attempt 2: stopped by a figure-code defect

Written 2026-09-30 from the attempt's `run.log`, `steps.json` and `comparison.md` (the evidence directory had no incident note for this attempt).

- Reproduction id: `3992840ddb3f`, the same id and identity as attempt 1 (commit `3992840ddb3f`, same `uv.lock`, `configs/`, Python, torch and transformers). It resumed attempt 1: the nine steps that had passed were skipped because their outputs still had the recorded SHA-256.
- Started (resumed `llm` step): 2026-09-30 01:13:40 UTC. Failed: 2026-09-30 01:31:53 UTC.
- Passed in this attempt: llm (Haiku 8,600/8,600, 0 parse failures, 20 predictions different from the original run), verify-llm, analysis, bench-cpu, llm-latency, cost.
- Haiku spend: US$3.178751, under the US$5 cap of this reproduction id (attempt 1 spent US$0 on the same journal).
- Failed step: `figures`. `figures.selected_aggregation` raised `ValueError: modernbert/k100: seeds chose different aggregations ['argmax', 'summed', 'summed']; pick one first`. In the original run all three ModernBERT k=100 seeds chose argmax on validation, so this path had never run.
- Verdict under the frozen criteria (docs/PLAN.md 5.1): FAIL (the flow did not complete). This is a defect in the program, not an infrastructure failure.
- Drew's decision (2026-09-30): fix the defect, then run AC1b again in full at the new commit as a new attempt with its own US$5 cap. The outputs of this attempt are not combined with figures or a report from another commit.
