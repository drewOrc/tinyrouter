# AC1b attempts

AC1b (docs/PLAN.md section 5.1) is the whole study rerun from a clean clone with `make setup && make reproduce`. This directory records every attempt, whether it passed or not, so that a failed attempt and its cost are never hidden by a later one.

- `attempts.json`: the ledger, one entry per attempt: commit, reproduction id, start and end (UTC), result and reason, actual Haiku spend. `python -m tinyrouter.ac1b_ledger` checks it and rewrites the budget block below and in the top-level README; `tests/test_ac1b_ledger.py` fails when either is stale.
- `attempt-<n>/`: the attempt's `comparison.md` as written by `make reproduce`, and `INCIDENT.md`, a note on why it stopped. Evidence files are kept as written: attempt 1's `INCIDENT.md` says it started "about 07:45" UTC; the ledger uses 07:40:02 UTC, the start of the first step in `steps.json`, which is authoritative.

Results (validated by `tinyrouter.ac1b_ledger`):

- `PASS` or `FAIL`: the comparison's verdict under the frozen criteria. `comparison_verdict` always records that mechanical verdict, and the evidence is never rewritten.
- `INFRASTRUCTURE INTERRUPTED`: allowed only with `failure_kind: infrastructure`, external incident evidence (incident, time, source, error type) and the decision that classified it. INFRASTRUCTURE INTERRUPTED is neither an AC1b failure nor a pass; AC1b acceptance is completed only by an attempt whose result is PASS. Attempt 1 is recorded this way under Drew's decision of 2026-09-29, while its comparison verdict stays FAIL.
- `INFRASTRUCTURE BLOCKED`: required when the same infrastructure error type recurs on the next attempt. AC1b is then paused, not resumed again.
- `failure_kind` is one of `infrastructure`, `program defect`, `ac2`; a program defect can only be `FAIL`.

Budget rules (Drew, 2026-09-29 and 2026-09-30):

- The original AC6 experiment cost is fixed at US$3.19. No AC1b attempt changes it, and reproduction-validation spend is never reported as original experiment cost or used to revisit the AC6 conclusion (Haiku spend at most US$5).
- Every AC1b attempt has its own US$5 cap. `make reproduce` writes the Haiku journal under `reproduction/<id>/results` and runs it with `MAX_USD=5`; the cap counts only that journal, so an attempt at a new commit starts at US$0. Attempt 2 resumed attempt 1 under the same id and identity and so shared its journal and cap (attempt 1 spent US$0).
- A comparison never mixes the outputs of two commits. After a code fix, AC1b is run again in full at the new commit.

<!-- BEGIN AC1b budget, generated from docs/ac1b/attempts.json; do not edit by hand -->
- Original experiment (AC6, fixed): US$3.19
- AC1b attempts (reproduction-validation, each capped at US$5):

| attempt | reproduction id | result | reason | Haiku spend |
|---|---|---|---|---|
| 1 | `3992840ddb3f` | INFRASTRUCTURE INTERRUPTED | infrastructure: Haiku count_tokens returned HTTP 503 credential validation failed 5 times during an Anthropic authentication incident; no classification request sent | US$0.000000 |
| 2 | `3992840ddb3f` | FAIL | program defect: make figures stopped because seeds chose different 8-way aggregations for modernbert/k100 (argmax, summed, summed); Haiku 8600/8600, 0 parse failures, 20 predictions differ | US$3.178751 |

- Reproduction-validation total (all AC1b attempts): US$3.178751

Reproduction-validation spend is never reported as original experiment cost and does not change the AC6 conclusion (Haiku spend at most US$5).

INFRASTRUCTURE INTERRUPTED is neither an AC1b failure nor a pass; AC1b acceptance is completed only by an attempt whose result is PASS.
<!-- END AC1b budget -->
