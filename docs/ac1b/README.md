# AC1b attempts

AC1b (docs/PLAN.md section 5.1) is the whole study rerun from a clean clone with `make setup && make reproduce`. This directory records every attempt, whether it passed or not, so that a failed attempt and its cost are never hidden by a later one.

- `attempts.json`: the ledger, one entry per attempt: commit, reproduction id, start and end (UTC), result and reason, actual Haiku spend. `python -m tinyrouter.ac1b_ledger` checks it and rewrites the budget block below and in the top-level README; `tests/test_ac1b_ledger.py` fails when either is stale.
- `attempt-<n>/`: the attempt's `comparison.md` as written by `make reproduce`, and `INCIDENT.md`, a note on why it stopped.

Budget rules (Drew, 2026-09-29 and 2026-09-30):

- The original AC6 experiment cost is fixed at US$3.19. No AC1b attempt changes it, and reproduction-validation spend is never reported as original experiment cost or used to revisit the AC6 conclusion (Haiku spend at most US$5).
- Every AC1b attempt has its own US$5 cap. `make reproduce` writes the Haiku journal under `reproduction/<id>/results` and runs it with `MAX_USD=5`; the cap counts only that journal, so an attempt at a new commit starts at US$0. Attempt 2 resumed attempt 1 under the same id and identity and so shared its journal and cap (attempt 1 spent US$0).
- A comparison never mixes the outputs of two commits. After a code fix, AC1b is run again in full at the new commit.

<!-- BEGIN AC1b budget, generated from docs/ac1b/attempts.json; do not edit by hand -->
- Original experiment (AC6, fixed): US$3.19
- AC1b attempts (reproduction-validation, each capped at US$5):

| attempt | reproduction id | result | reason | Haiku spend |
|---|---|---|---|---|
| 1 | `3992840ddb3f` | FAIL | infrastructure: Haiku count_tokens returned HTTP 503 credential validation failed 5 times during an Anthropic authentication incident; no classification request sent | US$0.000000 |
| 2 | `3992840ddb3f` | FAIL | program defect: make figures stopped because seeds chose different 8-way aggregations for modernbert/k100 (argmax, summed, summed); Haiku 8600/8600, 0 parse failures, 20 predictions differ | US$3.178751 |

- Reproduction-validation total (all AC1b attempts): US$3.178751

Reproduction-validation spend is never reported as original experiment cost and does not change the AC6 conclusion (Haiku spend at most US$5).
<!-- END AC1b budget -->
