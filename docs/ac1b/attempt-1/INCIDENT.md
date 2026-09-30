# AC1b attempt 1: interrupted by an external authentication incident

- Reproduction id: `3992840ddb3f` (clean clone of `main` at `3992840ddb3f`, `uv.lock` SHA-256 prefix `d6b5940a86b5c28c`).
- Started: 2026-09-29 about 07:45 UTC. Failed: 2026-09-29 about 14:24 UTC.
- Completed and passed before the failure (all at `3992840ddb3f`): original-haiku, ac2, pilot-lr, pilot-steps, baselines, curve-bert, curve-modernbert, oos-ablation, verify-logits.
  - AC2 rerun: 96.60 / 96.36 / 96.49 percent (seeds 42 / 43 / 44), all at or above 95.7.
  - Pilots selected the same values as the original run (lr 5e-5 for both encoders, S_min 400).
- Failed step: `llm`. The pre-run token count (`messages.count_tokens`) returned HTTP 503 `InternalServerError` with the message "credential validation failed" on all 5 attempts. No classification request was sent.
- Cost of attempt 1: US$0.00 (token counting is not billed; the reproduction journal holds no records).
- Cause: an Anthropic incident, "Elevated errors on claude.ai, Claude Code and Claude Cowork", Investigating since 2026-09-29 14:21 UTC, described as failed requests and being asked to sign in again (status.claude.com). The API component was listed as operational, but requests from this machine received 503 during the incident window.
- Verdict of attempt 1 under the frozen criteria: FAIL (flow did not complete). Drew's decision (2026-09-29): this is an infrastructure availability failure, not a failure of training or of the protocol; resume after the incident with the same reproduction id and identity. If the same authentication failure recurs, pause and mark AC1b infrastructure-blocked.
- Before resuming (2026-09-30 01:13 UTC): a free token count succeeded; HEAD, working tree, and `uv.lock` unchanged; the reproduction LLM journal was empty, so no prediction is paid for twice. No hyperparameter, threshold, or result file was changed.
- Evidence kept: `comparison.md`, `comparison.json`, `steps.json`, and `run.log` from attempt 1 (no API key in any of them).
