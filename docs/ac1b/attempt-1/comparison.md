# AC1b comparison

**Verdict: FAIL** (0 item(s) REVIEW REQUIRED). Criteria: docs/PLAN.md section 5.1 (frozen 2026-09-29). FAIL only when the flow or AC2 fails.

## Flow

| step | status | commit | completion line(s) |
|---|---|---|---|
| original-haiku | PASS | 3992840ddb3f | `verified 1/1 release files` |
| ac2 | PASS | 3992840ddb3f | `wrote reproduction/3992840ddb3f/results/ac2.json: PASS` |
| pilot-lr | PASS | 3992840ddb3f | `wrote reproduction/3992840ddb3f/results/pilots/lr.json` |
| pilot-steps | PASS | 3992840ddb3f | `wrote reproduction/3992840ddb3f/results/pilots/steps.json` |
| baselines | PASS | 3992840ddb3f | `completed 36/36 baseline points` |
| curve-bert | PASS | 3992840ddb3f | `completed 18/18 encoder points (bert)` |
| curve-modernbert | PASS | 3992840ddb3f | `completed 18/18 encoder points (modernbert)` |
| oos-ablation | PASS | 3992840ddb3f | `completed 3/3 ablation points` |
| verify-logits | PASS | 3992840ddb3f | `OK 75 archive(s) match reproduction/3992840ddb3f/results/logits-manifest.json` |
| llm | FAIL | 3992840ddb3f | none |

## AC2 (threshold 95.7%, every seed): PASS

| seed | rerun (%) | original (%) | passed |
|---|---|---|---|
| 42 | 96.60 | 96.51 | True |
| 43 | 96.36 | 96.40 | True |
| 44 | 96.49 | 96.22 | True |

## Haiku (reproduction-validation run)

```json
{
  "reached": false,
  "checks_passed": false,
  "problems": [
    "no reproduction run"
  ]
}
```

## Budget (separate, never summed)

- original AC6 experiment: US$3.19 (fixed)
- reproduction-validation: US$None (cap US$5)

## Pilots

| pilot | original | rerun | status |
|---|---|---|---|
| lr | {'bert': 5e-05, 'modernbert': 5e-05} | {'bert': 5e-05, 'modernbert': 5e-05} | OK |
| steps | 400 | 400 | OK |

Numbers: not reached (the analysis step did not finish).

