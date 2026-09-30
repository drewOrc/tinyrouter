# AC1b comparison

**Verdict: FAIL** (192 item(s) REVIEW REQUIRED). Criteria: docs/PLAN.md section 5.1 (frozen 2026-09-29). FAIL only when the flow or AC2 fails.

## Flow

| step | status | commit | completion line(s) |
|---|---|---|---|
| original-haiku | PASS (resumed) | 3992840ddb3f | `verified 1/1 release files` |
| ac2 | PASS (resumed) | 3992840ddb3f | `wrote reproduction/3992840ddb3f/results/ac2.json: PASS` |
| pilot-lr | PASS (resumed) | 3992840ddb3f | `wrote reproduction/3992840ddb3f/results/pilots/lr.json` |
| pilot-steps | PASS (resumed) | 3992840ddb3f | `wrote reproduction/3992840ddb3f/results/pilots/steps.json` |
| baselines | PASS (resumed) | 3992840ddb3f | `completed 36/36 baseline points` |
| curve-bert | PASS (resumed) | 3992840ddb3f | `completed 18/18 encoder points (bert)` |
| curve-modernbert | PASS (resumed) | 3992840ddb3f | `completed 18/18 encoder points (modernbert)` |
| oos-ablation | PASS (resumed) | 3992840ddb3f | `completed 3/3 ablation points` |
| verify-logits | PASS (resumed) | 3992840ddb3f | `OK 75 archive(s) match reproduction/3992840ddb3f/results/logits-manifest.json` |
| llm | PASS | 3992840ddb3f | `completed 8600/8600 llm predictions` |
| verify-llm | PASS | 3992840ddb3f | `completed 8600/8600 llm predictions` |
| analysis | PASS | 3992840ddb3f | `completed analysis (75/75 archives, 8600/8600 llm rows, 25 groups)` |
| bench-cpu | PASS | 3992840ddb3f | `wrote reproduction/3992840ddb3f/results/efficiency/cpu_latency.json` |
| llm-latency | PASS | 3992840ddb3f | `wrote reproduction/3992840ddb3f/results/efficiency/haiku_latency.json` |
| cost | PASS | 3992840ddb3f | `wrote reproduction/3992840ddb3f/results/cost/cost.json` |
| figures | FAIL | 3992840ddb3f | none |

## AC2 (threshold 95.7%, every seed): PASS

| seed | rerun (%) | original (%) | passed |
|---|---|---|---|
| 42 | 96.60 | 96.51 | True |
| 43 | 96.36 | 96.40 | True |
| 44 | 96.49 | 96.22 | True |

## Haiku (reproduction-validation run)

```json
{
  "reached": true,
  "rows": "8600/8600",
  "parse_failed": 0,
  "original_parse_failed": 0,
  "reproduction_cost_usd": 3.178751,
  "checks_passed": true,
  "problems": [],
  "predictions": {
    "compared_rows": 8600,
    "different_predictions": 20,
    "only_in_original": 0,
    "only_in_reproduction": 0
  }
}
```

## Budget (separate, never summed)

- original AC6 experiment: US$3.19 (fixed)
- reproduction-validation: US$3.178751 (cap US$5)

## Pilots

| pilot | original | rerun | status |
|---|---|---|---|
| lr | {'bert': 5e-05, 'modernbert': 5e-05} | {'bert': 5e-05, 'modernbert': 5e-05} | OK |
| steps | 400 | 400 | OK |

## README first screen (test)

| metric | original mean (%) | original std (%) | rerun (%) | difference (pp) | status |
|---|---|---|---|---|---|
| ModernBERT k=100 small-only 8-way accuracy | 91.87 | 0.13 | 91.42 | -0.45 | REVIEW REQUIRED |
| Haiku 4.5 8-way accuracy | 82.07 | n/a | 81.96 | -0.11 | REVIEW REQUIRED |
| ModernBERT k=10 small-only 8-way accuracy | 81.50 | 0.69 | 80.56 | -0.94 | REVIEW REQUIRED |
| ModernBERT k=10 hybrid 8-way accuracy | 87.95 | 0.39 | 87.60 | -0.35 | OK |
| ModernBERT k=10 hybrid Haiku call rate | 23.88 | 3.76 | 27.24 | 3.36 | OK |
| k=100 threshold, test selective risk | 7.34 | 0.86 | 7.75 | 0.40 | OK |
| k=100 threshold, validation selective risk | 1.50 | 0.13 | 1.57 | 0.07 | OK |
| validation OOS share | 3.23 | 0.00 | 3.23 | 0.00 | OK |
| test OOS share | 18.18 | 0.00 | 18.18 | 0.00 | OK |
| share of the gap explained by OOS share | 86.66 | 1.39 | 87.73 | 1.07 | OK |
| k=100 kept-OOS error rate, test | 36.48 | 3.04 | 39.25 | 2.77 | OK |
| k=100 kept-OOS error rate, validation | 19.91 | 2.93 | 22.75 | 2.84 | OK |
| ModernBERT k=100 hybrid Haiku call rate | 1.27 | 1.59 | 1.27 | 0.00 | OK |
| ModernBERT k=100 hybrid 8-way accuracy | 92.13 | 0.14 | 91.63 | -0.50 | REVIEW REQUIRED |

## README router table (test, 8-way)

| metric | original mean (%) | original std (%) | rerun (%) | difference (pp) | status |
|---|---|---|---|---|---|
| LLM-only accuracy_8 | 82.07 | n/a | 81.96 | -0.11 | REVIEW REQUIRED |
| LLM-only oos_recall | 56.80 | n/a | 56.60 | -0.20 | REVIEW REQUIRED |
| LLM-only llm_call_rate | 100.00 | n/a | 100.00 | 0.00 | OK |
| modernbert/k10 small-only accuracy_8 | 81.50 | 0.69 | 80.56 | -0.94 | REVIEW REQUIRED |
| modernbert/k10 small-only oos_recall | 23.13 | 3.18 | 20.67 | -2.47 | OK |
| modernbert/k10 small-only high_conf_oos_misroute_rate | 76.87 | 3.18 | 79.33 | 2.47 | OK |
| modernbert/k10 small-only llm_call_rate | 0.00 | 0.00 | 0.00 | 0.00 | OK |
| modernbert/k10 hybrid 0.02 accuracy_8 | 87.95 | 0.39 | 87.60 | -0.35 | OK |
| modernbert/k10 hybrid 0.02 oos_recall | 53.47 | 1.46 | 53.13 | -0.33 | OK |
| modernbert/k10 hybrid 0.02 high_conf_oos_misroute_rate | 15.37 | 4.72 | 13.23 | -2.13 | OK |
| modernbert/k10 hybrid 0.02 llm_call_rate | 23.88 | 3.76 | 27.24 | 3.36 | OK |
| modernbert/k10 hybrid 0.05 accuracy_8 | 86.05 | 0.59 | 85.75 | -0.30 | OK |
| modernbert/k10 hybrid 0.05 oos_recall | 43.53 | 3.54 | 43.00 | -0.53 | OK |
| modernbert/k10 hybrid 0.05 high_conf_oos_misroute_rate | 40.07 | 7.53 | 37.90 | -2.17 | OK |
| modernbert/k10 hybrid 0.05 llm_call_rate | 11.75 | 3.28 | 13.32 | 1.56 | OK |
| modernbert/k10 oracle accuracy_8 | 91.45 | 0.22 | 91.10 | -0.36 | REVIEW REQUIRED |
| modernbert/k10 oracle oos_recall | 61.23 | 0.68 | 60.57 | -0.67 | OK |
| modernbert/k10 oracle high_conf_oos_misroute_rate | 0.00 | 0.00 | 0.00 | 0.00 | OK |
| modernbert/k10 oracle llm_call_rate | 18.50 | 0.69 | 19.44 | 0.94 | REVIEW REQUIRED |
| modernbert/k100 small-only accuracy_8 | 91.87 | 0.13 | 91.42 | -0.45 | REVIEW REQUIRED |
| modernbert/k100 small-only oos_recall | 61.07 | 0.38 | 58.47 | -2.60 | REVIEW REQUIRED |
| modernbert/k100 small-only high_conf_oos_misroute_rate | 38.93 | 0.38 | 41.53 | 2.60 | REVIEW REQUIRED |
| modernbert/k100 small-only llm_call_rate | 0.00 | 0.00 | 0.00 | 0.00 | OK |
| modernbert/k100 hybrid 0.02 accuracy_8 | 92.13 | 0.14 | 91.63 | -0.50 | REVIEW REQUIRED |
| modernbert/k100 hybrid 0.02 oos_recall | 62.43 | 1.33 | 59.47 | -2.97 | REVIEW REQUIRED |
| modernbert/k100 hybrid 0.02 high_conf_oos_misroute_rate | 34.53 | 5.53 | 36.83 | 2.30 | OK |
| modernbert/k100 hybrid 0.02 llm_call_rate | 1.27 | 1.59 | 1.27 | 0.00 | OK |
| modernbert/k100 hybrid 0.05 accuracy_8 | 91.93 | 0.13 | 91.57 | -0.36 | REVIEW REQUIRED |
| modernbert/k100 hybrid 0.05 oos_recall | 61.37 | 0.21 | 59.23 | -2.13 | REVIEW REQUIRED |
| modernbert/k100 hybrid 0.05 high_conf_oos_misroute_rate | 38.30 | 0.17 | 39.90 | 1.60 | REVIEW REQUIRED |
| modernbert/k100 hybrid 0.05 llm_call_rate | 0.16 | 0.06 | 0.35 | 0.19 | REVIEW REQUIRED |
| modernbert/k100 oracle accuracy_8 | 95.15 | 0.05 | 94.83 | -0.32 | REVIEW REQUIRED |
| modernbert/k100 oracle oos_recall | 75.60 | 0.60 | 74.00 | -1.60 | REVIEW REQUIRED |
| modernbert/k100 oracle high_conf_oos_misroute_rate | 0.00 | 0.00 | 0.00 | 0.00 | OK |
| modernbert/k100 oracle llm_call_rate | 8.13 | 0.13 | 8.58 | 0.45 | REVIEW REQUIRED |

## OOS ablation (OOS 250 against OOS 0)

| metric | original mean (%) | original std (%) | rerun (%) | difference (pp) | status |
|---|---|---|---|---|---|
| oos_0/hybrid_test/0.02/accuracy_8 | 88.32 | 0.18 | 87.95 | -0.37 | REVIEW REQUIRED |
| oos_0/hybrid_test/0.02/high_conf_oos_misroute_rate | 37.67 | 2.90 | 38.57 | 0.90 | OK |
| oos_0/hybrid_test/0.02/llm_call_rate | 12.07 | 0.73 | 11.82 | -0.25 | OK |
| oos_0/hybrid_test/0.02/oos_recall | 40.60 | 1.04 | 38.90 | -1.70 | REVIEW REQUIRED |
| oos_0/hybrid_test/0.05/accuracy_8 | 81.03 | 0.07 | 81.00 | -0.03 | OK |
| oos_0/hybrid_test/0.05/high_conf_oos_misroute_rate | 99.33 | 0.23 | 99.13 | -0.20 | OK |
| oos_0/hybrid_test/0.05/llm_call_rate | 0.12 | 0.04 | 0.16 | 0.04 | REVIEW REQUIRED |
| oos_0/hybrid_test/0.05/oos_recall | 0.53 | 0.21 | 0.73 | 0.20 | OK |
| oos_0/oos_detection_test/auprc | 91.52 | 0.26 | 91.53 | 0.01 | OK |
| oos_0/oos_detection_test/auroc | 97.80 | 0.04 | 97.84 | 0.04 | REVIEW REQUIRED |
| oos_0/small_only_oos_recall | 0.00 | 0.00 | 0.00 | 0.00 | OK |
| oos_250/hybrid_test/0.02/accuracy_8 | 92.13 | 0.14 | 91.63 | -0.50 | REVIEW REQUIRED |
| oos_250/hybrid_test/0.02/high_conf_oos_misroute_rate | 34.53 | 5.53 | 36.83 | 2.30 | OK |
| oos_250/hybrid_test/0.02/llm_call_rate | 1.27 | 1.59 | 1.27 | 0.00 | OK |
| oos_250/hybrid_test/0.02/oos_recall | 62.43 | 1.33 | 59.47 | -2.97 | REVIEW REQUIRED |
| oos_250/hybrid_test/0.05/accuracy_8 | 91.93 | 0.13 | 91.57 | -0.36 | REVIEW REQUIRED |
| oos_250/hybrid_test/0.05/high_conf_oos_misroute_rate | 38.30 | 0.17 | 39.90 | 1.60 | REVIEW REQUIRED |
| oos_250/hybrid_test/0.05/llm_call_rate | 0.16 | 0.06 | 0.35 | 0.19 | REVIEW REQUIRED |
| oos_250/hybrid_test/0.05/oos_recall | 61.37 | 0.21 | 59.23 | -2.13 | REVIEW REQUIRED |
| oos_250/oos_detection_test/auprc | 94.34 | 0.25 | 94.13 | -0.21 | OK |
| oos_250/oos_detection_test/auroc | 98.33 | 0.07 | 98.26 | -0.06 | OK |
| oos_250/small_only_oos_recall | 61.07 | 0.38 | 58.47 | -2.60 | REVIEW REQUIRED |

## Machine-dependent numbers: 583 LISTED, NOT JUDGED (all rows in the JSON)

Latency, k=100 training time and peak memory, and the cost model vary with the machine; they are listed with their differences and never marked REVIEW REQUIRED.

## Learning curves (test, small model alone)

| metric | original mean (%) | original std (%) | rerun (%) | difference (pp) | status |
|---|---|---|---|---|---|
| modernbert k=1 small-only accuracy_8 | 41.71 | 2.47 | 43.30 | 1.59 | OK |
| modernbert k=1 small-only oos_recall | 1.73 | 0.71 | 2.43 | 0.70 | OK |
| modernbert k=5 small-only accuracy_8 | 70.99 | 3.40 | 71.71 | 0.72 | OK |
| modernbert k=5 small-only oos_recall | 11.00 | 4.29 | 12.30 | 1.30 | OK |
| modernbert k=10 small-only accuracy_8 | 81.50 | 0.69 | 80.56 | -0.94 | REVIEW REQUIRED |
| modernbert k=10 small-only oos_recall | 23.13 | 3.18 | 20.67 | -2.47 | OK |
| modernbert k=25 small-only accuracy_8 | 86.27 | 0.52 | 87.21 | 0.94 | REVIEW REQUIRED |
| modernbert k=25 small-only oos_recall | 36.43 | 3.00 | 40.93 | 4.50 | REVIEW REQUIRED |
| modernbert k=50 small-only accuracy_8 | 90.39 | 0.38 | 90.25 | -0.14 | OK |
| modernbert k=50 small-only oos_recall | 53.87 | 2.32 | 53.83 | -0.03 | OK |
| modernbert k=100 small-only accuracy_8 | 91.87 | 0.13 | 91.42 | -0.45 | REVIEW REQUIRED |
| modernbert k=100 small-only oos_recall | 61.07 | 0.38 | 58.47 | -2.60 | REVIEW REQUIRED |
| bert k=1 small-only accuracy_8 | 54.93 | 1.32 | 55.28 | 0.35 | OK |
| bert k=1 small-only oos_recall | 9.90 | 6.17 | 9.47 | -0.43 | OK |
| bert k=5 small-only accuracy_8 | 78.99 | 1.61 | 78.79 | -0.20 | OK |
| bert k=5 small-only oos_recall | 19.10 | 6.95 | 18.00 | -1.10 | OK |
| bert k=10 small-only accuracy_8 | 81.90 | 1.18 | 82.16 | 0.27 | OK |
| bert k=10 small-only oos_recall | 24.07 | 4.20 | 25.47 | 1.40 | OK |
| bert k=25 small-only accuracy_8 | 86.52 | 0.65 | 86.16 | -0.36 | OK |
| bert k=25 small-only oos_recall | 38.00 | 3.18 | 35.53 | -2.47 | OK |
| bert k=50 small-only accuracy_8 | 88.85 | 0.30 | 88.86 | 0.01 | OK |
| bert k=50 small-only oos_recall | 44.83 | 1.55 | 45.17 | 0.33 | OK |
| bert k=100 small-only accuracy_8 | 91.05 | 0.53 | 90.76 | -0.29 | OK |
| bert k=100 small-only oos_recall | 56.07 | 2.61 | 54.50 | -1.57 | OK |

## Threshold diagnostics: 155 of 758 flagged (all rows in the JSON)

| metric | original mean (%) | original std (%) | rerun (%) | difference (pp) | status |
|---|---|---|---|---|---|
| bert/k1 0.02/sensitivity_reweighted_validation/test/kept_oos_error_rate | 60.00 | n/a | 66.67 | 6.67 | REVIEW REQUIRED |
| bert/k1 0.02/sensitivity_reweighted_validation/test/kept_oos_share | 3.88 | n/a | 4.11 | 0.23 | REVIEW REQUIRED |
| bert/k1 0.02/sensitivity_reweighted_validation/test/selective_risk | 2.33 | n/a | 2.74 | 0.41 | REVIEW REQUIRED |
| bert/k1 0.02/share_of_gap_explained_by_oos_share | 84.49 | n/a | 84.46 | -0.03 | REVIEW REQUIRED |
| bert/k1 0.02/test/kept_oos_error_rate | 60.00 | n/a | 66.67 | 6.67 | REVIEW REQUIRED |
| bert/k1 0.02/test/kept_oos_share | 3.88 | n/a | 4.11 | 0.23 | REVIEW REQUIRED |
| bert/k1 0.02/test/selective_risk | 2.33 | n/a | 2.74 | 0.41 | REVIEW REQUIRED |
| bert/k1 0.02/test_reweighted_to_validation_oos_share/kept_oos_error_rate | 60.00 | n/a | 66.67 | 6.67 | REVIEW REQUIRED |
| bert/k1 0.02/test_reweighted_to_validation_oos_share/kept_oos_share | 0.60 | n/a | 0.64 | 0.04 | REVIEW REQUIRED |
| bert/k1 0.02/test_reweighted_to_validation_oos_share/selective_risk | 0.36 | n/a | 0.43 | 0.07 | REVIEW REQUIRED |
| bert/k1 0.05/sensitivity_reweighted_validation/test/high_conf_oos_misroute_rate | 0.40 | 0.35 | 0.77 | 0.37 | REVIEW REQUIRED |
| bert/k1 0.05/sensitivity_reweighted_validation/validation_weighted_coverage | 3.02 | 2.90 | 5.98 | 2.97 | REVIEW REQUIRED |
| bert/k1 0.05/validation/kept_oos_error_rate | 100.00 | 0.00 | 66.67 | -33.33 | REVIEW REQUIRED |
| bert/k1 0.05/validation/kept_oos_share | 0.43 | 0.16 | 0.78 | 0.35 | REVIEW REQUIRED |
| bert/k1 0.05/validation/selective_risk | 2.83 | 0.18 | 2.64 | -0.19 | REVIEW REQUIRED |
| bert/k10 0.02/sensitivity_reweighted_validation/tau | 96.88 | 2.54 | n/a | n/a | REVIEW REQUIRED |
| bert/k10 0.02/sensitivity_reweighted_validation/validation_weighted_coverage | 27.00 | 5.36 | 32.65 | 5.65 | REVIEW REQUIRED |
| bert/k10 0.02/tau | 81.51 | 7.57 | n/a | n/a | REVIEW REQUIRED |
| bert/k10 0.05/sensitivity_reweighted_validation/test/kept_oos_share | 3.85 | 0.84 | 4.84 | 0.99 | REVIEW REQUIRED |
| bert/k10 0.05/validation/kept_oos_share | 1.84 | 0.02 | 1.95 | 0.11 | REVIEW REQUIRED |
| bert/k100 0.02/test/high_conf_oos_misroute_rate | 27.27 | 1.07 | 29.57 | 2.30 | REVIEW REQUIRED |
| bert/k100 0.02/test/kept_oos_error_rate | 35.48 | 2.24 | 38.06 | 2.58 | REVIEW REQUIRED |
| bert/k100 0.02/test/selective_risk | 5.98 | 0.10 | 6.43 | 0.45 | REVIEW REQUIRED |
| bert/k100 0.02/test_reweighted_to_validation_oos_share/high_conf_oos_misroute_rate | 27.27 | 1.07 | 29.57 | 2.30 | REVIEW REQUIRED |
| bert/k100 0.02/test_reweighted_to_validation_oos_share/kept_oos_error_rate | 35.48 | 2.24 | 38.06 | 2.58 | REVIEW REQUIRED |
| bert/k100 0.02/test_reweighted_to_validation_oos_share/selective_risk | 1.77 | 0.08 | 1.88 | 0.10 | REVIEW REQUIRED |
| bert/k100 0.05/sensitivity_reweighted_validation/test/coverage | 93.12 | 0.62 | 93.79 | 0.68 | REVIEW REQUIRED |
| bert/k100 0.05/sensitivity_reweighted_validation/test/high_conf_oos_misroute_rate | 21.17 | 1.62 | 24.53 | 3.37 | REVIEW REQUIRED |
| bert/k100 0.05/sensitivity_reweighted_validation/test/kept_oos_error_rate | 31.11 | 2.89 | 34.91 | 3.80 | REVIEW REQUIRED |
| bert/k100 0.05/sensitivity_reweighted_validation/test/selective_risk | 4.80 | 0.40 | 5.46 | 0.65 | REVIEW REQUIRED |
| bert/k25 0.02/sensitivity_reweighted_validation/tau | 21.61 | 8.44 | n/a | n/a | REVIEW REQUIRED |
| bert/k25 0.02/sensitivity_reweighted_validation/test/kept_oos_share | 5.54 | 0.98 | 4.17 | -1.37 | REVIEW REQUIRED |
| bert/k25 0.02/tau | 11.31 | 2.09 | n/a | n/a | REVIEW REQUIRED |
| bert/k25 0.02/validation/high_conf_oos_misroute_rate | 12.67 | 2.52 | 17.00 | 4.33 | REVIEW REQUIRED |
| bert/k25 0.02/validation/in_scope_risk | 1.13 | 0.06 | 0.97 | -0.16 | REVIEW REQUIRED |
| bert/k25 0.02/validation/kept_oos_error_rate | 27.27 | 2.40 | 32.30 | 5.03 | REVIEW REQUIRED |
| bert/k25 0.02/validation/kept_oos_share | 1.62 | 0.23 | 1.89 | 0.27 | REVIEW REQUIRED |
| bert/k25 0.05/sensitivity_reweighted_validation/tau | 10.62 | 1.65 | n/a | n/a | REVIEW REQUIRED |
| bert/k25 0.05/sensitivity_reweighted_validation/test/coverage | 84.67 | 3.15 | 81.42 | -3.25 | REVIEW REQUIRED |
| bert/k25 0.05/sensitivity_reweighted_validation/test/in_scope_risk | 1.33 | 0.26 | 0.94 | -0.38 | REVIEW REQUIRED |
| bert/k25 0.05/sensitivity_reweighted_validation/validation_weighted_coverage | 86.37 | 2.25 | 83.66 | -2.70 | REVIEW REQUIRED |
| bert/k25 0.05/sensitivity_reweighted_validation/validation_weighted_risk | 4.06 | 0.05 | 4.11 | 0.05 | REVIEW REQUIRED |
| bert/k25 0.05/tau | 2.32 | 0.17 | n/a | n/a | REVIEW REQUIRED |
| bert/k25 0.05/test/coverage | 99.96 | 0.03 | 99.59 | -0.37 | REVIEW REQUIRED |
| bert/k25 0.05/test/kept_oos_share | 18.15 | 0.02 | 17.87 | -0.27 | REVIEW REQUIRED |
| bert/k25 0.05/test_reweighted_to_validation_oos_share/coverage | 99.99 | 0.00 | 99.90 | -0.10 | REVIEW REQUIRED |
| bert/k25 0.05/test_reweighted_to_validation_oos_share/kept_oos_share | 3.22 | 0.00 | 3.16 | -0.06 | REVIEW REQUIRED |
| bert/k25 0.05/validation/coverage | 100.00 | 0.00 | 99.97 | -0.03 | REVIEW REQUIRED |
| bert/k25 0.05/validation/kept_oos_share | 3.23 | 0.00 | 3.19 | -0.03 | REVIEW REQUIRED |
| bert/k5 0.02/sensitivity_reweighted_validation/tau | n/a | n/a | 98.40 | n/a | REVIEW REQUIRED |
| bert/k5 0.02/tau | n/a | n/a | 87.51 | n/a | REVIEW REQUIRED |
| bert/k5 0.02/test/coverage | 58.16 | 0.86 | 59.78 | 1.61 | REVIEW REQUIRED |
| bert/k5 0.02/test_reweighted_to_validation_oos_share/coverage | 66.07 | 0.97 | 67.95 | 1.88 | REVIEW REQUIRED |
| bert/k5 0.02/validation/coverage | 66.81 | 1.41 | 68.41 | 1.60 | REVIEW REQUIRED |
| bert/k5 0.05/sensitivity_reweighted_validation/validation_weighted_risk | 4.10 | 0.03 | 4.07 | -0.03 | REVIEW REQUIRED |
| bert/k5 0.05/tau | 60.05 | 1.53 | 57.94 | -2.11 | REVIEW REQUIRED |
| bert/k5 0.05/test/coverage | 81.46 | 1.09 | 82.74 | 1.28 | REVIEW REQUIRED |
| bert/k5 0.05/test_reweighted_to_validation_oos_share/coverage | 88.55 | 1.02 | 89.60 | 1.05 | REVIEW REQUIRED |
| bert/k5 0.05/validation/coverage | 89.11 | 0.47 | 89.76 | 0.66 | REVIEW REQUIRED |
| bert/k50 0.02/sensitivity_reweighted_validation/tau | n/a | n/a | -114.56 | n/a | REVIEW REQUIRED |
| bert/k50 0.02/sensitivity_reweighted_validation/validation_weighted_risk | 1.33 | 0.09 | 1.42 | 0.09 | REVIEW REQUIRED |
| bert/k50 0.02/tau | n/a | n/a | -246.17 | n/a | REVIEW REQUIRED |
| bert/k50 0.05/test/in_scope_risk | 1.36 | 0.03 | 1.42 | 0.05 | REVIEW REQUIRED |
| bert/k50 0.05/test_reweighted_to_validation_oos_share/in_scope_risk | 1.36 | 0.03 | 1.42 | 0.05 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.02/sensitivity_reweighted_validation/tau | -13.82 | 1.04 | n/a | n/a | REVIEW REQUIRED |
| modernbert-oos0/k100 0.02/sensitivity_reweighted_validation/test/in_scope_risk | 0.18 | 0.06 | 0.32 | 0.14 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.02/sensitivity_reweighted_validation/test/selective_risk | 2.19 | 0.57 | 2.81 | 0.62 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.02/share_of_gap_explained_by_oos_share | 93.87 | 1.34 | 92.22 | -1.65 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.02/tau | -119.38 | 44.08 | n/a | n/a | REVIEW REQUIRED |
| modernbert-oos0/k100 0.02/test/in_scope_risk | 0.75 | 0.03 | 0.86 | 0.11 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.02/test_reweighted_to_validation_oos_share/in_scope_risk | 0.75 | 0.03 | 0.86 | 0.11 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.02/test_reweighted_to_validation_oos_share/selective_risk | 1.99 | 0.12 | 2.13 | 0.14 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.05/sensitivity_reweighted_validation/tau | n/a | n/a | 50.79 | n/a | REVIEW REQUIRED |
| modernbert-oos0/k100 0.05/sensitivity_reweighted_validation/test/high_conf_oos_misroute_rate | 24.13 | 1.33 | 22.33 | -1.80 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.05/sensitivity_reweighted_validation/test/in_scope_risk | 0.50 | 0.04 | 0.56 | 0.06 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.05/sensitivity_reweighted_validation/test/kept_oos_share | 5.18 | 0.26 | 4.82 | -0.36 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.05/sensitivity_reweighted_validation/test/selective_risk | 5.65 | 0.30 | 5.35 | -0.30 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.05/sensitivity_reweighted_validation/validation_weighted_coverage | 83.84 | 0.25 | 83.57 | -0.27 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.05/share_of_gap_explained_by_oos_share | 97.97 | 0.26 | 97.33 | -0.64 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.05/tau | n/a | n/a | 10.34 | n/a | REVIEW REQUIRED |
| modernbert-oos0/k100 0.05/test/coverage | 99.88 | 0.04 | 99.84 | -0.04 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.05/test/in_scope_risk | 1.08 | 0.05 | 1.16 | 0.07 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.05/test_reweighted_to_validation_oos_share/coverage | 99.98 | 0.01 | 99.96 | -0.01 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.05/test_reweighted_to_validation_oos_share/in_scope_risk | 1.08 | 0.05 | 1.16 | 0.07 | REVIEW REQUIRED |
| modernbert-oos0/k100 0.05/test_reweighted_to_validation_oos_share/selective_risk | 4.25 | 0.05 | 4.32 | 0.07 | REVIEW REQUIRED |
| modernbert/k1 0.05/sensitivity_reweighted_validation/test/kept_oos_error_rate | 100.00 | 0.00 | 91.67 | -8.33 | REVIEW REQUIRED |
| modernbert/k1 0.05/test/kept_oos_error_rate | 100.00 | 0.00 | 91.67 | -8.33 | REVIEW REQUIRED |
| modernbert/k1 0.05/test_reweighted_to_validation_oos_share/kept_oos_error_rate | 100.00 | 0.00 | 91.67 | -8.33 | REVIEW REQUIRED |
| modernbert/k10 0.02/sensitivity_reweighted_validation/test/coverage | 61.01 | 0.49 | 62.42 | 1.41 | REVIEW REQUIRED |
| modernbert/k10 0.02/sensitivity_reweighted_validation/test/high_conf_oos_misroute_rate | 3.93 | 0.78 | 5.77 | 1.83 | REVIEW REQUIRED |
| modernbert/k10 0.02/sensitivity_reweighted_validation/test/in_scope_risk | 0.48 | 0.04 | 0.61 | 0.13 | REVIEW REQUIRED |
| modernbert/k10 0.02/sensitivity_reweighted_validation/test/kept_oos_error_rate | 54.19 | 0.72 | 61.73 | 7.54 | REVIEW REQUIRED |
| modernbert/k10 0.02/sensitivity_reweighted_validation/test/kept_oos_share | 2.16 | 0.42 | 2.75 | 0.59 | REVIEW REQUIRED |
| modernbert/k10 0.02/sensitivity_reweighted_validation/test/selective_risk | 1.64 | 0.21 | 2.26 | 0.62 | REVIEW REQUIRED |
| modernbert/k10 0.02/sensitivity_reweighted_validation/validation_weighted_coverage | 61.15 | 1.32 | 63.37 | 2.22 | REVIEW REQUIRED |
| modernbert/k10 0.02/sensitivity_reweighted_validation/validation_weighted_risk | 1.28 | 0.07 | 1.39 | 0.11 | REVIEW REQUIRED |
| modernbert/k10 0.02/validation/high_conf_oos_misroute_rate | 11.00 | 0.00 | 9.67 | -1.33 | REVIEW REQUIRED |
| modernbert/k10 0.02/validation/kept_oos_share | 1.03 | 0.12 | 0.87 | -0.16 | REVIEW REQUIRED |
| modernbert/k10 0.05/validation/selective_risk | 4.31 | 0.01 | 4.33 | 0.01 | REVIEW REQUIRED |
| modernbert/k100 0.02/sensitivity_reweighted_validation/test/in_scope_risk | 0.53 | 0.10 | 0.39 | -0.14 | REVIEW REQUIRED |
| modernbert/k100 0.05/sensitivity_reweighted_validation/test/in_scope_risk | 1.23 | 0.08 | 1.11 | -0.11 | REVIEW REQUIRED |
| modernbert/k100 0.05/sensitivity_reweighted_validation/validation_weighted_risk | 4.12 | 0.06 | 4.05 | -0.07 | REVIEW REQUIRED |
| modernbert/k100 0.05/test/coverage | 99.84 | 0.06 | 99.65 | -0.19 | REVIEW REQUIRED |
| modernbert/k100 0.05/test/high_conf_oos_misroute_rate | 38.30 | 0.17 | 39.90 | 1.60 | REVIEW REQUIRED |
| modernbert/k100 0.05/test/kept_oos_error_rate | 38.57 | 0.12 | 40.63 | 2.06 | REVIEW REQUIRED |
| modernbert/k100 0.05/test/kept_oos_share | 18.08 | 0.07 | 17.91 | -0.17 | REVIEW REQUIRED |
| modernbert/k100 0.05/test/selective_risk | 8.01 | 0.08 | 8.30 | 0.29 | REVIEW REQUIRED |
| modernbert/k100 0.05/test_reweighted_to_validation_oos_share/coverage | 99.94 | 0.01 | 99.93 | -0.02 | REVIEW REQUIRED |
| modernbert/k100 0.05/test_reweighted_to_validation_oos_share/high_conf_oos_misroute_rate | 38.30 | 0.17 | 39.90 | 1.60 | REVIEW REQUIRED |
| modernbert/k100 0.05/test_reweighted_to_validation_oos_share/kept_oos_error_rate | 38.57 | 0.12 | 40.63 | 2.06 | REVIEW REQUIRED |
| modernbert/k100 0.05/test_reweighted_to_validation_oos_share/kept_oos_share | 3.21 | 0.02 | 3.17 | -0.04 | REVIEW REQUIRED |
| modernbert/k25 0.02/sensitivity_reweighted_validation/test/high_conf_oos_misroute_rate | 6.20 | 0.26 | 6.57 | 0.37 | REVIEW REQUIRED |
| modernbert/k25 0.02/sensitivity_reweighted_validation/test/kept_oos_error_rate | 42.56 | 0.95 | 35.31 | -7.25 | REVIEW REQUIRED |
| modernbert/k25 0.02/sensitivity_reweighted_validation/test/kept_oos_share | 3.57 | 0.15 | 4.56 | 0.99 | REVIEW REQUIRED |
| modernbert/k25 0.02/test/kept_oos_error_rate | 49.52 | 4.21 | 44.33 | -5.19 | REVIEW REQUIRED |
| modernbert/k25 0.02/test/kept_oos_share | 8.62 | 0.30 | 9.54 | 0.92 | REVIEW REQUIRED |
| modernbert/k25 0.02/test_reweighted_to_validation_oos_share/kept_oos_error_rate | 49.52 | 4.21 | 44.33 | -5.19 | REVIEW REQUIRED |
| modernbert/k25 0.02/test_reweighted_to_validation_oos_share/kept_oos_share | 1.40 | 0.05 | 1.56 | 0.16 | REVIEW REQUIRED |
| modernbert/k25 0.02/validation/high_conf_oos_misroute_rate | 14.33 | 1.53 | 11.00 | -3.33 | REVIEW REQUIRED |
| modernbert/k25 0.02/validation/in_scope_risk | 1.09 | 0.05 | 1.20 | 0.11 | REVIEW REQUIRED |
| modernbert/k25 0.02/validation/kept_oos_error_rate | 32.81 | 1.66 | 24.00 | -8.81 | REVIEW REQUIRED |
| modernbert/k25 0.02/validation/selective_risk | 1.56 | 0.00 | 1.56 | -0.01 | REVIEW REQUIRED |
| modernbert/k25 0.05/sensitivity_reweighted_validation/test/coverage | 85.41 | 0.99 | 87.78 | 2.38 | REVIEW REQUIRED |
| modernbert/k25 0.05/sensitivity_reweighted_validation/test/high_conf_oos_misroute_rate | 20.10 | 0.62 | 23.17 | 3.07 | REVIEW REQUIRED |
| modernbert/k25 0.05/sensitivity_reweighted_validation/test/kept_oos_share | 8.70 | 0.54 | 10.34 | 1.65 | REVIEW REQUIRED |
| modernbert/k25 0.05/sensitivity_reweighted_validation/test/selective_risk | 5.20 | 0.57 | 5.85 | 0.65 | REVIEW REQUIRED |
| modernbert/k25 0.05/sensitivity_reweighted_validation/validation_weighted_coverage | 85.95 | 1.42 | 88.49 | 2.55 | REVIEW REQUIRED |
| modernbert/k25 0.05/sensitivity_reweighted_validation/validation_weighted_risk | 4.09 | 0.02 | 4.05 | -0.03 | REVIEW REQUIRED |
| modernbert/k25 0.05/test/kept_oos_error_rate | 63.18 | 3.12 | 59.02 | -4.16 | REVIEW REQUIRED |
| modernbert/k25 0.05/test/selective_risk | 13.51 | 0.46 | 12.74 | -0.77 | REVIEW REQUIRED |
| modernbert/k25 0.05/test_reweighted_to_validation_oos_share/kept_oos_error_rate | 63.18 | 3.12 | 59.02 | -4.16 | REVIEW REQUIRED |
| modernbert/k25 0.05/validation/high_conf_oos_misroute_rate | 51.33 | 3.79 | 45.67 | -5.67 | REVIEW REQUIRED |
| modernbert/k25 0.05/validation/kept_oos_error_rate | 52.28 | 4.96 | 45.67 | -6.61 | REVIEW REQUIRED |
| modernbert/k5 0.02/test/high_conf_oos_misroute_rate | 4.00 | 0.69 | 4.97 | 0.97 | REVIEW REQUIRED |
| modernbert/k5 0.02/test/selective_risk | 3.31 | 0.22 | 3.55 | 0.24 | REVIEW REQUIRED |
| modernbert/k5 0.02/test_reweighted_to_validation_oos_share/high_conf_oos_misroute_rate | 4.00 | 0.69 | 4.97 | 0.97 | REVIEW REQUIRED |
| modernbert/k5 0.02/validation/high_conf_oos_misroute_rate | 0.67 | 0.58 | 2.33 | 1.67 | REVIEW REQUIRED |
| modernbert/k5 0.02/validation/in_scope_risk | 1.33 | 0.02 | 1.26 | -0.07 | REVIEW REQUIRED |
| modernbert/k5 0.02/validation/kept_oos_error_rate | 8.33 | 7.22 | 15.87 | 7.54 | REVIEW REQUIRED |
| modernbert/k5 0.02/validation/kept_oos_share | 0.45 | 0.12 | 0.58 | 0.13 | REVIEW REQUIRED |
| modernbert/k5 0.05/sensitivity_reweighted_validation/tau | n/a | n/a | 202.52 | n/a | REVIEW REQUIRED |
| modernbert/k5 0.05/tau | n/a | n/a | 144.06 | n/a | REVIEW REQUIRED |
| modernbert/k5 0.05/validation/in_scope_risk | 3.68 | 0.04 | 3.60 | -0.08 | REVIEW REQUIRED |
| modernbert/k50 0.02/sensitivity_reweighted_validation/tau | 79.38 | 11.04 | n/a | n/a | REVIEW REQUIRED |
| modernbert/k50 0.02/sensitivity_reweighted_validation/test/in_scope_risk | 0.55 | 0.16 | 0.76 | 0.21 | REVIEW REQUIRED |
| modernbert/k50 0.02/sensitivity_reweighted_validation/validation_weighted_risk | 1.35 | 0.02 | 1.42 | 0.07 | REVIEW REQUIRED |
| modernbert/k50 0.02/tau | 35.06 | 7.34 | n/a | n/a | REVIEW REQUIRED |
| modernbert/k50 0.05/sensitivity_reweighted_validation/tau | n/a | n/a | -238.19 | n/a | REVIEW REQUIRED |
| modernbert/k50 0.05/sensitivity_reweighted_validation/test/in_scope_risk | 1.17 | 0.10 | 1.35 | 0.18 | REVIEW REQUIRED |
| modernbert/k50 0.05/sensitivity_reweighted_validation/validation_weighted_risk | 4.11 | 0.04 | 4.16 | 0.04 | REVIEW REQUIRED |
| modernbert/k50 0.05/share_of_gap_explained_by_oos_share | 89.41 | 1.38 | 88.02 | -1.39 | REVIEW REQUIRED |
| modernbert/k50 0.05/tau | n/a | n/a | -371.87 | n/a | REVIEW REQUIRED |
| modernbert/k50 0.05/test/in_scope_risk | 1.48 | 0.10 | 1.65 | 0.17 | REVIEW REQUIRED |
| modernbert/k50 0.05/test_reweighted_to_validation_oos_share/in_scope_risk | 1.48 | 0.10 | 1.65 | 0.17 | REVIEW REQUIRED |
| modernbert/k50 0.05/test_reweighted_to_validation_oos_share/selective_risk | 2.91 | 0.07 | 3.08 | 0.17 | REVIEW REQUIRED |

