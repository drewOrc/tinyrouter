# Data provenance

TinyRouter uses CLINC150 (Larson et al., 2019, "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction", EMNLP-IJCNLP).

## Where the files come from

| Item | Value |
|---|---|
| Loaded from | Hugging Face Hub, `clinc/clinc_oos`, config `plus` |
| Pinned revision | `155b9c710419136e17307b80d0a13e68cd46b4ec` |
| Integrity checks | SHA-256 and row count per split, label names must match `src/tinyrouter/resources/intent_names.json` (see `src/tinyrouter/data.py`) |
| Original release | `clinc/oos-eval` on GitHub, `data/data_oos_plus.json` at commit `828f8093932c8fe6ca7936c3d2e52903b1c523de` (SHA-256 `bfcca9ae515623541dc1983c94c4ed7cae9d26b42ae47d74b972e51bb6f7a21f`) |

## The `plus` config is the paper's OOS+ variant

Checked on 2026-09-23: for every split, the multiset of `(text, label)` pairs from the Hub parquet is identical to the original `data_oos_plus.json` (in-scope and out-of-scope lists combined). `tests/test_data.py::test_hub_split_is_identical_to_original_oos_plus_release` repeats the check (network marker; runs in the CI `network` job).

| Split | In-scope | Out-of-scope | Total |
|---|---:|---:|---:|
| train | 15,000 (100 per intent × 150) | 250 | 15,250 |
| validation | 3,000 (20 per intent) | 100 | 3,100 |
| test | 4,500 (30 per intent) | 1,000 | 5,500 |

OOS+ differs from the paper's default "Full" variant only in the training split: Full has 100 out-of-scope training examples, OOS+ has 250. Validation and test are the same.

## What this project changes

Nothing in the files. The learning-curve experiments sample from the training split: `k` examples per intent and `ceil(2.5k)` out-of-scope examples. The 2.5 ratio is a design choice that keeps the OOS+ training ratio (250 OOS to 100 per intent) at every point, so `k = 100` is the complete OOS+ training set. It is not the natural out-of-scope rate of real traffic, and the test split (18% out-of-scope) is never subsampled.
