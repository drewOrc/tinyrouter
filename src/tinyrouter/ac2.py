"""AC2 pipeline check: bert-base-uncased on the full OOS+ train split, seeds 42/43/44.

Pass means every seed's 150-way in-scope test accuracy is at least 95.7%
(docs/PLAN.md AC2; derived from Larson et al.'s 96.7%, not a claim of
exact reproduction). Writes ``results/ac2.json`` (with the config and
each seed's logits SHA-256) and exits 1 on FAIL. Any earlier ``ac2.json``
is deleted first, so a run that fails partway leaves no verdict behind.

Tuning after a FAIL uses the validation numbers only; ``ac2.json`` and the
printout list them for that purpose. The test numbers are the verdict and
are not a tuning signal.

Resumable, with the rules in runs.py: a seed is skipped only when its
results JSON was produced by the current config (output paths aside) and
its logits archive has the same SHA-256 in the results JSON, in the
manifest and on disk. Trained weights are reused only when their
``train_summary.json`` records the current config. Anything else is
cleared and redone. ``--force`` clears
every seed's outputs before training anything, so a crash halfway never
leaves old and new runs mixed. Weights of every seed except 42 are deleted
once that seed's logits are archived (disk is tight; seed 42 is kept as
the RQ7 ONNX fallback).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyrouter.config import RunConfig, load_config
from tinyrouter.evaluate import RunPaths
from tinyrouter.runs import (
    EvaluateFn,
    Log,
    TrainFn,
    clear_outputs,
    default_evaluate,
    default_train,
    is_done,
    reusable_weights,
    run_dir,
    run_one,
)

SEEDS = (42, 43, 44)
KEEP_WEIGHTS_SEEDS = frozenset({42})
THRESHOLD = 0.957
EXPECTED_MODEL = "google-bert/bert-base-uncased"
EXPECTED_TRAIN_ROWS = 15_250
EXPECTED_OOS_TRAIN_ROWS = 250
EXPECTED_TEST_ROWS = 5_500
REPORTED = ("in_scope_accuracy_150", "oos_recall_151", "accuracy_8", "oos_recall_8")
# Re-exported: the resume rules live in runs.py and are shared with the curves.
__all__ = [
    "clear_outputs",
    "default_evaluate",
    "default_train",
    "is_done",
    "reusable_weights",
    "run_ac2",
    "run_dir",
]


class SetupError(ValueError):
    """A results record is not the AC2 setup (wrong model, seed, data, or a duplicate)."""


def expected_run_name(seed: int) -> str:
    return f"bert-base-uncased-full-seed{seed}"


def run_seed(config: RunConfig, train_fn: TrainFn, evaluate_fn: EvaluateFn, log: Log) -> None:
    run_one(config, train_fn, evaluate_fn, log, keep_weights=config.seed in KEEP_WEIGHTS_SEEDS)


def check_setup(record: dict[str, object], seed: int) -> None:
    config, training = record["config"], record["training"]
    assert isinstance(config, dict) and isinstance(training, dict)
    test_rows = record["metrics"]["test"]["raw"]["n"]  # type: ignore[index]
    problems = []
    if record["run_name"] != expected_run_name(seed):
        problems.append(f"run_name {record['run_name']} != {expected_run_name(seed)}")
    if config["seed"] != seed or training.get("seed") != seed:
        problems.append(f"config seed {config['seed']}, training seed {training.get('seed')}")
    if config["model_name"] != EXPECTED_MODEL:
        problems.append(f"model {config['model_name']} != {EXPECTED_MODEL}")
    if config["per_intent"] is not None or config["eval_per_intent"] is not None:
        problems.append("data was subsampled")
    if training["train_rows"] != EXPECTED_TRAIN_ROWS:
        problems.append(f"train_rows {training['train_rows']} != {EXPECTED_TRAIN_ROWS}")
    if training["oos_train_rows"] != EXPECTED_OOS_TRAIN_ROWS:
        problems.append(f"oos_train_rows {training['oos_train_rows']} != 250")
    if test_rows != EXPECTED_TEST_ROWS:
        problems.append(f"test rows {test_rows} != {EXPECTED_TEST_ROWS}")
    if problems:
        raise SetupError(f"seed {seed} is not the AC2 setup: {'; '.join(problems)}")


def split_numbers(record: dict[str, object], split: str) -> dict[str, float]:
    raw = record["metrics"][split]["raw"]  # type: ignore[index]
    return {key: float(raw[key]) for key in REPORTED}


def judge(records: dict[int, dict[str, object]]) -> dict:
    """PASS iff every seed in SEEDS has test in-scope accuracy >= THRESHOLD."""
    missing = [s for s in SEEDS if s not in records]
    if missing:
        raise SetupError(f"no results for seeds {missing}")
    for seed in SEEDS:
        check_setup(records[seed], seed)
    shas = [records[s]["logits"]["sha256"] for s in SEEDS]  # type: ignore[index]
    if len(set(shas)) != len(SEEDS):
        raise SetupError("two seeds point at the same logits archive; they are not separate runs")
    per_seed = {}
    for seed in SEEDS:
        test = split_numbers(records[seed], "test")
        per_seed[str(seed)] = {
            "run_name": records[seed]["run_name"],
            "logits_sha256": records[seed]["logits"]["sha256"],  # type: ignore[index]
            "validation": split_numbers(records[seed], "validation"),
            "test": test,
            "passed": test["in_scope_accuracy_150"] >= THRESHOLD,
        }
    passed = all(entry["passed"] for entry in per_seed.values())
    return {
        "criterion": "AC2: test in_scope_accuracy_150 >= threshold for every seed",
        "split": "test",
        "threshold": THRESHOLD,
        "tuning_note": "tune on the validation numbers only; test is the verdict",
        "seeds": per_seed,
        "verdict": "PASS" if passed else "FAIL",
    }


def run_ac2(
    base: RunConfig,
    train_fn: TrainFn,
    evaluate_fn: EvaluateFn,
    force: bool = False,
    log: Log = print,
) -> dict:
    out = Path(base.results_root) / "ac2.json"
    if out.exists():
        # A verdict from an earlier config must not survive a run that fails partway.
        out.unlink()
        log(f"removed previous {out}")
    if force:
        for seed in SEEDS:
            clear_outputs(base.with_seed(seed), log)
    records: dict[int, dict[str, object]] = {}
    for seed in SEEDS:
        config = base.with_seed(seed)
        run_seed(config, train_fn, evaluate_fn, log)
        path = RunPaths.of(config).results_json
        records[seed] = json.loads(path.read_text(encoding="utf-8"))
    result = judge(records)
    result["config"] = {k: v for k, v in base.identity().items() if k != "seed"}
    result["weights_kept"] = {str(s): run_dir(base.with_seed(s)).exists() for s in SEEDS}
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {out}: {result['verdict']}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bert-base.yaml")
    parser.add_argument("--force", action="store_true", help="clear and rerun every seed")
    args = parser.parse_args(argv)
    result = run_ac2(load_config(args.config), default_train, default_evaluate, args.force)
    print("seed  val in-scope  val OOS recall  test in-scope  verdict (test only)")
    for seed, entry in result["seeds"].items():
        val, test = entry["validation"], entry["test"]
        mark = "pass" if entry["passed"] else "FAIL"
        print(
            f"{seed:>4}  {val['in_scope_accuracy_150']:.4f}       {val['oos_recall_151']:.4f}"
            f"          {test['in_scope_accuracy_150']:.4f}         {mark}"
        )
    if result["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
