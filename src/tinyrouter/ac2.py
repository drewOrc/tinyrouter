"""AC2 pipeline check: bert-base-uncased on the full OOS+ train split, seeds 42/43/44.

Pass means every seed's 150-way in-scope test accuracy is at least 95.7%
(docs/PLAN.md AC2; derived from Larson et al.'s 96.7%, not a claim of
exact reproduction). Writes ``results/ac2.json`` and exits 1 on FAIL.

Resumable: a seed whose results JSON exists and whose logits archive
matches the manifest is not rerun, unless ``--force``. Trained weights
of every seed except 42 are deleted once that seed's logits are archived
(disk is tight; seed 42 is kept as the RQ7 ONNX fallback).
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from pathlib import Path

from tinyrouter.archive import ArchiveError, check_against_manifest, load_logits
from tinyrouter.config import RunConfig, load_config

SEEDS = (42, 43, 44)
KEEP_WEIGHTS_SEEDS = frozenset({42})
THRESHOLD = 0.957
EXPECTED_MODEL = "google-bert/bert-base-uncased"
EXPECTED_TRAIN_ROWS = 15_250
EXPECTED_OOS_TRAIN_ROWS = 250
EXPECTED_TEST_ROWS = 5_500

TrainFn = Callable[[RunConfig], Path]
EvaluateFn = Callable[[RunConfig, Path], dict[str, object]]
Log = Callable[[str], None]


class SetupError(ValueError):
    """A results record is not the AC2 setup (wrong model, subsampled data, ...)."""


def run_dir(config: RunConfig) -> Path:
    return Path(config.checkpoint_root) / config.run_name


def is_done(config: RunConfig) -> bool:
    """Results JSON present and its logits archive loads and matches the manifest."""
    from tinyrouter.evaluate import RunPaths

    paths = RunPaths.of(config)
    if not (paths.results_json.exists() and paths.logits.exists()):
        return False
    try:
        check_against_manifest(paths.manifest, paths.logits)
        load_logits(paths.logits)
    except ArchiveError:
        return False
    return True


def has_final_weights(config: RunConfig) -> bool:
    final = run_dir(config) / "final"
    return (final / "model.safetensors").exists() and (final / "train_summary.json").exists()


def run_seed(
    config: RunConfig, train_fn: TrainFn, evaluate_fn: EvaluateFn, force: bool, log: Log
) -> None:
    if is_done(config) and not force:
        log(f"seed {config.seed}: results and logits already archived, skipping")
    else:
        if has_final_weights(config) and not force:
            log(f"seed {config.seed}: reusing trained weights in {run_dir(config)}/final")
            final_dir = run_dir(config) / "final"
        else:
            if run_dir(config).exists():
                # Stale or partial output from an earlier run must not mix with the new one.
                shutil.rmtree(run_dir(config))
                log(f"seed {config.seed}: removed earlier output in {run_dir(config)}")
            log(f"seed {config.seed}: training {config.run_name}")
            final_dir = train_fn(config)
        evaluate_fn(config, final_dir)
        log(f"seed {config.seed}: evaluated, logits archived")
    if config.seed not in KEEP_WEIGHTS_SEEDS and run_dir(config).exists():
        shutil.rmtree(run_dir(config))
        log(f"seed {config.seed}: deleted weights {run_dir(config)} (only seed 42 is kept)")


def check_setup(record: dict[str, object]) -> None:
    config, training = record["config"], record["training"]
    assert isinstance(config, dict) and isinstance(training, dict)
    test_rows = record["metrics"]["test"]["raw"]["n"]  # type: ignore[index]
    problems = []
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
        raise SetupError(f"{record['run_name']} is not the AC2 setup: {'; '.join(problems)}")


def judge(records: dict[int, dict[str, object]]) -> dict:
    """PASS iff every seed in SEEDS has test in-scope accuracy >= THRESHOLD."""
    missing = [s for s in SEEDS if s not in records]
    if missing:
        raise SetupError(f"no results for seeds {missing}")
    per_seed = {}
    for seed in SEEDS:
        record = records[seed]
        check_setup(record)
        test = record["metrics"]["test"]["raw"]  # type: ignore[index]
        accuracy = float(test["in_scope_accuracy_150"])
        per_seed[str(seed)] = {
            "run_name": record["run_name"],
            "in_scope_accuracy_150": accuracy,
            "oos_recall_151": float(test["oos_recall_151"]),
            "accuracy_8": float(test["accuracy_8"]),
            "oos_recall_8": float(test["oos_recall_8"]),
            "passed": accuracy >= THRESHOLD,
        }
    passed = all(entry["passed"] for entry in per_seed.values())
    return {
        "criterion": "AC2: test in_scope_accuracy_150 >= threshold for every seed",
        "split": "test",
        "threshold": THRESHOLD,
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
    from tinyrouter.evaluate import RunPaths

    records: dict[int, dict[str, object]] = {}
    for seed in SEEDS:
        config = base.with_seed(seed)
        run_seed(config, train_fn, evaluate_fn, force, log)
        path = RunPaths.of(config).results_json
        records[seed] = json.loads(path.read_text(encoding="utf-8"))
    result = judge(records)
    result["weights_kept"] = {str(s): run_dir(base.with_seed(s)).exists() for s in SEEDS}
    out = Path(base.results_root) / "ac2.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {out}: {result['verdict']}")
    return result


def default_train(config: RunConfig) -> Path:
    from tinyrouter.train import prepare_train_split, train

    return train(config, prepare_train_split(config), run_dir(config))


def default_evaluate(config: RunConfig, model_dir: Path) -> dict[str, object]:
    from tinyrouter.evaluate import evaluate

    return evaluate(config, model_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bert-base.yaml")
    parser.add_argument("--force", action="store_true", help="retrain and re-score every seed")
    args = parser.parse_args(argv)
    result = run_ac2(load_config(args.config), default_train, default_evaluate, args.force)
    for seed, entry in result["seeds"].items():
        mark = "pass" if entry["passed"] else "FAIL"
        print(f"  seed {seed}: in-scope {entry['in_scope_accuracy_150']:.4f} {mark}")
    if result["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
