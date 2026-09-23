"""Run lifecycle shared by AC2, the learning curves and the pilots.

A run is done only when its results JSON was produced by the current
config (output paths aside) and its logits archive has the same SHA-256 in
the results JSON, in the manifest and on disk. Anything else is cleared and
redone. Trained weights are reused only when their ``train_summary.json``
records the current config.

``equivalent_run`` answers a narrower question for reusing an existing run
under a different config: would training with ``config`` have run the
same code on the same rows with the same hyperparameters? See its
docstring for the two rewrites it accepts; everything else must be equal.

The pilots use ``reusable_equivalent(..., validation_only=True)``: it
hashes the whole archive file (a hash is not a result), reads only the
validation arrays and metadata from it, and keeps only the record fields
in ``VALIDATION_ONLY_FIELDS``, none of which holds a test number.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import fields
from importlib.metadata import version
from pathlib import Path

from tinyrouter.archive import (
    ArchiveError,
    load_logits,
    load_validation_logits,
    read_manifest,
    remove_from_manifest,
)
from tinyrouter.config import ADDED_FIELD_DEFAULTS, LOCATION_FIELDS, RunConfig
from tinyrouter.data import sha256_of
from tinyrouter.evaluate import RunPaths
from tinyrouter.sampling import FULL_K, OOS_TRAIN_ROWS, full_train_rows, planned_rows
from tinyrouter.steps import plan_steps

TrainFn = Callable[[RunConfig], Path]
EvaluateFn = Callable[[RunConfig, Path], dict[str, object]]
Log = Callable[[str], None]

# Fields whose effect equivalent_run() compares through the rows and the
# step plan they produce instead of by value.
DATA_AND_SCHEDULE_FIELDS = frozenset(
    {"per_intent", "k_shot", "oos_train", "num_train_epochs", "max_steps", "min_train_steps"}
)
# Record fields the validation-only path keeps; ``metrics`` (which has test) is dropped.
VALIDATION_ONLY_FIELDS = ("run_name", "config", "training", "logits", "environment")
# Libraries whose version must match for a recorded run to count as this code's run.
PINNED_LIBRARIES = ("torch", "transformers")


class RunIncompleteError(RuntimeError):
    """Evaluation returned but the run's results JSON and archive do not check out."""


def run_dir(config: RunConfig) -> Path:
    return Path(config.checkpoint_root) / config.run_name


def archive_intact(
    paths: RunPaths, record: dict[str, object], seed: int, validation_only: bool = False
) -> bool:
    """The archive's SHA-256 agrees in results JSON, manifest and file, and it is this seed's.

    ``validation_only`` checks the seed through the metadata and validation
    arrays alone, without reading the test arrays.
    """
    if not paths.logits.exists():
        return False
    logits = record.get("logits")
    recorded = logits.get("sha256") if isinstance(logits, dict) else None
    listed = read_manifest(paths.manifest).get(paths.logits.name, {}).get("sha256")
    if not recorded or not recorded == listed == sha256_of(paths.logits):
        return False
    try:
        if validation_only:
            metadata, _ = load_validation_logits(paths.logits)
        else:
            metadata = load_logits(paths.logits).metadata
    except ArchiveError:
        return False
    return metadata["seed"] == seed


def read_record(paths: RunPaths, keep: tuple[str, ...] | None = None) -> dict[str, object] | None:
    """The results JSON, or only the ``keep`` fields of it."""
    if not paths.results_json.exists():
        return None
    record = json.loads(paths.results_json.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        return None
    return record if keep is None else {k: record[k] for k in keep if k in record}


def is_done(config: RunConfig) -> bool:
    """This config's results JSON exists and its archive is intact and the one it was scored on."""
    paths = RunPaths.of(config)
    record = read_record(paths)
    if record is None or not config.matches(record.get("config")):
        return False
    return archive_intact(paths, record, config.seed)


def reusable_weights(config: RunConfig) -> bool:
    """Final weights exist and were trained with exactly this config."""
    final = run_dir(config) / "final"
    summary = final / "train_summary.json"
    if not ((final / "model.safetensors").exists() and summary.exists()):
        return False
    return config.matches(json.loads(summary.read_text(encoding="utf-8")).get("config"))


def clear_outputs(config: RunConfig, log: Log) -> None:
    """Remove this run's weights, results JSON, logits archive and manifest entry."""
    paths = RunPaths.of(config)
    removed = [str(p) for p in (paths.results_json, paths.logits) if p.exists()]
    for name in removed:
        Path(name).unlink()
    if remove_from_manifest(paths.manifest, paths.logits.name):
        removed.append(f"manifest entry {paths.logits.name}")
    if run_dir(config).exists():
        shutil.rmtree(run_dir(config))
        removed.append(str(run_dir(config)))
    if removed:
        log(f"{config.run_name}: cleared {', '.join(removed)}")


def run_one(
    config: RunConfig, train_fn: TrainFn, evaluate_fn: EvaluateFn, log: Log, keep_weights: bool
) -> None:
    """Train (or reuse weights) and evaluate unless done; delete weights unless kept."""
    if is_done(config):
        log(f"{config.run_name}: results and logits for this config already archived, skipping")
    else:
        if reusable_weights(config):
            log(f"{config.run_name}: reusing weights trained with this config")
            final_dir = run_dir(config) / "final"
        else:
            clear_outputs(config, log)
            log(f"{config.run_name}: training")
            final_dir = train_fn(config)
        evaluate_fn(config, final_dir)
        if not is_done(config):
            raise RunIncompleteError(
                f"{config.run_name}: evaluation returned but the results JSON, manifest and "
                f"archive do not agree; weights kept in {run_dir(config)} for a rerun"
            )
        log(f"{config.run_name}: evaluated, logits archived")
    if not keep_weights and run_dir(config).exists():
        shutil.rmtree(run_dir(config))
        log(f"{config.run_name}: deleted weights {run_dir(config)}")


def config_from_record(recorded: object) -> RunConfig | None:
    """Rebuild the RunConfig a results record was produced with; None if it cannot be exact."""
    if not isinstance(recorded, dict):
        return None
    filled = {**ADDED_FIELD_DEFAULTS, **recorded}
    names = {f.name for f in fields(RunConfig)}
    if set(filled) != names:
        return None
    try:
        return RunConfig(**filled)
    except (TypeError, ValueError):
        return None


def data_key(config: RunConfig) -> tuple[int, int] | None:
    """(k, oos rows) of the training sample; None for a per-intent cap (not compared this way).

    No sampling at all is the k=100 sample: CLINC150's train split has
    exactly 100 rows per intent and 250 oos rows (pinned in SPLIT_FILES and
    docs/DATA.md), and ``sample_k_shot`` at k=100 returns every row in the
    original order (tests/test_sampling.py), so the Trainer sees the same
    dataset.
    """
    if config.per_intent is not None:
        return None
    if config.k_shot is None:
        if full_train_rows() != planned_rows(FULL_K)[0]:
            return None
        return FULL_K, OOS_TRAIN_ROWS[FULL_K]
    return config.k_shot, planned_rows(config.k_shot, config.oos_train)[1]


def training_identity(config: RunConfig) -> dict[str, object] | None:
    """What the Trainer actually receives: rows, step count and every other field."""
    data = data_key(config)
    if data is None:
        return None
    plan = plan_steps(config, planned_rows(*data)[0])
    if plan.max_steps_arg < 0:
        schedule: tuple[str, float] = ("epochs", config.num_train_epochs)
    else:
        schedule = ("steps", plan.planned_steps)
    rest = {
        name: value
        for name, value in config.identity().items()
        if name not in DATA_AND_SCHEDULE_FIELDS | LOCATION_FIELDS
    }
    return {**rest, "data": data, "schedule": schedule}


def equivalent_run(config: RunConfig, record: dict[str, object]) -> bool:
    """Whether ``record`` is a run that ``config`` would have reproduced step for step.

    Two rewrites are accepted, and nothing else:

    1. no sampling (``k_shot`` and ``per_intent`` null) is the same data as
       ``k_shot=100`` with the table's 250 oos rows (see ``data_key``);
    2. a ``min_train_steps`` at or below the epoch step count does not
       change the run (``plan_steps`` keeps ``max_steps=-1`` then), and a
       fixed step count reached through ``max_steps`` or through
       ``min_train_steps`` is the same TrainingArguments.

    All other fields must be equal. The record's own training summary must
    also show the rows and steps this config plans: train rows, oos rows,
    and global_step; and the record must have been made with the torch and
    transformers versions installed now (``same_libraries``).
    """
    recorded = config_from_record(record.get("config"))
    training = record.get("training")
    if recorded is None or not isinstance(training, dict):
        return False
    if not same_libraries(record.get("environment")):
        return False
    mine, theirs = training_identity(config), training_identity(recorded)
    if mine is None or theirs is None or mine != theirs:
        return False
    k, oos = mine["data"]  # type: ignore[misc]
    rows = planned_rows(k, oos)[0]
    return (
        training.get("train_rows") == rows
        and training.get("oos_train_rows") == oos
        and training.get("global_step") == plan_steps(config, rows).planned_steps
        and training.get("seed") == config.seed
    )


def public_version(text: str) -> str:
    """Drop a local build label: the Linux CPU wheel ``2.14.0+cpu`` is release 2.14.0."""
    return text.split("+", 1)[0]


def same_libraries(environment: object) -> bool:
    """The recorded torch and transformers releases are the ones installed now."""
    if not isinstance(environment, dict):
        return False
    for name in PINNED_LIBRARIES:
        recorded = environment.get(name)
        if not isinstance(recorded, str):
            return False
        if public_version(recorded) != public_version(version(name)):
            return False
    return True


def default_train(config: RunConfig) -> Path:
    from tinyrouter.train import prepare_train_split, train

    return train(config, prepare_train_split(config), run_dir(config))


def default_evaluate(config: RunConfig, model_dir: Path) -> dict[str, object]:
    from tinyrouter.evaluate import evaluate

    return evaluate(config, model_dir)


def reusable_equivalent(
    config: RunConfig, donor: RunConfig, validation_only: bool = False
) -> dict[str, object] | None:
    """The donor run's record if it is done, intact, and equivalent to ``config``; else None.

    With ``validation_only`` the returned record holds only
    ``VALIDATION_ONLY_FIELDS`` and the archive's test arrays are never read.
    """
    paths = RunPaths.of(donor)
    record = read_record(paths, VALIDATION_ONLY_FIELDS if validation_only else None)
    if record is None or record.get("run_name") != donor.run_name:
        return None
    if not equivalent_run(config, record):
        return None
    intact = archive_intact(paths, record, config.seed, validation_only)
    return record if intact else None
