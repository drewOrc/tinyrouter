"""Learning curves (RQ1) and the OOS ablation (RQ2), docs/PLAN.md sections 3 to 5.

``make curve MODEL=bert|modernbert``: the baselines (baselines.py), then
6 values of k x 3 seeds with the
learning rate and S_min from ``configs/curve.yaml``. ``make oos-ablation``:
ModernBERT at k=100 with 0 OOS training rows x 3 seeds. Both resume with
the rules in runs.py (a run is skipped only when its results JSON came
from this config and its archive's SHA-256 agrees in three places), and
no curve weights are kept: each run's weights are deleted once its logits
are archived.

BERT at k=100 can be the AC2 run. ``protocol.ac2_donor`` proposes the AC2
run of the same seed, and ``runs.reusable_equivalent`` accepts it only if
training with the curve config would have been the same run (same fields,
same rows, same step count, and the AC2 record's own row and step counts
agree). If the pilot picked a learning rate other than 5e-5, or anything
else differs, the point is trained like any other.

Each command writes an index, ``results/curves/<name>.json``, with one
entry per (k, seed): the run that holds its logits, the archive's SHA-256,
the planned steps and which rule decided them (``epochs`` or
``min_train_steps``), the steps actually run, and ``reused_from`` when the
point is an AC2 run.

Completion is checked, not inferred from the exit code (completeness.py).
Before any point runs, the configs must be exactly 6 values of k x 3
seeds (the ablation: k=100 with 0 OOS rows x 3 seeds) of the named
model, each once. After, the index is written to a temporary file, read
back and checked again: the same points, each once, and every archive's
SHA-256 equal on disk, in the manifest and in the index. Only then is it
moved into place and ``completed 18/18 encoder points (bert)`` (or
``completed 3/3 ablation points``) printed; any failure raises and the
command exits non-zero.

Every point's training sample is checked against ``curve_sample(k, seed)``
recomputed now (``sample_fingerprint``). The encoder's resume check
compares configs only, so a numpy release that changed the sampler's
random stream would otherwise pair old encoder runs with new baseline
samples without notice. AC2 runs predate the fingerprint field; for them
the index records the fingerprint computed now and says so in
``train_sample_sha256_source``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import cache
from pathlib import Path

from tinyrouter.baselines import run_all as run_baselines
from tinyrouter.completeness import (
    ABLATION_POINTS,
    CURVE_POINTS,
    ENCODER_KEY,
    Key,
    check_points,
    publish_index,
)
from tinyrouter.config import RunConfig
from tinyrouter.evaluate import RunPaths
from tinyrouter.protocol import (
    ABLATION_MODEL,
    MODELS,
    CurveProtocol,
    ac2_donor,
    load_protocol,
)
from tinyrouter.runs import (
    EvaluateFn,
    Log,
    TrainFn,
    archive_intact,
    default_evaluate,
    default_train,
    read_record,
    reusable_equivalent,
    run_one,
)
from tinyrouter.sampling import curve_sample, sample_fingerprint
from tinyrouter.steps import plan_steps

FingerprintFn = Callable[[RunConfig], str]
BACKFILLED_SAMPLE = "computed at index time from curve_sample(k, seed); the run predates the field"
ABLATION_NAME = "oos-ablation"


@cache
def _fingerprint(k: int, seed: int, oos_train: int | None) -> str:
    return sample_fingerprint(curve_sample(k, seed, oos_train))


def expected_fingerprint(config: RunConfig) -> str:
    """Fingerprint of the rows ``config`` should train on, drawn with the numpy installed now."""
    assert config.k_shot is not None
    return _fingerprint(config.k_shot, config.seed, config.oos_train)


class CurveError(RuntimeError):
    """A finished curve point does not look like the run it was supposed to be."""


def checked_sample(
    config: RunConfig, record: dict, reused_from: str | None, fingerprint_fn: FingerprintFn
) -> tuple[str, str]:
    """(fingerprint, source): the run's recorded sample must be the one drawn now."""
    expected = fingerprint_fn(config)
    recorded = record["training"].get("train_sample_sha256")
    if recorded is None and reused_from is not None:
        return expected, BACKFILLED_SAMPLE
    if recorded != expected:
        raise CurveError(
            f"{record['run_name']}: trained on sample {recorded}, but curve_sample("
            f"{config.k_shot}, {config.seed}) now gives {expected}; the sampler's output "
            "changed (numpy upgrade?), so this run and the baselines no longer share rows"
        )
    return expected, "recorded by train()"


def index_entry(
    config: RunConfig,
    record: dict,
    reused_from: str | None,
    fingerprint_fn: FingerprintFn | None = None,
) -> dict[str, object]:
    training, logits = record["training"], record["logits"]
    plan = plan_steps(config, training["train_rows"])
    if training["global_step"] != plan.planned_steps:
        raise CurveError(
            f"{record['run_name']}: ran {training['global_step']} steps, "
            f"{config.run_name} plans {plan.planned_steps}"
        )
    if reused_from is None and training.get("step_plan") != plan.as_dict():
        raise CurveError(f"{record['run_name']}: recorded step plan differs from this config's")
    if reused_from is None and not archive_intact(RunPaths.of(config), record, config.seed):
        raise CurveError(f"{record['run_name']}: archive does not match its record and manifest")
    fingerprint_fn = fingerprint_fn or expected_fingerprint
    fingerprint, source = checked_sample(config, record, reused_from, fingerprint_fn)
    return {
        "k": config.k_shot,
        "seed": config.seed,
        "oos_train": config.oos_train,
        "run_name": record["run_name"],
        "logits_file": logits["file"],
        "logits_sha256": logits["sha256"],
        "train_rows": training["train_rows"],
        "oos_train_rows": training["oos_train_rows"],
        "train_sample_sha256": fingerprint,
        "train_sample_sha256_source": source,
        "planned_steps": plan.planned_steps,
        "epoch_steps": plan.epoch_steps,
        "decided_by": plan.decided_by,
        "global_step": training["global_step"],
        "reused_from": reused_from,
    }


def run_point(
    config: RunConfig,
    protocol: CurveProtocol,
    train_fn: TrainFn,
    evaluate_fn: EvaluateFn,
    log: Log,
    fingerprint_fn: FingerprintFn | None,
) -> dict[str, object]:
    donor = ac2_donor(protocol, config)
    if donor is not None:
        record = reusable_equivalent(config, donor)
        if record is not None:
            log(f"{config.run_name}: reusing {donor.run_name} (equivalent run, archive intact)")
            return index_entry(config, record, donor.run_name, fingerprint_fn)
        log(f"{config.run_name}: {donor.run_name} is not an equivalent intact run; training")
    run_one(config, train_fn, evaluate_fn, log, keep_weights=False)
    record = read_record(RunPaths.of(config))
    if record is None:
        raise CurveError(f"{config.run_name}: no results JSON after evaluation")
    return index_entry(config, record, None, fingerprint_fn)


def planned_shape(name: str, protocol: CurveProtocol) -> tuple[frozenset[Key], str]:
    """The (k, seed, oos_train) points and the model the index called ``name`` must hold."""
    if name == ABLATION_NAME:
        return ABLATION_POINTS, protocol.base(ABLATION_MODEL).model_name
    if name in MODELS:
        return CURVE_POINTS, protocol.base(name).model_name
    raise CurveError(f"unknown curve '{name}'; expected one of {(*MODELS, ABLATION_NAME)}")


def check_plan(name: str, configs: list[RunConfig], protocol: CurveProtocol) -> frozenset[Key]:
    """Refuse to start unless ``configs`` are the expected points of one model, each once."""
    expected, model = planned_shape(name, protocol)
    others = sorted({c.model_name for c in configs} - {model})
    if others:
        raise CurveError(f"{name}: configs for {others}; this index holds only {model}")
    check_points(((c.k_shot, c.seed, c.oos_train) for c in configs), expected, f"{name} plan")
    return expected


def completion_message(name: str, count: int, expected: int) -> str:
    if name == ABLATION_NAME:
        return f"completed {count}/{expected} ablation points"
    return f"completed {count}/{expected} encoder points ({name})"


def run_curve(
    name: str,
    configs: list[RunConfig],
    protocol: CurveProtocol,
    train_fn: TrainFn = default_train,
    evaluate_fn: EvaluateFn = default_evaluate,
    log: Log = print,
    fingerprint_fn: FingerprintFn | None = None,
) -> dict[str, object]:
    """Run or resume every config, then write and verify results/curves/<name>.json."""
    expected = check_plan(name, configs, protocol)
    entries = [run_point(c, protocol, train_fn, evaluate_fn, log, fingerprint_fn) for c in configs]
    shas = [e["logits_sha256"] for e in entries]
    if len(set(shas)) != len(shas):
        raise CurveError(f"{name}: two points share one logits archive")
    first = configs[0]
    body = {
        "name": name,
        "model_name": first.model_name,
        "model_revision": first.model_revision,
        "learning_rate": first.learning_rate,
        "min_train_steps": first.min_train_steps,
        "points": entries,
    }
    root = Path(first.results_root)
    out = root / "curves" / f"{name}.json"
    count = publish_index(out, body, ENCODER_KEY, expected, root, name)
    log(f"wrote {out} ({count} points)")
    log(completion_message(name, count, len(expected)))
    return body


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--model", choices=MODELS)
    which.add_argument("--ablation", action="store_true", help="ModernBERT, k=100, 0 OOS rows")
    parser.add_argument("--protocol", default="configs/curve.yaml")
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    # Build every config first: a protocol still missing a pilot value stops here,
    # before anything is computed or written.
    if args.ablation:
        run_curve(ABLATION_NAME, protocol.ablation_configs(), protocol)
        return
    configs = protocol.curve_configs(args.model)
    run_baselines(Path(configs[0].results_root))
    run_curve(args.model, configs, protocol)


if __name__ == "__main__":
    main()
