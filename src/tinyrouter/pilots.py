"""Validation-only pilots that fix the curve's learning rates and S_min (docs/PLAN.md section 4).

``make pilot-lr``: each encoder at k=100, seed 42, learning rate in
{1e-5, 2e-5, 5e-5}. Chosen by validation 150-way in-scope accuracy, then
validation OOS recall, then the smaller learning rate. BERT at 5e-5 is the
AC2 seed-42 run, reused when ``runs.equivalent_run`` confirms it. From
that run only the validation arrays and metadata of its archive are read
(the whole file is hashed), and only the results-JSON fields in
``runs.VALIDATION_ONLY_FIELDS``, which exclude ``metrics``.

``make pilot-steps``: both encoders at k=5, seed 42, with their chosen
learning rates, S_min in {100, 200, 400}. Chosen by the mean of the two
encoders' validation in-scope accuracy, then the smaller S_min.

Neither pilot scores, loads or writes anything from the test split:
``predict_validation`` and ``validation_scores`` raise ``LeakageError`` on
any other split, the reused AC2 archive and record are read through
``reusable_equivalent(..., validation_only=True)`` and
``load_validation_logits``, and nothing here calls ``evaluate()`` (which
scores test). Outputs go to ``results/pilots/{lr,steps}.json``. The
selected values are printed, not written into ``configs/curve.yaml``;
copying them there is a deliberate manual step.

Ties are compared on integer counts of correct rows, not on floats, and
every point is scored on the same 3,100 validation rows.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from tinyrouter.archive import load_validation_logits
from tinyrouter.calibrate import LeakageError, SplitLogits
from tinyrouter.config import RunConfig
from tinyrouter.data import Split
from tinyrouter.evaluate import RunPaths, read_training_summary
from tinyrouter.labels import load_label_space
from tinyrouter.protocol import (
    LR_GRID,
    MODELS,
    PILOT_LR_K,
    PILOT_SEED,
    PILOT_STEPS_K,
    S_MIN_GRID,
    CurveProtocol,
    ac2_donor,
    load_protocol,
)
from tinyrouter.runs import (
    Log,
    TrainFn,
    default_train,
    reusable_equivalent,
    reusable_weights,
    run_dir,
)

ValidationFn = Callable[[RunConfig, Path], SplitLogits]
PILOT_DIR = "pilots"
# Training-summary fields copied into a pilot point (no test data exists in a summary).
TRAINING_FIELDS = (
    "train_rows",
    "oos_train_rows",
    "k_shot",
    "train_sample_sha256",
    "step_plan",
    "global_step",
    "train_wall_seconds",
)


def require_validation(split_name: str) -> None:
    if split_name != "validation":
        raise LeakageError(
            f"pilots choose hyperparameters and may only see validation, got '{split_name}'"
        )


def predict_validation(model_dir: Path, split: Split, config: RunConfig) -> SplitLogits:
    """Logits of ``split``, which must be validation."""
    from tinyrouter.evaluate import predict_logits

    require_validation(split.name)
    return predict_logits(model_dir, split, config)


def validation_scores(val: SplitLogits) -> dict[str, float | int]:
    """In-scope accuracy and OOS recall on validation, with the integer counts ties use."""
    require_validation(val.split)
    oos = load_label_space().oos_intent_id
    pred, gold = val.logits.argmax(axis=1), val.labels
    in_scope, is_oos = gold != oos, gold == oos
    in_correct = int((pred[in_scope] == gold[in_scope]).sum())
    oos_correct = int((pred[is_oos] == oos).sum())
    n_in, n_oos = int(in_scope.sum()), int(is_oos.sum())
    return {
        "in_scope_correct": in_correct,
        "in_scope_n": n_in,
        "in_scope_accuracy_150": in_correct / n_in,
        "oos_correct": oos_correct,
        "oos_n": n_oos,
        "oos_recall_151": oos_correct / n_oos,
        "n": int(len(gold)),
    }


def default_validation(config: RunConfig, model_dir: Path) -> SplitLogits:
    from tinyrouter.data import load_split

    return predict_validation(model_dir, load_split("validation"), config)


@dataclass(frozen=True)
class PilotPoint:
    model: str
    value: float | int
    config: RunConfig


def pilot_location(config: RunConfig, tag: str) -> RunConfig:
    return replace(config, checkpoint_root=str(Path(config.checkpoint_root) / PILOT_DIR / tag))


def lr_points(protocol: CurveProtocol) -> list[PilotPoint]:
    """k=100 already exceeds every S_min (2,385 epoch steps), so S_min is left unset here."""
    points = []
    for model in MODELS:
        for lr in LR_GRID:
            config = replace(
                protocol.base(model),
                learning_rate=lr,
                k_shot=PILOT_LR_K,
                oos_train=None,
                min_train_steps=None,
                seed=PILOT_SEED,
            )
            points.append(PilotPoint(model, lr, pilot_location(config, f"lr{lr:g}")))
    return points


def steps_points(protocol: CurveProtocol) -> list[PilotPoint]:
    points = []
    for s_min in S_MIN_GRID:
        for model in MODELS:
            config = replace(
                protocol.base(model),
                learning_rate=protocol.learning_rate(model),
                k_shot=PILOT_STEPS_K,
                oos_train=None,
                min_train_steps=s_min,
                seed=PILOT_SEED,
            )
            points.append(PilotPoint(model, s_min, pilot_location(config, f"smin{s_min}")))
    return points


def point_entry(point: PilotPoint, training: dict, scores: dict, source: str) -> dict:
    return {
        "model": point.model,
        "value": point.value,
        "run_name": point.config.run_name,
        "config": point.config.identity(),
        "source": source,
        "training": {key: training.get(key) for key in TRAINING_FIELDS},
        "validation": scores,
    }


def reused_entry(point: PilotPoint, protocol: CurveProtocol) -> dict | None:
    """The AC2 run as this point, if it is an equivalent run; validation logits only."""
    donor = ac2_donor(protocol, point.config)
    record = (
        None if donor is None else reusable_equivalent(point.config, donor, validation_only=True)
    )
    if donor is None or record is None:
        return None
    _, val = load_validation_logits(RunPaths.of(donor).logits)
    training = record["training"]
    assert isinstance(training, dict)
    source = f"reused {donor.run_name} (equivalent run; validation logits from its archive)"
    return point_entry(point, training, validation_scores(val), source)


def trained_entry(
    point: PilotPoint, train_fn: TrainFn, validation_fn: ValidationFn, log: Log
) -> dict:
    if reusable_weights(point.config):
        final = run_dir(point.config) / "final"
        log(f"{point.model} {point.value}: reusing weights trained with this config")
    else:
        if run_dir(point.config).exists():
            shutil.rmtree(run_dir(point.config))
        log(f"{point.model} {point.value}: training {point.config.run_name}")
        final = train_fn(point.config)
    training = read_training_summary(final)
    scores = validation_scores(validation_fn(point.config, final))
    entry = point_entry(point, training, scores, "trained")
    shutil.rmtree(run_dir(point.config))
    return entry


def select_lr(entries: list[dict]) -> dict[str, float]:
    """Per model: most in-scope correct, then most OOS correct, then the smaller rate."""
    chosen = {}
    for model in MODELS:
        mine = [e for e in entries if e["model"] == model]
        same_rows(mine)
        best = max(
            mine,
            key=lambda e: (
                e["validation"]["in_scope_correct"],
                e["validation"]["oos_correct"],
                -e["value"],
            ),
        )
        chosen[model] = best["value"]
    return chosen


def select_steps(entries: list[dict]) -> int:
    """Largest total in-scope correct over both encoders (same as the mean), then smaller S_min."""
    same_rows(entries)
    totals = {
        s_min: sum(e["validation"]["in_scope_correct"] for e in entries if e["value"] == s_min)
        for s_min in S_MIN_GRID
    }
    return max(S_MIN_GRID, key=lambda s: (totals[s], -s))


def same_rows(entries: list[dict]) -> None:
    sizes = {(e["validation"]["in_scope_n"], e["validation"]["oos_n"]) for e in entries}
    if len(sizes) != 1:
        raise ValueError(f"pilot points were scored on different validation sets: {sizes}")


def load_previous(out: Path) -> list[dict]:
    if not out.exists():
        return []
    return list(json.loads(out.read_text(encoding="utf-8")).get("points", []))


def write_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_pilot(
    kind: str,
    protocol: CurveProtocol,
    out: Path,
    train_fn: TrainFn = default_train,
    validation_fn: ValidationFn = default_validation,
    log: Log = print,
) -> dict:
    """Run (or resume) one pilot and write its JSON; returns the body with the selection."""
    points = lr_points(protocol) if kind == "lr" else steps_points(protocol)
    previous = load_previous(out)
    entries: list[dict] = []
    for point in points:
        done = next(
            (
                e
                for e in previous
                if e["model"] == point.model
                and e["value"] == point.value
                and point.config.matches(e["config"])
            ),
            None,
        )
        if done is not None:
            log(f"{point.model} {point.value}: already scored with this config, skipping")
            entries.append(done)
            continue
        entry = reused_entry(point, protocol) if kind == "lr" else None
        entries.append(entry or trained_entry(point, train_fn, validation_fn, log))
        write_json(out, pilot_body(kind, entries, selected=None))
    selected = select_lr(entries) if kind == "lr" else select_steps(entries)
    body = pilot_body(kind, entries, selected)
    write_json(out, body)
    return body


def pilot_body(kind: str, entries: list[dict], selected: object) -> dict:
    if kind == "lr":
        setup = {"k": PILOT_LR_K, "seed": PILOT_SEED, "grid": list(LR_GRID)}
        rule = "per model: max validation in_scope_correct, then oos_correct, then smaller lr"
    else:
        setup = {"k": PILOT_STEPS_K, "seed": PILOT_SEED, "grid": list(S_MIN_GRID)}
        rule = "sum of both models' validation in_scope_correct, then smaller S_min"
    return {
        "pilot": "learning_rate" if kind == "lr" else "min_train_steps",
        "split": "validation",
        **setup,
        "rule": rule,
        "points": entries,
        "selected": selected,
        "note": "not applied automatically; copy into configs/curve.yaml by hand",
    }


def print_summary(body: dict) -> None:
    for e in body["points"]:
        v = e["validation"]
        print(
            f"{e['model']:<11} {e['value']!s:<7} in-scope {v['in_scope_accuracy_150']:.4f} "
            f"({v['in_scope_correct']}/{v['in_scope_n']})  OOS recall {v['oos_recall_151']:.4f}  "
            f"steps {e['training']['global_step']}  {e['source']}"
        )
    print(f"selected {body['pilot']}: {body['selected']} (validation only)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("lr", "steps"))
    parser.add_argument("--protocol", default="configs/curve.yaml")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    out = Path(args.results_root) / "pilots" / f"{args.kind}.json"
    body = run_pilot(args.kind, protocol, out)
    print_summary(body)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
