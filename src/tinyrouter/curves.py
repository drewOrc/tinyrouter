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
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tinyrouter.baselines import run_all as run_baselines
from tinyrouter.config import RunConfig
from tinyrouter.evaluate import RunPaths
from tinyrouter.protocol import MODELS, CurveProtocol, ac2_donor, load_protocol
from tinyrouter.runs import (
    EvaluateFn,
    Log,
    TrainFn,
    default_evaluate,
    default_train,
    read_record,
    reusable_equivalent,
    run_one,
)
from tinyrouter.steps import plan_steps


class CurveError(RuntimeError):
    """A finished curve point does not look like the run it was supposed to be."""


def index_entry(config: RunConfig, record: dict, reused_from: str | None) -> dict[str, object]:
    training, logits = record["training"], record["logits"]
    plan = plan_steps(config, training["train_rows"])
    if training["global_step"] != plan.planned_steps:
        raise CurveError(
            f"{record['run_name']}: ran {training['global_step']} steps, "
            f"{config.run_name} plans {plan.planned_steps}"
        )
    if reused_from is None and training.get("step_plan") != plan.as_dict():
        raise CurveError(f"{record['run_name']}: recorded step plan differs from this config's")
    return {
        "k": config.k_shot,
        "seed": config.seed,
        "oos_train": config.oos_train,
        "run_name": record["run_name"],
        "logits_file": logits["file"],
        "logits_sha256": logits["sha256"],
        "train_rows": training["train_rows"],
        "oos_train_rows": training["oos_train_rows"],
        "train_sample_sha256": training.get("train_sample_sha256"),
        "planned_steps": plan.planned_steps,
        "epoch_steps": plan.epoch_steps,
        "decided_by": plan.decided_by,
        "global_step": training["global_step"],
        "reused_from": reused_from,
    }


def run_point(
    config: RunConfig, protocol: CurveProtocol, train_fn: TrainFn, evaluate_fn: EvaluateFn, log: Log
) -> dict[str, object]:
    donor = ac2_donor(protocol, config)
    if donor is not None:
        record = reusable_equivalent(config, donor)
        if record is not None:
            log(f"{config.run_name}: reusing {donor.run_name} (equivalent run, archive intact)")
            return index_entry(config, record, donor.run_name)
        log(f"{config.run_name}: {donor.run_name} is not an equivalent intact run; training")
    run_one(config, train_fn, evaluate_fn, log, keep_weights=False)
    record = read_record(RunPaths.of(config))
    if record is None:
        raise CurveError(f"{config.run_name}: no results JSON after evaluation")
    return index_entry(config, record, None)


def run_curve(
    name: str,
    configs: list[RunConfig],
    protocol: CurveProtocol,
    train_fn: TrainFn = default_train,
    evaluate_fn: EvaluateFn = default_evaluate,
    log: Log = print,
) -> dict[str, object]:
    """Run or resume every config, then write results/curves/<name>.json."""
    entries = [run_point(c, protocol, train_fn, evaluate_fn, log) for c in configs]
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
    out = Path(first.results_root) / "curves" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    log(f"wrote {out} ({len(entries)} points)")
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
        run_curve("oos-ablation", protocol.ablation_configs(), protocol)
        return
    configs = protocol.curve_configs(args.model)
    run_baselines(Path(configs[0].results_root))
    run_curve(args.model, configs, protocol)


if __name__ == "__main__":
    main()
