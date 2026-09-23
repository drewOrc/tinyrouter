"""Short compatibility trial of an encoder on this machine (docs/PLAN.md section 4).

Trains ``--steps`` optimizer steps on the full train split with the
config's batch size and max_length, then scores validation once. Records
the wall time of every step, the sampled peak memory, the loss of every
step and whether any loss or logit was NaN. Used to check that
ModernBERT trains on MPS and to compare its per-step time with
bert-base-uncased under identical settings.

The numbers after 50 steps are not a result; the model is barely trained.
Validation only: this script never loads the test split.

    uv run python scripts/compat_trial.py --config configs/modernbert-base.yaml \\
        --learning-rate 5e-5 --out results/compat/modernbert-base.json --keep-weights DIR
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from tinyrouter.config import load_config
from tinyrouter.data import load_split
from tinyrouter.efficiency import (
    count_parameters,
    memory_report,
    sampler_callback,
    start_measurement,
)
from tinyrouter.evaluate import environment, predict_logits
from tinyrouter.labels import load_label_space
from tinyrouter.metrics import in_scope_accuracy, oos_recall
from tinyrouter.steps import plan_steps
from tinyrouter.train import (
    load_model_and_tokenizer,
    pick_device,
    to_hf_dataset,
    training_arguments,
)

AC2_EPOCH_STEPS = 2385


def step_timer(times: list[float]) -> object:
    from transformers import TrainerCallback

    class StepTimer(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):  # noqa: ANN001
            self.started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            times.append(time.perf_counter() - self.started)

    return StepTimer()


def run_trial(config_path: str, steps: int, lr: float, work: Path) -> tuple[dict, Path]:
    from transformers import DataCollatorWithPadding, Trainer, set_seed

    config = replace(
        load_config(config_path), learning_rate=lr, max_steps=steps, checkpoint_root=str(work)
    )
    set_seed(config.seed)
    device = pick_device(config.device)
    labels = load_label_space()
    model, tokenizer = load_model_and_tokenizer(config, labels)
    train_split = load_split("train")
    sampler = start_measurement(device)
    times: list[float] = []
    out_dir = work / config.run_name
    trainer = Trainer(
        model=model,  # type: ignore[arg-type]
        args=training_arguments(config, out_dir, device),  # type: ignore[arg-type]
        train_dataset=to_hf_dataset(train_split, tokenizer, config.max_length),  # type: ignore[arg-type]
        data_collator=DataCollatorWithPadding(tokenizer),  # type: ignore[arg-type]
        processing_class=tokenizer,  # type: ignore[arg-type]
        callbacks=[sampler_callback(sampler), step_timer(times)],  # type: ignore[list-item]
    )
    started = time.perf_counter()
    result = trainer.train()
    wall = time.perf_counter() - started
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    final = out_dir / "final"
    trainer.save_model(str(final))
    val = predict_logits(final, load_split("validation"), config)
    pred, oos = val.logits.argmax(axis=1), labels.oos_intent_id
    steady = times[5:] if len(times) > 10 else times
    record = {
        "config": config_path,
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "learning_rate": lr,
        "train_batch_size": config.train_batch_size,
        "max_length": config.max_length,
        "steps": result.global_step,
        "wall_seconds": round(wall, 3),
        "step_seconds_median": round(float(np.median(steady)), 4),
        "step_seconds_mean_after_5": round(float(np.mean(steady)), 4),
        "step_seconds_first": round(times[0], 4),
        "estimated_full_5_epoch_seconds": round(float(np.median(steady)) * AC2_EPOCH_STEPS, 1),
        "loss_first_5": losses[:5],
        "loss_last_5": losses[-5:],
        "loss_decreased": float(np.mean(losses[-10:])) < float(np.mean(losses[:10])),
        "any_nan_loss": any(not math.isfinite(x) for x in losses),
        "validation_logits_finite": bool(np.isfinite(val.logits).all()),
        "validation_in_scope_accuracy_150": in_scope_accuracy(pred, val.labels, oos),
        "validation_oos_recall_151": oos_recall(pred, val.labels, oos),
        "parameters": count_parameters(model),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "peak_memory": memory_report(device, sampler),
        "environment": environment(config),
        "plan_note": f"full 5 epochs = {plan_steps(replace(config, max_steps=-1), 15_250)}",
    }
    return record, final


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--work", default="checkpoints/compat")
    parser.add_argument("--out", required=True)
    parser.add_argument("--keep-weights", action="store_true", help="for the ONNX export trial")
    args = parser.parse_args(argv)
    work = Path(args.work)
    record, final = run_trial(args.config, args.steps, args.learning_rate, work)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "environment"}, indent=2))
    if args.keep_weights:
        print(f"weights kept at {final}")
    else:
        shutil.rmtree(final.parent)
        print(f"deleted {final.parent}")


if __name__ == "__main__":
    main()
