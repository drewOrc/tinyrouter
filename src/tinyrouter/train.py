"""Fine-tune a sequence classifier on the 151 CLINC150 intents with HF Trainer.

Every hyperparameter comes from a RunConfig. Only the final weights are
kept: ``save_total_limit=1`` bounds disk use during training and the
remaining ``checkpoint-*`` directory is deleted once ``final/`` is written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from tinyrouter.config import RunConfig, load_config
from tinyrouter.data import Split, load_split, subsample_per_intent
from tinyrouter.efficiency import (
    count_parameters,
    memory_report,
    sampler_callback,
    start_measurement,
)
from tinyrouter.labels import LabelSpace, load_label_space

SUMMARY_NAME = "train_summary.json"


def pick_device(requested: str = "auto") -> str:
    """Resolve 'auto' to cuda, then mps, then cpu; validate an explicit choice."""
    import torch

    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        return next(name for name in ("cuda", "mps", "cpu") if available[name])
    if not available.get(requested, False):
        raise RuntimeError(f"device '{requested}' requested but not available on this machine")
    return requested


def to_hf_dataset(split: Split, tokenizer: object, max_length: int) -> object:
    from datasets import Dataset

    ds = Dataset.from_dict({"text": list(split.texts), "label": split.intents.tolist()})
    return ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),  # type: ignore[operator]
        batched=True,
        remove_columns=["text"],
    )


def load_model_and_tokenizer(config: RunConfig, labels: LabelSpace) -> tuple[object, object]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        num_labels=labels.num_intents,
        id2label=dict(enumerate(labels.intent_names)),
        label2id={name: i for i, name in enumerate(labels.intent_names)},
        ignore_mismatched_sizes=config.replace_classifier_head,
    )
    return model, tokenizer


def training_arguments(config: RunConfig, output_dir: Path, device: str) -> object:
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        seed=config.seed,
        data_seed=config.seed,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        # transformers 5.x reads warmup_steps < 1 as a fraction of total steps.
        warmup_steps=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        eval_strategy="no",
        save_strategy="epoch" if config.max_steps < 0 else "no",
        save_total_limit=1,
        save_only_model=True,
        logging_strategy="epoch" if config.max_steps < 0 else "steps",
        logging_steps=1,
        report_to="none",
        use_cpu=device == "cpu",
        dataloader_pin_memory=device == "cuda",
        disable_tqdm=True,
    )


def train(config: RunConfig, train_split: Split, output_dir: Path) -> Path:
    """Train on ``train_split`` and return the directory holding the final weights.

    ``final/train_summary.json`` records the training set size, parameter
    counts, wall time, steps and peak memory; evaluation copies it into the
    results JSON, because the weights (and this file) may be deleted later.
    """
    from transformers import DataCollatorWithPadding, Trainer, set_seed

    if train_split.name != "train":
        raise ValueError(f"train() only accepts the train split, got '{train_split.name}'")
    set_seed(config.seed)
    labels = load_label_space()
    device = pick_device(config.device)
    model, tokenizer = load_model_and_tokenizer(config, labels)
    sampler = start_measurement(device)
    trainer = Trainer(
        model=model,  # type: ignore[arg-type]
        args=training_arguments(config, output_dir, device),  # type: ignore[arg-type]
        train_dataset=to_hf_dataset(train_split, tokenizer, config.max_length),  # type: ignore[arg-type]
        data_collator=DataCollatorWithPadding(tokenizer),  # type: ignore[arg-type]
        processing_class=tokenizer,  # type: ignore[arg-type]
        callbacks=[sampler_callback(sampler)],  # type: ignore[list-item]
    )
    started = time.perf_counter()
    result = trainer.train()
    wall_seconds = time.perf_counter() - started
    sampler.sample()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    for leftover in output_dir.glob("checkpoint-*"):
        shutil.rmtree(leftover)
    summary = {
        "run_name": config.run_name,
        "device": device,
        "train_rows": len(train_split),
        "oos_train_rows": int((train_split.intents == labels.oos_intent_id).sum()),
        "per_intent": config.per_intent,
        "global_step": result.global_step,
        "train_loss": result.training_loss,
        "parameters": count_parameters(model),
        "train_wall_seconds": round(wall_seconds, 3),
        "peak_memory": memory_report(device, sampler),
    }
    (final_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return final_dir


def prepare_train_split(config: RunConfig) -> Split:
    return subsample_per_intent(load_split("train"), config.per_intent, config.seed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.seed is not None:
        config = config.with_seed(args.seed)
    output_dir = Path(config.checkpoint_root) / config.run_name
    final_dir = train(config, prepare_train_split(config), output_dir)
    print(f"final weights: {final_dir}")


if __name__ == "__main__":
    main()
