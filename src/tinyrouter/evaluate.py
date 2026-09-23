"""Score a trained model on validation and test, raw and temperature-scaled.

The temperature is fitted on validation logits only (see calibrate.py) and
then applied unchanged to test. Results go to ``results/<run_name>.json``
with the config, versions, and metrics, so tables are rebuilt from JSON,
never typed by hand.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tinyrouter.calibrate import SplitLogits, apply_temperature, fit_temperature
from tinyrouter.config import RunConfig, load_config
from tinyrouter.data import DATASET_REVISION, Split, SplitName, load_split, subsample_per_intent
from tinyrouter.labels import load_label_space
from tinyrouter.metrics import routing_metrics, softmax
from tinyrouter.train import pick_device


def predict_logits(model_dir: Path, split: Split, config: RunConfig) -> SplitLogits:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = pick_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(split), config.eval_batch_size):
            texts = list(split.texts[start : start + config.eval_batch_size])
            batch = tokenizer(
                texts,
                truncation=True,
                max_length=config.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            chunks.append(model(**batch).logits.float().cpu().numpy())
    return SplitLogits(split.name, np.concatenate(chunks), split.intents)


def score(val: SplitLogits, test: SplitLogits) -> dict[str, object]:
    """Metrics for both splits before and after temperature scaling fitted on ``val``."""
    labels = load_label_space()
    temperature = fit_temperature(val)
    out: dict[str, object] = {"temperature": temperature}
    for split_logits in (val, test):
        out[split_logits.split] = {
            "raw": routing_metrics(softmax(split_logits.logits), split_logits.labels, labels),
            "calibrated": routing_metrics(
                apply_temperature(split_logits.logits, temperature), split_logits.labels, labels
            ),
        }
    return out


def environment() -> dict[str, str]:
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "dataset_revision": DATASET_REVISION,
    }


def eval_split(name: SplitName, config: RunConfig) -> Split:
    return subsample_per_intent(load_split(name), config.eval_per_intent, config.seed)


def evaluate(config: RunConfig, model_dir: Path, results_path: Path) -> dict[str, object]:
    val = predict_logits(model_dir, eval_split("validation", config), config)
    test = predict_logits(model_dir, eval_split("test", config), config)
    record = {
        "run_name": config.run_name,
        "config": asdict(config),
        "environment": environment(),
        "metrics": score(val, test),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-dir", help="defaults to <checkpoint_root>/<run_name>/final")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.seed is not None:
        config = config.with_seed(args.seed)
    model_dir = Path(args.model_dir or Path(config.checkpoint_root) / config.run_name / "final")
    results_path = Path(config.results_root) / f"{config.run_name}.json"
    evaluate(config, model_dir, results_path)
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
