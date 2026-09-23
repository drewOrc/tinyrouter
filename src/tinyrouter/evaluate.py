"""Score a trained model on validation and test, raw and temperature-scaled.

The temperature is fitted on validation logits only (see calibrate.py) and
then applied unchanged to test. Each run writes three things under
``results_root``:

- ``runs/<run_name>.json``: config, versions, training cost, metrics
  (committed; tables are rebuilt from these, never typed by hand);
- ``logits/<run_name>.npz``: per-example validation and test logits
  (not committed, see archive.py);
- an entry in ``logits-manifest.json`` with that archive's SHA-256
  (committed).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tinyrouter.archive import (
    FORMAT_VERSION,
    MANIFEST_NAME,
    git_state,
    load_logits,
    pinned_dataset_sha256,
    record_in_manifest,
    save_logits,
    utc_now,
)
from tinyrouter.calibrate import SplitLogits, apply_temperature, fit_temperature
from tinyrouter.config import RunConfig, load_config
from tinyrouter.data import DATASET_REVISION, Split, SplitName, load_split, subsample_per_intent
from tinyrouter.labels import load_label_space
from tinyrouter.metrics import routing_metrics, softmax
from tinyrouter.train import SUMMARY_NAME, pick_device


@dataclass(frozen=True)
class RunPaths:
    results_json: Path
    logits: Path
    manifest: Path

    @classmethod
    def of(cls, config: RunConfig) -> RunPaths:
        return cls.named(config.results_root, config.run_name)

    @classmethod
    def named(cls, results_root: str | Path, run_name: str) -> RunPaths:
        root = Path(results_root)
        return cls(
            results_json=root / "runs" / f"{run_name}.json",
            logits=root / "logits" / f"{run_name}.npz",
            manifest=root / MANIFEST_NAME,
        )


def predict_logits(model_dir: Path, split: Split, config: RunConfig) -> SplitLogits:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = pick_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    check_model_labels(model.config.id2label, model_dir)
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


class LabelMismatchError(ValueError):
    """The model's output columns are not the intents the current label space lists."""


def check_model_labels(id2label: dict, model_dir: Path) -> None:
    """Refuse a model whose id2label differs from resources/intent_names.json, in order."""
    names = load_label_space().intent_names
    model_names = tuple(str(label) for _, label in sorted((int(k), v) for k, v in id2label.items()))
    if model_names != names:
        diff = next(
            (i for i, (a, b) in enumerate(zip(model_names, names, strict=False)) if a != b),
            min(len(model_names), len(names)),
        )
        raise LabelMismatchError(
            f"{model_dir}: model id2label ({len(model_names)} labels) differs from "
            f"intent_names.json ({len(names)}) first at column {diff}; logit column i "
            "would not mean intent i"
        )


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


def device_name(device: str) -> str:
    import torch

    if device == "cuda":
        return torch.cuda.get_device_name()
    if platform.system() == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def environment(config: RunConfig) -> dict[str, object]:
    import torch
    import transformers

    device = pick_device(config.device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "device": device,
        "device_name": device_name(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "dataset_revision": DATASET_REVISION,
    }


def eval_split(name: SplitName, config: RunConfig) -> Split:
    return subsample_per_intent(load_split(name), config.eval_per_intent, config.seed)


def read_training_summary(model_dir: Path) -> dict[str, object]:
    path = model_dir / SUMMARY_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; evaluate needs the summary train() writes next to the weights "
            "(training set size goes into the logits metadata)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def archive_metadata(config: RunConfig, training: dict[str, object]) -> dict[str, object]:
    commit, dirty = git_state()
    return {
        "format_version": FORMAT_VERSION,
        "run_name": config.run_name,
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "seed": config.seed,
        "per_intent": config.per_intent,
        "k_shot": config.k_shot,
        "train_rows": training["train_rows"],
        "oos_train_rows": training["oos_train_rows"],
        "eval_per_intent": config.eval_per_intent,
        "dataset_revision": DATASET_REVISION,
        # load_split verified every file against these before anything read it.
        "dataset_sha256": pinned_dataset_sha256(),
        "label_space_sha256": load_label_space().sha256,
        "git_commit": commit,
        "git_dirty": dirty,
        "created_at": utc_now(),
    }


def evaluate(config: RunConfig, model_dir: Path) -> dict[str, object]:
    """Score the model, archive its logits, and write the results JSON; return the record."""
    paths = RunPaths.of(config)
    # A crash below must not leave the previous run's JSON next to this run's archive.
    paths.results_json.unlink(missing_ok=True)
    training = read_training_summary(model_dir)
    val = predict_logits(model_dir, eval_split("validation", config), config)
    test = predict_logits(model_dir, eval_split("test", config), config)
    save_logits(paths.logits, {"validation": val, "test": test}, archive_metadata(config, training))
    entry = record_in_manifest(paths.manifest, paths.logits)
    # Metrics come from the archive as written, so they match what later analysis reads.
    archived = load_logits(paths.logits)
    record = {
        "run_name": config.run_name,
        "config": asdict(config),
        "environment": environment(config),
        "training": training,
        "logits": {"file": paths.logits.name, "sha256": entry["sha256"], "bytes": entry["bytes"]},
        "metrics": score(archived.validation, archived.test),
    }
    paths.results_json.parent.mkdir(parents=True, exist_ok=True)
    paths.results_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
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
    evaluate(config, model_dir)
    paths = RunPaths.of(config)
    print(f"wrote {paths.results_json} and {paths.logits}")


if __name__ == "__main__":
    main()
