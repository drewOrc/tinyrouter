"""A stand-in for train() and evaluate() that writes the same files, for pilot and curve tests.

The training summary carries what the real one does for these tests: the
row counts ``planned_rows`` gives, the step plan and a ``global_step`` equal
to it. Evaluation writes a real archive, manifest entry and results JSON.
"""

import json
from dataclasses import asdict
from pathlib import Path

from archive_fakes import fake_metadata
from tinyrouter.archive import record_in_manifest, save_logits
from tinyrouter.calibrate import SplitLogits
from tinyrouter.config import RunConfig
from tinyrouter.evaluate import RunPaths, score
from tinyrouter.sampling import planned_rows
from tinyrouter.steps import plan_steps


def summary_for(config: RunConfig) -> dict[str, object]:
    if config.k_shot is None:
        rows, oos = 15_250, 250
    else:
        rows, oos = planned_rows(config.k_shot, config.oos_train)
    plan = plan_steps(config, rows)
    return {
        "run_name": config.run_name,
        "seed": config.seed,
        "config": asdict(config),
        "train_rows": rows,
        "oos_train_rows": oos,
        "k_shot": config.k_shot,
        "train_sample_sha256": f"sample-{config.k_shot}-{config.seed}",
        "step_plan": plan.as_dict(),
        "global_step": plan.planned_steps,
        "train_wall_seconds": 1.0,
    }


class FakeRuns:
    def __init__(self) -> None:
        self.trained: list[str] = []
        self.evaluated: list[str] = []

    def train(self, config: RunConfig) -> Path:
        self.trained.append(f"{config.run_name}@{config.checkpoint_root}")
        final = Path(config.checkpoint_root) / config.run_name / "final"
        final.mkdir(parents=True)
        (final / "model.safetensors").write_bytes(b"weights")
        (final / "train_summary.json").write_text(json.dumps(summary_for(config)))
        return final

    def evaluate(self, config: RunConfig, model_dir: Path) -> dict[str, object]:
        self.evaluated.append(config.run_name)
        paths = RunPaths.of(config)
        paths.results_json.unlink(missing_ok=True)
        training = json.loads((model_dir / "train_summary.json").read_text())
        shifted = fake_validation(100 + config.seed + (config.k_shot or 0))
        test = SplitLogits("test", shifted.logits, shifted.labels)
        splits = {"validation": self.validation(config, model_dir), "test": test}
        meta = fake_metadata(
            run_name=config.run_name,
            seed=config.seed,
            k_shot=config.k_shot,
            train_rows=training["train_rows"],
            oos_train_rows=training["oos_train_rows"],
        )
        save_logits(paths.logits, splits, meta)
        entry = record_in_manifest(paths.manifest, paths.logits)
        record = {
            "run_name": config.run_name,
            "config": asdict(config),
            "training": training,
            "logits": {"file": paths.logits.name, "sha256": entry["sha256"]},
            "metrics": score(splits["validation"], splits["test"]),
        }
        paths.results_json.parent.mkdir(parents=True, exist_ok=True)
        paths.results_json.write_text(json.dumps(record))
        return record

    def validation(self, config: RunConfig, model_dir: Path) -> SplitLogits:
        """Validation logits whose accuracy the test controls through ``self.correct``.

        The seed shifts the count by one so different seeds give different archives.
        """
        correct = getattr(self, "correct", {}).get(self.key(config), 0)
        return fake_validation(correct + config.seed % 42)

    @staticmethod
    def key(config: RunConfig) -> tuple[str, object]:
        value = config.min_train_steps if config.k_shot == 5 else config.learning_rate
        return config.model_name, value


def fake_validation(in_scope_correct: int, oos_correct: int = 0) -> SplitLogits:
    """3,000 in-scope rows (20 per intent) and 100 oos rows, with chosen counts right."""
    import numpy as np

    from tinyrouter.labels import load_label_space

    oos = load_label_space().oos_intent_id
    in_scope_ids = [i for i in range(151) if i != oos]
    gold = np.array([i for i in in_scope_ids for _ in range(20)] + [oos] * 100, dtype=np.int64)
    pred = np.where(gold == oos, (oos + 1) % 151, (gold + 1) % 151)
    pred[np.flatnonzero(pred == oos)] = (oos + 2) % 151
    pred[:in_scope_correct] = gold[:in_scope_correct]
    oos_rows = np.flatnonzero(gold == oos)[:oos_correct]
    pred[oos_rows] = oos
    logits = np.full((len(gold), 151), -3.0, dtype=np.float32)
    logits[np.arange(len(gold)), pred] = 3.0
    return SplitLogits("validation", logits, gold)
