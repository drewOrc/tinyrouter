import json
from pathlib import Path

import numpy as np
import pytest

from archive_fakes import fake_metadata, fake_splits
from tinyrouter import ac2
from tinyrouter.ac2 import SEEDS, THRESHOLD, SetupError, judge, run_ac2
from tinyrouter.archive import record_in_manifest, save_logits
from tinyrouter.config import RunConfig
from tinyrouter.evaluate import RunPaths


def record(seed: int, accuracy: float, **config_overrides: object) -> dict[str, object]:
    config = {"model_name": ac2.EXPECTED_MODEL, "per_intent": None, "eval_per_intent": None}
    config.update(config_overrides)
    test = {
        "in_scope_accuracy_150": accuracy,
        "oos_recall_151": 0.5,
        "accuracy_8": 0.9,
        "oos_recall_8": 0.5,
        "n": ac2.EXPECTED_TEST_ROWS,
    }
    return {
        "run_name": f"bert-base-uncased-full-seed{seed}",
        "config": config,
        "training": {"train_rows": 15_250, "oos_train_rows": 250},
        "metrics": {"test": {"raw": test}},
    }


def records(*accuracies: float) -> dict[int, dict[str, object]]:
    return {seed: record(seed, acc) for seed, acc in zip(SEEDS, accuracies, strict=True)}


def test_accuracy_exactly_at_the_threshold_passes():
    # 957 of 1000 correct, computed the way metrics.accuracy computes it.
    exact = float(np.mean(np.arange(1000) < 957))
    assert exact == THRESHOLD
    assert judge(records(exact, exact, exact))["verdict"] == "PASS"


def test_accuracy_just_below_the_threshold_fails():
    below = 4306 / 4500  # one query short of 95.7% on the 4,500 in-scope test rows
    result = judge(records(below, below, below))
    assert result["verdict"] == "FAIL"
    assert not any(entry["passed"] for entry in result["seeds"].values())


def test_one_seed_below_fails_even_when_the_mean_is_above():
    result = judge(records(0.99, 0.95, 0.99))
    assert result["verdict"] == "FAIL"
    assert [result["seeds"][str(s)]["passed"] for s in SEEDS] == [True, False, True]


def test_all_seeds_above_pass_and_report_every_number():
    result = judge(records(0.97, 0.968, 0.971))
    assert result["verdict"] == "PASS"
    assert result["threshold"] == THRESHOLD and result["split"] == "test"
    assert set(result["seeds"]["42"]) >= {"in_scope_accuracy_150", "oos_recall_151", "accuracy_8"}


def test_missing_seed_is_an_error_not_a_pass():
    partial = records(0.99, 0.99, 0.99)
    del partial[44]
    with pytest.raises(SetupError, match="44"):
        judge(partial)


@pytest.mark.parametrize(
    "overrides",
    [{"model_name": "answerdotai/ModernBERT-base"}, {"per_intent": 10}, {"eval_per_intent": 1}],
)
def test_a_record_from_a_different_setup_is_refused(overrides):
    bad = records(0.99, 0.99, 0.99)
    bad[43] = record(43, 0.99, **overrides)
    with pytest.raises(SetupError, match="seed43"):
        judge(bad)


class FakePipeline:
    """Stands in for train() and evaluate(): writes weights, an archive and a results JSON."""

    def __init__(self, accuracy: float = 0.97):
        self.accuracy = accuracy
        self.trained: list[int] = []
        self.evaluated: list[int] = []

    def train(self, config: RunConfig) -> Path:
        self.trained.append(config.seed)
        final = Path(config.checkpoint_root) / config.run_name / "final"
        final.mkdir(parents=True)
        (final / "model.safetensors").write_bytes(b"weights")
        (final / "train_summary.json").write_text("{}")
        return final

    def evaluate(self, config: RunConfig, model_dir: Path) -> dict[str, object]:
        self.evaluated.append(config.seed)
        paths = RunPaths.of(config)
        splits = fake_splits(seed=config.seed)
        save_logits(paths.logits, splits, fake_metadata(seed=config.seed))
        record_in_manifest(paths.manifest, paths.logits)
        rec = record(config.seed, self.accuracy)
        paths.results_json.parent.mkdir(parents=True, exist_ok=True)
        paths.results_json.write_text(json.dumps(rec))
        return rec


@pytest.fixture
def base(tmp_path) -> RunConfig:
    return RunConfig(
        model_name=ac2.EXPECTED_MODEL,
        model_revision="r",
        checkpoint_root=str(tmp_path / "ckpt"),
        results_root=str(tmp_path / "results"),
    )


def test_run_trains_all_seeds_writes_ac2_json_and_keeps_only_seed_42_weights(base, tmp_path):
    fake, logs = FakePipeline(), []
    result = run_ac2(base, fake.train, fake.evaluate, log=logs.append)
    assert fake.trained == fake.evaluated == list(SEEDS)
    assert result["weights_kept"] == {"42": True, "43": False, "44": False}
    assert (tmp_path / "ckpt" / "bert-base-uncased-full-seed42" / "final").exists()
    assert not (tmp_path / "ckpt" / "bert-base-uncased-full-seed43").exists()
    assert sum("deleted weights" in line for line in logs) == 2
    on_disk = json.loads((tmp_path / "results" / "ac2.json").read_text())
    assert on_disk["verdict"] == "PASS"


def test_rerun_skips_seeds_already_archived(base):
    fake = FakePipeline()
    run_ac2(base, fake.train, fake.evaluate, log=lambda _: None)
    again = FakePipeline()
    run_ac2(base, again.train, again.evaluate, log=lambda _: None)
    assert again.trained == again.evaluated == []


def test_force_retrains_every_seed(base):
    fake = FakePipeline()
    run_ac2(base, fake.train, fake.evaluate, log=lambda _: None)
    forced = FakePipeline()
    run_ac2(base, forced.train, forced.evaluate, force=True, log=lambda _: None)
    assert forced.trained == list(SEEDS)


def test_a_seed_whose_archive_no_longer_matches_the_manifest_is_rerun(base):
    fake = FakePipeline()
    run_ac2(base, fake.train, fake.evaluate, log=lambda _: None)
    tampered = RunPaths.of(base.with_seed(43)).logits
    tampered.write_bytes(tampered.read_bytes() + b"x")
    again = FakePipeline()
    run_ac2(base, again.train, again.evaluate, log=lambda _: None)
    assert again.trained == [43]


def test_weights_left_by_a_crashed_evaluation_are_reused_not_retrained(base):
    crashed = FakePipeline()
    crashed.train(base.with_seed(42))
    fake = FakePipeline()
    run_ac2(base, fake.train, fake.evaluate, log=lambda _: None)
    assert fake.trained == [43, 44]
    assert fake.evaluated == list(SEEDS)


def test_fail_verdict_makes_the_command_exit_nonzero(base, monkeypatch):
    fake = FakePipeline(accuracy=0.9)
    monkeypatch.setattr(ac2, "default_train", fake.train)
    monkeypatch.setattr(ac2, "default_evaluate", fake.evaluate)
    monkeypatch.setattr(ac2, "load_config", lambda _: base)
    with pytest.raises(SystemExit) as exc:
        ac2.main(["--config", "unused.yaml"])
    assert exc.value.code == 1
