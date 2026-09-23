import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from archive_fakes import fake_metadata, fake_splits
from tinyrouter import ac2
from tinyrouter.ac2 import SEEDS, THRESHOLD, SetupError, is_done, judge, run_ac2
from tinyrouter.archive import record_in_manifest, save_logits
from tinyrouter.config import RunConfig
from tinyrouter.evaluate import RunPaths


def metrics(accuracy: float) -> dict[str, object]:
    return {
        "in_scope_accuracy_150": accuracy,
        "oos_recall_151": 0.5,
        "accuracy_8": 0.9,
        "oos_recall_8": 0.5,
        "n": ac2.EXPECTED_TEST_ROWS,
    }


def record(seed: int, accuracy: float, **config_overrides: object) -> dict[str, object]:
    config = {
        "model_name": ac2.EXPECTED_MODEL,
        "seed": seed,
        "per_intent": None,
        "eval_per_intent": None,
    }
    config.update(config_overrides)
    return {
        "run_name": ac2.expected_run_name(seed),
        "config": config,
        "training": {"seed": seed, "train_rows": 15_250, "oos_train_rows": 250},
        "logits": {"sha256": f"{seed:064d}"},
        "metrics": {"validation": {"raw": metrics(0.96)}, "test": {"raw": metrics(accuracy)}},
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


def test_verdict_lists_validation_numbers_for_tuning_next_to_test():
    result = judge(records(0.97, 0.968, 0.971))
    assert result["verdict"] == "PASS"
    assert result["threshold"] == THRESHOLD and result["split"] == "test"
    seed42 = result["seeds"]["42"]
    assert seed42["validation"]["in_scope_accuracy_150"] == 0.96
    assert set(seed42["validation"]) == set(seed42["test"]) == set(ac2.REPORTED)
    assert "validation" in result["tuning_note"]


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
    with pytest.raises(SetupError, match="seed 43"):
        judge(bad)


def test_one_result_reused_for_all_three_seeds_is_refused():
    same = record(42, 0.99)
    with pytest.raises(SetupError, match="seed 43"):
        judge({42: same, 43: same, 44: same})


def test_records_whose_seed_fields_disagree_with_their_slot_are_refused():
    bad = records(0.99, 0.99, 0.99)
    bad[44]["training"]["seed"] = 42  # type: ignore[index]
    with pytest.raises(SetupError, match="training seed 42"):
        judge(bad)


def test_two_seeds_sharing_one_logits_archive_are_refused():
    bad = records(0.99, 0.99, 0.99)
    bad[44]["logits"] = bad[43]["logits"]
    with pytest.raises(SetupError, match="same logits archive"):
        judge(bad)


@pytest.mark.parametrize(("key", "value"), [("train_rows", 15_249), ("oos_train_rows", 100)])
def test_training_set_size_other_than_full_oos_plus_is_refused(key, value):
    bad = records(0.99, 0.99, 0.99)
    bad[42]["training"][key] = value  # type: ignore[index]
    with pytest.raises(SetupError, match=key):
        judge(bad)


class FakePipeline:
    """Stands in for train() and evaluate() with the same on-disk contract."""

    def __init__(self, accuracy: float = 0.97, fail_train: int | None = None):
        self.accuracy = accuracy
        self.fail_train = fail_train
        self.fail_evaluate: int | None = None
        self.trained: list[int] = []
        self.evaluated: list[int] = []

    def train(self, config: RunConfig) -> Path:
        if config.seed == self.fail_train:
            raise RuntimeError("simulated crash during training")
        self.trained.append(config.seed)
        final = Path(config.checkpoint_root) / config.run_name / "final"
        final.mkdir(parents=True)
        (final / "model.safetensors").write_bytes(b"weights")
        summary = {"seed": config.seed, "config": asdict(config)}
        (final / "train_summary.json").write_text(json.dumps(summary))
        return final

    def evaluate(self, config: RunConfig, model_dir: Path) -> dict[str, object]:
        if config.seed == self.fail_evaluate:
            raise RuntimeError("simulated crash during evaluation")
        self.evaluated.append(config.seed)
        paths = RunPaths.of(config)
        paths.results_json.unlink(missing_ok=True)
        save_logits(paths.logits, fake_splits(seed=config.seed), fake_metadata(seed=config.seed))
        entry = record_in_manifest(paths.manifest, paths.logits)
        rec = record(config.seed, self.accuracy)
        rec["config"] = asdict(config)
        rec["logits"] = {"sha256": entry["sha256"]}
        paths.results_json.parent.mkdir(parents=True, exist_ok=True)
        paths.results_json.write_text(json.dumps(rec))
        return rec


def quiet(_: str) -> None:
    pass


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
    run_ac2(base, FakePipeline().train, FakePipeline().evaluate, log=quiet)
    again = FakePipeline()
    run_ac2(base, again.train, again.evaluate, log=quiet)
    assert again.trained == again.evaluated == []


def test_rerun_after_changing_a_hyperparameter_retrains_every_seed(base):
    """S1: tuning after a FAIL must not be judged on the old config's results."""
    fake = FakePipeline()
    run_ac2(base, fake.train, fake.evaluate, log=quiet)
    tuned = replace(base, learning_rate=3e-5)
    again = FakePipeline()
    run_ac2(tuned, again.train, again.evaluate, log=quiet)
    assert again.trained == list(SEEDS)
    stored = json.loads(RunPaths.of(tuned.with_seed(42)).results_json.read_text())
    assert stored["config"]["learning_rate"] == 3e-5


def test_weights_trained_with_another_config_are_not_reused(base):
    FakePipeline().train(base.with_seed(42))
    fake = FakePipeline()
    run_ac2(replace(base, num_train_epochs=3.0), fake.train, fake.evaluate, log=quiet)
    assert fake.trained == list(SEEDS)


def test_changing_only_output_locations_does_not_count_as_a_new_config(base):
    assert base.matches(asdict(replace(base, results_root="elsewhere", checkpoint_root="x")))
    assert not base.matches(asdict(replace(base, model_revision="other")))


def test_force_retrains_every_seed(base):
    run_ac2(base, FakePipeline().train, FakePipeline().evaluate, log=quiet)
    forced = FakePipeline()
    run_ac2(base, forced.train, forced.evaluate, force=True, log=quiet)
    assert forced.trained == list(SEEDS)


def test_force_run_that_crashes_leaves_no_old_results_to_mix_in(base):
    """S2: after FORCE crashes at seed 43, a plain rerun must redo 43 and 44, not reuse old 44."""
    run_ac2(base, FakePipeline().train, FakePipeline().evaluate, log=quiet)
    crashing = FakePipeline(fail_train=43)
    with pytest.raises(RuntimeError, match="simulated"):
        run_ac2(base, crashing.train, crashing.evaluate, force=True, log=quiet)
    assert not RunPaths.of(base.with_seed(44)).results_json.exists()
    assert (
        "bert-base-uncased-full-seed44.npz"
        not in json.loads(RunPaths.of(base).manifest.read_text())["files"]
    )
    resumed = FakePipeline()
    run_ac2(base, resumed.train, resumed.evaluate, log=quiet)
    assert resumed.trained == [43, 44]


def test_old_results_json_next_to_a_newer_archive_is_not_done(base):
    """S3: evaluation wrote archive and manifest, then crashed before the results JSON."""
    run_ac2(base, FakePipeline().train, FakePipeline().evaluate, log=quiet)
    config = base.with_seed(43)
    paths = RunPaths.of(config)
    assert is_done(config)
    save_logits(paths.logits, fake_splits(seed=99), fake_metadata(seed=43))
    record_in_manifest(paths.manifest, paths.logits)
    assert not is_done(config)
    again = FakePipeline()
    run_ac2(base, again.train, again.evaluate, log=quiet)
    assert again.trained == [43]


def test_a_seed_whose_archive_no_longer_matches_the_manifest_is_rerun(base):
    run_ac2(base, FakePipeline().train, FakePipeline().evaluate, log=quiet)
    tampered = RunPaths.of(base.with_seed(43)).logits
    tampered.write_bytes(tampered.read_bytes() + b"x")
    again = FakePipeline()
    run_ac2(base, again.train, again.evaluate, log=quiet)
    assert again.trained == [43]


def test_weights_left_by_a_crashed_evaluation_are_reused_not_retrained(base):
    FakePipeline().train(base.with_seed(42))
    fake = FakePipeline()
    run_ac2(base, fake.train, fake.evaluate, log=quiet)
    assert fake.trained == [43, 44]
    assert fake.evaluated == list(SEEDS)


def test_weights_survive_when_evaluation_of_that_seed_raises(base):
    fake = FakePipeline()
    fake.fail_evaluate = 43
    with pytest.raises(RuntimeError, match="evaluation"):
        run_ac2(base, fake.train, fake.evaluate, log=quiet)
    final = Path(base.checkpoint_root) / "bert-base-uncased-full-seed43" / "final"
    assert (final / "model.safetensors").exists()


def test_fail_verdict_makes_the_command_exit_nonzero(base, monkeypatch, capsys):
    fake = FakePipeline(accuracy=0.9)
    monkeypatch.setattr(ac2, "default_train", fake.train)
    monkeypatch.setattr(ac2, "default_evaluate", fake.evaluate)
    monkeypatch.setattr(ac2, "load_config", lambda _: base)
    with pytest.raises(SystemExit) as exc:
        ac2.main(["--config", "unused.yaml"])
    assert exc.value.code == 1
    assert "val in-scope" in capsys.readouterr().out
