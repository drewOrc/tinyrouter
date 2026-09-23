"""runs.equivalent_run: when an existing run may stand in for a curve point, and when not."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from run_fakes import FakeRuns
from tinyrouter.config import ADDED_FIELD_DEFAULTS, load_config
from tinyrouter.evaluate import RunPaths
from tinyrouter.runs import equivalent_run, reusable_equivalent, run_one

pytestmark = pytest.mark.filterwarnings("ignore::tinyrouter.calibrate.TemperatureBoundWarning")

ROOT = Path(__file__).parent.parent
BERT = load_config(ROOT / "configs" / "bert-base.yaml")
AC2_RECORDS = {
    s: ROOT / "results" / "runs" / f"bert-base-uncased-full-seed{s}.json" for s in (42, 43, 44)
}


def curve_point(seed: int = 42, **fields: object):
    return replace(BERT, **{"k_shot": 100, "min_train_steps": 400, "seed": seed, **fields})


def ac2_like_record(seed: int = 42, training: dict | None = None) -> dict:
    """The shape of an AC2 results JSON: a config written before k_shot and S_min existed."""
    config = {
        k: v for k, v in asdict(BERT.with_seed(seed)).items() if k not in ADDED_FIELD_DEFAULTS
    }
    return {
        "run_name": BERT.with_seed(seed).run_name,
        "config": config,
        "training": {
            "seed": seed,
            "train_rows": 15_250,
            "oos_train_rows": 250,
            "global_step": 2385,
            **(training or {}),
        },
    }


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_the_committed_ac2_runs_are_equivalent_to_bert_k100_at_lr_5e5_for_every_s_min(seed):
    record = json.loads(AC2_RECORDS[seed].read_text())
    assert "k_shot" not in record["config"]  # written before the field existed
    for s_min in (None, 100, 200, 400):
        assert equivalent_run(curve_point(seed, min_train_steps=s_min), record)


def test_an_ac2_run_is_not_equivalent_to_another_seed():
    record = json.loads(AC2_RECORDS[42].read_text())
    assert not equivalent_run(curve_point(43), record)


@pytest.mark.parametrize(
    "change",
    [
        {"learning_rate": 2e-5},
        {"learning_rate": 1e-5},
        {"model_revision": "0" * 40},
        {"max_length": 128},
        {"warmup_ratio": 0.0},
        {"weight_decay": 0.0},
        {"train_batch_size": 16},
        {"num_train_epochs": 3.0},
        {"eval_batch_size": 64},
        {"device": "cpu"},
        {"k_shot": 50},
        {"oos_train": 0},
    ],
    ids=lambda c: next(iter(c)),
)
def test_any_other_difference_means_not_the_same_run(change):
    assert equivalent_run(curve_point(), ac2_like_record())
    assert not equivalent_run(curve_point(**change), ac2_like_record())


@pytest.mark.parametrize(
    "training",
    [{"global_step": 2384}, {"train_rows": 15_249}, {"oos_train_rows": 100}, {"seed": 43}],
    ids=lambda t: next(iter(t)),
)
def test_a_record_whose_own_training_summary_disagrees_is_refused(training):
    record = ac2_like_record(training=training)
    assert record["config"]["seed"] == 42
    assert not equivalent_run(curve_point(), record)


def test_a_record_missing_a_field_that_is_not_a_known_later_addition_is_refused():
    record = ac2_like_record()
    del record["config"]["warmup_ratio"]
    assert not equivalent_run(curve_point(), record)


def test_a_record_carrying_an_unknown_field_is_refused():
    record = ac2_like_record()
    record["config"]["label_smoothing"] = 0.1
    assert not equivalent_run(curve_point(), record)


def test_min_train_steps_that_decides_equals_the_same_fixed_max_steps():
    """Rewrite 2: 200 steps via S_min at k=5 is the same TrainingArguments as max_steps=200."""
    fixed = replace(BERT, k_shot=5, max_steps=200)
    by_floor = replace(BERT, k_shot=5, min_train_steps=200)
    record = {
        "run_name": fixed.run_name,
        "config": asdict(fixed),
        "training": {"seed": 42, "train_rows": 763, "oos_train_rows": 13, "global_step": 200},
    }
    assert equivalent_run(by_floor, record)
    assert not equivalent_run(replace(by_floor, min_train_steps=400), record)
    assert not equivalent_run(replace(BERT, k_shot=5), record)  # 120 epoch steps, not 200


def test_per_intent_cap_is_never_treated_as_a_k_shot_sample():
    record = ac2_like_record()
    record["config"]["per_intent"] = 100
    assert not equivalent_run(curve_point(), record)


@pytest.fixture
def tmp_bert(tmp_path):
    return replace(BERT, checkpoint_root=str(tmp_path / "ckpt"), results_root=str(tmp_path / "res"))


def test_reusable_equivalent_needs_an_intact_archive(tmp_bert):
    donor = tmp_bert.with_seed(42)
    fake = FakeRuns()
    run_one(donor, fake.train, fake.evaluate, lambda _: None, keep_weights=False)
    point = replace(tmp_bert, k_shot=100, min_train_steps=400)
    assert reusable_equivalent(point, donor) is not None
    archive = RunPaths.of(donor).logits
    archive.write_bytes(archive.read_bytes() + b"x")
    assert reusable_equivalent(point, donor) is None


def test_reusable_equivalent_is_none_when_the_donor_never_ran(tmp_bert):
    point = replace(tmp_bert, k_shot=100, min_train_steps=400)
    assert reusable_equivalent(point, tmp_bert) is None
