import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from archive_fakes import fake_splits
from run_fakes import FakeRuns, fake_validation
from tinyrouter import pilots
from tinyrouter.calibrate import LeakageError, SplitLogits
from tinyrouter.config import load_config
from tinyrouter.data import Split
from tinyrouter.protocol import CurveProtocol, ProtocolError
from tinyrouter.runs import run_one

pytestmark = pytest.mark.filterwarnings("ignore::tinyrouter.calibrate.TemperatureBoundWarning")

ROOT = Path(__file__).parent.parent
BERT_NAME = "google-bert/bert-base-uncased"
MODERN_NAME = "answerdotai/ModernBERT-base"


def make_protocol(tmp_path, s_min=None, bert_lr=None, modern_lr=None, **bert_changes):
    def here(config):
        return replace(
            config, checkpoint_root=str(tmp_path / "ckpt"), results_root=str(tmp_path / "res")
        )

    bert = here(replace(load_config(ROOT / "configs" / "bert-base.yaml"), **bert_changes))
    modern = here(load_config(ROOT / "configs" / "modernbert-base.yaml"))
    return CurveProtocol(
        s_min,
        {"bert": bert_lr, "modernbert": modern_lr},
        {"bert": bert, "modernbert": modern},
        {"bert": "b", "modernbert": "m"},
    )


def quiet(_: str) -> None:
    pass


def keys_anywhere(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(keys_anywhere(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(keys_anywhere(v) for v in value)) if value else set()
    return set()


def test_scoring_refuses_test_logits():
    with pytest.raises(LeakageError, match="only see validation"):
        pilots.validation_scores(fake_splits()["test"])


def test_prediction_refuses_the_test_split_before_touching_a_model(tmp_path):
    split = Split("test", ("a",), np.array([0]))
    with pytest.raises(LeakageError):
        pilots.predict_validation(
            tmp_path / "no-such-model", split, load_config(ROOT / "configs/bert-base.yaml")
        )


def test_validation_scores_count_in_scope_and_oos_hits():
    scores = pilots.validation_scores(fake_validation(2_901, oos_correct=68))
    assert scores["in_scope_correct"] == 2_901 and scores["in_scope_n"] == 3_000
    assert scores["oos_correct"] == 68 and scores["oos_n"] == 100
    assert scores["in_scope_accuracy_150"] == 2_901 / 3_000


def entry(model, value, in_scope, oos=0):
    return {
        "model": model,
        "value": value,
        "validation": pilots.validation_scores(fake_validation(in_scope, oos)),
    }


def test_lr_selection_breaks_ties_on_oos_recall_then_on_the_smaller_rate():
    entries = [
        entry("bert", 1e-5, 2_900, 60),
        entry("bert", 2e-5, 2_900, 70),
        entry("bert", 5e-5, 2_899, 99),
        entry("modernbert", 1e-5, 2_950, 50),
        entry("modernbert", 2e-5, 2_950, 50),
        entry("modernbert", 5e-5, 2_949, 90),
    ]
    assert pilots.select_lr(entries) == {"bert": 2e-5, "modernbert": 1e-5}


def test_steps_selection_uses_both_encoders_and_prefers_the_smaller_s_min_on_a_tie():
    entries = [
        entry("bert", 100, 1_000),
        entry("modernbert", 100, 1_300),
        entry("bert", 200, 1_200),
        entry("modernbert", 200, 1_100),
        entry("bert", 400, 1_500),
        entry("modernbert", 400, 700),
    ]
    assert pilots.select_steps(entries) == 100  # 2,300 each for 100 and 200
    entries[2]["validation"]["in_scope_correct"] += 1
    assert pilots.select_steps(entries) == 200


def ac2_donor_run(protocol: CurveProtocol, in_scope_correct: int = 0) -> None:
    fake = FakeRuns()
    fake.correct = {(BERT_NAME, 5e-5): in_scope_correct}
    run_one(
        protocol.base("bert").with_seed(42), fake.train, fake.evaluate, quiet, keep_weights=True
    )


def test_pilot_lr_reuses_ac2_for_bert_5e5_and_trains_the_other_five(tmp_path):
    protocol = make_protocol(tmp_path)
    ac2_donor_run(protocol, in_scope_correct=2_902)
    fake = FakeRuns()
    fake.correct = {(MODERN_NAME, 2e-5): 2_950}
    out = tmp_path / "res" / "pilots" / "lr.json"
    body = pilots.run_pilot("lr", protocol, out, fake.train, fake.validation, quiet)
    assert len(fake.trained) == 5
    assert all("lr5e-05" not in t or "ModernBERT" in t for t in fake.trained)
    reused = [p for p in body["points"] if p["source"].startswith("reused")]
    assert [(p["model"], p["value"]) for p in reused] == [("bert", 5e-5)]
    assert body["selected"] == {"bert": 5e-5, "modernbert": 2e-5}
    assert json.loads(out.read_text()) == json.loads(json.dumps(body))
    # The AC2 point's numbers come from its archive's validation logits.
    ac2 = next(p for p in body["points"] if p["source"].startswith("reused"))
    assert ac2["validation"]["in_scope_correct"] == 2_902
    assert ac2["training"]["global_step"] == 2385


def test_pilot_output_holds_no_test_numbers_at_all(tmp_path):
    protocol = make_protocol(tmp_path)
    ac2_donor_run(protocol)
    fake = FakeRuns()
    body = pilots.run_pilot(
        "lr", protocol, tmp_path / "lr.json", fake.train, fake.validation, quiet
    )
    assert not {k for k in keys_anywhere(body) if "test" in k}
    assert body["split"] == "validation"


def test_pilot_never_calls_the_full_evaluation(tmp_path, monkeypatch):
    import tinyrouter.evaluate as evaluate_module

    def forbidden(*_):
        raise AssertionError("pilots must not run evaluate(), which scores test")

    monkeypatch.setattr(evaluate_module, "evaluate", forbidden)
    monkeypatch.setattr(evaluate_module, "score", forbidden)
    protocol = make_protocol(tmp_path)
    fake = FakeRuns()
    pilots.run_pilot("lr", protocol, tmp_path / "lr.json", fake.train, fake.validation, quiet)


def test_an_ac2_run_made_with_other_settings_is_not_reused(tmp_path):
    ac2_donor_run(make_protocol(tmp_path, warmup_ratio=0.0))
    protocol = make_protocol(tmp_path)  # the config file now says warmup 0.1
    fake = FakeRuns()
    body = pilots.run_pilot(
        "lr", protocol, tmp_path / "lr.json", fake.train, fake.validation, quiet
    )
    assert len(fake.trained) == 6
    assert {p["source"] for p in body["points"]} == {"trained"}


def test_pilot_weights_are_deleted_and_ac2_weights_are_left_alone(tmp_path):
    protocol = make_protocol(tmp_path)
    ac2_donor_run(protocol)
    fake = FakeRuns()
    pilots.run_pilot("lr", protocol, tmp_path / "lr.json", fake.train, fake.validation, quiet)
    ckpt = tmp_path / "ckpt"
    assert not list((ckpt / "pilots").rglob("model.safetensors"))
    assert (ckpt / "bert-base-uncased-full-seed42" / "final" / "model.safetensors").exists()


def test_rerunning_a_finished_pilot_trains_nothing(tmp_path):
    protocol = make_protocol(tmp_path)
    out = tmp_path / "lr.json"
    pilots.run_pilot("lr", protocol, out, FakeRuns().train, FakeRuns().validation, quiet)
    again = FakeRuns()
    pilots.run_pilot("lr", protocol, out, again.train, again.validation, quiet)
    assert again.trained == []


def test_pilot_steps_needs_the_learning_rates_first(tmp_path):
    with pytest.raises(ProtocolError, match="pilot-lr"):
        pilots.run_pilot("steps", make_protocol(tmp_path), tmp_path / "s.json")


def test_pilot_steps_trains_k5_for_each_s_min_and_encoder(tmp_path):
    protocol = make_protocol(tmp_path, bert_lr=5e-5, modern_lr=2e-5)
    fake = FakeRuns()
    fake.correct = {(BERT_NAME, 200): 700, (MODERN_NAME, 200): 800, (MODERN_NAME, 400): 1_600}
    body = pilots.run_pilot(
        "steps", protocol, tmp_path / "s.json", fake.train, fake.validation, quiet
    )
    assert len(fake.trained) == 6
    assert body["selected"] == 400
    steps = {(p["model"], p["value"]): p["training"]["step_plan"] for p in body["points"]}
    assert steps[("bert", 100)]["decided_by"] == "epochs"  # 120 epoch steps at k=5
    assert steps[("bert", 400)]["decided_by"] == "min_train_steps"
    assert {p["config"]["k_shot"] for p in body["points"]} == {5}


def test_a_crash_after_training_keeps_the_weights_for_the_rerun(tmp_path):
    protocol = make_protocol(tmp_path)
    fake = FakeRuns()

    def broken(config, model_dir) -> SplitLogits:
        raise RuntimeError("prediction crashed")

    with pytest.raises(RuntimeError, match="crashed"):
        pilots.run_pilot("lr", protocol, tmp_path / "lr.json", fake.train, broken, quiet)
    again = FakeRuns()
    pilots.run_pilot("lr", protocol, tmp_path / "lr.json", again.train, again.validation, quiet)
    assert len(again.trained) == len(pilots.lr_points(protocol)) - 1
