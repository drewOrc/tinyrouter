from dataclasses import asdict, replace
from pathlib import Path

import pytest

from tinyrouter.config import load_config

# curve.yaml is the curve protocol (tests/test_protocol.py), not a run config.
CONFIGS = sorted(
    p for p in (Path(__file__).parent.parent / "configs").glob("*.yaml") if p.name != "curve.yaml"
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_committed_configs_load_with_pinned_model_revision(path):
    config = load_config(path)
    assert len(config.model_revision) == 40


def test_unknown_config_key_is_an_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model_name: x\nmodel_revision: y\nlearning_rat: 1e-5\n")
    with pytest.raises(ValueError, match="learning_rat"):
        load_config(path)


def test_warmup_ratio_of_one_or_more_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model_name: x\nmodel_revision: y\nwarmup_ratio: 1.0\n")
    with pytest.raises(ValueError, match="warmup_ratio"):
        load_config(path)


def test_run_name_encodes_model_size_and_seed(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model_name: org/bert-x\nmodel_revision: r\nper_intent: 16\nseed: 43\n")
    assert load_config(path).run_name == "bert-x-cap16-seed43"
    assert load_config(path).with_seed(44).run_name == "bert-x-cap16-seed44"


def test_run_name_encodes_k_shot_and_the_oos_override():
    from tinyrouter.config import RunConfig

    config = RunConfig(model_name="answerdotai/ModernBERT-base", model_revision="r", k_shot=100)
    assert config.run_name == "ModernBERT-base-k100-seed42"
    assert replace(config, oos_train=0).run_name == "ModernBERT-base-k100-oos0-seed42"


def test_a_record_written_before_the_step3_fields_matches_their_defaults():
    from tinyrouter.config import ADDED_FIELD_DEFAULTS, RunConfig

    config = RunConfig(model_name="m", model_revision="r")
    old = {k: v for k, v in asdict(config).items() if k not in ADDED_FIELD_DEFAULTS}
    assert config.matches(old)
    assert not replace(config, min_train_steps=100).matches(old)
    assert not replace(config, k_shot=5).matches(old)
