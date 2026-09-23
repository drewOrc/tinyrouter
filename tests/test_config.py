from pathlib import Path

import pytest

from tinyrouter.config import load_config

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))


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
    assert load_config(path).run_name == "bert-x-k16-seed43"
    assert load_config(path).with_seed(44).run_name == "bert-x-k16-seed44"
