from dataclasses import asdict, replace
from pathlib import Path

import pytest

from tinyrouter.config import load_config
from tinyrouter.protocol import (
    LR_GRID,
    S_MIN_GRID,
    SEEDS,
    CurveProtocol,
    ProtocolError,
    ac2_donor,
    load_protocol,
)
from tinyrouter.sampling import CURVE_KS

ROOT = Path(__file__).parent.parent


def protocol_file(tmp_path, s_min="null", bert="null", modern="null", extra="") -> Path:
    path = tmp_path / "curve.yaml"
    path.write_text(
        f"min_train_steps: {s_min}\nmodels:\n"
        f"  bert:\n    config: {ROOT}/configs/bert-base.yaml\n    learning_rate: {bert}\n"
        f"  modernbert:\n    config: {ROOT}/configs/modernbert-base.yaml\n"
        f"    learning_rate: {modern}\n{extra}"
    )
    return path


def decided(tmp_path) -> CurveProtocol:
    return load_protocol(protocol_file(tmp_path, s_min=400, bert="5.0e-5", modern="2.0e-5"))


def test_the_committed_protocol_loads_with_both_pilots_still_open():
    protocol = load_protocol(ROOT / "configs" / "curve.yaml")
    assert protocol.min_train_steps is None
    assert protocol.learning_rates == {"bert": None, "modernbert": None}
    with pytest.raises(ProtocolError, match="make pilot-lr"):
        protocol.curve_configs("bert")


def test_curve_refuses_to_start_until_s_min_is_chosen(tmp_path):
    protocol = load_protocol(protocol_file(tmp_path, bert="5.0e-5", modern="2.0e-5"))
    with pytest.raises(ProtocolError, match="make pilot-steps"):
        protocol.curve_configs("modernbert")


@pytest.mark.parametrize(
    "fields",
    [
        {"bert": "3.0e-5"},
        {"bert": "5e-5"},  # YAML reads this as a string
        {"modern": "true"},
        {"s_min": 150},
        {"s_min": "100.0"},
        {"extra": "  extra_model:\n    config: x\n    learning_rate: null\n"},
    ],
)
def test_values_off_the_protocol_grid_are_refused(tmp_path, fields):
    with pytest.raises(ProtocolError):
        load_protocol(protocol_file(tmp_path, **fields))


def test_curve_configs_cover_every_k_and_seed_with_one_lr_and_one_s_min(tmp_path):
    protocol = decided(tmp_path)
    configs = protocol.curve_configs("modernbert")
    assert [(c.k_shot, c.seed) for c in configs] == [(k, s) for k in CURVE_KS for s in SEEDS]
    assert {c.learning_rate for c in configs} == {2e-5}
    assert {c.min_train_steps for c in configs} == {400}
    assert {c.oos_train for c in configs} == {None}


def test_a_curve_run_differs_from_its_base_config_only_in_lr_s_min_k_and_seed(tmp_path):
    base = load_config(ROOT / "configs" / "bert-base.yaml")
    point = decided(tmp_path).curve_config("bert", 25, 43)
    changed = {k for k, v in asdict(point).items() if asdict(base)[k] != v}
    assert changed == {"min_train_steps", "k_shot", "seed"}  # bert's lr 5e-5 equals the file's


def test_the_two_encoders_differ_only_in_model_and_learning_rate():
    bert = asdict(load_config(ROOT / "configs" / "bert-base.yaml"))
    modern = asdict(load_config(ROOT / "configs" / "modernbert-base.yaml"))
    assert {k for k in bert if bert[k] != modern[k]} == {
        "model_name",
        "model_revision",
        "learning_rate",
    }


def test_the_ablation_is_modernbert_at_k100_with_no_oos_rows(tmp_path):
    configs = decided(tmp_path).ablation_configs()
    assert [c.seed for c in configs] == list(SEEDS)
    assert all(c.k_shot == 100 and c.oos_train == 0 for c in configs)
    assert {c.model_name for c in configs} == {"answerdotai/ModernBERT-base"}
    assert configs[0].run_name == "ModernBERT-base-k100-oos0-seed42"


def test_only_bert_at_k100_with_the_table_oos_count_has_an_ac2_donor(tmp_path):
    protocol = decided(tmp_path)
    donor = ac2_donor(protocol, protocol.curve_config("bert", 100, 44))
    assert donor == protocol.base("bert").with_seed(44)
    assert donor.run_name == "bert-base-uncased-full-seed44"
    assert ac2_donor(protocol, protocol.curve_config("bert", 50, 44)) is None
    assert ac2_donor(protocol, protocol.curve_config("modernbert", 100, 44)) is None
    ablation_like = replace(protocol.curve_config("bert", 100, 44), oos_train=0)
    assert ac2_donor(protocol, ablation_like) is None


def test_grids_are_the_ones_the_plan_fixes():
    assert LR_GRID == (1e-5, 2e-5, 5e-5)
    assert S_MIN_GRID == (100, 200, 400)
