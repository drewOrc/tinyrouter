import numpy as np
import pytest

from tinyrouter.analysis import Routed
from tinyrouter.diagnostics import (
    DECLARED_OOS_SHARE,
    gap_explained,
    kept_rates,
    prior_weights,
    sensitivity,
    threshold_diagnostics,
    transfer,
)
from tinyrouter.labels import AGENTS

F, T, OOS = AGENTS.index("finance_agent"), AGENTS.index("travel_agent"), AGENTS.index("oos")


def routed(split, pred, gold, msp):
    return Routed(split, np.array(pred), np.array(gold), {"msp": np.array(msp, dtype=float)})


def test_prior_weights_hit_the_share_and_average_one():
    gold = np.array([True, False, False, False])
    w = prior_weights(gold, 0.5)
    assert w.tolist() == pytest.approx([2.0, 2 / 3, 2 / 3, 2 / 3])
    assert w.mean() == pytest.approx(1.0)
    assert w[gold].sum() / w.sum() == pytest.approx(0.5)


def test_kept_rates_by_hand():
    # Kept at tau 0.5: rows 0-2. Row 1 in-scope error, row 2 kept OOS routed to finance.
    # Row 4 is OOS routed to finance too, but deferred, so it is not a high-confidence misroute.
    r = routed("test", [F, T, F, OOS, F], [F, F, OOS, OOS, OOS], [0.9, 0.8, 0.7, 0.2, 0.1])
    got = kept_rates(r, "msp", 0.5)
    assert got["coverage"] == pytest.approx(3 / 5)
    assert got["selective_risk"] == pytest.approx(2 / 3)
    assert got["in_scope_risk"] == pytest.approx(1 / 2)
    assert got["kept_oos_error_rate"] == 1.0
    assert got["kept_oos_share"] == pytest.approx(1 / 3)
    assert got["high_conf_oos_misroute_rate"] == pytest.approx(1 / 3)


def test_reweighting_moves_risk_by_the_oos_share_by_hand():
    r = routed("test", [F, T, F, OOS], [F, F, OOS, OOS], [0.9, 0.8, 0.7, 0.2])
    # Weights for a 25% OOS share: oos rows 0.5, in-scope rows 1.5. Kept weight 3.5,
    # errors 1.5 (row 1) + 0.5 (row 2).
    got = kept_rates(r, "msp", 0.5, prior_weights(r.gold == OOS, 0.25))
    assert got["selective_risk"] == pytest.approx(2.0 / 3.5)


def test_gap_explained_by_hand_and_undefined_cases():
    assert gap_explained(0.015, 0.0734, 0.0229) == pytest.approx((0.0734 - 0.0229) / 0.0584)
    assert gap_explained(0.05, 0.04, 0.03) is None
    assert gap_explained(None, 0.04, 0.03) is None


def small_pair():
    rng = np.random.default_rng(0)
    n_val, n_test = 400, 400
    val_gold = np.where(rng.random(n_val) < 0.05, OOS, F)
    test_gold = np.where(rng.random(n_test) < 0.3, OOS, F)
    val = routed("validation", np.full(n_val, F), val_gold, np.where(val_gold == OOS, 0.6, 0.9))
    test = routed("test", np.full(n_test, F), test_gold, np.where(test_gold == OOS, 0.6, 0.9))
    return val, test


def test_transfer_explains_a_pure_prior_shift_completely():
    val, test = small_pair()
    got = transfer(val, test, "msp", 0.5)
    assert got["validation_oos_share"] < got["test_oos_share"]
    rw = got["test_reweighted_to_validation_oos_share"]["selective_risk"]
    assert rw == pytest.approx(got["validation"]["selective_risk"])
    assert got["share_of_gap_explained_by_oos_share"] == pytest.approx(1.0)


def test_sensitivity_chooses_tau_on_reweighted_validation_only():
    val, test = small_pair()
    got = sensitivity(val, test, "msp", 0.05)
    # With OOS reweighted to 18.2%, keeping the OOS rows breaks 5%, so only 0.9 is kept.
    assert got["tau"] == 0.9
    assert got["validation_weighted_coverage"] == pytest.approx(1 - DECLARED_OOS_SHARE)
    assert got["test"]["kept_oos_share"] == 0.0


def test_diagnostics_follow_the_final_hybrid_and_skip_a_run_without_signals():
    val, test = small_pair()
    block = {
        "fallback": {
            "0.05": {"selected_signal": "msp", "hybrid": {"tau": 0.5}},
            "0.02": {"selected_signal": None, "hybrid": None},
        }
    }
    got = threshold_diagnostics(val, test, block)
    assert got["0.02"] is None
    assert got["0.05"]["tau"] == 0.5 and got["0.05"]["signal"] == "msp"
    assert got["0.05"]["validation"]["coverage"] == 1.0
