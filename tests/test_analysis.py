import numpy as np
import pytest

from tinyrouter.analysis import (
    AGGREGATIONS,
    Routed,
    analyze_run,
    route,
    router_metrics,
    select_aggregation,
    select_signal,
)
from tinyrouter.calibrate import LeakageError, SplitLogits
from tinyrouter.labels import AGENTS, load_label_space
from tinyrouter.selective import select_threshold

F, T, OOS = AGENTS.index("finance_agent"), AGENTS.index("travel_agent"), AGENTS.index("oos")
SPACE = load_label_space()


def routed(split, pred, gold, signals=None):
    return Routed(split, np.array(pred), np.array(gold), signals or {})


def hand_case():
    small = routed("test", [F, T, F, OOS, T], [F, F, OOS, OOS, T])
    llm_pred = np.array([F, F, OOS, F, F])
    cost = np.array([1, 2, 3, 4, 5]) * 1e-3
    return small, llm_pred, cost


def test_hybrid_router_metrics_by_hand():
    small, llm_pred, cost = hand_case()
    m = router_metrics(small, llm_pred, cost, np.array([0, 1, 0, 1, 0], bool))
    assert m["coverage"] == pytest.approx(3 / 5)
    assert m["selective_risk"] == pytest.approx(1 / 3)
    assert m["accuracy_8"] == pytest.approx(3 / 5)
    assert m["oos_recall"] == 0.0 and m["oos_misroute_rate"] == 1.0
    # Gold oos rows 2 and 3: row 2 kept and routed to finance; row 3 predicted oos.
    assert m["high_conf_oos_misroute_rate"] == pytest.approx(1 / 2)
    # Small errors: rows 1, 2. Haiku fixes both; only row 1 was deferred.
    assert m["error_recovery_rate"] == pytest.approx(1 / 2)
    assert m["recoverable_caught"] == pytest.approx(1 / 2)
    assert m["llm_call_rate"] == pytest.approx(2 / 5)
    assert m["llm_cost_usd_per_1k"] == pytest.approx(1000 * (0.002 + 0.004) / 5)


def test_oracle_defers_exactly_the_small_models_errors():
    small, llm_pred, cost = hand_case()
    m = router_metrics(small, llm_pred, cost, small.error)
    assert m["accuracy_8"] == 1.0
    assert m["llm_call_rate"] == pytest.approx(2 / 5)
    assert m["high_conf_oos_misroute_rate"] == 0.0
    assert m["error_recovery_rate"] == 1.0 and m["recoverable_caught"] == 1.0


def test_small_only_misroute_equals_one_minus_oos_recall():
    small, llm_pred, cost = hand_case()
    m = router_metrics(small, llm_pred, cost, np.zeros(5, bool))
    assert m["oos_recall"] == pytest.approx(1 / 2)
    assert m["oos_misroute_rate"] == pytest.approx(1 / 2)
    assert m["high_conf_oos_misroute_rate"] == m["oos_misroute_rate"]
    assert m["llm_call_rate"] == 0.0 and m["error_recovery_rate"] == 0.0


def one_row_logits():
    """oos is the top intent; two finance intents together outweigh it."""
    names = SPACE.intent_names
    finance = [i for i, n in enumerate(names) if SPACE.intent_to_agent[n] == "finance_agent"]
    z = np.full((1, SPACE.num_intents), -20.0)
    z[0, SPACE.oos_intent_id] = 2.0
    z[0, finance[:2]] = 1.5
    return SplitLogits("validation", z.astype(np.float32), np.array([SPACE.oos_intent_id]))


def test_argmax_and_summed_aggregations_disagree_where_they_should():
    split = one_row_logits()
    assert route(split, 1.0, "argmax", SPACE, "encoder").pred.tolist() == [OOS]
    assert route(split, 1.0, "summed", SPACE, "encoder").pred.tolist() == [F]


def test_argmax_margin_is_the_score_margin_over_t():
    split = one_row_logits()
    margin = route(split, 2.0, "argmax", SPACE, "encoder").signals["margin"]
    assert margin == pytest.approx([(2.0 - 1.5) / 2.0])


def test_signals_follow_the_plan_by_kind():
    split = one_row_logits()
    assert set(route(split, 1.0, "argmax", SPACE, "encoder").signals) == {
        "msp",
        "entropy",
        "margin",
        "msp_t",
    }
    assert set(route(split, 1.0, "argmax", SPACE, "tfidf").signals) == {"margin", "msp_t"}
    assert route(split, 1.0, "summed", SPACE, "majority").signals == {}


def test_aggregation_is_chosen_on_validation_and_ties_go_to_argmax():
    better = {
        "argmax": routed("validation", [F, F], [F, T]),
        "summed": routed("validation", [F, T], [F, T]),
    }
    assert select_aggregation(better)[0] == "summed"
    tied = {"argmax": routed("validation", [F], [F]), "summed": routed("validation", [F], [F])}
    assert select_aggregation(tied)[0] == "argmax"


def test_aggregation_and_signal_choices_refuse_test():
    leaked = {a: routed("test", [F], [F]) for a in AGGREGATIONS}
    with pytest.raises(LeakageError, match="test"):
        select_aggregation(leaked)
    with pytest.raises(LeakageError, match="test"):
        select_signal(routed("test", [F], [F], {"msp": np.array([1.0])}), {})


def test_signal_choice_prefers_coverage_then_aurc():
    val = routed(
        "validation",
        [F, F, F, T],
        [F, F, F, F],
        {"msp": np.array([4.0, 3, 2, 1]), "entropy": np.array([1.0, 2, 3, 4])},
    )
    choices = {
        s: select_threshold("validation", val.signals[s], val.error, 0.8) for s in val.signals
    }
    # Both keep every row at this target; msp ranks the error last, so its AURC is lower.
    assert select_signal(val, choices) == "msp"


def synthetic_splits(seed=0, n_val=300, n_test=500):
    """Logits that are right about 70% of the time; about 20% of rows are oos."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, n in (("validation", n_val), ("test", n_test)):
        labels = rng.integers(0, SPACE.num_intents, size=n)
        labels[rng.random(n) < 0.2] = SPACE.oos_intent_id
        z = rng.normal(size=(n, SPACE.num_intents))
        right = rng.random(n) < 0.7
        z[np.arange(n)[right], labels[right]] += 4.0
        out[name] = SplitLogits(name, z.astype(np.float32), labels)
    return out


def fake_llm(n, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(AGENTS), size=n), np.full(n, 3.7e-4)


def test_analyze_run_reports_both_aggregations_and_a_final_router():
    splits = synthetic_splits()
    pred, cost = fake_llm(500)
    scalars, curves = analyze_run(
        splits["validation"], splits["test"], "encoder", SPACE, pred, cost
    )
    assert scalars["selected_aggregation"] in AGGREGATIONS
    for a in AGGREGATIONS:
        assert set(scalars[a]["signals"]) == {"msp", "entropy", "margin", "msp_t"}
        assert set(scalars[a]["fallback"]) == {"0.02", "0.05"}
        assert len(curves[a]["msp"]["risk_coverage"]) == 20
    assert scalars["final"]["small_only"] == scalars[scalars["selected_aggregation"]]["small_only"]


def test_the_oracle_calls_the_llm_exactly_on_the_small_models_errors():
    splits = synthetic_splits()
    pred, cost = fake_llm(500)
    scalars, _ = analyze_run(splits["validation"], splits["test"], "encoder", SPACE, pred, cost)
    for a in AGGREGATIONS:
        small, oracle = scalars[a]["small_only"], scalars[a]["oracle"]
        assert oracle["llm_call_rate"] == pytest.approx(1 - small["accuracy_8"])
        assert oracle["recoverable_caught"] == 1.0
        assert oracle["accuracy_8"] == pytest.approx(
            small["accuracy_8"] + oracle["recovered_errors"] / 500
        )


def test_majority_has_no_temperature_signals_or_hybrid():
    splits = synthetic_splits()
    const = {n: SplitLogits(n, np.zeros_like(s.logits), s.labels) for n, s in splits.items()}
    pred, cost = fake_llm(500)
    scalars, _ = analyze_run(const["validation"], const["test"], "majority", SPACE, pred, cost)
    assert scalars["calibration"]["temperature"] is None
    assert scalars["calibration"]["test_ece"] is None
    assert scalars["final"]["fallback"]["0.05"]["hybrid"] is None


def test_tfidf_reports_no_uncalibrated_ece():
    splits = synthetic_splits()
    pred, cost = fake_llm(500)
    scalars, _ = analyze_run(splits["validation"], splits["test"], "tfidf", SPACE, pred, cost)
    assert scalars["calibration"]["test_ece"]["uncalibrated"] is None
    assert scalars["calibration"]["test_ece"]["calibrated"]["ece_151"] >= 0


def test_analyze_run_refuses_test_logits_in_the_validation_slot():
    splits = synthetic_splits()
    pred, cost = fake_llm(500)
    with pytest.raises(LeakageError):
        analyze_run(splits["test"], splits["test"], "encoder", SPACE, pred, cost)


def choices_of(scalars):
    fallback = {
        a: {
            t: {s: v["tau"] for s, v in f["by_signal"].items()}
            for t, f in scalars[a]["fallback"].items()
        }
        for a in AGGREGATIONS
    }
    return scalars["calibration"]["temperature"], scalars["selected_aggregation"], fallback


def test_test_labels_do_not_move_any_choice():
    splits = synthetic_splits()
    pred, cost = fake_llm(500)
    before, _ = analyze_run(splits["validation"], splits["test"], "encoder", SPACE, pred, cost)
    shuffled = np.random.default_rng(9).permutation(splits["test"].labels)
    test = SplitLogits("test", splits["test"].logits, shuffled)
    after, _ = analyze_run(splits["validation"], test, "encoder", SPACE, pred, cost)
    assert choices_of(before) == choices_of(after)
    assert before["final"]["small_only"] != after["final"]["small_only"]
