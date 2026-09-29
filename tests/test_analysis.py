import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from tinyrouter.analysis import (
    AGGREGATIONS,
    Routed,
    analyze_run,
    fallback,
    fit_run_temperature,
    oos_detection,
    route,
    router_metrics,
    select_aggregation,
    select_signal,
    signal_quality,
)
from tinyrouter.calibrate import LeakageError, SplitLogits
from tinyrouter.labels import AGENTS, load_label_space
from tinyrouter.selective import ThresholdChoice, select_threshold

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
    choices = {s: select_threshold(val.scored(s), 0.8) for s in val.signals}
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


@pytest.mark.parametrize("kind", ["encoder", "tfidf"])
def test_the_temperature_is_fitted_on_validation_only(kind):
    with pytest.raises(LeakageError, match="test"):
        fit_run_temperature(synthetic_splits()["test"], kind)


def choices_of(scalars, curves):
    """Everything chosen on validation: T, aggregation, every tau, curve taus, (b) taus."""
    fallback = {
        a: {
            t: ({s: v["tau"] for s, v in f["by_signal"].items()}, f["selected_signal"])
            for t, f in scalars[a]["fallback"].items()
        }
        for a in AGGREGATIONS
    }
    operating = {a: {s: c["operating"]["tau"] for s, c in curves[a].items()} for a in AGGREGATIONS}
    sensitivity = {
        t: (d["sensitivity_reweighted_validation"]["tau"], d["signal"], d["tau"])
        for t, d in scalars["diagnostics"].items()
    }
    return (
        scalars["calibration"]["temperature"],
        scalars["selected_aggregation"],
        fallback,
        operating,
        sensitivity,
    )


def test_test_labels_do_not_move_any_choice():
    splits = synthetic_splits()
    pred, cost = fake_llm(500)
    before = analyze_run(splits["validation"], splits["test"], "encoder", SPACE, pred, cost)
    shuffled = np.random.default_rng(9).permutation(splits["test"].labels)
    test = SplitLogits("test", splits["test"].logits, shuffled)
    after = analyze_run(splits["validation"], test, "encoder", SPACE, pred, cost)
    assert choices_of(*before) == choices_of(*after)
    assert before[0]["final"]["small_only"] != after[0]["final"]["small_only"]


def test_test_scores_do_not_move_any_choice():
    """Replacing the test logits with noise must leave every validation choice alone."""
    splits = synthetic_splits()
    pred, cost = fake_llm(500)
    before = analyze_run(splits["validation"], splits["test"], "encoder", SPACE, pred, cost)
    noise = np.random.default_rng(11).normal(size=splits["test"].logits.shape).astype(np.float32)
    test = SplitLogits("test", noise, splits["test"].labels)
    after = analyze_run(splits["validation"], test, "encoder", SPACE, pred, cost)
    assert choices_of(*before) == choices_of(*after)
    assert before[1]["argmax"]["msp"]["risk_coverage"] != after[1]["argmax"]["msp"]["risk_coverage"]


def oos_rows_unsure(n=200, seed=3):
    """A test split where every gold-oos row gets low confidence and in-scope rows high."""
    rng = np.random.default_rng(seed)
    gold = np.where(rng.random(n) < 0.25, OOS, F)
    confidence = np.where(gold == OOS, rng.uniform(0.1, 0.6, n), rng.uniform(0.4, 1.0, n))
    oos_score = np.where(gold == OOS, rng.uniform(0.3, 1.0, n), rng.uniform(0.0, 0.5, n))
    return routed("test", np.full(n, F), gold, {"msp": confidence}), confidence, oos_score


def test_low_confidence_on_oos_rows_gives_high_oos_auroc_matching_sklearn():
    r, confidence, _ = oos_rows_unsure()
    quality = signal_quality(r, "msp")
    gold_oos = r.gold == OOS
    assert quality["oos_auroc"] > 0.5
    assert quality["oos_auroc"] == pytest.approx(roc_auc_score(gold_oos, -confidence))
    assert quality["oos_auprc"] == pytest.approx(average_precision_score(gold_oos, -confidence))


def test_oos_detection_ranks_by_the_oos_score_itself():
    base, _, oos_score = oos_rows_unsure()
    r = Routed(base.split, base.pred, base.gold, base.signals, oos_score)
    got = oos_detection(r)
    gold_oos = r.gold == OOS
    assert got["auroc"] > 0.5
    assert got["auroc"] == pytest.approx(roc_auc_score(gold_oos, oos_score))
    assert got["auprc"] == pytest.approx(average_precision_score(gold_oos, oos_score))


def test_oos_score_is_one_minus_the_largest_in_scope_probability_at_t():
    split = one_row_logits()
    t = 2.0
    p = np.exp(split.logits[0] / t) / np.exp(split.logits[0] / t).sum()
    in_scope = np.delete(p, SPACE.oos_intent_id).max()
    got = route(split, t, "argmax", SPACE, "encoder").oos_score
    assert got == pytest.approx([1 - in_scope], rel=1e-5)
    assert route(split, t, "argmax", SPACE, "majority").oos_score is None


def test_signal_choice_by_hand_coverage_first_then_aurc():
    val = routed(
        "validation",
        [F, F, F, T],
        [F, F, F, F],
        {"msp": np.array([4.0, 3, 2, 1]), "entropy": np.array([1.0, 2, 3, 4])},
    )

    def choice(coverage):
        return ThresholdChoice(0.0, 0.05, coverage, 0.0, 0.0, 4)

    # entropy has the worse AURC (its error ranks first) but more coverage: it wins.
    assert select_signal(val, {"msp": choice(0.5), "entropy": choice(0.75)}) == "entropy"
    # Equal coverage: the lower validation AURC (msp) wins, whatever the dict order.
    assert select_signal(val, {"entropy": choice(0.5), "msp": choice(0.5)}) == "msp"


def test_entropy_signal_is_lower_for_a_uniform_row_than_a_one_hot_row():
    z = np.full((2, SPACE.num_intents), 0.0)
    z[1, 0] = 50.0
    split = SplitLogits("validation", z.astype(np.float32), np.array([0, 0]))
    entropy = route(split, 1.0, "argmax", SPACE, "encoder").signals["entropy"]
    assert entropy[0] < entropy[1]
    assert entropy[0] == pytest.approx(-np.log(SPACE.num_intents))


def test_summed_aggregation_uses_the_temperature_scaled_probabilities():
    """At T = 1 oos outweighs two finance intents; at T = 10 the pair wins."""
    names = SPACE.intent_names
    finance = [i for i, n in enumerate(names) if SPACE.intent_to_agent[n] == "finance_agent"]
    z = np.full((1, SPACE.num_intents), -1000.0)
    z[0, SPACE.oos_intent_id] = 3.0
    z[0, finance[:2]] = 2.0
    split = SplitLogits("validation", z.astype(np.float32), np.array([SPACE.oos_intent_id]))
    assert route(split, 1.0, "summed", SPACE, "encoder").pred.tolist() == [OOS]
    assert route(split, 10.0, "summed", SPACE, "encoder").pred.tolist() == [F]


def test_a_test_row_exactly_at_tau_is_kept_by_the_small_model(monkeypatch):
    import tinyrouter.analysis as analysis

    monkeypatch.setattr(analysis, "TARGET_RISKS", (0.6,))
    # Validation: two error-free rows (upper bound 0.575) then an error (3 rows: 0.746),
    # so at target 0.6 tau is the second score, 0.8.
    val = routed("validation", [F, F, T], [F, F, F], {"msp": np.array([0.9, 0.8, 0.1])})
    # Test row 0 sits exactly on tau and must be kept; row 1 is just below it.
    test = routed("test", [F, T, OOS], [F, F, OOS], {"msp": np.array([0.8, 0.79, 0.95])})
    llm = np.array([F, F, OOS])
    got = fallback(val, test, llm, np.zeros(3))["0.60"]["by_signal"]["msp"]
    assert got["tau"] == 0.8
    assert got["test"]["coverage"] == pytest.approx(2 / 3)
    assert got["test"]["llm_call_rate"] == pytest.approx(1 / 3)
    assert got["test"]["accuracy_8"] == 1.0
