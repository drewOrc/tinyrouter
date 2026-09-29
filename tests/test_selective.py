import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from tinyrouter.calibrate import LeakageError
from tinyrouter.metrics import wilson_interval
from tinyrouter.selective import (
    ONE_SIDED_95_Z,
    Scored,
    aurc,
    auroc,
    average_precision,
    coverage_thresholds,
    deferred_below,
    detection_counts,
    high_confidence_oos_misroute,
    risk_at_coverages,
    risk_coverage,
    select_threshold,
    wilson_upper,
)


def test_aurc_by_hand_for_distinct_scores():
    # Keep 1: risk 0; keep 2: 1/2; keep 3: 1/3.
    assert aurc(np.array([0.9, 0.8, 0.7]), np.array([0, 1, 0])) == pytest.approx(
        (0 + 1 / 2 + 1 / 3) / 3
    )


def test_aurc_ties_use_the_expected_order_so_a_constant_score_gives_the_error_rate():
    assert aurc(np.ones(3), np.array([0, 1, 0])) == pytest.approx(1 / 3)


def test_risk_coverage_spreads_a_tie_groups_errors_evenly():
    coverage, risk = risk_coverage(np.array([1.0, 1.0, 0.0]), np.array([1, 0, 0]))
    assert coverage.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert risk.tolist() == pytest.approx([0.5, 0.5, 1 / 3])


def test_risk_at_coverages_keeps_the_ceiling_of_c_times_n_rows():
    confidence = np.array([0.9, 0.8, 0.7, 0.6])
    error = np.array([0, 1, 0, 0])
    assert risk_at_coverages(confidence, error, (0.25, 0.5, 0.6, 1.0)) == pytest.approx(
        [0.0, 0.5, 1 / 3, 0.25]
    )


def test_auroc_and_average_precision_by_hand():
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    positive = np.array([True, False, True, False])
    # Pairs (pos, neg): 0.9>0.8, 0.9>0.1, 0.7<0.8, 0.7>0.1.
    assert auroc(scores, positive) == pytest.approx(3 / 4)
    # Recall steps 1/2 at precision 1, then 1/2 more at precision 2/3.
    assert average_precision(scores, positive) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_auroc_of_a_constant_score_is_one_half():
    assert auroc(np.zeros(4), np.array([True, False, False, True])) == 0.5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_auroc_and_average_precision_match_sklearn_with_ties(seed):
    rng = np.random.default_rng(seed)
    scores = rng.integers(0, 6, size=300).astype(float)
    positive = rng.random(300) < 0.3
    assert auroc(scores, positive) == pytest.approx(roc_auc_score(positive, scores))
    assert average_precision(scores, positive) == pytest.approx(
        average_precision_score(positive, scores)
    )


def test_detection_needs_both_classes():
    with pytest.raises(ValueError, match="positive and one negative"):
        auroc(np.array([0.1, 0.2]), np.array([True, True]))


@pytest.mark.parametrize(("k", "n"), [(0, 1), (0, 50), (3, 50), (50, 50), (17, 3100)])
def test_wilson_upper_equals_the_metrics_interval(k, n):
    expected = wilson_interval(k, n, z=ONE_SIDED_95_Z)[1]
    assert wilson_upper(np.array([k]), np.array([n]), ONE_SIDED_95_Z)[0] == pytest.approx(expected)


def brute_force_threshold(confidence, error, target):
    """Try every distinct score as a threshold; keep the feasible one with most rows."""
    best = None
    for tau in np.unique(confidence):
        kept = confidence >= tau
        upper = wilson_interval(int(error[kept].sum()), int(kept.sum()), z=ONE_SIDED_95_Z)[1]
        if upper <= target and (best is None or kept.sum() > best[1]):
            best = (tau, kept.sum())
    return best


@pytest.mark.parametrize("target", [0.02, 0.05, 0.1, 0.3])
def test_select_threshold_matches_a_brute_force_search(target):
    rng = np.random.default_rng(7)
    confidence = np.round(rng.random(400), 2)
    error = rng.random(400) < 0.3 * (1 - confidence)
    choice = select_threshold(Scored("validation", confidence, error), target)
    expected = brute_force_threshold(confidence, error, target)
    if expected is None:
        assert not choice.feasible and choice.coverage == 0.0
    else:
        assert choice.tau == expected[0]
        assert choice.coverage == pytest.approx(expected[1] / 400)


def test_select_threshold_accepts_everything_when_the_target_is_loose():
    choice = select_threshold(
        Scored("validation", np.array([3.0, 2.0, 1.0]), np.array([0, 1, 1])), 1.0
    )
    assert choice.tau == 1.0 and choice.coverage == 1.0 and choice.risk == pytest.approx(2 / 3)


def test_select_threshold_reports_an_unreachable_target():
    choice = select_threshold(Scored("validation", np.array([3.0, 2.0]), np.array([0, 0])), 1e-4)
    assert choice.tau is None and not choice.feasible and choice.coverage == 0.0


def test_select_threshold_never_splits_tied_scores():
    # Upper bounds: 1 row, 0 errors 0.73; 2 rows, 1 error 0.88; 3 rows, 1 error 0.746.
    # Splitting the tie would keep the error-free first row alone at target 0.74.
    choice = select_threshold(
        Scored("validation", np.array([1.0, 1.0, 0.0]), np.array([0, 1, 0])), 0.74
    )
    assert choice.tau is None


@pytest.mark.parametrize("split", ["test", "train"])
def test_threshold_and_curve_choices_refuse_any_split_but_validation(split):
    with pytest.raises(LeakageError, match=split):
        select_threshold(Scored(split, np.array([1.0, 0.0]), np.array([0, 1])), 0.05)
    with pytest.raises(LeakageError, match=split):
        coverage_thresholds(Scored(split, np.array([1.0, 0.0]), np.array([0, 1])))


def test_coverage_thresholds_are_the_validation_scores_at_each_coverage():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert coverage_thresholds(
        Scored("validation", scores, np.zeros(4, bool)), (0.25, 0.5, 1.0)
    ) == [0.9, 0.7, 0.1]


def test_detection_counts_by_hand_and_zero_precision_when_nothing_is_predicted():
    counts = detection_counts(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool))
    assert (counts["recall"], counts["precision"], counts["f1"]) == (0.5, 0.5, 0.5)
    none = detection_counts(np.zeros(3, bool), np.array([1, 0, 0], bool))
    assert (none["recall"], none["precision"], none["f1"]) == (0.0, 0.0, 0.0)


def test_high_confidence_misroute_counts_only_kept_oos_sent_to_an_agent():
    gold_oos = np.array([1, 1, 1, 1, 0], bool)
    pred_oos = np.array([1, 0, 0, 0, 0], bool)
    accepted = np.array([1, 1, 0, 1, 1], bool)
    # Rows 1 and 3: gold oos, routed to an agent, kept. Row 2 was deferred.
    assert high_confidence_oos_misroute(pred_oos, gold_oos, accepted) == pytest.approx(2 / 4)


def test_the_bound_is_one_sided_95_percent():
    from scipy.stats import norm

    assert ONE_SIDED_95_Z == pytest.approx(norm.ppf(0.95))
    # 60 rows, no error: upper 0.043 one-sided, 0.060 two-sided; the target sits between.
    choice = select_threshold(Scored("validation", np.arange(60.0), np.zeros(60, bool)), 0.05)
    assert choice.coverage == 1.0


def test_a_row_exactly_at_tau_is_kept_and_no_tau_defers_everything():
    confidence = np.array([0.9, 0.5, 0.4999])
    assert deferred_below(confidence, 0.5).tolist() == [False, False, True]
    assert deferred_below(confidence, None).tolist() == [True, True, True]


def test_scored_refuses_arrays_that_do_not_line_up():
    with pytest.raises(ValueError, match="equal"):
        Scored("validation", np.zeros(3), np.zeros(2, bool))
    with pytest.raises(ValueError, match="equal"):
        Scored("validation", np.zeros(3), np.zeros(3, bool), np.ones(4))


def test_weighted_threshold_counts_weighted_rows_by_hand():
    # Rows by confidence: ok, ok, error. Error weight 3, others 0.5 each (total 4).
    scored = Scored(
        "validation",
        np.array([3.0, 2.0, 1.0]),
        np.array([0, 0, 1], bool),
        np.array([0.5, 0.5, 3.0]),
    )
    loose = select_threshold(scored, 1.0)
    assert loose.coverage == 1.0 and loose.risk == pytest.approx(3 / 4)
    # Kish: n_eff = 4^2 / (0.25 + 0.25 + 9) = 16 / 9.5, with 0.75 * n_eff errors.
    n_eff = 16 / 9.5
    expected = wilson_upper(np.array([0.75 * n_eff]), np.array([n_eff]), ONE_SIDED_95_Z)[0]
    assert loose.risk_upper == pytest.approx(expected)
    # Two kept rows of weight 0.5, no error: n_eff = 2, bound z^2 / (2 + z^2) = 0.575.
    upper = ONE_SIDED_95_Z**2 / (2 + ONE_SIDED_95_Z**2)
    mid = select_threshold(scored, upper + 1e-9)
    assert mid.tau == 2.0 and mid.coverage == pytest.approx(1 / 4)


@pytest.mark.parametrize("target", [0.05, 0.1, 0.3])
def test_unit_weights_give_the_unweighted_choice(target):
    rng = np.random.default_rng(5)
    confidence = np.round(rng.random(300), 2)
    error = rng.random(300) < 0.3 * (1 - confidence)
    plain = select_threshold(Scored("validation", confidence, error), target)
    unit = select_threshold(Scored("validation", confidence, error, np.ones(300)), target)
    assert unit.tau == plain.tau and unit.coverage == pytest.approx(plain.coverage)
    assert unit.risk_upper == pytest.approx(plain.risk_upper)


def test_unequal_weights_widen_the_bound_beyond_the_nominal_count():
    rng = np.random.default_rng(6)
    error = rng.random(200) < 0.1
    weight = np.where(rng.random(200) < 0.1, 5.0, 0.5)
    weight = weight / weight.mean()
    got = select_threshold(Scored("validation", np.ones(200), error, weight), 1.0)
    nominal = wilson_upper(
        np.array([(weight * error).sum()]), np.array([weight.sum()]), ONE_SIDED_95_Z
    )
    assert got.risk_upper > nominal[0]
