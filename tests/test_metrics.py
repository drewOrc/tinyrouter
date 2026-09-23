import numpy as np
import pytest
from sklearn.metrics import accuracy_score, log_loss

from tinyrouter.labels import AGENTS, OOS, build_label_space
from tinyrouter.metrics import (
    accuracy,
    expected_calibration_error,
    in_scope_accuracy,
    negative_log_likelihood,
    oos_recall,
    routing_metrics,
    softmax,
    wilson_interval,
)

OOS_ID = 9


def test_accuracy_counts_exact_matches_and_agrees_with_sklearn():
    pred, gold = np.array([1, 2, 3, 4]), np.array([1, 2, 0, 4])
    assert accuracy(pred, gold) == 0.75
    assert accuracy(pred, gold) == accuracy_score(gold, pred)


def test_accuracy_rejects_mismatched_or_empty_inputs():
    with pytest.raises(ValueError):
        accuracy(np.array([1, 2]), np.array([1]))
    with pytest.raises(ValueError):
        accuracy(np.array([]), np.array([]))


def test_in_scope_accuracy_ignores_gold_oos_and_counts_oos_predictions_as_wrong():
    gold = np.array([1, 2, 3, OOS_ID, OOS_ID])
    pred = np.array([1, OOS_ID, 3, 5, OOS_ID])
    # in-scope rows: [1,2,3] vs [1,oos,3] -> 2/3
    assert in_scope_accuracy(pred, gold, OOS_ID) == pytest.approx(2 / 3)


def test_oos_recall_is_share_of_gold_oos_predicted_oos():
    gold = np.array([OOS_ID, OOS_ID, OOS_ID, OOS_ID, 1])
    pred = np.array([OOS_ID, 2, OOS_ID, 3, OOS_ID])
    assert oos_recall(pred, gold, OOS_ID) == 0.5


def test_oos_recall_raises_without_oos_examples():
    with pytest.raises(ValueError, match="no oos"):
        oos_recall(np.array([1]), np.array([1]), OOS_ID)


def test_ece_matches_hand_computed_value():
    # Two bins used with n_bins=10:
    #   conf 0.9 (bin 8): rows 0,1 -> accuracy 1/2, mean conf 0.9, gap 0.4
    #   conf 0.6 (bin 5): rows 2,3 -> accuracy 2/2, mean conf 0.6, gap 0.4
    # ECE = 0.5*0.4 + 0.5*0.4 = 0.4
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.4, 0.6], [0.4, 0.6]])
    gold = np.array([0, 1, 1, 1])
    assert expected_calibration_error(probs, gold, n_bins=10) == pytest.approx(0.4)


def test_ece_is_zero_when_confidence_equals_accuracy():
    probs = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert expected_calibration_error(probs, np.array([0, 0])) == 0.0


def test_ece_puts_confidence_on_a_bin_edge_into_the_lower_bin():
    # n_bins=2. Row 0: conf 0.5, correct. Row 1: conf 0.8, wrong.
    # 0.5 in (0, 0.5]: gaps |1-0.5| and |0-0.8| -> 0.5*0.5 + 0.5*0.8 = 0.65.
    # Had 0.5 gone to the upper bin, both rows share it: |0.5-0.65| = 0.15.
    probs = np.array([[0.5, 0.5], [0.2, 0.8]])
    assert expected_calibration_error(probs, np.array([0, 0]), n_bins=2) == pytest.approx(0.65)


def test_nll_and_softmax_agree_with_sklearn_log_loss():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(20, 4))
    gold = rng.integers(0, 4, size=20)
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    expected = log_loss(gold, probs, labels=[0, 1, 2, 3])
    assert negative_log_likelihood(logits, gold) == pytest.approx(expected)


def test_routing_metrics_on_a_tiny_label_space():
    labels = build_label_space(
        ["pay_bill", "credit_score", "book_flight", OOS],
        {
            "pay_bill": "finance_agent",
            "credit_score": "finance_agent",
            "book_flight": "travel_agent",
            OOS: OOS,
        },
    )
    probs = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],  # gold pay_bill, pred pay_bill
            [0.6, 0.2, 0.1, 0.1],  # gold credit_score, pred pay_bill (right agent)
            [0.1, 0.1, 0.7, 0.1],  # gold book_flight, correct
            [0.3, 0.3, 0.0, 0.4],  # gold oos: argmax intent oos, summed finance 0.6
        ]
    )
    gold = np.array([0, 1, 2, 3])
    m = routing_metrics(probs, gold, labels)
    assert m["in_scope_accuracy_150"] == pytest.approx(2 / 3)
    assert m["oos_recall_151"] == 1.0
    assert m["accuracy_8"] == 1.0
    assert m["oos_recall_8"] == 1.0
    # Summing moves the oos row to finance: the structural bias the docstring describes.
    assert m["accuracy_8_summed"] == 0.75
    assert m["oos_recall_8_summed"] == 0.0
    assert m["n"] == 4
    assert len(AGENTS) == 8


@pytest.mark.parametrize(
    ("successes", "n", "low", "high"),
    [
        (0, 10, 0.0, 0.2775),  # upper = z^2 / (n + z^2)
        (5, 10, 0.2366, 0.7634),
        (10, 10, 0.7225, 1.0),
    ],
)
def test_wilson_interval_matches_hand_computed_values(successes, n, low, high):
    got = wilson_interval(successes, n)
    assert got == pytest.approx((low, high), abs=1e-4)


@pytest.mark.parametrize(("successes", "n"), [(583, 1000), (1, 3), (59, 100), (990, 1000)])
def test_wilson_interval_agrees_with_scipy(successes, n):
    from scipy.stats import binomtest, norm

    ref = binomtest(successes, n).proportion_ci(confidence_level=0.95, method="wilson")
    got = wilson_interval(successes, n, z=float(norm.ppf(0.975)))
    assert got == pytest.approx((ref.low, ref.high), abs=1e-12)


def test_wilson_interval_is_mirror_symmetric_and_narrows_with_n():
    low, high = wilson_interval(30, 100)
    mirror = wilson_interval(70, 100)
    assert (1 - high, 1 - low) == pytest.approx(mirror)
    small, big = wilson_interval(58, 100), wilson_interval(583, 1000)
    assert big[1] - big[0] < small[1] - small[0]
    assert wilson_interval(583, 1000, z=2.576)[0] < big[0]


@pytest.mark.parametrize(
    ("successes", "n", "z"), [(1, 0, 1.96), (-1, 10, 1.96), (11, 10, 1.96), (5, 10, 0.0)]
)
def test_wilson_interval_refuses_impossible_inputs(successes, n, z):
    with pytest.raises(ValueError):
        wilson_interval(successes, n, z)
