import numpy as np
import pytest

from tinyrouter.calibrate import (
    LeakageError,
    SplitLogits,
    TemperatureBoundWarning,
    apply_temperature,
    fit_temperature,
)
from tinyrouter.metrics import negative_log_likelihood


def overconfident_logits(seed: int = 0, n: int = 400, k: int = 5) -> tuple:
    """Logits whose argmax is right 70% of the time but scaled so softmax says ~100%."""
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, k, size=n)
    pred = np.where(rng.random(n) < 0.7, gold, (gold + 1) % k)
    logits = rng.normal(scale=0.5, size=(n, k))
    logits[np.arange(n), pred] += 3.0
    return logits * 5.0, gold


def test_temperature_reduces_nll_of_overconfident_logits():
    logits, gold = overconfident_logits()
    val = SplitLogits("validation", logits, gold)
    temperature = fit_temperature(val)
    assert temperature > 1.0
    before = negative_log_likelihood(logits, gold)
    after = negative_log_likelihood(logits / temperature, gold)
    assert after < before


def test_fitted_temperature_is_the_nll_minimum_on_a_grid():
    logits, gold = overconfident_logits(seed=1)
    temperature = fit_temperature(SplitLogits("validation", logits, gold))
    grid = np.linspace(0.5, 15, 400)
    best = grid[np.argmin([negative_log_likelihood(logits / t, gold) for t in grid])]
    assert temperature == pytest.approx(best, rel=0.05)


def test_underconfident_logits_get_temperature_below_one():
    logits, gold = overconfident_logits(seed=2)
    temperature = fit_temperature(SplitLogits("validation", logits / 50.0, gold))
    assert temperature < 1.0


@pytest.mark.parametrize("split", ["test", "train"])
def test_fit_temperature_refuses_any_split_but_validation(split):
    logits, gold = overconfident_logits()
    with pytest.raises(LeakageError, match=split):
        fit_temperature(SplitLogits(split, logits, gold))


def test_fit_warns_when_the_optimum_hits_the_search_bound():
    rng = np.random.default_rng(3)
    uniform = np.zeros((50, 4))
    uniform[:, 0] = 1e-4
    with pytest.warns(TemperatureBoundWarning):
        fit_temperature(SplitLogits("validation", uniform, rng.integers(0, 4, 50)))


def test_split_logits_rejects_misaligned_shapes():
    with pytest.raises(ValueError, match="line up"):
        SplitLogits("validation", np.zeros((3, 2)), np.zeros(4, dtype=int))


def test_apply_temperature_returns_probabilities_and_rejects_nonpositive_t():
    probs = apply_temperature(np.array([[2.0, 0.0], [0.0, 1.0]]), 2.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 0] == pytest.approx(1 / (1 + np.exp(-1.0)))
    with pytest.raises(ValueError):
        apply_temperature(np.zeros((1, 2)), 0.0)
