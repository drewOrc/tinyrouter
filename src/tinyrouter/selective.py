"""Selective prediction and OOS detection metrics (RQ2 to RQ4). Pure numpy.

Every score here is a confidence: higher means the small model trusts its
decision more. OOS detection flips the sign (low confidence means "this
looks out of scope").

Ties are resolved without randomness. The risk-coverage curve and AURC
use the expected value under a uniformly random order within each group
of tied scores, which is what a stable sort would give on average; AUROC
uses average ranks (a tie counts one half); average precision and the
threshold sweep only ever cut between distinct scores, so tied rows are
always accepted or rejected together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinyrouter.calibrate import FIT_SPLIT, LeakageError

# One-sided 95% upper confidence bound: the threshold is chosen so that the
# validation selective risk is below the target with 95% confidence.
ONE_SIDED_95_Z = 1.6448536269514722
COVERAGE_GRID = tuple(round(0.05 * i, 2) for i in range(1, 21))


def require_validation(split: str, what: str) -> None:
    if split != FIT_SPLIT:
        raise LeakageError(
            f"refusing to choose {what} on the '{split}' split; it is chosen on "
            f"'{FIT_SPLIT}' only and then applied unchanged to test"
        )


def wilson_upper(successes: np.ndarray, n: np.ndarray, z: float) -> np.ndarray:
    """Upper end of the Wilson score interval, elementwise; equals ``metrics.wilson_interval``.

    Counts may be fractional (weighted rows); they must satisfy 0 <= successes <= n, n > 0.
    """
    k, m = np.asarray(successes, dtype=np.float64), np.asarray(n, dtype=np.float64)
    if np.any(m <= 0) or np.any(k < 0) or np.any(k > m):
        raise ValueError("need 0 <= successes <= n and n > 0")
    p, z2 = k / m, z * z
    centre = (p + z2 / (2 * m)) / (1 + z2 / m)
    half = z * np.sqrt(p * (1 - p) / m + z2 / (4 * m * m)) / (1 + z2 / m)
    return np.minimum(1.0, centre + half)


def _tie_groups(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(descending order, start of each tie group in it, size of each group)."""
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]])
    sizes = np.diff(np.r_[starts, ranked.size])
    return order, starts, sizes


def risk_coverage(confidence: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage i/n and expected selective risk of the i most confident rows, i = 1..n."""
    confidence, error = np.asarray(confidence, dtype=np.float64), np.asarray(error, dtype=bool)
    if confidence.shape != error.shape or confidence.ndim != 1 or confidence.size == 0:
        raise ValueError(f"need equal non-empty 1-d arrays, got {confidence.shape}, {error.shape}")
    order, starts, sizes = _tie_groups(confidence)
    group_errors = np.add.reduceat(error[order].astype(np.float64), starts)
    before = np.r_[0.0, np.cumsum(group_errors)[:-1]]
    group = np.repeat(np.arange(starts.size), sizes)
    position = np.arange(confidence.size) - starts[group] + 1
    expected_errors = before[group] + group_errors[group] * position / sizes[group]
    taken = np.arange(1, confidence.size + 1)
    return taken / confidence.size, expected_errors / taken


def aurc(confidence: np.ndarray, error: np.ndarray) -> float:
    """Area under the risk-coverage curve: mean selective risk over coverages 1/n .. n/n."""
    return float(risk_coverage(confidence, error)[1].mean())


def risk_at_coverages(
    confidence: np.ndarray, error: np.ndarray, grid: tuple[float, ...] = COVERAGE_GRID
) -> list[float]:
    """Selective risk when keeping the ceil(c * n) most confident rows, for each c in ``grid``."""
    _, risk = risk_coverage(confidence, error)
    n = risk.size
    return [float(risk[max(1, int(np.ceil(c * n - 1e-9))) - 1]) for c in grid]


def _check_binary(scores: np.ndarray, positive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores, positive = np.asarray(scores, dtype=np.float64), np.asarray(positive, dtype=bool)
    if scores.shape != positive.shape or scores.ndim != 1:
        raise ValueError(f"need equal 1-d arrays, got {scores.shape} and {positive.shape}")
    if positive.all() or not positive.any():
        raise ValueError("need at least one positive and one negative row")
    return scores, positive


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Probability a random positive outscores a random negative (ties count one half)."""
    scores, positive = _check_binary(scores, positive)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    average_rank = (upper - (counts - 1) / 2.0)[inverse]
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((average_rank[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    """Sum over distinct thresholds of (recall gain) x precision; sklearn's definition."""
    scores, positive = _check_binary(scores, positive)
    order, starts, sizes = _tie_groups(scores)
    hits = np.cumsum(np.add.reduceat(positive[order].astype(np.float64), starts))
    taken = np.cumsum(sizes)
    recall = hits / positive.sum()
    precision = hits / taken
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def detection_counts(pred_oos: np.ndarray, gold_oos: np.ndarray) -> dict[str, float | int | None]:
    """OOS recall, precision and F1 (precision 0 when nothing is predicted oos, as sklearn)."""
    pred_oos, gold_oos = np.asarray(pred_oos, dtype=bool), np.asarray(gold_oos, dtype=bool)
    tp = int(np.sum(pred_oos & gold_oos))
    positives, predicted = int(gold_oos.sum()), int(pred_oos.sum())
    if positives == 0:
        raise ValueError("no gold oos rows")
    recall = tp / positives
    precision = tp / predicted if predicted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "gold_oos": positives,
        "predicted_oos": predicted,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def high_confidence_oos_misroute(
    pred_oos: np.ndarray, gold_oos: np.ndarray, accepted: np.ndarray
) -> float:
    """True OOS sent to an in-scope agent by the small model without asking the LLM, / all OOS."""
    pred_oos, gold_oos = np.asarray(pred_oos, dtype=bool), np.asarray(gold_oos, dtype=bool)
    accepted = np.asarray(accepted, dtype=bool)
    if not gold_oos.any():
        raise ValueError("no gold oos rows")
    return float(np.sum(gold_oos & ~pred_oos & accepted) / gold_oos.sum())


@dataclass(frozen=True)
class Scored:
    """One split's confidence scores and 8-way errors, taken together from one routed split.

    The threshold choosers take this object instead of a split name and
    loose arrays, so the name they check and the numbers they use cannot
    come from different splits. Build it with ``analysis.Routed.scored``.
    ``weight`` (optional, one per row) reweights rows, e.g. to a declared
    OOS share in the prior-shift diagnostic.
    """

    split: str
    confidence: np.ndarray
    error: np.ndarray
    weight: np.ndarray | None = None

    def __post_init__(self) -> None:
        shapes = {np.shape(self.confidence), np.shape(self.error)}
        if self.weight is not None:
            shapes.add(np.shape(self.weight))
        if len(shapes) != 1 or np.ndim(self.confidence) != 1 or np.size(self.confidence) == 0:
            raise ValueError(f"confidence, error and weight must be equal 1-d arrays, got {shapes}")


@dataclass(frozen=True)
class ThresholdChoice:
    """The chosen confidence threshold and what it did on validation.

    ``tau`` is None when no threshold meets the target; the router then
    sends every query to the LLM (coverage 0), which is what a deployed
    router with an unreachable risk target would have to do.
    """

    tau: float | None
    target_risk: float
    coverage: float
    risk: float | None
    risk_upper: float | None
    candidates: int

    @property
    def feasible(self) -> bool:
        return self.tau is not None


def deferred_below(confidence: np.ndarray, tau: float | None) -> np.ndarray:
    """Rows sent to the LLM: confidence strictly below ``tau`` (a row equal to tau is kept).

    ``tau`` None (no feasible threshold) defers every row.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    if tau is None:
        return np.ones(confidence.size, dtype=bool)
    return confidence < tau


def select_threshold(
    scored: Scored, target_risk: float, z: float = ONE_SIDED_95_Z
) -> ThresholdChoice:
    """Largest-coverage threshold whose selective-risk Wilson upper bound is <= ``target_risk``.

    Candidates are the distinct validation scores; threshold t accepts
    every row with confidence >= t. With weights, the kept count and the
    errors are weighted sums and coverage is the kept share of the total
    weight.
    """
    require_validation(scored.split, "a deferral threshold")
    confidence = np.asarray(scored.confidence, dtype=np.float64)
    error = np.asarray(scored.error, dtype=bool)
    weight = np.ones(confidence.size) if scored.weight is None else np.asarray(scored.weight, float)
    order, starts, sizes = _tie_groups(confidence)
    accepted = np.cumsum(np.add.reduceat(weight[order], starts))
    errors = np.cumsum(np.add.reduceat((weight * error)[order], starts))
    upper = wilson_upper(np.minimum(errors, accepted), accepted, z)
    ok = np.flatnonzero(upper <= target_risk)
    if ok.size == 0:
        return ThresholdChoice(None, target_risk, 0.0, None, None, int(starts.size))
    best = ok[-1]
    return ThresholdChoice(
        tau=float(confidence[order][starts[best]]),
        target_risk=target_risk,
        coverage=float(accepted[best] / weight.sum()),
        risk=float(errors[best] / accepted[best]),
        risk_upper=float(upper[best]),
        candidates=int(starts.size),
    )


def coverage_thresholds(scored: Scored, grid: tuple[float, ...] = COVERAGE_GRID) -> list[float]:
    """For each c in ``grid``, the score of the ceil(c * n)-th most confident validation row."""
    require_validation(scored.split, "operating-curve thresholds")
    ranked = np.sort(np.asarray(scored.confidence, dtype=np.float64))[::-1]
    n = ranked.size
    return [float(ranked[max(1, int(np.ceil(c * n - 1e-9))) - 1]) for c in grid]
