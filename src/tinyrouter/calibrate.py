"""Temperature scaling, fitted on validation logits only.

The fit function takes a ``SplitLogits`` rather than bare arrays so the
split name travels with the numbers. Passing anything but validation
raises ``LeakageError``: the temperature (and later the deferral
threshold) must never see test data (docs/PLAN.md AC4).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from tinyrouter.data import SplitName
from tinyrouter.metrics import negative_log_likelihood, softmax

FIT_SPLIT: SplitName = "validation"
# Search range for 1/T. NLL(z * beta) is convex in beta, so golden-section
# search on a bracket finds the global minimum inside it.
BETA_MIN, BETA_MAX = 0.01, 20.0


class LeakageError(ValueError):
    """A split other than validation was passed to a fitting function."""


class TemperatureBoundWarning(UserWarning):
    """The fitted temperature sits on the edge of the search range."""


@dataclass(frozen=True)
class SplitLogits:
    """Model logits for one named split, with gold labels."""

    split: SplitName
    logits: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        if self.logits.ndim != 2 or self.logits.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"logits {self.logits.shape} and labels {self.labels.shape} do not line up"
            )


def require_fit_split(split_logits: SplitLogits) -> None:
    if split_logits.split != FIT_SPLIT:
        raise LeakageError(
            f"refusing to fit on the '{split_logits.split}' split; "
            f"calibration parameters are chosen on '{FIT_SPLIT}' only"
        )


def fit_temperature(val: SplitLogits, tol: float = 1e-6) -> float:
    """Return the temperature T > 0 minimising validation NLL of softmax(logits / T)."""
    require_fit_split(val)

    def loss(beta: float) -> float:
        return negative_log_likelihood(val.logits * beta, val.labels)

    lo, hi = BETA_MIN, BETA_MAX
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = loss(a), loss(b)
    while hi - lo > tol:
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = loss(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = loss(b)
    beta = (lo + hi) / 2
    if beta - BETA_MIN < 1e-3 or BETA_MAX - beta < 1e-3:
        warnings.warn(
            f"temperature fit hit the search bound (T={1 / beta:.3g}); the logits are "
            "probably near-uniform or degenerate, so T is not meaningful",
            TemperatureBoundWarning,
            stacklevel=2,
        )
    return 1.0 / beta


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrated probabilities softmax(logits / T)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return softmax(np.asarray(logits, dtype=np.float64) / temperature)
