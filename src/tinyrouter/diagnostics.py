"""Why validation-chosen thresholds miss the target risk on test (diagnosis, not method).

CLINC150's validation split is 3.2% OOS (100 of 3,100) and test is 18.2%
(1,000 of 5,500). For each encoder point's final hybrid, this reports the
selective risk on validation and on test, split into in-scope rows and
kept OOS rows, and the test risk after reweighting test rows to the
validation OOS share. The share of the gap that reweighting closes is how
much the OOS proportion explains; the rest is test OOS being harder.

``sensitivity_reweighted_validation`` is PLAN option (b): validation rows
reweighted to ``DECLARED_OOS_SHARE`` before choosing the threshold, then
applied to test. That share is the test split's, known only because we
looked at test, so this is a diagnostic sensitivity check and never the
reported method (the main result stays validation-only).

Routed splits are used through their attributes only (``signals``,
``error``, ``gold_oos``, ``pred_oos``, ``scored``), so this module does
not import ``analysis``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tinyrouter.selective import deferred_below, select_threshold

DECLARED_OOS_SHARE = 0.182
SENSITIVITY_NOTE = (
    "validation rows reweighted to a 18.2% OOS share (the test split's, known only after looking "
    "at test) before choosing tau; diagnostic only, not the reported method"
)


def prior_weights(gold_oos: np.ndarray, share: float) -> np.ndarray:
    """Per-row weights that make the weighted OOS share ``share``; they average to 1."""
    gold_oos = np.asarray(gold_oos, dtype=bool)
    p = float(gold_oos.mean())
    if not 0 < p < 1 or not 0 < share < 1:
        raise ValueError(f"need both classes and 0 < share < 1, got OOS share {p} and {share}")
    return np.where(gold_oos, share / p, (1 - share) / (1 - p))


def kept_rates(routed: Any, signal: str, tau: float | None, weight: np.ndarray | None = None):
    """Coverage and error rates of the rows the small model keeps at ``tau`` (weighted)."""
    kept = ~deferred_below(routed.signals[signal], tau)
    w = np.ones(kept.size) if weight is None else np.asarray(weight, dtype=np.float64)
    error, gold_oos = routed.error, routed.gold_oos

    def rate(mask: np.ndarray, hit: np.ndarray) -> float | None:
        denominator = float(w[mask].sum())
        return float(w[mask & hit].sum() / denominator) if denominator > 0 else None

    return {
        "coverage": float(w[kept].sum() / w.sum()),
        "selective_risk": rate(kept, error),
        "in_scope_risk": rate(kept & ~gold_oos, error),
        "kept_oos_error_rate": rate(kept & gold_oos, error),
        "kept_oos_share": rate(kept, gold_oos),
        "high_conf_oos_misroute_rate": rate(gold_oos, kept & ~routed.pred_oos),
    }


def gap_explained(val_risk: float | None, test_risk: float | None, reweighted: float | None):
    """(test - reweighted) / (test - validation); None when undefined."""
    if None in (val_risk, test_risk, reweighted) or test_risk <= val_risk:
        return None
    return float((test_risk - reweighted) / (test_risk - val_risk))


def sensitivity(val: Any, test: Any, signal: str, target: float) -> dict[str, object]:
    weight = prior_weights(val.gold_oos, DECLARED_OOS_SHARE)
    choice = select_threshold(val.scored(signal, weight), target)
    return {
        "note": SENSITIVITY_NOTE,
        "tau": choice.tau,
        "feasible": choice.feasible,
        "validation_weighted_coverage": choice.coverage,
        "validation_weighted_risk": choice.risk,
        "test": kept_rates(test, signal, choice.tau),
    }


def transfer(val: Any, test: Any, signal: str, tau: float | None) -> dict[str, object]:
    val_share = float(val.gold_oos.mean())
    v = kept_rates(val, signal, tau)
    t = kept_rates(test, signal, tau)
    rw = kept_rates(test, signal, tau, prior_weights(test.gold_oos, val_share))
    return {
        "validation_oos_share": val_share,
        "test_oos_share": float(test.gold_oos.mean()),
        "validation": v,
        "test": t,
        "test_reweighted_to_validation_oos_share": rw,
        "share_of_gap_explained_by_oos_share": gap_explained(
            v["selective_risk"], t["selective_risk"], rw["selective_risk"]
        ),
    }


def threshold_diagnostics(val: Any, test: Any, block: dict[str, Any]) -> dict[str, object]:
    """Per target risk, the final hybrid's threshold transfer and the (b) sensitivity."""
    out: dict[str, object] = {}
    for target, entry in block["fallback"].items():
        signal = entry["selected_signal"]
        if signal is None:
            out[target] = None
            continue
        tau = entry["hybrid"]["tau"]
        out[target] = {
            "signal": signal,
            "tau": tau,
            **transfer(val, test, signal, tau),
            "sensitivity_reweighted_validation": sensitivity(val, test, signal, float(target)),
        }
    return out
