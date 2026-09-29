"""RQ2 to RQ4 for one archived run: 8-way routing, OOS, uncertainty, LLM fallback.

Everything is computed from one logits archive (AC3) and the stored Haiku
predictions; nothing is trained and no API is called. Every choice is
made on validation and applied unchanged to test (AC4), and each choosing
function refuses any other split with ``LeakageError``:

- the temperature (``calibrate.fit_temperature``);
- the 8-way aggregation (``select_aggregation``): higher validation 8-way
  accuracy, ties go to argmax, the PLAN's default;
- the deferral threshold per signal and target risk
  (``selective.select_threshold``);
- which signal the hybrid router uses (``select_signal``): largest
  validation coverage at the target, then lower validation AURC, then the
  order of ``SIGNALS``.

Aggregations. ``argmax``: the argmax intent's agent; the signals are read
off the 151-way distribution. ``summed``: the argmax of the 8 agent
probabilities summed over each agent's intents (temperature-scaled, since
that argmax depends on T); the signals are read off the 8-way
distribution. Signals (higher = more confident): ``msp`` max probability
and ``entropy`` minus the entropy, both at T = 1; ``msp_t`` max
probability at the fitted T; ``margin`` the gap between the two largest
log-probabilities at the fitted T. For argmax that gap is
``(z1 - z2) / T``, which ranks rows exactly as the raw score margin does,
so it is the "score margin" PLAN section 4.1 asks for on TF-IDF.

Which signals a run gets (PLAN section 4.1): encoders all four; TF-IDF
only ``msp_t`` and ``margin`` (its 100x cosine scale distorts the T = 1
signals and the raw ECE); majority none, and no temperature (every row
has the same scores, so every signal is a constant).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from tinyrouter.calibrate import SplitLogits, TemperatureBoundWarning, fit_temperature
from tinyrouter.labels import AGENTS, OOS, LabelSpace
from tinyrouter.metrics import expected_calibration_error, log_softmax, softmax, wilson_interval
from tinyrouter.selective import (
    COVERAGE_GRID,
    ThresholdChoice,
    aurc,
    auroc,
    average_precision,
    coverage_thresholds,
    detection_counts,
    high_confidence_oos_misroute,
    require_validation,
    risk_at_coverages,
    select_threshold,
)

AGGREGATIONS = ("argmax", "summed")
SIGNALS = ("msp", "entropy", "margin", "msp_t")
SIGNALS_BY_KIND: dict[str, tuple[str, ...]] = {
    "encoder": SIGNALS,
    "tfidf": ("margin", "msp_t"),
    "majority": (),
}
TARGET_RISKS = (0.02, 0.05)
OOS_AGENT = AGENTS.index(OOS)


@dataclass(frozen=True)
class Routed:
    """One split routed in the 8-way space by one aggregation."""

    split: str
    pred: np.ndarray
    gold: np.ndarray
    signals: dict[str, np.ndarray]

    @property
    def error(self) -> np.ndarray:
        return self.pred != self.gold

    @property
    def gold_oos(self) -> np.ndarray:
        return self.gold == OOS_AGENT

    @property
    def pred_oos(self) -> np.ndarray:
        return self.pred == OOS_AGENT


def top_two(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    part = np.partition(values, -2, axis=1)
    return part[:, -1], part[:, -2]


def signals_of(raw: np.ndarray, cal: np.ndarray, log_cal: np.ndarray) -> dict[str, np.ndarray]:
    safe = np.clip(raw, 1e-300, None)
    first, second = top_two(log_cal)
    return {
        "msp": raw.max(axis=1),
        "entropy": np.sum(raw * np.log(safe), axis=1),
        "margin": first - second,
        "msp_t": cal.max(axis=1),
    }


def route(
    split: SplitLogits, temperature: float, aggregation: str, space: LabelSpace, kind: str
) -> Routed:
    """8-way predictions and the ``kind``'s signals for one split."""
    z = np.asarray(split.logits, dtype=np.float64)
    raw, cal = softmax(z), softmax(z / temperature)
    if aggregation == "argmax":
        pred = space.agents_of(z.argmax(axis=1))
        signals = signals_of(raw, cal, log_softmax(z / temperature))
    elif aggregation == "summed":
        raw8, cal8 = space.aggregate_probs(raw), space.aggregate_probs(cal)
        pred = cal8.argmax(axis=1)
        signals = signals_of(raw8, cal8, np.log(np.clip(cal8, 1e-300, None)))
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}; expected one of {AGGREGATIONS}")
    kept = {name: signals[name] for name in SIGNALS_BY_KIND[kind]}
    return Routed(split.split, pred, space.agents_of(split.labels), kept)


def fit_run_temperature(val: SplitLogits, kind: str) -> tuple[float, dict[str, object]]:
    """T on validation (none for majority), and a note of how it was obtained."""
    if kind == "majority":
        return 1.0, {"temperature": None, "note": "not fitted: every row has the same scores"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", TemperatureBoundWarning)
        temperature = fit_temperature(val)
    hit_bound = any(issubclass(w.category, TemperatureBoundWarning) for w in caught)
    return temperature, {"temperature": temperature, "hit_search_bound": hit_bound}


def select_aggregation(candidates: dict[str, Routed]) -> tuple[str, dict[str, float]]:
    """The validation-best aggregation (ties: argmax) and each one's validation accuracy."""
    for routed in candidates.values():
        require_validation(routed.split, "the 8-way aggregation")
    accuracy = {name: float(np.mean(~r.error)) for name, r in candidates.items()}
    best = max(AGGREGATIONS, key=lambda name: (accuracy[name], name == "argmax"))
    return best, accuracy


def router_metrics(
    small: Routed, llm_pred: np.ndarray, llm_cost: np.ndarray, deferred: np.ndarray
) -> dict[str, object]:
    """Small model where it keeps the query, the LLM where it defers; all rates over this split."""
    final = np.where(deferred, llm_pred, small.pred)
    accepted = ~deferred
    recoverable = small.error & (llm_pred == small.gold)
    recovered = recoverable & deferred
    counts = detection_counts(final == OOS_AGENT, small.gold_oos)
    errors, n_recoverable = int(small.error.sum()), int(recoverable.sum())
    return {
        "coverage": float(accepted.mean()),
        "selective_risk": float(small.error[accepted].mean()) if accepted.any() else None,
        "accuracy_8": float(np.mean(final == small.gold)),
        "oos_recall": counts["recall"],
        "oos_recall_wilson95": list(wilson_interval(counts["tp"], counts["gold_oos"])),
        "oos_precision": counts["precision"],
        "oos_f1": counts["f1"],
        "oos_misroute_rate": 1.0 - counts["recall"],
        "high_conf_oos_misroute_rate": high_confidence_oos_misroute(
            small.pred_oos, small.gold_oos, accepted
        ),
        "llm_call_rate": float(deferred.mean()),
        "llm_cost_usd_per_1k": float(1000 * llm_cost[deferred].sum() / deferred.size),
        "error_recovery_rate": float(recovered.sum() / errors) if errors else None,
        "recoverable_caught": float(recovered.sum() / n_recoverable) if n_recoverable else None,
        "small_errors": errors,
        "recoverable_errors": n_recoverable,
        "recovered_errors": int(recovered.sum()),
    }


def signal_quality(routed: Routed, signal: str) -> dict[str, object]:
    """Threshold-free quality of one signal on one split (RQ2 detection, RQ3 selection)."""
    confidence = routed.signals[signal]
    return {
        "aurc": aurc(confidence, routed.error),
        "oos_auroc": auroc(-confidence, routed.gold_oos),
        "oos_auprc": average_precision(-confidence, routed.gold_oos),
    }


def select_signal(val: Routed, choices: dict[str, ThresholdChoice]) -> str | None:
    """Signal with the largest validation coverage at the target (see module docstring)."""
    require_validation(val.split, "the hybrid's signal")
    if not choices:
        return None
    order = {name: i for i, name in enumerate(SIGNALS)}

    def key(name: str) -> tuple[float, float, int]:
        return (-choices[name].coverage, aurc(val.signals[name], val.error), order[name])

    return min(choices, key=key)


def choice_record(choice: ThresholdChoice) -> dict[str, object]:
    return {
        "tau": choice.tau,
        "feasible": choice.feasible,
        "validation_coverage": choice.coverage,
        "validation_selective_risk": choice.risk,
        "validation_risk_upper95": choice.risk_upper,
    }


def fallback(
    val: Routed, test: Routed, llm_pred: np.ndarray, llm_cost: np.ndarray
) -> dict[str, object]:
    """RQ4: per target risk, every signal's validation-chosen threshold applied to test."""
    out: dict[str, object] = {}
    for target in TARGET_RISKS:
        choices = {
            s: select_threshold(val.split, val.signals[s], val.error, target) for s in val.signals
        }
        per_signal = {}
        for signal, choice in choices.items():
            confidence = test.signals[signal]
            deferred = np.ones(confidence.size, dtype=bool)
            if choice.tau is not None:
                deferred = confidence < choice.tau
            test_metrics = router_metrics(test, llm_pred, llm_cost, deferred)
            per_signal[signal] = {**choice_record(choice), "test": test_metrics}
        chosen = select_signal(val, choices)
        out[f"{target:.2f}"] = {
            "selected_signal": chosen,
            "hybrid": per_signal[chosen] if chosen else None,
            "by_signal": per_signal,
        }
    return out


def operating_curve(
    val: Routed, test: Routed, llm_pred: np.ndarray, llm_cost: np.ndarray, signal: str
) -> dict[str, list[float | None]]:
    """For each validation coverage in the grid, its threshold and what it does on test."""
    taus = coverage_thresholds(val.split, val.signals[signal])
    keys = (
        "coverage",
        "selective_risk",
        "accuracy_8",
        "high_conf_oos_misroute_rate",
        "oos_misroute_rate",
        "llm_call_rate",
    )
    curve: dict[str, list[float | None]] = {"tau": taus, **{k: [] for k in keys}}
    for tau in taus:
        metrics = router_metrics(test, llm_pred, llm_cost, test.signals[signal] < tau)
        for k in keys:
            curve[k].append(metrics[k])
    return curve


def aggregation_block(
    val: Routed, test: Routed, llm_pred: np.ndarray, llm_cost: np.ndarray
) -> tuple[dict[str, object], dict[str, object]]:
    """(scalar results, curves) for one aggregation."""
    never = np.zeros(test.pred.size, dtype=bool)
    scalars = {
        "validation_accuracy_8": float(np.mean(~val.error)),
        "small_only": router_metrics(test, llm_pred, llm_cost, never),
        "oracle": router_metrics(test, llm_pred, llm_cost, test.error),
        "signals": {
            s: {"validation": signal_quality(val, s), "test": signal_quality(test, s)}
            for s in test.signals
        },
        "fallback": fallback(val, test, llm_pred, llm_cost),
    }
    curves = {
        s: {
            "risk_coverage": risk_at_coverages(test.signals[s], test.error),
            "operating": operating_curve(val, test, llm_pred, llm_cost, s),
        }
        for s in test.signals
    }
    return scalars, curves


def calibration_block(val: SplitLogits, test: SplitLogits, t: float, space: LabelSpace, kind: str):
    """Test ECE before and after temperature scaling, 151-way and 8-way summed."""
    if kind == "majority":
        return None
    z = np.asarray(test.logits, dtype=np.float64)
    gold8 = space.agents_of(test.labels)
    out = {}
    for name, probs in (("uncalibrated", softmax(z)), ("calibrated", softmax(z / t))):
        if kind == "tfidf" and name == "uncalibrated":
            out[name] = None
            continue
        out[name] = {
            "ece_151": expected_calibration_error(probs, test.labels),
            "ece_8_summed": expected_calibration_error(space.aggregate_probs(probs), gold8),
        }
    return out


def analyze_run(
    val: SplitLogits,
    test: SplitLogits,
    kind: str,
    space: LabelSpace,
    llm_pred: np.ndarray,
    llm_cost: np.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    """(scalars, curves) for one archive; ``llm_*`` are the test-split Haiku arrays."""
    temperature, t_record = fit_run_temperature(val, kind)
    routed = {
        a: (route(val, temperature, a, space, kind), route(test, temperature, a, space, kind))
        for a in AGGREGATIONS
    }
    selected, val_accuracy = select_aggregation({a: v for a, (v, _) in routed.items()})
    scalars: dict[str, object] = {
        "calibration": {
            **t_record,
            "test_ece": calibration_block(val, test, temperature, space, kind),
        },
        "selected_aggregation": selected,
        "aggregation_validation_accuracy_8": val_accuracy,
        "signals_reported": list(SIGNALS_BY_KIND[kind]),
    }
    curves: dict[str, object] = {}
    for a, (v, t) in routed.items():
        scalars[a], curves[a] = aggregation_block(v, t, llm_pred, llm_cost)
    scalars["final"] = final_view(scalars[selected])
    return scalars, curves


def final_view(block: dict[str, object]) -> dict[str, object]:
    """The selected aggregation's router numbers, without the per-signal detail."""
    fallback_block = block["fallback"]
    assert isinstance(fallback_block, dict)
    return {
        "small_only": block["small_only"],
        "oracle": block["oracle"],
        "signals": block["signals"],
        "fallback": {
            target: {"selected_signal": v["selected_signal"], "hybrid": v["hybrid"]}
            for target, v in fallback_block.items()
        },
    }


__all__ = ["COVERAGE_GRID", "analyze_run", "route", "router_metrics", "select_aggregation"]
