"""Evaluation metrics. Pure numpy; no model code here.

Definitions follow Larson et al. (2019): in-scope accuracy is exact-intent
accuracy over queries whose gold label is not oos (predicting oos on an
in-scope query counts as wrong); OOS recall is the share of gold-oos
queries predicted as oos.
"""

from __future__ import annotations

import numpy as np

from tinyrouter.labels import LabelSpace


def accuracy(pred: np.ndarray, gold: np.ndarray) -> float:
    pred, gold = np.asarray(pred), np.asarray(gold)
    if pred.shape != gold.shape or pred.size == 0:
        raise ValueError(f"need equal, non-empty shapes; got {pred.shape} and {gold.shape}")
    return float(np.mean(pred == gold))


def in_scope_accuracy(pred: np.ndarray, gold: np.ndarray, oos_id: int) -> float:
    pred, gold = np.asarray(pred), np.asarray(gold)
    mask = gold != oos_id
    if not mask.any():
        raise ValueError("no in-scope examples")
    return accuracy(pred[mask], gold[mask])


def oos_recall(pred: np.ndarray, gold: np.ndarray, oos_id: int) -> float:
    pred, gold = np.asarray(pred), np.asarray(gold)
    mask = gold == oos_id
    if not mask.any():
        raise ValueError("no oos examples")
    return float(np.mean(pred[mask] == oos_id))


def expected_calibration_error(probs: np.ndarray, gold: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins over (0, 1].

    A prediction with confidence c falls in bin b when b/n < c <= (b+1)/n.
    ECE = sum over bins of (bin size / n) * |accuracy(bin) - mean confidence(bin)|.
    """
    probs, gold = np.asarray(probs, dtype=np.float64), np.asarray(gold)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == gold).astype(np.float64)
    bins = np.clip(np.ceil(confidence * n_bins).astype(np.int64) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
            ece += in_bin.mean() * gap
    return float(ece)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(logits))


def negative_log_likelihood(logits: np.ndarray, gold: np.ndarray) -> float:
    """Mean NLL of the gold labels under softmax(logits)."""
    logp = log_softmax(logits)
    return float(-logp[np.arange(len(gold)), np.asarray(gold)].mean())


def probs_nll(probs: np.ndarray, gold: np.ndarray) -> float:
    """Mean NLL of the gold labels given probabilities (clipped at 1e-12)."""
    picked = np.asarray(probs, dtype=np.float64)[np.arange(len(gold)), np.asarray(gold)]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


def routing_metrics(
    intent_probs: np.ndarray, gold_intents: np.ndarray, label_space: LabelSpace
) -> dict[str, float]:
    """All headline numbers for one split from 151-way probabilities.

    ``accuracy_8`` / ``oos_recall_8`` map the argmax intent to its agent.
    The ``_summed`` variants instead take the argmax of agent probabilities
    summed over each agent's intents. The two differ structurally: oos owns
    one intent while finance_agent owns 38, so summing moves mass away from
    oos. Both are reported so the choice is made on validation data, not
    assumed.
    """
    oos_intent, oos_agent = label_space.oos_intent_id, label_space.oos_agent_id
    pred_intent = intent_probs.argmax(axis=1)
    pred_agent = label_space.agents_of(pred_intent)
    agent_probs = label_space.aggregate_probs(intent_probs)
    pred_agent_summed = agent_probs.argmax(axis=1)
    gold_agent = label_space.agents_of(gold_intents)
    return {
        "accuracy_8": accuracy(pred_agent, gold_agent),
        "oos_recall_8": oos_recall(pred_agent, gold_agent, oos_agent),
        "accuracy_8_summed": accuracy(pred_agent_summed, gold_agent),
        "oos_recall_8_summed": oos_recall(pred_agent_summed, gold_agent, oos_agent),
        "in_scope_accuracy_150": in_scope_accuracy(pred_intent, gold_intents, oos_intent),
        "oos_recall_151": oos_recall(pred_intent, gold_intents, oos_intent),
        "ece_151": expected_calibration_error(intent_probs, gold_intents),
        "ece_8_summed": expected_calibration_error(agent_probs, gold_agent),
        "nll_151": probs_nll(intent_probs, gold_intents),
        "n": int(len(gold_intents)),
    }
