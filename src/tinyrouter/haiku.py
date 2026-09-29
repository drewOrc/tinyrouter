"""Stored Haiku predictions as arrays, plus the old project's parsing rule for comparison.

Reads ``results/llm/haiku-8way.jsonl`` (checked by ``llm_run.verify``
before this is called) and never calls the API. The replies are
re-parsed two ways:

- ``new``: ``llm.parse_agent``, the rule the run used and the hybrid uses;
- ``legacy``: cost-aware-hybrid-router ``src/routers/llm_router.py``:
  lower-case and strip the reply, then take the first label of
  ``VALID_AGENTS`` (a ``set``) that is a substring of it, else ``oos``.
  Set iteration order depends on string hashing, so a reply containing two
  labels had no fixed answer there. Here it takes the first in ``AGENTS``
  order and the number of such replies is reported as ``order_dependent``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinyrouter.labels import AGENTS, OOS, load_label_space
from tinyrouter.llm import parse_agent
from tinyrouter.metrics import wilson_interval
from tinyrouter.selective import detection_counts


class HaikuMismatchError(RuntimeError):
    """A stored prediction disagrees with the dataset labels the archives hold."""


def legacy_parse(raw_text: str) -> tuple[str, int]:
    """(agent, number of labels found as substrings) under the old project's rule."""
    text = raw_text.strip().lower()
    hits = [agent for agent in AGENTS if agent in text]
    return (hits[0] if hits else OOS), len(hits)


@dataclass(frozen=True)
class HaikuSplit:
    """One split's stored predictions, row i = dataset index i."""

    split: str
    gold_intent: np.ndarray
    pred: np.ndarray
    legacy_pred: np.ndarray
    parse_failed: np.ndarray
    legacy_hits: np.ndarray
    cost_usd: np.ndarray
    raw_text: tuple[str, ...]


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_arrays(records: list[dict], split: str, rows: int) -> HaikuSplit:
    """Arrays for ``split`` ordered by index; indexes must be exactly 0..rows-1."""
    chosen = sorted((r for r in records if r["split"] == split), key=lambda r: r["index"])
    if [r["index"] for r in chosen] != list(range(rows)):
        raise HaikuMismatchError(f"{split}: indexes are not exactly 0..{rows - 1}")
    space = load_label_space()
    for r in chosen:
        expected = AGENTS[int(space.intent_to_agent_id[r["gold_intent"]])]
        if r["gold_agent"] != expected:
            raise HaikuMismatchError(
                f"{split}[{r['index']}]: gold_agent {r['gold_agent']} but intent "
                f"{r['gold_intent']} maps to {expected}"
            )
        if parse_agent(r["raw_text"])[0] != r["agent"]:
            raise HaikuMismatchError(f"{split}[{r['index']}]: stored agent is not parse_agent's")
    legacy = [legacy_parse(r["raw_text"]) for r in chosen]
    return HaikuSplit(
        split=split,
        gold_intent=np.array([r["gold_intent"] for r in chosen], dtype=np.int64),
        pred=np.array([AGENTS.index(r["agent"]) for r in chosen], dtype=np.int64),
        legacy_pred=np.array([AGENTS.index(a) for a, _ in legacy], dtype=np.int64),
        parse_failed=np.array([r["parse_failed"] for r in chosen], dtype=bool),
        legacy_hits=np.array([h for _, h in legacy], dtype=np.int64),
        cost_usd=np.array([r["cost_usd"] for r in chosen], dtype=np.float64),
        raw_text=tuple(r["raw_text"] for r in chosen),
    )


def check_gold(haiku: HaikuSplit, labels: np.ndarray, where: str) -> None:
    """Row i of the predictions must carry the gold intent row i of the archive holds."""
    if haiku.gold_intent.shape != labels.shape or not np.array_equal(haiku.gold_intent, labels):
        bad = (
            np.flatnonzero(haiku.gold_intent != labels)
            if haiku.gold_intent.shape == labels.shape
            else []
        )
        raise HaikuMismatchError(
            f"{haiku.split}: Haiku gold_intent differs from {where} labels "
            f"(shapes {haiku.gold_intent.shape} vs {labels.shape}; first rows {list(bad[:5])})"
        )


def routing_summary(pred: np.ndarray, gold_agent: np.ndarray) -> dict[str, object]:
    oos = AGENTS.index(OOS)
    counts = detection_counts(pred == oos, gold_agent == oos)
    low, high = wilson_interval(counts["tp"], counts["gold_oos"])
    return {
        "accuracy_8": float(np.mean(pred == gold_agent)),
        "oos": counts,
        "oos_recall_wilson95": [low, high],
        "oos_misroute_rate": 1.0 - float(counts["recall"]),
        "n": int(pred.size),
    }


def summarize_split(haiku: HaikuSplit) -> dict[str, object]:
    gold_agent = load_label_space().agents_of(haiku.gold_intent)
    differ = np.flatnonzero(haiku.pred != haiku.legacy_pred)
    wrong_if_failed = (haiku.pred == gold_agent) & ~haiku.parse_failed
    return {
        "new_parser": routing_summary(haiku.pred, gold_agent),
        "legacy_parser": routing_summary(haiku.legacy_pred, gold_agent),
        "parsers_disagree_rows": int(differ.size),
        "parsers_disagree_examples": [
            {
                "index": int(i),
                "raw_text": haiku.raw_text[i],
                "new": AGENTS[haiku.pred[i]],
                "legacy": AGENTS[haiku.legacy_pred[i]],
            }
            for i in differ[:10]
        ],
        "legacy_order_dependent_rows": int(np.sum(haiku.legacy_hits > 1)),
        "parse_failed_rows": int(haiku.parse_failed.sum()),
        "accuracy_8_parse_failed_as_wrong": float(wrong_if_failed.mean()),
        "raw_reply_counts": dict(sorted(_counts(haiku.raw_text).items())),
        "cost_usd_total": float(haiku.cost_usd.sum()),
        "cost_usd_per_1k_queries": float(1000 * haiku.cost_usd.mean()),
    }


def _counts(texts: tuple[str, ...]) -> dict[str, int]:
    values, counts = np.unique(np.array(texts, dtype=object).astype(str), return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts, strict=True)}
