"""k-shot training samples for the learning curves (docs/PLAN.md section 4, AC5).

Every in-scope intent gets exactly ``k`` rows. Out-of-scope gets the
number in ``OOS_TRAIN_ROWS``, a hardcoded table (not computed at run
time) equal to ceil(2.5k): the OOS+ train split's ratio of 250 oos rows
to 100 per intent, so ``k = 100`` is the whole train split. The table is
checked against ceil(2.5k) by a test, not by the code that uses it.

The sample depends only on the split's content, ``k``, the oos count and
``seed``; the selected rows keep their original order.
"""

from __future__ import annotations

import hashlib

import numpy as np

from tinyrouter.data import SPLIT_FILES, Split, load_split
from tinyrouter.labels import load_label_space

OOS_TRAIN_ROWS: dict[int, int] = {1: 3, 5: 13, 10: 25, 25: 63, 50: 125, 100: 250}
CURVE_KS: tuple[int, ...] = tuple(OOS_TRAIN_ROWS)
FULL_K = 100
IN_SCOPE_INTENTS = 150


class SamplingError(ValueError):
    """A k-shot sample cannot be drawn as specified."""


def oos_rows_for(k: int, oos_override: int | None = None) -> int:
    """Out-of-scope rows in the k-shot sample; ``oos_override`` replaces the table (ablation)."""
    if k not in OOS_TRAIN_ROWS:
        raise SamplingError(f"k={k} is not a curve point; allowed: {CURVE_KS}")
    if oos_override is None:
        return OOS_TRAIN_ROWS[k]
    if oos_override < 0:
        raise SamplingError(f"oos override must be >= 0, got {oos_override}")
    return oos_override


def planned_rows(k: int, oos_override: int | None = None) -> tuple[int, int]:
    """(train rows, oos rows) a k-shot sample will have, without reading any data."""
    oos = oos_rows_for(k, oos_override)
    return IN_SCOPE_INTENTS * k + oos, oos


def k_shot_indices(
    intents: np.ndarray, k: int, seed: int, oos_override: int | None = None
) -> np.ndarray:
    """Sorted row indices of the k-shot sample of a train split's intent column."""
    oos_count = oos_rows_for(k, oos_override)
    oos_id = load_label_space().oos_intent_id
    present = set(np.unique(intents).tolist())
    expected = set(range(load_label_space().num_intents))
    if present != expected:
        raise SamplingError(
            f"train split has {len(present)} distinct intents, expected all {len(expected)}"
        )
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for intent in sorted(present):
        rows = np.flatnonzero(intents == intent)
        want = oos_count if intent == oos_id else k
        if len(rows) < want:
            raise SamplingError(f"intent {intent} has {len(rows)} rows, need {want}")
        keep.append(rows if len(rows) == want else rng.choice(rows, size=want, replace=False))
    return np.sort(np.concatenate(keep)).astype(np.int64)


def sample_k_shot(split: Split, k: int, seed: int, oos_override: int | None = None) -> Split:
    """The k-shot training sample; at k=100 with the table's 250 oos rows it is the whole split."""
    if split.name != "train":
        raise SamplingError(f"only the train split is subsampled, got '{split.name}'")
    return split.take(k_shot_indices(split.intents, k, seed, oos_override))


def curve_sample(k: int, seed: int, oos_override: int | None = None) -> Split:
    """The training rows of curve point (k, seed): shared by the encoders and the baselines."""
    return sample_k_shot(load_split("train"), k, seed, oos_override)


def sample_fingerprint(split: Split) -> str:
    """SHA-256 over the sample's (text, intent) rows in order.

    Two runs with the same fingerprint trained on the same rows in the same
    order, so an encoder and a baseline can be checked to share a sample.
    """
    digest = hashlib.sha256()
    for text, intent in zip(split.texts, split.intents.tolist(), strict=True):
        digest.update(f"{intent}\t{text}\n".encode())
    return digest.hexdigest()


def full_train_rows() -> int:
    return SPLIT_FILES["train"].rows
