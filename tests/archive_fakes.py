"""Synthetic logits archives shared by the archive and AC2 tests."""

import numpy as np

from tinyrouter.calibrate import SplitLogits
from tinyrouter.labels import load_label_space

NUM_INTENTS = 151


def fake_splits(seed: int = 0, n_val: int = 7, n_test: int = 11) -> dict[str, SplitLogits]:
    rng = np.random.default_rng(seed)
    return {
        name: SplitLogits(
            name,
            rng.normal(scale=4.0, size=(n, NUM_INTENTS)).astype(np.float32),
            rng.integers(0, NUM_INTENTS, size=n),
        )
        for name, n in (("validation", n_val), ("test", n_test))
    }


def fake_metadata(**overrides: object) -> dict[str, object]:
    meta: dict[str, object] = {
        "format_version": 1,
        "run_name": "bert-base-uncased-full-seed42",
        "model_name": "google-bert/bert-base-uncased",
        "model_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
        "seed": 42,
        "per_intent": None,
        "train_rows": 15_250,
        "oos_train_rows": 250,
        "eval_per_intent": None,
        "dataset_revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
        "label_space_sha256": load_label_space().sha256,
        "git_commit": "0" * 40,
        "git_dirty": False,
        "created_at": "2026-09-23T00:00:00+00:00",
    }
    meta.update(overrides)
    return meta
