"""CLINC150 (``clinc_oos``, ``plus`` config) from the Hugging Face Hub, pinned and verified.

Files are fetched by exact commit and each one is checked against its
SHA-256 (the LFS object id the Hub reports for that commit) and its row
count before anything reads it. The label names embedded in the parquet
metadata must equal the committed ``resources/intent_names.json``; a
mismatch means the integer ids no longer mean what the mapping assumes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq

from tinyrouter.labels import load_label_space

SplitName = Literal["train", "validation", "test"]

DATASET_REPO = "clinc/clinc_oos"
# Commit of https://huggingface.co/datasets/clinc/clinc_oos (2024-01-18),
# checked 2026-09-23 against the Hub API.
DATASET_REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"


@dataclass(frozen=True)
class SplitFile:
    path: str
    sha256: str
    rows: int


SPLIT_FILES: dict[str, SplitFile] = {
    "train": SplitFile(
        "plus/train-00000-of-00001.parquet",
        "30188119cf9f86fc9db27e1c22442d091cb5cb0913c9496f945fe11e7a02a28f",
        15_250,
    ),
    "validation": SplitFile(
        "plus/validation-00000-of-00001.parquet",
        "fbd545b46c611c4a7ba4b48cae6c7f09bb5b59f33ff56206ad1cd366c85cdfaa",
        3_100,
    ),
    "test": SplitFile(
        "plus/test-00000-of-00001.parquet",
        "3e60e45b25bf86543aa5df8ba4fcc674114164e6184f0197690648c2908d0102",
        5_500,
    ),
}


class DataIntegrityError(RuntimeError):
    """A downloaded file does not match its pinned checksum, row count, or label names."""


@dataclass(frozen=True)
class Split:
    """One dataset split. ``name`` travels with the data so downstream code can refuse test."""

    name: SplitName
    texts: tuple[str, ...]
    intents: np.ndarray

    def __len__(self) -> int:
        return len(self.texts)

    def take(self, indices: np.ndarray) -> Split:
        idx = np.asarray(indices, dtype=np.int64)
        return Split(self.name, tuple(self.texts[i] for i in idx), self.intents[idx])


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_split_file(split: SplitName) -> Path:
    """Fetch one split's parquet at the pinned revision and verify its SHA-256."""
    from huggingface_hub import hf_hub_download

    spec = SPLIT_FILES[split]
    local = Path(
        hf_hub_download(DATASET_REPO, spec.path, repo_type="dataset", revision=DATASET_REVISION)
    )
    actual = sha256_of(local)
    if actual != spec.sha256:
        raise DataIntegrityError(
            f"{spec.path}: sha256 {actual} != pinned {spec.sha256}. "
            f"Delete the cached file ({local}) and retry; if it persists the pin is wrong."
        )
    return local


def read_split_file(path: Path, split: SplitName) -> Split:
    """Read a verified parquet file into a Split, checking rows and embedded label names."""
    table = pq.read_table(path)
    spec = SPLIT_FILES[split]
    if table.num_rows != spec.rows:
        raise DataIntegrityError(f"{split}: {table.num_rows} rows, expected {spec.rows}")
    metadata = table.schema.metadata or {}
    if b"huggingface" not in metadata:
        raise DataIntegrityError(f"{path}: no Hugging Face feature metadata")
    features = json.loads(metadata[b"huggingface"])["info"]["features"]
    embedded_names = tuple(features["intent"]["names"])
    if embedded_names != load_label_space().intent_names:
        raise DataIntegrityError(
            f"{split}: label names in the parquet differ from resources/intent_names.json"
        )
    texts = tuple(str(t) for t in table.column("text").to_pylist())
    intents = np.asarray(table.column("intent").to_pylist(), dtype=np.int64)
    return Split(split, texts, intents)


def load_split(split: SplitName) -> Split:
    return read_split_file(download_split_file(split), split)


def subsample_per_intent(split: Split, per_intent: int | None, seed: int) -> Split:
    """Keep at most ``per_intent`` rows of every intent, oos included.

    ``None`` keeps everything. Selection depends only on ``seed`` and the
    split's content, and the result keeps the original row order.
    """
    if per_intent is None:
        return split
    if per_intent < 1:
        raise ValueError(f"per_intent must be >= 1 or None, got {per_intent}")
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for intent in np.unique(split.intents):
        rows = np.flatnonzero(split.intents == intent)
        if len(rows) > per_intent:
            rows = rng.choice(rows, size=per_intent, replace=False)
        keep.append(rows)
    return split.take(np.sort(np.concatenate(keep)))
