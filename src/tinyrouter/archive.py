"""Per-example logits archive: the only input the analysis steps read (docs/PLAN.md AC3).

One ``.npz`` per run holds validation and test logits, gold intent ids,
and a JSON metadata block. Logits are stored as float32, the dtype the
model produced them in, so nothing downstream (temperature fit, ECE,
threshold sweeps, risk-coverage ranking) is computed on rounded numbers.

Archives live in ``results/logits/`` and are not committed. Their SHA-256
is recorded in ``results/logits-manifest.json``, which is committed, so a
copy downloaded from a GitHub Release can be checked byte for byte against
what the run produced.

Format versions (``format_version`` in the metadata):

- 1 (AC2 archives, 2026-09-23): no per-file dataset checksums, only
  ``dataset_revision``.
- 2: adds ``dataset_sha256``, the SHA-256 of the train, validation and
  test parquet files the run read (each verified against ``SPLIT_FILES``
  before use). Only format 2 is written.

``load_logits`` still reads format 1. It fills ``dataset_sha256`` from the
pinned ``SPLIT_FILES`` when the archive's ``dataset_revision`` is the
revision this code pins (the files at a revision are immutable, and
``download_split_file`` refuses any other bytes), marks the value as
back-filled, and refuses a format-1 archive from any other revision.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tinyrouter.calibrate import SplitLogits
from tinyrouter.data import DATASET_REVISION, SPLIT_FILES, SplitName, sha256_of
from tinyrouter.labels import load_label_space

FORMAT_VERSION = 2
READABLE_FORMAT_VERSIONS = (1, 2)
# The manifest's own layout has not changed since it was introduced.
MANIFEST_FORMAT_VERSION = 1
DATASET_SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")
BACKFILL_NOTE = "back-filled from SPLIT_FILES when read (format 1 archive)"
SPLITS: tuple[SplitName, ...] = ("validation", "test")
LOGITS_DTYPE = np.float32
MANIFEST_NAME = "logits-manifest.json"

# Every key must be present. Values may be None only where listed in NULLABLE.
REQUIRED_METADATA: dict[str, type | tuple[type, ...]] = {
    "format_version": int,
    "run_name": str,
    "model_name": str,
    "model_revision": str,
    "seed": int,
    "per_intent": int,
    "train_rows": int,
    "oos_train_rows": int,
    "eval_per_intent": int,
    "dataset_revision": str,
    "label_space_sha256": str,
    "git_commit": str,
    "git_dirty": bool,
    "created_at": str,
}
NULLABLE = {"per_intent", "eval_per_intent"}
# Added in format 2: file checksums, and the k-shot size (null unless a curve run).
DATASET_SHA_KEY = "dataset_sha256"
K_SHOT_KEY = "k_shot"


def pinned_dataset_sha256() -> dict[str, str]:
    """SHA-256 of each split's parquet file at the pinned revision."""
    return {name: SPLIT_FILES[name].sha256 for name in DATASET_SPLITS}


class ArchiveError(ValueError):
    """A logits archive is incomplete, inconsistent, or does not match the manifest."""


@dataclass(frozen=True)
class LogitsArchive:
    metadata: dict[str, object]
    splits: dict[str, SplitLogits]

    @property
    def validation(self) -> SplitLogits:
        return self.splits["validation"]

    @property
    def test(self) -> SplitLogits:
        return self.splits["test"]


def git_state(repo_dir: Path | None = None) -> tuple[str, bool]:
    """Current commit and whether tracked files outside ``results/`` have uncommitted changes.

    ``results/`` is excluded because runs rewrite it themselves (the
    manifest, results JSON); counting that would mark every later run dirty.
    Returns ``("unknown", True)`` outside a git checkout, so the archive
    still says it cannot be traced to a commit instead of omitting the key.
    """
    cwd = repo_dir or Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ":(top,exclude)results"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return commit, bool(status.strip())


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def validate_metadata(metadata: dict[str, object]) -> None:
    """Refuse metadata that is not a complete, current (format 2) record."""
    validate_common_metadata(metadata)
    if metadata["format_version"] != FORMAT_VERSION:
        raise ArchiveError(
            f"format_version {metadata['format_version']} cannot be written (current is "
            f"{FORMAT_VERSION})"
        )
    validate_format2_fields(metadata)


def validate_format2_fields(metadata: dict[str, object]) -> None:
    if K_SHOT_KEY not in metadata:
        raise ArchiveError(f"metadata is missing [{K_SHOT_KEY!r}]")
    k_shot = metadata[K_SHOT_KEY]
    if k_shot is not None and (isinstance(k_shot, bool) or not isinstance(k_shot, int)):
        raise ArchiveError(f"metadata[{K_SHOT_KEY!r}] = {k_shot!r} is not an int or null")
    validate_dataset_sha256(metadata)


def validate_dataset_sha256(metadata: dict[str, object]) -> None:
    recorded = metadata.get(DATASET_SHA_KEY)
    if not isinstance(recorded, dict) or set(recorded) != set(DATASET_SPLITS):
        raise ArchiveError(
            f"metadata[{DATASET_SHA_KEY!r}] must map exactly {DATASET_SPLITS} to SHA-256 digests"
        )
    if metadata["dataset_revision"] != DATASET_REVISION or recorded != pinned_dataset_sha256():
        raise ArchiveError(
            f"dataset (revision {metadata['dataset_revision']}, files {recorded}) is not the "
            f"pinned CLINC150 revision {DATASET_REVISION}; the analysis code assumes those files"
        )


def upgrade_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Metadata of a readable archive in the current format; format 1 gets dataset_sha256."""
    validate_common_metadata(metadata)
    version = metadata["format_version"]
    if version == FORMAT_VERSION:
        validate_format2_fields(metadata)
        return metadata
    for key in (DATASET_SHA_KEY, K_SHOT_KEY):
        if key in metadata:
            raise ArchiveError(f"format {version} archive already carries {key!r}")
    if metadata["dataset_revision"] != DATASET_REVISION:
        raise ArchiveError(
            f"format {version} archive from dataset revision {metadata['dataset_revision']}; "
            f"its file checksums can only be back-filled for the pinned revision "
            f"{DATASET_REVISION}"
        )
    return {
        **metadata,
        # Format 1 predates k-shot sampling: those runs used per_intent only.
        K_SHOT_KEY: None,
        DATASET_SHA_KEY: pinned_dataset_sha256(),
        "dataset_sha256_source": BACKFILL_NOTE,
    }


def validate_common_metadata(metadata: dict[str, object]) -> None:
    missing = sorted(set(REQUIRED_METADATA) - set(metadata))
    if missing:
        raise ArchiveError(f"metadata is missing {missing}")
    for key, expected in REQUIRED_METADATA.items():
        value = metadata[key]
        if value is None and key in NULLABLE:
            continue
        # bool is a subclass of int; do not let True pass as a seed.
        wrong_bool = isinstance(value, bool) and expected is not bool
        if wrong_bool or not isinstance(value, expected):
            raise ArchiveError(f"metadata[{key!r}] = {value!r} is not {expected}")
        if isinstance(value, str) and not value.strip():
            raise ArchiveError(f"metadata[{key!r}] is empty")
    if metadata["format_version"] not in READABLE_FORMAT_VERSIONS:
        raise ArchiveError(
            f"format_version {metadata['format_version']} is not supported (readable: "
            f"{READABLE_FORMAT_VERSIONS})"
        )
    expected_hash = load_label_space().sha256
    if metadata["label_space_sha256"] != expected_hash:
        raise ArchiveError(
            "label space fingerprint differs from resources/*.json; the logit columns in this "
            "archive may not mean what the current code assumes"
        )


def validate_split(split: SplitLogits) -> None:
    num_intents = load_label_space().num_intents
    logits, labels = split.logits, split.labels
    if logits.shape[1] != num_intents:
        raise ArchiveError(
            f"{split.split}: {logits.shape[1]} logit columns, expected {num_intents}"
        )
    if logits.dtype != LOGITS_DTYPE:
        raise ArchiveError(f"{split.split}: logits are {logits.dtype}, expected {LOGITS_DTYPE}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ArchiveError(f"{split.split}: labels are {labels.dtype}, expected integers")
    if labels.size and (labels.min() < 0 or labels.max() >= num_intents):
        raise ArchiveError(f"{split.split}: label ids outside [0, {num_intents})")
    if not np.isfinite(logits).all():
        raise ArchiveError(f"{split.split}: logits contain NaN or inf")


def save_logits(path: Path, splits: dict[str, SplitLogits], metadata: dict[str, object]) -> Path:
    """Validate and write an archive atomically (a crash never leaves a half-written file)."""
    if set(splits) != set(SPLITS):
        raise ArchiveError(f"need exactly the splits {SPLITS}, got {sorted(splits)}")
    arrays: dict[str, np.ndarray] = {}
    for name in SPLITS:
        split = splits[name]
        if split.split != name:
            raise ArchiveError(f"splits[{name!r}] holds the '{split.split}' split")
        stored = SplitLogits(name, split.logits.astype(LOGITS_DTYPE), split.labels.astype(np.int64))
        validate_split(stored)
        arrays[f"{name}_logits"] = stored.logits
        arrays[f"{name}_labels"] = stored.labels
    validate_metadata(metadata)
    arrays["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)
    return path


def load_logits(path: Path) -> LogitsArchive:
    """Read an archive back, refusing it if metadata or shapes are incomplete or inconsistent.

    The returned metadata is in the current format (see ``upgrade_metadata``).
    """
    expected_keys = {"metadata"} | {f"{s}_{part}" for s in SPLITS for part in ("logits", "labels")}
    with np.load(path, allow_pickle=False) as npz:
        missing = sorted(expected_keys - set(npz.files))
        if missing:
            raise ArchiveError(f"{path}: arrays missing {missing}")
        metadata = json.loads(str(npz["metadata"]))
        arrays = {key: npz[key] for key in expected_keys - {"metadata"}}
    if not isinstance(metadata, dict):
        raise ArchiveError(f"{path}: metadata is not a JSON object")
    metadata = upgrade_metadata(metadata)
    splits: dict[str, SplitLogits] = {}
    for name in SPLITS:
        logits, labels = arrays[f"{name}_logits"], arrays[f"{name}_labels"]
        if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
            raise ArchiveError(
                f"{path}: {name} logits {logits.shape} and labels {labels.shape} do not line up"
            )
        split = SplitLogits(name, logits, labels)
        validate_split(split)
        splits[name] = split
    return LogitsArchive(metadata=metadata, splits=splits)


def load_validation_logits(path: Path) -> tuple[dict[str, object], SplitLogits]:
    """Metadata and the validation split only; the test arrays are never read.

    For the pilots, which choose hyperparameters and must not touch any
    test output (docs/PLAN.md section 4, hyperparameter protocol). An
    ``.npz`` is a zip, so reading one member leaves the others unread.
    """
    with np.load(path, allow_pickle=False) as npz:
        needed = {"metadata", "validation_logits", "validation_labels"}
        missing = sorted(needed - set(npz.files))
        if missing:
            raise ArchiveError(f"{path}: arrays missing {missing}")
        metadata = json.loads(str(npz["metadata"]))
        logits, labels = npz["validation_logits"], npz["validation_labels"]
    if not isinstance(metadata, dict):
        raise ArchiveError(f"{path}: metadata is not a JSON object")
    metadata = upgrade_metadata(metadata)
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ArchiveError(f"{path}: validation logits {logits.shape}, labels {labels.shape}")
    split = SplitLogits("validation", logits, labels)
    validate_split(split)
    return metadata, split


def read_manifest(manifest_path: Path) -> dict[str, dict[str, object]]:
    if not manifest_path.exists():
        return {}
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return dict(raw["files"])


def record_in_manifest(manifest_path: Path, archive_path: Path) -> dict[str, object]:
    """Add or replace the entry for ``archive_path``; keys are file names relative to its dir."""
    archive = load_logits(archive_path)
    entry = {
        "sha256": sha256_of(archive_path),
        "bytes": archive_path.stat().st_size,
        "run_name": archive.metadata["run_name"],
        "git_commit": archive.metadata["git_commit"],
        "created_at": archive.metadata["created_at"],
        "rows": {name: int(split.labels.shape[0]) for name, split in archive.splits.items()},
    }
    files = read_manifest(manifest_path)
    files[archive_path.name] = entry
    write_manifest(manifest_path, files)
    return entry


def write_manifest(manifest_path: Path, files: dict[str, dict[str, object]]) -> None:
    body = {"format_version": MANIFEST_FORMAT_VERSION, "files": dict(sorted(files.items()))}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)


def remove_from_manifest(manifest_path: Path, archive_name: str) -> bool:
    """Drop one entry; returns whether it was there."""
    files = read_manifest(manifest_path)
    if files.pop(archive_name, None) is None:
        return False
    write_manifest(manifest_path, files)
    return True


def check_against_manifest(manifest_path: Path, archive_path: Path) -> None:
    """Raise unless the manifest lists ``archive_path`` with this exact SHA-256."""
    entry = read_manifest(manifest_path).get(archive_path.name)
    if entry is None:
        raise ArchiveError(f"{archive_path.name} is not listed in {manifest_path}")
    actual = sha256_of(archive_path)
    if actual != entry["sha256"]:
        raise ArchiveError(
            f"{archive_path.name}: sha256 {actual} != manifest {entry['sha256']}; the file "
            "changed after it was recorded, or it is a different run's archive"
        )


def verify_all(manifest_path: Path, logits_dir: Path) -> list[str]:
    """Problems found checking every manifest entry against the files in ``logits_dir``."""
    problems = []
    for name in read_manifest(manifest_path):
        path = logits_dir / name
        if not path.exists():
            problems.append(f"{name}: listed in the manifest but not in {logits_dir}")
            continue
        try:
            check_against_manifest(manifest_path, path)
            load_logits(path)
        except ArchiveError as exc:
            problems.append(str(exc))
    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify logits archives against the manifest.")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args(argv)
    root = Path(args.results_root)
    manifest = root / MANIFEST_NAME
    problems = verify_all(manifest, root / "logits")
    for problem in problems:
        print(f"FAIL {problem}")
    count = len(read_manifest(manifest))
    if problems:
        raise SystemExit(1)
    print(f"OK {count} archive(s) match {manifest}")


if __name__ == "__main__":
    main()
