"""Completion checks for the curve, ablation and baseline indexes.

A command that exits 0 is not evidence that it ran anything: a shell
loop once handed ``make`` a single argument ``curve MODEL=bert``, make
read it as a variable assignment, ran its default target and exited 0,
and two curves were reported done with no point run. So each index
command checks its own work twice and says so in one line only when
both checks pass:

1. before running, the planned points are exactly the expected set,
   each once (``check_points``);
2. after running, the index read back from disk has exactly the expected
   set, each once, and every point's archive has the same SHA-256 on
   disk, in the manifest and in the index, and no two points share an
   archive (``publish_index``). The index is written to a temporary file
   and that file, not the in-memory body, is what gets checked, so a
   short or failed write is caught too; it is moved into place only after
   this passes.

The expected sets are written out here as literals, not derived from
``sampling.CURVE_KS`` or ``protocol.SEEDS``: a check that reads the same
constant as the loop it guards would agree with a wrong loop.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from tinyrouter.archive import MANIFEST_NAME, read_manifest
from tinyrouter.data import sha256_of
from tinyrouter.evaluate import RunPaths

EXPECTED_KS = (1, 5, 10, 25, 50, 100)
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_BASELINES = ("majority", "tfidf-centroid")
ABLATION_K = 100
ABLATION_OOS_ROWS = 0

Key = tuple[object, ...]
# Index fields that identify a point; ``oos_train`` is None on a curve and 0 on the ablation.
ENCODER_KEY = ("k", "seed", "oos_train")
BASELINE_KEY = ("baseline", "k", "seed")
CURVE_POINTS: frozenset[Key] = frozenset((k, s, None) for k in EXPECTED_KS for s in EXPECTED_SEEDS)
ABLATION_POINTS: frozenset[Key] = frozenset(
    (ABLATION_K, s, ABLATION_OOS_ROWS) for s in EXPECTED_SEEDS
)
BASELINE_POINTS: frozenset[Key] = frozenset(
    (b, k, s) for b in EXPECTED_BASELINES for k in EXPECTED_KS for s in EXPECTED_SEEDS
)


class IncompleteError(RuntimeError):
    """A planned or written set of points is not exactly the expected one."""


def check_points(keys: Iterable[Key], expected: frozenset[Key], what: str) -> None:
    """``keys`` must be ``expected`` with every key exactly once."""
    counts = Counter(keys)
    duplicated = sorted((k for k, n in counts.items() if n > 1), key=repr)
    missing = sorted(expected - set(counts), key=repr)
    unexpected = sorted(set(counts) - expected, key=repr)
    if duplicated or missing or unexpected:
        raise IncompleteError(
            f"{what}: expected {len(expected)} unique points, got {sum(counts.values())} "
            f"({len(counts)} unique); missing {missing}, duplicated {duplicated}, "
            f"unexpected {unexpected}"
        )


def check_archives(points: Sequence[dict], results_root: Path, what: str) -> None:
    """Each point has its own archive, whose SHA-256 agrees on disk, in manifest and index."""
    manifest = read_manifest(results_root / MANIFEST_NAME)
    owner: dict[object, object] = {}
    for point in points:
        paths = RunPaths.named(results_root, str(point["run_name"]))
        name = paths.logits.name
        for held in (point.get("logits_file"), point.get("logits_sha256")):
            if held in owner:
                raise IncompleteError(
                    f"{what}: {point['run_name']} and {owner[held]} share one logits archive"
                )
            owner[held] = point["run_name"]
        if point.get("logits_file") != name:
            raise IncompleteError(
                f"{what}: {point['run_name']} lists archive {point.get('logits_file')}"
            )
        if not paths.logits.exists():
            raise IncompleteError(f"{what}: {paths.logits} is missing")
        listed = manifest.get(name, {}).get("sha256")
        on_disk = sha256_of(paths.logits)
        if not point.get("logits_sha256") == listed == on_disk:
            raise IncompleteError(
                f"{what}: {name} SHA-256 differs: index {point.get('logits_sha256')}, "
                f"manifest {listed}, file {on_disk}"
            )


def verify_index(
    path: Path, key: Sequence[str], expected: frozenset[Key], results_root: Path, what: str
) -> int:
    """Read the index at ``path`` and run both checks on it; returns the number of points."""
    points = json.loads(path.read_text(encoding="utf-8"))["points"]
    check_points((tuple(p[f] for f in key) for p in points), expected, what)
    check_archives(points, results_root, what)
    return len(points)


def publish_index(
    out: Path,
    body: dict[str, object],
    key: Sequence[str],
    expected: frozenset[Key],
    results_root: Path,
    what: str,
) -> int:
    """Write ``body`` to ``out`` only if, read back from disk, it passes ``verify_index``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    try:
        count = verify_index(tmp, key, expected, results_root, what)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    return count
