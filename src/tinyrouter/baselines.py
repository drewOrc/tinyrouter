"""Cheap baselines on every curve point, stored exactly like an encoder run (docs/PLAN.md AC5).

For each k and seed, both baselines are fit on ``sampling.curve_sample(k,
seed)``, the same rows the encoders train on (the results JSON records the
sample's fingerprint, comparable with an encoder's ``train_sample_sha256``).
Each run writes a logits archive of 151 scores per validation and test
row, an entry in the logits manifest, and ``results/runs/<run>.json`` with
the same metrics block as an encoder, so the step-4 analysis reads all of
them the same way. How the scores are defined:

- ``majority``: the log of each intent's share of the training sample,
  the same 151 numbers for every query (it never looks at the text). Its
  argmax is the most frequent intent, which is ``oos`` (ceil(2.5k) rows
  against k for every other intent). Summed per agent it is the most
  frequent agent, ``finance_agent`` (38 of the 150 intents): that is the
  8-way majority class, so ``accuracy_8_summed`` is the majority-class
  accuracy and ``accuracy_8`` (argmax intent, then its agent) always says
  oos. Its softmax is the training class distribution, a calibrated
  prior; the temperature fit on validation may still move it.
- ``tfidf-centroid``: cosine similarity between the query's TF-IDF vector
  and each intent's centroid. The vectorizer (word unigrams and bigrams,
  sublinear tf, L2-normalised rows) is fit on the training sample only;
  a centroid is the mean of its intent's training vectors, L2-normalised.
  The score is ``COSINE_SCALE`` (100) times the cosine, the convention
  CLIP uses to turn cosines into logits. The factor changes no argmax;
  it only moves the scores to a range where the validation temperature
  fit (search range T in [0.05, 100]) has an interior optimum. On the
  raw cosines (range [0, 1]) the fitted T hit the lower bound at every
  k >= 5 (checked on seed 42 before this was written). A query with no
  known term scores 0 everywhere and its argmax is intent 0.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from tinyrouter.archive import (
    FORMAT_VERSION,
    git_state,
    load_logits,
    pinned_dataset_sha256,
    record_in_manifest,
    save_logits,
    utc_now,
)
from tinyrouter.calibrate import SplitLogits
from tinyrouter.data import DATASET_REVISION, Split, load_split
from tinyrouter.evaluate import RunPaths, score
from tinyrouter.labels import load_label_space
from tinyrouter.protocol import SEEDS
from tinyrouter.runs import Log, archive_intact, read_record
from tinyrouter.sampling import CURVE_KS, curve_sample, sample_fingerprint

# Bump when a definition below changes; a stored run with another version is redone.
BASELINE_VERSION = "1"
TFIDF_PARAMS: dict[str, object] = {"ngram_range": (1, 2), "sublinear_tf": True}
COSINE_SCALE = 100.0
DEFINITIONS = {
    "majority": "log share of each intent in the training sample, same for every query",
    "tfidf-centroid": "100 * cosine(query tf-idf, L2-normalised intent centroid); vectorizer "
    "fit on the training sample, word 1-2 grams, sublinear tf",
}
BASELINES = tuple(DEFINITIONS)
SampleFn = Callable[[int, int], Split]
EvalFn = Callable[[str], Split]


def majority_scores(train: Split, rows: int) -> np.ndarray:
    counts = np.bincount(train.intents, minlength=load_label_space().num_intents)
    if (counts == 0).any():
        raise ValueError(f"intents {np.flatnonzero(counts == 0).tolist()} have no training rows")
    prior = np.log(counts / counts.sum())
    return np.tile(prior, (rows, 1)).astype(np.float32)


class TfidfCentroid:
    """Nearest-centroid scorer over TF-IDF vectors, fit on the training sample only."""

    def __init__(self, train: Split) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        self.vectorizer = TfidfVectorizer(**TFIDF_PARAMS)  # type: ignore[arg-type]
        vectors = self.vectorizer.fit_transform(train.texts)
        num_intents = load_label_space().num_intents
        centroids = np.zeros((num_intents, vectors.shape[1]))
        for intent in range(num_intents):
            rows = np.flatnonzero(train.intents == intent)
            if len(rows) == 0:
                raise ValueError(f"intent {intent} has no training rows")
            centroids[intent] = np.asarray(vectors[rows].mean(axis=0)).ravel()
        self.centroids = normalize(centroids)

    def scores(self, texts: tuple[str, ...]) -> np.ndarray:
        cosine = self.vectorizer.transform(texts) @ self.centroids.T
        return (COSINE_SCALE * np.asarray(cosine)).astype(np.float32)


def baseline_logits(name: str, train: Split, evals: dict[str, Split]) -> dict[str, SplitLogits]:
    if name == "majority":
        make = lambda split: majority_scores(train, len(split))  # noqa: E731
    elif name == "tfidf-centroid":
        model = TfidfCentroid(train)
        make = lambda split: model.scores(split.texts)  # noqa: E731
    else:
        raise ValueError(f"unknown baseline '{name}'; expected one of {BASELINES}")
    return {n: SplitLogits(n, make(split), split.intents) for n, split in evals.items()}  # type: ignore[arg-type]


def run_name(name: str, k: int, seed: int) -> str:
    return f"{name}-k{k}-seed{seed}"


def definition(name: str) -> dict[str, object]:
    params = {k: list(v) if isinstance(v, tuple) else v for k, v in TFIDF_PARAMS.items()}
    return {
        "name": name,
        "version": BASELINE_VERSION,
        "scores": DEFINITIONS[name],
        "params": params if name == "tfidf-centroid" else {},
    }


def training_block(train: Split, k: int, seed: int) -> dict[str, object]:
    oos = load_label_space().oos_intent_id
    return {
        "seed": seed,
        "k_shot": k,
        "train_rows": len(train),
        "oos_train_rows": int((train.intents == oos).sum()),
        "train_sample_sha256": sample_fingerprint(train),
    }


def metadata(name: str, k: int, seed: int, training: dict[str, object]) -> dict[str, object]:
    commit, dirty = git_state()
    return {
        "format_version": FORMAT_VERSION,
        "run_name": run_name(name, k, seed),
        "model_name": f"baseline/{name}",
        "model_revision": f"v{BASELINE_VERSION}",
        "seed": seed,
        "per_intent": None,
        "k_shot": k,
        "train_rows": training["train_rows"],
        "oos_train_rows": training["oos_train_rows"],
        "eval_per_intent": None,
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": pinned_dataset_sha256(),
        "label_space_sha256": load_label_space().sha256,
        "git_commit": commit,
        "git_dirty": dirty,
        "created_at": utc_now(),
    }


def is_done(paths: RunPaths, name: str, training: dict[str, object], seed: int) -> bool:
    record = read_record(paths)
    if record is None or record.get("baseline") != definition(name):
        return False
    if record.get("training") != training:
        return False
    return archive_intact(paths, record, seed)


def run_baseline(
    name: str, k: int, seed: int, train: Split, evals: dict[str, Split], results_root: Path
) -> dict[str, object]:
    """Fit, score, archive and record one baseline run; skip if an identical one is intact."""
    paths = RunPaths.named(results_root, run_name(name, k, seed))
    training = training_block(train, k, seed)
    if is_done(paths, name, training, seed):
        record = read_record(paths)
        assert record is not None
        return record
    paths.results_json.unlink(missing_ok=True)
    splits = baseline_logits(name, train, evals)
    save_logits(paths.logits, splits, metadata(name, k, seed, training))
    entry = record_in_manifest(paths.manifest, paths.logits)
    archived = load_logits(paths.logits)
    record = {
        "run_name": run_name(name, k, seed),
        "baseline": definition(name),
        "environment": environment(),
        "training": training,
        "logits": {"file": paths.logits.name, "sha256": entry["sha256"], "bytes": entry["bytes"]},
        "metrics": score(archived.validation, archived.test),
    }
    paths.results_json.parent.mkdir(parents=True, exist_ok=True)
    paths.results_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def environment() -> dict[str, object]:
    import platform

    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "dataset_revision": DATASET_REVISION,
    }


def run_all(
    results_root: Path,
    sample_fn: SampleFn = curve_sample,
    eval_fn: EvalFn = load_split,  # type: ignore[assignment]
    log: Log = print,
) -> dict[str, object]:
    """Every baseline at every (k, seed); writes results/curves/baselines.json."""
    evals = {name: eval_fn(name) for name in ("validation", "test")}
    entries = []
    for k in CURVE_KS:
        for seed in SEEDS:
            train = sample_fn(k, seed)
            for name in BASELINES:
                record = run_baseline(name, k, seed, train, evals, results_root)
                entries.append(index_entry(name, k, seed, record))
            log(f"baselines k={k} seed={seed}: done")
    body = {"baselines": {n: definition(n) for n in BASELINES}, "points": entries}
    out = results_root / "curves" / "baselines.json"
    write_index(out, body)
    return body


def index_entry(name: str, k: int, seed: int, record: dict[str, object]) -> dict[str, object]:
    training, logits = record["training"], record["logits"]
    assert isinstance(training, dict) and isinstance(logits, dict)
    return {
        "baseline": name,
        "k": k,
        "seed": seed,
        "run_name": record["run_name"],
        "logits_file": logits["file"],
        "logits_sha256": logits["sha256"],
        "train_rows": training["train_rows"],
        "oos_train_rows": training["oos_train_rows"],
        "train_sample_sha256": training["train_sample_sha256"],
    }


def write_index(path: Path, body: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args(argv)
    body = run_all(Path(args.results_root))
    print(f"wrote {len(body['points'])} baseline runs")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
