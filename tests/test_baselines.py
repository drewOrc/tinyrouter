import json
from collections import Counter

import numpy as np
import pytest

from test_sampling import FULL
from tinyrouter import baselines
from tinyrouter.archive import check_against_manifest, load_logits
from tinyrouter.config import RunConfig
from tinyrouter.data import Split
from tinyrouter.evaluate import RunPaths, score
from tinyrouter.labels import AGENTS, load_label_space
from tinyrouter.sampling import OOS_TRAIN_ROWS, sample_fingerprint, sample_k_shot

pytestmark = pytest.mark.filterwarnings("ignore::tinyrouter.calibrate.TemperatureBoundWarning")

LABELS = load_label_space()
OOS = LABELS.oos_intent_id


def worded(split: Split) -> Split:
    """Give each intent its own vocabulary so a text is closest to its own intent's centroid."""
    texts = tuple(f"w{i}a w{i}b common n{n % 7}" for n, i in enumerate(split.intents))
    return Split(split.name, texts, split.intents)


TRAIN = worded(FULL)


def eval_split(name: str) -> Split:
    """Validation or test: every intent twice, oos five times."""
    intents = np.array([i for i in range(151) for _ in range(5 if i == OOS else 2)])
    return worded(Split(name, tuple("" for _ in intents), intents))  # type: ignore[arg-type]


EVALS = {name: eval_split(name) for name in ("validation", "test")}


def test_majority_scores_are_the_same_prior_for_every_row():
    train = sample_k_shot(TRAIN, 5, seed=42)
    scores = baselines.majority_scores(train, 4)
    assert scores.shape == (4, 151) and scores.dtype == np.float32
    assert (scores == scores[0]).all()
    assert np.exp(scores[0].astype(np.float64)).sum() == pytest.approx(1.0, rel=1e-6)
    assert np.exp(scores[0, OOS]) == pytest.approx(OOS_TRAIN_ROWS[5] / len(train), rel=1e-6)


def test_majority_argmax_is_oos_and_its_agent_sum_is_finance():
    scores = baselines.majority_scores(sample_k_shot(TRAIN, 10, seed=42), 1)
    assert scores.argmax() == OOS
    probs = np.exp(scores.astype(np.float64))
    assert AGENTS[LABELS.aggregate_probs(probs).argmax()] == "finance_agent"


def test_majority_refuses_a_sample_missing_an_intent():
    train = sample_k_shot(TRAIN, 5, seed=42)
    keep = np.flatnonzero(train.intents != 9)
    with pytest.raises(ValueError, match=r"\[9\]"):
        baselines.majority_scores(train.take(keep), 1)


def test_tfidf_centroid_puts_each_query_nearest_its_own_intent():
    model = baselines.TfidfCentroid(sample_k_shot(TRAIN, 5, seed=42))
    scores = model.scores(EVALS["validation"].texts)
    assert (scores.argmax(axis=1) == EVALS["validation"].intents).all()
    assert scores.max() <= baselines.COSINE_SCALE + 1e-3 and scores.min() >= 0


def test_tfidf_is_fit_on_the_training_sample_only():
    model = baselines.TfidfCentroid(sample_k_shot(TRAIN, 1, seed=42))
    unseen = model.scores(("zzz_never_seen_in_training",))
    assert (unseen == 0).all()


def test_baseline_run_is_archived_and_scored_like_an_encoder_run(tmp_path):
    train = sample_k_shot(TRAIN, 5, seed=43)
    record = baselines.run_baseline("tfidf-centroid", 5, 43, train, EVALS, tmp_path)
    paths = RunPaths.named(tmp_path, "tfidf-centroid-k5-seed43")
    check_against_manifest(paths.manifest, paths.logits)
    archive = load_logits(paths.logits)
    meta = archive.metadata
    assert (meta["model_name"], meta["k_shot"], meta["seed"]) == ("baseline/tfidf-centroid", 5, 43)
    assert meta["train_rows"] == 150 * 5 + 13 and meta["oos_train_rows"] == 13
    assert archive.validation.logits.shape == (len(EVALS["validation"]), 151)
    expected = score(archive.validation, archive.test)
    assert set(record["metrics"]) == set(expected)
    assert set(record["metrics"]["test"]["raw"]) == set(expected["test"]["raw"])
    assert (
        json.loads(paths.results_json.read_text())["logits"]["sha256"] == record["logits"]["sha256"]
    )


def test_baseline_rerun_on_the_same_sample_does_nothing(tmp_path, monkeypatch):
    train = sample_k_shot(TRAIN, 5, seed=42)
    baselines.run_baseline("majority", 5, 42, train, EVALS, tmp_path)

    def refit(*_):
        raise AssertionError("an intact run with the same sample must not be recomputed")

    monkeypatch.setattr(baselines, "baseline_logits", refit)
    baselines.run_baseline("majority", 5, 42, train, EVALS, tmp_path)
    other = sample_k_shot(TRAIN, 5, seed=44)
    with pytest.raises(AssertionError, match="must not be recomputed"):
        baselines.run_baseline("majority", 5, 42, other, EVALS, tmp_path)


def test_baselines_see_exactly_the_rows_the_encoder_of_that_point_trains_on(tmp_path, monkeypatch):
    import tinyrouter.sampling as sampling
    from tinyrouter.train import prepare_train_split

    monkeypatch.setattr(sampling, "load_split", lambda name: TRAIN)
    body = baselines.run_all(tmp_path, eval_fn=EVALS.__getitem__, log=lambda _: None)
    assert len(body["points"]) == 6 * 3 * 2
    for point in body["points"]:
        encoder = RunConfig(
            model_name="m", model_revision="r", k_shot=point["k"], seed=point["seed"]
        )
        assert point["train_sample_sha256"] == sample_fingerprint(prepare_train_split(encoder))
    counts = Counter((p["baseline"], p["k"]) for p in body["points"])
    assert set(counts.values()) == {3}
    shas = [p["logits_sha256"] for p in body["points"]]
    assert len(set(shas)) == len(shas)
    assert (tmp_path / "curves" / "baselines.json").exists()


def test_unknown_baseline_is_refused():
    with pytest.raises(ValueError, match="unknown baseline"):
        baselines.baseline_logits("knn", TRAIN, EVALS)


def test_temperature_scaling_undoes_the_cosine_scale():
    """1x and 100x cosines, each with its own validation-fitted T, give the same probabilities."""
    from tinyrouter.calibrate import SplitLogits, apply_temperature, fit_temperature

    rng = np.random.default_rng(0)
    n = 400
    gold = rng.integers(0, 151, size=n)
    cosine = rng.uniform(0.0, 0.35, size=(n, 151))
    cosine[np.arange(n), gold] += rng.uniform(0.0, 0.1, size=n)
    plain = SplitLogits("validation", cosine, gold)
    scaled = SplitLogits("validation", baselines.COSINE_SCALE * cosine, gold)
    t_plain, t_scaled = fit_temperature(plain), fit_temperature(scaled)
    assert 0.05 < t_plain < 1 and t_scaled == pytest.approx(100 * t_plain, rel=1e-4)
    np.testing.assert_allclose(
        apply_temperature(plain.logits, t_plain),
        apply_temperature(scaled.logits, t_scaled),
        rtol=1e-4,
        atol=1e-8,
    )
