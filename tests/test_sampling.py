import math
from collections import Counter

import numpy as np
import pytest

from tinyrouter.data import Split
from tinyrouter.labels import load_label_space
from tinyrouter.sampling import (
    CURVE_KS,
    OOS_TRAIN_ROWS,
    SamplingError,
    curve_sample,
    planned_rows,
    sample_fingerprint,
    sample_k_shot,
)

OOS = load_label_space().oos_intent_id


def full_train() -> Split:
    """Shaped like the OOS+ train split: 100 rows per in-scope intent, 250 oos, interleaved."""
    rng = np.random.default_rng(0)
    intents = np.array([i for i in range(151) if i != OOS for _ in range(100)] + [OOS] * 250)
    intents = intents[rng.permutation(len(intents))]
    texts = tuple(f"row{n} intent{i}" for n, i in enumerate(intents))
    return Split("train", texts, intents.astype(np.int64))


FULL = full_train()


def test_table_is_ceil_of_two_and_a_half_k():
    assert {k: math.ceil(2.5 * k) for k in CURVE_KS} == OOS_TRAIN_ROWS
    assert CURVE_KS == (1, 5, 10, 25, 50, 100)


@pytest.mark.parametrize("k", CURVE_KS)
def test_every_in_scope_intent_gets_exactly_k_rows_and_oos_the_table_count(k):
    counts = Counter(sample_k_shot(FULL, k, seed=42).intents.tolist())
    assert counts.pop(OOS) == OOS_TRAIN_ROWS[k]
    assert len(counts) == 150 and set(counts.values()) == {k}


@pytest.mark.parametrize("k", CURVE_KS)
def test_planned_rows_match_the_drawn_sample(k):
    sample = sample_k_shot(FULL, k, seed=7)
    assert planned_rows(k) == (len(sample), int((sample.intents == OOS).sum()))


def test_k_100_is_the_whole_train_split_in_its_original_order():
    sample = sample_k_shot(FULL, 100, seed=43)
    assert Counter(zip(sample.texts, sample.intents.tolist(), strict=True)) == Counter(
        zip(FULL.texts, FULL.intents.tolist(), strict=True)
    )
    assert sample.texts == FULL.texts
    assert sample_fingerprint(sample) == sample_fingerprint(FULL)


@pytest.mark.parametrize("k", [1, 5, 10, 25, 50])
def test_same_seed_draws_the_same_rows_and_another_seed_different_ones(k):
    a, b = sample_k_shot(FULL, k, seed=42), sample_k_shot(FULL, k, seed=42)
    assert a.texts == b.texts
    assert sample_k_shot(FULL, k, seed=43).texts != a.texts


def test_selected_rows_keep_their_original_order():
    sample = sample_k_shot(FULL, 5, seed=42)
    positions = [FULL.texts.index(t) for t in sample.texts]
    assert positions == sorted(positions)


@pytest.mark.parametrize("k", [0, 2, 20, 99, 101, -1])
def test_k_not_in_the_table_is_refused(k):
    with pytest.raises(SamplingError, match="not a curve point"):
        sample_k_shot(FULL, k, seed=42)


def test_oos_override_zero_drops_every_oos_row_for_the_ablation():
    sample = sample_k_shot(FULL, 100, seed=42, oos_override=0)
    assert OOS not in sample.intents.tolist()
    assert len(sample) == 15_000
    assert planned_rows(100, 0) == (15_000, 0)


def test_negative_oos_override_is_refused():
    with pytest.raises(SamplingError, match=">= 0"):
        sample_k_shot(FULL, 5, seed=42, oos_override=-1)


def test_only_the_train_split_is_sampled():
    with pytest.raises(SamplingError, match="only the train split"):
        sample_k_shot(Split("test", FULL.texts, FULL.intents), 5, seed=42)


def test_a_split_missing_an_intent_is_refused():
    keep = np.flatnonzero(FULL.intents != 3)
    with pytest.raises(SamplingError, match="distinct intents"):
        sample_k_shot(FULL.take(keep), 5, seed=42)


def test_an_intent_with_too_few_rows_is_refused():
    rows_of_7 = np.flatnonzero(FULL.intents == 7)
    keep = np.setdiff1d(np.arange(len(FULL)), rows_of_7[:95])
    with pytest.raises(SamplingError, match="intent 7 has 5 rows, need 10"):
        sample_k_shot(FULL.take(keep), 10, seed=42)


def test_curve_sample_reads_the_train_split(monkeypatch):
    import tinyrouter.sampling as sampling

    asked: list[str] = []
    monkeypatch.setattr(sampling, "load_split", lambda name: asked.append(name) or FULL)
    assert curve_sample(5, 42).texts == sample_k_shot(FULL, 5, 42).texts
    assert asked == ["train"]


@pytest.mark.network
def test_on_the_real_train_split_k_100_is_everything_and_k_5_is_balanced():
    from tinyrouter.data import load_split

    real = load_split("train")
    assert sample_k_shot(real, 100, seed=42).texts == real.texts
    counts = Counter(sample_k_shot(real, 5, seed=42).intents.tolist())
    assert counts.pop(OOS) == 13 and set(counts.values()) == {5}


def rng_free_train() -> Split:
    """OOS+-shaped split built without any random generator, so only the sampler draws."""
    in_scope = [i for i in range(151) if i != OOS]
    intents = [i for r in range(100) for i in in_scope]
    for n in range(250):
        intents.insert(n * 61, OOS)
    texts = tuple(f"q{n}-{i}" for n, i in enumerate(intents))
    return Split("train", texts, np.array(intents, dtype=np.int64))


# SHA-256 of sample_fingerprint on rng_free_train(), numpy 2.5.3 (uv.lock). If
# these change, numpy's Generator stream changed: every k < 100 sample differs
# from the one the recorded runs trained on. Do not update the values to make
# the test pass; the curves must be rerun (curves.py also refuses to index such
# runs, by comparing each run's recorded fingerprint with curve_sample now).
GOLDEN_FINGERPRINTS = {
    (1, 42): "73b3054b10915c25150fbaf12f841a8dbf9b7d501050100c9b1c1b8ffed9b9eb",
    (5, 42): "bcb7915540c4392dbb999b056e7e0e90deeeaa07260b45850b36447f8549cfc1",
    (5, 43): "8616df6a42cf0d60893c7b32f64cbd9bcfc99d645c57df25c150442c89e86bfd",
    (25, 44): "ebd8fa4ebcaa263c378315dff1a627bf6d49dd53215a0d6d3b4a159266dca1cb",
    (50, 42): "7829d6e72d2af9c9a602787ccf1890c19a9d6ddacaff98762c9eaadde178a5b1",
}


@pytest.mark.parametrize(("k", "seed"), sorted(GOLDEN_FINGERPRINTS))
def test_samples_are_pinned_across_numpy_releases(k, seed):
    sample = sample_k_shot(rng_free_train(), k, seed)
    assert sample_fingerprint(sample) == GOLDEN_FINGERPRINTS[(k, seed)]


def test_the_rng_free_split_is_oos_plus_shaped():
    split = rng_free_train()
    counts = Counter(split.intents.tolist())
    assert counts.pop(OOS) == 250 and set(counts.values()) == {100}
