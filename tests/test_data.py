import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import tinyrouter.data as data
from tinyrouter.data import (
    SPLIT_FILES,
    DataIntegrityError,
    Split,
    SplitFile,
    read_split_file,
    subsample_per_intent,
)
from tinyrouter.labels import load_label_space


def write_parquet(path: Path, texts: list[str], intents: list[int], names: tuple) -> Path:
    features = {"text": {"dtype": "string"}, "intent": {"names": list(names)}}
    meta = {b"huggingface": json.dumps({"info": {"features": features}}).encode()}
    table = pa.table({"text": texts, "intent": intents}).replace_schema_metadata(meta)
    pq.write_table(table, path)
    return path


@pytest.fixture
def three_row_validation(monkeypatch, tmp_path):
    monkeypatch.setitem(SPLIT_FILES, "validation", SplitFile("v.parquet", "unused", 3))
    return tmp_path


def test_read_split_file_returns_named_split(three_row_validation):
    names = load_label_space().intent_names
    path = write_parquet(three_row_validation / "v.parquet", ["a", "b", "c"], [0, 42, 5], names)
    split = read_split_file(path, "validation")
    assert split.name == "validation"
    assert split.texts == ("a", "b", "c")
    np.testing.assert_array_equal(split.intents, [0, 42, 5])


def test_read_split_file_rejects_wrong_row_count(three_row_validation):
    names = load_label_space().intent_names
    path = write_parquet(three_row_validation / "v.parquet", ["a", "b"], [0, 1], names)
    with pytest.raises(DataIntegrityError, match="2 rows, expected 3"):
        read_split_file(path, "validation")


def test_read_split_file_rejects_reordered_label_names(three_row_validation):
    names = list(load_label_space().intent_names)
    names[0], names[1] = names[1], names[0]
    path = write_parquet(three_row_validation / "v.parquet", ["a", "b", "c"], [0, 1, 2], names)
    with pytest.raises(DataIntegrityError, match="label names"):
        read_split_file(path, "validation")


def test_download_rejects_a_file_whose_checksum_differs(monkeypatch, tmp_path):
    tampered = tmp_path / "tampered.parquet"
    tampered.write_bytes(b"not the pinned file")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(tampered))
    with pytest.raises(DataIntegrityError, match="sha256"):
        data.download_split_file("test")


def test_split_files_pin_the_published_row_counts():
    assert {k: v.rows for k, v in SPLIT_FILES.items()} == {
        "train": 15_250,
        "validation": 3_100,
        "test": 5_500,
    }


def make_split(counts: dict[int, int]) -> Split:
    intents = np.array([i for i, c in counts.items() for _ in range(c)])
    return Split("train", tuple(f"q{j}" for j in range(len(intents))), intents)


def test_subsample_caps_each_intent_and_keeps_small_ones_whole():
    split = make_split({0: 10, 1: 3, 42: 8})
    sub = subsample_per_intent(split, 4, seed=42)
    assert dict(zip(*np.unique(sub.intents, return_counts=True), strict=True)) == {
        0: 4,
        1: 3,
        42: 4,
    }
    assert sub.name == "train"


def test_subsample_is_deterministic_per_seed_and_differs_across_seeds():
    split = make_split({0: 50, 1: 50})
    a = subsample_per_intent(split, 5, seed=42)
    b = subsample_per_intent(split, 5, seed=42)
    c = subsample_per_intent(split, 5, seed=43)
    assert a.texts == b.texts
    assert a.texts != c.texts


def test_subsample_none_returns_everything_and_zero_is_rejected():
    split = make_split({0: 3})
    assert subsample_per_intent(split, None, seed=1) is split
    with pytest.raises(ValueError):
        subsample_per_intent(split, 0, seed=1)


@pytest.mark.network
@pytest.mark.parametrize("name", ["train", "validation", "test"])
def test_hub_split_matches_pins_and_label_space(name):
    split = data.load_split(name)
    labels = load_label_space()
    assert len(split) == SPLIT_FILES[name].rows
    assert split.intents.min() >= 0 and split.intents.max() < labels.num_intents
    assert (split.intents == labels.oos_intent_id).any()
    assert len(np.unique(split.intents)) == labels.num_intents
