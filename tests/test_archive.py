import json

import numpy as np
import pytest

from archive_fakes import NUM_INTENTS, fake_metadata, fake_splits
from tinyrouter.archive import (
    REQUIRED_METADATA,
    ArchiveError,
    check_against_manifest,
    git_state,
    load_logits,
    read_manifest,
    record_in_manifest,
    save_logits,
    verify_all,
)
from tinyrouter.calibrate import SplitLogits
from tinyrouter.labels import build_label_space, load_label_space


def test_round_trip_preserves_logits_labels_and_metadata_exactly(tmp_path):
    splits, meta = fake_splits(), fake_metadata()
    path = save_logits(tmp_path / "run.npz", splits, meta)
    archive = load_logits(path)
    assert archive.metadata == meta
    for name in ("validation", "test"):
        got = archive.splits[name]
        assert got.split == name
        assert got.logits.dtype == np.float32
        np.testing.assert_array_equal(got.logits, splits[name].logits)
        np.testing.assert_array_equal(got.labels, splits[name].labels)
    assert archive.validation.logits.shape == (7, NUM_INTENTS)
    assert archive.test.logits.shape == (11, NUM_INTENTS)


def test_float64_logits_are_stored_as_float32(tmp_path):
    splits = fake_splits()
    wide = splits["test"].logits.astype(np.float64)
    splits["test"] = SplitLogits("test", wide, splits["test"].labels)
    archive = load_logits(save_logits(tmp_path / "r.npz", splits, fake_metadata()))
    assert archive.test.logits.dtype == np.float32


@pytest.mark.parametrize("key", sorted(REQUIRED_METADATA))
def test_save_refuses_metadata_missing_any_required_key(tmp_path, key):
    meta = fake_metadata()
    del meta[key]
    with pytest.raises(ArchiveError, match=key):
        save_logits(tmp_path / "r.npz", fake_splits(), meta)
    assert not (tmp_path / "r.npz").exists()


def rewrite_metadata(path, mutate) -> None:
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    meta = json.loads(str(arrays["metadata"]))
    mutate(meta)
    arrays["metadata"] = np.array(json.dumps(meta))
    with path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)


def test_load_refuses_an_archive_whose_metadata_lost_a_key(tmp_path):
    path = save_logits(tmp_path / "r.npz", fake_splits(), fake_metadata())
    rewrite_metadata(path, lambda m: m.pop("model_revision"))
    with pytest.raises(ArchiveError, match="model_revision"):
        load_logits(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [("seed", True), ("seed", "42"), ("git_commit", ""), ("oos_train_rows", None)],
)
def test_metadata_with_wrong_type_or_empty_value_is_refused(tmp_path, key, value):
    with pytest.raises(ArchiveError, match=key):
        save_logits(tmp_path / "r.npz", fake_splits(), fake_metadata(**{key: value}))


def test_metadata_for_a_different_label_space_is_refused(tmp_path):
    other = build_label_space(["a", "oos"], {"a": "finance_agent", "oos": "oos"}).sha256
    with pytest.raises(ArchiveError, match="label space"):
        save_logits(tmp_path / "r.npz", fake_splits(), fake_metadata(label_space_sha256=other))


def test_label_space_fingerprint_changes_when_an_intent_moves_agent():
    labels = load_label_space()
    mapping = dict(labels.intent_to_agent)
    name = next(n for n, a in mapping.items() if a == "finance_agent")
    mapping[name] = "travel_agent"
    moved = build_label_space(list(labels.intent_names), mapping)
    assert moved.sha256 != labels.sha256
    assert len(labels.sha256) == 64


def test_load_refuses_logits_and_labels_of_different_length(tmp_path):
    path = save_logits(tmp_path / "r.npz", fake_splits(), fake_metadata())
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays["test_labels"] = arrays["test_labels"][:-1]
    with path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    with pytest.raises(ArchiveError, match="line up"):
        load_logits(path)


def test_save_refuses_wrong_column_count_nan_and_out_of_range_labels(tmp_path):
    base = fake_splits()
    narrow = dict(base, test=SplitLogits("test", base["test"].logits[:, :150], base["test"].labels))
    with pytest.raises(ArchiveError, match="columns"):
        save_logits(tmp_path / "a.npz", narrow, fake_metadata())
    nan_logits = base["test"].logits.copy()
    nan_logits[0, 0] = np.nan
    with pytest.raises(ArchiveError, match="NaN"):
        save_logits(
            tmp_path / "b.npz",
            dict(base, test=SplitLogits("test", nan_logits, base["test"].labels)),
            fake_metadata(),
        )
    bad_labels = base["test"].labels.copy()
    bad_labels[0] = NUM_INTENTS
    with pytest.raises(ArchiveError, match="label ids"):
        save_logits(
            tmp_path / "c.npz",
            dict(base, test=SplitLogits("test", base["test"].logits, bad_labels)),
            fake_metadata(),
        )


def test_save_refuses_mislabelled_or_missing_splits(tmp_path):
    base = fake_splits()
    swapped = {"validation": base["test"], "test": base["validation"]}
    with pytest.raises(ArchiveError, match="holds the 'test' split"):
        save_logits(tmp_path / "r.npz", swapped, fake_metadata())
    with pytest.raises(ArchiveError, match="exactly the splits"):
        save_logits(tmp_path / "r.npz", {"test": base["test"]}, fake_metadata())


def test_manifest_records_sha256_and_detects_a_changed_file(tmp_path):
    manifest = tmp_path / "logits-manifest.json"
    path = save_logits(tmp_path / "logits" / "run.npz", fake_splits(), fake_metadata())
    entry = record_in_manifest(manifest, path)
    assert read_manifest(manifest)["run.npz"] == entry
    assert entry["rows"] == {"validation": 7, "test": 11}
    check_against_manifest(manifest, path)
    assert verify_all(manifest, tmp_path / "logits") == []

    save_logits(path, fake_splits(seed=1), fake_metadata())
    with pytest.raises(ArchiveError, match="sha256"):
        check_against_manifest(manifest, path)
    assert len(verify_all(manifest, tmp_path / "logits")) == 1


def test_manifest_check_fails_for_an_unlisted_or_missing_file(tmp_path):
    manifest = tmp_path / "logits-manifest.json"
    a = save_logits(tmp_path / "logits" / "a.npz", fake_splits(), fake_metadata())
    b = save_logits(tmp_path / "logits" / "b.npz", fake_splits(seed=2), fake_metadata())
    record_in_manifest(manifest, a)
    with pytest.raises(ArchiveError, match="not listed"):
        check_against_manifest(manifest, b)
    record_in_manifest(manifest, b)
    assert sorted(read_manifest(manifest)) == ["a.npz", "b.npz"]
    a.unlink()
    assert verify_all(manifest, tmp_path / "logits") == [
        f"a.npz: listed in the manifest but not in {tmp_path / 'logits'}"
    ]


def test_git_state_outside_a_repository_says_unknown(tmp_path):
    assert git_state(tmp_path) == ("unknown", True)


def test_git_state_ignores_changes_under_results_but_not_elsewhere(tmp_path):
    import subprocess

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "logits-manifest.json").write_text("{}")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    commit, dirty = git_state(tmp_path)
    assert len(commit) == 40 and dirty is False
    (tmp_path / "results" / "logits-manifest.json").write_text('{"files": {}}')
    assert git_state(tmp_path)[1] is False
    (tmp_path / "code.py").write_text("x = 2\n")
    assert git_state(tmp_path)[1] is True
