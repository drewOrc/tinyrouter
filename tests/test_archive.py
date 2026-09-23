import json
from pathlib import Path

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


def write_format1(path, **overrides):
    """An archive as the AC2 code wrote it: format 1, no dataset_sha256, no k_shot."""
    meta = fake_metadata(format_version=1, **overrides)
    del meta["dataset_sha256"], meta["k_shot"]
    arrays = {"metadata": np.array(json.dumps(meta, sort_keys=True))}
    for name, split in fake_splits().items():
        arrays[f"{name}_logits"] = split.logits
        arrays[f"{name}_labels"] = split.labels.astype(np.int64)
    with path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    return path


def test_new_archives_record_the_sha256_of_all_three_split_files(tmp_path):
    from tinyrouter.data import SPLIT_FILES

    archive = load_logits(save_logits(tmp_path / "r.npz", fake_splits(), fake_metadata()))
    assert archive.metadata["format_version"] == 2
    assert archive.metadata["dataset_sha256"] == {
        name: SPLIT_FILES[name].sha256 for name in ("train", "validation", "test")
    }
    assert "dataset_sha256_source" not in archive.metadata


def test_format1_archive_at_the_pinned_revision_is_back_filled_on_read(tmp_path):
    from tinyrouter.archive import BACKFILL_NOTE, pinned_dataset_sha256

    archive = load_logits(write_format1(tmp_path / "old.npz"))
    assert archive.metadata["format_version"] == 1
    assert archive.metadata["dataset_sha256"] == pinned_dataset_sha256()
    assert archive.metadata["dataset_sha256_source"] == BACKFILL_NOTE
    assert archive.metadata["k_shot"] is None


def test_format1_archive_from_another_dataset_revision_is_refused(tmp_path):
    path = write_format1(tmp_path / "old.npz", dataset_revision="f" * 40)
    with pytest.raises(ArchiveError, match="can only be back-filled"):
        load_logits(path)


def test_format1_archive_that_already_claims_checksums_is_refused(tmp_path):
    path = write_format1(tmp_path / "old.npz")
    rewrite_metadata(path, lambda m: m.update(dataset_sha256={"train": "x"}))
    with pytest.raises(ArchiveError, match="already carries"):
        load_logits(path)


@pytest.mark.parametrize(
    "bad",
    [
        {"train": "0" * 64, "validation": "0" * 64, "test": "0" * 64},
        {"train": "x", "validation": "y"},
        None,
    ],
)
def test_format2_archive_with_other_dataset_checksums_is_refused(tmp_path, bad):
    with pytest.raises(ArchiveError, match="dataset_sha256|pinned CLINC150"):
        save_logits(tmp_path / "r.npz", fake_splits(), fake_metadata(dataset_sha256=bad))


def test_format2_archive_of_another_revision_is_refused_even_with_pinned_checksums(tmp_path):
    with pytest.raises(ArchiveError, match="pinned CLINC150"):
        save_logits(tmp_path / "r.npz", fake_splits(), fake_metadata(dataset_revision="f" * 40))


def test_format1_cannot_be_written_any_more(tmp_path):
    with pytest.raises(ArchiveError, match="cannot be written"):
        save_logits(tmp_path / "r.npz", fake_splits(), fake_metadata(format_version=1))


def test_load_validation_logits_never_needs_the_test_arrays(tmp_path):
    from tinyrouter.archive import load_validation_logits

    path = save_logits(tmp_path / "r.npz", fake_splits(), fake_metadata())
    with np.load(path, allow_pickle=False) as npz:
        kept = {k: npz[k] for k in npz.files if not k.startswith("test_")}
    with path.open("wb") as fh:
        np.savez_compressed(fh, **kept)
    with pytest.raises(ArchiveError, match="test_"):
        load_logits(path)
    meta, val = load_validation_logits(path)
    assert val.split == "validation" and val.logits.shape == (7, NUM_INTENTS)
    assert meta["run_name"] == "bert-base-uncased-full-seed42"


def test_load_validation_logits_back_fills_a_format1_archive(tmp_path):
    from tinyrouter.archive import load_validation_logits

    meta, _ = load_validation_logits(write_format1(tmp_path / "old.npz"))
    assert meta["dataset_sha256_source"]


AC2_ARCHIVES = sorted(
    (Path(__file__).parent.parent / "results" / "logits").glob("bert-base-uncased-full-*.npz")
)


@pytest.mark.skipif(not AC2_ARCHIVES, reason="AC2 archives are not in git (GitHub Release)")
@pytest.mark.parametrize("path", AC2_ARCHIVES, ids=lambda p: p.name)
def test_the_real_ac2_archives_still_load_as_format1(path):
    archive = load_logits(path)
    assert archive.metadata["format_version"] == 1
    assert archive.test.logits.shape == (5_500, NUM_INTENTS)
