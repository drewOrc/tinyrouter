"""Release downloads are checked against the committed manifests, never trusted as given."""

import hashlib
from pathlib import Path

import pytest

from tinyrouter.release import (
    AC2_TAG,
    CURVES_TAG,
    HAIKU_TAG,
    Asset,
    ReleaseError,
    download,
    ensure,
    planned_assets,
)

RESULTS = Path(__file__).parent.parent / "results"


def test_the_committed_manifests_list_76_files_in_three_releases():
    assets = planned_assets(RESULTS)
    by_tag = {tag: [a for a in assets if a.tag == tag] for tag in (AC2_TAG, CURVES_TAG, HAIKU_TAG)}
    assert len(assets) == 76
    assert sorted(a.name for a in by_tag[AC2_TAG]) == [
        f"bert-base-uncased-full-seed{s}.npz" for s in (42, 43, 44)
    ]
    assert len(by_tag[CURVES_TAG]) == 72
    assert [a.name for a in by_tag[HAIKU_TAG]] == ["haiku-8way.jsonl"]
    assert by_tag[HAIKU_TAG][0].subdir == "llm"
    assert by_tag[HAIKU_TAG][0].url.endswith(
        "/releases/download/haiku-predictions/haiku-8way.jsonl"
    )
    assert len(planned_assets(RESULTS, only="llm")) == 1


def asset_for(content: bytes) -> Asset:
    return Asset("t", "f.npz", hashlib.sha256(content).hexdigest(), "logits")


def test_a_download_with_the_manifest_sha_is_moved_into_place(tmp_path):
    asset = asset_for(b"right")
    assert ensure(asset, tmp_path, lambda url, dest: dest.write_bytes(b"right")) is True
    assert (tmp_path / "logits" / "f.npz").read_bytes() == b"right"
    assert not list(tmp_path.rglob("*.part"))


def test_a_download_with_other_bytes_is_refused_and_leaves_nothing(tmp_path):
    asset = asset_for(b"right")
    with pytest.raises(ReleaseError, match="differs"):
        ensure(asset, tmp_path, lambda url, dest: dest.write_bytes(b"tampered"))
    assert not list(tmp_path.rglob("*.npz")) and not list(tmp_path.rglob("*.part"))


def test_a_file_already_present_and_correct_is_not_downloaded_again(tmp_path):
    asset = asset_for(b"right")
    (tmp_path / "logits").mkdir()
    (tmp_path / "logits" / "f.npz").write_bytes(b"right")

    def refuse(url, dest):
        raise AssertionError("should not download")

    assert ensure(asset, tmp_path, refuse) is False


def test_a_present_but_wrong_file_is_replaced(tmp_path):
    asset = asset_for(b"right")
    (tmp_path / "logits").mkdir()
    (tmp_path / "logits" / "f.npz").write_bytes(b"stale")
    assert ensure(asset, tmp_path, lambda url, dest: dest.write_bytes(b"right")) is True
    assert (tmp_path / "logits" / "f.npz").read_bytes() == b"right"


def test_download_counts_every_file_of_the_manifest(tmp_path):
    manifests = tmp_path / "m"
    manifests.mkdir()
    sha = hashlib.sha256(b"j").hexdigest()
    (manifests / "llm-manifest.json").write_text(
        f'{{"files": {{"haiku-8way.jsonl": {{"sha256": "{sha}"}}}}}}'
    )
    downloaded, total = download(
        manifests, tmp_path / "d", only="llm", fetch=lambda url, dest: dest.write_bytes(b"j")
    )
    assert (downloaded, total) == (1, 1)
    assert download(manifests, tmp_path / "d", only="llm", fetch=None) == (0, 1)  # type: ignore[arg-type]
