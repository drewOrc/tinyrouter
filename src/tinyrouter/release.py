"""Download the GitHub Release files the analysis reads, and check each against its SHA-256.

Three Releases hold what git does not: ``ac2-bert-logits`` (the three AC2
archives), ``curves-logits`` (every other logits archive) and
``haiku-predictions`` (``haiku-8way.jsonl``). The list of files and their
SHA-256 come from the committed manifests, ``results/logits-manifest.json``
and ``results/llm-manifest.json``, never from the Release itself: a Release
asset that changed after the commit fails here instead of being trusted.

A file already on disk with the right SHA-256 is kept. Anything else is
downloaded to a temporary name, checked, and only then moved into place, so
an interrupted download never leaves a file that looks complete. The last
line, printed only when every file checks out, is
``verified N/N release files``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tinyrouter.data import sha256_of

REPOSITORY = "drewOrc/tinyrouter"
AC2_TAG = "ac2-bert-logits"
CURVES_TAG = "curves-logits"
HAIKU_TAG = "haiku-predictions"
AC2_ARCHIVES = frozenset(f"bert-base-uncased-full-seed{seed}.npz" for seed in (42, 43, 44))
ATTEMPTS = 3
TIMEOUT_SECONDS = 120

Fetch = Callable[[str, Path], None]


class ReleaseError(RuntimeError):
    """A Release file is missing, or its bytes are not the ones the manifest records."""


@dataclass(frozen=True)
class Asset:
    tag: str
    name: str
    sha256: str
    subdir: str

    @property
    def url(self) -> str:
        return f"https://github.com/{REPOSITORY}/releases/download/{self.tag}/{self.name}"


def manifest_files(path: Path) -> dict[str, dict]:
    return dict(json.loads(path.read_text(encoding="utf-8"))["files"])


def planned_assets(manifest_root: Path, only: str | None = None) -> list[Asset]:
    """Every Release file the committed manifests list, with the Release that holds it."""
    assets: list[Asset] = []
    if only in (None, "logits"):
        for name, entry in sorted(manifest_files(manifest_root / "logits-manifest.json").items()):
            tag = AC2_TAG if name in AC2_ARCHIVES else CURVES_TAG
            assets.append(Asset(tag, name, entry["sha256"], "logits"))
    if only in (None, "llm"):
        for name, entry in sorted(manifest_files(manifest_root / "llm-manifest.json").items()):
            assets.append(Asset(HAIKU_TAG, name, entry["sha256"], "llm"))
    return assets


def http_fetch(url: str, dest: Path) -> None:
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
                dest.write_bytes(response.read())
            return
        except OSError as exc:
            last = exc
            time.sleep(2**attempt)
    raise ReleaseError(f"could not download {url} after {ATTEMPTS} attempts: {last}")


def ensure(asset: Asset, dest_root: Path, fetch: Fetch) -> bool:
    """Put ``asset`` under ``dest_root`` with the recorded SHA-256; True if it was downloaded."""
    dest = dest_root / asset.subdir / asset.name
    if dest.is_file() and sha256_of(dest) == asset.sha256:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    fetch(asset.url, partial)
    actual = sha256_of(partial)
    if actual != asset.sha256:
        partial.unlink()
        raise ReleaseError(
            f"{asset.tag}/{asset.name}: SHA-256 {actual} is not the manifest's {asset.sha256}; "
            "the Release asset differs from the one this commit recorded"
        )
    os.replace(partial, dest)
    return True


def download(
    manifest_root: Path, dest_root: Path, only: str | None = None, fetch: Fetch = http_fetch
) -> tuple[int, int]:
    """(downloaded, total); raises ReleaseError on the first file that does not check out."""
    assets = planned_assets(manifest_root, only)
    downloaded = sum(ensure(asset, dest_root, fetch) for asset in assets)
    for asset in assets:
        if sha256_of(dest_root / asset.subdir / asset.name) != asset.sha256:
            raise ReleaseError(f"{asset.name} changed on disk after it was checked")
    return downloaded, len(assets)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("download",))
    parser.add_argument("--manifest-root", default="results", help="where the manifests are")
    parser.add_argument("--dest-root", default="results", help="logits/ and llm/ go under here")
    parser.add_argument("--only", choices=("logits", "llm"), default=None)
    args = parser.parse_args(argv)
    downloaded, total = download(Path(args.manifest_root), Path(args.dest_root), args.only)
    print(f"downloaded {downloaded}, kept {total - downloaded} already present")
    print(f"verified {total}/{total} release files")


if __name__ == "__main__":
    main()
