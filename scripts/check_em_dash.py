"""Fail if any project text file contains an em dash (U+2014).

Workspace rule: prose uses a colon, semicolon, parentheses, or comma
instead. Scans every text file under the project root except environments,
caches, and model output.

ALLOWED pins an exact count per file. A file over its count fails; a file
under its count, or missing, also fails, so a fixed file forces its entry
to be removed instead of leaving a silent exemption behind.

Mutation check (2026-09-23): appending an em dash to README.md made this
exit 1 naming README.md; raising a pinned file's allowed count by one (as if
one of its em dashes had been fixed) reported the entry as stale.
tests/test_em_dash_check.py repeats both on a temporary tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

EM_DASH = chr(0x2014)  # spelled as a code point so this file passes its own check
ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {
    ".venv",
    ".venv.nosync",
    "checkpoints",
    "checkpoints.nosync",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".toml", ".json", ".txt", ".cfg", ".example"}
TEXT_NAMES = {"Makefile", "LICENSE", ".gitignore"}
ALLOWED: dict[str, tuple[int, str]] = {}


def iter_text_files(root: Path) -> list[Path]:
    out = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts) or not path.is_file():
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
            out.append(path)
    return sorted(out)


def find_problems(root: Path, allowed: dict[str, tuple[int, str]]) -> list[str]:
    problems = []
    counts = {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8", errors="replace").count(
            EM_DASH
        )
        for p in iter_text_files(root)
    }
    for rel, count in counts.items():
        limit = allowed.get(rel, (0, ""))[0]
        if count > limit:
            problems.append(f"{rel}: {count} em dash(es), allowed {limit}")
    for rel, (limit, reason) in allowed.items():
        if not reason.strip():
            problems.append(f"allowlist entry {rel} has no reason")
        if counts.get(rel, 0) < limit:
            problems.append(f"allowlist entry {rel} is stale ({counts.get(rel, 0)} < {limit})")
    return problems


def main() -> int:
    problems = find_problems(ROOT, ALLOWED)
    for line in problems:
        print(line)
    if problems:
        print("replace em dashes with a colon, semicolon, parentheses, or comma")
        return 1
    print("em dash check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
