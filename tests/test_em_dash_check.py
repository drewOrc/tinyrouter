import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "check_em_dash.py"
spec = importlib.util.spec_from_file_location("check_em_dash", SCRIPT)
assert spec and spec.loader
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

DASH = chr(0x2014)  # spelled as a code point so this file passes its own check


def test_clean_tree_passes(tmp_path):
    (tmp_path / "README.md").write_text("fine: no dashes here\n")
    assert check.find_problems(tmp_path, {}) == []


def test_em_dash_in_prose_fails(tmp_path):
    (tmp_path / "README.md").write_text(f"bad {DASH} here\n")
    assert check.find_problems(tmp_path, {}) == ["README.md: 1 em dash(es), allowed 0"]


def test_files_inside_the_virtualenv_are_skipped(tmp_path):
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "x.py").write_text(DASH)
    assert check.find_problems(tmp_path, {}) == []


def test_allowlist_is_exact_count_and_goes_stale(tmp_path):
    doc = tmp_path / "PLAN.md"
    doc.write_text(f"a {DASH} b\n")
    allowed = {"PLAN.md": (1, "reason")}
    assert check.find_problems(tmp_path, allowed) == []
    doc.write_text(f"a {DASH} b {DASH} c\n")
    assert check.find_problems(tmp_path, allowed) == ["PLAN.md: 2 em dash(es), allowed 1"]
    doc.write_text("fixed\n")
    assert check.find_problems(tmp_path, allowed) == ["allowlist entry PLAN.md is stale (0 < 1)"]


def test_allowlist_entry_needs_a_reason(tmp_path):
    (tmp_path / "PLAN.md").write_text(DASH)
    assert check.find_problems(tmp_path, {"PLAN.md": (1, " ")}) == [
        "allowlist entry PLAN.md has no reason"
    ]
