"""The Makefile has no default target (incident of 2026-09-23).

A zsh loop ran ``make $t`` with ``t="curve MODEL=bert"``. zsh does not
split an unquoted parameter, so make got one argument, read it as the
variable assignment ``curve MODEL`` = ``bert``, ran the first target
(``setup``) and exited 0; two curves were reported done with no point
run. ``make -n`` only prints the recipe, so these tests run nothing.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent


def make_dry_run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "-n", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def test_a_quoted_target_and_variable_is_refused_not_run_as_setup():
    result = make_dry_run("curve MODEL=bert")
    assert result.returncode != 0
    assert "no target given" in result.stderr
    assert "uv sync" not in result.stdout


def test_a_bare_make_is_refused():
    assert make_dry_run().returncode != 0


def test_the_same_words_split_into_target_and_variable_still_work():
    result = make_dry_run("curve", "MODEL=bert")
    assert result.returncode == 0, result.stderr
    assert "tinyrouter.curves --model bert" in result.stdout


def test_an_explicit_setup_still_works():
    result = make_dry_run("setup")
    assert result.returncode == 0, result.stderr
    assert "uv sync --locked" in result.stdout
