"""The Makefile has no default target (incident of 2026-09-23).

A zsh loop ran ``make $t`` with ``t="curve MODEL=bert"``. zsh does not
split an unquoted parameter, so make got one argument, read it as the
variable assignment ``curve MODEL`` = ``bert``, ran the first target
(``setup``) and exited 0; two curves were reported done with no point
run. ``make -n`` only prints the recipe, so these tests run nothing.

How make reads the quoted argument depends on its version: GNU make 3.81
(macOS) takes it as an assignment and, with no goal left, now stops at
the Makefile's no-target guard; GNU make 4.x (the CI runner) takes it as
a target name and stops with "No rule to make target". Either way it must
exit non-zero without reaching setup, which is what these tests check.
"""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent


def make_dry_run(*args: str) -> subprocess.CompletedProcess[str]:
    # Run as a top-level make even under `make test`: no inherited flags or level.
    env = {k: v for k, v in os.environ.items() if k not in {"MAKEFLAGS", "MFLAGS", "MAKELEVEL"}}
    return subprocess.run(
        ["make", "-n", *args], cwd=ROOT, env=env, capture_output=True, text=True, check=False
    )


def test_a_quoted_target_and_variable_is_refused_not_run_as_setup():
    result = make_dry_run("curve MODEL=bert")
    assert result.returncode != 0
    assert "no target given" in result.stderr or "No rule to make target" in result.stderr
    assert "uv sync" not in result.stdout


def test_a_bare_make_is_refused_by_the_guard():
    result = make_dry_run()
    assert result.returncode != 0
    assert "no target given" in result.stderr
    assert "uv sync" not in result.stdout


def test_only_a_variable_assignment_is_refused_by_the_guard():
    """What make 3.81 made of the quoted argument, spelled so every make version sees it."""
    result = make_dry_run("MODEL=bert")
    assert result.returncode != 0
    assert "no target given" in result.stderr
    assert "uv sync" not in result.stdout


def test_the_same_words_split_into_target_and_variable_still_work():
    result = make_dry_run("curve", "MODEL=bert")
    assert result.returncode == 0, result.stderr
    assert "tinyrouter.curves --model bert" in result.stdout


def test_an_explicit_setup_still_works():
    result = make_dry_run("setup")
    assert result.returncode == 0, result.stderr
    assert "uv sync --locked" in result.stdout
