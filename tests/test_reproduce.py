"""``make reproduce`` (AC1b) and ``make reproduce-artifacts`` (AC1a): driver behaviour.

No test here trains, downloads or calls an API: steps run through a fake
executor, and preflight reads fake probes.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from tinyrouter import reproduce
from tinyrouter.comparison import FAIL, PASS
from tinyrouter.reproduce import (
    Layout,
    PathIsolationError,
    Probes,
    Step,
    artifact_steps,
    check_isolation,
    full_steps,
    judge_step,
    preflight,
    reproduce_full,
    run_steps,
)

ROOT = Path(__file__).parent.parent
LAYOUT = Layout("abc123def456")


def test_every_rerun_output_is_under_reproduction_id_and_none_is_an_original():
    for step in full_steps(LAYOUT):
        for path in step.outputs:
            assert path.startswith("reproduction/abc123def456/"), (step.name, path)
    commands = " ".join(" ".join(s.command) for s in full_steps(LAYOUT))
    assert "RESULTS_ROOT=reproduction/abc123def456/results" in commands
    assert "README_OUT=reproduction/abc123def456/README.md" in commands


def test_training_steps_move_both_results_and_checkpoints():
    training = {"ac2", "pilot-lr", "pilot-steps", "curve-bert", "curve-modernbert", "oos-ablation"}
    for step in full_steps(LAYOUT):
        if step.name in training:
            assert "CHECKPOINT_ROOT=reproduction/abc123def456/checkpoints" in step.command
            assert "RESULTS_ROOT=reproduction/abc123def456/results" in step.command


def test_the_full_flow_keeps_the_original_order_and_haiku_gets_its_own_five_dollar_cap():
    names = [s.name for s in full_steps(LAYOUT)]
    assert names == [
        "original-haiku",
        "ac2",
        "pilot-lr",
        "pilot-steps",
        "baselines",
        "curve-bert",
        "curve-modernbert",
        "oos-ablation",
        "verify-logits",
        "llm",
        "verify-llm",
        "analysis",
        "bench-cpu",
        "llm-latency",
        "cost",
        "figures",
        "report",
        "originals-untouched",
    ]
    llm = next(s for s in full_steps(LAYOUT) if s.name == "llm")
    assert "MAX_USD=5" in llm.command


def make_dry_run(*args: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in {"MAKEFLAGS", "MFLAGS", "MAKELEVEL"}}
    done = subprocess.run(
        ["make", "-n", *args], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    )
    return done.stdout


def test_each_make_command_of_the_rerun_passes_the_moved_roots_to_python():
    """The Makefile, not only the step list, carries the roots (a dropped $(ROOT_ARGS) fails)."""
    root = "reproduction/abc123def456/results"
    for step in full_steps(LAYOUT):
        if step.command[0] != "make":
            continue
        recipe = make_dry_run(*step.command[1:])
        python_lines = [line for line in recipe.splitlines() if "python -m tinyrouter" in line]
        assert python_lines, step.name
        for line in python_lines:
            moved = f"--results-root {root}" in line or f"--results-dir {root}" in line
            assert moved, (step.name, line)
            if "CHECKPOINT_ROOT=reproduction/abc123def456/checkpoints" in step.command:
                assert "--checkpoint-root reproduction/abc123def456/checkpoints" in line
        if step.name == "report":
            assert "--readme-out reproduction/abc123def456/README.md" in recipe


def test_without_roots_the_make_targets_keep_the_configs_paths():
    recipe = make_dry_run("ac2")
    assert "--results-root" not in recipe and "--checkpoint-root" not in recipe


def test_a_reproduction_directory_pointing_into_results_is_refused(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "reproduction").symlink_to(tmp_path / "results")
    with pytest.raises(PathIsolationError):
        check_isolation(LAYOUT, tmp_path)


def test_an_id_that_escapes_the_reproduction_directory_is_refused():
    for bad in ("", "../results", ".hidden", "a/b"):
        with pytest.raises(PathIsolationError):
            Layout(bad)


def test_a_normal_layout_passes_the_isolation_check(tmp_path):
    (tmp_path / "results").mkdir()
    check_isolation(LAYOUT, tmp_path)


STEP = Step("curve-bert", ("make", "curve"), ("completed 18/18 encoder points (bert)",))


def test_a_step_passes_only_on_its_whole_completion_line_and_exit_0():
    good = ["training", "completed 18/18 encoder points (bert)"]
    assert judge_step(STEP, 0, good)["status"] == PASS
    assert judge_step(STEP, 1, good)["status"] == FAIL
    assert judge_step(STEP, 0, ["completed 36/36 baseline points"])["status"] == FAIL
    padded = ["completed 18/18 encoder points (bert) and more"]
    assert judge_step(STEP, 0, padded)["status"] == FAIL
    assert judge_step(STEP, 0, ["x completed 18/18 encoder points (bert)"])["status"] == FAIL


def test_a_step_with_several_completion_lines_needs_all_of_them():
    step = Step("report", ("make", "report"), ("wrote a", "wrote b"))
    assert judge_step(step, 0, ["wrote a"])["status"] == FAIL
    assert judge_step(step, 0, ["wrote b", "wrote a"])["status"] == PASS


class FakeExecutor:
    """Prints each step's completion lines and writes its outputs, unless told to fail it."""

    def __init__(self, repo: Path, fail: str | None = None, write: bool = True) -> None:
        self.repo, self.fail, self.write, self.ran = repo, fail, write, []

    def __call__(self, step: Step, log: Path) -> tuple[int, list[str]]:
        self.ran.append(step.name)
        if step.name == self.fail:
            return 1, ["Traceback: boom"]
        for path in step.outputs if self.write else ():
            (self.repo / path).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / path).write_text(f"{step.name} output\n")
        return 0, list(step.completion)


def three_steps() -> list[Step]:
    return [
        Step(n, ("true",), (f"done {n}",), (f"out/{n}.txt",)) for n in ("first", "second", "third")
    ]


def test_the_first_failing_step_stops_the_run_and_is_recorded(tmp_path):
    executor = FakeExecutor(tmp_path, fail="second")
    state = tmp_path / "steps.json"
    results = run_steps(three_steps(), tmp_path, state, tmp_path / "logs", executor)
    assert [r["status"] for r in results] == [PASS, FAIL]
    assert executor.ran == ["first", "second"]
    assert json.loads(state.read_text())["second"]["completion_missing"] == ["done second"]


def test_a_rerun_skips_passed_steps_whose_outputs_are_unchanged(tmp_path):
    state = tmp_path / "steps.json"
    run_steps(three_steps(), tmp_path, state, tmp_path / "logs", FakeExecutor(tmp_path, "third"))
    executor = FakeExecutor(tmp_path)
    results = run_steps(three_steps(), tmp_path, state, tmp_path / "logs", executor)
    assert executor.ran == ["third"]
    assert [r.get("resumed", False) for r in results] == [True, True, False]
    assert all(r["status"] == PASS for r in results)


def test_a_changed_output_reruns_that_step_and_every_later_one(tmp_path):
    state = tmp_path / "steps.json"
    run_steps(three_steps(), tmp_path, state, tmp_path / "logs", FakeExecutor(tmp_path))
    (tmp_path / "out" / "second.txt").write_text("edited by hand\n")
    executor = FakeExecutor(tmp_path)
    run_steps(three_steps(), tmp_path, state, tmp_path / "logs", executor)
    assert executor.ran == ["second", "third"]


def test_a_deleted_output_is_not_resumed(tmp_path):
    state = tmp_path / "steps.json"
    run_steps(three_steps(), tmp_path, state, tmp_path / "logs", FakeExecutor(tmp_path))
    (tmp_path / "out" / "first.txt").unlink()
    executor = FakeExecutor(tmp_path)
    run_steps(three_steps(), tmp_path, state, tmp_path / "logs", executor)
    assert executor.ran == ["first", "second", "third"]


def probes(
    dirty: str = "",
    merged: int = 0,
    key: bool = True,
    free_gib: float = 50.0,
    sync: int = 0,
    head: str = "f" * 40,
    status_code: int = 0,
    head_code: int = 0,
) -> Probes:
    def git(args: list[str]) -> tuple[int, str]:
        if args[0] == "status":
            return status_code, dirty
        if args[0] == "rev-parse":
            return head_code, "" if head_code else head
        if args[0] == "merge-base":
            return merged, ""
        return 0, ""

    return Probes(
        git=git,
        free_bytes=lambda _: int(free_gib * reproduce.GIB),
        key_present=lambda: key,
        uv_sync=lambda: sync,
    )


def test_preflight_passes_on_a_clean_merged_synced_checkout_with_a_key_and_disk(tmp_path):
    checks = preflight(tmp_path, probes())
    assert checks["problems"] == []
    assert checks["head"] == "f" * 40
    assert checks["api_key_present"] is True


@pytest.mark.parametrize(
    ("change", "needle"),
    [
        ({"dirty": " M src/x.py"}, "not clean"),
        ({"merged": 1}, "not a commit merged"),
        ({"key": False}, "ANTHROPIC_API_KEY"),
        ({"free_gib": 8.0}, "GiB free"),
        ({"sync": 1}, "uv.lock"),
    ],
)
def test_preflight_names_each_problem(tmp_path, change, needle):
    problems = preflight(tmp_path, probes(**change))["problems"]
    assert len(problems) == 1 and needle in problems[0]


def test_preflight_never_writes_the_key_value(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-value-not-printed")
    checks = preflight(tmp_path, Probes(git=probes().git, uv_sync=lambda: 0))
    assert "sk-test" not in json.dumps(checks)


def repo_files(path: Path) -> Path:
    """What run_identity and the snapshot read: results/, README.md, uv.lock, configs/."""
    (path / "results").mkdir(exist_ok=True)
    (path / "README.md").write_text("# x\n")
    (path / "uv.lock").write_text("lock\n")
    (path / "configs").mkdir(exist_ok=True)
    (path / "configs" / "a.yaml").write_text("a: 1\n")
    return path


def test_a_failed_preflight_runs_no_step_and_says_not_started(tmp_path):
    repo_files(tmp_path)
    executor = FakeExecutor(tmp_path)
    verdict = reproduce_full(LAYOUT, tmp_path, probes(key=False), executor)
    assert verdict == "NOT STARTED"
    assert executor.ran == []
    written = json.loads((tmp_path / LAYOUT.base / "preflight.json").read_text())
    assert written["problems"]


def test_a_flow_with_every_line_but_no_real_outputs_fails_on_ac2(tmp_path, monkeypatch):
    """Completion lines alone do not make a PASS: the comparison still needs AC2's JSON."""
    monkeypatch.setattr(reproduce, "pins", lambda repo: {})
    repo_files(tmp_path)
    verdict = reproduce_full(LAYOUT, tmp_path, probes(), FakeExecutor(tmp_path, write=False))
    assert verdict == FAIL
    body = json.loads((tmp_path / LAYOUT.base / "comparison.json").read_text())
    assert body["flow"]["passed"] is True
    assert body["ac2"]["verdict"] == FAIL


def test_the_artifacts_flow_rebuilds_into_results_without_bench_cpu_and_ends_with_the_diff():
    steps = artifact_steps()
    names = [s.name for s in steps]
    assert names[0] == "release-download" and names[-1] == "originals-untouched"
    assert "bench-cpu" not in names
    assert steps[0].completion == ("verified 76/76 release files",)
    for step in steps:
        assert not any("RESULTS_ROOT" in part for part in step.command)


def test_check_originals_reports_a_change_to_results_or_readme(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "a.json").write_text("{}\n")
    (tmp_path / "README.md").write_text("# x\n")
    (tmp_path / "other.txt").write_text("not checked\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t"}
    env |= {"GIT_COMMITTER_EMAIL": "t@t", "PATH": os.environ["PATH"]}
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True, env=env)

    def git(args):
        done = subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, text=True)
        return done.returncode, done.stdout.strip()

    (tmp_path / "other.txt").write_text("changed\n")
    assert reproduce.originals_changed(git) == ""
    (tmp_path / "results" / "new.json").write_text("{}\n")
    assert "results/new.json" in reproduce.originals_changed(git)
    (tmp_path / "results" / "new.json").unlink()
    (tmp_path / "README.md").write_text("# y\n")
    assert "README.md" in reproduce.originals_changed(git)


@pytest.mark.parametrize(
    ("change", "needle"),
    [({"status_code": 128}, "git status failed"), ({"head_code": 128}, "rev-parse HEAD failed")],
)
def test_preflight_treats_a_failing_git_as_a_problem_not_as_clean(tmp_path, change, needle):
    problems = preflight(tmp_path, probes(**change))["problems"]
    assert any(needle in p for p in problems)


IDENTITY = {
    "head": "a" * 40,
    "uv_lock_sha256": "1" * 64,
    "configs_sha256": "2" * 64,
    "python": "3.12.12",
    "torch": "2.14.0",
    "transformers": "5.17.0",
}


def test_a_state_from_the_same_identity_resumes(tmp_path):
    state = tmp_path / "steps.json"
    run_steps(three_steps(), tmp_path, state, tmp_path / "logs", FakeExecutor(tmp_path), IDENTITY)
    executor = FakeExecutor(tmp_path)
    run_steps(three_steps(), tmp_path, state, tmp_path / "logs", executor, dict(IDENTITY))
    assert executor.ran == []


@pytest.mark.parametrize(
    "field", ["head", "uv_lock_sha256", "configs_sha256", "python", "torch", "transformers"]
)
def test_a_state_from_another_identity_is_refused_and_nothing_runs(tmp_path, field):
    state = tmp_path / "steps.json"
    run_steps(three_steps(), tmp_path, state, tmp_path / "logs", FakeExecutor(tmp_path), IDENTITY)
    executor = FakeExecutor(tmp_path)
    with pytest.raises(reproduce.StaleStateError, match=f"{field} differ.*new REPRO_ID"):
        run_steps(
            three_steps(), tmp_path, state, tmp_path / "logs", executor, {**IDENTITY, field: "x"}
        )
    assert executor.ran == []


def test_a_state_without_an_identity_is_refused(tmp_path):
    state = tmp_path / "steps.json"
    run_steps(three_steps(), tmp_path, state, tmp_path / "logs", FakeExecutor(tmp_path))
    with pytest.raises(reproduce.StaleStateError, match="no identity recorded"):
        run_steps(
            three_steps(), tmp_path, state, tmp_path / "logs", FakeExecutor(tmp_path), IDENTITY
        )


def test_run_identity_covers_head_lockfile_configs_and_libraries(tmp_path):
    repo_files(tmp_path)
    first = reproduce.run_identity(tmp_path, probes().git)
    assert first["head"] == "f" * 40 and first["torch"] and first["python"]
    (tmp_path / "configs" / "a.yaml").write_text("a: 2\n")
    assert (
        reproduce.run_identity(tmp_path, probes().git)["configs_sha256"] != first["configs_sha256"]
    )
    (tmp_path / "uv.lock").write_text("other\n")
    assert (
        reproduce.run_identity(tmp_path, probes().git)["uv_lock_sha256"] != first["uv_lock_sha256"]
    )


def test_the_same_id_after_a_new_commit_is_refused_before_any_step(tmp_path, monkeypatch):
    """The review's case: same REPRO_ID, one more (empty) commit, then make reproduce again."""
    monkeypatch.setattr(reproduce, "pins", lambda repo: {})
    repo_files(tmp_path)
    reproduce_full(LAYOUT, tmp_path, probes(head="a" * 40), FakeExecutor(tmp_path, write=False))
    first = json.loads((tmp_path / LAYOUT.base / "steps.json").read_text())
    assert {s["identity"]["head"] for s in first.values()} == {"a" * 40}
    executor = FakeExecutor(tmp_path)
    verdict = reproduce_full(LAYOUT, tmp_path, probes(head="b" * 40), executor)
    assert verdict == "NOT STARTED"
    assert executor.ran == []
    problems = json.loads((tmp_path / LAYOUT.base / "preflight.json").read_text())["problems"]
    assert "head differ" in problems[0] and "new REPRO_ID" in problems[0]


def test_check_originals_with_a_snapshot_sees_gitignored_files(tmp_path):
    repo_files(tmp_path)
    (tmp_path / "results" / "logits").mkdir()
    (tmp_path / "results" / "logits" / "a.npz").write_bytes(b"archive")
    clean = probes().git
    snapshot = tmp_path / "snap.json"
    snapshot.write_text(json.dumps(reproduce.snapshot_originals(tmp_path)))
    assert reproduce.check_originals(tmp_path, snapshot, clean) == ""
    (tmp_path / "results" / "logits" / "a.npz").write_bytes(b"overwritten")
    (tmp_path / "results" / "logits" / "b.npz").write_bytes(b"new")
    changed = reproduce.check_originals(tmp_path, snapshot, clean)
    assert "changed results/logits/a.npz" in changed and "added results/logits/b.npz" in changed
    (tmp_path / "README.md").write_text("# y\n")
    assert "changed README.md" in reproduce.check_originals(tmp_path, snapshot, clean)


def test_both_flows_end_with_a_snapshot_check():
    full = full_steps(LAYOUT)[-1]
    assert full.command[-2:] == ("--snapshot", "reproduction/abc123def456/originals-snapshot.json")
    names = [s.name for s in artifact_steps()]
    assert names.index("snapshot-originals") == names.index("verify-llm") + 1
    assert "--snapshot" in artifact_steps()[-1].command
