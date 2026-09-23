"""The shared attribution check used by both the commit-hygiene and pr-text-hygiene CI jobs."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / ".github" / "scripts" / "check-disallowed-text.sh"
WORKFLOWS = ROOT / ".github" / "workflows"


def check(text: str, patterns_file: Path | None = None) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin"}
    if patterns_file is not None:
        env["PATTERNS_FILE"] = str(patterns_file)
    return subprocess.run(
        ["bash", str(SCRIPT), "test input"],
        input=text,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )


@pytest.mark.parametrize(
    "line",
    [
        "Co-Authored-By: Hygiene Canary <tinyrouter-hygiene-canary@example.com>",
        "TINYROUTER-HYGIENE-CANARY",
        "co-authored-by: Some Bot <noreply@anthropic.com>",
    ],
)
def test_a_disallowed_line_fails_and_is_printed(line):
    result = check(f"Fix the sampler\n\nBody text\n{line}\n")
    assert result.returncode == 1
    assert line in result.stdout


def test_clean_text_and_human_co_authors_pass():
    result = check("Add pilots\n\nCo-Authored-By: Jane Doe <jane@example.com>\n")
    assert result.returncode == 0, result.stdout
    assert "clean" in result.stdout


def test_an_empty_pattern_file_is_an_error_not_a_pass(tmp_path):
    empty = tmp_path / "patterns.txt"
    empty.write_text("# only comments\n\n")
    assert check("anything\n", empty).returncode == 2


def steps_of(workflow: str, job: str) -> list[dict]:
    return yaml.safe_load((WORKFLOWS / workflow).read_text())["jobs"][job]["steps"]


def test_pr_text_job_runs_on_edits_and_passes_title_and_body_through_env_only():
    workflow = yaml.safe_load((WORKFLOWS / "pr-text.yml").read_text())
    # PyYAML reads the bare key `on` as True.
    trigger = workflow[True]["pull_request"]["types"]
    assert set(trigger) == {"opened", "edited", "synchronize", "reopened"}
    step = steps_of("pr-text.yml", "pr-text-hygiene")[-1]
    assert step["env"]["PR_TITLE"] == "${{ github.event.pull_request.title }}"
    assert step["env"]["PR_BODY"] == "${{ github.event.pull_request.body }}"
    assert "${{" not in step["run"]
    assert "check-disallowed-text.sh" in step["run"]


def test_no_workflow_interpolates_pr_or_commit_text_inside_run():
    risky = re.compile(r"\$\{\{[^}]*(title|body|head_ref|message)[^}]*\}\}")
    for path in WORKFLOWS.glob("*.yml"):
        for job in yaml.safe_load(path.read_text())["jobs"].values():
            for step in job["steps"]:
                assert not risky.search(step.get("run", "")), (path.name, step.get("name"))


def test_commit_hygiene_uses_the_same_check():
    run = steps_of("ci.yml", "commit-hygiene")[-1]["run"]
    assert "check-disallowed-text.sh" in run
    assert "disallowed-trailers.txt" not in run
