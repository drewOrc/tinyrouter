"""``make reproduce`` (AC1b) and ``make reproduce-artifacts`` (AC1a), docs/PLAN.md section 5.

AC1b, ``full``: the whole study again from a clean clone, in the order it
was first run: AC2, the two pilots, the baselines, both curves, the OOS
ablation, Haiku, then analysis, latency, cost, figures and report. Every
rerun output goes under ``reproduction/<id>/`` (``<id>`` defaults to the
first 12 characters of HEAD), never into the committed ``results/`` or
``README.md``:

- ``results/``: what the make targets write, through ``RESULTS_ROOT``;
- ``checkpoints/``: weights while training, through ``CHECKPOINT_ROOT``;
- ``original/haiku-8way.jsonl``: the original Haiku predictions from the
  Release, checked against the committed manifest, for the row comparison;
- ``README.md``: the rerun's README, through ``README_OUT``;
- ``logs/<step>.log``, ``preflight.json``, ``steps.json`` and
  ``comparison.{json,md}``.

Haiku writes a fresh journal under the rerun's own results root, so its
US$5 cap (``MAX_USD=5``) covers the reproduction-validation run alone and
the original run's spend is neither reused nor touched.

A step passes only when every one of its completion lines appears as a
whole line in its output, and its exit code is 0; the exit code alone never
counts as done. The first failing step stops the run, and the comparison is
still written, with verdict FAIL. Rerunning resumes: a step recorded as
passed whose output files still have the recorded SHA-256 is skipped, until
one step has to run again; from there every later step runs (each make
target resumes its own finished work, so that costs little). The last step
checks that ``results/`` and ``README.md`` still match HEAD.

AC1a, ``artifacts``: download the three Releases into ``results/``, verify
them, rebuild analysis, Haiku latency, cost, figures and report, and check
that ``results/`` and ``README.md`` are byte-identical to HEAD. CPU latency
is not rerun (it depends on the machine); the committed file is used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from tinyrouter import comparison
from tinyrouter.archive import utc_now
from tinyrouter.data import DATASET_REVISION, sha256_of

ORIGINAL_RESULTS = Path("results")
REPRODUCTION_DIR = Path("reproduction")
MAX_USD = 5
KEY_ENV = "ANTHROPIC_API_KEY"
GIB = 1024**3
# Peak extra disk while it runs: base models in the Hugging Face cache (about
# 1.1 GB), one ModernBERT run's final weights plus its one kept checkpoint
# with optimizer state (about 2.4 GB), AC2 seed 42's kept weights (0.44 GB),
# 75 logits archives (0.36 GB) and the Haiku journal. About 4.4 GB; twice
# that is required so a full disk cannot stop a six-hour run near the end.
REQUIRED_FREE_GIB = 8.8
ORIGINALS = ("results", "README.md")
ORIGINALS_OK = "originals untouched: results/ and README.md match HEAD"
ANALYSIS_DONE = "completed analysis (75/75 archives, 8600/8600 llm rows, 25 groups)"
LLM_DONE = "completed 8600/8600 llm predictions"
FIGURES = ("learning_curves.png", "risk_coverage.png", "routers.png", "threshold_transfer.png")

Executor = Callable[["Step", Path], tuple[int, list[str]]]
Git = Callable[[list[str]], tuple[int, str]]


class PathIsolationError(ValueError):
    """A rerun would write into the committed results or README."""


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...]
    completion: tuple[str, ...]
    outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Layout:
    """Where one rerun writes; every path is relative to the repository root."""

    run_id: str
    base: Path = field(init=False)

    def __post_init__(self) -> None:
        if not self.run_id or "/" in self.run_id or self.run_id.startswith("."):
            raise PathIsolationError(f"bad reproduction id {self.run_id!r}")
        object.__setattr__(self, "base", REPRODUCTION_DIR / self.run_id)

    @property
    def results(self) -> Path:
        return self.base / "results"

    @property
    def checkpoints(self) -> Path:
        return self.base / "checkpoints"

    @property
    def original(self) -> Path:
        return self.base / "original"

    @property
    def readme(self) -> Path:
        return self.base / "README.md"


def check_isolation(layout: Layout, repo: Path) -> None:
    """Refuse any rerun path that is, or is inside, the committed results or README."""
    protected = [(repo / name).resolve() for name in ORIGINALS]
    for path in (layout.results, layout.checkpoints, layout.original, layout.readme):
        resolved = (repo / path).resolve()
        for guard in protected:
            if resolved == guard or guard in resolved.parents:
                raise PathIsolationError(f"{path} resolves to {resolved}, inside {guard}")


def make(target: str, *variables: str) -> tuple[str, ...]:
    return ("make", target, *variables)


def full_steps(layout: Layout) -> list[Step]:
    """AC1b, in the order and with the settings of the original runs."""
    r = str(layout.results)
    roots = (f"RESULTS_ROOT={r}", f"CHECKPOINT_ROOT={layout.checkpoints}")
    return [
        Step(
            "original-haiku",
            ("uv", "run", "python", "-m", "tinyrouter.release", "download", "--only", "llm")
            + ("--dest-root", str(layout.original)),
            ("verified 1/1 release files",),
            (f"{layout.original}/llm/haiku-8way.jsonl",),
        ),
        Step("ac2", make("ac2", *roots), (f"wrote {r}/ac2.json: PASS",), (f"{r}/ac2.json",)),
        Step(
            "pilot-lr",
            make("pilot-lr", *roots),
            (f"wrote {r}/pilots/lr.json",),
            (f"{r}/pilots/lr.json",),
        ),
        Step(
            "pilot-steps",
            make("pilot-steps", *roots),
            (f"wrote {r}/pilots/steps.json",),
            (f"{r}/pilots/steps.json",),
        ),
        Step(
            "baselines",
            make("baselines", roots[0]),
            ("completed 36/36 baseline points",),
            (f"{r}/curves/baselines.json",),
        ),
        *curve_steps(r, roots),
        Step(
            "verify-logits",
            make("verify-logits", roots[0]),
            (f"OK 75 archive(s) match {r}/logits-manifest.json",),
            (f"{r}/logits-manifest.json",),
        ),
        Step(
            "llm",
            make("llm", roots[0], f"MAX_USD={MAX_USD}"),
            (LLM_DONE,),
            (f"{r}/llm/haiku-8way.jsonl", f"{r}/llm/haiku-8way.json"),
        ),
        Step("verify-llm", make("verify-llm", roots[0]), (LLM_DONE,), (f"{r}/llm-manifest.json",)),
        *rebuild_steps(r, (roots[0],), str(layout.readme), bench=True),
        Step("originals-untouched", module("check-originals"), (ORIGINALS_OK,)),
    ]


def curve_steps(r: str, roots: tuple[str, str]) -> list[Step]:
    steps = [
        Step(
            f"curve-{model}",
            make("curve", f"MODEL={model}", *roots),
            (f"completed 18/18 encoder points ({model})",),
            (f"{r}/curves/{model}.json",),
        )
        for model in ("bert", "modernbert")
    ]
    ablation = Step(
        "oos-ablation",
        make("oos-ablation", *roots),
        ("completed 3/3 ablation points",),
        (f"{r}/curves/oos-ablation.json",),
    )
    return [*steps, ablation]


def rebuild_steps(r: str, variables: tuple[str, ...], readme: str, bench: bool) -> list[Step]:
    """analysis, latency, cost, figures and report from the logits and Haiku predictions.

    ``bench`` reruns the CPU latency benchmark (AC1b); AC1a keeps the committed
    file, since latency depends on the machine.
    """
    analysis_files = tuple(f"{r}/analysis/{n}.json" for n in ("summary", "curves", "haiku"))
    latency = ("cpu_latency", "bench-cpu"), ("haiku_latency", "llm-latency")
    steps = [Step("analysis", make("analysis", *variables), (ANALYSIS_DONE,), analysis_files)]
    for stem, target in latency[0 if bench else 1 :]:
        out = f"{r}/efficiency/{stem}.json"
        steps.append(Step(target, make(target, *variables), (f"wrote {out}",), (out,)))
    cost = f"{r}/cost/cost.json"
    figures = tuple(f"{r}/figures/{n}" for n in FIGURES)
    readme_var = () if readme == "README.md" else (f"README_OUT={readme}",)
    return [
        *steps,
        Step("cost", make("cost", *variables), (f"wrote {cost}",), (cost,)),
        Step("figures", make("figures", *variables), tuple(f"wrote {f}" for f in figures), figures),
        Step(
            "report",
            make("report", *variables, *readme_var),
            (f"wrote {r}/report.md", f"wrote {readme}"),
            (f"{r}/report.md", readme),
        ),
    ]


def artifact_steps() -> list[Step]:
    """AC1a: Release files into results/, verified, then everything rebuilt offline."""
    r = str(ORIGINAL_RESULTS)
    return [
        Step(
            "release-download",
            ("uv", "run", "python", "-m", "tinyrouter.release", "download"),
            ("verified 76/76 release files",),
        ),
        Step(
            "verify-logits",
            make("verify-logits"),
            (f"OK 75 archive(s) match {r}/logits-manifest.json",),
        ),
        Step("verify-llm", make("verify-llm"), (LLM_DONE,)),
        *rebuild_steps(r, (), "README.md", bench=False),
        Step("originals-untouched", module("check-originals"), (ORIGINALS_OK,)),
    ]


def module(action: str) -> tuple[str, ...]:
    return ("uv", "run", "python", "-m", "tinyrouter.reproduce", action)


def run_command(step: Step, log_path: Path) -> tuple[int, list[str]]:
    """Run ``step`` and stream its output to the terminal and to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with (
        log_path.open("a", encoding="utf-8") as log,
        subprocess.Popen(
            step.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        ) as proc,
    ):
        log.write(f"$ {' '.join(step.command)}\n")
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            lines.append(line.rstrip("\n"))
    return proc.returncode, lines


def judge_step(step: Step, code: int, lines: list[str]) -> dict[str, object]:
    """PASS needs every completion line as a whole output line, and exit code 0."""
    seen = set(lines)
    found = [line for line in step.completion if line in seen]
    missing = [line for line in step.completion if line not in seen]
    passed = not missing and code == 0
    return {
        "name": step.name,
        "status": comparison.PASS if passed else comparison.FAIL,
        "exit_code": code,
        "completion_found": found,
        "completion_missing": missing,
    }


def output_hashes(step: Step, repo: Path) -> dict[str, str | None]:
    return {
        path: sha256_of(repo / path) if (repo / path).is_file() else None for path in step.outputs
    }


def resumable(step: Step, prior: dict | None, repo: Path) -> bool:
    """Passed before, has outputs, and every output still has the recorded SHA-256."""
    if prior is None or prior.get("status") != comparison.PASS or not step.outputs:
        return False
    recorded = prior.get("outputs")
    current = output_hashes(step, repo)
    return recorded == current and None not in current.values()


def run_steps(
    steps: list[Step], repo: Path, state_path: Path | None, logs: Path, executor: Executor
) -> list[dict]:
    """Run in order, resuming passed steps; stop at the first failure."""
    state = read_state(state_path)
    results: list[dict] = []
    rerun = False
    for step in steps:
        prior = state.get(step.name)
        if not rerun and resumable(step, prior, repo):
            print(f"[{step.name}] passed earlier and outputs unchanged, skipping", flush=True)
            results.append({**prior, "resumed": True})  # type: ignore[dict-item]
            continue
        rerun = True
        started = utc_now()
        code, lines = executor(step, logs / f"{step.name}.log")
        outcome = {**judge_step(step, code, lines), "started": started, "finished": utc_now()}
        outcome["outputs"] = output_hashes(step, repo)
        state[step.name] = outcome
        write_state(state_path, state)
        results.append(outcome)
        if outcome["status"] != comparison.PASS:
            print(f"[{step.name}] FAILED: missing {outcome['completion_missing']}", flush=True)
            break
    return results


def read_state(path: Path | None) -> dict[str, dict]:
    if path is None or not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def write_state(path: Path | None, state: dict[str, dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_git(args: list[str]) -> tuple[int, str]:
    done = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return done.returncode, done.stdout.strip()


def originals_changed(git: Git = run_git) -> str:
    """``git status`` of results/ and README.md, untracked files included; empty when clean."""
    code, out = git(["status", "--porcelain", "--untracked-files=all", "--", *ORIGINALS])
    return out if code == 0 else f"git status failed ({code})"


@dataclass(frozen=True)
class Probes:
    """What preflight looks at; replaced in tests."""

    git: Git = run_git
    free_bytes: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free
    key_present: Callable[[], bool] = lambda: bool(os.environ.get(KEY_ENV, "").strip())
    uv_sync: Callable[[], int] = lambda: (
        subprocess.run(
            ["uv", "sync", "--locked", "--group", "llm", "--group", "figures"], check=False
        ).returncode
    )


def preflight(repo: Path, probes: Probes) -> dict[str, object]:
    """Clean tree, HEAD merged into origin/main, lockfile in sync, API key set, enough disk."""
    problems = []
    _, dirty = probes.git(["status", "--porcelain", "--untracked-files=all"])
    if dirty:
        problems.append("working tree is not clean (git status --porcelain is not empty)")
    _, head = probes.git(["rev-parse", "HEAD"])
    fetched, _ = probes.git(["fetch", "--quiet", "origin", "main"])
    merged, _ = probes.git(["merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"])
    if fetched != 0 or merged != 0:
        problems.append(f"HEAD {head} is not a commit merged into origin/main")
    if probes.uv_sync() != 0:
        problems.append("uv sync --locked failed: uv.lock does not match pyproject.toml")
    key = probes.key_present()
    if not key:
        problems.append(f"{KEY_ENV} is not set (export it or put it in .env)")
    free = probes.free_bytes(repo) / GIB
    if free < REQUIRED_FREE_GIB:
        problems.append(f"{free:.1f} GiB free, need {REQUIRED_FREE_GIB} GiB")
    return {
        "checked_at": utc_now(),
        "head": head,
        "api_key_present": key,
        "free_gib": round(free, 1),
        "required_free_gib": REQUIRED_FREE_GIB,
        "problems": problems,
    }


def pins(repo: Path) -> dict[str, object]:
    """Lockfile, data and model revisions, and the Release manifests the rerun is held to."""
    revisions = {}
    for name in ("bert-base.yaml", "modernbert-base.yaml"):
        for line in (repo / "configs" / name).read_text(encoding="utf-8").splitlines():
            if line.startswith("model_revision:"):
                revisions[name] = line.split(":", 1)[1].strip()
    return {
        "uv_lock_sha256": sha256_of(repo / "uv.lock"),
        "dataset_revision": DATASET_REVISION,
        "model_revisions": revisions,
        "logits_manifest_sha256": sha256_of(repo / "results" / "logits-manifest.json"),
        "llm_manifest_sha256": sha256_of(repo / "results" / "llm-manifest.json"),
    }


def write_comparison(layout: Layout, repo: Path, steps: list[dict], context: dict) -> dict:
    body = comparison.build(
        original_root=repo / ORIGINAL_RESULTS,
        reproduced_root=repo / layout.results,
        original_haiku=repo / layout.original / "llm" / "haiku-8way.jsonl",
        steps=steps,
        expected_steps=[s.name for s in full_steps(layout)],
        context=context,
    )
    (repo / layout.base).mkdir(parents=True, exist_ok=True)
    (repo / layout.base / "comparison.json").write_text(
        json.dumps(body, indent=2) + "\n", encoding="utf-8"
    )
    (repo / layout.base / "comparison.md").write_text(
        comparison.render_markdown(body), encoding="utf-8"
    )
    return body


def reproduce_full(
    layout: Layout, repo: Path, probes: Probes | None = None, executor: Executor = run_command
) -> str:
    """AC1b; returns the verdict (PASS, FAIL, or NOT STARTED when preflight refuses)."""
    check_isolation(layout, repo)
    checks = preflight(repo, probes or Probes())
    (repo / layout.base).mkdir(parents=True, exist_ok=True)
    (repo / layout.base / "preflight.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    if checks["problems"]:
        for problem in checks["problems"]:  # type: ignore[attr-defined]
            print(f"preflight: {problem}", file=sys.stderr)
        return "NOT STARTED"
    steps = run_steps(
        full_steps(layout),
        repo,
        repo / layout.base / "steps.json",
        repo / layout.base / "logs",
        executor,
    )
    context = {"reproduction_id": layout.run_id, "preflight": checks, "pins": pins(repo)}
    body = write_comparison(layout, repo, steps, context)
    return str(body["verdict"])


def reproduce_artifacts(repo: Path, executor: Executor = run_command) -> bool:
    """AC1a; True when every step passed and results/ and README.md match HEAD."""
    steps = artifact_steps()
    results = run_steps(steps, repo, None, repo / REPRODUCTION_DIR / "artifacts-logs", executor)
    return [r["name"] for r in results if r["status"] == comparison.PASS] == [s.name for s in steps]


def default_id(git: Git = run_git) -> str:
    code, head = git(["rev-parse", "HEAD"])
    if code != 0 or not head:
        raise SystemExit("error: not a git checkout; pass --id")
    return head[:12]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("full", "artifacts", "check-originals", "compare"))
    parser.add_argument("--id", default=None, help="reproduction/<id>; default: HEAD[:12]")
    args = parser.parse_args(argv)
    repo = Path(".")
    if args.action == "check-originals":
        changed = originals_changed()
        if changed:
            print(f"changed since HEAD:\n{changed}")
            raise SystemExit(1)
        print(ORIGINALS_OK)
        return
    if args.action == "artifacts":
        if not reproduce_artifacts(repo):
            raise SystemExit(1)
        print("completed reproduce-artifacts: results/ and README.md byte-identical to HEAD")
        return
    layout = Layout(args.id or default_id())
    if args.action == "compare":
        state = read_state(repo / layout.base / "steps.json")
        body = write_comparison(
            layout, repo, list(state.values()), {"reproduction_id": layout.run_id}
        )
        print(f"verdict {body['verdict']} ({body['review_required']} review required)")
        return
    verdict = reproduce_full(layout, repo)
    record = "preflight.json" if verdict == "NOT STARTED" else "comparison.md"
    print(f"AC1b verdict: {verdict}; see {layout.base}/{record}")
    if verdict != comparison.PASS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
