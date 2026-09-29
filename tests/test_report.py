import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tinyrouter import report
from tinyrouter.report import BEGIN, END, Results, pm, readme_block, replace_block

RESULTS = Path("results")
EM_DASH = chr(0x2014)


@pytest.fixture(scope="module")
def results():
    return Results.load(RESULTS)


def generated_block(text: str) -> str:
    return text[text.index(BEGIN) : text.index(END) + len(END)]


def test_readme_generated_block_matches_the_committed_json(results):
    """AC8: every first-screen number comes from make report; run it if this fails."""
    readme = Path("README.md").read_text(encoding="utf-8")
    assert generated_block(readme) == readme_block(results)


def test_results_report_md_matches_the_committed_json(results):
    assert (RESULTS / "report.md").read_text(encoding="utf-8") == report.report_md(results)


def test_check_mode_exits_1_on_a_stale_readme_and_0_on_a_fresh_one(tmp_path, results):
    shutil.copytree(
        RESULTS, tmp_path / "results", ignore=shutil.ignore_patterns("logits", "*.jsonl")
    )
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n\n{BEGIN}\n91.9 typed by hand\n{END}\n", encoding="utf-8")
    argv = ["--results-dir", str(tmp_path / "results"), "--repo", str(tmp_path), "--check"]
    with pytest.raises(SystemExit) as stale:
        report.main(argv)
    assert stale.value.code == 1
    report.main(argv[:-1])
    report.main(argv)  # no SystemExit: fresh
    assert "typed by hand" not in readme.read_text()
    assert readme.read_text().startswith("# x\n\n")


def test_a_changed_number_in_the_json_changes_the_block(results):
    summary = copy.deepcopy(results.summary)
    stat = summary["groups"]["modernbert/k100"]["result"]["final"]["small_only"]["accuracy_8"]
    stat["mean"] = 0.5
    changed = Results(**{**results.__dict__, "summary": summary})
    assert "50.0 ± 0.1%" in readme_block(changed)
    assert readme_block(changed) != readme_block(results)


def test_replace_block_refuses_a_readme_without_markers():
    with pytest.raises(ValueError, match="markers"):
        replace_block("# no block here\n", f"{BEGIN}\n{END}")


def test_pm_shows_std_only_when_there_is_one():
    assert pm({"mean": 0.9187, "std": 0.0013}) == "91.9 ± 0.1"
    assert pm({"mean": 0.9187, "std": None}) == "91.9"
    assert pm({"mean": 0.07345, "std": 0.00857}, 2) == "7.34 ± 0.86"


def test_headline_numbers_and_wording(results):
    block = readme_block(results)
    diag = results.group("modernbert/k100")["diagnostics"]["0.02"]
    assert f"{100 * diag['test']['selective_risk']['mean']:.2f}" in block
    share = round(100 * diag["share_of_gap_explained_by_oos_share"]["mean"])
    assert f"closes about {share}% of the gap between validation and test risk" in block
    assert "In a reweighting diagnostic" in block
    for causal in ("caused", "causes", "because of the OOS"):
        assert causal not in block
    assert "not a risk guarantee for production" in block


def test_hybrid_call_counts_are_reported_per_seed_as_absolute_numbers(results):
    hybrid = results.hybrid("modernbert/k100")["test"]["llm_call_rate"]["values"]
    counts = ", ".join(str(round(v * results.test_n)) for v in hybrid)
    assert f"({counts} of {results.test_n:,} for seeds 42, 43, 44)" in readme_block(results)


def test_generated_text_has_no_em_dash(results):
    assert EM_DASH not in readme_block(results)
    assert EM_DASH not in report.report_md(results)


def test_every_referenced_figure_is_committed():
    for name, _, _ in report.FIGURES:
        assert (RESULTS / "figures" / name).is_file(), name


def test_efficiency_table_has_both_encoders_and_haiku_latency(results):
    block = readme_block(results)
    assert "BERT-base (historical baseline)" in block
    assert "ModernBERT-base (main model)" in block
    lat = results.haiku_latency["splits"]["test"]
    assert f"{lat['p50_ms']:.0f} / {lat['p95_ms']:.0f} ms (API, incl. network)" in block


def test_limitations_name_the_grid_edges_and_the_step_budget(results):
    block = readme_block(results)
    assert "is the top of {1e-05, 2e-05, 5e-05}" in block
    assert "S_min 400 is the top of {100, 200, 400}" in block
    assert "k in {1, 5, 10} trains for S_min = 400 steps" in block


def test_every_file_report_reads_is_tracked_in_git():
    """The README check runs in CI from a plain checkout, so no input may be a Release download."""
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "results"], capture_output=True, text=True, check=True
        ).stdout.split()
    )
    for name in (
        "analysis/summary.json",
        "analysis/haiku.json",
        "llm/haiku-8way.json",
        "efficiency/cpu_latency.json",
        "efficiency/haiku_latency.json",
        "cost/cost.json",
        "curves/bert.json",
        "curves/modernbert.json",
        "pilots/lr.json",
        "pilots/steps.json",
        "report.md",
    ):
        assert f"results/{name}" in tracked, name


SUMMARY = json.loads((RESULTS / "analysis" / "summary.json").read_text(encoding="utf-8"))
K100 = ("groups", "modernbert/k100", "result")
K10 = ("groups", "modernbert/k10", "result")
DIAG = (*K100, "diagnostics", "0.02")
K10_HYBRID = (*K10, "final", "fallback", "0.02", "hybrid", "test")
K100_HYBRID = (*K100, "final", "fallback", "0.02", "hybrid", "test")


def at(path: tuple[str, ...]):
    node = SUMMARY
    for key in path:
        node = node[key]
    return node


def one(path: tuple[str, ...], digits: int = 1) -> str:
    """A fraction at ``path`` as a percentage, formatted independently of report.py."""
    value = at(path)
    return f"{100 * (value['mean'] if isinstance(value, dict) else value):.{digits}f}"


def both(path: tuple[str, ...], digits: int = 1) -> str:
    stat = at(path)
    return f"{100 * stat['mean']:.{digits}f} ± {100 * stat['std']:.{digits}f}"


def counts(path: tuple[str, ...]) -> str:
    n = at(("llm_only_test", "n"))
    return ", ".join(f"{round(v * n):,}" for v in at(path)["values"]) + f" of {n:,}"


# Each first-screen number, the exact JSON field it must come from, and the text around it.
# A number rendered from the wrong field (another router, another split) fails here even
# after `make report` regenerates the README.
FIRST_SCREEN = {
    "k=100 small-only accuracy": lambda: (
        f"routes {both((*K100, 'final', 'small_only', 'accuracy_8'))}% of test queries"
    ),
    "Haiku accuracy": lambda: (
        f"(Claude Haiku 4.5 zero-shot: {one(('llm_only_test', 'accuracy_8'))}%)"
    ),
    "k=10 small-only accuracy": lambda: (
        f"accuracy from {both((*K10, 'final', 'small_only', 'accuracy_8'))}% (small model alone)"
    ),
    "k=10 hybrid accuracy": lambda: f"to {both((*K10_HYBRID, 'accuracy_8'))}%, with Haiku called",
    "k=10 hybrid call rate and counts": lambda: (
        f"called on {both((*K10_HYBRID, 'llm_call_rate'))}% of queries "
        f"({counts((*K10_HYBRID, 'llm_call_rate'))} for seeds 42, 43, 44)"
    ),
    "k=100 test selective risk": lambda: (
        f"gives {both((*DIAG, 'test', 'selective_risk'), 2)}% selective risk on test"
    ),
    "k=100 validation selective risk": lambda: (
        f"(ModernBERT k=100; validation: {both((*DIAG, 'validation', 'selective_risk'), 2)}%)"
    ),
    "OOS shares": lambda: (
        f"Validation is {one((*DIAG, 'validation_oos_share'))}% OOS and test "
        f"{one((*DIAG, 'test_oos_share'))}%."
    ),
    "share of gap": lambda: (
        f"closes about {round(100 * at((*DIAG, 'share_of_gap_explained_by_oos_share'))['mean'])}%"
        " of the gap"
    ),
    "kept OOS error rates": lambda: (
        f"({both((*DIAG, 'test', 'kept_oos_error_rate'))}% vs "
        f"{both((*DIAG, 'validation', 'kept_oos_error_rate'))}% on validation)"
    ),
    "k=100 hybrid call rate and counts": lambda: (
        f"sends {both((*K100_HYBRID, 'llm_call_rate'))}% of test queries to Haiku "
        f"({counts((*K100_HYBRID, 'llm_call_rate'))} for seeds 42, 43, 44)"
    ),
    "k=100 hybrid accuracy": lambda: (
        f"from {both((*K100, 'final', 'small_only', 'accuracy_8'))}% to "
        f"{both((*K100_HYBRID, 'accuracy_8'))}%. There the fallback"
    ),
}


@pytest.mark.parametrize("name", sorted(FIRST_SCREEN))
def test_each_first_screen_number_comes_from_its_own_json_field(name):
    readme = Path("README.md").read_text(encoding="utf-8")
    assert FIRST_SCREEN[name]() in generated_block(readme)


def test_k10_hybrid_counts_are_the_reviewed_absolute_numbers():
    assert counts((*K10_HYBRID, "llm_call_rate")) == "1,235, 1,157, 1,548 of 5,500"
    assert counts((*K100_HYBRID, "llm_call_rate")) == "170, 9, 31 of 5,500"
