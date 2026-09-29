"""AC1b comparison verdict (docs/PLAN.md 5.1): only the flow or AC2 can FAIL it."""

import json
import shutil
from pathlib import Path

import pytest

from tinyrouter import comparison
from tinyrouter.comparison import (
    FAIL,
    OK,
    ORIGINAL_AC6_USD,
    PASS,
    REVIEW,
    ac2_section,
    budget_section,
    build,
    compare_stat,
    haiku_disagreements,
    haiku_section,
    render_markdown,
    verdict,
)

RESULTS = Path(__file__).parent.parent / "results"
STEPS = ["ac2", "analysis"]
IDENTITY = json.loads((RESULTS / "llm" / "haiku-8way.json").read_text())["identity_sha256"]


def stat(mean, std, n=3):
    return {"mean": mean, "std": std, "values": [mean] * n}


def test_a_rerun_inside_the_original_mean_plus_minus_std_is_ok():
    assert compare_stat("x", stat(0.90, 0.01), stat(0.909, 0.02))["status"] == OK
    assert compare_stat("x", stat(0.90, 0.01), stat(0.891, 0.0))["status"] == OK


def test_a_rerun_outside_the_band_is_flagged_for_review():
    row = compare_stat("x", stat(0.90, 0.01), stat(0.911, 0.0))
    assert row["status"] == REVIEW
    assert row["difference"] == pytest.approx(0.011)


def test_a_null_std_from_a_single_feasible_seed_flags_any_difference():
    single = {"mean": 0.5, "std": None, "n": 1, "values": [None, 0.5, None]}
    assert compare_stat("x", single, dict(single))["status"] == OK
    moved = {**single, "mean": 0.5001}
    row = compare_stat("x", single, moved)
    assert row["status"] == REVIEW
    assert "no seed spread" in row["rule"]


def test_a_null_std_with_another_number_of_feasible_seeds_is_flagged_even_at_the_same_mean():
    single = {"mean": 0.5, "std": None, "n": 1, "values": [None, 0.5, None]}
    assert compare_stat("x", single, stat(0.5, None))["status"] == REVIEW


def test_a_zero_std_or_a_single_run_number_needs_an_exact_match():
    assert compare_stat("share", stat(0.181818, 0.0), stat(0.181818, 0.0))["status"] == OK
    assert compare_stat("haiku", 0.820727, 0.820727)["status"] == OK
    assert compare_stat("haiku", 0.820727, 0.821)["status"] == REVIEW


def test_a_number_in_only_one_run_is_flagged_and_absent_in_both_is_ok():
    assert compare_stat("x", stat(0.5, 0.1), None)["status"] == REVIEW
    assert compare_stat("x", None, stat(0.5, 0.1))["status"] == REVIEW
    assert compare_stat("x", None, None)["status"] == OK


def ac2_json(*accuracies):
    seeds = {
        str(seed): {"test": {"in_scope_accuracy_150": acc}}
        for seed, acc in zip((42, 43, 44), accuracies, strict=True)
    }
    return {"seeds": seeds}


def test_ac2_below_957_on_one_seed_fails_the_section():
    section = ac2_section(ac2_json(0.97, 0.9569, 0.97), ac2_json(0.97, 0.97, 0.97))
    assert section["verdict"] == FAIL
    assert [s["passed"] for s in section["seeds"].values()] == [True, False, True]
    assert ac2_section(ac2_json(0.957, 0.957, 0.957), {})["verdict"] == PASS


def test_ac2_missing_fails():
    assert ac2_section(None, {})["verdict"] == FAIL


def test_verdict_fails_only_on_flow_or_ac2():
    good_haiku, good_ac2 = {"checks_passed": True}, {"verdict": PASS}
    assert verdict(True, good_haiku, good_ac2) == PASS
    assert verdict(False, good_haiku, good_ac2) == FAIL
    assert verdict(True, good_haiku, {"verdict": FAIL}) == FAIL
    assert verdict(True, {"checks_passed": False}, good_ac2) == FAIL


def rows(agents, split="test", parse_failed=()):
    return [
        {"split": split, "index": i, "agent": a, "parse_failed": i in parse_failed}
        for i, a in enumerate(agents)
    ]


def test_haiku_differences_are_counted_row_by_row_on_split_and_index():
    before = rows(["finance", "oos", "travel", "home"])
    after = rows(["finance", "travel", "travel"])
    after.append({"split": "validation", "index": 0, "agent": "oos", "parse_failed": False})
    counts = haiku_disagreements(before, after)
    assert counts == {
        "compared_rows": 3,
        "different_predictions": 1,
        "only_in_original": 1,
        "only_in_reproduction": 1,
    }


def test_the_same_agents_in_another_order_do_not_count_as_differences():
    before = rows(["finance", "oos", "travel"])
    assert haiku_disagreements(before, list(reversed(before)))["different_predictions"] == 0


def summary(cost, identity=IDENTITY):
    return {"identity_sha256": identity, "totals": {"cost_usd": cost}}


def test_haiku_section_reports_parse_failures_and_spend():
    full = rows(["oos"] * comparison.EXPECTED_LLM_ROWS, parse_failed={3, 7})
    body = haiku_section(full, full, summary(3.2), summary(3.18))
    assert body["checks_passed"] is True
    assert body["rows"] == "8600/8600"
    assert body["parse_failed"] == 2
    assert body["reproduction_cost_usd"] == 3.2
    assert body["predictions"]["different_predictions"] == 0


def test_haiku_checks_fail_on_missing_rows_another_identity_or_spend_above_the_cap():
    short = rows(["oos"] * 10)
    assert not haiku_section(short, short, summary(1.0), summary(1.0))["checks_passed"]
    full = rows(["oos"] * comparison.EXPECTED_LLM_ROWS)
    other = haiku_section(full, full, summary(1.0, "0" * 64), summary(1.0))
    assert "identity" in other["problems"][0]
    over = haiku_section(full, full, summary(5.01), summary(1.0))
    assert not over["checks_passed"]


def test_the_two_budgets_stay_separate_and_the_original_is_fixed():
    budget = budget_section(summary(4.2))
    assert budget["original_ac6_experiment"]["usd"] == ORIGINAL_AC6_USD == 3.19
    assert budget["reproduction_validation"]["usd"] == 4.2
    assert budget["reproduction_validation"]["cap_usd"] == 5.0
    flat = json.dumps(budget)
    assert "7.39" not in flat and "total" not in flat


@pytest.fixture
def roots(tmp_path):
    """Committed results as the original, a copy as the rerun, and a Haiku journal for both."""
    original = tmp_path / "original"
    for name in ("analysis/summary.json", "ac2.json", "pilots/lr.json", "pilots/steps.json"):
        (original / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(RESULTS / name, original / name)
    (original / "llm").mkdir()
    shutil.copy(RESULTS / "llm" / "haiku-8way.json", original / "llm" / "haiku-8way.json")
    rerun = tmp_path / "rerun"
    shutil.copytree(original, rerun)
    journal = "\n".join(json.dumps(r) for r in rows(["oos"] * 8600)) + "\n"
    (rerun / "llm" / "haiku-8way.jsonl").write_text(journal)
    (tmp_path / "haiku-original.jsonl").write_text(journal)
    return original, rerun, tmp_path / "haiku-original.jsonl"


def run_build(roots, steps=None):
    original, rerun, haiku = roots
    steps = steps or [{"name": n, "status": PASS} for n in STEPS]
    return build(
        original_root=original,
        reproduced_root=rerun,
        original_haiku=haiku,
        steps=steps,
        expected_steps=STEPS,
        context={},
    )


def test_the_committed_results_against_themselves_pass_with_nothing_to_review(roots):
    body = run_build(roots)
    assert body["verdict"] == PASS
    assert body["review_required"] == 0
    assert len(body["headline"]) == 14
    assert len(body["learning_curves"]) == 24
    assert len(body["threshold_diagnostics"]) > 100
    assert "Verdict: PASS" in render_markdown(body)


def edit(path: Path, change) -> None:
    body = json.loads(path.read_text())
    change(body)
    path.write_text(json.dumps(body))


def test_a_number_off_by_more_than_its_std_is_reviewed_but_still_passes(roots):
    _, rerun, _ = roots

    def shift(body):
        stat = body["groups"]["modernbert/k100"]["result"]["final"]["small_only"]["accuracy_8"]
        stat["mean"] += 0.05

    edit(rerun / "analysis" / "summary.json", shift)
    body = run_build(roots)
    assert body["verdict"] == PASS
    assert body["review_required"] >= 2  # first screen and the learning curve row
    flagged = [r["metric"] for r in body["headline"] if r["status"] == REVIEW]
    assert flagged == ["ModernBERT k=100 small-only 8-way accuracy"]


def test_a_different_pilot_choice_is_reviewed_not_failed(roots):
    _, rerun, _ = roots
    edit(rerun / "pilots" / "steps.json", lambda b: b.update(selected=200))
    body = run_build(roots)
    assert body["verdict"] == PASS
    assert body["pilots"]["steps"]["status"] == REVIEW
    assert body["review_required"] == 1


def test_an_ac2_seed_at_95_69_fails_the_whole_reproduction(roots):
    _, rerun, _ = roots
    edit(
        rerun / "ac2.json",
        lambda b: b["seeds"]["43"]["test"].update(in_scope_accuracy_150=0.9569),
    )
    assert run_build(roots)["verdict"] == FAIL


def test_a_failed_step_fails_the_reproduction_even_when_every_number_matches(roots):
    steps = [{"name": "ac2", "status": PASS}, {"name": "analysis", "status": FAIL}]
    body = run_build(roots, steps)
    assert body["verdict"] == FAIL
    assert body["flow"]["passed"] is False


def test_a_step_that_never_ran_fails_the_reproduction(roots):
    assert run_build(roots, [{"name": "ac2", "status": PASS}])["verdict"] == FAIL


def test_numbers_are_marked_not_reached_when_the_analysis_never_ran(roots):
    _, rerun, _ = roots
    (rerun / "analysis" / "summary.json").unlink()
    body = run_build(roots, [{"name": "ac2", "status": FAIL}])
    assert body["headline"] is None
    assert "not reached" in render_markdown(body)
