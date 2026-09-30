"""AC1b attempt ledger and budget (docs/PLAN.md 5.1, Drew 2026-09-30)."""

import copy
import json
from pathlib import Path

import pytest

from tinyrouter import ac1b_ledger
from tinyrouter.ac1b_ledger import LedgerError
from tinyrouter.reproduce import Layout, full_steps

ROOT = Path(__file__).parent.parent
EVIDENCE = ROOT / "docs" / "ac1b"


def committed() -> dict:
    return ac1b_ledger.load(ROOT / ac1b_ledger.LEDGER)


def test_the_committed_ledger_records_both_attempts_at_3992840():
    attempts = committed()["attempts"]
    assert [
        (a["attempt"], a["result"], a["comparison_verdict"], a["haiku_usd"]) for a in attempts
    ] == [
        (1, "INFRASTRUCTURE INTERRUPTED", "FAIL", 0.0),
        (2, "FAIL", "FAIL", 3.178751),
    ]
    assert [a["failure_kind"] for a in attempts] == ["infrastructure", "program defect"]
    assert {a["reproduction_id"] for a in attempts} == {"3992840ddb3f"}
    assert attempts[0]["decision"]["source"] == (
        "Drew's decision, relayed by the coordinator (2026-09-29)"
    )
    assert "status.claude.com" in attempts[0]["evidence"]["source"]
    for a in attempts:
        assert (ROOT / a["evidence_dir"] / "comparison.md").is_file()
        assert (ROOT / a["evidence_dir"] / "INCIDENT.md").is_file()


def test_the_comparison_verdict_in_the_evidence_is_kept_as_written():
    """Attempt 1 is INTERRUPTED in the ledger; its comparison still says FAIL."""
    text = (EVIDENCE / "attempt-1" / "comparison.md").read_text(encoding="utf-8")
    assert text.startswith("# AC1b comparison\n\n**Verdict: FAIL**")


def interrupted() -> dict:
    return copy.deepcopy(committed()["attempts"][0])


def with_attempts(*attempts: dict) -> dict:
    ledger = copy.deepcopy(committed())
    ledger["attempts"] = [{**a, "attempt": i} for i, a in enumerate(attempts, start=1)]
    return ledger


@pytest.mark.parametrize(
    "change",
    [
        {"evidence": None},
        {"evidence": {"incident": "x", "time_utc": "t", "source": "s", "error_type": " "}},
        {"decision": None},
        {"failure_kind": "program defect"},
        {"failure_kind": "ac2"},
        {"failure_kind": None},
    ],
    ids=["no evidence", "blank error type", "no decision", "program defect", "ac2", "no kind"],
)
def test_infrastructure_interrupted_is_refused_without_its_conditions(change):
    with pytest.raises(LedgerError):
        ac1b_ledger.validate(with_attempts({**interrupted(), **change}))


def test_a_program_defect_can_only_be_fail():
    defect = copy.deepcopy(committed()["attempts"][1])
    for result in ("INFRASTRUCTURE INTERRUPTED", "INFRASTRUCTURE BLOCKED", "PASS"):
        with pytest.raises(LedgerError):
            ac1b_ledger.validate(with_attempts({**defect, "result": result}))


def test_failure_kind_is_an_enumeration():
    defect = copy.deepcopy(committed()["attempts"][1])
    with pytest.raises(LedgerError, match="failure_kind"):
        ac1b_ledger.validate(with_attempts({**defect, "failure_kind": "flaky"}))


def test_the_same_infrastructure_error_twice_in_a_row_must_be_blocked():
    first = interrupted()
    with pytest.raises(LedgerError, match="INFRASTRUCTURE BLOCKED"):
        ac1b_ledger.validate(with_attempts(first, interrupted()))
    with pytest.raises(LedgerError, match="INFRASTRUCTURE BLOCKED"):
        ac1b_ledger.validate(with_attempts(first, {**interrupted(), "result": "FAIL"}))
    blocked = {**interrupted(), "result": "INFRASTRUCTURE BLOCKED"}
    assert ac1b_ledger.validate(with_attempts(first, blocked))


def test_a_different_infrastructure_error_may_be_interrupted_again():
    other = interrupted()
    other["evidence"] = {**other["evidence"], "error_type": "HTTP 529 overloaded"}
    assert ac1b_ledger.validate(with_attempts(interrupted(), other))


def test_blocked_needs_infrastructure_evidence_too():
    blocked = {**interrupted(), "result": "INFRASTRUCTURE BLOCKED", "evidence": None}
    with pytest.raises(LedgerError):
        ac1b_ledger.validate(with_attempts(blocked))


def test_pass_needs_a_pass_comparison_and_no_failure_kind():
    good = {**copy.deepcopy(committed()["attempts"][1]), "result": "PASS"}
    with pytest.raises(LedgerError):
        ac1b_ledger.validate(with_attempts({**good, "failure_kind": None}))
    assert ac1b_ledger.validate(
        with_attempts({**good, "failure_kind": None, "comparison_verdict": "PASS"})
    )


def test_the_budget_states_that_only_a_pass_completes_ac1b():
    text = "\n".join(ac1b_ledger.markdown(ac1b_ledger.budget(committed())))
    assert ac1b_ledger.FINAL_RULE in text
    assert "neither an AC1b failure nor a pass" in ac1b_ledger.FINAL_RULE
    for doc in ac1b_ledger.BUDGET_DOCS:
        assert ac1b_ledger.FINAL_RULE in (ROOT / doc).read_text(encoding="utf-8")


def test_the_original_experiment_cost_and_the_cap_cannot_be_edited_in_the_ledger():
    for key, value in (("original_ac6_usd", 6.37), ("cap_usd_per_attempt", 10.0)):
        ledger = copy.deepcopy(committed())
        ledger[key] = value
        with pytest.raises(LedgerError):
            ac1b_ledger.validate(ledger)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attempt", 3),
        ("result", "PASSED"),
        ("reason", " "),
        ("haiku_usd", 5.01),
        ("haiku_usd", -1),
        ("comparison_verdict", "NOT STARTED"),
    ],
)
def test_a_malformed_attempt_is_refused(field, value):
    ledger = copy.deepcopy(committed())
    ledger["attempts"][1][field] = value
    with pytest.raises(LedgerError):
        ac1b_ledger.validate(ledger)


def test_the_readme_budget_blocks_match_the_ledger():
    """Rewrite with `python -m tinyrouter.ac1b_ledger` after editing attempts.json."""
    ledger = committed()
    for doc in ac1b_ledger.BUDGET_DOCS:
        text = (ROOT / doc).read_text(encoding="utf-8")
        assert ac1b_ledger.with_block(text, ledger) == text, doc
        assert "US$3.19" in text and "US$3.178751" in text


def test_the_total_is_only_reproduction_validation_spend():
    budget = ac1b_ledger.budget(committed())
    assert budget["original_ac6_experiment"]["usd"] == 3.19
    assert budget["reproduction_validation_total"]["usd"] == 3.178751
    assert "6.37" not in json.dumps(budget)


def test_the_copied_evidence_holds_no_key_or_home_path():
    for path in EVIDENCE.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert "sk-ant" not in text and "/Users/" not in text and "/private/" not in text, path


def test_each_new_commit_gets_its_own_journal_and_five_dollar_cap():
    """The cap is per reproduction id: a new id's Haiku step starts on an empty journal."""
    commands = {}
    for run_id in ("3992840ddb3f", "abcdef012345"):
        llm = next(s for s in full_steps(Layout(run_id)) if s.name == "llm")
        commands[run_id] = llm.command
        assert "MAX_USD=5" in llm.command
        assert f"RESULTS_ROOT=reproduction/{run_id}/results" in llm.command
    assert commands["3992840ddb3f"] != commands["abcdef012345"]


def test_spend_under_one_reproduction_id_does_not_count_against_another(tmp_path, monkeypatch):
    pytest.importorskip("anthropic")
    from llm_fakes import fake_client, fake_split
    from tinyrouter import llm, llm_run

    rows = {"validation": 3, "test": 2}
    monkeypatch.setattr(llm_run, "EXPECTED_ROWS", rows)
    splits = {name: fake_split(name, n + 2) for name, n in rows.items()}
    queries = llm_run.build_queries(splits, rows)
    cap = 5.5 * llm.cost_usd(262, llm.MAX_TOKENS)
    outcomes = []
    for run_id in ("first-attempt", "second-attempt"):
        target = llm_run.full_target(tmp_path / "reproduction" / run_id / "results")
        client = fake_client()
        outcome = llm_run.run(
            queries, client, target, max_usd=cap, sleep=lambda _: None, log=lambda _: None
        )
        outcomes.append((outcome, len(client.messages.calls)))
    for outcome, calls in outcomes:
        assert outcome.spent_before == 0.0
        assert calls == 5 and outcome.stopped is None
