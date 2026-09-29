import json

import pytest

from tinyrouter import llm, llm_run
from tinyrouter.llm_run import IncompleteError, JournalError, RunLockedError, Target

pytest.importorskip("anthropic")

from llm_fakes import api_error, fake_client, fake_split  # noqa: E402

ROWS = {"validation": 3, "test": 2}
FAKE_KEY = "sk-ant-api03-FAKE-leak-canary-0123456789"
# Fake usage: 250 input + 4 output tokens per call.
CALL_COST = llm.cost_usd(250, 4)


@pytest.fixture
def small_rows(monkeypatch):
    """``main`` runs the official target; shrink it to ROWS so the fake splits cover it."""
    monkeypatch.setattr(llm_run, "EXPECTED_ROWS", ROWS)


def small_target(tmp_path, rows=ROWS) -> Target:
    return Target("haiku-test", tmp_path / "llm", rows, tmp_path / "llm-manifest.json")


def queries(rows=ROWS):
    splits = {name: fake_split(name, n + 2) for name, n in rows.items()}
    return llm_run.build_queries(splits, rows)


def run(tmp_path, client, rows=ROWS, **kwargs):
    kwargs.setdefault("max_usd", 5.0)
    kwargs.setdefault("sleep", lambda _: None)
    kwargs.setdefault("log", lambda _: None)
    return llm_run.run(queries(rows), client, small_target(tmp_path, rows), **kwargs)


def journal_lines(tmp_path, rows=ROWS):
    path = small_target(tmp_path, rows).journal(llm.identity_sha256())
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_build_queries_takes_the_first_rows_of_each_split_in_order():
    keys = [q.key for q in queries()]
    assert keys == [
        ("validation", 0),
        ("validation", 1),
        ("validation", 2),
        ("test", 0),
        ("test", 1),
    ]


def test_a_full_run_stores_every_field_and_passes_the_completion_check(tmp_path):
    client = fake_client(reply=lambda q: "Kitchen_Agent" if q.endswith("0") else "hmm")
    outcome = run(tmp_path, client)
    target = small_target(tmp_path)
    assert llm_run.finalize(target, outcome, 5.0) == 5
    records = [json.loads(line) for line in target.predictions.read_text().splitlines()]
    first = records[0]
    assert set(llm_run.RECORD_FIELDS) <= set(first)
    assert (first["split"], first["index"], first["agent"], first["parse_failed"]) == (
        "validation",
        0,
        "kitchen_agent",
        False,
    )
    assert first["raw_text"] == "Kitchen_Agent"
    assert first["request_id"].startswith("req_fake_")
    assert first["gold_intent"] == 0 and first["gold_agent"] in llm.AGENT_DESCRIPTIONS
    assert records[1]["agent"] == "oos" and records[1]["parse_failed"] is True
    assert "text" not in first and "validation-0" not in json.dumps(records)


def test_summary_and_manifest_record_spend_and_the_same_sha256(tmp_path):
    outcome = run(tmp_path, fake_client())
    target = small_target(tmp_path)
    llm_run.finalize(target, outcome, 5.0)
    summary = json.loads(target.summary.read_text())
    manifest = json.loads(target.manifest.read_text())["files"][target.predictions.name]
    assert summary["predictions_sha256"] == manifest["sha256"]
    assert summary["totals"]["calls"] == 5
    assert summary["totals"]["input_tokens"] == 5 * 250
    assert summary["totals"]["cost_usd"] == pytest.approx(5 * CALL_COST)
    assert summary["last_invocation"]["cost_usd"] == pytest.approx(5 * CALL_COST)
    assert summary["pricing_usd_per_mtok"]["input"] == 1.00


def test_a_rerun_calls_only_the_rows_without_a_stored_reply(tmp_path):
    # A cap that covers two calls' upper bound (base 250 + ~12 bytes, 20 out) stops the first run.
    first_cap = 2.2 * llm.cost_usd(250 + 12, llm.MAX_TOKENS)
    first = fake_client()
    outcome = run(tmp_path, first, max_usd=first_cap, workers=1)
    assert outcome.stopped == "cost cap" and len(outcome.records) == 2
    second = fake_client()
    outcome = run(tmp_path, second)
    called_first = set(first.messages.queries_called())
    called_second = set(second.messages.queries_called())
    assert not called_first & called_second
    assert len(called_first | called_second) == 5
    assert len(outcome.records) == 5 and outcome.spent_before == pytest.approx(2 * CALL_COST)


def test_a_complete_journal_makes_no_calls_at_all(tmp_path):
    run(tmp_path, fake_client())
    again = fake_client()
    outcome = run(tmp_path, again)
    assert again.messages.calls == [] and again.messages.count_calls == 0
    assert len(outcome.records) == 5


def test_changing_max_tokens_does_not_reuse_stored_replies(tmp_path, monkeypatch):
    run(tmp_path, fake_client())
    monkeypatch.setattr(llm, "MAX_TOKENS", 21)
    client = fake_client()
    outcome = run(tmp_path, client)
    assert len(client.messages.calls) == 5
    assert all(c["max_tokens"] == 21 for c in client.messages.calls)
    assert outcome.spent_before == 0


def test_changing_the_prompt_does_not_reuse_stored_replies(tmp_path, monkeypatch):
    run(tmp_path, fake_client())
    monkeypatch.setattr(llm, "SYSTEM_PROMPT", llm.SYSTEM_PROMPT.replace("ONE", "one"))
    client = fake_client()
    run(tmp_path, client)
    assert len(client.messages.calls) == 5


def test_a_torn_last_journal_line_is_redone_and_trimmed(tmp_path):
    run(tmp_path, fake_client(), workers=1)
    path = small_target(tmp_path).journal(llm.identity_sha256())
    text = path.read_text()
    path.write_text(text[: text.rstrip("\n").rfind("\n") + 1] + '{"split": "te')
    client = fake_client()
    outcome = run(tmp_path, client)
    assert client.messages.queries_called() == ["test-1"]
    assert len(outcome.records) == 5
    stored = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["split"], r["index"]) for r in stored][-1] == ("test", 1)
    assert (
        len(llm_run.load_journal(path, {q.key: q for q in queries()}, llm.identity_sha256())) == 5
    )


def test_a_journal_record_that_does_not_match_its_dataset_row_is_refused(tmp_path):
    run(tmp_path, fake_client())
    path = small_target(tmp_path).journal(llm.identity_sha256())
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["query_sha256"] = "0" * 64
    path.write_text("\n".join([json.dumps(record), *lines[1:]]) + "\n")
    with pytest.raises(JournalError, match="does not match the dataset row"):
        run(tmp_path, fake_client())


def test_the_cost_cap_is_never_passed_with_calls_in_flight(tmp_path):
    rows = {"validation": 40}
    cap = 10 * llm.cost_usd(240 + 20, llm.MAX_TOKENS)
    client = fake_client(input_tokens=240 + 12, output_tokens=llm.MAX_TOKENS, base_tokens=240)
    outcome = run(tmp_path, client, rows=rows, max_usd=cap, workers=8)
    assert outcome.stopped == "cost cap"
    assert 0 < len(client.messages.calls) < 40
    assert outcome.spent_now <= cap
    assert sum(r["cost_usd"] for r in journal_lines(tmp_path, rows)) <= cap


def test_the_cap_counts_what_earlier_runs_spent(tmp_path):
    run(tmp_path, fake_client(), max_usd=2.2 * llm.cost_usd(262, llm.MAX_TOKENS), workers=1)
    client = fake_client()
    outcome = run(tmp_path, client, max_usd=2 * CALL_COST)
    assert client.messages.calls == [] and outcome.stopped == "cost cap"


def test_main_exits_1_and_reports_spend_when_the_cap_stops_the_run(small_rows, tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        main(tmp_path, ["--max-usd", "0.0005", "--workers", "1"], fake_client())
    assert info.value.code == 1
    out = capsys.readouterr().out
    assert "cost cap" in out and "this run: 1 calls" in out
    assert "completed" not in out


def test_a_call_that_runs_out_of_retries_is_recorded_and_the_run_exits_1(
    small_rows, tmp_path, capsys
):
    client = fake_client(failures={"test-1": [api_error(500)] * 99})
    with pytest.raises(SystemExit) as info:
        main(tmp_path, [], client)
    assert info.value.code == 1
    failed = [r for r in journal_lines_full(tmp_path) if r.get("status") == "failed"]
    assert [(r["split"], r["index"], r["attempts"]) for r in failed] == [
        ("test", 1, llm.MAX_ATTEMPTS)
    ]
    assert "1 call(s) failed after retries" in capsys.readouterr().out
    # A rerun retries only that row, and then completes.
    retry = fake_client()
    main(tmp_path, [], retry)
    assert retry.messages.queries_called() == ["test-1"]


def test_a_non_retryable_error_stops_starting_new_calls(small_rows, tmp_path):
    client = fake_client(failures={"validation-0": [api_error(401)]})
    with pytest.raises(SystemExit):
        main(tmp_path, ["--workers", "1"], client)
    assert client.messages.queries_called() == ["validation-0"]


def test_main_prints_completed_only_after_the_checks(small_rows, tmp_path, capsys):
    main(tmp_path, [], fake_client())
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "completed 5/5 llm predictions"
    main(tmp_path, ["--verify"], None)
    assert capsys.readouterr().out.strip() == "completed 5/5 llm predictions"


def test_smoke_writes_its_own_files_and_leaves_the_official_ones_alone(tmp_path, capsys):
    client = fake_client()
    llm_run.main(
        ["--smoke", "--results-root", str(tmp_path)],
        client_factory=lambda: client,
        load=lambda name: fake_split(name, 25),
        sleep=lambda _: None,
    )
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "completed 20/20 llm smoke predictions"
    assert f"at that rate 8600 queries cost US${8600 * CALL_COST:.2f}" in out
    assert len(client.messages.calls) == 20
    assert (tmp_path / "llm-smoke" / "haiku-8way-smoke.jsonl").exists()
    assert not (tmp_path / "llm").exists() and not (tmp_path / "llm-manifest.json").exists()


def test_official_expected_rows_are_the_full_validation_and_test_splits():
    assert llm_run.EXPECTED_ROWS == {"validation": 3100, "test": 5500}
    assert llm_run.FULL_QUERIES == sum(llm_run.EXPECTED_ROWS.values()) == 8600


def completed_target(tmp_path) -> Target:
    target = small_target(tmp_path)
    llm_run.finalize(target, run(tmp_path, fake_client()), 5.0)
    return target


def test_completion_check_fails_when_one_row_is_missing(tmp_path):
    target = completed_target(tmp_path)
    lines = target.predictions.read_text().splitlines()
    target.predictions.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(IncompleteError, match=r"missing \[\('test', 1\)\]"):
        llm_run.check_predictions(target.predictions, ROWS, llm.identity_sha256())


def test_completion_check_fails_when_one_row_is_duplicated(tmp_path):
    target = completed_target(tmp_path)
    lines = target.predictions.read_text().splitlines()
    target.predictions.write_text("\n".join([*lines, lines[0]]) + "\n")
    with pytest.raises(IncompleteError, match=r"duplicated \[\('validation', 0\)\]"):
        llm_run.check_predictions(target.predictions, ROWS, llm.identity_sha256())


def test_completion_check_fails_on_a_row_from_another_identity(tmp_path):
    target = completed_target(tmp_path)
    lines = target.predictions.read_text().splitlines()
    record = json.loads(lines[0])
    record["identity_sha256"] = "f" * 64
    target.predictions.write_text("\n".join([json.dumps(record), *lines[1:]]) + "\n")
    with pytest.raises(IncompleteError, match="identity"):
        llm_run.check_predictions(target.predictions, ROWS, llm.identity_sha256())


def test_verify_fails_when_the_file_changed_after_the_manifest_was_written(tmp_path):
    target = completed_target(tmp_path)
    lines = target.predictions.read_text().splitlines()
    record = json.loads(lines[0])
    record["latency_ms"] += 1
    target.predictions.write_text("\n".join([json.dumps(record), *lines[1:]]) + "\n")
    with pytest.raises(IncompleteError, match="SHA-256 differs"):
        llm_run.verify(target)


def test_missing_key_exits_2_with_a_clear_message_before_loading_data(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as info:
        llm_run.main(
            ["--results-root", str(tmp_path)],
            load=lambda name: pytest.fail("loaded data without a key"),
        )
    assert info.value.code == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_the_key_never_reaches_output_journal_or_exceptions(
    small_rows, tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    leaky = api_error(401, message=f"invalid x-api-key {FAKE_KEY}")
    client = fake_client(failures={"validation-0": [leaky]})
    with pytest.raises(SystemExit):
        main(tmp_path, ["--workers", "1"], client)
    captured = capsys.readouterr()
    assert "invalid x-api-key [redacted]" in captured.out
    assert FAKE_KEY not in captured.out + captured.err
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert FAKE_KEY not in path.read_text()
    with pytest.raises(llm.LLMCallError) as info:
        llm.classify("validation-0", fake_client(failures={"validation-0": [leaky]}))
    assert FAKE_KEY not in str(info.value) and info.value.__cause__ is None
    assert FAKE_KEY not in repr(info.value)


def test_workers_and_cap_arguments_are_bounded():
    with pytest.raises(SystemExit):
        llm_run.parse_args(["--workers", "9"])
    with pytest.raises(SystemExit):
        llm_run.parse_args(["--max-usd", "0"])


def main(tmp_path, argv, client):
    llm_run.main(
        [*argv, "--results-root", str(tmp_path)],
        client_factory=lambda: client,
        load=lambda name: fake_split(name, 4),
        sleep=lambda _: None,
    )


def journal_lines_full(tmp_path):
    target = llm_run.full_target(tmp_path)
    path = target.journal(llm.identity_sha256())
    return [json.loads(line) for line in path.read_text().splitlines()]


def rewrite_first_journal_record(tmp_path, **changes):
    path = small_target(tmp_path).journal(llm.identity_sha256())
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record.update(changes)
    path.write_text("\n".join([json.dumps(record), *lines[1:]]) + "\n")
    return path, lines


def test_a_journal_row_of_another_identity_is_refused(tmp_path):
    run(tmp_path, fake_client())
    rewrite_first_journal_record(tmp_path, identity_sha256="e" * 64)
    with pytest.raises(JournalError, match="identity"):
        run(tmp_path, fake_client())


def test_a_journal_row_whose_gold_label_changed_is_refused(tmp_path):
    run(tmp_path, fake_client())
    rewrite_first_journal_record(tmp_path, gold_intent=7)
    with pytest.raises(JournalError, match="gold label"):
        run(tmp_path, fake_client())


def test_a_row_recorded_twice_in_the_journal_is_refused(tmp_path):
    run(tmp_path, fake_client())
    path = small_target(tmp_path).journal(llm.identity_sha256())
    first = path.read_text().splitlines()[0]
    with path.open("a") as fh:
        fh.write(first + "\n")
    with pytest.raises(JournalError, match="recorded twice"):
        run(tmp_path, fake_client())


def test_verify_fails_when_only_the_manifest_sha256_changed(tmp_path):
    target = completed_target(tmp_path)
    body = json.loads(target.manifest.read_text())
    body["files"][target.predictions.name]["sha256"] = "0" * 64
    target.manifest.write_text(json.dumps(body))
    with pytest.raises(IncompleteError, match="SHA-256 differs"):
        llm_run.verify(target)


def test_the_cost_bound_counts_utf8_bytes_not_characters(tmp_path):
    """A non-ASCII query at its worst case (one token per byte) is inside its bound and the cap."""
    rows = {"validation": 30}
    prefix = "\u00e9" * 10  # 10 characters, 20 bytes; the query is e.g. "éééééééééé-7"
    query = llm_run.Query("validation", 0, f"{prefix}-0", 0)
    assert llm_run.input_token_bound(query, 240) == 240 + 22
    splits = {"validation": fake_split("validation", 30, prefix=prefix)}
    worst = 240 + len(f"{prefix}-0".encode())
    client = fake_client(input_tokens=worst, output_tokens=llm.MAX_TOKENS, base_tokens=240)
    cap = 7.5 * llm.cost_usd(worst, llm.MAX_TOKENS)
    outcome = llm_run.run(
        llm_run.build_queries(splits, rows),
        client,
        small_target(tmp_path, rows),
        max_usd=cap,
        workers=8,
        sleep=lambda _: None,
        log=lambda _: None,
    )
    assert outcome.stopped == "cost cap"
    assert outcome.spent_now <= cap


@pytest.mark.parametrize(("extra_in", "extra_out"), [(1, 0), (0, 1)])
def test_a_response_over_its_bound_stops_the_run_and_exits_1(
    small_rows, tmp_path, capsys, extra_in, extra_out
):
    # validation-0 is 12 bytes; base 250.
    client = fake_client(input_tokens=250 + 12 + extra_in, output_tokens=llm.MAX_TOKENS + extra_out)
    with pytest.raises(SystemExit) as info:
        main(tmp_path, ["--workers", "1"], client)
    assert info.value.code == 1
    assert "bound violated" in capsys.readouterr().out
    assert len(client.messages.calls) == 1
    stored = journal_lines_full(tmp_path)
    assert len(stored) == 1 and "status" not in stored[0]


def test_a_second_runner_on_the_same_target_is_refused(tmp_path):
    target = small_target(tmp_path)
    client = fake_client()
    with llm_run.exclusive(target), pytest.raises(RunLockedError):
        run(tmp_path, client)
    assert client.messages.calls == []
    assert len(run(tmp_path, client).records) == 5


def test_main_exits_2_when_another_process_holds_the_lock(small_rows, tmp_path, capsys):
    import subprocess
    import sys

    target = llm_run.full_target(tmp_path)
    target.out_dir.mkdir(parents=True)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys, time; f = open(sys.argv[1], 'a'); "
            "fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True); time.sleep(30)",
            str(target.lock),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        client = fake_client()
        with pytest.raises(SystemExit) as info:
            main(tmp_path, [], client)
        assert info.value.code == 2 and client.messages.calls == []
        assert "held by another process" in capsys.readouterr().err
    finally:
        holder.kill()
        holder.wait()


def test_a_whole_last_record_without_its_newline_is_kept(tmp_path):
    run(tmp_path, fake_client(), workers=1)
    path = small_target(tmp_path).journal(llm.identity_sha256())
    path.write_text(path.read_text().rstrip("\n"))
    client = fake_client()
    outcome = run(tmp_path, client)
    assert client.messages.calls == [] and len(outcome.records) == 5
    assert path.read_text().endswith("\n") and len(path.read_text().splitlines()) == 5


def test_an_unexpected_exception_is_journaled_as_a_failure_and_stops_the_run(small_rows, tmp_path):
    client = fake_client()
    client.messages.response_override = object()  # no .content or .usage
    with pytest.raises(SystemExit) as info:
        main(tmp_path, ["--workers", "1"], client)
    assert info.value.code == 1
    failed = journal_lines_full(tmp_path)
    assert len(failed) == 1 and failed[0]["status"] == "failed"
    assert "AttributeError" in failed[0]["error"]


def test_calls_in_flight_at_an_interrupt_are_journaled(tmp_path, monkeypatch):
    real_wait = llm_run.wait
    state = {"calls": 0}

    def interrupted_wait(fs, return_when):
        state["calls"] += 1
        if state["calls"] == 1:
            raise KeyboardInterrupt
        return real_wait(fs, return_when=return_when)

    monkeypatch.setattr(llm_run, "wait", interrupted_wait)
    client = fake_client()
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, client, workers=3)
    assert len(client.messages.calls) == 3
    assert len(journal_lines(tmp_path)) == 3


def test_token_count_failure_exits_1_without_the_key(small_rows, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    client = fake_client()
    client.messages.count_failures = [api_error(401, f"invalid x-api-key {FAKE_KEY}")]
    with pytest.raises(SystemExit) as info:
        main(tmp_path, [], client)
    assert info.value.code == 1
    captured = capsys.readouterr()
    assert "token count failed" in captured.out
    assert FAKE_KEY not in captured.out + captured.err
    assert client.messages.calls == []


def test_summary_records_the_parser_hash_and_the_cap_scope(tmp_path):
    target = completed_target(tmp_path)
    summary = json.loads(target.summary.read_text())
    assert summary["parser_sha256"] == llm.parser_sha256()
    assert "smoke" in summary["cap_scope"]


def test_a_bound_violation_on_the_last_row_still_exits_1(small_rows, tmp_path, capsys):
    """Every row has a reply, but one used more tokens than its bound: not a clean finish."""
    client = fake_client(
        input_tokens_for=lambda q: 250 + len(q.encode()) + (1 if q == "test-1" else 0)
    )
    with pytest.raises(SystemExit) as info:
        main(tmp_path, ["--workers", "1"], client)
    assert info.value.code == 1
    out = capsys.readouterr().out
    assert "every row has a reply, but the run stopped: bound violated" in out
    assert "completed" not in out
    assert len(client.messages.calls) == 5
