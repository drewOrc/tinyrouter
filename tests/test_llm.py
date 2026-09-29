import hashlib

import pytest

from tinyrouter import llm
from tinyrouter.llm import SYSTEM_PROMPT, classify, parse_agent

anthropic = pytest.importorskip("anthropic")

from llm_fakes import api_error, connection_error, fake_client  # noqa: E402

# SHA-256 of SYSTEM_PROMPT in cost-aware-hybrid-router, src/routers/llm_router.py
# (github.com/drewOrc/cost-aware-hybrid-router, main, fetched 2026-09-29): the
# prompt there is built from the same AGENT_DESCRIPTIONS by the same join.
OLD_PROJECT_PROMPT_SHA256 = "560d22c59164f1125e4ec787f9002b4c9bc6bf03b4cc78472550ae850d5df574"


@pytest.mark.parametrize(
    ("reply", "agent", "parsed"),
    [
        ("finance_agent", "finance_agent", True),
        ('  "Travel_Agent".  ', "travel_agent", True),
        ("\n  DEVICE_AGENT \n", "device_agent", True),
        ("oos", "oos", True),
        ("OOS.", "oos", True),
        ("I think kitchen_agent", "kitchen_agent", True),
        ("finance_agent or travel_agent", "oos", False),
        ("no idea", "oos", False),
        ("", "oos", False),
        ("oos (not travel_agent)", "oos", False),
        ("This is oos, not travel_agent", "oos", False),
        ("travel_agent.", "travel_agent", True),
        ("`travel_agent`", "travel_agent", True),
        ("**travel_agent**", "travel_agent", True),
        ("travel_agent\nThe query asks about flights.", "travel_agent", True),
        ("**oos**", "oos", True),
        ("travel agent", "oos", False),
        ("out of scope", "oos", False),
        ("noose", "oos", False),
    ],
)
def test_parse_agent(reply, agent, parsed):
    assert parse_agent(reply) == (agent, parsed)


def test_system_prompt_is_byte_identical_to_the_old_project():
    digest = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert digest == OLD_PROJECT_PROMPT_SHA256


def test_system_prompt_lists_all_eight_labels():
    for label in ("finance_agent", "meta_agent", "oos"):
        assert f"- {label}:" in SYSTEM_PROMPT


def test_request_matches_the_old_project_settings():
    params = llm.request_params("book a flight")
    assert params == {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 20,
        "temperature": 0,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": "book a flight"}],
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MODEL", "claude-haiku-4-5"),
        ("MAX_TOKENS", 21),
        ("TEMPERATURE", 1),
        ("SYSTEM_PROMPT", SYSTEM_PROMPT + " "),
    ],
)
def test_identity_changes_when_any_request_setting_changes(monkeypatch, name, value):
    before = llm.identity_sha256()
    monkeypatch.setattr(llm, name, value)
    assert llm.identity_sha256() != before


def test_cost_uses_haiku_rates_per_million_tokens():
    assert llm.cost_usd(1_000_000, 0) == pytest.approx(1.00)
    assert llm.cost_usd(0, 1_000_000) == pytest.approx(5.00)
    assert llm.cost_usd(250, 4) == pytest.approx(250e-6 + 20e-6)


def test_classify_returns_prediction_with_usage_and_request_id():
    client = fake_client()
    result = classify("when is my oil change due", client, sleep=lambda _: None)
    assert result.agent == "auto_agent" and result.parsed
    assert (result.input_tokens, result.output_tokens) == (250, 4)
    assert result.request_id == "req_fake_1"
    assert result.attempts == 1
    assert client.messages.calls[-1]["temperature"] == 0


def test_classify_retries_a_429_then_succeeds_honouring_retry_after():
    client = fake_client(failures={"hi": [api_error(429, retry_after="7"), api_error(529)]})
    sleeps: list[float] = []
    result = classify("hi", client, sleep=sleeps.append)
    assert result.attempts == 3
    assert sleeps == [7.0, 2.0]


def test_classify_retries_connection_errors():
    client = fake_client(failures={"hi": [connection_error()]})
    assert classify("hi", client, sleep=lambda _: None).attempts == 2


def test_classify_gives_up_after_max_attempts_as_a_retryable_failure():
    client = fake_client(failures={"hi": [api_error(500)] * 99})
    sleeps: list[float] = []
    with pytest.raises(llm.LLMCallError) as info:
        classify("hi", client, sleep=sleeps.append)
    assert info.value.attempts == llm.MAX_ATTEMPTS
    assert info.value.retryable
    assert sleeps == [1.0, 2.0, 4.0, 8.0]
    assert len(client.messages.calls) == llm.MAX_ATTEMPTS


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_classify_does_not_retry_client_errors(status):
    client = fake_client(failures={"hi": [api_error(status)]})
    with pytest.raises(llm.LLMCallError) as info:
        classify("hi", client, sleep=lambda _: pytest.fail("slept on a non-retryable error"))
    assert not info.value.retryable
    assert len(client.messages.calls) == 1


def test_retry_delay_is_capped():
    assert llm.retry_delay(api_error(429, retry_after="3600"), 1) == llm.MAX_BACKOFF_S
    assert llm.retry_delay(api_error(429, retry_after="soon"), 3) == 4.0


def test_make_client_refuses_without_a_key_and_names_the_variable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(llm.MissingKeyError, match="ANTHROPIC_API_KEY is not set"):
        llm.make_client()


def test_make_client_turns_sdk_retries_off(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    assert llm.make_client().max_retries == 0


def test_parser_hash_follows_the_parser_source(monkeypatch):
    before = llm.parser_sha256()
    monkeypatch.setattr(llm, "OOS_WORD", __import__("re").compile(r"oos"))
    assert llm.parser_sha256() != before


def test_token_count_is_retried_and_its_errors_redacted(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-FAKE-count-canary-99")
    client = fake_client()
    client.messages.count_failures = [api_error(529), connection_error()]
    sleeps: list[float] = []
    assert llm.prompt_base_tokens(client, sleeps.append) == client.messages.base_tokens
    assert sleeps == [1.0, 2.0]
    client.messages.count_failures = [api_error(401, "bad key sk-ant-api03-FAKE-count-canary-99")]
    with pytest.raises(llm.LLMCallError) as info:
        llm.prompt_base_tokens(client, sleeps.append)
    assert "canary" not in str(info.value) and info.value.__cause__ is None
