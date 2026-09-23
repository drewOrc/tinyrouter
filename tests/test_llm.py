from types import SimpleNamespace

import pytest

from tinyrouter.llm import SYSTEM_PROMPT, classify, parse_agent


@pytest.mark.parametrize(
    ("reply", "agent", "parsed"),
    [
        ("finance_agent", "finance_agent", True),
        ('  "Travel_Agent".  ', "travel_agent", True),
        ("oos", "oos", True),
        ("I think kitchen_agent", "kitchen_agent", True),
        ("finance_agent or travel_agent", "oos", False),
        ("no idea", "oos", False),
    ],
)
def test_parse_agent(reply, agent, parsed):
    assert parse_agent(reply) == (agent, parsed)


def test_system_prompt_lists_all_eight_labels():
    for label in ("finance_agent", "meta_agent", "oos"):
        assert f"- {label}:" in SYSTEM_PROMPT


class FakeMessages:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise ConnectionError("simulated")
        return SimpleNamespace(
            content=[SimpleNamespace(text="auto_agent")],
            usage=SimpleNamespace(input_tokens=120, output_tokens=3),
        )


def test_classify_retries_then_returns_prediction_with_token_counts():
    fake = SimpleNamespace(messages=FakeMessages(failures=2))
    sleeps: list[float] = []
    result = classify("when is my oil change due", fake, sleep=sleeps.append)
    assert result.agent == "auto_agent"
    assert (result.input_tokens, result.output_tokens) == (120, 3)
    assert sleeps == [1, 2]
    assert fake.messages.calls[-1]["temperature"] == 0


def test_classify_gives_up_after_max_attempts():
    fake = SimpleNamespace(messages=FakeMessages(failures=99))
    with pytest.raises(RuntimeError, match="5 attempts"):
        classify("hi", fake, sleep=lambda _: None)
