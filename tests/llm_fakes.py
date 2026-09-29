"""A fake Anthropic client for the Haiku runner tests; nothing here touches the network."""

from __future__ import annotations

import threading
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np

from tinyrouter.data import Split


def api_error(status: int, message: str = "simulated", retry_after: str | None = None):
    """A real SDK status error (needs the ``llm`` group), as the client would raise it."""
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx2.Response(status, request=request, headers=headers)
    classes = {
        401: anthropic.AuthenticationError,
        429: anthropic.RateLimitError,
        500: anthropic.InternalServerError,
        529: anthropic.InternalServerError,
    }
    return classes.get(status, anthropic.APIStatusError)(message, response=response, body=None)


def connection_error():
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(request=request)


class FakeMessages:
    """``create`` replies ``reply(query)``; ``failures[query]`` is raised first, in order."""

    def __init__(
        self,
        reply: Callable[[str], str] = lambda q: "auto_agent",
        failures: dict[str, list[BaseException]] | None = None,
        input_tokens: int = 250,
        output_tokens: int = 4,
        base_tokens: int = 240,
    ) -> None:
        self.reply = reply
        self.failures = {k: list(v) for k, v in (failures or {}).items()}
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.base_tokens = base_tokens
        self.calls: list[dict] = []
        self.count_calls = 0
        self._lock = threading.Lock()

    def create(self, **kwargs):
        query = kwargs["messages"][0]["content"]
        with self._lock:
            self.calls.append(kwargs)
            pending = self.failures.get(query)
            error = pending.pop(0) if pending else None
        if error is not None:
            raise error
        n = len(self.calls)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.reply(query))],
            usage=SimpleNamespace(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=None,
            ),
            stop_reason="end_turn",
            _request_id=f"req_fake_{n}",
        )

    def count_tokens(self, **kwargs):
        self.count_calls += 1
        return SimpleNamespace(input_tokens=self.base_tokens)

    def queries_called(self) -> list[str]:
        return [c["messages"][0]["content"] for c in self.calls]


def fake_client(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(messages=FakeMessages(**kwargs))


def fake_split(name: str, n: int) -> Split:
    """``n`` rows named ``<name>-<i>``, alternating intents 0 and 1."""
    texts = tuple(f"{name}-{i}" for i in range(n))
    return Split(name, texts, np.array([i % 2 for i in range(n)], dtype=np.int64))  # type: ignore[arg-type]
