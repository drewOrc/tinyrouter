"""Claude Haiku zero-shot router: the LLM baseline and cascade fallback.

Prompt and model carried over from cost-aware-hybrid-router
(src/routers/llm_router.py) so the two projects score the same LLM: the
system prompt is byte-identical (tests/test_llm.py pins its SHA-256), the
query is sent verbatim as the only user message, temperature 0,
max_tokens 20. The ``anthropic`` package (``llm`` dependency group) is
imported only when a client is built or an API error is classified.

Retries are done here, not by the SDK (the client is built with
``max_retries=0``), so the attempt count and the delays are recorded and
testable. 429, 5xx (529 overloaded included), 408, 409 and connection
errors or timeouts are retried with exponential backoff, honouring
``retry-after``; anything else (400, 401, 403, 404, ...) fails at once.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tinyrouter.labels import AGENTS, OOS

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 20
TEMPERATURE = 0
MAX_ATTEMPTS = 5
MAX_BACKOFF_S = 60.0
RETRYABLE_STATUS = frozenset({408, 409, 429})
KEY_ENV = "ANTHROPIC_API_KEY"

# USD per million tokens for Claude Haiku 4.5, from the claude-api skill's
# "Current Models" table (cached 2026-06-24; first-party API rates), which
# points to https://platform.claude.com/docs/en/about-claude/pricing.md.
# Cache multipliers (write 1.25x, read 0.1x of input) are from the same
# skill; this prompt is not cached, so those two should stay at zero.
PRICE_USD_PER_MTOK: dict[str, float] = {
    "input": 1.00,
    "output": 5.00,
    "cache_write": 1.25,
    "cache_read": 0.10,
}
PRICING_SOURCE = (
    "claude-api skill, Current Models table (cached 2026-06-24): Claude Haiku 4.5 "
    "$1.00 input / $5.00 output per MTok; https://platform.claude.com/docs/en/about-claude/pricing.md"
)

AGENT_DESCRIPTIONS: dict[str, str] = {
    "finance_agent": "Banking, credit cards, accounts, bills, taxes, insurance, rewards, "
    "credit score, transfers, payments",
    "travel_agent": "Flights, hotels, reservations, visas, luggage, travel alerts, vaccines, "
    "plug types",
    "auto_agent": "Vehicle maintenance, oil change, tires, gas, car rental, uber, directions, "
    "traffic, distance",
    "kitchen_agent": "Recipes, nutrition, calories, ingredients, meal suggestions, restaurants, "
    "shopping lists, food storage",
    "productivity_agent": "Calendar, reminders, alarms, timers, todo lists, PTO, meetings, "
    "weather, translate, spelling, calculator, time, orders",
    "device_agent": "Smart home, music playback, playlists, volume, device settings, sync, "
    "language/accent changes, user name",
    "meta_agent": "Greetings, goodbye, jokes, fun facts, bot identity questions, yes/no/maybe, "
    "coin flip, dice roll, hobbies, cancel, repeat",
    OOS: "Out-of-scope: queries that don't fit any of the above categories",
}

SYSTEM_PROMPT = (
    "You are a query router for a multi-agent system. Given a user query, classify it into "
    "exactly ONE of these agent categories:\n\n"
    + "\n".join(f"- {agent}: {desc}" for agent, desc in AGENT_DESCRIPTIONS.items())
    + "\n\nRules:\n"
    '1. Reply with ONLY the agent name (e.g., "finance_agent"), nothing else.\n'
    "2. If the query is ambiguous, nonsensical, or doesn't clearly fit any agent, "
    'reply "oos".\n'
    "3. Do not explain your reasoning."
)


class MissingKeyError(RuntimeError):
    """ANTHROPIC_API_KEY is not set."""


class LLMCallError(RuntimeError):
    """A call failed for good: a non-retryable error, or retries used up."""

    def __init__(self, detail: str, *, attempts: int, retryable: bool) -> None:
        super().__init__(detail)
        self.detail = detail
        self.attempts = attempts
        self.retryable = retryable


@dataclass(frozen=True)
class LLMPrediction:
    agent: str
    raw_text: str
    parsed: bool
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    request_id: str | None = None
    stop_reason: str | None = None
    attempts: int = 1

    @property
    def cost_usd(self) -> float:
        return cost_usd(
            self.input_tokens,
            self.output_tokens,
            self.cache_creation_input_tokens,
            self.cache_read_input_tokens,
        )


def cost_usd(
    input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0
) -> float:
    """Dollar cost of one call's usage at ``PRICE_USD_PER_MTOK``."""
    p = PRICE_USD_PER_MTOK
    total = (
        input_tokens * p["input"]
        + output_tokens * p["output"]
        + cache_write * p["cache_write"]
        + cache_read * p["cache_read"]
    )
    return total / 1_000_000


def request_params(query: str) -> dict[str, Any]:
    """The exact Messages API arguments for one query; the single place they are built."""
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": query}],
    }


def identity() -> dict[str, object]:
    """Everything that decides what the model is asked; a change invalidates stored replies."""
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "user_content": "query text verbatim, one user message",
    }


def identity_sha256() -> str:
    canonical = json.dumps(identity(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_agent(raw_text: str) -> tuple[str, bool]:
    """Map the model's reply to an agent; unparseable replies become oos and are flagged."""
    text = raw_text.strip().strip("\"'`.").lower()
    if text in AGENTS:
        return text, True
    hits = [agent for agent in AGENTS if agent != OOS and agent in text]
    if len(hits) == 1:
        return hits[0], True
    return OOS, False


def redact(text: str) -> str:
    """``text`` with the API key's value, if set and present, replaced."""
    key = os.environ.get(KEY_ENV, "").strip()
    if len(key) >= 8:
        text = text.replace(key, "[redacted]")
    return text


def describe_error(exc: BaseException) -> str:
    """Class, status, request id and message of an API error, with the key redacted."""
    parts = [type(exc).__name__]
    for attr in ("status_code", "request_id"):
        value = getattr(exc, attr, None)
        if value is not None:
            parts.append(f"{attr}={value}")
    return redact(f"{' '.join(parts)}: {exc}")[:500]


def is_retryable(exc: BaseException) -> bool:
    """Rate limits, server errors, 408/409 and network failures; see the module docstring."""
    import anthropic

    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in RETRYABLE_STATUS or exc.status_code >= 500
    return False


def retry_delay(exc: BaseException, attempt: int) -> float:
    """Seconds to wait after failed attempt ``attempt`` (1-based): 1, 2, 4, ... or retry-after."""
    delay = float(2 ** (attempt - 1))
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}).get("retry-after") if response is not None else None
    try:
        delay = max(delay, float(header)) if header is not None else delay
    except ValueError:
        pass
    return min(delay, MAX_BACKOFF_S)


def reply_text(response: Any) -> str:
    """Concatenated text blocks; an empty reply (no text block) is the empty string."""
    return "".join(
        block.text for block in response.content if getattr(block, "type", "text") == "text"
    )


def classify(
    query: str,
    client: Any,
    sleep: Callable[[float], None] = time.sleep,
    retryable: Callable[[BaseException], bool] = is_retryable,
) -> LLMPrediction:
    """One zero-shot call, retried per the module docstring; LLMCallError when it gives up."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            response = client.messages.create(**request_params(query))
        except Exception as exc:  # classified below; anything unknown is not retried
            can_retry = retryable(exc)
            if not can_retry or attempt == MAX_ATTEMPTS:
                raise LLMCallError(
                    describe_error(exc), attempts=attempt, retryable=can_retry
                ) from None
            sleep(retry_delay(exc, attempt))
            continue
        latency_ms = round((time.monotonic() - started) * 1000)
        return prediction_from(response, latency_ms, attempt)
    raise AssertionError("unreachable")


def prediction_from(response: Any, latency_ms: int, attempts: int) -> LLMPrediction:
    raw_text = reply_text(response)
    agent, parsed = parse_agent(raw_text)
    usage = response.usage
    return LLMPrediction(
        agent=agent,
        raw_text=raw_text,
        parsed=parsed,
        input_tokens=int(usage.input_tokens),
        output_tokens=int(usage.output_tokens),
        latency_ms=latency_ms,
        cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        request_id=getattr(response, "_request_id", None),
        stop_reason=getattr(response, "stop_reason", None),
        attempts=attempts,
    )


def prompt_base_tokens(client: Any) -> int:
    """Input tokens of the system prompt plus a one-character query (token counting endpoint)."""
    counted = client.messages.count_tokens(
        model=MODEL, system=SYSTEM_PROMPT, messages=[{"role": "user", "content": "x"}]
    )
    return int(counted.input_tokens)


def make_client() -> Any:
    """Anthropic client from ANTHROPIC_API_KEY with SDK retries off; needs the ``llm`` group."""
    if not os.environ.get(KEY_ENV, "").strip():
        raise MissingKeyError(
            f"{KEY_ENV} is not set. Put it in a .env file at the repository root "
            "(gitignored; the Makefile passes it to uv) or export it, then rerun."
        )
    import anthropic

    return anthropic.Anthropic(max_retries=0)
