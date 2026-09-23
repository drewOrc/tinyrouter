"""Claude Haiku zero-shot router: the LLM baseline and cascade fallback.

Prompt and model carried over from cost-aware-hybrid-router
(src/routers/llm_router.py) so the two projects score the same LLM. Nothing
in this skeleton calls the API; ``classify`` needs the ``llm`` dependency
group and ANTHROPIC_API_KEY, and the unit tests exercise it with a fake
client only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from tinyrouter.labels import AGENTS, OOS

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 20
MAX_ATTEMPTS = 5

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


@dataclass(frozen=True)
class LLMPrediction:
    agent: str
    raw_text: str
    parsed: bool
    input_tokens: int
    output_tokens: int
    latency_ms: int


def parse_agent(raw_text: str) -> tuple[str, bool]:
    """Map the model's reply to an agent; unparseable replies become oos and are flagged."""
    text = raw_text.strip().strip("\"'`.").lower()
    if text in AGENTS:
        return text, True
    hits = [agent for agent in AGENTS if agent != OOS and agent in text]
    if len(hits) == 1:
        return hits[0], True
    return OOS, False


def classify(query: str, client: Any, sleep: Any = time.sleep) -> LLMPrediction:
    """One zero-shot call with exponential backoff on API errors."""
    for attempt in range(MAX_ATTEMPTS):
        started = time.monotonic()
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
            )
        except Exception as exc:  # the SDK raises several error types; all are retryable here
            if attempt == MAX_ATTEMPTS - 1:
                raise RuntimeError(f"LLM call failed after {MAX_ATTEMPTS} attempts") from exc
            sleep(2**attempt)
            continue
        raw_text = response.content[0].text
        agent, parsed = parse_agent(raw_text)
        return LLMPrediction(
            agent=agent,
            raw_text=raw_text,
            parsed=parsed,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=round((time.monotonic() - started) * 1000),
        )
    raise AssertionError("unreachable")


def make_client() -> Any:
    """Anthropic client from ANTHROPIC_API_KEY; requires `uv sync --group llm`."""
    import anthropic

    return anthropic.Anthropic()
