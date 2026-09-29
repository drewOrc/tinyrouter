import numpy as np
import pytest

from tinyrouter.haiku import (
    HaikuMismatchError,
    check_gold,
    legacy_parse,
    split_arrays,
    summarize_split,
)
from tinyrouter.labels import AGENTS, load_label_space
from tinyrouter.llm import parse_agent

SPACE = load_label_space()


def record(index, raw, gold_intent, split="validation"):
    agent, parsed = parse_agent(raw)
    return {
        "split": split,
        "index": index,
        "gold_intent": gold_intent,
        "gold_agent": AGENTS[int(SPACE.intent_to_agent_id[gold_intent])],
        "raw_text": raw,
        "agent": agent,
        "parse_failed": not parsed,
        "cost_usd": 0.001,
    }


def gold_of(agent):
    return int(np.flatnonzero(SPACE.intent_to_agent_id == AGENTS.index(agent))[0])


def test_legacy_parse_takes_the_first_substring_hit_or_oos():
    assert legacy_parse(" Finance_Agent ") == ("finance_agent", 1)
    assert legacy_parse("no idea") == ("oos", 0)
    # The new rule calls this ambiguous (oos, parse_failed); the old one routed it.
    assert legacy_parse("oos (not travel_agent)") == ("travel_agent", 2)
    assert parse_agent("oos (not travel_agent)") == ("oos", False)


def fake_records():
    return [
        record(0, "finance_agent", gold_of("finance_agent")),
        record(1, "oos (not travel_agent)", gold_of("oos")),
        record(2, "travel_agent", gold_of("finance_agent")),
    ]


def test_summary_counts_parser_disagreements_and_parse_failures():
    summary = summarize_split(split_arrays(fake_records(), "validation", 3))
    assert summary["parsers_disagree_rows"] == 1
    assert summary["legacy_order_dependent_rows"] == 1
    assert summary["parse_failed_rows"] == 1
    assert summary["new_parser"]["accuracy_8"] == pytest.approx(2 / 3)
    assert summary["legacy_parser"]["accuracy_8"] == pytest.approx(1 / 3)
    # Row 1 is right under the new rule only because the failure fell back to oos.
    assert summary["accuracy_8_parse_failed_as_wrong"] == pytest.approx(1 / 3)
    assert summary["cost_usd_per_1k_queries"] == pytest.approx(1.0)


def test_split_arrays_needs_every_index_once():
    records = fake_records()
    with pytest.raises(HaikuMismatchError, match="indexes"):
        split_arrays(records[:1] + records[2:], "validation", 3)


def test_split_arrays_refuses_a_gold_agent_that_the_intent_does_not_map_to():
    records = fake_records()
    records[0]["gold_agent"] = "travel_agent"
    with pytest.raises(HaikuMismatchError, match="gold_agent"):
        split_arrays(records, "validation", 3)


def test_gold_must_equal_the_archive_labels_row_by_row():
    haiku = split_arrays(fake_records(), "validation", 3)
    check_gold(haiku, haiku.gold_intent.copy(), "fake.npz")
    shifted = np.roll(haiku.gold_intent, 1)
    with pytest.raises(HaikuMismatchError, match="fake.npz"):
        check_gold(haiku, shifted, "fake.npz")
    with pytest.raises(HaikuMismatchError):
        check_gold(haiku, haiku.gold_intent[:2], "fake.npz")
