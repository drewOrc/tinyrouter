import json
from importlib import resources

import numpy as np
import pytest

from tinyrouter.labels import AGENTS, OOS, build_label_space, load_label_space


def raw_mapping() -> dict:
    text = resources.files("tinyrouter.resources").joinpath("intent_to_agent.json").read_text()
    return json.loads(text)


def test_all_151_intents_map_to_a_known_agent():
    labels = load_label_space()
    assert labels.num_intents == 151
    assert set(labels.intent_to_agent) == set(labels.intent_names)
    assert set(labels.intent_to_agent.values()) <= set(AGENTS)


def test_every_agent_owns_at_least_one_intent():
    labels = load_label_space()
    counts = np.bincount(labels.intent_to_agent_id, minlength=len(AGENTS))
    assert (counts >= 1).all(), dict(zip(AGENTS, counts.tolist(), strict=True))


def test_oos_intent_maps_to_oos_agent_and_only_oos_does():
    labels = load_label_space()
    assert labels.intent_to_agent[OOS] == OOS
    assert [n for n, a in labels.intent_to_agent.items() if a == OOS] == [OOS]


def test_oos_has_hf_integer_id_42():
    # clinc_oos 'plus' ClassLabel order; the network test re-checks against the Hub.
    assert load_label_space().oos_intent_id == 42


def test_meta_keys_are_excluded_from_the_mapping():
    raw = raw_mapping()
    assert "_meta" in raw
    assert "_meta" not in load_label_space().intent_to_agent


def test_agent_counts_match_the_source_repository():
    counts = np.bincount(load_label_space().intent_to_agent_id, minlength=len(AGENTS))
    expected = {
        "finance_agent": 38,
        "travel_agent": 15,
        "auto_agent": 15,
        "kitchen_agent": 12,
        "productivity_agent": 32,
        "device_agent": 16,
        "meta_agent": 22,
        OOS: 1,
    }
    assert dict(zip(AGENTS, counts.tolist(), strict=True)) == expected


@pytest.mark.parametrize(
    ("names", "mapping", "message"),
    [
        (["a", OOS], {"a": "finance_agent"}, "missing"),
        (["a", OOS], {"a": "finance_agent", OOS: OOS, "b": "auto_agent"}, "extra"),
        (["a", OOS], {"a": "weather_agent", OOS: OOS}, "unknown agents"),
        (["a", OOS], {"a": "finance_agent", OOS: "meta_agent"}, "must map"),
        (["a", "a", OOS], {"a": "finance_agent", OOS: OOS}, "duplicates"),
    ],
)
def test_build_label_space_rejects_broken_mappings(names, mapping, message):
    with pytest.raises(ValueError, match=message):
        build_label_space(names, mapping)


def test_aggregate_probs_sums_intent_mass_per_agent():
    labels = build_label_space(
        ["pay_bill", "credit_score", "book_flight", OOS],
        {
            "pay_bill": "finance_agent",
            "credit_score": "finance_agent",
            "book_flight": "travel_agent",
            OOS: OOS,
        },
    )
    probs = np.array([[0.2, 0.3, 0.1, 0.4]])
    agent = labels.aggregate_probs(probs)
    assert agent.shape == (1, len(AGENTS))
    assert agent[0, AGENTS.index("finance_agent")] == pytest.approx(0.5)
    assert agent[0, AGENTS.index("travel_agent")] == pytest.approx(0.1)
    assert agent[0, AGENTS.index(OOS)] == pytest.approx(0.4)
    assert agent.sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(labels.agents_of(np.array([0, 2, 3])), [0, 1, 7])
