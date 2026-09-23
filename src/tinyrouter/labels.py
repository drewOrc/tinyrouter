"""Label spaces: 151 CLINC150 intents (150 + oos) and the 8 routing targets.

The model is trained on the 151 fine-grained intents; routing decisions are
made in the 8-way agent space (7 agents + oos). Both files under
``resources/`` are committed: ``intent_names.json`` is the Hugging Face
``clinc_oos`` ``plus`` ClassLabel order (index == integer label id), and
``intent_to_agent.json`` maps each intent name to an agent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from importlib import resources

import numpy as np

OOS = "oos"
AGENTS: tuple[str, ...] = (
    "finance_agent",
    "travel_agent",
    "auto_agent",
    "kitchen_agent",
    "productivity_agent",
    "device_agent",
    "meta_agent",
    OOS,
)


def _read_resource(name: str) -> object:
    text = resources.files("tinyrouter.resources").joinpath(name).read_text(encoding="utf-8")
    return json.loads(text)


@dataclass(frozen=True)
class LabelSpace:
    """The two label spaces and the fixed map between them."""

    intent_names: tuple[str, ...]
    intent_to_agent: dict[str, str]
    intent_to_agent_id: np.ndarray

    @property
    def num_intents(self) -> int:
        return len(self.intent_names)

    @property
    def oos_intent_id(self) -> int:
        return self.intent_names.index(OOS)

    @property
    def oos_agent_id(self) -> int:
        return AGENTS.index(OOS)

    def agents_of(self, intent_ids: np.ndarray) -> np.ndarray:
        """Map integer intent ids (any shape) to integer agent ids."""
        return self.intent_to_agent_id[np.asarray(intent_ids, dtype=np.int64)]

    @property
    def sha256(self) -> str:
        """Fingerprint of intent order plus the intent-to-agent map.

        Two archives with the same fingerprint agree on what every logit
        column means and which agent it rolls up to.
        """
        canonical = json.dumps(
            [[name, self.intent_to_agent[name]] for name in self.intent_names],
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def aggregate_probs(self, intent_probs: np.ndarray) -> np.ndarray:
        """Sum 151-way probabilities into 8-way agent probabilities.

        Args:
            intent_probs: shape (n, 151), rows summing to 1.

        Returns:
            shape (n, 8), rows summing to 1.
        """
        n = intent_probs.shape[0]
        out = np.zeros((n, len(AGENTS)), dtype=intent_probs.dtype)
        np.add.at(out.T, self.intent_to_agent_id, intent_probs.T)
        return out


def build_label_space(intent_names: list[str], intent_to_agent: dict[str, object]) -> LabelSpace:
    """Validate and assemble a LabelSpace; keys starting with ``_`` are metadata and skipped."""
    mapping = {k: v for k, v in intent_to_agent.items() if not k.startswith("_")}
    names = tuple(intent_names)
    if len(set(names)) != len(names):
        raise ValueError("intent_names has duplicates")
    missing = [n for n in names if n not in mapping]
    extra = [k for k in mapping if k not in set(names)]
    if missing or extra:
        raise ValueError(
            f"intent_to_agent does not cover intent_names exactly: missing={missing} extra={extra}"
        )
    unknown = sorted({str(v) for v in mapping.values()} - set(AGENTS))
    if unknown:
        raise ValueError(f"intent_to_agent uses unknown agents {unknown}; allowed: {AGENTS}")
    if mapping.get(OOS) != OOS:
        raise ValueError("the 'oos' intent must map to the 'oos' agent")
    agent_ids = np.array([AGENTS.index(str(mapping[n])) for n in names], dtype=np.int64)
    return LabelSpace(
        intent_names=names,
        intent_to_agent={n: str(mapping[n]) for n in names},
        intent_to_agent_id=agent_ids,
    )


@cache
def load_label_space() -> LabelSpace:
    """Load the committed label space shipped with the package."""
    names = _read_resource("intent_names.json")
    mapping = _read_resource("intent_to_agent.json")
    assert isinstance(names, list) and isinstance(mapping, dict)
    return build_label_space(names, mapping)
