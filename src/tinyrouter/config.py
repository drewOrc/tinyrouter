"""Run configuration loaded from ``configs/*.yaml``. Unknown keys are an error."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Literal

import yaml

Device = Literal["auto", "cpu", "mps", "cuda"]

# Where output goes, not what it is: two configs differing only here produce the same run.
LOCATION_FIELDS = frozenset({"checkpoint_root", "results_root"})


@dataclass(frozen=True)
class RunConfig:
    model_name: str
    model_revision: str
    seed: int = 42
    per_intent: int | None = None
    max_length: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_train_epochs: float = 5.0
    max_steps: int = -1
    train_batch_size: int = 32
    eval_batch_size: int = 128
    device: Device = "auto"
    # Allow the checkpoint's own classifier head to be replaced by a fresh
    # 151-way head. Off by default so a real backbone fails loudly on any
    # unexpected shape mismatch; the smoke model ships a head and needs it.
    replace_classifier_head: bool = False
    checkpoint_root: str = "checkpoints"
    results_root: str = "results"
    # Subsample validation/test per intent. Only the smoke config sets it;
    # every reported number uses the full splits.
    eval_per_intent: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")

    @property
    def run_name(self) -> str:
        size = "full" if self.per_intent is None else f"k{self.per_intent}"
        short = self.model_name.rstrip("/").split("/")[-1]
        return f"{short}-{size}-seed{self.seed}"

    def identity(self) -> dict[str, object]:
        """Every field that can change the trained model or its scores.

        Deliberately conservative: fields that only affect evaluation
        (``eval_batch_size``, ``eval_per_intent``) also count, so changing
        one of them retrains instead of just re-scoring. This could later be
        split into a training identity and an evaluation identity.
        """
        return {k: v for k, v in asdict(self).items() if k not in LOCATION_FIELDS}

    def matches(self, recorded: object) -> bool:
        """Whether a config dict saved with earlier output describes this same run."""
        if not isinstance(recorded, dict):
            return False
        return {k: v for k, v in recorded.items() if k not in LOCATION_FIELDS} == self.identity()

    def with_seed(self, seed: int) -> RunConfig:
        return replace(self, seed=seed)


def load_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    allowed = {f.name for f in fields(RunConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{path}: unknown config keys {unknown}; allowed: {sorted(allowed)}")
    return RunConfig(**raw)
