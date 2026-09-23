"""Run configuration loaded from ``configs/*.yaml``. Unknown keys are an error."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Literal

import yaml

Device = Literal["auto", "cpu", "mps", "cuda"]

# Where output goes, not what it is: two configs differing only here produce the same run.
LOCATION_FIELDS = frozenset({"checkpoint_root", "results_root"})

# Fields added after results were first recorded (AC2, format of 2026-09-23),
# with the default that reproduces the behaviour of the code before they
# existed. A recorded config that lacks one of these keys is read as having
# this default; any other missing key still means "not the same run".
ADDED_FIELD_DEFAULTS: dict[str, object] = {
    "k_shot": None,
    "oos_train": None,
    "min_train_steps": None,
}


@dataclass(frozen=True)
class RunConfig:
    model_name: str
    model_revision: str
    seed: int = 42
    per_intent: int | None = None
    # Learning-curve sample (sampling.py): k rows per intent, oos from the
    # hardcoded table unless ``oos_train`` overrides it (OOS ablation).
    k_shot: int | None = None
    oos_train: int | None = None
    max_length: int = 64
    # None means "not chosen yet" (a pilot decides it); training refuses it.
    learning_rate: float | None = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_train_epochs: float = 5.0
    max_steps: int = -1
    # Train for max(min_train_steps, the steps num_train_epochs gives).
    # None trains by epochs only. See train.plan_steps.
    min_train_steps: int | None = None
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
        if self.k_shot is not None and self.per_intent is not None:
            raise ValueError("set k_shot (curve sample) or per_intent (per-intent cap), not both")
        if self.oos_train is not None and self.k_shot is None:
            raise ValueError("oos_train overrides the k-shot oos count; it needs k_shot")
        if self.min_train_steps is not None:
            if self.min_train_steps < 1:
                raise ValueError(f"min_train_steps must be >= 1, got {self.min_train_steps}")
            if self.max_steps > 0:
                raise ValueError("min_train_steps and max_steps > 0 contradict each other")

    @property
    def run_name(self) -> str:
        if self.k_shot is not None:
            size = f"k{self.k_shot}"
            if self.oos_train is not None:
                size += f"-oos{self.oos_train}"
        elif self.per_intent is not None:
            size = f"cap{self.per_intent}"
        else:
            size = "full"
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
        """Whether a config dict saved with earlier output describes this same run.

        Exact field-by-field equality, except that keys in
        ``ADDED_FIELD_DEFAULTS`` missing from an older record count as
        their default.
        """
        if not isinstance(recorded, dict):
            return False
        filled = {**ADDED_FIELD_DEFAULTS, **recorded}
        return {k: v for k, v in filled.items() if k not in LOCATION_FIELDS} == self.identity()

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
