"""Learning-curve hyperparameter protocol (docs/PLAN.md section 4, hyperparameter protocol).

``configs/curve.yaml`` holds the two values the pilots choose: one
``min_train_steps`` (S_min) shared by every k and both encoders, and one
learning rate per encoder shared by its whole curve. Both are written by
hand from ``results/pilots/*.json``; a null means that pilot has not been
run, and the curve refuses to start. Values outside the protocol's grids
are refused too, so a typo cannot quietly become a new setting.

Every curve run is the model's base config (``configs/<model>.yaml``,
which also fixes max_length, batch size, epochs, warmup and weight decay)
with the chosen learning rate and S_min, plus a k and a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from tinyrouter.config import RunConfig, load_config
from tinyrouter.sampling import CURVE_KS, FULL_K

MODELS = ("bert", "modernbert")
SEEDS = (42, 43, 44)
LR_GRID = (1e-5, 2e-5, 5e-5)
S_MIN_GRID = (100, 200, 400)
PILOT_SEED = 42
PILOT_LR_K = 100
PILOT_STEPS_K = 5
ABLATION_MODEL = "modernbert"
ABLATION_K = 100
ABLATION_OOS_ROWS = 0
DEFAULT_PROTOCOL = Path("configs/curve.yaml")


class ProtocolError(ValueError):
    """configs/curve.yaml is malformed, off the protocol's grids, or not filled in yet."""


@dataclass(frozen=True)
class CurveProtocol:
    min_train_steps: int | None
    learning_rates: dict[str, float | None]
    base_configs: dict[str, RunConfig]
    base_paths: dict[str, str]

    def base(self, model: str) -> RunConfig:
        """The model's config file as is (for BERT this is the AC2 config)."""
        if model not in MODELS:
            raise ProtocolError(f"unknown model '{model}'; expected one of {MODELS}")
        return self.base_configs[model]

    def learning_rate(self, model: str) -> float:
        lr = self.learning_rates[model]
        if lr is None:
            raise ProtocolError(
                f"configs/curve.yaml: learning_rate for {model} is null; run `make pilot-lr` "
                "and copy the selected value in"
            )
        return lr

    def s_min(self) -> int:
        if self.min_train_steps is None:
            raise ProtocolError(
                "configs/curve.yaml: min_train_steps is null; run `make pilot-steps` and copy "
                "the selected value in"
            )
        return self.min_train_steps

    def curve_config(
        self, model: str, k: int, seed: int, oos_train: int | None = None
    ) -> RunConfig:
        """One curve (or ablation) run; needs both pilots decided."""
        if k not in CURVE_KS:
            raise ProtocolError(f"k={k} is not a curve point {CURVE_KS}")
        return replace(
            self.base(model),
            learning_rate=self.learning_rate(model),
            min_train_steps=self.s_min(),
            k_shot=k,
            oos_train=oos_train,
            seed=seed,
        )

    def curve_configs(self, model: str) -> list[RunConfig]:
        return [self.curve_config(model, k, seed) for k in CURVE_KS for seed in SEEDS]

    def ablation_configs(self) -> list[RunConfig]:
        return [
            self.curve_config(ABLATION_MODEL, ABLATION_K, seed, oos_train=ABLATION_OOS_ROWS)
            for seed in SEEDS
        ]


def ac2_donor(protocol: CurveProtocol, config: RunConfig) -> RunConfig | None:
    """The AC2 run that could stand in for ``config``: BERT's own config file at this seed.

    Only a candidate. Whether it really is the same run is decided by
    ``runs.reusable_equivalent``, which compares every field, the rows and
    the step count, and checks the archive's SHA-256 in three places.
    """
    bert = protocol.base("bert")
    if config.model_name != bert.model_name or config.k_shot != FULL_K:
        return None
    if config.oos_train is not None:
        return None
    return bert.with_seed(config.seed)


def _on_grid(value: object, grid: tuple, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float) or value not in grid:
        raise ProtocolError(f"{name} = {value!r} is not one of the protocol's values {grid}")


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> CurveProtocol:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"min_train_steps", "models"}:
        raise ProtocolError(f"{path}: top level must have exactly min_train_steps and models")
    models = raw["models"]
    if not isinstance(models, dict) or set(models) != set(MODELS):
        raise ProtocolError(f"{path}: models must be exactly {MODELS}")
    _on_grid(raw["min_train_steps"], S_MIN_GRID, "min_train_steps")
    if isinstance(raw["min_train_steps"], float):
        raise ProtocolError(f"min_train_steps must be an integer, got {raw['min_train_steps']!r}")
    lrs: dict[str, float | None] = {}
    bases: dict[str, RunConfig] = {}
    paths: dict[str, str] = {}
    for model in MODELS:
        entry = models[model]
        if not isinstance(entry, dict) or set(entry) != {"config", "learning_rate"}:
            raise ProtocolError(f"{path}: models.{model} needs exactly config and learning_rate")
        _on_grid(entry["learning_rate"], LR_GRID, f"models.{model}.learning_rate")
        lr = entry["learning_rate"]
        lrs[model] = None if lr is None else float(lr)
        paths[model] = str(entry["config"])
        bases[model] = load_config(entry["config"])
    return CurveProtocol(raw["min_train_steps"], lrs, bases, paths)
