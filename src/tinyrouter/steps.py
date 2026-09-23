"""How many optimizer steps a run takes, decided before training starts.

The learning-curve protocol (docs/PLAN.md section 4, hyperparameter
protocol) trains for max(S_min, the steps of ``num_train_epochs`` epochs).
Small k-shot samples give only a handful of steps per epoch, so S_min keeps
the tiny points from being judged on an undertrained model; the large
points are unaffected because their epoch steps already exceed S_min.

The epoch count reproduces HF Trainer's own arithmetic (transformers
5.17, ``Trainer.set_initial_training_values``, one device, no gradient
accumulation): ceil(epochs * ceil(rows / batch)). ``train()`` checks the
step count it actually ran against this plan and raises on a mismatch.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

from tinyrouter.config import RunConfig

DecidedBy = Literal["max_steps", "epochs", "min_train_steps"]


@dataclass(frozen=True)
class StepPlan:
    epoch_steps: int | None
    min_train_steps: int | None
    planned_steps: int
    decided_by: DecidedBy
    # What TrainingArguments.max_steps receives: -1 means "train by epochs".
    max_steps_arg: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def epoch_steps(train_rows: int, batch_size: int, epochs: float) -> int:
    if train_rows < 1:
        raise ValueError("cannot plan steps for an empty training set")
    return math.ceil(epochs * math.ceil(train_rows / batch_size))


def plan_steps(config: RunConfig, train_rows: int) -> StepPlan:
    """Steps for ``config`` on ``train_rows`` rows; ties between the two rules go to epochs.

    Going to epochs on a tie keeps the code path of a run without
    ``min_train_steps`` (``max_steps=-1``), so such a run is the same run.
    """
    if config.max_steps > 0:
        return StepPlan(None, None, config.max_steps, "max_steps", config.max_steps)
    by_epochs = epoch_steps(train_rows, config.train_batch_size, config.num_train_epochs)
    floor = config.min_train_steps
    if floor is not None and floor > by_epochs:
        return StepPlan(by_epochs, floor, floor, "min_train_steps", floor)
    return StepPlan(by_epochs, floor, by_epochs, "epochs", -1)
