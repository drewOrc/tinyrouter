from dataclasses import replace

import pytest

from tinyrouter.config import RunConfig
from tinyrouter.steps import epoch_steps, plan_steps

BASE = RunConfig(model_name="m", model_revision="r")


def test_epoch_steps_follow_the_trainer_arithmetic():
    assert epoch_steps(15_250, 32, 5) == 2385  # the AC2 run's recorded global_step
    assert epoch_steps(763, 32, 5) == 120  # k=5: 24 batches per epoch
    assert epoch_steps(153, 32, 5) == 25  # k=1
    assert epoch_steps(33, 32, 1.5) == 3  # ceil(1.5 * 2)


def test_empty_training_set_cannot_be_planned():
    with pytest.raises(ValueError, match="empty"):
        epoch_steps(0, 32, 5)


def test_without_min_train_steps_the_epochs_decide():
    plan = plan_steps(BASE, 763)
    assert (plan.planned_steps, plan.decided_by, plan.max_steps_arg) == (120, "epochs", -1)


@pytest.mark.parametrize(("s_min", "rows", "steps"), [(400, 153, 400), (200, 763, 200)])
def test_min_train_steps_above_the_epoch_count_sets_the_steps(s_min, rows, steps):
    plan = plan_steps(replace(BASE, min_train_steps=s_min), rows)
    assert plan.planned_steps == plan.max_steps_arg == steps
    assert plan.decided_by == "min_train_steps"
    assert plan.epoch_steps == epoch_steps(rows, 32, 5)


@pytest.mark.parametrize("s_min", [100, 120])
def test_min_train_steps_at_or_below_the_epoch_count_changes_nothing(s_min):
    plan = plan_steps(replace(BASE, min_train_steps=s_min), 763)
    assert (plan.planned_steps, plan.decided_by, plan.max_steps_arg) == (120, "epochs", -1)
    assert plan.min_train_steps == s_min


def test_full_data_is_never_extended_by_any_s_min_on_the_grid():
    for s_min in (100, 200, 400):
        assert plan_steps(replace(BASE, min_train_steps=s_min), 15_250).decided_by == "epochs"


def test_an_explicit_max_steps_wins():
    plan = plan_steps(replace(BASE, max_steps=7), 15_250)
    assert (plan.planned_steps, plan.decided_by, plan.max_steps_arg) == (7, "max_steps", 7)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"min_train_steps": 0}, ">= 1"),
        ({"min_train_steps": 10, "max_steps": 5}, "contradict"),
        ({"k_shot": 5, "per_intent": 5}, "not both"),
        ({"oos_train": 0}, "needs k_shot"),
    ],
)
def test_contradictory_configs_are_refused(fields, message):
    with pytest.raises(ValueError, match=message):
        replace(BASE, **fields)
