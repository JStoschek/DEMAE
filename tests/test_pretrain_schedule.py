"""The LR schedule: linear warmup from 0, cosine to exactly 0 (#93, plan §6).

A learning-rate curve is the easiest thing in a training loop to get subtly wrong and the
hardest to notice: every wrong version still trains. A warmup that starts at the peak instead
of 0 trains, a cosine that bottoms out at 4% of the peak instead of 0 trains, an off-by-one
that shifts the whole curve by an epoch trains. So the tests here are about **boundary
values** — the three points the plan actually names (0 at step 0, the peak at the end of
warmup, exactly 0 at the final epoch) — and about the shape between them, rather than about
the schedule producing numbers.

The peak is not typed anywhere: it is :attr:`RunConfig.lr <pretrain.config.RunConfig>`, which
is §6's ``blr x eff_batch / 256``. That rule is checked here too, at the value the plan states
(1.2e-3 at effective batch 2048), because a schedule scaled off the wrong peak is the same
class of invisible error.
"""

from __future__ import annotations

import itertools
import math

import pytest

from pretrain.config import Optimization, RunConfig, Schedule
from pretrain.schedule import LrSchedule

#: Steps per epoch small enough to enumerate, large enough that a fractional epoch is real.
STEPS = 10


@pytest.fixture
def schedule() -> LrSchedule:
    """R1's schedule: 400 epochs, 20 of warmup, cosine to 0, peak 1.2e-3."""
    return LrSchedule.from_config(RunConfig(), steps_per_epoch=STEPS)


# --- The rule that sets the peak ----------------------------------------------


def test_the_peak_is_the_linear_scaling_rules_lr_not_a_typed_number(schedule):
    # §6: lr = blr x eff_batch / 256 = 1.5e-4 x 2048 / 256.
    assert RunConfig().effective_batch == 2048
    assert schedule.peak_lr == pytest.approx(1.2e-3)
    assert schedule.peak_lr == RunConfig().lr


def test_the_rule_follows_the_effective_batch_not_the_physical_one():
    # This used to lean on ViT-B shipping a halved physical batch. R0 retired that hedge
    # (#81, ADR-0010 §1), so every preset now runs 1024 x 2 and the property would be
    # asserted by a config that cannot exercise it. Move the physical batch explicitly
    # instead: the peak LR must not notice, because §6's rule reads the *effective* batch.
    vit_b = RunConfig.vit_base()
    assert (vit_b.optimization.batch_size, vit_b.optimization.accum_iter) == (1024, 2)

    halved = vit_b.replace(
        optimization=vit_b.optimization.replace(batch_size=512, accum_iter=4)
    )
    assert halved.effective_batch == vit_b.effective_batch == 2048
    for cfg in (vit_b, halved):
        assert cfg.lr == pytest.approx(1.2e-3)
        assert LrSchedule.from_config(cfg, steps_per_epoch=STEPS).peak_lr == pytest.approx(1.2e-3)


# --- Warmup -------------------------------------------------------------------


def test_warmup_starts_at_exactly_zero(schedule):
    assert schedule.lr_at_epoch(0.0) == 0.0
    assert schedule.lr_at_step(0) == 0.0


def test_warmup_rises_linearly_to_the_peak_over_the_warmup_epochs(schedule):
    warmup = RunConfig().schedule.warmup_epochs
    assert warmup == 20
    for epoch in range(warmup + 1):
        assert schedule.lr_at_epoch(epoch) == pytest.approx(schedule.peak_lr * epoch / warmup)


def test_the_peak_is_reached_at_the_end_of_warmup_and_nowhere_before(schedule):
    warmup = RunConfig().schedule.warmup_epochs
    assert schedule.lr_at_epoch(warmup) == pytest.approx(schedule.peak_lr)
    assert schedule.lr_at_epoch(warmup - 1e-6) < schedule.peak_lr


def test_warmup_is_measured_in_steps_not_only_whole_epochs(schedule):
    # Half an epoch into a 20-epoch warmup is 1/40th of the peak; a schedule that only moved
    # on epoch boundaries would hold 0 for a whole epoch of steps.
    assert schedule.lr_at_step(STEPS // 2) == pytest.approx(schedule.peak_lr * 0.5 / 20)


def test_a_zero_warmup_schedule_starts_at_the_peak():
    # Legal in the config, and the division by `warmup_epochs` must not be reached.
    cfg = RunConfig().replace(schedule=Schedule(epochs=10, warmup_epochs=0, milestones=(10,)))
    assert LrSchedule.from_config(cfg, steps_per_epoch=STEPS).lr_at_epoch(0.0) == pytest.approx(
        cfg.lr
    )


# --- Cosine decay -------------------------------------------------------------


def test_the_cosine_reaches_exactly_zero_at_the_final_epoch(schedule):
    # Exactly, not approximately: §6 says "cosine to 0", and a schedule that bottoms out at a
    # few percent of the peak is a different recipe that no curve would distinguish.
    assert schedule.lr_at_epoch(400) == 0.0
    assert schedule.lr_at_step(schedule.total_steps) == 0.0
    # And *the cosine* is what reaches it, rather than an early return that happens to agree:
    # the value has to fall continuously into the endpoint from inside the schedule.
    assert schedule.lr_at_epoch(400 - 1e-9) == pytest.approx(0.0, abs=1e-12)
    assert schedule.lr_at_epoch(399) > 0.0


def test_the_cosine_is_half_the_peak_at_the_midpoint_of_the_decay(schedule):
    midpoint = 20 + (400 - 20) / 2
    assert schedule.lr_at_epoch(midpoint) == pytest.approx(schedule.peak_lr / 2)


def test_the_cosine_matches_maes_own_formula(schedule):
    # MAE's `adjust_learning_rate`, with min_lr = 0: the recipe this reproduces.
    for epoch in (20, 100, 250, 399):
        expected = (
            schedule.peak_lr * 0.5 * (1 + math.cos(math.pi * (epoch - 20) / (400 - 20)))
        )
        assert schedule.lr_at_epoch(epoch) == pytest.approx(expected)


def test_the_decay_never_rises(schedule):
    lrs = [schedule.lr_at_step(s) for s in range(20 * STEPS, schedule.total_steps + 1)]
    assert all(b <= a for a, b in itertools.pairwise(lrs))
    assert lrs[0] == pytest.approx(schedule.peak_lr) and lrs[-1] == 0.0


def test_running_past_the_schedule_holds_zero_rather_than_riding_the_cosine_back_up(schedule):
    # cos(pi * x) climbs again past x = 1. An overrun must not silently restart the LR: R5's
    # 400 -> 800 extension (§6) is a config change, not a loop that kept going.
    assert schedule.lr_at_epoch(401) == 0.0
    assert schedule.lr_at_epoch(1_000) == 0.0


# --- Steps, epochs, and the optimizer -----------------------------------------


def test_total_steps_is_the_schedule_measured_in_optimizer_steps(schedule):
    assert schedule.total_steps == 400 * STEPS


def test_a_step_maps_onto_its_fractional_epoch(schedule):
    assert schedule.epoch_at_step(0) == 0.0
    assert schedule.epoch_at_step(STEPS) == 1.0
    assert schedule.epoch_at_step(3 * STEPS + 5) == pytest.approx(3.5)


def test_applying_it_writes_the_lr_into_every_parameter_group(schedule):
    import torch

    p = torch.nn.Parameter(torch.zeros(2))
    q = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW(
        [{"params": [p], "weight_decay": 0.05}, {"params": [q], "weight_decay": 0.0}], lr=1.0
    )
    lr = schedule.apply(optimizer, step=10 * STEPS)  # ten epochs in, halfway up the warmup
    assert lr == pytest.approx(schedule.peak_lr * 10 / 20)
    assert [g["lr"] for g in optimizer.param_groups] == [lr, lr]


def test_a_schedule_needs_at_least_one_step_per_epoch():
    with pytest.raises(ValueError, match="steps_per_epoch"):
        LrSchedule.from_config(RunConfig(), steps_per_epoch=0)


def test_the_schedule_reads_the_config_rather_than_taking_loose_numbers():
    cfg = RunConfig().replace(
        schedule=Schedule(epochs=150, warmup_epochs=8, milestones=(50, 100)),
        optimization=Optimization(batch_size=256, accum_iter=1),
    )
    s = LrSchedule.from_config(cfg, steps_per_epoch=STEPS)
    assert (s.epochs, s.warmup_epochs) == (150, 8)
    assert s.peak_lr == pytest.approx(1.5e-4)  # eff batch 256 -> lr = blr exactly
    assert s.lr_at_epoch(8) == pytest.approx(1.5e-4)
    assert s.lr_at_epoch(150) == 0.0
