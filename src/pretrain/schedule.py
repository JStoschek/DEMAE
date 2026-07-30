"""§6's learning-rate curve: linear warmup from 0, then cosine to exactly 0 (#93).

Three numbers off the [plan](../../docs/phase1-pretraining-plan.md) §6 table make this whole
module — a peak, a warmup length, and a schedule length — and it exists as its own file for one
reason: **every wrong version of it trains**. A warmup that starts at the peak trains. A cosine
that bottoms out at 4% of the peak trains. An off-by-one that shifts the curve by an epoch
trains. None of them shows up as anything but a slightly different loss curve, which is exactly
what a different recipe also looks like. So the curve is a pure function of (step, config) with
its boundary values pinned by name, rather than a few lines inside the training loop.

## The peak is derived, never typed

:attr:`RunConfig.lr <pretrain.config.RunConfig>` is §6's linear scaling rule — ``blr x
eff_batch / lr_batch_base``, 1.2e-3 at the default 2048 — and this module reads it. That is
what keeps ViT-S (1024 x 2) and ViT-B (512 x 4) on the same learning rate: they differ in the
physical batch that fits in 32 GB and agree on the effective one, so the rule lands on the same
number and the two runs stay comparable.

## Fractional epochs, because a 20-epoch warmup is 20 epochs of *steps*

MAE adjusts the LR once per accumulation cycle at ``epoch + step_in_epoch / steps_per_epoch``,
not once per epoch, and so does this. At the shipped shape an epoch is ~244 optimizer steps, so
a per-epoch schedule would hold the LR flat across each of them and step it 400 times over the
run — visibly different early, where the warmup is doing the work it exists to do.

## Where it ends, and what happens after

``lr_at_epoch(epochs)`` is **exactly** 0.0, and it gets there *through the cosine* rather than
through a special case: at the final epoch ``progress`` is 1.0, ``cos(pi)`` is exactly -1 in
IEEE arithmetic, and ``0.5 * (1 + -1)`` is exactly 0. §6's "cosine to 0" is therefore literal
rather than approximate, and it is the formula saying so. Only *past* the end is the curve
clamped, because ``cos`` climbs back up after pi: a loop that overran its schedule would
otherwise silently ramp the LR back toward the peak. R5's 400 -> 800 extension (§6, #87) is a
*config change* that rebuilds this object, not a loop that kept going.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import RunConfig


@dataclass(frozen=True)
class LrSchedule:
    """The §6 curve, in optimizer steps. Build it with :meth:`from_config`.

    Frozen and free of torch: it is arithmetic over four numbers, so it can be evaluated in a
    test, plotted, or asserted at a boundary without a model or a device. :meth:`apply` is the
    one method that touches an optimizer, and it does nothing a caller could not do itself.
    """

    #: §6's ``blr x eff_batch / 256`` — :attr:`RunConfig.lr <pretrain.config.RunConfig>`.
    peak_lr: float
    #: The schedule length, in epochs (§6: 400).
    epochs: int
    #: Linear-warmup length, in epochs (§6: 20, which is 5% of 400, matching MAE's 40/800).
    warmup_epochs: int
    #: Optimizer steps per epoch — ``len(loader) // accum_iter`` at the run's shape. The
    #: schedule is written in these rather than in micro-batches so that changing
    #: ``accum_iter`` at a fixed effective batch (ViT-S vs ViT-B) leaves the curve alone.
    steps_per_epoch: int

    def __post_init__(self) -> None:
        if self.steps_per_epoch < 1:
            raise ValueError(
                f"steps_per_epoch {self.steps_per_epoch} must be >= 1: a schedule cannot be "
                "measured in steps if an epoch holds none (batch x accum_iter larger than the "
                "split, most likely)"
            )
        if self.peak_lr <= 0:
            raise ValueError(f"peak_lr {self.peak_lr} must be positive")

    @classmethod
    def from_config(cls, cfg: RunConfig, *, steps_per_epoch: int) -> LrSchedule:
        """The only way a run builds a schedule: §6's three numbers, off the config object."""
        return cls(
            peak_lr=cfg.lr,
            epochs=cfg.schedule.epochs,
            warmup_epochs=cfg.schedule.warmup_epochs,
            steps_per_epoch=steps_per_epoch,
        )

    @property
    def total_steps(self) -> int:
        """The whole schedule in optimizer steps — what a progress bar and #94 count against."""
        return self.epochs * self.steps_per_epoch

    def epoch_at_step(self, step: int) -> float:
        """The fractional epoch a step sits in. Step 0 is epoch 0.0, and warmup starts there."""
        return step / self.steps_per_epoch

    def lr_at_epoch(self, epoch: float) -> float:
        """The LR at a (possibly fractional) epoch — §6's curve, and MAE's own formula.

        Warmup is ``peak * epoch / warmup_epochs``, so it leaves 0 at epoch 0 and arrives at the
        peak exactly at the warmup boundary. After it, cosine over the remaining epochs down to
        0 — reached *by the cosine* at the final epoch, and only clamped **beyond** it (see the
        module docstring).
        """
        if epoch > self.epochs:
            return 0.0
        if epoch < self.warmup_epochs:
            return self.peak_lr * epoch / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
        return self.peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def lr_at_step(self, step: int) -> float:
        """The LR for the optimizer step about to be taken."""
        return self.lr_at_epoch(self.epoch_at_step(step))

    def apply(self, optimizer, step: int) -> float:
        """Set every parameter group's LR for ``step``, and return the value written.

        Every group gets the same rate — the decay and no-decay groups (:mod:`pretrain.optim`)
        differ in weight decay only. MAE's per-group ``lr_scale`` belongs to layer-decay
        fine-tuning, which is Phase 2's problem and not a knob §6 turns.
        """
        lr = self.lr_at_step(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
