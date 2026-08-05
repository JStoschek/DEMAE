"""§10.1's failure conditions, as things that fire (#97).

Plan §10.1 names three: *sudden collapse toward zero*, *val loss rising while train falls*, and
*a plateau at high loss in the first 20 epochs*. §10.4's throughput dashboard adds a fourth, the
~10% dataloader wait fraction. All four were sentences in a document; this module is what makes
them alarms, and the thresholds they need are :class:`~pretrain.config.Monitoring` knobs so that
"why did it fire?" is answerable from the run's own recorded config.

:func:`diverged` is a fifth that the plan does not name, and it is here for a structural reason
rather than an appetite for more alarms: every threshold below compares *false* against a NaN,
so without it the harness goes silent at the exact moment a run dies. §13 lists bf16 loss spikes
as a real risk, so this is not a hypothetical shape.

## Pure functions over trajectories, deliberately

Nothing here touches a model, a logger or a config beyond the threshold group. An alarm takes
``[(epoch, loss), ...]`` and returns an :class:`Alarm` or ``None``, which is what lets every one
of them be fired on a synthetic curve built to trigger it and held quiet on one built not to —
the acceptance criterion this module exists to satisfy. :mod:`pretrain.monitor` is what
assembles the trajectories and what does something about the answer.

## Thresholds relative to the run's own start, not absolute

The loss *scale* is a recipe knob. With §4's `norm_pix` on, the target is unit-variance per
patch and an untrained model sits near 1.0; R3 (#84) ablates it off and the same model scores
something else entirely. So "collapsed toward zero" and "still at a high loss" are both
expressed as fractions of the run's **first recorded epoch**, which is scale-free and reads the
same on both arms. The cost is one blind spot, stated rather than hidden: a run that is already
broken at its first recorded epoch has a broken baseline, and no ratio against it will fire.
§4's per-patch loss *distribution* is what covers that case — a broken `norm_pix` target
collapses the distribution onto a spike at zero on the very first pass, which is visible in the
histogram :mod:`pretrain.monitor` logs whether or not anything here fires. An absolute floor
was considered instead and rejected: it would only be meaningful on the `norm_pix`-on arm, and
an alarm that silently does not apply to R3 is worse than one whose limit is written down.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .config import Monitoring

#: The run should stop and be fixed before more GPU hours are spent on it.
STOP: Final = "stop"
#: The run should continue, and someone should look at this before the next decision.
WATCH: Final = "watch"

#: ``[(epoch, value), ...]`` — what every check below reads.
Trajectory = Sequence[tuple[int, float]]


@dataclass(frozen=True)
class Alarm:
    """One fired condition: what, when, what to do, and the numbers that fired it."""

    #: Stable identifier — ``collapse``, ``overfitting``, ``plateau``, ``loader_wait``. This is
    #: what a W&B panel and a log filter key on, so it does not change when wording does.
    name: str
    #: The epoch the condition was evaluated at.
    epoch: int
    #: :data:`STOP` or :data:`WATCH`. Advice, not enforcement: this module never kills a run.
    #: What to do about a stop-grade alarm is R0/R1's call and #95's entrypoint's to wire, and
    #: an alarm that terminated a 40-hour run on its own reading of a threshold would be a
    #: worse failure than the one it caught.
    action: str
    #: One line, carrying the actual numbers. A message that only restates the name gives the
    #: reader nothing the alarm's existence had not already told them.
    message: str

    def __str__(self) -> str:
        return f"[{self.action.upper()}] {self.name} @ epoch {self.epoch}: {self.message}"


def _at(trajectory: Trajectory, epoch: int) -> float | None:
    """The value at ``epoch``, or the last one recorded before it."""
    seen = [v for e, v in trajectory if e <= epoch]
    return seen[-1] if seen else None


def diverged(trajectory: Trajectory, *, what: str) -> Alarm | None:
    """A non-finite loss — the floor underneath §10.1's three conditions.

    Not one of the plan's named conditions, and it is here for a structural reason rather than
    an ambition to add a fourth: **every comparison against a NaN is false**, so a run that has
    gone to NaN or inf passes :func:`collapse`, :func:`overfitting` and :func:`plateau` in
    silence. The monitoring harness would be quietest at the exact moment the run died, which
    is not a defensible thing for a monitoring harness to do.

    It is also a condition the plan expects to be possible: §13 lists *"bf16 instability / loss
    spikes"* as a risk, with grad-clip 1.0 as the mitigation to reach for if it appears.
    """
    if not trajectory:
        return None
    epoch, latest = trajectory[-1]
    if math.isfinite(latest):  # covers both NaN and +/-inf
        return None
    return Alarm(
        "diverged",
        epoch,
        STOP,
        f"{what} loss is {latest} at epoch {epoch} — not a number. Every threshold below "
        f"compares false against it, so nothing else here can fire: this alarm exists so the "
        f"harness is not silent exactly when the run has died. §13 lists bf16 loss spikes as a "
        f"risk with grad-clip 1.0 as the mitigation; check that before restarting",
    )


def collapse(trajectory: Trajectory, *, monitoring: Monitoring, what: str) -> Alarm | None:
    """§10.1's *sudden collapse toward zero* — the broken-`norm_pix`-target symptom.

    Two ways in, because the symptom has two shapes and neither implies the other:

    - **The level.** The loss is at or under :attr:`Monitoring.collapse_frac
      <pretrain.config.Monitoring>` of where the run started. A model that reconstructs terrain
      50x better than an untrained one is not what a masked autoencoder does; a model whose
      target has gone identically flat scores exactly this, because predicting zero is right.
    - **The fall.** A drop by :attr:`Monitoring.collapse_drop_factor
      <pretrain.config.Monitoring>` between two consecutive recorded epochs. §10.1 expects a
      fast warmup drop, so the level alone would miss a target that broke at epoch 100 and
      would risk firing on a healthy epoch 1; the factor is what makes "sudden" mean something.

    The reference for the mechanism is [transformers issue
    29416](https://github.com/huggingface/transformers/issues/29416), which §10.1 cites: a
    `norm_pix` target computed in a narrow dtype measures as having zero variance, the
    normalized target becomes all zeros, and the loss goes to zero while nothing is learned.
    """
    if len(trajectory) < 2:
        return None
    baseline = trajectory[0][1]
    epoch, latest = trajectory[-1]
    _, previous = trajectory[-2]

    if baseline > 0 and latest <= monitoring.collapse_frac * baseline:
        return Alarm(
            "collapse",
            epoch,
            STOP,
            f"{what} loss {latest:.4g} is {latest / baseline:.1%} of the run's first epoch "
            f"({baseline:.4g}), under the {monitoring.collapse_frac:.0%} floor. A masked "
            f"autoencoder does not reconstruct terrain this well; a broken norm_pix target "
            f"scores exactly this, because a target that measures as flat is predicted by zero "
            f"(§10.1, transformers issue 29416). Check §4's fp32 target statistics and the "
            f"variance floor, and read the per-patch distribution for a spike at zero",
        )
    if latest * monitoring.collapse_drop_factor <= previous:
        return Alarm(
            "collapse",
            epoch,
            STOP,
            f"{what} loss fell {previous:.4g} -> {latest:.4g} in one epoch, a "
            f"{previous / max(latest, 1e-12):.0f}x drop against the "
            f"{monitoring.collapse_drop_factor:.0f}x threshold. §10.1 expects a fast warmup "
            f"drop and then slow decay for hundreds of epochs; a fall this sharp mid-run is the "
            f"target moving, not the model learning. Check §4's norm_pix guards",
        )
    return None


def overfitting(
    train: Trajectory, val: Trajectory, *, monitoring: Monitoring
) -> Alarm | None:
    """§10.1's *val loss rising while train falls*.

    Plausible here in a way it is not for most SSL: the corpus card (§4.6) warns that 500k
    tiles drawn from 425 geocells are not 500k independent samples, so the effective dataset is
    smaller than the count and §9's 2% val split exists to watch for exactly this.

    Three conditions, and dropping any one of them produces a false positive that would teach
    people to ignore the alarm:

    1. Val is above its own best by more than :attr:`Monitoring.overfit_min_rise
       <pretrain.config.Monitoring>`. Without this, a *converged* run — val flat at its minimum,
       train still creeping down — alarms as overfitting.
    2. It has stayed there for :attr:`Monitoring.overfit_patience
       <pretrain.config.Monitoring>` recorded val points. Without this, noise in a ~10k-tile
       val pass fires it.
    3. **Train has fallen over the same span.** Without this, a run where both losses are rising
       — §13's bf16 instability — is diagnosed as overfitting, which points at the data when the
       problem is the optimizer.
    """
    if len(val) < 2:
        return None
    best_index = min(range(len(val)), key=lambda i: val[i][1])
    best_epoch, best = val[best_index]
    epoch, latest = val[-1]
    # Epochs past the minimum, not val *points* past it: the threshold is documented in epochs
    # and has to keep meaning that whatever cadence a run scores val on.
    since = epoch - best_epoch
    if since < monitoring.overfit_patience:
        return None
    if best <= 0 or latest < best * (1 + monitoring.overfit_min_rise):
        return None

    train_then, train_now = _at(train, best_epoch), _at(train, epoch)
    if train_then is None or train_now is None or train_now >= train_then:
        return None
    return Alarm(
        "overfitting",
        epoch,
        WATCH,
        f"val loss has been above its best for {since} epochs — {best:.4g} at epoch "
        f"{best_epoch}, now {latest:.4g} ({latest / best - 1:+.1%}) — while train fell "
        f"{train_then:.4g} -> {train_now:.4g} over the same span. §10.1's overfitting "
        f"condition; the corpus card (§4.6) makes it plausible at 500k non-independent tiles. "
        f"§13's mitigation is to stop the 400->800 epoch extension rather than the run",
    )


def plateau(trajectory: Trajectory, *, monitoring: Monitoring) -> Alarm | None:
    """§10.1's *plateau at high loss in the first 20 epochs* — an LR or normalization bug.

    Adjudicated at :attr:`Monitoring.plateau_by_epoch <pretrain.config.Monitoring>`, which is
    §6's warmup length: the LR first reaches its peak there, so a model that is going to learn
    has visibly started. Two conditions together, because either alone is ordinary — a run can
    be high and improving fast (early), or low and barely improving (late in the cosine), and
    neither is a bug. High *and* stuck is.

    It keeps firing past epoch 20 while both conditions hold, rather than being a one-shot check
    at exactly 20. A run that is still at its starting loss at epoch 50 has the same bug, and
    silencing the alarm on the grounds that its window closed would be a strange thing to
    explain to whoever is paying for the GPU.
    """
    if not trajectory:
        return None
    epoch, latest = trajectory[-1]
    if epoch < monitoring.plateau_by_epoch:
        return None
    baseline = trajectory[0][1]
    if baseline <= 0 or latest < monitoring.plateau_frac * baseline:
        return None

    window = trajectory[max(0, len(trajectory) - monitoring.plateau_window - 1)]
    then = window[1]
    improvement = (then - latest) / then if then > 0 else 0.0
    if improvement >= monitoring.plateau_min_improvement:
        return None
    return Alarm(
        "plateau",
        epoch,
        STOP,
        f"train loss is {latest:.4g} at epoch {epoch}, still {latest / baseline:.0%} of the "
        f"run's first epoch ({baseline:.4g}), and improved only {improvement:.2%} since epoch "
        f"{window[0]}. §6's warmup reaches its peak learning rate at epoch "
        f"{monitoring.plateau_by_epoch}, so a run that is going to learn has moved by now. "
        f"§10.1 reads this as a learning-rate or normalization bug",
    )


def loader_wait(fraction: float, epoch: int, *, monitoring: Monitoring) -> Alarm | None:
    """§10.4's throughput alarm: the fraction of wall-clock spent waiting for the loader.

    The plan's #1 named risk (§7.5). DailyMAE got 5.8x end-to-end out of IO alone, and past
    ~10% wait the GPU hours are buying idle time, so this is a stop-and-fix rather than a note.
    """
    if fraction <= monitoring.loader_wait_alarm:
        return None
    return Alarm(
        "loader_wait",
        epoch,
        STOP,
        f"the step spent {fraction:.1%} of its wall-clock waiting for the dataloader, over "
        f"the {monitoring.loader_wait_alarm:.0%} threshold. §7.5 calls the loader this plan's "
        f"#1 risk and §10.4 says to stop and fix IO before burning more GPU hours: check "
        f"num_workers, whether the corpus has page-cached (>=64 GB RAM host), and prefetch",
    )


@dataclass(frozen=True)
class FrameCounts:
    """Dynamo frames a run has compiled, split by which path paid for them (#109).

    Mirrors :class:`~pretrain.runtime.CompileWatch`'s two tallies without importing it, so this
    module keeps reading nothing but plain numbers.
    """

    #: Frames the **training step** compiled after warmup. §7.3's claim is that this is 0.
    train: int = 0
    #: Frames §10's boundary work compiled. Expected to be nonzero exactly once, at boundary 0.
    monitoring: int = 0


def recompiled(frames: FrameCounts, before: FrameCounts, epoch: int) -> Alarm | None:
    """§7's other silent failure: a graph recompiling after the warmup step paid for it.

    Fires on **growth**, never on a standing count, so a run that compiled once and settled
    says so once instead of 400 times — while a genuine per-step loop, which grows every epoch,
    fires every epoch exactly as :func:`loader_wait` does.

    The two paths do not carry the same remedy, which is the whole of #109. A **first**
    train-path frame is a ~60 s one-time cost and :data:`WATCH`: recommending
    ``runtime.compile=false`` for it would trade one minute for a ~2.4x tax on every remaining
    epoch (R0's measurement). A **second** one means the count is still climbing — the recompile
    loop the lever actually loses to — and only then is turning compile off the cheaper trade.

    The monitoring path is silent on its first appearance, because that appearance is the val
    pass and the grids compiling their own graphs the first time they run, which
    :meth:`ReconGrids.due <pretrain.recon.ReconGrids.due>` pins to boundary 0 by design. Growth
    *after* that is a real per-epoch cost — val runs at every one of 400 boundaries — so it is
    a :data:`WATCH` rather than nothing.
    """
    if frames.train > before.train:
        first = before.train == 0
        return Alarm(
            "recompile",
            epoch,
            WATCH if first else STOP,
            (
                f"the training step compiled {frames.train} frame(s) after warmup, first seen "
                f"at epoch {epoch}. §7.3 expects none — the mask count is fixed (#90) and short "
                f"batches are dropped (#93) — so a shape or a guard is moving. One frame is a "
                f"one-time cost; compile is still worth ~2.4x, so this is worth understanding "
                f"and not worth disabling the lever for"
            )
            if first
            else (
                f"the training step's compile count grew again, {before.train} -> "
                f"{frames.train} by epoch {epoch}. It is still climbing, which is the recompile "
                f"loop §7.3's win is lost to: compiling is now costing this run rather than "
                f"paying for it. `--set runtime.compile=false` is the cheaper fallback until it "
                f"is understood"
            ),
        )
    if before.monitoring > 0 and frames.monitoring > before.monitoring:
        return Alarm(
            "recompile_monitoring",
            epoch,
            WATCH,
            f"§10's boundary work compiled {frames.monitoring - before.monitoring} more "
            f"frame(s) by epoch {epoch}, {before.monitoring} -> {frames.monitoring}. The val "
            f"pass and the grids compile once, at the first boundary; anything later repeats at "
            f"every boundary for the rest of the run. This does not affect §7.3's training-step "
            f"claim and does not warrant turning compile off",
        )
    return None


def check_alarms(
    monitoring: Monitoring,
    *,
    train: Trajectory,
    val: Trajectory,
    loader_wait_fraction: float | None = None,
    frames: FrameCounts | None = None,
    frames_before: FrameCounts | None = None,
) -> list[Alarm]:
    """Every §10 condition against one run's history so far, in a stable order.

    Divergence and collapse are each checked on **both** series and reported once: train and val
    are scored against the same target, so a broken target shows in whichever pass runs, and
    firing twice for one cause would just be noise. Train is checked first because it is the
    objective the optimizer steps on and therefore the first place a symptom appears.

    :func:`diverged` **replaces** the three loss conditions rather than joining them: against a
    NaN every threshold below compares false, so reporting one real alarm beside three silent
    passes would read as three conditions checked and cleared. The loader-wait check is not a
    loss condition and still runs — a diverged run on a starved loader has two problems. The
    same goes for :func:`recompiled`: a diverged run that is also recompiling has two, and the
    §7 checks are about what the kernels are doing rather than about the loss.
    """
    divergence = diverged(train, what="train") or diverged(val, what="val")
    fired: list[Alarm | None] = (
        [divergence]
        if divergence is not None
        else [
            collapse(train, monitoring=monitoring, what="train")
            or collapse(val, monitoring=monitoring, what="val"),
            overfitting(train, val, monitoring=monitoring),
            plateau(train, monitoring=monitoring),
        ]
    )
    epoch = train[-1][0] if train else 0
    if loader_wait_fraction is not None:
        fired.append(loader_wait(loader_wait_fraction, epoch, monitoring=monitoring))
    if frames is not None:
        fired.append(recompiled(frames, frames_before or FrameCounts(), epoch))
    return [alarm for alarm in fired if alarm is not None]
