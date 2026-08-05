"""What a run reports every epoch, and the alarms wired onto it (#97, plan §10, #106).

Plan §10 opens with the reason this module exists: *"MAE gives few in-flight signals; build the
ones that exist."* There is no accuracy to watch, no held-out metric that means anything on its
own, and a reconstruction loss that goes down proves very little. What there is, this builds:

1. **Train and val reconstruction loss, every epoch** (§10.1). The val pass runs under a
   **committed fixed masking seed** (:data:`~pretrain.config.VAL_MASK_SEED`, §15's checklist
   item 3), which is the entire reason the number is comparable epoch to epoch.
2. **§10.1's three alarm conditions, as things that fire.** They live in :mod:`pretrain.alarms`
   as pure functions; this module assembles the trajectories and does something about the answer.
3. **§4's per-patch loss distribution** — :class:`LossHistogram`, cross-tabbed against patch
   flatness, because §4's rule is *"if the top tail is dominated by near-flat patches, raise the
   variance floor before concluding anything about the model"* and a mean cannot answer that.
4. **§10.4's throughput dashboard**, displayed and alarmed on. The counters themselves are
   :mod:`pretrain.throughput`'s (#95); this reads them.
5. **The sink** every one of those goes to, which is :mod:`pretrain.sink` (§11).

## This module is not a run path

There is no ``python -m pretrain.monitor``. There was, and #106 removed it: it trained, wrote
no checkpoints, and — because its ``main`` called ``build_config`` without ``overrides`` — it
accepted ``--set`` and discarded it, so ``monitor --set masking.ratio=0.9`` masked 75% and
reported R1's config hash. Two commands that both train are two recipes to keep in step, and
the one nobody watched was the one that drifted.

What is here now is :func:`train_epochs`, which *is* the shipping run — #94's resumable loop
with this module's panel on its epoch boundary — and ``python -m pretrain.launch`` is the one
command that calls it. Everything §10 asks for is still exercisable against a real corpus
without renting anything; it is ``pretrain.launch --epochs 2`` rather than a second entrypoint.

## The seam this is on both sides of

#98's reconstruction grids and #99's kNN probe compute their own payloads and write them
through :mod:`pretrain.sink`; what they share with this module is the logger and its step
axis — the epoch — so a run's grids, probes and losses land on one page under one set of tags.
This module is the first caller of that seam, the one that opens and closes the run, and the
only place that knows what epoch it is — which is why §10.2's grids are *driven* from
:meth:`EpochMonitor.end_of_epoch`, on their own 10-epoch cadence, without this module ever
learning what a hillshade is.

## What the val pass must not do

It must not perturb the run it is measuring. The mask draw takes an **explicit**
:class:`torch.Generator` rather than seeding the ambient RNG, so a val pass consumes nothing
from the stream the next training step draws its masks from. Left to `torch.manual_seed`, the
trajectory of a run would depend on whether anyone was watching it — and no curve would show
that. `test_the_val_pass_draws_nothing_from_the_training_stream` is the pin.

## Per-patch distribution on the val pass, not the train loop

§4's distribution is accumulated during the val pass only, and that is a deliberate cost
decision rather than an oversight. Reading per-patch tensors off the device in the training loop
is a sync per step — §7.3's `torch.compile` graph break, on the hot path, for a diagnostic that
is read once an epoch. The val pass already runs under fixed masks over a fixed tile order, so
its distribution is the *comparable* one anyway: two epochs' histograms differ because the model
did, not because the sample did.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
import torch

from .alarms import Alarm, FrameCounts, check_alarms
from .checkpoint import CheckpointPolicy
from .config import RunConfig
from .loss import patch_statistics
from .probe import KnnProbe, ProbeReport, probe_for
from .recon import ReconGrids
from .runtime import CompileWatch
from .sink import Histogram, RunLogger, open_run_logger
from .throughput import ThroughputSummary, format_throughput
from .train import StepRecord, Trainer, build_loader, train_resumable

#: The bins §4's distribution is accumulated into: one bin for *exactly nothing*, then 64
#: logarithmic bins from 1e-6 to 1e3.
#:
#: Logarithmic because the quantity is a squared error whose interesting structure is at both
#: ends — a near-flat patch under §4's variance floor scores orders of magnitude *below* the
#: bulk, and the top tail the rule adjudicates on sits orders of magnitude above it. Linear bins
#: over that range would put the entire distribution in one of them.
#:
#: The first bin, ``[0, 1e-6)``, is not a rounding artefact: it is where a **broken `norm_pix`
#: target** puts every patch (the target measures as identically flat, so predicting zero is
#: exactly right), and :attr:`LossHistogram.collapsed_fraction` reads it. That is the one
#: symptom §10.1's collapse alarm structurally cannot catch, because a run already broken at its
#: first epoch has no healthy baseline to be a fraction of.
#:
#: Values above the top edge are counted in the last bin rather than dropped. A histogram that
#: silently lost its tail would be worst exactly where §4's rule reads it.
LOSS_BIN_EDGES: Final = np.concatenate([[0.0], np.logspace(-6, 3, 65)])


# --- §4: the per-patch loss distribution --------------------------------------


class LossHistogram:
    """Per-patch losses binned, and cross-tabbed against whether the patch was near-flat.

    The cross-tab is the point. §4's rule is not "log the distribution" but *"if the top tail is
    dominated by near-flat patches, raise the variance floor before concluding anything about the
    model"*, and that is a question about the **joint** distribution of loss and flatness. Two
    marginals cannot answer it: a run can have 40% flat patches and a top tail made entirely of
    real terrain, or 4% flat patches that are the whole tail, and those call for opposite
    actions.

    Mergeable by construction — fixed bins, integer counts — because it is accumulated over a
    whole val pass and holding every per-patch value would be ~2M floats on the shipped split.
    Quantiles are therefore accurate to a bin, which at ~1.38x per bin is far finer than any
    decision anyone takes off them.
    """

    def __init__(self) -> None:
        self.edges = np.array(LOSS_BIN_EDGES, dtype=np.float64)
        self.counts = np.zeros(len(self.edges) - 1, dtype=np.int64)
        #: Of those counts, how many were patches whose target variance sat at or under §4's
        #: floor. The second half of the cross-tab.
        self.flat_counts = np.zeros_like(self.counts)
        self.n = 0
        self.n_flat = 0
        #: Patches whose loss was NaN or inf. Counted, never binned — see :meth:`update`.
        self.n_nonfinite = 0
        self._total = 0.0
        self.largest = 0.0

    def update(self, per_patch: torch.Tensor, weight: torch.Tensor, flat: torch.Tensor) -> None:
        """Add one batch: ``(N, n_patches)`` losses, their weights, and their flatness.

        Only patches with a **non-zero weight** are counted — the ones §4 actually put in the
        objective. Binning the rest would describe a population the loss never saw: every
        unmasked patch, and every masked one the valid-fraction gate excluded.

        **A non-finite loss is counted separately and never binned.** `numpy.digitize` sorts a
        NaN *past the last boundary*, so binning it would file it as the largest loss in the
        pass — corrupting :meth:`quantile`, :attr:`largest` and :meth:`tail_flat_fraction`,
        which are exactly the numbers §4's top-tail rule is read off. A run whose loss has gone
        to NaN has a much louder problem than its distribution's shape
        (:func:`pretrain.alarms.diverged`), and this keeps that problem from also quietly
        rewriting the diagnostic used to decide whether the *variance floor* is wrong.
        """
        scored = weight.detach().reshape(-1) > 0
        if not bool(scored.any()):
            return
        values = per_patch.detach().reshape(-1)[scored].to(torch.float32).cpu().numpy()
        is_flat = flat.detach().reshape(-1)[scored].cpu().numpy().astype(bool)

        finite = np.isfinite(values)
        self.n_nonfinite += int((~finite).sum())
        values, is_flat = values[finite], is_flat[finite]
        if values.size == 0:
            return

        # Clip into the range rather than dropping: see LOSS_BIN_EDGES on why the tail matters.
        binned = np.clip(
            np.digitize(values, self.edges[1:-1], right=False), 0, len(self.counts) - 1
        )
        self.counts += np.bincount(binned, minlength=len(self.counts))
        self.flat_counts += np.bincount(binned[is_flat], minlength=len(self.counts))
        self.n += values.size
        self.n_flat += int(is_flat.sum())
        self._total += float(values.sum())
        self.largest = max(self.largest, float(values.max()))

    @property
    def mean(self) -> float:
        """The exact mean — accumulated, not read back off the bins."""
        return self._total / self.n if self.n else float("nan")

    @property
    def flat_fraction(self) -> float:
        """How much of the scored population was near-flat at all — the marginal."""
        return self.n_flat / self.n if self.n else float("nan")

    @property
    def collapsed_fraction(self) -> float:
        """The share sitting in the bottom bin, i.e. scoring essentially exactly zero."""
        return float(self.counts[0]) / self.n if self.n else float("nan")

    def _bin_at(self, q: float) -> tuple[int, np.ndarray]:
        cumulative = np.cumsum(self.counts)
        index = int(np.searchsorted(cumulative, q * self.n, side="left"))
        return min(index, len(self.counts) - 1), cumulative

    def quantile(self, q: float) -> float:
        """The ``q``-quantile, interpolated linearly inside the bin it falls in."""
        if self.n == 0:
            return float("nan")
        index, cumulative = self._bin_at(q)
        below = cumulative[index - 1] if index else 0
        within = self.counts[index]
        offset = (q * self.n - below) / within if within else 0.0
        low, high = self.edges[index], self.edges[index + 1]
        return float(low + offset * (high - low))

    def tail_flat_fraction(self, q: float) -> float:
        """§4's rule, as a number: what share of the top tail was near-flat.

        The tail is every bin from the one holding the ``q``-quantile upward, so this is
        bin-granular rather than exactly the top ``1 - q``. Deliberate: the alternative is
        holding the raw values, and the answer people act on is "mostly flat" or "mostly not",
        which no bin boundary is going to flip.
        """
        if self.n == 0:
            return float("nan")
        index, _ = self._bin_at(q)
        in_tail = int(self.counts[index:].sum())
        if in_tail == 0:
            return float("nan")
        return float(self.flat_counts[index:].sum()) / in_tail

    def payload(self) -> Histogram:
        """The sink's payload type — see :class:`pretrain.sink.Histogram`."""
        return Histogram(counts=tuple(int(c) for c in self.counts), edges=tuple(self.edges))

    def scalars(self, prefix: str, *, tail_quantile: float) -> dict[str, Any]:
        """The panel §4 asks for: the distribution itself, and the numbers read off it."""
        return {
            f"{prefix}/per_patch_loss": self.payload(),
            f"{prefix}/per_patch_mean": self.mean,
            f"{prefix}/per_patch_p50": self.quantile(0.5),
            f"{prefix}/per_patch_p90": self.quantile(0.9),
            f"{prefix}/per_patch_p99": self.quantile(0.99),
            f"{prefix}/per_patch_max": self.largest,
            f"{prefix}/flat_patch_frac": self.flat_fraction,
            f"{prefix}/tail_flat_frac": self.tail_flat_fraction(tail_quantile),
            f"{prefix}/collapsed_patch_frac": self.collapsed_fraction,
            # Kept out of every number above, so it has to be reported on its own or a pass
            # that scored nothing but NaNs would look like a pass with no patches in it.
            f"{prefix}/nonfinite_patches": self.n_nonfinite,
        }


# --- §10.1: the fixed-mask val pass -------------------------------------------


@dataclass(frozen=True)
class ValPass:
    """One val evaluation: the comparable loss, what it was over, and its distribution."""

    #: §4's objective over the whole split — a **patch-weighted** mean, not a mean of batch
    #: means. With ``drop_last`` off the last batch is short, and averaging batch means would
    #: weight its tiles more heavily than the rest, giving a val number that moves with the
    #: batch size.
    loss: float
    tiles: int
    #: Patches that entered the objective: masked, and past §4's valid-fraction gate.
    patches: int
    distribution: LossHistogram


class ScoredBatch(NamedTuple):
    """What one val batch contributed — the two running sums, and how many tiles they came from.

    Named rather than a bare ``tuple[float, float, int]`` because the two sums are only
    meaningful together: :attr:`ValPass.loss` is their quotient over the whole split, and a
    caller that averaged :attr:`weighted_loss` alone would be back to a mean of batch means.
    """

    #: ``Σ per_patch · weight`` over the batch — the numerator of a patch-weighted mean.
    weighted_loss: float
    #: ``Σ weight`` — its denominator, i.e. how many patches actually entered the objective.
    weight: float
    tiles: int


def accumulate(
    histogram: LossHistogram,
    trainer: Trainer,
    batch: dict,
    *,
    var_floor: float,
    generator: torch.Generator | None = None,
) -> ScoredBatch:
    """Score one batch, fold it into ``histogram``, and return what it contributed.

    The flatness flag is derived here from the target rather than being carried out of the loss
    module, and that is not a duplication worth removing: what §4's rule needs is *"was this
    patch near-flat"*, a property of the **target**, whereas the loss's own use of the floor is a
    clamp inside a normalization it does not report. Recomputing
    :func:`~pretrain.loss.patch_statistics` costs one pass over the targets, once an epoch.

    What it does **not** re-derive is the pixel weights: those come back from
    :meth:`Trainer.score <pretrain.train.Trainer.score>` exactly as the loss used them. Under
    §4's ocean weighting the variance the floor is compared against is the variance over a
    patch's *valid* pixels, and computed unweighted a half-ocean patch measures its variance
    across the land/ocean step — the largest gradient in the tile — and never looks flat. Two
    derivations of one thing is how that drifts, and the drifted version trains fine while
    misclassifying precisely the population the rule exists to find.
    """
    scored = trainer.score(batch, generator=generator)
    _, variance = patch_statistics(scored.target, weights=scored.pixel_weights)
    out = scored.out
    histogram.update(out.per_patch, out.weight, variance.squeeze(-1) <= var_floor)
    # Detached rather than relying on the caller's `no_grad`: #92 deliberately does not detach
    # its diagnostics, and this function is reachable outside a val pass (a pre-flight check on
    # one synthetic tile, §15.6). A monitoring accumulator is never differentiated through.
    weighted = (out.per_patch * out.weight).detach()
    return ScoredBatch(
        weighted_loss=float(weighted.sum()),
        weight=float(out.weight.detach().sum()),
        tiles=int(batch["image"].shape[0]),
    )


def evaluate(trainer: Trainer, loader, *, mask_seed: int, var_floor: float = 1e-4) -> ValPass:
    """§10.1's val pass: every val tile once, under masks pinned by ``mask_seed``.

    Three properties, and each is a separate way for this number to stop meaning anything:

    - **Fixed masks.** One :class:`torch.Generator`, seeded per pass, so epoch 40 hides the same
      tokens epoch 39 did. Without it the val loss moves with the mask draw and comparing two
      epochs compares two measurements.
    - **Nothing drawn from the ambient RNG.** The generator is explicit, so a val pass does not
      shift the masks the next training step draws. A run whose trajectory depended on whether
      it was being measured would be reproducible only by accident.
    - **A patch-weighted mean.** See :attr:`ValPass.loss`.

    The model is put in eval mode and restored, and the whole pass runs under
    :func:`torch.no_grad`, so nothing here can leak a gradient into the next optimizer step.
    """
    was_training = trainer.model.training
    trainer.model.eval()
    generator = torch.Generator(device=trainer.device).manual_seed(mask_seed)
    histogram = LossHistogram()
    weighted = weight = 0.0
    tiles = 0
    try:
        with torch.no_grad():
            for batch in loader:
                scored = accumulate(
                    histogram, trainer, batch, var_floor=var_floor, generator=generator
                )
                weighted += scored.weighted_loss
                weight += scored.weight
                tiles += scored.tiles
    finally:
        trainer.model.train(was_training)
    return ValPass(
        loss=weighted / weight if weight else float("nan"),
        tiles=tiles,
        patches=histogram.n,
        distribution=histogram,
    )


# --- The epoch ----------------------------------------------------------------


@dataclass(frozen=True)
class EpochRecord:
    """What one epoch reported. The row of §10's dashboard, and what #94 checkpoints beside."""

    epoch: int
    #: Optimizer steps the epoch took.
    steps: int
    #: §4's objective averaged over those steps — the *training* number, under fresh masks and
    #: §6's augmentation, so it is not comparable to :attr:`val_loss` and is not meant to be.
    train_loss: float
    #: The LR the epoch's last step ran at (§6's warmup/cosine curve).
    lr: float
    #: ``None`` only when the monitor was built with no val loader at all (a test, or a corpus
    #: whose val split is empty) — §10.1 asks for a pass every epoch and there is no cadence
    #: knob to skip one. Deliberately not carried forward from the previous epoch either way: a
    #: repeated number in a trend reads as a measurement that did not move.
    val: ValPass | None
    #: #95's counters for the epoch, or ``None`` when the caller measured nothing.
    throughput: ThroughputSummary | None
    alarms: tuple[Alarm, ...]
    #: §10.3's probe point (#99), on the epochs its cadence lands on and ``None`` between them.
    #: Carried here so an ``on_epoch`` caller can print it; the numbers themselves go to the
    #: sink through :mod:`pretrain.probe`, not through :meth:`scalars` — see the module
    #: docstring on which side of the seam each ticket sits.
    probe: ProbeReport | None = None

    @property
    def val_loss(self) -> float | None:
        return None if self.val is None else self.val.loss

    def scalars(self, *, tail_quantile: float) -> dict[str, Any]:
        """The whole epoch as one sink payload, under keys that do not move between runs."""
        values: dict[str, Any] = {
            "train/loss": self.train_loss,
            "train/lr": self.lr,
            "train/steps": self.steps,
        }
        if self.val is not None:
            values["val/loss"] = self.val.loss
            values["val/tiles"] = self.val.tiles
            values["val/scored_patches"] = self.val.patches
            values.update(self.val.distribution.scalars("val", tail_quantile=tail_quantile))
        if self.throughput is not None:
            values["throughput/samples_per_s"] = self.throughput.samples_per_second
            values["throughput/step_time_s"] = (
                self.throughput.seconds / self.throughput.steps if self.throughput.steps else 0.0
            )
            values["throughput/loader_wait_frac"] = self.throughput.loader_wait_fraction
            values["throughput/steps"] = self.throughput.steps
        values["alarms/fired"] = len(self.alarms)
        values.update({f"alarms/{alarm.name}": 1 for alarm in self.alarms})
        return values


def format_epoch(record: EpochRecord) -> str:
    """One epoch, as a line. The console half of §10's dashboard."""
    val = "   val    -   " if record.val is None else f"   val {record.val_loss:8.4f}"
    throughput = (
        "" if record.throughput is None else f"   {format_throughput(record.throughput)}"
    )
    return (
        f"epoch {record.epoch:4d}  lr {record.lr:.3e}  train {record.train_loss:8.4f}"
        f"{val}{throughput}"
    )


class EpochMonitor:
    """Accumulates a run's steps into epochs, scores val, checks §10's alarms, logs the lot.

    Driven rather than driving: :meth:`on_step` per optimizer step and :meth:`end_of_epoch` at
    the boundary. That is what lets it hang off #94's resumable loop unchanged —
    :func:`train_epochs` is those two callbacks and nothing else — instead of §10's reporting
    being a loop of its own that a run would have to choose between.
    """

    def __init__(
        self,
        cfg: RunConfig,
        *,
        trainer: Trainer | None,
        logger: RunLogger,
        val_loader=None,
        on_alarm: Callable[[Alarm], None] | None = None,
        probe: KnnProbe | None = None,
        recon: ReconGrids | None = None,
        watch: CompileWatch | None = None,
    ) -> None:
        self.cfg = cfg
        self.trainer = trainer
        self.logger = logger
        self.val_loader = val_loader
        self.on_alarm = on_alarm
        #: #109's split counter, or ``None`` for a caller that is not measuring compiles. Held
        #: so the boundary can *declare itself* — everything §10 does here forwards through the
        #: same compiled module in ``eval()`` under `no_grad`, which Dynamo guards on, so it
        #: earns its own graphs the first time it runs. Attributing them to the training step
        #: is what made §7.3's check fire on three consecutive healthy runs.
        self.watch = watch
        self._frames = FrameCounts()
        self._stepped = False
        #: §10.2's reconstruction grids (#98), or ``None`` when no panel is committed at the
        #: configured path or the cadence is 0. Held for the life of the run rather than rebuilt
        #: per render: it carries the panel's pixels, read once, so epoch 400's grid is drawn
        #: from the same bytes epoch 0's was.
        self.recon = recon
        #: §10.3's probe (#99), or ``None``. Driven from here — the epoch boundary is the only
        #: place that knows what epoch it is — but it writes its own numbers into the sink, so
        #: this module never learns what a kNN accuracy is.
        self.probe = probe
        #: ``[(epoch, loss), ...]`` — what :mod:`pretrain.alarms` reads. **Epochs, not steps**:
        #: §10.1's conditions are written in epochs ("a plateau in the first 20 epochs"), and
        #: fed a step series they would fire on within-epoch noise.
        self.train_trajectory: list[tuple[int, float]] = []
        self.val_trajectory: list[tuple[int, float]] = []
        self.records: list[EpochRecord] = []
        self._fired: list[Alarm] = []
        self._losses: list[float] = []
        self._lr = 0.0

    def _attributed(self):
        """The boundary declaring itself to #109's watch — or nothing, when there is no watch."""
        return nullcontext() if self.watch is None else self.watch.monitoring()

    def _read_frames(self) -> FrameCounts:
        """Both tallies as one value. ``FrameCounts()`` when nobody is watching."""
        if self.watch is None:
            return FrameCounts()
        return FrameCounts(train=self.watch.compiled, monitoring=self.watch.monitored)

    def on_step(self, record: StepRecord) -> None:
        """Fold one optimizer step into the epoch being accumulated.

        Only the loss and the LR: the throughput counters are #95's
        (:class:`~pretrain.throughput.ThroughputMeter`) and arrive whole at
        :meth:`end_of_epoch`, so this does not count samples or read a clock. A second
        stopwatch here would give a second answer to a question that already has one.

        It is also where #109's watch is baselined. The first step *this process* takes is
        §7.3's compile — which is the first step of a resumed run too, so this keys on the
        process rather than on step 0 — and everything counted after it is a recompile. Owned
        here rather than by each caller, because a caller that forgot would charge the warmup
        compile to §7.3's claim and report a recompile on every healthy run.
        """
        if not self._stepped:
            self._stepped = True
            if self.watch is not None:
                self.watch.reset()
        self._losses.append(record.loss)
        self._lr = record.lr

    def end_of_epoch(
        self, epoch: int, *, throughput: ThroughputSummary | None = None
    ) -> EpochRecord:
        """Close the epoch: score val, check §10's alarms, log the panel, return the row."""
        monitoring = self.cfg.monitoring
        train_loss = float(np.mean(self._losses)) if self._losses else float("nan")
        # Read *before* this boundary runs, so what the §7 check sees is complete: every frame
        # the previous boundary compiled, plus every frame this epoch's steps did. Reading it
        # afterwards would put half of this boundary's own compiles inside the number and the
        # other half outside it, depending on where in the sequence the check happened to sit.
        frames_before, frames = self._frames, self._read_frames()
        self._frames = frames
        # Every epoch, unconditionally — §10.1 sets no cadence, and a knob to skip it would take
        # the per-patch distribution and the val-side collapse alarm off the skipped epochs too.
        val = None
        if self.val_loader is not None and self.trainer is not None:
            with self._attributed():
                val = evaluate(
                    self.trainer,
                    self.val_loader,
                    mask_seed=monitoring.val_mask_seed,
                    var_floor=self.cfg.loss.norm_pix_var_floor,
                )

        self.train_trajectory.append((epoch, train_loss))
        if val is not None:
            self.val_trajectory.append((epoch, val.loss))
        alarms = check_alarms(
            monitoring,
            train=self.train_trajectory,
            val=self.val_trajectory,
            loader_wait_fraction=(
                None if throughput is None else throughput.loader_wait_fraction
            ),
            frames=None if self.watch is None else frames,
            frames_before=frames_before,
        )

        # §10.3's probe, on its own cadence — after val, so an epoch's cheap numbers are all
        # taken before the expensive one. It logs itself; what comes back is for the console.
        with self._attributed():
            probe = (
                None
                if self.probe is None or self.trainer is None
                else self.probe.maybe_run(self.trainer.model, epoch)
            )

        record = EpochRecord(
            epoch=epoch,
            steps=len(self._losses),
            train_loss=train_loss,
            lr=self._lr,
            val=val,
            throughput=throughput,
            alarms=tuple(alarms),
            probe=probe,
        )
        values = record.scalars(tail_quantile=monitoring.per_patch_tail_quantile)
        # §10.2's grids, on their own cadence, merged into the epoch's own payload rather than
        # logged separately: W&B's step axis is the epoch and a grid belongs to the epoch whose
        # losses sit beside it. Rendered last, after every number is taken, because it is the
        # one part of a boundary that runs a forward pass purely to be looked at.
        if self.recon is not None and self.trainer is not None and self.recon.due(epoch):
            with self._attributed():
                values.update(self.recon.render(self.trainer))
        if self.watch is not None:
            # Logged every epoch so §7.3's precondition is chartable from a laptop rather than
            # legible only in an ssh scrollback on a box that will be destroyed. Two flat lines
            # at 0 and 18 is what a healthy run looks like; #109 is what it cost to have to
            # infer that from throughput.
            values["compile/train_frames"] = frames.train
            values["compile/monitor_frames"] = frames.monitoring
        self.logger.log(values, step=epoch)
        for alarm in alarms:
            self._fired.append(alarm)
            if self.on_alarm is not None:
                self.on_alarm(alarm)
        self.records.append(record)
        self._losses = []
        return record

    def finish(self) -> None:
        """The run's last word: the bests, and what fired over the whole run.

        The alarm names go into the summary as well as the per-epoch series, because a run's
        page should answer "did anything fire?" without anyone scrubbing a 400-point chart.
        """
        summary: dict[str, Any] = {
            "alarms/fired": len(self._fired),
            "alarms/names": ", ".join(sorted({a.name for a in self._fired})) or "none",
            # What §10.2 was rendered against, or that it was not. Recorded rather than raised
            # (:mod:`pretrain.fingerprint`'s rule for the split): an absent panel is a §15
            # checklist failure, not a reason to refuse to train — but a run with no grids and
            # a run whose grids nobody scrolled to look identical afterwards.
            "recon/panel": (
                "absent"
                if self.recon is None
                else f"{len(self.recon.panel)} tiles over {len(self.recon.panel.strata)} strata, "
                f"md5 {self.recon.panel_md5 or '-'}"
            ),
        }
        if self.train_trajectory:
            summary["train/best_loss"] = min(loss for _, loss in self.train_trajectory)
        if self.val_trajectory:
            summary["val/best_loss"] = min(loss for _, loss in self.val_trajectory)
            summary["val/best_epoch"] = min(self.val_trajectory, key=lambda p: p[1])[0]
        self.logger.summary(summary)


# --- The loop ------------------------------------------------------------------


def train_epochs(
    cfg: RunConfig,
    *,
    epochs: int | None = None,
    steps: int | None = None,
    device: torch.device | str | None = None,
    logger: RunLogger | None = None,
    policy: CheckpointPolicy | None = None,
    resume: bool | str | Path = False,
    on_epoch: Callable[[EpochRecord], None] | None = None,
    on_alarm: Callable[[Alarm], None] | None = None,
    on_step: Callable[[StepRecord], None] | None = None,
    on_start: Callable[[Trainer], None] | None = None,
    watch: CompileWatch | None = None,
) -> list[EpochRecord]:
    """**The run.** #94's resumable loop with §10's panel on its epoch boundary (#106).

    This is the whole composition, and until #106 it did not exist. #93 built the loop, #94
    built checkpoints and exact resume, #95 built the entrypoint and the counters, #97/#98/#99
    built the instruments — and nothing put the three together, so the command the ops runbook
    tells a rented box to run produced no checkpoints, no W&B run and no val pass. Every piece
    was built and tested; this is the wrap the docstrings had been promising.

    What it adds to :func:`~pretrain.train.train_resumable` is the epoch boundary's *content*:
    where §10.1's val pass runs, where its alarms are checked, where §4's per-patch distribution
    is accumulated, where §10.2's grids and §10.3's probe fire on their cadences, and where all
    of it is written to the sink. The loop, the checkpoints and the counters are not
    reimplemented here — they are the function this calls.

    ``epochs`` is the run's **total**, not an increment, and defaults to the config's own
    schedule. ``steps`` truncates it for a smoke run, in which case a final partial epoch simply
    never reaches a boundary and is never scored — which is honest: half an epoch's train loss
    is not a row of §10's table.

    ``policy`` and ``resume`` are #94's, passed straight through: no policy is a run that writes
    no checkpoints, which is a test or a laptop, never a rented box. ``logger`` is #97's sink; a
    caller that does not supply one gets :func:`~pretrain.sink.open_run_logger`'s and this closes
    it.

    ``watch`` is #109's split frame counter. Passing one is what lets the boundary attribute its
    own compiles to the monitoring path instead of to §7.3's training-step claim; without it the
    §7 checks simply do not run, which is what a test or a laptop wants.

    The :class:`EpochMonitor` is built inside ``on_start`` rather than before the call, because
    the :class:`~pretrain.train.Trainer` it scores with is the resumed one — built from the
    config and then loaded over from the checkpoint. A monitor holding a trainer built out here
    would run its val pass against a model the run had already replaced.
    """
    epochs = cfg.schedule.epochs if epochs is None else epochs
    val_loader = build_loader(cfg, cfg.data.val_split, augment=False, shuffle=False)
    owned = logger is None
    logger = logger or open_run_logger(cfg)
    monitor: EpochMonitor | None = None

    def start(trainer: Trainer) -> None:
        nonlocal monitor
        monitor = EpochMonitor(
            cfg,
            trainer=trainer,
            logger=logger,
            val_loader=val_loader,
            on_alarm=on_alarm,
            probe=probe_for(cfg, logger),
            recon=ReconGrids.from_config(cfg),
            watch=watch,
        )
        if on_start is not None:
            on_start(trainer)

    def step(record: StepRecord) -> None:
        monitor.on_step(record)
        if on_step is not None:
            on_step(record)

    def boundary(epoch: int, throughput: ThroughputSummary) -> None:
        result = monitor.end_of_epoch(epoch, throughput=throughput)
        if on_epoch is not None:
            on_epoch(result)

    try:
        train_resumable(
            cfg,
            steps=steps,
            epochs=None if steps is not None else epochs,
            policy=policy,
            resume=resume,
            device=device,
            on_start=start,
            on_step=step,
            on_epoch=boundary,
        )
        # ``monitor`` is unguarded inside the three callbacks above and guarded here, and the
        # asymmetry is real rather than sloppy: a callback cannot run before ``on_start`` built
        # it, but the *call* can fail before ``on_start`` is reached at all — a resumed
        # checkpoint that names a different corpus is refused in `train_resumable` before the
        # trainer is handed over. Then there is no run to finish and no records to return.
        if monitor is not None:
            monitor.finish()
    finally:
        if owned:
            logger.close()
    return [] if monitor is None else monitor.records

