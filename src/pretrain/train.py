"""The training step, and the loop around it: the tracer bullet that actually trains (#93).

Everything Phase 1 has built so far is a piece — a config (#89), a model (#90), an augmented
read path (#91), an objective (#92, #96), an optional warm start (#102). This is the file where
they are one thing: loader -> augment -> mask -> encode -> decode -> loss -> backward -> step,
running, with the loss going down.

Two loops live here. :func:`train` is the tracer bullet — steps, no checkpoints, no
monitoring — and ``python -m pretrain.train`` runs it so a step can be watched going by.
:func:`train_resumable` is #94's: the same :class:`Trainer`, driven through an
:class:`EpochCursor` so it survives the box vanishing, timed by #95's counters, and reporting
each epoch boundary as a seam. **That is the loop that ships.**
:func:`pretrain.monitor.train_epochs` puts §10's panel on its boundary and
``python -m pretrain.launch`` is the one command that calls it (#106); nothing here knows what
a val loss or a W&B run is, which is why the seams are callbacks.

## What the step is, precisely

One optimizer step is ``accum_iter`` micro-batches: each is forwarded under §7.1's bf16
autocast, scored by §4's objective in fp32, and backwarded at ``1/N`` so the accumulated
gradient is the mean over the effective batch — the *same gradient* a single batch of ``N x``
the size would have produced. That equivalence is what makes §6's 1024 x 2 (ViT-S) and 512 x 4
(ViT-B) the same recipe at two memory footprints, and it is asserted rather than assumed:
`test_gradient_accumulation_matches_one_batch_of_twice_the_size`.

## bf16, and the absence of a GradScaler

§7.1 chooses bf16 autocast on Ada/Blackwell and no gradient scaler, and the *absence* is the
point: fp16 needs a scaler because its dynamic range cannot hold the gradients, and the official
MAE repo documents the NaNs that follow when the scaler's bookkeeping and the accumulation
interact badly. bf16 has fp32's exponent range, so there is nothing to scale. The loss and the
`norm_pix` statistics stay fp32 regardless (#92 disables autocast over its own body and refuses
a target that arrives narrower), so what autocast actually covers here is the encoder and
decoder matmuls — which is where the speed is.

## Determinism, and what one seed buys

§6 asks for a fixed seed on model init and the DataLoader generator, so that two runs of one
config differ only by nondeterministic kernels. :func:`seed_run` sets both, and there are
exactly three streams to think about:

- **Model init** seeds itself inside :meth:`MaskedAutoencoder.from_config
  <pretrain.model.MaskedAutoencoder.from_config>`, in a forked RNG, so building a model does not
  consume from the stream the run then draws its masks from.
- **The DataLoader's shuffle** takes an explicit :class:`torch.Generator` seeded from the config.
- **Masking** draws from the ambient torch RNG.
- **Augmentation** draws from no stream at all: since #104 the dihedral element is a pure
  function of ``(config seed, epoch, sample index)``, so it is the same at every worker count and
  the same after a resume — see :mod:`pretrain.augment`.

`test_the_same_seed_and_config_reproduce_the_loss_trajectory` pins the result on CPU. It is a
weaker claim on a GPU, where the kernels themselves are nondeterministic, and #94's kill-and-
resume test is the one that has to hold there.
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor
from torch.utils.data import Sampler

from dataset.loader import build_dataloader

from .augment import EpochIndex, make_augmented
from .checkpoint import (
    DEFAULT_INTERVAL_SECONDS,
    CheckpointPolicy,
    RunState,
    capture_rng,
    load_checkpoint,
    resolve_resume,
    restore_rng,
)
from .config import RunConfig
from .data import SplitDataset
from .fingerprint import fingerprint_for
from .header import add_config_arguments, config_from_args, format_header
from .init import initialize_from_config
from .loss import MaeLoss, MaeLossOutput, pixel_weights_from_land_mask
from .model import MaeOutput, MaskedAutoencoder
from .optim import build_optimizer
from .runtime import compile_model, verify_attention
from .schedule import LrSchedule
from .throughput import StepThroughput, ThroughputMeter, ThroughputSummary, summarize

# --- The read path ------------------------------------------------------------


def seeded_generator(cfg: RunConfig) -> torch.Generator:
    """The DataLoader's generator, from :attr:`Runtime.seed <pretrain.config.Runtime>` (§6)."""
    generator = torch.Generator()
    generator.manual_seed(cfg.runtime.seed)
    return generator


def seed_run(cfg: RunConfig) -> torch.Generator:
    """Seed the run's RNG streams and hand back the DataLoader's generator (§6).

    The ambient seed is what the in-model mask draw (#90) reads. Model init does *not* read it —
    it forks its own — so adding a decoder block cannot silently reorder the corpus. Neither
    does the dihedral augmentation, which since #104 hashes ``cfg.runtime.seed`` with the epoch
    and the sample index instead of drawing from any stream.
    """
    torch.manual_seed(cfg.runtime.seed)
    return seeded_generator(cfg)


def epoch_seed(seed: int, epoch: int) -> int:
    """The seed epoch ``epoch`` of a run seeded ``seed`` shuffles with.

    A keyed hash rather than ``seed + epoch``, for the same reason
    :attr:`RunConfig.config_hash <pretrain.config.RunConfig>` is one: adjacent seeds should not
    produce related streams, and two runs one seed apart would otherwise share every epoch's
    order but the first. BLAKE2b is the hash family every tie-break in :mod:`dataset.sampling`
    already uses.
    """
    import hashlib

    digest = hashlib.blake2b(f"{seed}:{epoch}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1  # torch seeds are non-negative int64


class EpochShuffleSampler(Sampler[int]):
    """The training order: a pure function of ``(seed, epoch)``, sliceable mid-epoch (#94, §11).

    This exists because of one bug, and the bug is invisible. A run killed mid-epoch and resumed
    against ``DataLoader(shuffle=True)`` gets a **fresh permutation**: the epoch it resumes into
    shows some tiles twice and never shows others, and nothing about the loss curve, the
    gradient norm or the checkpoint says so. Owning the permutation is what makes "resume
    continues the same order" a property rather than a hope.

    Deriving each epoch's order from ``(seed, epoch)`` — rather than from a rolling generator —
    is what makes it *reproducible from a fresh process*. A generator's state at the start of
    epoch 300 depends on every draw before it, including how many times a DataLoader iterator
    was constructed, which differs between a first launch and a resumed one (and again between
    ``persistent_workers`` on and off). A pure function has no such history to get wrong.

    :attr:`skip` is how a resumed epoch starts partway through: the same permutation, minus the
    samples already seen. With ``drop_last`` that is exactly right —
    ``(n - k*B) // B == n // B - k`` — so the resumed epoch has precisely the batches the
    interrupted one had left, in the order it would have read them.

    Iterating to exhaustion advances to the next epoch and clears :attr:`skip`, so a plain
    ``for epoch in range(N): for batch in loader:`` still reshuffles every pass (#93 relies on
    it). :meth:`set_epoch` overrides that, and is what :class:`EpochCursor` drives.

    What it yields is an :class:`~pretrain.augment.EpochIndex` — an ``int``, plus the epoch it
    was drawn for (#104). That is how the epoch reaches a *worker* process, which is handed
    indices and knows nothing else; :class:`~pretrain.augment.DihedralDataset` needs it because
    its element is a pure function of ``(seed, epoch, index)`` rather than a stream. Nothing
    else downstream has to know: an unaugmented shuffled loader indexes
    :class:`~pretrain.data.SplitDataset` with these directly and they behave as the ints they
    are.
    """

    def __init__(self, n: int, *, seed: int, epoch: int = 0, skip: int = 0) -> None:
        if n < 0 or skip < 0:
            raise ValueError(f"a sampler over {n} samples cannot skip {skip}")
        self.n = n
        self.seed = seed
        self.epoch = epoch
        self.skip = skip

    def set_epoch(self, epoch: int, *, skip: int = 0) -> None:
        """Read epoch ``epoch``'s order next, starting ``skip`` samples into it."""
        self.epoch = epoch
        self.skip = skip

    def permutation(self) -> list[int]:
        """Epoch :attr:`epoch`'s whole order, before :attr:`skip` is applied."""
        generator = torch.Generator()
        generator.manual_seed(epoch_seed(self.seed, self.epoch))
        return torch.randperm(self.n, generator=generator).tolist()

    def __len__(self) -> int:
        return max(self.n - self.skip, 0)

    def __iter__(self):
        epoch, skip = self.epoch, self.skip
        # Keyed with the epoch captured above, not with `self.epoch`: a `set_epoch` issued
        # mid-iteration (which is what a resume does) must not relabel the tail of the
        # permutation this pass is still handing out with an epoch it did not come from.
        yield from (EpochIndex(i, epoch) for i in self.permutation()[skip:])
        # Exhausted, and nobody moved us meanwhile: the next pass is the next epoch. Guarded so
        # that a `set_epoch` issued *during* iteration (which is what a resume does, and what
        # prefetching workers make possible) is not undone by the iterator that preceded it.
        if (self.epoch, self.skip) == (epoch, skip):
            self.set_epoch(epoch + 1)


def build_loader(
    cfg: RunConfig,
    split: str,
    *,
    augment: bool = True,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
    **overrides: Any,
):
    """ADR-0003's read path for one split, with §6's augmentation wrapped around it.

    Three layers, and the order matters: :class:`~pretrain.data.SplitDataset` selects the rows
    (and refuses a contaminated split at construction, plan §15.2),
    :func:`~pretrain.augment.make_augmented` wraps the sample dict, and
    :func:`dataset.loader.build_dataloader` supplies ADR-0003 §1's batching flags. The read path
    is called, never subclassed and never reimplemented: the ETL core stays torch-free and
    augmentation-free, and these three layers are the whole of what the training package adds to
    it. (#93 did have to fix one thing *in* `dataset.loader` — its Dataset class was unpicklable,
    so ``num_workers > 0`` failed under the `spawn` start method — but that is the same class,
    reading the same bytes, with the same API.)

    ``drop_last`` is on for training and off elsewhere. A short final batch is a second tensor
    shape, which is a `torch.compile(dynamic=False)` recompile (§7.3) — and at 500k tiles the
    dropped remainder is under a batch per epoch, re-drawn next epoch by the shuffle. A val pass
    keeps its tail, because there the point is to score every tile exactly once.

    ``augment=False`` is how §10.1's val loader is built: a fresh dihedral draw per read would
    make the val loss a moving measurement rather than a comparable one.

    A shuffled loader reads through an :class:`EpochShuffleSampler` rather than through
    ``DataLoader(shuffle=True)``, so that #94 can resume mid-epoch into the order it was
    interrupted in. A val pass keeps manifest order and needs neither.
    """
    dataset = SplitDataset(
        cfg.data.corpus_dir,
        split,
        blocks_path=cfg.data.split_blocks_path,
        assignment_path=cfg.data.split_assignment_path,
        normalization=cfg.data.normalization,
    )
    if augment:
        dataset = make_augmented(dataset, cfg.data, seed=cfg.runtime.seed)
    kwargs: dict[str, Any] = {
        "batch_size": cfg.optimization.batch_size,
        "num_workers": cfg.data.num_workers,
        # The order is the sampler's; `shuffle` stays False because DataLoader refuses both.
        "shuffle": False,
        "sampler": EpochShuffleSampler(len(dataset), seed=cfg.runtime.seed) if shuffle else None,
        "drop_last": shuffle,
        "generator": generator if generator is not None else seeded_generator(cfg),
    }
    kwargs.update(overrides)
    return build_dataloader(dataset, **kwargs)


def steps_per_epoch_for(loader, cfg: RunConfig) -> int:
    """Optimizer steps in one epoch — micro-batches divided by the accumulation factor.

    Exact, not approximate: :func:`micro_batch_groups` drops the remainder rather than stepping
    on it, so this is the number of steps an epoch actually produces. That matters because it is
    the unit the LR schedule is written in — a count that ran ahead of the loop by a step per
    epoch would slide the whole 400-epoch curve by a couple of epochs by the end.
    """
    return len(loader) // cfg.optimization.accum_iter


def epoch_groups(batches, accum_iter: int) -> Iterator[list[dict]]:
    """**One** pass over ``batches``, grouped into one list per optimizer step.

    **A short group at the end is dropped**, for the same reason ``drop_last`` drops a short
    batch: an optimizer step is defined as *the effective batch*, and a step taken on half of one
    at the LR the whole batch earns is a step outside the recipe. At the shipped shape that costs
    at most one micro-batch per epoch, re-drawn by the next epoch's shuffle. It also keeps the
    count :func:`steps_per_epoch_for` gives the schedule exact.

    Takes an iterable rather than the loader itself, so a caller can wrap the loader first —
    which is how #95's :class:`~pretrain.throughput.ThroughputMeter` charges the time spent
    waiting for a batch to §10.4's loader-wait fraction without this function knowing anything
    about it.
    """
    group: list[dict] = []
    for batch in batches:
        group.append(batch)
        if len(group) == accum_iter:
            yield group
            group = []


def assert_a_step_fits(loader, accum_iter: int) -> None:
    """Refuse a split too small to accumulate one optimizer step out of.

    With ``drop_last`` on, such a loader yields batches that never fill a group, so a loop
    waiting for its first step waits forever — on a rented box, an hour of paid nothing.
    """
    if len(loader) // accum_iter == 0:
        raise ValueError(
            f"{len(loader)} batch(es) of {accum_iter} accumulated is no complete step — a split "
            "smaller than one effective batch, with drop_last on. Left alone this would spin "
            "forever without taking a step"
        )


def micro_batch_groups(loader, accum_iter: int, *, steps: int) -> Iterator[list[dict]]:
    """Group the loader's batches into one list per optimizer step, for ``steps`` steps.

    Each pass over ``loader`` is an epoch, and re-iterating it is what re-shuffles: the sample
    order in epoch 2 is not epoch 1's. (Cycling one materialized epoch instead would train fine
    and quietly turn a 400-epoch run into 400 repetitions of one ordering.)
    """
    assert_a_step_fits(loader, accum_iter)
    produced = 0
    while produced < steps:
        for group in epoch_groups(loader, accum_iter):
            yield group
            produced += 1
            if produced == steps:
                return


@dataclass(frozen=True)
class EpochGroup:
    """One optimizer step's micro-batches, and where in the schedule they came from (#94)."""

    micro_batches: list[dict]
    #: The epoch these batches were drawn from. 0-based, so it is the *index*, while
    #: :attr:`RunState.epoch <pretrain.checkpoint.RunState>` counts epochs completed.
    epoch: int
    #: This group is the last complete step of its epoch — §11's per-epoch checkpoint trigger.
    ends_epoch: bool


class EpochCursor:
    """Where the loop is in the corpus's order, and the iterator that keeps it there (#94, §11).

    :func:`micro_batch_groups` walks the same batches and is the right thing for a run that
    will not be interrupted. This one adds the two things a resume needs and nothing else:
    it drives an :class:`EpochShuffleSampler` so a resumed epoch continues the order it was
    interrupted in, and it tracks a :class:`~pretrain.checkpoint.RunState` precise enough to
    restart from.

    It does its own grouping rather than calling :func:`epoch_groups`, for one reason worth
    stating: **the epoch has to be over before its last group is handed out.**

    What is left over at an epoch's end is the batches an incomplete *group* would have been
    built from — ``len(loader) % accum_iter`` of them, which at ``accum_iter = 2`` is one. Those
    batches are genuinely fetched, and a fetched batch is one the epoch has consumed. What is at
    stake is that count rather than the samples: since #104 the augmentation is a pure function
    of ``(seed, epoch, index)``, so a re-read tail comes back bit-identical — but it is re-*trained*
    on, which moves weights and optimizer moments. (Before #104 a re-read silently changed the
    transforms as well, which is the stronger claim this paragraph used to make.) The remainder
    ``drop_last`` drops is a different thing and costs nothing here: torch's ``BatchSampler``
    abandons a short batch's indices without ever fetching them, so those samples are never
    read. So the drain loop exists for the accumulation remainder, and it
    runs *before* the final group is yielded: a checkpoint written at an epoch boundary is then
    taken on the far side of it, with a clean ``batches_into_epoch = 0``, and the resumed run
    does not re-read a tail the original had already consumed.
    """

    def __init__(
        self,
        loader,
        *,
        accum_iter: int,
        steps_per_epoch: int,
        state: RunState = RunState(),
    ) -> None:
        self.loader = loader
        self.accum_iter = accum_iter
        #: A *whole* epoch's step count — the schedule's unit. Computed by the caller before any
        #: skip is applied, since ``len(loader)`` shrinks under one.
        self.steps_per_epoch = steps_per_epoch
        self.sampler = loader.sampler
        if not isinstance(self.sampler, EpochShuffleSampler):
            raise TypeError(
                f"an exactly resumable run needs an EpochShuffleSampler, not "
                f"{type(self.sampler).__name__}: an order that is not a function of "
                f"(seed, epoch) cannot be resumed into (plan §11)"
            )
        self.epoch = state.epoch
        self.batches_into_epoch = state.batches_into_epoch

    def state(self, step: int) -> RunState:
        """The position to checkpoint, given the optimizer step count the Trainer holds."""
        return RunState(
            epoch=self.epoch, step=step, batches_into_epoch=self.batches_into_epoch
        )

    def groups(self) -> Iterator[EpochGroup]:
        """Micro-batch groups, epoch after epoch, forever — the caller decides when to stop."""
        assert_a_step_fits(self.loader, self.accum_iter)
        while True:
            self.sampler.set_epoch(
                self.epoch, skip=self.batches_into_epoch * self.loader.batch_size
            )
            done = self.batches_into_epoch // self.accum_iter
            iterator = iter(self.loader)
            group: list[dict] = []
            final: list[dict] | None = None
            for batch in iterator:
                self.batches_into_epoch += 1
                group.append(batch)
                if len(group) < self.accum_iter:
                    continue
                done += 1
                if done < self.steps_per_epoch:
                    yield EpochGroup(group, epoch=self.epoch, ends_epoch=False)
                    group = []
                    continue
                final = group
                for _ in iterator:  # the remainder `drop_last` discards — read, then dropped
                    self.batches_into_epoch += 1
                break
            epoch = self.epoch
            self.epoch += 1
            self.batches_into_epoch = 0
            if final is not None:
                yield EpochGroup(final, epoch=epoch, ends_epoch=True)


# --- One step -----------------------------------------------------------------


@dataclass(frozen=True)
class StepRecord:
    """What one optimizer step did. The seam #97's monitoring reads, and #94's resume writes."""

    #: Optimizer steps completed before this one — 0 for the first.
    step: int
    #: The fractional epoch the step sat in, which is what the LR was computed from.
    epoch: float
    #: The learning rate the step was taken at (§6's warmup/cosine curve).
    lr: float
    #: §4's objective over the effective batch: the mean of the micro-batch losses.
    loss: float
    #: How many micro-batches the step accumulated. Always ``accum_iter`` from
    #: :func:`micro_batch_groups`, which drops a short tail; :meth:`Trainer.step` still divides
    #: by what it was handed, so a caller driving it directly (#94's resume, a test) gets a mean
    #: over whatever it passed rather than a step scaled by the size of the remainder.
    micro_batches: int
    #: The pre-clip gradient norm, or ``None`` when :attr:`Optimization.grad_clip
    #: <pretrain.config.Optimization>` is unset. §6 clips nothing by default, and computing the
    #: norm anyway would put a device sync in every step for a number nobody reads.
    grad_norm: float | None
    #: What the step cost in wall clock (#95, §7.5) — ``None`` when a caller drove
    #: :meth:`Trainer.step` directly, since a step timed outside a loop has no interval to be
    #: measured against. :func:`train` fills it for every step it takes.
    throughput: StepThroughput | None = None


class Scored(NamedTuple):
    """One scoring of one micro-batch — see :meth:`Trainer.score`.

    A :class:`~typing.NamedTuple` for the same reason :class:`~pretrain.loss.MaeLossOutput` is
    one: it survives `torch.compile`. The four travel together because they are four views of
    a single forward pass, and a caller that has one and re-derives another is the drift
    :meth:`Trainer.score` documents.
    """

    #: §4's objective: the scalar, the per-patch losses, and the weight each entered with.
    out: MaeLossOutput
    #: The patchified fp32 image the loss scored against.
    target: Tensor
    #: The per-pixel validity weights used, or ``None`` when §4's ocean weighting is ablated off.
    pixel_weights: Tensor | None
    #: The model's own output — the per-patch predictions and the mask that hid them. Carried
    #: for §10.2's grids (#98), which need the reconstruction *and* the mask that produced it
    #: beside the target and the weights, all from one forward. A renderer that ran its own
    #: forward would draw a different mask than the loss beside it was computed under, and the
    #: grid would then be captioned with a number describing a different masking of the tile.
    #: Required, not optional: it is a view of the same forward pass as the other three, and a
    #: ``None`` here would be an ``AttributeError`` at the one call site rather than a degraded
    #: render.
    model_out: MaeOutput


class Trainer:
    """Model, objective, optimizer and schedule, and the one method that steps them (§6, §7.1).

    Build it with :meth:`from_config`. It owns no data and no loop: a step is
    ``trainer.step(micro_batches)``, which makes the accumulation testable against a single
    large batch and lets #94 drive the same object from a resumed state.
    """

    def __init__(
        self,
        model: MaskedAutoencoder,
        loss_fn: MaeLoss,
        optimizer: torch.optim.Optimizer,
        schedule: LrSchedule,
        *,
        accum_iter: int,
        grad_clip: float | None,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
        forward_model: torch.nn.Module | None = None,
    ) -> None:
        self.model = model
        #: What a forward actually goes through: the model, or §7.3's `torch.compile` wrapper
        #: around it. Kept *beside* :attr:`model` rather than replacing it, because the wrapper
        #: is not a drop-in for every purpose — its ``state_dict`` keys carry an ``_orig_mod.``
        #: prefix, so a Trainer that swapped it in would hand #94 checkpoints that no eager run
        #: could load. Everything structural (parameters, ``patchify``, the state dict) reads
        #: :attr:`model`; only the forward reads this.
        self.forward_model = model if forward_model is None else forward_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.schedule = schedule
        self.accum_iter = accum_iter
        self.grad_clip = grad_clip
        self.device = device
        #: ``None`` means fp32 — §7.1's debugging escape hatch. Anything else is bf16; the
        #: config refuses fp16 outright, so there is no third case to handle here.
        self.autocast_dtype = autocast_dtype
        #: Steps taken. #94 restores it; :meth:`step` accepts an explicit step so a caller can
        #: drive the schedule without owning this counter.
        self.completed_steps = 0

    @classmethod
    def from_config(
        cls,
        cfg: RunConfig,
        *,
        steps_per_epoch: int,
        device: torch.device | str | None = None,
    ) -> Trainer:
        """Every piece of the step, off one config object (§8's warm start included).

        :func:`~pretrain.init.initialize_from_config` is called unconditionally and does nothing
        at ``init.mode = 'scratch'``: R4 and R5 run the same code path, which is what stops the
        continual arm from being a branch that only one run exercises.

        §7.3's compile is applied last, after the weights are loaded and the model is on its
        device — the graph is traced on the first forward, so anything that moves the module
        afterwards is a guard failure waiting to happen. The optimizer is built over the *eager*
        model, which costs nothing: `torch.compile` wraps a module, it does not copy its
        parameters.

        §7.2's attention path is **verified here rather than at the entrypoint**, so every way
        of building a trainer is covered by it — this loop, #94's resume, a test. The case it
        catches is not hypothetical: #90 only touches `attn.fused_attn` to turn it *off* for
        `attention='math'`, so a config asking for `sdpa` inherits whatever timm's own
        `use_fused_attn()` decided, and that reads an environment variable. A box with it
        disabled would train the recipe correctly and measure the throughput of a model nobody
        is going to ship.
        """
        device = resolve_device(device)
        model = MaskedAutoencoder.from_config(cfg)
        verify_attention(model, cfg)
        initialize_from_config(model, cfg)
        model.to(device)
        return cls(
            model,
            MaeLoss.from_config(cfg),
            build_optimizer(model, cfg),
            LrSchedule.from_config(cfg, steps_per_epoch=steps_per_epoch),
            accum_iter=cfg.optimization.accum_iter,
            grad_clip=cfg.optimization.grad_clip,
            device=device,
            autocast_dtype=torch.bfloat16 if cfg.runtime.precision == "bf16" else None,
            forward_model=compile_model(model, cfg),
        )

    def autocast(self):
        """§7.1's context: bf16 where the config asks for it, disabled for fp32 debugging."""
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype or torch.float32,
            enabled=self.autocast_dtype is not None,
        )

    def score(self, batch: dict, *, generator: torch.Generator | None = None) -> Scored:
        """Forward one micro-batch and score it: §4's objective in fp32, and what it scored.

        Only the model runs under autocast. The target is :meth:`patchify
        <pretrain.model.MaskedAutoencoder.patchify>` of the *same* fp32 image the model was
        shown — a reshape, so it stays fp32 whatever the surrounding policy is, which is what
        #92's guard 2 requires and its input check enforces.

        The pixel weights follow §4's ``ocean_weighted`` flag, read off the loss module rather
        than the config so the two cannot disagree: the loss refuses weights it was not
        configured for, and refuses to score without them when it was.

        Returns all three of the things one scoring produces, and the third is the one that is
        easy to leave out. #97's per-patch distribution (§4) needs the losses to bin, the
        **target** to say which of those patches were near-flat, and the **same pixel weights**
        this used — because under §4's ocean weighting "near-flat" means flat over a patch's
        *valid* pixels, and computed unweighted a half-ocean patch measures its variance across
        the land/ocean step and never looks flat at all. Deriving them a second time at the call
        site is the version of this that trains fine and silently misclassifies exactly the
        population §4's variance-floor rule exists to find.

        :meth:`step` takes ``.loss`` and drops the rest, which costs nothing — every one of
        those tensors exists either way.

        ``generator`` pins the mask draw. §10.1's val pass passes a seeded one so its loss is
        comparable epoch to epoch; a training step passes nothing and draws from the ambient RNG.
        """
        image = batch["image"].to(self.device, non_blocking=True)
        with self.autocast():
            out = self.forward_model(image, generator=generator)
        target = self.model.patchify(image)
        weights = None
        if self.loss_fn.ocean_weighted:
            land_mask = batch["land_mask"].to(self.device, non_blocking=True)
            weights = pixel_weights_from_land_mask(land_mask, self.model.patchify)
        return Scored(
            self.loss_fn(out.pred, target, out.mask, pixel_weights=weights),
            target,
            weights,
            out,
        )

    def micro_loss(self, batch: dict) -> torch.Tensor:
        """§4's scalar for one micro-batch — :meth:`score` with the diagnostics dropped."""
        return self.score(batch).out.loss

    def step(self, micro_batches: Sequence[dict], *, step: int | None = None) -> StepRecord:
        """One optimizer step over an effective batch delivered in pieces (§6).

        The LR is written **before** the step from :class:`~pretrain.schedule.LrSchedule`, which
        is why step 0 is taken at exactly 0 and nothing moves.

        The gradient is zeroed at the start rather than the end, so the gradients a step
        produced are still readable afterwards — which is what #97 needs to log a norm and what
        the clipping test inspects.
        """
        if not micro_batches:
            raise ValueError("a step needs at least one micro-batch")
        step = self.completed_steps if step is None else step
        lr = self.schedule.apply(self.optimizer, step)
        self.optimizer.zero_grad(set_to_none=True)

        n = len(micro_batches)
        total = torch.zeros((), device=self.device)
        for batch in micro_batches:
            loss = self.micro_loss(batch)
            # 1/N here, not on the accumulated total: the gradient is what accumulates, and
            # scaling it per micro-batch is what makes it the mean over the effective batch.
            (loss / n).backward()
            total += loss.detach()

        grad_norm = None
        if self.grad_clip is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            )
        self.optimizer.step()
        self.completed_steps = step + 1
        return StepRecord(
            step=step,
            epoch=self.schedule.epoch_at_step(step),
            lr=lr,
            loss=float(total) / n,  # one device sync per step, not per micro-batch
            micro_batches=n,
            grad_norm=grad_norm,
        )


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    """The device to train on: what was asked for, else CUDA, else CPU.

    MPS and other accelerators are reachable by name but never chosen by default. §7 is written
    for an Ada/Blackwell box, and silently landing on a device whose bf16 path and fused AdamW
    kernel differ would make a local run look like the rented one it is not.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- The loop -----------------------------------------------------------------


def train(
    cfg: RunConfig,
    *,
    steps: int,
    device: torch.device | str | None = None,
    on_step: Callable[[StepRecord], None] | None = None,
    on_start: Callable[[Trainer], None] | None = None,
    loader=None,
    clock: Callable[[], float] = time.perf_counter,
) -> list[StepRecord]:
    """Train for ``steps`` optimizer steps and return what each one did.

    ``steps`` rather than epochs, deliberately: this is the tracer bullet and R0's smoke run
    (#81) is measured in steps. The *schedule* is still the config's — 400 epochs of warmup and
    cosine — so a short run is the first N steps of the real curve rather than a compressed
    imitation of it. #94's :func:`train_resumable` owns running one to its end and surviving
    the box dying halfway, and it — not this — is what ``python -m pretrain.launch`` drives.
    What this is for is watching the loop work with nothing else attached.

    ``on_step`` receives each :class:`StepRecord` as it happens; that is the seam #97's logging
    and #94's checkpoint triggers hang off, and it is why nothing here prints. ``on_start``
    receives the built :class:`Trainer` before the first step, which is where a caller checks
    what the model will actually do — §7.2's attention path, §7.3's compile — while there is
    still time to stop.

    Every step is timed (§7.5): the loader is wrapped in a
    :class:`~pretrain.throughput.ThroughputMeter`, so each record carries samples/s, its step
    time, and the fraction of it spent blocked on the loader. Two ``perf_counter`` calls per
    micro-batch, which is nothing next to a forward pass — and the plan's #1 named risk is
    worth measuring on every run rather than on the runs someone remembered to instrument.
    """
    generator = seed_run(cfg)
    if loader is None:
        loader = build_loader(cfg, cfg.data.train_split, generator=generator)
    trainer = Trainer.from_config(
        cfg, steps_per_epoch=steps_per_epoch_for(loader, cfg), device=device
    )
    if on_start is not None:
        on_start(trainer)
    records: list[StepRecord] = []
    # Built here, immediately before the loop: its clock starts on construction, so anything
    # done between the two would be charged to the first step.
    meter = ThroughputMeter(loader, clock=clock)
    for group in micro_batch_groups(meter, trainer.accum_iter, steps=steps):
        record = trainer.step(group)
        record = replace(
            record,
            throughput=meter.mark(
                step=record.step,
                epoch=int(record.epoch),
                samples=sum(len(batch["image"]) for batch in group),
            ),
        )
        records.append(record)
        if on_step is not None:
            on_step(record)
    return records


def train_resumable(
    cfg: RunConfig,
    *,
    steps: int | None = None,
    epochs: int | None = None,
    policy: CheckpointPolicy | None = None,
    resume: bool | str | Path = False,
    device: torch.device | str | None = None,
    on_step: Callable[[StepRecord], None] | None = None,
    on_start: Callable[[Trainer], None] | None = None,
    on_epoch: Callable[[int, ThroughputSummary], None] | None = None,
) -> list[StepRecord]:
    """:func:`train`, but it survives the box going away (#94, plan §11).

    Same loop and the same :class:`Trainer`; the differences are all bookkeeping. The order
    comes from an :class:`EpochShuffleSampler` through an :class:`EpochCursor`, so a mid-epoch
    resume continues the interrupted shuffle. ``policy`` decides when to write and what to keep.
    ``resume`` picks what to continue from — ``False`` for a fresh run, ``True`` for "the newest
    checkpoint in the policy's directory, if there is one", or an explicit path.

    ``steps`` is the run's **total**, not an increment: a resumed run stops where the original
    would have. ``epochs`` says the same thing in the schedule's own unit and is converted
    through :func:`steps_per_epoch_for`, so it counts the steps the split actually yields.
    Asked for neither, this runs the config's whole schedule — which is what a real run does.

    Every step is timed, exactly as :func:`train` times one: the loader is wrapped in a
    :class:`~pretrain.throughput.ThroughputMeter` and each record carries samples/s, its step
    time and the loader-wait fraction. That is not decoration on this path either — §7.5's gate
    is read off these numbers and #97's ~10% loader alarm fires on them, and a resumable run
    that reported none of it would be the shipping path measuring nothing.

    ``on_epoch`` is the epoch boundary as a seam: the epoch index that has just ended, and its
    throughput summed over the steps in it. It is what #97's :class:`~pretrain.monitor.
    EpochMonitor` hangs off, and it fires **after the checkpoint for that boundary is written**.
    The order is deliberate and worth stating: an epoch's val pass, its per-patch histogram, its
    grids and its probe are minutes of work, and a box that vanishes during them should not take
    the epoch with it. It also keeps §11's bit-identity independent of what monitoring does —
    the RNG streams a checkpoint records are the training loop's, captured before anything else
    has had a chance to draw from them.

    The restore order is the part worth reading twice. Everything is rebuilt from the config
    first — model, optimizer, schedule, loader — and only then are the checkpoint's tensors and
    RNG states written over it. Restoring RNG *last* matters because building a loader and a
    model touches generators (:meth:`MaskedAutoencoder.from_config
    <pretrain.model.MaskedAutoencoder.from_config>` forks its own, but nothing guarantees the
    next thing to be added will), and a stream restored before something else advances it is a
    stream that was not restored at all.
    """
    if steps is not None and epochs is not None:
        raise ValueError(
            f"a run's length is steps or epochs, not both (got steps={steps}, epochs={epochs})"
        )
    generator = seed_run(cfg)
    loader = build_loader(cfg, cfg.data.train_split, generator=generator)
    steps_per_epoch = steps_per_epoch_for(loader, cfg)  # before any skip shortens the sampler
    trainer = Trainer.from_config(cfg, steps_per_epoch=steps_per_epoch, device=device)
    if steps is None:
        steps = (cfg.schedule.epochs if epochs is None else epochs) * steps_per_epoch

    state = RunState()
    fingerprint = fingerprint_for(cfg)
    path = resolve_resume(policy.directory, resume) if policy is not None else None
    if path is not None:
        checkpoint = load_checkpoint(path, map_location=trainer.device)
        # Refused before a step is taken, and before anything is loaded over the fresh model:
        # a run that resumed across a corpus change would release an encoder whose recorded
        # `dataset_version` names data it was only partly trained on (§11, §14).
        checkpoint.verify_against(fingerprint)
        if checkpoint.config_hash != cfg.config_hash:
            # Warned, not refused. §6/#87 lets R5 extend 400 -> 800 epochs by *rebuilding the
            # schedule* and resuming, and the plan calls that a config change rather than a new
            # run — so refusing here would block the one legitimate case. But an unnoticed knob
            # move is how a run reports one recipe and trains another, which is exactly what
            # `config_hash` exists to make visible.
            warnings.warn(
                f"resuming {path} under a different recipe: the checkpoint records config "
                f"{checkpoint.config_hash} and this run is {cfg.config_hash}. Intended for an "
                f"R5-style epoch extension (§6, #87); anything else means the resumed run is "
                f"not the run that was killed",
                stacklevel=2,
            )
        trainer.model.load_state_dict(checkpoint.model)
        trainer.optimizer.load_state_dict(checkpoint.optimizer)
        trainer.completed_steps = checkpoint.state.step
        state = checkpoint.state
        restore_rng(checkpoint.rng, generator=generator)

    if on_start is not None:
        on_start(trainer)

    # Built here, immediately before the loop, for the reason :func:`train` gives: its clock
    # starts on construction, so anything done between the two is charged to the first step.
    # It wraps the loader rather than sitting beside it, and the cursor drives it as the loader
    # (see :meth:`ThroughputMeter.__getattr__`), so the wait it measures is the one the
    # resumable order actually paid.
    meter = ThroughputMeter(loader)
    cursor = EpochCursor(
        meter, accum_iter=trainer.accum_iter, steps_per_epoch=steps_per_epoch, state=state
    )
    records: list[StepRecord] = []
    # Accumulated, never accrued: R1 is ~195,000 steps and a per-epoch list of all of them
    # would be a memory leak inside the thing measuring the run's health.
    epoch_counters = summarize([], epoch=state.epoch)
    groups = cursor.groups()  # endless; the step count below is what stops the run
    while trainer.completed_steps < steps:
        group = next(groups)
        record = trainer.step(group.micro_batches)
        counters = meter.mark(
            step=record.step,
            epoch=group.epoch,
            samples=sum(len(batch["image"]) for batch in group.micro_batches),
        )
        record = replace(record, throughput=counters)
        if counters.epoch != epoch_counters.epoch:
            epoch_counters = summarize([], epoch=counters.epoch)
        epoch_counters = epoch_counters.plus(counters)
        records.append(record)
        if on_step is not None:
            on_step(record)
        if policy is not None and policy.due(ends_epoch=group.ends_epoch):
            policy.save(
                config=cfg,
                fingerprint=fingerprint,
                state=cursor.state(trainer.completed_steps),
                model=trainer.model,
                optimizer=trainer.optimizer,
                rng=capture_rng(generator=generator),
                ends_epoch=group.ends_epoch,
            )
        # After the checkpoint, always — see the docstring. An epoch that ended because the run
        # ended still closes, so the last epoch of a run reports like every other one.
        if group.ends_epoch and on_epoch is not None:
            on_epoch(group.epoch, epoch_counters)
    return records


# --- The entrypoint -----------------------------------------------------------


def format_step(record: StepRecord, total: int) -> str:
    """One step, as a line. Placeholder-grade by design: #97 owns the real reporting."""
    clip = "" if record.grad_norm is None else f"  |g| {record.grad_norm:.3f}"
    return (
        f"step {record.step + 1}/{total}  epoch {record.epoch:7.3f}  "
        f"lr {record.lr:.3e}  loss {record.loss:.4f}{clip}"
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m pretrain.train --corpus … --steps N`` — the loop, runnable.

    Enough of an entrypoint to watch the tracer bullet fly and to smoke-test a box. The launch
    entrypoint a rented run actually invokes — compile, throughput counters, the full override
    surface — is #95's, and it wraps this.
    """
    p = add_config_arguments(
        argparse.ArgumentParser(
            prog="python -m pretrain.train",
            description="Train the MAE for N steps against a corpus (#93).",
        )
    )
    p.add_argument("--steps", type=int, required=True, help="optimizer steps to run")
    p.add_argument("--device", default=None, help="cuda | cpu | … (default: cuda if present)")
    p.add_argument("--log-every", type=int, default=10, help="steps between printed lines")
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="write §11 checkpoints here (per epoch and per --checkpoint-minutes)",
    )
    p.add_argument(
        "--checkpoint-minutes",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS / 60,
        help="wall-clock checkpoint interval; §11 asks for 30-60",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue from the newest checkpoint in --checkpoint-dir, if there is one",
    )
    args = p.parse_args(argv)

    cfg = config_from_args(args)
    print(format_header(cfg, fingerprint_for(cfg)))
    print()

    def report(record: StepRecord) -> None:
        if record.step % args.log_every == 0 or record.step + 1 == args.steps:
            print(format_step(record, args.steps), flush=True)

    if args.checkpoint_dir is None:
        if args.resume:
            p.error("--resume needs --checkpoint-dir to resume from")
        records = train(cfg, steps=args.steps, device=args.device, on_step=report)
    else:
        policy = CheckpointPolicy.from_config(
            cfg,
            directory=args.checkpoint_dir,
            interval_seconds=args.checkpoint_minutes * 60,
        )
        records = train_resumable(
            cfg,
            steps=args.steps,
            policy=policy,
            resume=args.resume,
            device=args.device,
            on_step=report,
        )
    if records:
        print(f"\n{len(records)} steps: loss {records[0].loss:.4f} -> {records[-1].loss:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
