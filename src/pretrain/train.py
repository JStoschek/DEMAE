"""The training step, and the loop around it: the tracer bullet that actually trains (#93).

Everything Phase 1 has built so far is a piece — a config (#89), a model (#90), an augmented
read path (#91), an objective (#92, #96), an optional warm start (#102). This is the file where
they are one thing: loader -> augment -> mask -> encode -> decode -> loss -> backward -> step,
running, with the loss going down.

It deliberately stops short of two neighbours. **Survival is #94's**: checkpoints, RNG state and
exact resume. **Speed is #95's**: `torch.compile`, throughput counters, and the launch
entrypoint a rented box invokes. What is here is the loop those two wrap, plus a
``python -m pretrain.train`` that runs it for N steps so the thing can be watched before either
lands.

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
- **Masking and augmentation** draw from the ambient torch RNG (at ``num_workers > 0`` the
  augmentation's entropy is the loader's ``base_seed + worker_id``, which is drawn from that same
  seeded generator — see :mod:`pretrain.augment`).

`test_the_same_seed_and_config_reproduce_the_loss_trajectory` pins the result on CPU. It is a
weaker claim on a GPU, where the kernels themselves are nondeterministic, and #94's kill-and-
resume test is the one that has to hold there.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from dataset.loader import build_dataloader

from .augment import make_augmented
from .config import RunConfig
from .data import SplitDataset
from .fingerprint import fingerprint_for
from .header import add_config_arguments, build_config, format_header
from .init import initialize_from_config
from .loss import MaeLoss, pixel_weights_from_land_mask
from .model import MaskedAutoencoder
from .optim import build_optimizer
from .schedule import LrSchedule

# --- The read path ------------------------------------------------------------


def seeded_generator(cfg: RunConfig) -> torch.Generator:
    """The DataLoader's generator, from :attr:`Runtime.seed <pretrain.config.Runtime>` (§6)."""
    generator = torch.Generator()
    generator.manual_seed(cfg.runtime.seed)
    return generator


def seed_run(cfg: RunConfig) -> torch.Generator:
    """Seed the run's RNG streams and hand back the DataLoader's generator (§6).

    The ambient seed is what the in-model mask draw (#90) and, at ``num_workers=0``, the
    dihedral augmentation (#91) read. Model init does *not* read it — it forks its own — so
    adding a decoder block cannot silently reorder the corpus.
    """
    torch.manual_seed(cfg.runtime.seed)
    return seeded_generator(cfg)


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
    """
    dataset = SplitDataset(
        cfg.data.corpus_dir,
        split,
        blocks_path=cfg.data.split_blocks_path,
        assignment_path=cfg.data.split_assignment_path,
        normalization=cfg.data.normalization,
    )
    if augment:
        dataset = make_augmented(dataset, cfg.data)
    kwargs: dict[str, Any] = {
        "batch_size": cfg.optimization.batch_size,
        "num_workers": cfg.data.num_workers,
        "shuffle": shuffle,
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


def micro_batch_groups(loader, accum_iter: int, *, steps: int) -> Iterator[list[dict]]:
    """Group the loader's batches into one list per optimizer step, for ``steps`` steps.

    Each pass over ``loader`` is an epoch, and re-iterating it is what re-shuffles: the sample
    order in epoch 2 is not epoch 1's. (Cycling one materialized epoch instead would train fine
    and quietly turn a 400-epoch run into 400 repetitions of one ordering.)

    **A short group at an epoch boundary is dropped**, for the same reason ``drop_last`` drops a
    short batch: an optimizer step is defined as *the effective batch*, and a step taken on half
    of one at the LR the whole batch earns is a step outside the recipe. At the shipped shape
    that costs at most one micro-batch per epoch, re-drawn by the next epoch's shuffle. It also
    keeps the count :func:`steps_per_epoch_for` gives the schedule exact.
    """
    if len(loader) // accum_iter == 0:
        raise ValueError(
            f"{len(loader)} batch(es) of {accum_iter} accumulated is no complete step — a split "
            "smaller than one effective batch, with drop_last on. Left alone this would spin "
            "forever without taking a step"
        )
    produced = 0
    while produced < steps:
        group: list[dict] = []
        for batch in loader:
            group.append(batch)
            if len(group) == accum_iter:
                yield group
                produced += 1
                group = []
                if produced == steps:
                    return


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
    ) -> None:
        self.model = model
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
        """
        device = resolve_device(device)
        model = MaskedAutoencoder.from_config(cfg)
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
        )

    def autocast(self):
        """§7.1's context: bf16 where the config asks for it, disabled for fp32 debugging."""
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype or torch.float32,
            enabled=self.autocast_dtype is not None,
        )

    def micro_loss(self, batch: dict) -> torch.Tensor:
        """Forward one micro-batch and score it: §4's scalar, in fp32.

        Only the model runs under autocast. The target is :meth:`patchify
        <pretrain.model.MaskedAutoencoder.patchify>` of the *same* fp32 image the model was
        shown — a reshape, so it stays fp32 whatever the surrounding policy is, which is what
        #92's guard 2 requires and its input check enforces.

        The pixel weights follow §4's ``ocean_weighted`` flag, read off the loss module rather
        than the config so the two cannot disagree: the loss refuses weights it was not
        configured for, and refuses to score without them when it was.
        """
        image = batch["image"].to(self.device, non_blocking=True)
        with self.autocast():
            out = self.model(image)
        target = self.model.patchify(image)
        weights = None
        if self.loss_fn.ocean_weighted:
            land_mask = batch["land_mask"].to(self.device, non_blocking=True)
            weights = pixel_weights_from_land_mask(land_mask, self.model.patchify)
        return self.loss_fn(out.pred, target, out.mask, pixel_weights=weights).loss

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
    loader=None,
) -> list[StepRecord]:
    """Train for ``steps`` optimizer steps and return what each one did.

    ``steps`` rather than epochs, deliberately: this is the tracer bullet and R0's smoke run
    (#81) is measured in steps. The *schedule* is still the config's — 400 epochs of warmup and
    cosine — so a short run is the first N steps of the real curve rather than a compressed
    imitation of it. #94 owns running one to its end and surviving the box dying halfway.

    ``on_step`` receives each :class:`StepRecord` as it happens; that is the seam #97's logging
    and #94's checkpoint triggers hang off, and it is why nothing here prints.
    """
    generator = seed_run(cfg)
    if loader is None:
        loader = build_loader(cfg, cfg.data.train_split, generator=generator)
    trainer = Trainer.from_config(
        cfg, steps_per_epoch=steps_per_epoch_for(loader, cfg), device=device
    )
    records: list[StepRecord] = []
    for group in micro_batch_groups(loader, trainer.accum_iter, steps=steps):
        record = trainer.step(group)
        records.append(record)
        if on_step is not None:
            on_step(record)
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
    args = p.parse_args(argv)

    cfg = build_config(
        config_path=args.config,
        vit_base=args.vit_base,
        corpus=args.corpus,
        assignment=args.assignment,
        blocks=args.blocks,
    )
    print(format_header(cfg, fingerprint_for(cfg)))
    print()

    def report(record: StepRecord) -> None:
        if record.step % args.log_every == 0 or record.step + 1 == args.steps:
            print(format_step(record, args.steps), flush=True)

    records = train(cfg, steps=args.steps, device=args.device, on_step=report)
    if records:
        print(f"\n{len(records)} steps: loss {records[0].loss:.4f} -> {records[-1].loss:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
