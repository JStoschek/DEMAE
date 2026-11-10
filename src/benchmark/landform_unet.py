"""B7's training pass: Task 1's A7 UNet on one channel, from scratch, under §4.5's schedule (#209).

Plan §4.3: *the same 4-down/4-up UNet spec as Task 1's A7 (~17M params), 1-channel input, from
scratch at every label fraction ... Batch 64 and 40 epochs, B2's.* The trunk is built by Task
1's builder, :func:`benchmark.unet.build`, with ``in_channels=1`` and ten class logits: A7's
depth, base width, GroupNorm and seeded init, and nothing about A7's output changed.

Everything after the trunk is :mod:`benchmark.landform_train`'s, so B7 is scored the way every
other Task 2 cell is: the cycling sampler, the ten val reads with the best kept, the joint
dihedral turn, weighted cross-entropy ignoring 255 under the committed class weights, and per-tile
confusion and boundary counts on test. The loop is its own rather than a branch of
:func:`benchmark.landform_train.train_linear` because the two differ in what they build and what
they check (a frozen encoder there, everything trainable here), not in the schedule.

## The input

A7's conditioning (:func:`benchmark.unet.condition`): each tile standardized by its own mean and
std over its land pixels, through :func:`benchmark.extract.tile_statistics`, the one code path
for those statistics. B7 has no void, so the raw tile is both the float image and the int16 the
statistics are taken from. A from-scratch trunk fed raw metres would train badly for reasons
that have nothing to do with the comparison, and §4.3 asks for a bar that is not crippled.

## Precision

fp32, as the linear probe trains (#197 review). A7 ran bf16 under Task 1's head recipe; the run
record says fp32 so a reader comparing B7's seconds per step against A7's sees why.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dataset.normalize import land_mask

from . import geomorphon_pass, train, unet
from . import config as task1_config
from .landform_config import LandformConfig
from .landform_train import (
    CLASSES,
    DIHEDRAL,
    CellTiles,
    Fitted,
    ValRead,
    _loader,
    _val_miou,
    cycling_batches,
    val_read_steps,
)


def trunk_parameters() -> int:
    """B7's trainable count: A7's spec on one channel with ten logits, derived, not built."""
    return unet.parameter_count(task1_config.Unet(), in_channels=1, classes=CLASSES)


def build(cfg: LandformConfig):
    """B7's trunk for this cell, seeded by the cell's seed, through Task 1's builder."""
    return unet.build(task1_config.Unet(), seed=cfg.seed, in_channels=cfg.unet.in_channels,
                      classes=CLASSES)


@dataclass
class ScratchForward:
    """The tile, standardized per tile over its land, through the trunk — train and eval."""

    module: Any
    device: str

    def logits(self, raw: np.ndarray):
        import torch

        raw = np.asarray(raw)
        image, _ = unet.condition(raw.astype(np.float32), raw, land_mask(raw))
        return self.module(torch.from_numpy(image).unsqueeze(1).to(self.device))


def train_unet(
    cfg: LandformConfig,
    *,
    lr: float,
    tiles: CellTiles,
    extractor=None,
    on_read: Callable[[ValRead], None] | None = None,
    steps: int | None = None,
    val_reads: int | None = None,
    checkpoint_dir: Path | str | None = None,
) -> Fitted:
    """Train B7 for one cell from scratch; return the best val read's trunk.

    The signature is :func:`benchmark.landform_train.train_linear`'s so #207's tuning pass can
    call either; ``extractor`` is accepted and unused, since B7 reads no encoder. ``steps``,
    ``val_reads`` and ``checkpoint_dir`` are the tuning pass's and nothing else's.
    """
    import torch

    from .evaluate import TileStore

    if cfg.arm != "B7" or cfg.head != "unet":
        raise ValueError(f"the from-scratch UNet pass is B7's alone; this cell is arm {cfg.arm}, "
                         f"head {cfg.head}")
    started = time.perf_counter()
    recipe = cfg.recipe
    device = train.resolve_device(cfg)
    ignore = cfg.geomorphon.ignore_label
    weights = torch.tensor(geomorphon_pass.committed_class_weights(cfg), dtype=torch.float32,
                           device=device)

    built = build(cfg)
    module = built.module.to(device)
    expected = trunk_parameters()
    train.assert_everything_trains(module, expected=expected)
    forward = ScratchForward(module, device)
    optimizer = torch.optim.AdamW(
        train.parameter_groups(module, weight_decay=recipe.weight_decay),
        lr=lr, weight_decay=recipe.weight_decay,
    )
    steps = cfg.steps if steps is None else steps
    if steps < 1:
        raise ValueError(f"a pass of {steps} steps trains nothing")
    schedule = train.Cosine(
        peak_lr=lr, total_steps=steps,
        warmup_steps=max(1, int(round(steps * cfg.schedule.warmup_frac))),
    )
    reads_at = set(val_read_steps(steps, cfg.schedule.val_reads if val_reads is None
                                  else val_reads))
    index = TileStore.index_for(cfg.paths.corpus_dir, [*tiles.train, *tiles.val])

    rng = np.random.default_rng(cfg.seed)
    sampler = cycling_batches(tiles.train, recipe.batch_size, rng)
    items = [(t, int(rng.integers(DIHEDRAL))) for _ in range(steps) for t in next(sampler)]

    directory = (Path(cfg.runtime.checkpoint_dir) / f"{cfg.arm}-{cfg.rung}-seed{cfg.seed}"
                 if checkpoint_dir is None else Path(checkpoint_dir))
    directory.mkdir(parents=True, exist_ok=True)
    best_path = directory / "best.pt"
    mark = train._clock(device)
    reads: list[ValRead] = []
    best: ValRead | None = None
    best_state = None
    train_seconds = 0.0
    step = 0
    module.train()
    tick = mark()
    for raw, label in _loader(cfg, items, index, recipe.batch_size):
        schedule.apply(optimizer, step)
        # An all-ignore batch takes no optimizer step but keeps its place (#206's rule).
        if bool((label != ignore).any()):
            optimizer.zero_grad(set_to_none=True)
            target = torch.from_numpy(label.astype(np.int64)).to(device)
            loss = torch.nn.functional.cross_entropy(
                forward.logits(raw), target, weight=weights, ignore_index=ignore
            )
            loss.backward()
            optimizer.step()
        step += 1
        train_seconds += mark() - tick
        if step in reads_at:
            read_started = mark()
            score = _val_miou(forward, cfg, tiles.val, index, recipe.batch_size, ignore)
            read = ValRead(step=step, miou=score, seconds=mark() - read_started)
            reads.append(read)
            if best is None or read.miou > best.miou:
                best = read
                best_state = copy.deepcopy(module.state_dict())
                torch.save({"step": step, "val_miou": score, "state_dict": best_state},
                           best_path)
            if on_read is not None:
                on_read(read)
            module.train()
        tick = mark()
    if step != steps or best is None:
        raise train.TrainError(f"the pass took {step} of {steps} steps and read val "
                               f"{len(reads)} times")
    module.load_state_dict(best_state)
    module.eval()
    train.assert_everything_trains(module, expected=expected)
    return Fitted(
        module=module,
        head=cfg.head,
        parameters=built.parameters,
        width=0,
        steps=steps,
        lr=lr,
        reads=reads,
        best=best,
        seconds_per_step=train_seconds / steps,
        seconds=time.perf_counter() - started,
        device=device,
        checkpoint=str(best_path),
        train_tiles=len(tiles.train),
        encoder=None,
        extractor=None,
        precision=None,
        forward=forward,
    )
