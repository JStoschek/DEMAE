"""Task 2's training pass for a frozen-encoder probe, and the test scoring after it (#206).

Plan §4.5 and §4.1, in the order this module runs them:

* **Length is steps**, :attr:`benchmark.landform_config.LandformConfig.steps`: ``epochs x
  ceil(max(n_train, floor) / batch)``. 5% linear warmup then cosine to zero over those steps
  (:class:`benchmark.train.Cosine`, Task 1's curve).
* **The sampler cycles** its rung's tiles, reshuffled each pass, and a batch may span two passes
  (:func:`cycling_batches`), so the linear probe's batch of 256 stands at the 1% rung's 200 tiles.
* **Val is read ten times**, at each tenth of the steps with the last step always included
  (:func:`val_read_steps`), over every val tile; the read with the best val mIoU is the
  checkpoint scored on test. No early stopping.
* **Joint dihedral augmentation**: each training sample is turned by one of the eight
  symmetries of the square, the tile and its label raster together (:func:`dihedral`), before the
  encoder sees it. Nothing is cached, which is what makes that possible (#184).
* **Valid pixels are land** (story 41): the mask handed to the extractor is the tile's
  non-ocean pixels, from which ADR-0012's extractor recomputes the per-tile statistics.
* **Weighted cross-entropy ignoring 255**, under the committed class weights
  (:func:`benchmark.geomorphon_pass.committed_class_weights`).

Task 1's machinery is reused, not copied: the device (:func:`benchmark.train.resolve_device`),
the accelerator-synced clock, the frozen-encoder checksum and its assertion, AdamW's two
parameter groups, the cosine, and the loader worker count (:func:`benchmark.train.workers_for`).

Scoring writes, per test tile, the 10x10 confusion counts over non-ignore pixels (truth rows,
prediction columns) and the four boundary-F1 counts at 2 px (:func:`boundary_counts`). Every
printed metric is a function of those counts.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from dataset.normalize import land_mask
from pretrain.artifact import read_tile_strata

from . import arms, artifacts, geomorphon_pass, ladders, landform_arms, probe, train
from .landform_config import GEOMORPHON_CLASSES, LandformConfig

#: The eight symmetries of the square: ``k % 4`` quarter turns, then a left-right flip if
#: ``k >= 4``.
DIHEDRAL: Final = 8

#: §6's boundary-F1 tolerance, in pixels.
BOUNDARY_TOLERANCE_PX: Final = 2

CLASSES: Final = len(GEOMORPHON_CLASSES)


# --- §4.5's schedule pieces ---------------------------------------------------


def cycling_batches(ids: Sequence[str], batch_size: int, rng: np.random.Generator
                    ) -> Iterator[list[str]]:
    """Batches of exactly ``batch_size`` ids forever: each pass a fresh permutation, and a batch
    spanning two passes when the pass runs out mid-batch (§4.5's 1% sampler)."""
    if not ids:
        raise ValueError("the sampler has no tiles to cycle")
    pending: list[str] = []
    while True:
        while len(pending) < batch_size:
            pending.extend(ids[i] for i in rng.permutation(len(ids)))
        yield pending[:batch_size]
        pending = pending[batch_size:]


def val_read_steps(steps: int, reads: int) -> tuple[int, ...]:
    """After which optimizer steps (1-based) val is read: each tenth, the last step included."""
    return tuple(sorted({max(1, math.ceil(k * steps / reads)) for k in range(1, reads + 1)}))


def dihedral(a: np.ndarray, k: int) -> np.ndarray:
    """Symmetry ``k`` of :data:`DIHEDRAL` applied to a 2-D array."""
    out = np.rot90(a, k % 4)
    if k >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def augment(raw: np.ndarray, label: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The tile and its label raster under the same symmetry — one transform, two arrays."""
    return dihedral(raw, k), dihedral(label, k)


# --- The counts ---------------------------------------------------------------


def confusion(truth: np.ndarray, pred: np.ndarray, *, ignore: int,
              classes: int = CLASSES) -> np.ndarray:
    """``classes x classes`` counts over the pixels whose truth is not ``ignore``, flattened
    row-major: entry ``t * classes + p`` is truth ``t`` predicted ``p``."""
    keep = truth != ignore
    t = truth[keep].astype(np.int64)
    p = pred[keep].astype(np.int64)
    return np.bincount(t * classes + p, minlength=classes * classes)


def _boundary(label: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pixels with a 4-neighbour of another class, both pixels valid."""
    edge = np.zeros(label.shape, dtype=bool)
    for axis in (0, 1):
        a = np.swapaxes(label, 0, axis)
        v = np.swapaxes(valid, 0, axis)
        diff = (a[1:] != a[:-1]) & v[1:] & v[:-1]
        e = np.swapaxes(edge, 0, axis)
        e[1:] |= diff
        e[:-1] |= diff
    return edge


def boundary_counts(truth: np.ndarray, pred: np.ndarray, *, ignore: int,
                    tolerance: int = BOUNDARY_TOLERANCE_PX) -> dict[str, int]:
    """Boundary-F1's four counts at ``tolerance`` px over the non-ignore pixels.

    ``pred_boundary`` predicted boundary pixels, ``pred_matched`` those within ``tolerance`` of a
    true boundary pixel (precision's numerator), and the same two for truth (recall's). Pooled
    over tiles they give precision ``sum(pred_matched) / sum(pred_boundary)`` and recall
    likewise, so boundary-F1 pools like the confusion counts do.
    """
    from scipy.ndimage import distance_transform_edt

    valid = truth != ignore
    tb = _boundary(truth, valid)
    pb = _boundary(pred, valid)

    def near(edges: np.ndarray) -> np.ndarray:
        if not edges.any():
            return np.zeros(edges.shape, dtype=bool)
        return distance_transform_edt(~edges) <= tolerance

    return {
        "bf1_pred_boundary": int(pb.sum()),
        "bf1_pred_matched": int((pb & near(tb)).sum()),
        "bf1_true_boundary": int(tb.sum()),
        "bf1_true_matched": int((tb & near(pb)).sum()),
    }


def iou_per_class(matrix: np.ndarray, classes: int = CLASSES) -> np.ndarray:
    """Per-class IoU from summed counts; ``nan`` for a class with no truth and no prediction."""
    m = np.asarray(matrix, dtype=np.float64).reshape(classes, classes)
    tp = np.diag(m)
    union = m.sum(0) + m.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, tp / union, np.nan)


def miou(matrix: np.ndarray) -> float:
    iou = iou_per_class(matrix)
    return float(np.nanmean(iou)) if np.isfinite(iou).any() else 0.0


def class_collapse(matrix: np.ndarray) -> list[str]:
    """L10: classes present in test truth whose pooled test IoU is 0 — flagged, never excluded."""
    m = np.asarray(matrix).reshape(CLASSES, CLASSES)
    iou = iou_per_class(m)
    return [GEOMORPHON_CLASSES[c] for c in range(CLASSES) if m[c].sum() > 0 and iou[c] == 0]


# --- The cell's tiles ---------------------------------------------------------


@dataclass(frozen=True)
class CellTiles:
    """The three tile sets one cell reads: its rung of its ladder, all val, all test."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    #: ``(tile_id, stratum_id)`` in the committed test list's order.
    test: tuple[tuple[str, int], ...]

    @classmethod
    def read(cls, cfg: LandformConfig) -> CellTiles:
        directory = Path(cfg.paths.artifacts_dir)
        walk = read_tile_strata(directory / ladders.ladder_name(cfg.ladder), what="ladder")
        train_ids = tuple(t for t, _ in walk[: cfg.train_tiles])
        if len(train_ids) != cfg.train_tiles:
            raise ValueError(
                f"ladder {cfg.ladder} holds {len(walk)} tiles and rung {cfg.rung} is the first "
                f"{cfg.train_tiles:,}"
            )
        task1 = Path(cfg.paths.task1_artifacts_dir)
        val = tuple(t for t, _ in read_tile_strata(task1 / artifacts.VAL_TILES_NAME,
                                                   what="val list"))
        test = tuple(read_tile_strata(task1 / artifacts.TEST_TILES_NAME, what="test list"))
        overlap = set(train_ids) & (set(val) | {t for t, _ in test})
        if overlap:
            raise ValueError(
                f"{len(overlap)} rung tiles are also val or test tiles (e.g. "
                f"{sorted(overlap)[:3]}): the splits are block-disjoint by construction"
            )
        return cls(train=train_ids, val=val, test=test)

    @property
    def every(self) -> list[str]:
        return [*self.train, *self.val, *(t for t, _ in self.test)]


# --- The pass -----------------------------------------------------------------


class _Samples:
    """``(tile_id, k)`` -> the turned tile and label: a ``torch`` dataset, paths and ids only."""

    def __init__(self, cfg: LandformConfig, items: Sequence[tuple[str, int]], index: dict):
        self.cfg = cfg
        self.items = list(items)
        self.index = index
        self._store = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        from .evaluate import TileStore

        if self._store is None:
            self._store = TileStore(self.cfg.paths.corpus_dir, index=self.index)
        tile_id, k = self.items[i]
        raw = self._store.read(tile_id).raw_i16
        label = geomorphon_pass.read_label(self.cfg, tile_id)
        return augment(raw, label, k)


def _collate(items):
    raws, labels = zip(*items)
    return np.stack(raws), np.stack(labels)


def _loader(cfg: LandformConfig, items: Sequence[tuple[str, int]], index: dict,
            batch_size: int):
    """Batches of ``(raw_i16, label)`` in order, on Task 1's worker rule."""
    dataset = _Samples(cfg, items, index)
    workers = train.workers_for(len(items), batch_size, cfg.runtime.loader_workers)
    if workers <= 0:
        for start in range(0, len(dataset), batch_size):
            yield _collate([dataset[i] for i in range(start, min(start + batch_size,
                                                                 len(dataset)))])
        return
    from torch.utils.data import DataLoader

    yield from DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                          collate_fn=_collate)


@dataclass(frozen=True)
class ValRead:
    step: int
    miou: float
    seconds: float

    def line(self) -> str:
        return f"step {self.step:>6}  val mIoU {self.miou:.4f}  {self.seconds:6.1f}s"


@dataclass
class Fitted:
    """What one probe pass produced, and what the run record quotes about it."""

    module: Any
    parameters: int
    width: int
    steps: int
    lr: float
    reads: list[ValRead]
    best: ValRead
    seconds_per_step: float
    seconds: float
    device: str
    checkpoint: str
    train_tiles: int
    encoder: str
    extractor: Any
    precision: dict | None = None

    def to_dict(self) -> dict:
        return {
            "kind": "linear",
            "lr": self.lr,
            "steps": self.steps,
            "train_tiles": self.train_tiles,
            "val_reads": [{"step": r.step, "miou": round(r.miou, 6)} for r in self.reads],
            "best_step": self.best.step,
            "best_val_miou": round(self.best.miou, 6),
            "checkpoint": self.checkpoint,
            "device": self.device,
            "precision_head": "fp32",
            "seconds": round(self.seconds, 3),
        }


class _Forward:
    """The frozen encoder through the contract, then the probe — one place, train and eval."""

    def __init__(self, extractor, encoder: str, module, device: str):
        self.extractor = extractor
        self.encoder = encoder
        self.module = module
        self.device = device

    def logits(self, raw: np.ndarray):
        features = self.extractor.extract(raw, land_mask(raw), encoder_name=self.encoder)
        return self.module(features.output.to(self.device))


def _val_miou(forward: _Forward, cfg: LandformConfig, val: Sequence[str], index: dict,
              batch_size: int, ignore: int) -> float:
    import torch

    total = np.zeros(CLASSES * CLASSES, dtype=np.int64)
    forward.module.eval()
    with torch.no_grad():
        for raw, label in _loader(cfg, [(t, 0) for t in val], index, batch_size):
            pred = forward.logits(raw).argmax(1).cpu().numpy()
            for i in range(len(pred)):
                total += confusion(label[i], pred[i], ignore=ignore)
    return miou(total)


def train_linear(
    cfg: LandformConfig,
    *,
    lr: float,
    tiles: CellTiles,
    extractor=None,
    on_read: Callable[[ValRead], None] | None = None,
) -> Fitted:
    """Train §4.1's linear probe for one cell over a frozen encoder; return the best val read."""
    import torch

    from . import extract
    from .evaluate import TileStore

    started = time.perf_counter()
    arm = landform_arms.get(cfg.arm)
    encoder = arms.ENCODERS[arm.encoder]
    recipe = cfg.recipe
    device = train.resolve_device(cfg)
    ignore = cfg.geomorphon.ignore_label
    weights = torch.tensor(geomorphon_pass.committed_class_weights(cfg), dtype=torch.float32,
                           device=device)

    if extractor is None:
        spec = Path(cfg.paths.task1_artifacts_dir) / extract.SPEC_FILENAME
        extractor = extract.Extractor(extract.load_spec(spec), cfg, device=device)
    model = extractor.encoder(encoder_name=encoder.name)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    before = train.encoder_checksums(model)

    built = probe.build(encoder.width, seed=cfg.seed)
    module = built.module.to(device)
    forward = _Forward(extractor, encoder.name, module, device)
    optimizer = torch.optim.AdamW(
        train.parameter_groups(module, weight_decay=recipe.weight_decay),
        lr=lr, weight_decay=recipe.weight_decay,
    )
    steps = cfg.steps
    schedule = train.Cosine(
        peak_lr=lr, total_steps=steps,
        warmup_steps=max(1, int(round(steps * cfg.schedule.warmup_frac))),
    )
    reads_at = set(val_read_steps(steps, cfg.schedule.val_reads))
    index = TileStore.index_for(cfg.paths.corpus_dir, [*tiles.train, *tiles.val])

    rng = np.random.default_rng(cfg.seed)
    sampler = cycling_batches(tiles.train, recipe.batch_size, rng)
    items = [(t, int(rng.integers(DIHEDRAL))) for _ in range(steps) for t in next(sampler)]

    directory = Path(cfg.runtime.checkpoint_dir) / f"{cfg.arm}-{cfg.rung}-seed{cfg.seed}"
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
        optimizer.zero_grad(set_to_none=True)
        target = torch.from_numpy(label.astype(np.int64)).to(device)
        logits = forward.logits(raw)
        if bool((target != ignore).any()):
            loss = torch.nn.functional.cross_entropy(
                logits, target, weight=weights, ignore_index=ignore
            )
        else:  # a batch of all-ignored pixels teaches nothing, and CE's mean over it is NaN
            loss = logits.sum() * 0.0
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
    train.assert_encoder_frozen(model, name=encoder.name, before=before)
    return Fitted(
        module=module,
        parameters=built.parameters,
        width=built.width,
        steps=steps,
        lr=lr,
        reads=reads,
        best=best,
        seconds_per_step=train_seconds / steps,
        seconds=time.perf_counter() - started,
        device=device,
        checkpoint=str(best_path),
        train_tiles=len(tiles.train),
        encoder=encoder.name,
        extractor=extractor,
        precision=extractor.precision_agreement(encoder_name=encoder.name),
    )


@dataclass(frozen=True)
class ScoredTile:
    tile_id: str
    stratum: int
    band: int
    confusion: np.ndarray
    boundary: dict[str, int]


def score_test(cfg: LandformConfig, fitted: Fitted, tiles: CellTiles) -> list[ScoredTile]:
    """The best val read's probe over every test tile: per-tile counts, nothing averaged."""
    import torch

    from .evaluate import TileStore
    from .voids import Bands

    ids = [t for t, _ in tiles.test]
    stratum = dict(tiles.test)
    index = TileStore.index_for(cfg.paths.corpus_dir, ids)
    bands = Bands.from_manifest(cfg.paths.corpus_dir, split="test")
    forward = _Forward(fitted.extractor, fitted.encoder, fitted.module, fitted.device)
    ignore = cfg.geomorphon.ignore_label
    out: list[ScoredTile] = []
    fitted.module.eval()
    order = iter(ids)
    with torch.no_grad():
        for raw, label in _loader(cfg, [(t, 0) for t in ids], index, cfg.recipe.batch_size):
            pred = forward.logits(raw).argmax(1).cpu().numpy()
            for i in range(len(pred)):
                tile_id = next(order)
                out.append(ScoredTile(
                    tile_id=tile_id,
                    stratum=int(stratum[tile_id]),
                    band=bands.of(index[tile_id][3]),
                    confusion=confusion(label[i], pred[i], ignore=ignore),
                    boundary=boundary_counts(label[i], pred[i], ignore=ignore),
                ))
    return out
