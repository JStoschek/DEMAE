"""Task 2's training pass for a frozen-encoder head, and the test scoring after it (#206, #208).

One pass for both heads over a frozen encoder: §4.1's linear probe, which reads the extractor's
``output``, and §4.2's UperNet, which reads its ``intermediates`` at ``upernet.layers``
(:func:`build_head` is the one place a head is chosen, built and fed). Plan §4.5, §4.1 and §4.2,
in the order this module runs them:

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

from . import arms, artifacts, geomorphon_pass, ladders, landform_arms, probe, train, upernet
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
    #: The head that trained, as the cell names it.
    head: str
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
    #: What the head takes off the extractor's features (:attr:`Head.takes`).
    takes: Callable[[Any], Any] = None  # type: ignore[assignment]  # set by train_head
    #: How a tile becomes logits, for a pass with no frozen encoder behind it (B7, #209; B3's
    #: fine-tuned one, #210). ``None`` means the frozen encoder through the contract, then the
    #: probe.
    forward: Any = None
    #: Encoder weights this pass trained: 0 for a frozen encoder, R5's less its sin-cos for B3.
    encoder_trained: int = 0
    #: The precision the head trained in, as declared and as this box could honour it
    #: (:func:`benchmark.train.precision_on`). The probes train in fp32; B3 declares bf16.
    precision_declared: str = "fp32"
    precision_used: str = "fp32"

    def to_dict(self) -> dict:
        return {
            "kind": "finetune" if self.encoder_trained else self.head,
            "lr": self.lr,
            "steps": self.steps,
            "train_tiles": self.train_tiles,
            "val_reads": [{"step": r.step, "miou": round(r.miou, 6)} for r in self.reads],
            "best_step": self.best.step,
            "best_val_miou": round(self.best.miou, 6),
            "checkpoint": self.checkpoint,
            "device": self.device,
            "precision_head": self.precision_used,
            "precision_declared": self.precision_declared,
            "precision_used": self.precision_used,
            "seconds": round(self.seconds, 3),
        }


@dataclass(frozen=True)
class Head:
    """A built head and what it reads off the extractor's :class:`~benchmark.extract.Features`."""

    module: Any
    parameters: int
    width: int
    #: ``Features -> the module's input``: ``output`` for the linear probe, the four chosen
    #: ``intermediates`` for UperNet.
    takes: Callable[[Any], Any]


def build_head(cfg: LandformConfig, encoder: arms.Encoder) -> Head:
    """The cell's head over ``encoder``, initialized under the cell's seed.

    The linear probe reads `output`, so it is built at the encoder's `output` width — 5 for B8's
    pre-FSQ latent (#211) — and UperNet reads the blocks, at the block width.
    """
    width = encoder.width
    if cfg.head == "linear":
        built = probe.build(encoder.output_width, seed=cfg.seed)
        return Head(built.module, built.parameters, built.width, lambda f: f.output)
    if cfg.head == "upernet":
        recipe = cfg.upernet
        built = upernet.build(width, channels=recipe.channels, pool_scales=recipe.pool_scales,
                              seed=cfg.seed)
        layers = tuple(recipe.layers)
        return Head(built.module, built.parameters, built.width,
                    lambda f: upernet.read(f, layers))
    raise NotImplementedError(
        f"the {cfg.head} head reads no frozen encoder; B7's UNet trains in "
        f"benchmark.landform_unet"
    )


# --- §4.3's B3: the encoder unfrozen under layer-wise decay -------------------

#: Encoder tensors B3 does not train. R5's ``pos_embed`` is fixed sin-cos, verified as such when
#: the extractor loads it (ADR-0011): it was never a learned weight, so the fine-tune does not
#: learn one.
FIXED_ENCODER_TENSORS: Final = ("pos_embed",)

#: Encoder tensors that skip weight decay beside every 1-d tensor (MAE's no-decay list).
NO_DECAY_TENSORS: Final = ("cls_token",)


def unfreeze(model) -> None:
    """Every encoder weight trains except :data:`FIXED_ENCODER_TENSORS`."""
    for name, param in model.named_parameters():
        param.requires_grad_(name not in FIXED_ENCODER_TENSORS)


def finetune_groups(encoder, head, recipe, *, weight_decay: float) -> list[dict]:
    """§4.3's AdamW groups: the encoder at ``recipe.lr`` under ``recipe.layer_decay`` per layer,
    the head at ``recipe.head_lr``.

    The ladder is MAE's, the one Task 1's fine-tune climbs
    (:func:`benchmark.finetune._encoder_layer`): the patch embedding and the CLS token are layer
    0, block *i* is layer *i + 1*, the final norm is the top, and layer *l* trains at
    ``lr * layer_decay ** (top - l)``. Weights decay at ``weight_decay``; 1-d tensors and the CLS
    token do not. Each group carries ``part``, ``layer`` and ``lr``, which torch ignores beside
    its own keys and a test reads.
    """
    from .finetune import _encoder_layer

    top = len(encoder.blocks) + 1
    groups: dict[tuple, dict] = {}

    def add(part: str, layer: int, lr: float, name: str, param) -> None:
        if not param.requires_grad or name in FIXED_ENCODER_TENSORS:
            return
        decays = param.ndim > 1 and name not in NO_DECAY_TENSORS
        group = groups.setdefault((part, layer, decays), {
            "part": part, "layer": layer, "lr": lr,
            "weight_decay": weight_decay if decays else 0.0, "params": [],
        })
        group["params"].append(param)

    for name, param in encoder.named_parameters():
        layer = _encoder_layer(name, top=top)
        add("encoder", layer, recipe.lr * recipe.layer_decay ** (top - layer), name, param)
    for name, param in head.named_parameters():
        add("head", top + 1, recipe.head_lr, name, param)
    return [groups[key] for key in sorted(groups)]


def optimizer_for(cfg: LandformConfig, groups, *, lr: float):
    """The head recipe's optimizer over ``groups``: AdamW, the one every Task 2 recipe names.

    The recipe's ``optimizer`` is read rather than stated beside a hard-coded AdamW, so a line
    frozen under another name is refused here instead of trained under AdamW and recorded as
    whatever it said (#197 review).
    """
    import torch

    recipe = cfg.recipe
    if recipe.optimizer != "adamw":
        raise ValueError(
            f"{cfg.head}.optimizer {recipe.optimizer!r}: Task 2's heads train under AdamW "
            f"(Task 1's §4.1 optimizer block), and no other optimizer is built"
        )
    return torch.optim.AdamW(groups, lr=lr, weight_decay=recipe.weight_decay)


def _to(value, device: str):
    return [v.to(device) for v in value] if isinstance(value, (list, tuple)) else value.to(device)


class _Forward:
    """The frozen encoder through the contract, then the head — one place, train and eval."""

    def __init__(self, extractor, encoder: str, module, device: str,
                 takes: Callable[[Any], Any], *, precision: str | None = None):
        self.extractor = extractor
        self.encoder = encoder
        self.module = module
        self.device = device
        self.takes = takes
        #: B3's declared precision: set, the encoder runs with a graph whenever gradients are on
        #: and the head under :func:`benchmark.train._autocast`. ``None`` is a frozen encoder.
        self.precision = precision

    def logits(self, raw: np.ndarray):
        if self.precision is None:
            features = self.extractor.extract(raw, land_mask(raw), encoder_name=self.encoder)
            return self.module(_to(self.takes(features), self.device))
        import torch

        with train._autocast(self.device, self.precision):
            features = self.extractor.extract(raw, land_mask(raw), encoder_name=self.encoder,
                                              grad=torch.is_grad_enabled())
            return self.module(_to(self.takes(features), self.device)).float()


def _val_miou(forward: _Forward, cfg: LandformConfig, val: Sequence[str], index: dict,
              batch_size: int, ignore: int) -> float:
    import torch

    total = np.zeros(CLASSES * CLASSES, dtype=np.int64)
    forward.module.eval()
    if getattr(forward, "precision", None) is not None:  # B3's encoder reads val in eval
        forward.extractor.encoder(encoder_name=forward.encoder).eval()
    with torch.no_grad():
        for raw, label in _loader(cfg, [(t, 0) for t in val], index, batch_size):
            pred = forward.logits(raw).argmax(1).cpu().numpy()
            for i in range(len(pred)):
                total += confusion(label[i], pred[i], ignore=ignore)
    return miou(total)


def train_head(
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
    """Train the cell's head (§4.1's linear probe or §4.2's UperNet) over a frozen encoder;
    return the best val read.

    B3 (#210) runs the same pass with its encoder unfrozen: :func:`finetune_groups`' per-layer
    lrs scaled by the one cosine, a graph through the extractor while training, bf16 where the
    box has it, and the best read's encoder restored with its head, since test scores through
    that same model.

    ``steps``, ``val_reads`` and ``checkpoint_dir`` are #207's tuning pass and nothing else: a
    point trains a fifth of the cell's steps (warmup and cosine over that shorter run, as Task 1's
    10 of 50 epochs did), reads val twice, and checkpoints into its own directory so five points
    do not overwrite one ``best.pt``. They are parameters rather than config knobs because a
    config with fewer epochs would land in the frozen line's recipe. A scored cell passes none.
    """
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
    fine_tunes = cfg.fine_tunes
    built = build_head(cfg, encoder)
    module = built.module.to(device)
    if fine_tunes:
        # B3 (§4.3): the encoder trains with the head, each layer at its own lr. The frozen
        # checksum is not taken, because moving the encoder is this arm's whole point.
        unfreeze(model)
        before = None
        groups = finetune_groups(model, module, cfg.finetune, weight_decay=recipe.weight_decay)
        forward = _Forward(extractor, encoder.name, module, device, built.takes,
                           precision=cfg.finetune.precision)
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        before = train.encoder_checksums(model)
        groups = train.parameter_groups(module, weight_decay=recipe.weight_decay)
        forward = _Forward(extractor, encoder.name, module, device, built.takes)
    optimizer = optimizer_for(cfg, groups, lr=lr)
    steps = cfg.steps if steps is None else steps
    if steps < 1:
        raise ValueError(f"a pass of {steps} steps trains nothing")
    # The fine-tune's groups each peak at their own lr, so its curve is a factor (peak 1) that
    # scales every group's; a frozen head's one lr is the curve itself, as Task 1's was.
    schedule = train.Cosine(
        peak_lr=1.0 if fine_tunes else lr, total_steps=steps,
        warmup_steps=max(1, int(round(steps * cfg.schedule.warmup_frac))),
    )
    peaks = [group["lr"] for group in optimizer.param_groups]
    reads_at = set(val_read_steps(steps, cfg.schedule.val_reads if val_reads is None
                                  else val_reads))
    index = TileStore.index_for(cfg.paths.corpus_dir, [*tiles.train, *tiles.val])

    rng = np.random.default_rng(cfg.seed)
    sampler = cycling_batches(tiles.train, recipe.batch_size, rng)
    items = [(t, int(rng.integers(DIHEDRAL))) for _ in range(steps) for t in next(sampler)]

    directory = (Path(cfg.runtime.checkpoint_dir)
                 / f"{cfg.arm}-{cfg.head}-{cfg.rung}-seed{cfg.seed}"
                 if checkpoint_dir is None else Path(checkpoint_dir))
    directory.mkdir(parents=True, exist_ok=True)
    best_path = directory / "best.pt"
    mark = train._clock(device)
    reads: list[ValRead] = []
    best: ValRead | None = None
    best_state = None
    train_seconds = 0.0
    step = 0

    def training_mode() -> None:
        module.train()
        if fine_tunes:
            model.train()

    def snapshot() -> dict:
        state = {"state_dict": copy.deepcopy(module.state_dict())}
        if fine_tunes:
            state["encoder_state_dict"] = copy.deepcopy(model.state_dict())
        return state

    training_mode()
    tick = mark()
    for raw, label in _loader(cfg, items, index, recipe.batch_size):
        if fine_tunes:
            factor = schedule.at(step)
            for group, peak in zip(optimizer.param_groups, peaks):
                group["lr"] = peak * factor
        else:
            schedule.apply(optimizer, step)
        # A batch of all-ignored pixels teaches nothing (and CE's mean over it is NaN). It takes
        # no optimizer step, since AdamW would still decay the weights and advance its moments,
        # but it keeps its step: the count and the cosine stay those of the cell's schedule.
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
                best_state = snapshot()
                torch.save({"step": step, "val_miou": score, **best_state}, best_path)
            if on_read is not None:
                on_read(read)
            training_mode()
        tick = mark()
    if step != steps or best is None:
        raise train.TrainError(f"the pass took {step} of {steps} steps and read val "
                               f"{len(reads)} times")
    module.load_state_dict(best_state["state_dict"])
    module.eval()
    if fine_tunes:
        # The best read's encoder is the one test scores: the extractor holds this very model.
        model.load_state_dict(best_state["encoder_state_dict"])
        model.eval()
    else:
        train.assert_encoder_frozen(model, name=encoder.name, before=before)
    return Fitted(
        module=module,
        head=cfg.head,
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
        takes=built.takes,
        forward=forward if fine_tunes else None,
        encoder_trained=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fine_tunes else 0,
        precision_declared=cfg.finetune.precision if fine_tunes else "fp32",
        precision_used=train.precision_on(device, cfg.finetune.precision) if fine_tunes
        else "fp32",
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
    forward = fitted.forward or _Forward(fitted.extractor, fitted.encoder, fitted.module,
                                         fitted.device, fitted.takes)
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
                    band=bands.of(index[tile_id].slope_mean),
                    confusion=confusion(label[i], pred[i], ignore=ignore),
                    boundary=boundary_counts(label[i], pred[i], ignore=ignore),
                ))
    return out
