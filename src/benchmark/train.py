"""§4.1's head-training pass: the bank in, one trained head out (#159).

`run --arm A4-R5` is *train, then score*, and this is the first half of it. It assembles the one
thing three arms share — the A1 fill, the frozen encoder's tokens over it, and §4.1's head — and
trains that head against the committed train mask bank.

## Not a way to produce a result

Documented as such, exactly as ``pretrain.train`` is (ADR-0009, and the spec names this module
by name for it). ``python -m benchmark.launch`` is the only command that scores anything. There
is no ``main`` here and there should not be one: a second entry point that trained a head would
be a seam nobody is on both sides of, which is the failure Phase 1 shipped four times.

## The composition, and why training and eval share it

:func:`compose` builds the model's input — A1 fill, valid mask, void — and :func:`residual_of`
runs the encoder and the head over it. **Both are called from the eval path too**
(:func:`predict`, which `benchmark.evaluate` reaches through
:func:`benchmark.evaluate._head_fill`). That is not code tidiness. A head trained against one
composition and scored against a slightly different one — a valid mask that included the void, a
fill computed at a different variant, a normalization applied twice — converges normally and
prints a table exactly as wide as a good one. There is one composition because there is one way
for that to be true.

## Three things about the composition worth stating

**The encoder is fed the A1 fill, rounded to int16 metres.** §2.4's ``a1_fill_plus_mask``
convention hands the encoder a *complete* DEM, and ADR-0012 §1's door takes raw int16 metres and
applies its own declared diet. The corpus is int16 metres, so rounding the fill to the corpus's
own quantization is not a loss of information the encoder could have used — it is putting the
fill into the units the encoder was pretrained on. The **head's** residual is added to the
float32 fill (:func:`benchmark.head.apply_residual`), so the rounding never reaches a score.

**The valid mask excludes the void.** ADR-0012 §1's ``land_and_not_void``: per-tile
standardization reads only measured ground, so an interpolated patch cannot set the statistics
the encoder normalizes by. That matters most on exactly the instances that matter — a 158 px
void is 40% of a tile.

**The truth is the raw tile.** Not the fill, not a standardized tensor. §4.1's loss is masked L1
**in metres** and §5 scores in metres, so metres is what the loss sees — and what
:mod:`benchmark.head` emits, since the encoder's diet is applied on the way in and never
inverted.

## Val, and the split guard

§4.1 tunes and early-stops on **val**. :class:`TrainingTiles` is the only tile source this module
opens, it is built from §3.1's head-train and val lists alone, and it **refuses a tile id that is
in neither** — so "test tiles are not reachable from the tuning path" is a property of the object
rather than of the discipline of whoever wrote the loop. :data:`TRAINING_ARTIFACTS` is the other
half: the committed files this module is allowed to read, with the test list and the eval mask
list conspicuously absent.

Val masks are **drawn**, not banked, and that is not a hole in "training reads the bank rather
than drawing masks". §3.2 commits one mask per *test* tile and §3.4 banks K per *head-train*
tile; no artifact covers val, because nothing that a val number gates is printed. What val gets
instead is :func:`val_instances`' deal — one void per val tile, §3.2's six geometries dealt
round-robin over the sorted list, at the same seed the eval generator would have given that tile
— so every arm, every seed and every re-run early-stops against the identical set.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Sequence

import numpy as np

from dataset.metrics import land_mask
from pretrain.artifact import read_tile_strata

from . import arms, extract, fill, head, masks
from .artifacts import TILE_LIST_NAMES, VAL_TILES_NAME
from .config import BenchmarkConfig

#: The committed artifacts a training pass may open. The test list and the eval mask list are
#: **not** here, and that absence is #159's fourth acceptance criterion: tuning reads val only.
#: A frozenset rather than a comment, so :func:`assert_reads_val_only` can check a caller's
#: intent against it and a test can read the roster without running a pass.
TRAINING_ARTIFACTS: Final = frozenset(
    {*TILE_LIST_NAMES.values(), VAL_TILES_NAME, "train_mask_bank.parquet"}
)

#: Per-epoch checkpoint filename, and the one the best val epoch is copied to. §7: "heads
#: checkpoint per epoch like everything else in this repo and that is the whole of it."
EPOCH_CHECKPOINT: Final = "epoch{epoch:03d}.pt"
BEST_CHECKPOINT: Final = "best.pt"


class TrainError(RuntimeError):
    """A training pass this box cannot run, or a composition that has come apart."""


# --- The box ------------------------------------------------------------------


def resolve_device(cfg: BenchmarkConfig) -> str:
    """``runtime.device``, with ``auto`` resolved against what this box actually has.

    CUDA, then MPS, then CPU. MPS is in the list rather than being a laptop convenience: this
    project's own note is that the laptop has it, so "needs a rented GPU" is usually false and a
    non-CPU bug reproduces here deterministically. A named device is taken literally — a box with
    CUDA can still be asked for ``cpu``, which is what a reproduction run wants.
    """
    named = cfg.runtime.device
    if named != "auto":
        return named
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _clock(device: str):
    """A wall clock that waits for the accelerator before reading itself.

    Without the synchronize, every phase timer measures how fast Python *queued* work rather than
    how long the device took to do it — which on a GPU attributes almost everything to whichever
    call happens to block first, and is how a profile lies.
    """
    import torch

    if device == "cuda":
        def mark() -> float:
            torch.cuda.synchronize()
            return time.perf_counter()

        return mark
    if device == "mps":
        def mark() -> float:
            torch.mps.synchronize()
            return time.perf_counter()

        return mark
    return time.perf_counter


def precision_on(device: str, precision: str) -> str:
    """What a box will actually compute in, given what the recipe declared.

    §4.1 declares bf16. CPU bf16 autocast is real but slow, and the box that runs this suite has
    no accelerator at all; a precision the recipe declares must not become the reason the tests
    cannot run. So the declared precision is honoured on CUDA and MPS and dropped to fp32 on CPU
    — and **said**, here, so :meth:`Trained.to_dict` can record both. A run record naming only
    what was asked for would be claiming a number was produced under a precision it was not.
    """
    return "fp32" if device == "cpu" else precision


def _autocast(device: str, precision: str):
    """§4.1's bf16 where the box has it, and a no-op context where :func:`precision_on` says not."""
    import contextlib

    import torch

    if precision_on(device, precision) == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16)


def assert_encoder_frozen(model, *, name: str, before: dict[str, Any] | None = None) -> None:
    """Eval mode, no gradient, no parameter moved — asserted rather than assumed.

    #160's criterion and A6's too: a frozen arm whose encoder quietly trained would produce the
    best number in the table and no column would say why. ``before`` is a checksum per parameter
    taken at the start of the pass; passing it is what turns "no gradient is attached" into "no
    weight moved", which are different claims and only the second one is the one C1 rests on.
    """
    import torch

    if model.training:
        raise TrainError(
            f"the {name} encoder is in training mode at the end of a frozen-encoder pass. "
            f"§4.1 trains the head; the encoder's dropout and norm statistics must not move"
        )
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise TrainError(
            f"the {name} encoder has {len(trainable)} parameter(s) with `requires_grad` set "
            f"({trainable[:3]}). A4/A5/A6 are frozen-encoder arms (§1)"
        )
    if before is None:
        return
    with torch.no_grad():
        moved = [
            n
            for n, p in model.named_parameters()
            if n in before and not torch.equal(p.detach().cpu(), before[n])
        ]
    if moved:
        raise TrainError(
            f"the {name} encoder moved {len(moved)} parameter tensor(s) during a frozen pass "
            f"({moved[:3]}). C1 is a claim about a representation held still"
        )


def encoder_checksums(model) -> dict[str, Any]:
    """A copy of every encoder parameter, for :func:`assert_encoder_frozen` to compare against."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


# --- §3.1's two draws, and the guard ------------------------------------------


@dataclass(frozen=True)
class TrainingTiles:
    """The head-train and val draws — **the only tile ids a training pass can reach**.

    The guard is :meth:`allow`, and it is a refusal rather than a filter on purpose: a pass that
    silently dropped a test tile would train on fewer tiles than it reported, which is a quieter
    bug than the one it is guarding against. What it is defending is #159's acceptance criterion
    — *a test shows test tiles are not reachable from the tuning path* — and the way to make that
    testable was to give the tuning path one door and put the check in it.
    """

    #: §3.1's head-train draw, at the size this cell trains on (the nested subsets are prefixes
    #: of one committed draw, so ``1k`` is the first 1,000 rows of the same file family).
    head_train: tuple[str, ...]
    #: §3.1's val draw. What §4.1 tunes and early-stops on, and nothing else.
    val: tuple[str, ...]
    #: The committed files this object was built from, by name — quoted in the run record so a
    #: reader can see that the test list is not among them.
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.head_train) & set(self.val)
        if overlap:
            raise TrainError(
                f"{len(overlap)} tile(s) are in both §3.1's head-train and val draws "
                f"({sorted(overlap)[:3]}). The splits are block-disjoint by construction "
                f"(ADR-0005), so this is a `freeze` against a different corpus"
            )

    @classmethod
    def read(cls, cfg: BenchmarkConfig) -> TrainingTiles:
        """Both draws off the committed lists, at this cell's size. Opens no test list."""
        directory = Path(cfg.paths.artifacts_dir)
        head_train_name = TILE_LIST_NAMES[cfg.size]
        head_train = [
            tile_id
            for tile_id, _ in read_tile_strata(
                directory / head_train_name, what="head-train list"
            )
        ]
        val = [
            tile_id
            for tile_id, _ in read_tile_strata(directory / VAL_TILES_NAME, what="val list")
        ]
        if not head_train or not val:
            raise TrainError(
                f"§3.1's draws are empty in {directory} ({len(head_train)} head-train, "
                f"{len(val)} val). `python -m benchmark.launch freeze` writes both"
            )
        return cls(
            head_train=tuple(head_train),
            val=tuple(val),
            sources=(head_train_name, VAL_TILES_NAME),
        )

    @property
    def reachable(self) -> frozenset[str]:
        return frozenset(self.head_train) | frozenset(self.val)

    def allow(self, tile_id: str) -> str:
        """``tile_id`` if a training pass may read it; a refusal naming the rule if not."""
        if tile_id in self.reachable:
            return tile_id
        raise TrainError(
            f"{tile_id} is in neither §3.1's head-train draw nor its val draw, and those are the "
            f"only two a training or tuning pass may read (#159). §4.1's hyperparameters are "
            f"tuned on val alone precisely so that no thumb lands on the scale, and a test tile "
            f"reaching a training loop is how that stops being true without anything printing "
            f"differently"
        )


def assert_reads_val_only(names: Iterable[str]) -> None:
    """Refuse a training pass that names a committed artifact outside :data:`TRAINING_ARTIFACTS`."""
    stray = sorted(set(names) - TRAINING_ARTIFACTS)
    if stray:
        raise TrainError(
            f"a training pass named {stray}, which is not among the committed inputs §4.1 lets "
            f"one read ({sorted(TRAINING_ARTIFACTS)}). The test list and the eval mask list are "
            f"deliberately absent: tuning reads val only (#159)"
        )


# --- §3.4's bank, and val's deal ----------------------------------------------


@dataclass(frozen=True)
class BankedVoids:
    """The committed train mask bank, restricted to this cell's tiles and indexed for the loop.

    ``by_tile[tile_id][epoch]`` is the RLE of the void that tile trains against in that epoch —
    §3.4's K draws per head-train tile, read rather than redrawn, which is #159's last acceptance
    criterion and §3.4's whole determinism contract.

    Held in memory, and it is worth saying how much: the full bank is 1,000,000 rows and §7
    prices the RLE payload at ~740 MB uncompressed. A cell at ``--size 1k`` holds a twentieth of
    that, because the label-efficiency rungs are nested prefixes of one draw. The alternative —
    re-reading the parquet per epoch — costs a full 104 MB decode fifty times over, because the
    bank is ordered tile-then-epoch and epoch *e*'s rows are one in every fifty throughout the
    file. So this is the cheap side of a measured trade, not an oversight.
    """

    by_tile: dict[str, list[str]]
    epochs: int

    def rle(self, tile_id: str, epoch: int) -> str:
        """The void ``tile_id`` trains against in ``epoch``, cycling if the schedule outruns K."""
        draws = self.by_tile[tile_id]
        return draws[epoch % len(draws)]


def read_bank(cfg: BenchmarkConfig, tile_ids: Sequence[str]) -> BankedVoids:
    """§3.4's bank for these tiles, off the committed parquet.

    The md5 guard is **not** here — it is `run`'s, ahead of everything, so that a moved bank is
    refused before an encoder loads rather than after (:func:`benchmark.masks.assert_bank_matches`).
    What this refuses is narrower and structural: a bank that does not carry every tile this cell
    trains on, which is what a bank frozen against a different head-train draw looks like.
    """
    import pyarrow.parquet as pq

    path = Path(cfg.paths.mask_bank_path)
    wanted = set(tile_ids)
    table = pq.read_table(path, columns=["tile_id", "epoch", "mask_rle"])
    column = {name: table.column(name) for name in ("tile_id", "epoch", "mask_rle")}
    ids = column["tile_id"].to_pylist()
    epochs = column["epoch"].to_pylist()
    rles = column["mask_rle"].to_pylist()

    by_tile: dict[str, list[tuple[int, str]]] = {}
    for tile_id, epoch, rle in zip(ids, epochs, rles):
        if tile_id in wanted:
            by_tile.setdefault(tile_id, []).append((int(epoch), rle))

    absent = sorted(wanted - set(by_tile))
    if absent:
        raise TrainError(
            f"§3.4's bank at {path} carries no draws for {len(absent):,} of this cell's "
            f"{len(wanted):,} head-train tiles ({absent[:3]}). The bank is K draws per "
            f"head-train tile and the label-efficiency rungs are nested prefixes of one draw "
            f"(§3.1), so a tile with no rows is a bank frozen against a different draw"
        )
    ordered = {
        tile_id: [rle for _, rle in sorted(draws)] for tile_id, draws in by_tile.items()
    }
    return BankedVoids(
        by_tile=ordered, epochs=min(len(draws) for draws in ordered.values())
    )


@dataclass(frozen=True)
class ValInstance:
    """One val tile and the single void it is early-stopped against."""

    tile_id: str
    family: str
    bucket: str
    mask_rle: str


def val_instances(cfg: BenchmarkConfig, val_tiles: Sequence[str]) -> list[ValInstance]:
    """One void per val tile, §3.2's six geometries dealt round-robin over the sorted list.

    **One** rather than six, which is the whole difference between a val pass that costs a
    fiftieth of an eval pass and one that costs six times a fiftieth fifty times over. The deal
    is what buys back what the single draw gives up: every geometry is equally represented across
    the val set, so a head that has learned to fill large blobs and nothing else is visible in
    the val RMSE rather than averaged into it.

    The seed is :func:`benchmark.masks.seed_for` — the *eval* generator's — so a val tile's void
    is the one §3.2 would have given it. Val and test are block-disjoint splits (ADR-0005), so
    reusing the seed function reaches no test tile; what it buys is that val is not a third mask
    convention with its own generator to keep in step.
    """
    cells = [
        (family, bucket)
        for family in cfg.masks.synthetic_families
        for bucket, _ in cfg.masks.buckets
    ]
    out = []
    for index, tile_id in enumerate(sorted(val_tiles)):
        family, bucket = cells[index % len(cells)]
        drawn = masks.draw(tile_id, family, bucket, geometry=cfg.masks)
        out.append(
            ValInstance(
                tile_id=tile_id, family=family, bucket=bucket, mask_rle=drawn.mask_rle
            )
        )
    return out


# --- The composition ----------------------------------------------------------


@dataclass(frozen=True)
class Composed:
    """One batch, as both the training loop and the eval pass build it. See the module docstring.

    Every array is a batch: ``(N, 224, 224)``, with the elevation arrays in **metres**.
    """

    #: The corpus tile as the shard holds it — the loss's truth and the fill's input.
    raw_i16: np.ndarray
    #: A1's fill over the void, float32 metres. What the head's residual is added to (§2.4).
    a1_fill: np.ndarray
    #: The same fill at the corpus's own quantization — ADR-0012 §1's door. See the docstring.
    a1_i16: np.ndarray
    #: ``land ∧ ¬void`` — ADR-0012 §1's ``land_and_not_void``.
    valid: np.ndarray
    #: The pixels this arm is asked to predict.
    void: np.ndarray


#: :class:`Composed`'s five arrays, in the order the class declares them. Read off the dataclass
#: so the collate function cannot fall out of step with it.
COMPOSED_FIELDS: Final = ("raw_i16", "a1_fill", "a1_i16", "valid", "void")


def compose_one(raw_i16: np.ndarray, void: np.ndarray, cfg) -> dict[str, np.ndarray]:
    """One tile's composition — the expensive half, and the half a loader worker does.

    §7 measures head training as **4.1x loader-bound**, and this function is why: the A1 fill of
    one banked void is milliseconds of CPU against a frozen forward that is microseconds of GPU.
    So it is written to be callable from a worker process — it takes arrays and a recipe, holds
    no handle, and returns plain numpy.
    """
    raw = np.asarray(raw_i16, dtype=np.int16)
    mask = np.asarray(void, dtype=bool)
    filled = fill.fill(raw, mask, recipe=cfg.fill)
    rounded = np.rint(filled).astype(np.int16)
    valid = land_mask(rounded) & ~mask
    if not valid.any():
        raise TrainError(
            "this tile has no valid pixel — valid = land ∧ ¬void (ADR-0012 §1), so an all-ocean "
            "tile, or a void covering every land pixel, has no per-tile statistic to normalize "
            f"by. §3.2 caps a void at {cfg.masks.max_area_frac:.0%} of a tile, so this is a tile "
            "the draw should not have reached"
        )
    return {
        "raw_i16": raw,
        "a1_fill": filled,
        "a1_i16": rounded,
        "valid": valid,
        "void": mask,
    }


def collate(items: Sequence[dict[str, np.ndarray]]) -> Composed:
    """Stack per-tile compositions into one batch. A named function so it survives pickling."""
    return Composed(
        **{name: np.stack([item[name] for item in items]) for name in COMPOSED_FIELDS}
    )


def compose(raw_i16: Sequence[np.ndarray], voids: Sequence[np.ndarray], cfg) -> Composed:
    """A1's fill and ADR-0012's two masks, for a batch of tiles. No torch, no encoder.

    ``cfg`` is a :class:`benchmark.config.BenchmarkConfig` or anything carrying its ``fill`` and
    ``masks`` groups; the fill recipe is the whole of what this reads, because the composition is
    §2.4's convention and not a training detail.
    """
    return collate(
        [compose_one(tile, mask, cfg) for tile, mask in zip(raw_i16, voids)]
    )


class Examples:
    """``(tile_id, mask_rle)`` pairs, composed on demand — a ``torch.utils.data.Dataset``.

    Holds **paths and strings only**, never an open handle: each worker builds its own
    :class:`benchmark.evaluate.TileStore` the first time it is asked for a tile, which is what
    makes this picklable under ``spawn`` (macOS's default) as well as ``fork``.

    The split guard is :meth:`TrainingTiles.allow`, called here rather than in the loop above:
    it has to survive the process boundary or it guards only the main process's copy, and "test
    tiles are not reachable from the tuning path" is a claim about the pass rather than about
    where the pass happens to run. :class:`TrainingTiles` is a frozen dataclass of tuples, so it
    crosses that boundary intact.
    """

    def __init__(
        self,
        cfg,
        items: Sequence[tuple[str, str]],
        tiles: TrainingTiles,
        index: dict | None = None,
    ) -> None:
        self.cfg = cfg
        self.items = list(items)
        self.tiles = tiles
        #: :meth:`benchmark.evaluate.TileStore.index_for` for the tiles this pass can reach,
        #: built **once by the caller** and inherited by every worker. A worker that built its
        #: own read the whole 499,934-row manifest to do it — 1.24 s each, per epoch — which at
        #: 64 workers over 50 epochs is 66 minutes of a cell spent re-deriving one table, and is
        #: why a 192-core box ran slower than a 16-core one (#159).
        self.index = index
        self._store = None

    def __len__(self) -> int:
        return len(self.items)

    def store(self):
        if self._store is None:
            from .evaluate import TileStore

            self._store = TileStore(self.cfg.paths.corpus_dir, index=self.index)
        return self._store

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        tile_id, mask_rle = self.items[index]
        return compose_one(
            self.store().read(self.tiles.allow(tile_id)).raw_i16,
            masks.decode_rle(mask_rle),
            self.cfg,
        )


def batches(
    cfg,
    items: Sequence[tuple[str, str]],
    tiles: TrainingTiles,
    *,
    batch_size: int,
    index: dict | None = None,
):
    """Yield :class:`Composed` batches over ``items``, in order, on ``runtime.loader_workers``.

    Zero workers is a plain loop rather than a one-worker :class:`~torch.utils.data.DataLoader`,
    because the suite runs sharded under ``pytest-xdist`` and a fixture-sized pass that forked a
    worker per shard would spend more time on process setup than on the four tiles it composes.
    """
    dataset = Examples(cfg, items, tiles, index)
    workers = cfg.runtime.loader_workers
    if workers <= 0:
        for start in range(0, len(dataset), batch_size):
            yield collate(
                [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            )
        return

    from torch.utils.data import DataLoader

    yield from DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate,
    )


@dataclass(frozen=True)
class Stack:
    """What produces a residual: a frozen encoder, a head, the arm naming them, and a device.

    The four travel together everywhere — the training step, the val pass, the eval pass and
    :class:`Trained` — so they are one object rather than four keyword arguments threaded through
    five call sites. The clump is also the thing that must not be reassembled differently in two
    places: a head scored against a second load of its own encoder, or against a different arm's
    normalization, converges and prints normally.
    """

    #: :class:`benchmark.extract.Extractor` — the contract, holding the loaded encoder.
    extractor: Any
    #: §4.1's head (:func:`benchmark.head.build`).
    module: Any
    #: The registered arm, which is what selects the encoder and its declared diet.
    arm: str
    device: str


def residual_of(batch: Composed, stack: Stack, *, precision: str):
    """The frozen encoder over the A1 fill, then §4.1's head — one ``(N, 1, 224, 224)`` residual.

    The encoder runs under :meth:`benchmark.extract.Extractor.extract`'s own ``no_grad``, so
    nothing here has to remember to detach: the tokens arrive as a leaf and the only parameters
    on the graph are the head's. That is why the frozen half is reached through the contract
    rather than by calling the model — the contract is where the diet, the layout and the
    absence of a gradient are all one declared thing.
    """
    import torch

    features = stack.extractor.extract(batch.a1_i16, batch.valid, arm=stack.arm)
    void = torch.from_numpy(batch.void.astype(np.float32)).unsqueeze(1).to(stack.device)
    tokens = features.output.to(stack.device)
    with _autocast(stack.device, precision):
        residual = stack.module(tokens, void)
    return residual.float(), void


def predict_composed(
    batch: Composed, stack: Stack, *, precision: str
) -> list[np.ndarray]:
    """§2.4's fill for one already-composed batch: A1's fill plus the head's masked residual."""
    import torch

    stack.module.eval()
    with torch.no_grad():
        residual, _ = residual_of(batch, stack, precision=precision)
    values = residual.detach().cpu().numpy()
    return [
        head.apply_residual(batch.a1_fill[i], values[i, 0], batch.void[i])
        for i in range(batch.raw_i16.shape[0])
    ]


def predict(
    raw_i16: Sequence[np.ndarray],
    voids: Sequence[np.ndarray],
    cfg: BenchmarkConfig,
    stack: Stack,
) -> list[np.ndarray]:
    """§2.4's fill for a trained arm: the A1 fill plus the head's residual, inside the void.

    The eval path's entry into this module, reached through
    :func:`benchmark.evaluate._head_fill`. Returns one float32 tile per input, in metres, ready for
    :func:`benchmark.metrics.score` — and, when the head's output convolution is still zero,
    returns arrays **equal** to A1's (:func:`benchmark.head.apply_residual`).
    """
    return predict_composed(
        compose(raw_i16, voids, cfg), stack, precision=cfg.head.precision
    )


# --- §4.1's schedule ----------------------------------------------------------


@dataclass(frozen=True)
class Cosine:
    """§4.1's curve: 5% linear warmup, then cosine to 0, measured in optimizer steps.

    :class:`pretrain.schedule.LrSchedule`'s sibling rather than an import of it, because that one
    is written in *epochs* against §6's 400-epoch pretraining recipe and this one is written in
    steps against §4.1's 50. Same two formulas; the shared thing is the shape, and a shared class
    parameterized over both would have been a third thing to keep in step with two plans.
    """

    peak_lr: float
    total_steps: int
    warmup_steps: int

    @classmethod
    def of(cls, recipe, *, steps_per_epoch: int, epochs: int) -> Cosine:
        total = max(1, steps_per_epoch * epochs)
        return cls(
            peak_lr=recipe.lr,
            total_steps=total,
            warmup_steps=max(1, int(round(total * recipe.warmup_frac))),
        )

    def at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return self.peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def apply(self, optimizer, step: int) -> float:
        lr = self.at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr


def parameter_groups(module, *, weight_decay: float) -> list[dict]:
    """AdamW's two groups: weights decay, biases and norm affines do not.

    :func:`pretrain.optim.parameter_groups`' rule at this scale. Not an import of it: that one
    reads :class:`pretrain.config.RunConfig` and reconciles timm's own no-decay list against a
    structural one, and this head has four module types and no timm in it. What is shared is the
    rule, which is stated here rather than inherited so that a reader of §4.1's optimizer block
    can check it against one file.
    """
    decay, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 else decay).append(param)
    return [
        {"name": "decay", "params": decay, "weight_decay": weight_decay},
        {"name": "no_decay", "params": no_decay, "weight_decay": 0.0},
    ]


# --- The pass -----------------------------------------------------------------


@dataclass(frozen=True)
class Epoch:
    """One epoch's line: what it cost, what it learned, and the val number it is judged on."""

    epoch: int
    loss: float
    elevation: float
    slope: float
    lr: float
    val_rmse: float
    seconds: float
    #: Wall seconds per phase — ``load`` (waiting on the dataloader), ``forward`` (the frozen
    #: encoder plus the head), ``backward`` (loss, gradients, optimizer step) and ``val``.
    #:
    #: Here because an epoch time on its own cannot say *what* is slow, and this benchmark spent
    #: three wrong diagnoses finding that out (#159): the tile index, then `normalize`, then the
    #: unused intermediates, each argued from the code and each refuted by a measurement. A run
    #: that reports where its own seconds went is cheaper than a fourth argument.
    phases: dict = field(default_factory=dict)

    def line(self) -> str:
        return (
            f"epoch {self.epoch:>3}  loss {self.loss:9.4f}  l1 {self.elevation:8.4f} m  "
            f"slope {self.slope:7.4f}  lr {self.lr:.2e}  val_rmse {self.val_rmse:8.4f} m  "
            f"{self.seconds:6.1f}s"
            + (
                "  [" + " ".join(f"{k} {v:.1f}s" for k, v in self.phases.items()) + "]"
                if self.phases
                else ""
            )
        )


def save_checkpoint(module, line: Epoch, path: Path) -> Path:
    """One epoch's state, with the two numbers that say which epoch it is.

    §7: *heads checkpoint per epoch like everything else in this repo and that is the whole of
    it.* Written by one function rather than by two identical ``torch.save`` calls, because the
    per-epoch file and the best-so-far file are the same payload under two names and a payload
    that drifted between them would make the kept head unidentifiable.
    """
    import torch

    torch.save(
        {
            "epoch": line.epoch,
            "val_rmse": line.val_rmse,
            "state_dict": module.state_dict(),
        },
        path,
    )
    return path


@dataclass(frozen=True)
class Trained:
    """What one training pass produced — the head, and everything a run record quotes about it."""

    module: Any
    width: int
    parameters: int
    #: The epoch whose val RMSE was lowest. Its weights are what ``module`` holds.
    best_epoch: int
    best_val_rmse: float
    history: tuple[Epoch, ...] = ()
    seconds: float = 0.0
    checkpoint: str | None = None
    tiles: int = 0
    sources: tuple[str, ...] = ()
    #: The encoder, head, arm and device this was trained as. Held so that :meth:`predict` is
    #: the *same* composition the val pass used, over the *same* loaded encoder — the eval pass
    #: rebuilding either would be a second chance to get them different.
    stack: Stack | None = None
    #: §4.1's declared precision, and the one this box could honour. See :func:`_autocast`.
    precision_declared: str = "bf16"
    precision_used: str = "bf16"

    @property
    def arm(self) -> str:
        """Which registered arm these weights are for.

        Read off :attr:`stack` rather than stored beside it, so there is one answer:
        :func:`benchmark.evaluate._head_fill` refuses a head trained for another arm, and a copy
        of the name that could disagree with the encoder the stack actually holds would make that
        refusal check the wrong thing.
        """
        return "" if self.stack is None else self.stack.arm

    def predict(
        self, raw_i16: Sequence[np.ndarray], voids: Sequence[np.ndarray], cfg
    ) -> list[np.ndarray]:
        """:func:`predict`, bound to this trained head — what the ``a1_fill_plus_mask`` factory
        closes over when `run` hands it this object (:func:`benchmark.evaluate._head_fill`)."""
        return predict(raw_i16, voids, cfg, self.stack)

    def to_dict(self) -> dict:
        """The training half of a run record: what was trained, on what, and how it went."""
        return {
            "parameters": self.parameters,
            "width": self.width,
            "epochs_run": len(self.history),
            "best_epoch": self.best_epoch,
            "best_val_rmse": round(float(self.best_val_rmse), 6),
            "final_val_rmse": (
                round(float(self.history[-1].val_rmse), 6) if self.history else None
            ),
            "train_tiles": self.tiles,
            "seconds": round(float(self.seconds), 3),
            "checkpoint": self.checkpoint,
            "read": list(self.sources),
            "device": None if self.stack is None else self.stack.device,
            # What the recipe asked for, and what the box actually ran — two fields, because
            # `_autocast` skips bf16 on CPU and a record naming only the declared precision would
            # be claiming a number was produced under one it was not.
            "precision_declared": self.precision_declared,
            "precision_used": self.precision_used,
        }


def val_rmse(
    instances: Sequence[ValInstance],
    cfg: BenchmarkConfig,
    stack: Stack,
    *,
    tiles: TrainingTiles,
    index: dict | None = None,
) -> float:
    """Pooled RMSE in metres over every val void — the number §4.1 early-stops on.

    **Pooled over pixels, not averaged over instances.** A per-instance mean would weight a 16 px
    void the same as a 158 px one, a factor of ~100 in area, and the early-stopping decision
    would then be made mostly by the small voids §5 reports separately anyway. Pooling is also
    what makes this number comparable to the ``rmse`` column :mod:`benchmark.metrics` computes,
    without being that column: it is one number over a set §5 never prints.
    """
    squares = 0.0
    count = 0
    items = [(item.tile_id, item.mask_rle) for item in instances]
    for batch in batches(cfg, items, tiles, batch_size=cfg.head.batch_size, index=index):
        filled = predict_composed(batch, stack, precision=cfg.head.precision)
        for i, out in enumerate(filled):
            void = batch.void[i]
            err = out[void] - batch.raw_i16[i].astype(np.float32)[void]
            squares += float(np.square(err.astype(np.float64)).sum())
            count += int(void.sum())
    if not count:
        raise TrainError("§3.1's val draw produced no void pixels to early-stop on")
    return math.sqrt(squares / count)


def train_head(
    cfg: BenchmarkConfig,
    arm: arms.Arm,
    *,
    extractor=None,
    epochs: int | None = None,
    recipe=None,
    tag: str | None = None,
    on_epoch: Callable[[Epoch], None] | None = None,
) -> Trained:
    """Train §4.1's head for one cell. The encoder is frozen and asserted so on the way out.

    ``recipe`` overrides ``cfg.head`` **for the tuning pass and nothing else**
    (:func:`benchmark.tune.tune`); a `run` passes none and gets the frozen one, which
    :func:`benchmark.tune.assert_frozen` has already checked. ``epochs`` is the tuning pass's
    other short-cut, for the same reason and with the same restriction.

    ``tag`` distinguishes this pass's checkpoint directory from a sibling's. A cell does not need
    one — ``(arm, size, seed)`` is its identity — but a **tuning pass** trains several heads at
    the same identity, and without a tag each grid point overwrote the last one's epoch files and
    left a ``best.pt`` holding the *last* point's weights rather than the winner's. Nothing reads
    those files, so no number was ever wrong; what was wrong is that a directory named for a cell
    held something else, which is the same shape of problem as a printed count nothing checked.

    The order below is the argument:

    1. **The two draws, behind the guard** — :class:`TrainingTiles`, which is the only tile
       source opened here and which cannot reach a test tile.
    2. **The bank, restricted to this cell's tiles.** Read, never drawn.
    3. **The encoder, checksummed.** Before a step, so that "no parameter moved" is a comparison
       rather than an assertion about intent.
    4. **Fifty epochs, val-scored, best-epoch restored.** §4.1 says *early stop on val RMSE*, and
       what that means here is **checkpoint selection**: the full cosine runs and the weights the
       pass returns are the lowest-val epoch's. Stopping the schedule early instead would need a
       patience §4.1 does not state, and would leave a cosine-to-0 curve cut off mid-decay — two
       recipes fighting, one of them invented here. Every epoch is checkpointed either way, so a
       run that wants a different rule can apply it to the files rather than re-training.
    """
    import torch

    started = time.perf_counter()
    recipe = cfg.head if recipe is None else recipe
    epochs = recipe.epochs if epochs is None else epochs
    device = resolve_device(cfg)
    encoder = arms.ENCODERS[arm.encoder] if arm.encoder else None
    if encoder is None:
        raise TrainError(
            f"arm {arm.name} runs no encoder, and §4.1's head reads patch tokens. A7's UNet "
            f"trains from scratch and is #162's, not this path's"
        )

    tiles = TrainingTiles.read(cfg)
    assert_reads_val_only(tiles.sources)
    train_ids = list(tiles.head_train[: cfg.train_tiles])
    banked = read_bank(cfg, train_ids)
    val = val_instances(cfg, tiles.val)

    if extractor is None:
        extractor = extract.Extractor(
            extract.load_spec(extract.spec_path(cfg)), cfg, device=device
        )
    model = extractor.encoder(arm.name)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    before = encoder_checksums(model)

    built = head.build(encoder.width, recipe, seed=cfg.seed)
    module = built.module.to(device)
    stack = Stack(extractor=extractor, module=module, arm=arm.name, device=device)
    optimizer = torch.optim.AdamW(
        parameter_groups(module, weight_decay=recipe.weight_decay),
        lr=recipe.lr,
        weight_decay=recipe.weight_decay,
    )
    steps_per_epoch = max(1, math.ceil(len(train_ids) / recipe.batch_size))
    schedule = Cosine.of(recipe, steps_per_epoch=steps_per_epoch, epochs=epochs)

    cell = f"{cfg.arm}-{cfg.size}-seed{cfg.seed}"
    directory = Path(cfg.runtime.checkpoint_dir) / (cell if tag is None else f"{cell}-{tag}")
    directory.mkdir(parents=True, exist_ok=True)

    # Built once here and inherited by every worker of every epoch. See `Examples.index`: a
    # worker that builds its own reads the whole manifest, and that cost is paid per worker per
    # epoch rather than once per run.
    from .evaluate import TileStore

    index = TileStore.index_for(
        cfg.paths.corpus_dir, [*train_ids, *(item.tile_id for item in val)]
    )

    order = np.random.default_rng(cfg.seed)
    history: list[Epoch] = []
    #: The lowest-val epoch so far, as the :class:`Epoch` line itself rather than as a pair of
    #: loose numbers — it is what the pass returns and what the run record quotes.
    best: Epoch | None = None
    step = 0

    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        module.train()
        # The epoch's pairing of tile to void: §3.4's epoch-`e` draw for each tile, in an order
        # this cell's seed decides. The *voids* come off the bank and the *order* comes off the
        # seed — two different things, and only the second one is what separates three seeds of
        # the same cell.
        shuffled = [
            (train_ids[i], banked.rle(train_ids[i], epoch))
            for i in order.permutation(len(train_ids))
        ]
        totals = {"loss": 0.0, "l1_m": 0.0, "slope_l1": 0.0}
        phases = {"load": 0.0, "forward": 0.0, "backward": 0.0, "val": 0.0}
        lr = recipe.lr
        # GPU work is asynchronous, so a wall clock that does not synchronize measures how fast
        # Python queued the kernels rather than how long they ran. `mark` is the only honest way
        # to attribute seconds to a phase on an accelerator.
        mark = _clock(device)
        tick = mark()
        for batch in batches(cfg, shuffled, tiles, batch_size=recipe.batch_size, index=index):
            phases["load"] += mark() - tick
            tick = mark()
            lr = schedule.apply(optimizer, step)
            residual, void = residual_of(batch, stack, precision=recipe.precision)
            truth = torch.from_numpy(
                batch.raw_i16.astype(np.float32)
            ).unsqueeze(1).to(device)
            a1 = torch.from_numpy(batch.a1_fill).unsqueeze(1).to(device)
            # §2.4's convention, in torch and in metres, and the same one `predict` applies in
            # numpy: the prediction is the fill plus the residual *inside the void*.
            prediction = a1 + residual * void
            terms = head.loss(prediction, truth, void, slope_weight=recipe.slope_weight)
            phases["forward"] += mark() - tick
            tick = mark()
            optimizer.zero_grad(set_to_none=True)
            terms.total.backward()
            optimizer.step()
            step += 1
            for key, value in terms.detached().items():
                totals[key] += value
            phases["backward"] += mark() - tick
            tick = mark()

        module.eval()
        tick = mark()
        scored = val_rmse(val, cfg, stack, tiles=tiles, index=index)
        phases["val"] = mark() - tick
        line = Epoch(
            epoch=epoch,
            loss=totals["loss"] / steps_per_epoch,
            elevation=totals["l1_m"] / steps_per_epoch,
            slope=totals["slope_l1"] / steps_per_epoch,
            lr=lr,
            val_rmse=scored,
            seconds=time.perf_counter() - epoch_started,
            phases={k: round(v, 1) for k, v in phases.items()},
        )
        history.append(line)
        save_checkpoint(module, line, directory / EPOCH_CHECKPOINT.format(epoch=epoch))
        if best is None or line.val_rmse < best.val_rmse:
            best = line
            save_checkpoint(module, line, directory / BEST_CHECKPOINT)
        if on_epoch is not None:
            on_epoch(line)

    if best is not None:
        module.load_state_dict(
            torch.load(directory / BEST_CHECKPOINT, map_location=device)["state_dict"]
        )
    module.eval()
    assert_encoder_frozen(model, name=encoder.name, before=before)

    return Trained(
        module=module,
        width=built.width,
        parameters=built.parameters,
        best_epoch=-1 if best is None else best.epoch,
        best_val_rmse=math.inf if best is None else best.val_rmse,
        history=tuple(history),
        seconds=time.perf_counter() - started,
        checkpoint=None if best is None else str(directory / BEST_CHECKPOINT),
        tiles=len(train_ids),
        sources=tiles.sources,
        stack=stack,
        precision_declared=recipe.precision,
        precision_used=precision_on(device, recipe.precision),
    )


def format_trained(trained: Trained) -> list[str]:
    """What `run` prints once the head is trained — the counts, and the epoch it kept."""
    return [
        f"{'head':<16} {head.format_parameters(trained.width)}",
        f"{'trained':<16} {len(trained.history)} epochs over {trained.tiles:,} tiles in "
        f"{trained.seconds:.1f}s, reading {', '.join(trained.sources)}",
        f"{'kept':<16} epoch {trained.best_epoch} at val_rmse "
        f"{trained.best_val_rmse:.4f} m — {trained.checkpoint}",
        f"{'precision':<16} {trained.precision_declared} declared, "
        f"{trained.precision_used} on {'' if trained.stack is None else trained.stack.device}",
    ]
