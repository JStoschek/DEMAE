"""The eval pass — committed masks over committed tiles, scored into rows (#158).

`run` executes exactly one ``(arm, size, seed)`` cell, and this is what that means once the
resolution has been printed and the box has been refused or accepted: join the committed eval
mask list to the committed test tiles, hand each instance to the arm, score the fill inside the
void in metres, and hand :mod:`benchmark.results` one dict per instance.

## Why this is a module and not a stretch of `launch`

The spec lists a module per plan section and does not name this one, so it is worth saying what
it is: :mod:`benchmark.results` owns *the file* — the schema, the run record, the replace-by-cell
rule — and this owns *the pass*, which is the part that reads pixels. Keeping them apart is what
lets the results file be tested without a corpus and the pass be tested without a parquet. What
is emphatically not here is a ``main``: `launch` is the only entry point (ADR-0009), and this
module is documented as **not a way to produce a result** exactly as ``pretrain.train`` is.

## Two conventions today, a table tomorrow

:data:`FILLERS` dispatches on the arm's **input convention** rather than on its name, because
that is the field §1's table actually varies: ``classical_fill`` is A1 and is built here;
``a1_fill_plus_mask`` is A4, A5 and A6's frozen-encoder-plus-head and arrived with #159;
``token_drop`` and ``a1_fill_through_tokenizer`` are the two the later arm tickets attach to,
and each names its own ticket when asked for rather than failing as an unknown key. Dispatching
on the convention is what made §2.4's residual rule expressible without a second pass, and it is
why A4, A5 and A6 are one entry rather than three: what differs between them is the encoder the
registry names, which this module never asks about.

The head is **not built here**. A learned cell is train-then-score, and what crosses into this
module is :class:`Learned` — a callable and a batch size — so that the pass scores the weights
the training run early-stopped rather than a fresh head, and so that A1's scoring path still
imports no torch.

## Reading tiles without torch

The corpus's own read path is :class:`dataset.loader.TileDataset`, and it imports
``torch.utils.data``. §3.4 has the scoring path CPU-only and torch-free so that A1 runs on a box
with no CUDA stack, so :class:`TileStore` resolves ``tile_id -> (shard_id, row_index) ->
memmap[shard_id][row_index]`` itself. That is a second reader of one storage format and the risk
is real, so it is kept to the smallest possible surface: it reads the raw int16 array and the
manifest columns beside it, applies **no** normalization, and derives its pixel spacing from
:func:`dataset.geometry.band_pixel_spacing_m` — the shard-backed reader's own function, which
:mod:`dataset.gallery` already uses for the same reason.

Raw int16 is metres, so A1 needs no inversion at all: :func:`benchmark.metrics.to_metres` is for
the arms whose output arrives per-tile standardized, and A1's never is. That is worth knowing
when reading the two gates below, because it means A1 cannot fail the units check by accident —
which is precisely why the gates fire against the *pipeline* rather than against `score`.

## The two gates, and what each actually proves

Both fire before a single instance is scored, because the failure they close is a table that
looks normal (§8's risk row):

* ``fill(identity-mask) == input`` (:func:`identity_gate`) — a real corpus tile through the real
  fill with an empty void. It proves the *plumbing*: the mask decoded to the polarity everything
  downstream assumes, the tile arrived in metres, and nothing rounded. An inverted mask fills the
  whole tile here and fails loudly instead of scoring a whole pass of plausible numbers.
* **The golden tile** (:func:`golden_gate`) — §8's committed fixture, hand-checked, driven
  through the same :func:`benchmark.metrics.score` call the pass uses and asserted against the
  numbers ticket 10 committed. It proves the *scoring*, at the CLI altitude, against the same
  file `tests/test_benchmark_metrics.py` asserts at the function seam.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np

from dataset.geometry import PixelSize, band_pixel_spacing_m
from pretrain.artifact import read_tile_strata

from . import arms, fill, masks, metrics, prereg, results, voids
from .artifacts import EVAL_MASKS_NAME, TEST_TILES_NAME, file_md5
from .config import BenchmarkConfig

#: The manifest columns the pass reads. Five, and no pixel: everything else about a tile comes
#: off the shard. ``slope_mean`` is here because §5's ruggedness bands are cut on it — see
#: :func:`bands_for` on why that column and not ``stratum_id``.
MANIFEST_COLUMNS: Final = ("tile_id", "shard_id", "row_index", "lat", "slope_mean")

#: How tight the golden-tile gate is. The same tolerance
#: ``tests/test_benchmark_metrics.py`` asserts the fixture at, so the two altitudes agree about
#: what "the same numbers" means rather than each picking a bar.
GOLDEN_TOLERANCE: Final = 1e-12


@dataclass(frozen=True)
class Tile:
    """One corpus tile as the scoring path wants it: raw metres, and where it is."""

    tile_id: str
    #: ``(224, 224)`` int16 **metres**, exactly as the shard stores it. No normalization has
    #: been applied and none should be: §5 scores in metres.
    raw_i16: np.ndarray
    lat: float
    #: ``(dy, dx)`` ground metres per pixel. Anisotropic, and applied to its own axis: Copernicus
    #: columns are up to ~1.55x wider on the ground than its rows near 50 deg (ADR-0001 §1), and
    #: one averaged number would quietly rotate every aspect poleward.
    px: PixelSize
    #: The manifest's own value, and what :class:`benchmark.voids.Bands` digitizes.
    slope_mean: float


class TileStore:
    """``tile_id -> raw int16 tile``, off the shards, without torch. See the module docstring.

    Memmaps are opened lazily and kept, which is :class:`dataset.loader.TileDataset`'s own
    strategy: an eval pass walks 5,000 test tiles scattered across every shard, so opening per
    read would re-`mmap` the same handful of files 30,000 times.
    """

    def __init__(self, corpus_dir: Path | str) -> None:
        import pyarrow.parquet as pq

        self.corpus_dir = Path(corpus_dir)
        manifest = self.corpus_dir / "manifest.parquet"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"{manifest} is not on this box, and the eval pass scores committed masks over "
                f"corpus tiles. `paths.corpus_dir` is where the 47 GB `corpus_500k` is mounted"
            )
        table = pq.read_table(manifest, columns=list(MANIFEST_COLUMNS))
        column = {
            name: table.column(name).to_numpy(zero_copy_only=False)
            for name in MANIFEST_COLUMNS
        }
        self._row = {
            str(tile_id): i for i, tile_id in enumerate(column["tile_id"].astype(str))
        }
        self._column = column
        self._memmaps: dict[int, np.memmap] = {}

    def __contains__(self, tile_id: str) -> bool:
        return tile_id in self._row

    def _memmap(self, shard_id: int) -> np.memmap:
        from dataset.shards import shard_path

        handle = self._memmaps.get(shard_id)
        if handle is None:
            handle = np.load(
                shard_path(self.corpus_dir / "shards", shard_id), mmap_mode="r"
            )
            self._memmaps[shard_id] = handle
        return handle

    def read(self, tile_id: str) -> Tile:
        """One tile, by id, in raw int16 metres."""
        try:
            i = self._row[tile_id]
        except KeyError:
            raise KeyError(
                f"{tile_id} is named by a committed artifact and is not in "
                f"{self.corpus_dir}/manifest.parquet. The eval mask list and the corpus have "
                f"come apart — re-freeze against this corpus, or mount the one the artifacts "
                f"were frozen from (their md5s are in every run record)"
            ) from None
        lat = float(self._column["lat"][i])
        raw = self._memmap(int(self._column["shard_id"][i]))[int(self._column["row_index"][i])]
        return Tile(
            tile_id=tile_id,
            raw_i16=np.asarray(raw),
            lat=lat,
            px=band_pixel_spacing_m(lat),
            slope_mean=float(self._column["slope_mean"][i]),
        )


def bands_for(cfg: BenchmarkConfig) -> voids.Bands:
    """§5's four ruggedness bands — **quartiles of** ``slope_mean``, not k-means strata.

    Delegated whole to :meth:`benchmark.voids.Bands.from_manifest` rather than recut here, and
    the distinction it encodes is the one #158 asks for explicitly. The corpus's ``stratum_id``
    is a k=20 k-means label over the six-metric frozen vector (ADR-0004): its numbers are
    *names*, carrying no ordering at all, so "band 1 is flatter than band 2" is a sentence that
    cannot be said about them. §5's bands are quartile cuts of one ordered quantity over the
    eligible test pool, which is what makes "a majority of the four ruggedness bands" (R12, and
    A2's negative-result rule) a statement with a direction.
    """
    return voids.Bands.from_manifest(cfg.paths.corpus_dir, split=cfg.tiles.test_split)


# --- The arm's fill -----------------------------------------------------------


@dataclass(frozen=True)
class Learned:
    """A trained arm's inference half, handed to the pass rather than built by it (#159).

    :mod:`benchmark.train` assembles this — the frozen encoder, the trained head, the device —
    and `launch` passes it in. This module deliberately does **not** import that one: the eval
    pass is torch-free for A1 (§3.4), and a scoring path that imported a training module to score
    a classical interpolation would have made that untrue for the sake of one type annotation.
    What crosses the seam is a callable and a batch size.

    ``predict`` has :func:`benchmark.train.predict`'s signature — raw int16 tiles and boolean
    voids in, one float32 fill in metres per instance out — and it is the *same function the
    training loop scored its val set with*. That identity is the whole reason the head is passed
    in already trained rather than rebuilt here.
    """

    #: The registered arm this was trained for, checked against the pass's own arm.
    arm: str
    #: ``(raw_i16, voids, cfg) -> [filled]``. See above.
    predict: Callable[[Sequence[np.ndarray], Sequence[np.ndarray], BenchmarkConfig], list]
    #: How many instances go through one encoder forward. §4.1's training batch by default,
    #: because the eval pass runs the identical composition and a frozen forward has no reason
    #: to be narrower here than it was there.
    batch_size: int = 1
    #: What the run record says about the head behind this — :meth:`benchmark.train.Trained.to_dict`.
    detail: Mapping[str, Any] = field(default_factory=dict)


def _classical_fill(
    tiles: Sequence[Tile],
    voids: Sequence[np.ndarray],
    cfg: BenchmarkConfig,
    learned: Learned | None,
) -> list[np.ndarray]:
    """A1 — §2.4's classical interpolation over the raw metres the shard holds."""
    return [fill.fill(tile.raw_i16, void, recipe=cfg.fill) for tile, void in zip(tiles, voids)]


def _head_fill(
    tiles: Sequence[Tile],
    voids: Sequence[np.ndarray],
    cfg: BenchmarkConfig,
    learned: Learned | None,
) -> list[np.ndarray]:
    """A4, A5 and A6 — the A1 fill plus §4.1's head residual, inside the void (§2.4, #159).

    The head is **not** built here. A cell of this convention is *train, then score*, and the
    thing scored has to be the object the training pass early-stopped: a pass that instantiated
    a fresh head at eval time would score a zero-output head, which by construction scores
    exactly A1 — the quietest possible way for a whole arm to be wrong.
    """
    if learned is None:
        raise NotImplementedError(
            "an `a1_fill_plus_mask` cell is train-then-score and this pass was handed no "
            "trained head. `benchmark.launch` trains one (§4.1, #159) and passes it in; a pass "
            "reached any other way has nothing to score but A1 wearing another arm's name"
        )
    return learned.predict([tile.raw_i16 for tile in tiles], voids, cfg)


def _unbuilt(convention: str, what: str, ticket: str) -> Callable[..., list[np.ndarray]]:
    """A convention that is registered and not yet built, saying which ticket builds it."""

    def refuse(
        tiles: Sequence[Tile],
        voids: Sequence[np.ndarray],
        cfg: BenchmarkConfig,
        learned: Learned | None,
    ) -> list[np.ndarray]:
        raise NotImplementedError(
            f"the {convention!r} input convention is not built: {what} ({ticket}). A1 runs "
            f"first and its numbers gate every learned arm, because each of them is a residual "
            f"over this fill (§2.4, §4.5)"
        )

    return refuse


#: One entry per :data:`benchmark.arms.INPUT_CONVENTIONS` value — the table §1's rows vary over.
#: Complete rather than partial, so an unbuilt arm names its ticket instead of failing as a
#: missing key, and so that adding a convention to the registry without building it is a visible
#: gap here rather than a `KeyError` on a rented box.
#:
#: Every filler takes and returns a **batch**. One instance at a time is what A1 wants and what
#: keeps its memory flat; one encoder forward per instance is not what the frozen arms want, and
#: a table whose signature could only express the first would have made the second a second pass.
FILLERS: Final[
    dict[str, Callable[[Sequence[Tile], Sequence[np.ndarray], BenchmarkConfig, Any], list]]
] = {
    "classical_fill": _classical_fill,
    "a1_fill_plus_mask": _head_fill,
    "token_drop": _unbuilt(
        "token_drop", "DEMAE's native MAE inpainting path", "#160's zero-shot and fine-tuned arms"
    ),
    "a1_fill_through_tokenizer": _unbuilt(
        "a1_fill_through_tokenizer", "A9's encode/FSQ/diffusion-decode pass", "#163"
    ),
}


# --- The two gates ------------------------------------------------------------


def identity_gate(tile: Tile, cfg: BenchmarkConfig) -> dict[str, Any]:
    """``fill(identity-mask) == input`` — the first of §8's two norm-inversion gates.

    Run against a **real corpus tile**, because what is being held shut is this pipeline reading
    this corpus, and a synthetic array would only prove the arithmetic.

    ## Where the gate is aimed, and why it is not aimed at `fill`

    :func:`benchmark.fill.fill` ends by restoring every pixel outside the void from its input.
    With an empty void that is the whole tile, so ``fill(identity-mask) == input`` is true **by
    construction** there and a gate pointed at it would assert a tautology — it would pass on a
    variant that returned noise. So the gate is pointed one level down, at the **variant's own
    output**, before the restore. That is a claim that can fail: a variant that rescales, that
    recentres, or that round-trips through a dtype too narrow for the corpus's metres returns
    something else, and GDAL's fill is only identity-preserving here because a tile is int16
    metres — a fact about the corpus, not about the method.

    The composed :func:`~benchmark.fill.fill` is checked too, second, so the restore itself is
    covered. Both are exact rather than approximate; see :mod:`benchmark.fill`.

    ## What this gate does not prove

    Not mask polarity — there is no mask to invert when the mask is empty. That check belongs
    where a *committed* mask is decoded, and it is :func:`score_instance`'s area cap.
    """
    empty = np.zeros(tile.raw_i16.shape, dtype=bool)
    expected = np.asarray(tile.raw_i16, dtype=fill.DTYPE)

    def refuse(got: np.ndarray, what: str) -> None:
        worst = int(np.argmax(np.abs(np.asarray(got, dtype=np.float64) - expected)))
        raise ValueError(
            f"fill(identity-mask) != input on {tile.tile_id}: with no void to fill, {what} "
            f"returned {np.ravel(got)[worst]} where the tile holds {expected.ravel()[worst]} "
            f"metres. §8's gate. Every learned arm's output is a residual over this fill (§2.4), "
            f"so a fill that moves untouched ground moves every arm's row by the same amount and "
            f"no column would say so"
        )

    variant = fill.VARIANTS[cfg.fill.variant](tile.raw_i16, empty, cfg.fill)
    if not np.array_equal(variant, expected):
        refuse(variant, f"the {cfg.fill.variant} variant")
    composed = fill.fill(tile.raw_i16, empty, recipe=cfg.fill)
    if not np.array_equal(composed, expected):
        refuse(composed, "the composed fill")
    return {
        "tile_id": tile.tile_id,
        "variant": cfg.fill.variant,
        "pixels": int(expected.size),
        "checked": ["variant", "composed"],
    }


def golden_gate(cfg: BenchmarkConfig, bar: prereg.Bar) -> dict[str, Any]:
    """§8's golden tile, through the shipping path, against its committed hand-checked scores.

    The fixture's fill is *right in shape and wrong in level* — the truth plus a 4 m offset plus
    a ±1.5 m checkerboard ripple — which lights up every registered column for a different
    reason (see :data:`benchmark.metrics.GOLDEN_RECIPE`). It is **not** an A1 fill and is not
    meant to be: what is being gated is the scoring half of this pass, so the fixture's job is to
    be a fill whose every score is known in closed form, not one this arm would have produced.

    Driven through the same :func:`benchmark.metrics.score` call, with the same bar, that the
    pass below uses. That is the whole point of asserting it here as well as at the function
    seam: a `run` that resolved the wrong bar, or scored a subset of the roster, is caught before
    it scores 30,000 instances against it.
    """
    directory = Path(cfg.paths.artifacts_dir)
    if not (directory / metrics.GOLDEN_SIDECAR).is_file():
        raise FileNotFoundError(
            f"§8's golden tile is not in {directory}: {metrics.GOLDEN_ARRAYS} and "
            f"{metrics.GOLDEN_SIDECAR} gate every arm's eval, and they are committed to the "
            f"repository rather than written by `freeze` (they are a fixture a test reads, not "
            f"an input a run scores). Point `paths.artifacts_dir` at the committed directory"
        )
    golden = metrics.read_golden(directory)
    scored = metrics.score(golden.truth, golden.filled, golden.void, golden.px, bar=bar)

    if set(scored) != set(golden.expected):
        raise ValueError(
            f"§8's golden tile is committed with {sorted(golden.expected)} and this run's bar "
            f"scores {sorted(scored)}. The roster comes from R1 in "
            f"{bar.path}, so the fixture and the registered metric set have come apart"
        )
    off = {
        name: (value, golden.expected[name])
        for name, value in scored.items()
        if not np.isclose(
            value, golden.expected[name], rtol=GOLDEN_TOLERANCE, atol=GOLDEN_TOLERANCE
        )
    }
    if off:
        detail = ", ".join(f"{n} {got!r} != {want!r}" for n, (got, want) in sorted(off.items()))
        raise ValueError(
            f"§8's golden tile does not score what it is committed to score: {detail}. Its fill "
            f"is hand-checked and its recipe is arithmetic, so this is the scoring path having "
            f"moved rather than the fixture being stale — and it is caught here, before an arm "
            f"scores a pass against it (§8, #155/#158)"
        )
    return {
        "arrays_md5": file_md5(directory / metrics.GOLDEN_ARRAYS),
        "metrics": sorted(scored),
        "tolerance": GOLDEN_TOLERANCE,
    }


# --- The pass -----------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One committed eval instance: which tile, which mask, and where in the row key it sits.

    The committed row's fifth column, ``seed``, is deliberately **not** carried. It is what
    :func:`benchmark.masks.draw_from_seed` reproduces the mask *from*, and this pass does not
    reproduce anything — it reads the committed RLE. What a results row needs is the mask's
    *identity*, and that is :attr:`Scored.mask_rle_md5`, which pins the actual bytes rather than
    the recipe that would regenerate them.
    """

    tile_id: str
    family: str
    bucket: str
    mask_rle: str
    instance: int


@dataclass(frozen=True)
class Scored:
    """One scored instance, as the pass hands it on — everything but the cell it belongs to.

    A dataclass rather than a dict, so the whole chain from ``Instance`` through ``Scored`` to
    :class:`benchmark.results.ResultRow` is typed end to end: the pass knows nothing about the
    arm, the size or the seed, and the results module fills those in once
    (:func:`benchmark.results.rows_from`) rather than eleven string keys being re-indexed on the
    way past.
    """

    tile_id: str
    family: str
    bucket: str
    band: str
    instance: int
    void_area_px: int
    void_eqdiam_px: float
    mask_rle_md5: str
    #: R1's roster by name, straight from :func:`benchmark.metrics.score`.
    metrics: dict[str, float]


def read_instances(cfg: BenchmarkConfig) -> list[Instance]:
    """The committed eval mask list, in its committed order, containment-checked.

    Two refusals on the way in, both about *this box* rather than about the code:

    * a mask list naming a tile the committed test list does not — which is what a re-drawn test
      list beside a stale mask list looks like, and which would score an arm on ground §3.1 never
      put in the test split;
    * an empty list, which `freeze` cannot produce and which would otherwise land a run record
      claiming a scored cell with no rows in it.

    ``instance`` counts within ``(tile_id, family, bucket)`` in committed order. It is 0 for
    every synthetic row — one void per tile per bucket — and is what separates the real voids
    that share a host tile: #153 landed 9,273 of them onto 1,116 hosts, a mean of 8.3 each and
    at most 23, so ``(tile_id, family, bucket)`` is not a key on that half of the list.

    ``family`` and ``bucket`` are carried as **labels**, never mapped back to §3.2's buckets.
    That is load-bearing now that both families are in the file: a real row's ``bucket`` is one
    of §3.3's size sub-rows (``real-small``, ``real-medium+``, ``count-only-128-160``), a
    different vocabulary from the synthetic ``small``/``medium``/``large``. Anything downstream
    that groups on ``bucket`` has to split on ``family`` first or it will pool two axes that
    mean different things.
    """
    import pyarrow.parquet as pq

    directory = Path(cfg.paths.artifacts_dir)
    table = pq.read_table(directory / EVAL_MASKS_NAME, columns=list(masks.COLUMNS))
    column = {name: table.column(name).to_pylist() for name in masks.COLUMNS}

    committed = {
        tile_id
        for tile_id, _ in read_tile_strata(directory / TEST_TILES_NAME, what="test list")
    }
    named = set(column["tile_id"])
    stray = sorted(named - committed)
    if stray:
        raise ValueError(
            f"{EVAL_MASKS_NAME} names {len(stray):,} tile(s) the committed {TEST_TILES_NAME} "
            f"does not — {stray[:3]}{'...' if len(stray) > 3 else ''}. §3.2 draws one eval mask "
            f"per test tile, so the two artifacts are one draw; a mask list scored against a "
            f"different test list is a benchmark scoring tiles §3.1 never put in the test split"
        )

    seen: dict[tuple[str, str, str], int] = {}
    instances = []
    for i in range(table.num_rows):
        key = (column["tile_id"][i], column["family"][i], column["bucket"][i])
        instances.append(
            Instance(
                tile_id=key[0],
                family=key[1],
                bucket=key[2],
                mask_rle=column["mask_rle"][i],
                instance=seen.get(key, 0),
            )
        )
        seen[key] = seen.get(key, 0) + 1
    if not instances:
        raise ValueError(
            f"{directory / EVAL_MASKS_NAME} holds no rows. §3.2 commits one mask per test tile "
            f"per family per bucket, so an empty list is a failed `freeze` rather than a cell "
            f"with nothing to score"
        )
    return instances


def decode_void(tile: Tile, instance: Instance, cfg: BenchmarkConfig) -> np.ndarray:
    """The committed RLE for this instance, decoded and checked against §3.2's area cap.

    Separate from the scoring below because the two happen at different rhythms once an arm
    batches: every instance in a batch is decoded and checked before any of them is filled, and
    the cap is a **polarity guard** that has to fire before a fill rather than after one.

    `freeze` checks every draw against this cap and never trims, so a committed mask over it did
    not come from the generator: an inverted decode is the way that happens, and it turns a 3%
    void into a 97% one, which scores a plausible table nothing refuses.
    """
    void = masks.decode_rle(instance.mask_rle)
    if void.shape != tile.raw_i16.shape:
        raise ValueError(
            f"the committed mask for {instance.tile_id} decodes to {void.shape} and the tile is "
            f"{tile.raw_i16.shape}"
        )
    area = int(void.sum())
    cap = cfg.masks.max_area_frac * void.size
    if area > cap:
        raise ValueError(
            f"the committed mask for {instance.tile_id} ({instance.family}/{instance.bucket}) "
            f"decodes to {area:,} px, {area / void.size:.1%} of the tile, over §3.2's "
            f"{cfg.masks.max_area_frac:.0%} cap. Every draw is checked against that cap at "
            f"`freeze` and none is ever trimmed, so this mask did not come from the generator — "
            f"the usual cause is a decode that inverted the void"
        )
    return void


def score_filled(
    tile: Tile,
    instance: Instance,
    void: np.ndarray,
    filled: np.ndarray,
    *,
    bands: voids.Bands,
    bar: prereg.Bar,
) -> Scored:
    """One arm's fill for one instance, scored, and said what it was.

    The band comes off the **tile's** ``slope_mean`` rather than the void's own terrain, on
    :class:`benchmark.voids.Bands`' rule: the bands describe the ground a void sits on, so a band
    index means the same thing here as it does for every other tile in the corpus.
    """
    area = int(void.sum())
    return Scored(
        tile_id=instance.tile_id,
        family=instance.family,
        bucket=instance.bucket,
        band=bands.label(bands.of(tile.slope_mean)),
        instance=instance.instance,
        void_area_px=area,
        void_eqdiam_px=masks.equivalent_diameter(area),
        mask_rle_md5=results.mask_rle_md5(instance.mask_rle),
        metrics=metrics.score(
            np.asarray(tile.raw_i16, dtype=np.float64), filled, void, tile.px, bar=bar
        ),
    )


def score_instance(
    tile: Tile,
    instance: Instance,
    cfg: BenchmarkConfig,
    *,
    arm: arms.Arm,
    bands: voids.Bands,
    bar: prereg.Bar,
    learned: Learned | None = None,
) -> Scored:
    """One instance, end to end — the batch of one the pass below generalizes."""
    void = decode_void(tile, instance, cfg)
    filled = FILLERS[arm.input_convention]([tile], [void], cfg, learned)[0]
    return score_filled(tile, instance, void, filled, bands=bands, bar=bar)


def format_gates(gates: dict[str, Any]) -> str:
    """The two gates, on one line, as something a build session can read at a glance.

    Printed even though they passed — a gate that only speaks when it fires is one nobody can
    tell ran. The golden tile's array md5 is in it because that is what says *which* fixture
    passed, and the fixture is two files that can drift apart (:func:`benchmark.metrics.
    read_golden`).
    """
    identity = gates["identity_mask"]
    golden = gates["golden_tile"]
    return (
        f"fill(identity-mask) == input on {identity['tile_id']} "
        f"({identity['pixels']:,} px, {identity['variant']}); "
        f"golden tile md5 {golden['arrays_md5']} scores its committed "
        f"{len(golden['metrics'])} to {golden['tolerance']:g}"
    )


@dataclass(frozen=True)
class Pass:
    """What one cell's eval produced: the scored instances, the gates, and what it cost."""

    scored: list[Scored]
    gates: dict[str, Any]
    seconds: float
    bar: prereg.Bar


def run_pass(
    cfg: BenchmarkConfig,
    arm: arms.Arm,
    *,
    learned: Learned | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Pass:
    """Score one cell over the committed eval mask list, both gates first.

    The order is the argument: the bar is read before anything (R1's roster is what `score`
    returns and R7's threshold is what the seam column is read against), then the gates, then
    the corpus is opened. A gate that fired after 30,000 instances had been scored would be a
    gate that costs 2.0–5.2 CPU-hours to consult.

    ``learned`` is a trained arm's inference half (:class:`Learned`, #159). ``None`` is A1 and
    every arm that trains nothing; a convention that needs one and is handed ``None`` refuses
    rather than scoring an untrained head, which by construction would score exactly A1.

    Instances are scored in batches of :attr:`Learned.batch_size`, which is **one** when there is
    no learned arm. That keeps A1 exactly where #158 left it — one tile in memory at a time over
    30,000 instances — while letting a frozen encoder see §4.1's batch in one forward instead of
    128 of them.
    """
    started = time.perf_counter()
    bar = metrics.bar_for(cfg)
    gates: dict[str, Any] = {"golden_tile": golden_gate(cfg, bar)}

    if learned is not None and learned.arm != arm.name:
        raise ValueError(
            f"this pass is scoring {arm.name} and was handed a head trained for {learned.arm}. "
            f"A4, A5 and A6 share one head *architecture* (§4.1) and never one set of weights — "
            f"the weights are what C1 compares"
        )

    instances = read_instances(cfg)
    store = TileStore(cfg.paths.corpus_dir)
    bands = bands_for(cfg)

    # The identity gate wants a real tile, so it fires here — after the store is open and before
    # the first instance is scored. The first committed instance's tile is as good as any and is
    # deterministic, which matters for a gate that gets quoted in a run record.
    gates["identity_mask"] = identity_gate(store.read(instances[0].tile_id), cfg)

    batch_size = 1 if learned is None else max(1, learned.batch_size)
    filler = FILLERS[arm.input_convention]
    scored: list[Scored] = []
    for start in range(0, len(instances), batch_size):
        chunk = instances[start : start + batch_size]
        tiles = [store.read(item.tile_id) for item in chunk]
        chunk_voids = [
            decode_void(tile, item, cfg) for tile, item in zip(tiles, chunk)
        ]
        filled = filler(tiles, chunk_voids, cfg, learned)
        for tile, item, void, out in zip(tiles, chunk, chunk_voids, filled):
            scored.append(score_filled(tile, item, void, out, bands=bands, bar=bar))
        if progress is not None:
            progress(len(scored), len(instances))
    return Pass(
        scored=scored,
        gates=gates,
        seconds=time.perf_counter() - started,
        bar=bar,
    )
