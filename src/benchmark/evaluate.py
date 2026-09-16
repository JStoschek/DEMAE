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

## A1 today, a table tomorrow

:data:`FILLERS` dispatches on the arm's **input convention** rather than on its name, because
that is the field §1's table actually varies: ``classical_fill`` is A1 and is built here;
``a1_fill_plus_mask``, ``token_drop`` and ``a1_fill_through_tokenizer`` are the three the later
arm tickets attach to, and each names its own ticket when asked for rather than failing as an
unknown key. Dispatching on the convention is also what makes §2.4's residual rule expressible
later without a second pass: three of the four conventions *begin* with the fill this one
produces.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Sequence

import numpy as np

from dataset.geometry import PixelSize, band_pixel_spacing_m
from pretrain.artifact import read_tile_strata

from . import arms, fill, masks, metrics, prereg, results, voids, zeroshot
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

    def __init__(self, corpus_dir: Path | str, *, index: dict | None = None) -> None:
        """Open the corpus. ``index`` is :meth:`index_for`'s output, for a caller that has one.

        Building the index reads the whole manifest — 499,934 rows, **1.24 s** — and a dataloader
        worker that builds its own pays that per worker per epoch. At 64 workers over §4.1's 50
        epochs that is 3,200 constructions and **66 minutes** of a cell spent re-deriving the
        same table, which is what made a 192-core box slower than a 16-core one (#159). A caller
        that knows which tiles it can reach builds the index once and hands it over; workers
        inherit it by fork and read nothing.
        """
        self.corpus_dir = Path(corpus_dir)
        self._memmaps: dict[int, np.memmap] = {}
        if index is not None:
            self._index = index
            return
        self._index = self.index_for(self.corpus_dir)

    @staticmethod
    def index_for(corpus_dir: Path | str, tile_ids: Iterable[str] | None = None) -> dict:
        """``tile_id -> (shard_id, row_index, lat, slope_mean)`` — everything :meth:`read` needs.

        ``tile_ids`` narrows it to the tiles a caller can actually reach, which is what makes the
        result small enough to hand to workers: a training pass touches 27,000 of the corpus's
        500,000, and the whole index for those is a few megabytes against the manifest's 40.
        ``None`` indexes everything, which is what an eval pass over an unknown tile set wants.
        """
        import pyarrow.parquet as pq

        corpus_dir = Path(corpus_dir)
        manifest = corpus_dir / "manifest.parquet"
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
        wanted = None if tile_ids is None else set(tile_ids)
        out: dict[str, tuple] = {}
        for i, tile_id in enumerate(column["tile_id"].astype(str)):
            name = str(tile_id)
            if wanted is None or name in wanted:
                out[name] = (
                    int(column["shard_id"][i]),
                    int(column["row_index"][i]),
                    float(column["lat"][i]),
                    float(column["slope_mean"][i]),
                )
        return out

    def __contains__(self, tile_id: str) -> bool:
        return tile_id in self._index

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
            shard_id, row_index, lat, slope_mean = self._index[tile_id]
        except KeyError:
            raise KeyError(
                f"{tile_id} is named by a committed artifact and is not in "
                f"{self.corpus_dir}/manifest.parquet. The eval mask list and the corpus have "
                f"come apart — re-freeze against this corpus, or mount the one the artifacts "
                f"were frozen from (their md5s are in every run record)"
            ) from None
        raw = self._memmap(shard_id)[row_index]
        return Tile(
            tile_id=tile_id,
            raw_i16=np.asarray(raw),
            lat=lat,
            px=band_pixel_spacing_m(lat),
            slope_mean=slope_mean,
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


#: What a **batch** of fills looks like once the cell is known: tiles and voids in, one filled
#: tile in metres out per instance.
#:
#: Batched rather than per-instance, and #159 is why: A4, A5 and A6 run a frozen ViT-B over every
#: instance, and one forward per instance is 36,411 of them where one per batch of 128 is 285.
#: A1 and A2 are unaffected — they declare a batch of one below and loop inside — so the batch
#: axis costs nothing to a convention that cannot use it.
Filler = Callable[[Sequence["Tile"], Sequence[np.ndarray]], list]

#: How a convention turns *the cell* into one of those. A factory rather than a plain function,
#: and #167 is why: a filler that runs an encoder has to load one, and a signature that took the
#: config per instance would load a ViT-B thirty thousand times. It is also what lets a
#: convention see the **arm** — ``token_drop`` is shared by A2-R5 and A2-R4, and a filler that
#: could not tell them apart would score one cell's rows with the other's weights.
#: ``trained`` is #159's half and is keyword-only with a default, so the conventions that train
#: nothing never mention it. It carries :class:`benchmark.train.Trained` for the arms that train
#: §4.1's head — the *weights*, not a recipe for getting them. A factory cannot train one itself:
#: training inside the eval pass is the seam that would let a cell be scored against a head no
#: run record describes, and `launch` trains before it scores for that reason.
FillerFor = Callable[..., "Fill"]


@dataclass(frozen=True)
class Fill:
    """What a convention hands the pass: how to fill a batch, and how big a batch it wants.

    The batch size travels **with** the filler rather than being the pass's to choose, because it
    is a fact about the convention: a frozen-encoder arm wants §4.1's 128 so one forward covers
    the batch, and A1 wants one so its memory stays flat over 36,411 instances. A pass that
    picked the number would be deciding something only the convention knows.
    """

    fill: Filler
    #: Instances per call. One unless the convention says otherwise.
    batch: int = 1

    def __post_init__(self) -> None:
        if self.batch < 1:
            raise ValueError(f"a filler's batch is at least one instance; got {self.batch}")


def _classical_fill(cfg: BenchmarkConfig, arm: arms.Arm, *, trained=None) -> Fill:
    """A1 — §2.4's classical interpolation over the raw metres the shard holds.

    A batch of one, which is what it wants: `fillnodata` is CPU work with no fixed cost to
    amortize, so batching would buy nothing and would hold 128 tiles in memory to do it.
    """

    def fill_batch(tiles: Sequence[Tile], voids: Sequence[np.ndarray]) -> list:
        return [fill.fill(t.raw_i16, v, recipe=cfg.fill) for t, v in zip(tiles, voids)]

    return Fill(fill=fill_batch)


def _token_drop(cfg: BenchmarkConfig, arm: arms.Arm, *, trained=None) -> Fill:
    """A2 — §4.3's native MAE inpainting, no trained head (#167, :mod:`benchmark.zeroshot`).

    The model is loaded **once here**, when the pass is built, and its own refusals fire before
    the first instance rather than 30,000 rows in — the same rule §8's two gates follow, and for
    the same reason: a refusal that costs an eval pass is one nobody consults until the bill
    arrives.

    A3 shares this convention and is not this ticket's: it trains ``encoder+decoder``, so what it
    needs from a checkpoint is #162's question. It reaches the branch below and is told so.
    """
    if not arm.runs_pretrained_decoder:
        return _unbuilt(
            "token_drop",
            f"{arm.name} trains {arm.trains} over this convention, and the fine-tune that "
            f"produces those weights is not built",
            "#162",
        )(cfg, arm)

    field = arms.ENCODERS[arm.encoder].mae_checkpoint_field
    path = None if field is None else getattr(cfg.paths, field)
    if path is None:
        raise FileNotFoundError(
            f"{arm.name} runs the pretrained MAE decoder and this box has not mounted one. "
            f"§14 exports encoder-only weights, so the release `encoder.pt` cannot drive it — "
            f"see the resolution line's `mae checkpoint` row"
        )
    inpainter = zeroshot.load_inpainter(path, expect=arms.ENCODERS[arm.encoder])
    inpainter.check_ready()

    def fill_batch(tiles: Sequence[Tile], voids: Sequence[np.ndarray]) -> list:
        # One at a time: #167's inpainter takes a tile. The batch axis is available to it the day
        # someone measures that batching the decoder is worth the memory, and costs nothing until.
        return [inpainter.infill(t.raw_i16, v) for t, v in zip(tiles, voids)]

    return Fill(fill=fill_batch)


def _head_fill(cfg: BenchmarkConfig, arm: arms.Arm, *, trained=None) -> Fill:
    """A4, A5 and A6 — the A1 fill plus §4.1's head residual, inside the void (§2.4, #159).

    The head is **not built here**, and that is the one thing this factory does differently from
    :func:`_token_drop`'s. A2's model is pretrained, so loading it is all there is to do; a head
    has to be *trained*, and training inside the eval pass would let a cell be scored against
    weights no run record describes. So `launch` trains first and hands the result in, and this
    refuses rather than improvising when it is handed nothing — because a freshly built head
    outputs zero, and a zero-output head scores **exactly A1** (:func:`benchmark.head.build`
    zeroes the output convolution). That is the quietest possible way for a whole arm to be
    wrong: A4's row would be A1's row wearing A4's name, and no column would say so.

    §4.1's batch, because the composition here is the training loop's: one frozen forward covers
    128 instances (:func:`benchmark.train.predict`).
    """
    if trained is None:
        raise NotImplementedError(
            f"{arm.name} scores §4.1's head over a frozen encoder and this pass was handed no "
            f"trained head. `benchmark.launch` trains one before it scores (#159); a pass "
            f"reached any other way has nothing to score but A1 wearing another arm's name"
        )
    if trained.arm != arm.name:
        raise ValueError(
            f"this pass is scoring {arm.name} and was handed a head trained for {trained.arm}. "
            f"A4, A5 and A6 share one head *architecture* (§4.1) and never one set of weights — "
            f"the weights are what C1 compares"
        )

    def fill_batch(tiles: Sequence[Tile], voids: Sequence[np.ndarray]) -> list:
        return trained.predict([t.raw_i16 for t in tiles], voids, cfg)

    return Fill(fill=fill_batch, batch=cfg.head.batch_size)


def _tokenizer_pass_through(cfg: BenchmarkConfig, arm: arms.Arm, *, trained=None) -> Fill:
    """A9 — §1's reference row, the A1 fill through TerraMind's whole tokenizer (#168).

    The model is loaded **here**, when the pass is built, on :func:`_token_drop`'s rule and for
    a sharper version of its reason: this is the arm that reads 1.15 GB and 286 M parameters, so
    an unmounted checkpoint, a moved md5 or a terratorch whose DEM affine has been retuned all
    stop the run before the first instance rather than 30,000 rows in.

    §4.1's composition is shared with the learned arms rather than rebuilt —
    :func:`benchmark.train.compose` is what turns tiles and voids into the A1 fill, its int16
    rounding and ADR-0012's valid mask. A5 and A9 read the same encoder over the same input, and
    a second composition here would make that a claim instead of a fact.

    The batch is :attr:`benchmark.config.Tokenizer.batch_size` rather than §4.1's 128, and it is
    the convention's to declare for :class:`Fill`'s stated reason — except that here it is also
    recipe, because the decoder's noise is drawn once per batch. See
    :class:`benchmark.config.Tokenizer`.
    """
    # Imported here rather than at the top, on this module's standing rule: `benchmark.train`
    # and `benchmark.passthrough` both reach torch, and A1's scoring path runs on a box with no
    # CUDA stack (§3.4).
    from . import passthrough, train

    model = passthrough.load(cfg, arm)

    def fill_batch(tiles: Sequence[Tile], voids: Sequence[np.ndarray]) -> list:
        return model.run(train.compose([t.raw_i16 for t in tiles], voids, cfg))

    return Fill(fill=fill_batch, batch=cfg.tokenizer.batch_size)


def _unbuilt(convention: str, what: str, ticket: str) -> FillerFor:
    """A convention that is registered and not yet built, saying which ticket builds it."""

    def build(cfg: BenchmarkConfig, arm: arms.Arm, *, trained=None) -> Fill:
        def refuse(tiles: Sequence[Tile], voids: Sequence[np.ndarray]) -> list:
            raise NotImplementedError(
                f"the {convention!r} input convention is not built: {what} ({ticket}). A1 runs "
                f"first and its numbers gate every learned arm, because each of them is a "
                f"residual over this fill (§2.4, §4.5)"
            )

        return Fill(fill=refuse)

    return build


#: One entry per :data:`benchmark.arms.INPUT_CONVENTIONS` value — the table §1's rows vary over.
#: Complete rather than partial, so an unbuilt arm names its ticket instead of failing as a
#: missing key, and so that adding a convention to the registry without building it is a visible
#: gap here rather than a `KeyError` on a rented box.
FILLERS: Final[dict[str, FillerFor]] = {
    "classical_fill": _classical_fill,
    "a1_fill_plus_mask": _head_fill,
    "token_drop": _token_drop,
    "a1_fill_through_tokenizer": _tokenizer_pass_through,
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

    Separate from the scoring below because the two happen at different rhythms once a convention
    batches (#159): every instance in a batch is decoded and checked *before* any of them is
    filled, and the cap is a **polarity guard** that has to fire before a fill rather than after
    one.

    The band comes off the **tile's** ``slope_mean`` rather than the void's own terrain, on
    :class:`benchmark.voids.Bands`' rule: the bands describe the ground a void sits on, so a band
    index means the same thing here as it does for every other tile in the corpus.
    """
    void = masks.decode_rle(instance.mask_rle)
    if void.shape != tile.raw_i16.shape:
        raise ValueError(
            f"the committed mask for {instance.tile_id} decodes to {void.shape} and the tile is "
            f"{tile.raw_i16.shape}"
        )
    # §3.2's area cap, re-checked on the *decoded* mask — the polarity guard the identity gate
    # cannot be. `freeze` checks every draw against this cap and never trims, so a committed mask
    # over it did not come from the generator: an inverted decode is the way that happens, and it
    # turns a 3% void into a 97% one, which scores a plausible table nothing refuses.
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
    trained=None,
    progress: Callable[[int, int], None] | None = None,
) -> Pass:
    """Score one cell over the committed eval mask list, both gates first.

    The order is the argument: the bar is read before anything (R1's roster is what `score`
    returns and R7's threshold is what the seam column is read against), then the gates, then
    the corpus is opened. A gate that fired after 30,000 instances had been scored would be a
    gate that costs 2.0–5.2 CPU-hours to consult.

    ``trained`` is :class:`benchmark.train.Trained` for the arms that train §4.1's head, and
    ``None`` for every other (#159). It is handed *in* rather than made here: `launch` trains
    before it scores, so a cell's weights exist — and are described by its run record — before a
    single instance is filled.

    Instances are scored in batches of :attr:`Fill.batch`, which the convention chose: one for
    A1 and A2, §4.1's 128 for the frozen-encoder arms.
    """
    started = time.perf_counter()
    bar = metrics.bar_for(cfg)
    gates: dict[str, Any] = {"golden_tile": golden_gate(cfg, bar)}

    # Built here, with the gates, and for the gates' reason: an arm that loads an 86M-parameter
    # model refuses an unmounted checkpoint — or a decoder with no registered inversion (#167) —
    # while the clock is still cheap. A1's factory does nothing at all, so this costs it nothing.
    filler = FILLERS[arm.input_convention](cfg, arm, trained=trained)

    instances = read_instances(cfg)
    store = TileStore(cfg.paths.corpus_dir)
    bands = bands_for(cfg)

    # The identity gate wants a real tile, so it fires here — after the store is open and before
    # the first instance is scored. The first committed instance's tile is as good as any and is
    # deterministic, which matters for a gate that gets quoted in a run record.
    gates["identity_mask"] = identity_gate(store.read(instances[0].tile_id), cfg)

    scored: list[Scored] = []
    for start in range(0, len(instances), filler.batch):
        chunk = instances[start : start + filler.batch]
        tiles = [store.read(item.tile_id) for item in chunk]
        chunk_voids = [decode_void(t, item, cfg) for t, item in zip(tiles, chunk)]
        filled = filler.fill(tiles, chunk_voids)
        if len(filled) != len(chunk):
            raise ValueError(
                f"the {arm.input_convention!r} filler returned {len(filled)} fills for "
                f"{len(chunk)} instances — a batched filler answers one per instance, in order"
            )
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
