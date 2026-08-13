"""§3.2's void-mask generator, the RLE codec and the committed eval mask list (#150).

Promoted from `prototypes/p119_fill_headroom_proto.py`'s generator and the `p121_{masks,bank}.py`
probes, which are the code plan §3.2's measured constraint numbers were run against. Promotion is
not a file move: none of this was in ``src/``, none of it was under test, and the prototype's
constraints were `assert` statements, which `python -O` deletes.

## The one thing the generator must never do

Silently trim. §3.2's three constraints — fully interior at >= 8 px from the tile edge, never more
than 40% of the tile, equivalent diameter inside the bucket the row files it under — are
**checked on every draw** (:func:`check`), and a draw that breaks one is thrown away and redrawn,
or refused (:data:`MAX_ATTEMPTS`). It is never repaired. A trimmed mask is not a smaller void: it
is a void filed under a bucket it is no longer in, scored beside 4,999 tiles that *are*, in a
column whose caption says the bucket axis means one thing.

That is also why the large bucket tops at **158** px rather than the published 160. §3.2's
constraints are jointly unsatisfiable above ``d = 159.86`` px ([#121]), so a 160 px top would put
the generator in exactly the position of having to trim. The boundary was lowered in the plan,
the check lives in :class:`benchmark.config.Masks`, and this module simply draws inside whatever
it is handed — a bucket the cap cannot satisfy fails at the config, before a draw exists.

## What a seed is, and what it is for

Every draw is keyed: ``blake2b(GENERATOR | geometry.seed | tile_id | family | bucket)``, never a
process-salted ``hash`` and never a ``Generator`` handed down from a caller. The 63-bit result is
the row's own ``seed`` column, and :func:`draw_from_seed` re-draws the mask from it alone — so a
row is reproducible from what the row carries, and a ``mask_rle`` that its own seed no longer
produces is the one corruption an md5 over the whole file cannot localise.

## The RLE, and why it carries its shape

``"224x224:0,37,12,..."`` — alternating run lengths over the row-major flattening, opening with a
background run that may be zero-length. Text rather than packed ``int32``, measured rather than
assumed: run lengths are small numbers, so the text is **662 B/row** against int32's 743 B, and
under the parquet's zstd the two land within 4% of each other (103 vs 99 B/row). The text wins
the tie on being readable in a diff and hashable without an integer-width convention.

The ``HxW`` prefix is load-bearing. A mask decoded at the wrong width does not raise — it is a
plausible mask, scored normally, in the one column that says what every arm was asked to fill. A
codec whose inverse needs an out-of-band shape leaves that failure available.

## Cost

A draw is 1.9 ms in the prototype and 1.2 ms here, the difference being :func:`_dilate_to_area`:
iterated 3x3 dilation is the taxicab distance transform thresholded at a radius, so the same set
comes out of one pass instead of ~36. `tests/test_benchmark_masks.py` asserts the two are the
same set rather than similar ones. It is worth having twice — this generator also writes #151's
1,000,000-row bank, where the same factor is 20 minutes.

The eval mask list is 30,000 draws, ~40 s, and one file: 5,000 test tiles x §3.2's two synthetic
families x three buckets, one void per tile per instance, every eval tile appearing once per
bucket in both families so bucket scores are computed over identical tile sets. The real family's
9,978 rows are #153's, appended into the same five columns.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final

import numpy as np
from scipy.ndimage import distance_transform_cdt, distance_transform_edt, label

from dataset.config import TILE_PX
from pretrain.artifact import read_tile_strata

from .artifacts import EVAL_MASKS_NAME, TEST_TILES_NAME, write_committed
from .config import BenchmarkConfig, Masks

#: The generator's identity, and the first component of every seed. A change here is a change to
#: every mask in both artifacts, which is what makes it a version rather than a label.
GENERATOR: Final = "benchmark.masks-v1"

#: How many whole draws a cell gets before it is refused, and how many redraws a family gets
#: inside one. Reached only if a constraint is unsatisfiable for the geometry, which is a bug in
#: the draw rather than a reason to trim (see the module docstring); the ceiling is generous
#: because it is a refusal boundary, not a working budget.
#:
#: Measured over 500 draws per cell: **1.000** attempts for blob in all three buckets, and
#: 1.014 / 1.006 / 1.008 for rectangle in small / medium / large, where flooring the second side
#: can land an integer rectangle a hair under the bucket floor. The maximum ever seen is 2.
#:
#: These count *whole* draws, and are therefore not the number plan §3.2 publishes. Its "blob
#: draws retry **1.44x**" in the large bucket is :func:`blob_mask`'s internal walk-shortening
#: loop, which is invisible here because it resolves before a mask is returned. Re-measured on
#: this generator it is **1.158** walks per draw — see :func:`blob_mask`.
MAX_ATTEMPTS: Final = 24


class MaskError(ValueError):
    """A draw that §3.2's constraints refuse. Never repaired — see the module docstring."""


class NotInterior(MaskError):
    """The void reaches into the edge margin the plan keeps every void clear of."""


class OverAreaCap(MaskError):
    """The void covers more of the tile than §3.2's cap allows."""


class OutOfBucket(MaskError):
    """The void's own equivalent diameter is outside the bucket the row files it under."""


# --- Determinism --------------------------------------------------------------


def stable_seed(*parts: object) -> int:
    """A 63-bit seed from the string form of ``parts`` — no RNG state, and no ``hash``.

    Python's ``hash`` is salted per process, so a seed built from one would put a different
    number in a committed artifact on every run. blake2b does the same job for free, and is
    :mod:`dataset.splits`'s answer to the same question one phase earlier.
    """
    digest = hashlib.blake2b("|".join(str(part) for part in parts).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "little") >> 1


def seed_for(tile_id: str, family: str, bucket: str, geometry: Masks) -> int:
    """The seed one (tile, family, bucket) draw is committed under — the row's fifth column."""
    return stable_seed(GENERATOR, geometry.seed, tile_id, family, bucket)


# --- The RLE codec ------------------------------------------------------------


def encode_rle(mask: np.ndarray) -> str:
    """``"HxW:r0,r1,..."`` — alternating run lengths over the row-major flattening.

    The first run is background and may be zero-length, which is how a mask whose first pixel is
    void encodes without a special case. The shape rides in front; the module docstring says why.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"a void mask is a non-empty 2-D array; got shape {array.shape}")
    flat = array.ravel()
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    runs = np.diff(np.concatenate(([0], change, [flat.size])))
    if flat[0]:
        runs = np.concatenate(([0], runs))
    height, width = array.shape
    return f"{height}x{width}:" + ",".join(str(int(run)) for run in runs)


def decode_rle(text: str) -> np.ndarray:
    """The exact inverse of :func:`encode_rle`. Reads its shape out of the string it is given.

    Every way the text can fail to be an RLE is a refusal rather than a partial read: a mask
    decoded from a damaged column would be a plausible void, and would be scored as one.
    """
    shape, _, body = str(text).partition(":")
    height, _, width = shape.partition("x")
    try:
        rows, cols = int(height), int(width)
        runs = np.array([int(run) for run in body.split(",")], dtype=np.int64)
    except ValueError:
        raise ValueError(f"{text[:40]!r} is not an RLE ('HxW:r0,r1,...')") from None
    if rows < 1 or cols < 1 or runs.size == 0 or (runs < 0).any():
        raise ValueError(f"{text[:40]!r} is not an RLE ('HxW:r0,r1,...')")
    total = int(runs.sum())
    if total != rows * cols:
        raise ValueError(
            f"{rows}x{cols} needs {rows * cols} pixels and these runs cover {total}"
        )
    void = np.zeros(runs.size, dtype=bool)
    void[1::2] = True
    return np.repeat(void, runs).reshape(rows, cols)


# --- The geometry §3.2 measures in --------------------------------------------


def bucket_area(d: float) -> int:
    """Pixels covered by a void of equivalent diameter ``d``."""
    return int(round(math.pi * d * d / 4.0))


def equivalent_diameter(area: int) -> float:
    """``d = 2*sqrt(area/pi)`` — §3.2's unit, and the only thing a bucket boundary is in."""
    return 2.0 * math.sqrt(area / math.pi)


def bucket_of(eqdiam_px: float, geometry: Masks | None = None) -> str | None:
    """The bucket ``eqdiam_px`` falls in, or ``None`` in the gaps between them.

    Closed on both ends. §3.2's buckets are disjoint with room between them, so no diameter is
    in two — the config asserts that where the boundaries live.
    """
    geometry = Masks() if geometry is None else geometry
    for name, (lo, hi) in geometry.buckets:
        if lo <= eqdiam_px <= hi:
            return name
    return None


# --- The constraints ----------------------------------------------------------


def check(mask: np.ndarray, *, bucket: str, geometry: Masks | None = None) -> np.ndarray:
    """Raise unless ``mask`` satisfies all three of §3.2's constraints. Returns it if it does.

    Read off the mask's own shape rather than off a configured tile size, so the answer is about
    the array in hand. Called on **every** draw (:func:`draw_from_seed`), and public so that a
    reader of the committed column can make the *other* claim — that the artifact every arm
    scores against satisfies them — against the decoded rows rather than against the generator's
    word. `tests/test_benchmark_masks.py` makes exactly that pass over all 30,000.
    """
    geometry = Masks() if geometry is None else geometry
    margin = geometry.edge_margin_px
    area = int(np.asarray(mask, dtype=bool).sum())
    if area == 0:
        raise MaskError(f"a {bucket} instance came back with no void at all (0 px)")

    if margin and (
        mask[:margin].any()
        or mask[-margin:].any()
        or mask[:, :margin].any()
        or mask[:, -margin:].any()
    ):
        raise NotInterior(
            f"a {bucket} void reaches within {margin} px of the tile edge; §3.2 keeps every "
            f"void fully interior, so the fill is always surrounded by valid ground"
        )

    cap = geometry.max_area_frac * mask.size
    if area > cap:
        raise OverAreaCap(
            f"a {bucket} void covers {area:,} px of {mask.size:,} "
            f"({area / mask.size:.1%}), over §3.2's {geometry.max_area_frac:.0%} area cap"
        )

    eqdiam = equivalent_diameter(area)
    if bucket_of(eqdiam, geometry) != bucket:
        lo, hi = dict(geometry.buckets)[bucket]
        raise OutOfBucket(
            f"a void of {area:,} px has equivalent diameter {eqdiam:.2f} px and is filed under "
            f"{bucket} ({lo}-{hi} px). §3.2's buckets are cut on the achieved diameter, so the "
            f"draw is redrawn rather than trimmed to fit"
        )
    return mask


# --- Family (i): random-walk blobs dilated to target area ---------------------


def blob_mask(
    d: float, rng: np.random.Generator, *, margin: int, tile_px: int = TILE_PX
) -> np.ndarray:
    """§3.2 family (i) — the radar-shadow proxy, promoted from p119 unchanged in construction.

    The walk runs in an unbounded local frame; a dilated result that will not fit the inset
    interior is redrawn shorter, which is how interiority stays a *constraint* rather than a
    clip. :func:`check` still has the last word over what comes back.

    That redraw is the one plan §3.2 prices at **1.44x** in the large bucket. Re-measured here it
    is **1.158** walks per draw, and the difference is #121's own doing: the 1.44 was taken with
    ``d`` drawn up to the published 160 px top, and the top 1.86 px of that range — the part the
    plan then lowered away to 158 — is where the fits fail. Every other cell is 1.000.
    """
    target = bucket_area(d)
    interior = tile_px - 2 * margin
    steps = max(4, int(target / 3))
    for _ in range(MAX_ATTEMPTS):
        trace = _rasterize_walk(_random_walk(steps, rng))
        grown, trace, _ = _dilate_to_area(trace, target)
        height, width = grown.shape
        if height <= interior and width <= interior:
            blob = _largest_component(_trim_to_area(grown, trace, target, rng))
            canvas = np.zeros((tile_px, tile_px), dtype=bool)
            row0 = margin + int(rng.integers(0, interior - height + 1))
            col0 = margin + int(rng.integers(0, interior - width + 1))
            canvas[row0 : row0 + height, col0 : col0 + width] = blob
            return canvas
        steps = max(4, int(steps * 0.7))  # too sprawling — walk less far
    raise NotInterior(
        f"a blob of d={d:.2f} px will not fit the {interior} px interior after "
        f"{MAX_ATTEMPTS} redraws"
    )


def _random_walk(steps: int, rng: np.random.Generator) -> np.ndarray:
    """8-connected lattice random walk, as ``(steps + 1, 2)`` integer positions from the origin."""
    moves = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    picked = moves[rng.integers(0, len(moves), size=steps)]
    return np.vstack([np.zeros((1, 2), dtype=int), np.cumsum(picked, axis=0)])


def _rasterize_walk(walk: np.ndarray) -> np.ndarray:
    """The walk's visited cells, cropped to their own bounding box."""
    index = walk - walk.min(axis=0)
    trace = np.zeros(tuple(int(v) + 1 for v in index.max(axis=0)), dtype=bool)
    trace[index[:, 0], index[:, 1]] = True
    return trace


def _dilate_to_area(trace: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Grow ``trace`` to the first radius covering ``target`` px. Returns it padded to match.

    p119 dilated one step at a time and measured the area after each. Iterating scipy's default
    (4-connected) structuring element ``r`` times is exactly the taxicab distance transform
    thresholded at ``r``, so the whole sequence is available from **one** pass and the radius is
    a ``searchsorted`` on the cumulative histogram — 3.5x faster in the large bucket, and the
    same set, which `tests/test_benchmark_masks.py` asserts against the iterated reference.

    The trace comes back padded by the same radius, because the trim step measures distance
    *from the trace* on the grown array's own grid.
    """
    # A taxicab ball of radius r holds 2r^2 + 2r + 1 px, so this reach covers `target` even for
    # the degenerate single-pixel trace. Padding shorter would silently cap the radius.
    reach = math.isqrt(target // 2) + 2
    canvas = np.pad(trace, reach)
    distance = distance_transform_cdt(~canvas, metric="taxicab")
    covered = np.cumsum(np.bincount(distance.ravel()))
    radius = int(np.searchsorted(covered, target))
    if radius > reach:  # cannot happen for §3.2's buckets; a guard, not a path
        raise MaskError(f"a trace of {int(trace.sum())} px did not reach {target:,} px")
    grown = distance <= radius
    height, width = trace.shape
    return (
        grown[reach - radius : reach + height + radius, reach - radius : reach + width + radius],
        np.pad(trace, radius),
        radius,
    )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """The biggest connected piece — a void is one void, not a scatter."""
    labelled, found = label(mask)
    if found <= 1:
        return mask
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return labelled == int(sizes.argmax())


def _trim_to_area(
    mask: np.ndarray, trace: np.ndarray, target: int, rng: np.random.Generator
) -> np.ndarray:
    """Shrink ``mask`` to exactly ``target`` px by dropping the pixels furthest from ``trace``.

    Not the trimming the module docstring forbids: that one repairs a *finished* draw to make it
    pass a constraint it failed. This is how a dilated blob is brought to its drawn area at all,
    before any constraint is consulted, and :func:`check` still judges the result.

    Keeping the pixels *closest* to a connected trace keeps a sublevel set of the
    distance-to-trace function, which is connected — so the trim cannot shatter the blob. Ties on
    the threshold ring break on a seeded key, so two runs drop the same pixels.
    """
    if mask.sum() <= target:
        return mask
    distance = distance_transform_edt(~trace)
    rows, cols = np.nonzero(mask)
    keys = distance[rows, cols] + rng.random(rows.size) * 1e-6
    keep = np.argsort(keys, kind="stable")[:target]
    trimmed = np.zeros_like(mask)
    trimmed[rows[keep], cols[keep]] = True
    return trimmed


# --- Family (ii): axis-aligned rectangles -------------------------------------


def rectangle_mask(
    d: float, rng: np.random.Generator, *, margin: int, tile_px: int = TILE_PX
) -> np.ndarray:
    """§3.2 family (ii) — the sensor-dropout proxy, area-matched to family (i) through ``d``.

    Aspect ratio log-uniform in [1/2, 2]. The second side **floors**, so the rectangle lands at
    or just under the blob's pixel count and can never be the family that breaches the area cap;
    the cost is that a bottom-of-bucket draw occasionally lands a hair under the bucket floor,
    which :func:`draw` redraws. Rounding up instead was measured and is worse — it puts 5 in 300
    large-bucket draws over the cap, which is the constraint that has no margin at all.
    """
    target = bucket_area(d)
    interior = tile_px - 2 * margin
    for _ in range(MAX_ATTEMPTS):
        ratio = float(np.exp(rng.uniform(-math.log(2), math.log(2))))
        height = max(1, int(round(math.sqrt(target * ratio))))
        width = max(1, int(target // height))
        if height <= interior and width <= interior:
            canvas = np.zeros((tile_px, tile_px), dtype=bool)
            row0 = margin + int(rng.integers(0, interior - height + 1))
            col0 = margin + int(rng.integers(0, interior - width + 1))
            canvas[row0 : row0 + height, col0 : col0 + width] = True
            return canvas
    raise NotInterior(
        f"a rectangle of d={d:.2f} px will not fit the {interior} px interior after "
        f"{MAX_ATTEMPTS} draws"
    )


#: §3.2's two synthetic families, by the names :class:`benchmark.config.Masks` registers them
#: under. A table rather than a branch, so #153's real family is an entry rather than an ``elif``.
FAMILIES: Final[dict[str, Callable[..., np.ndarray]]] = {
    "blob": blob_mask,
    "rectangle": rectangle_mask,
}


# --- One draw -----------------------------------------------------------------


@dataclass(frozen=True)
class Drawn:
    """One void, and everything the committed row says about how it got there."""

    mask: np.ndarray
    #: The row's fifth column, and all :func:`draw_from_seed` needs to reproduce this mask.
    seed: int
    family: str
    bucket: str
    geometry: Masks
    #: How many whole draws the constraints cost. 1 almost always — see :data:`MAX_ATTEMPTS`.
    attempts: int

    @property
    def area_px(self) -> int:
        return int(self.mask.sum())

    @property
    def eqdiam_px(self) -> float:
        return equivalent_diameter(self.area_px)

    @property
    def mask_rle(self) -> str:
        return encode_rle(self.mask)


def draw(
    tile_id: str,
    family: str,
    bucket: str,
    *,
    geometry: Masks | None = None,
) -> Drawn:
    """The one void this (tile, family, bucket) gets — identical for every arm, seed and config."""
    geometry = Masks() if geometry is None else geometry
    return draw_from_seed(
        seed_for(tile_id, family, bucket, geometry),
        family=family,
        bucket=bucket,
        geometry=geometry,
    )


def draw_from_seed(
    seed: int,
    *,
    family: str,
    bucket: str,
    geometry: Masks | None = None,
) -> Drawn:
    """Draw from the committed seed alone, checking every constraint on every attempt.

    A violation is **retried** — a fresh diameter and a fresh shape, off the same stream, so the
    redraw is as deterministic as the draw — and after :data:`MAX_ATTEMPTS` it is **refused**.
    Nothing is trimmed to fit, which is the one repair that would leave a wrong number looking
    like a right one.
    """
    geometry = Masks() if geometry is None else geometry
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; §3.2's synthetic two are {sorted(FAMILIES)}")
    spans = dict(geometry.buckets)
    if bucket not in spans:
        named = [name for name, _ in geometry.buckets]
        raise ValueError(f"unknown bucket {bucket!r}; §3.2's three are {named}")

    lo, hi = spans[bucket]
    rng = np.random.default_rng(seed)
    refused: MaskError | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        diameter = float(rng.uniform(lo, hi))
        try:
            mask = FAMILIES[family](diameter, rng, margin=geometry.edge_margin_px)
            check(mask, bucket=bucket, geometry=geometry)
        except MaskError as exc:
            refused = exc
            continue
        return Drawn(
            mask=mask,
            seed=seed,
            family=family,
            bucket=bucket,
            geometry=geometry,
            attempts=attempt,
        )
    raise MaskError(
        f"no {family} void satisfied §3.2's constraints in the {bucket} bucket after "
        f"{MAX_ATTEMPTS} draws (last: {refused}). The boundaries are a settled record, so this "
        f"is a bug in the draw rather than an invitation to move one"
    )


# --- The eval mask list -------------------------------------------------------


@dataclass(frozen=True)
class MaskRow:
    """One committed eval instance — §3.4's five columns, in their committed order."""

    tile_id: str
    family: str
    bucket: str
    mask_rle: str
    seed: int


#: The five columns, in the order §3.4 states them — read off :class:`MaskRow` rather than
#: retyped beside it, so the row and the schema cannot disagree about what a row is.
COLUMNS: Final = tuple(f.name for f in fields(MaskRow))

#: The one column that is not text. ``mask_rle`` is decimal digits by measurement rather than by
#: default — see the module docstring — and the other three are names.
INTEGER_COLUMN: Final = "seed"


def eval_rows(tile_ids: Iterable[str], geometry: Masks | None = None) -> list[MaskRow]:
    """§3.2's synthetic half of the eval mask list: one row per (tile, family, bucket).

    Every eval tile appears once per bucket in **both** synthetic families, which is what makes
    bucket scores comparable across families — they are computed over identical tile sets. The
    order is tile, then §3.2's family order, then ascending bucket: a total order, so two runs
    that agree on the draw produce the same bytes.
    """
    geometry = Masks() if geometry is None else geometry
    rows = []
    for tile_id in sorted(tile_ids):
        for family in geometry.synthetic_families:
            for bucket, _ in geometry.buckets:
                drawn = draw(tile_id, family, bucket, geometry=geometry)
                rows.append(
                    MaskRow(
                        tile_id=tile_id,
                        family=family,
                        bucket=bucket,
                        mask_rle=drawn.mask_rle,
                        seed=drawn.seed,
                    )
                )
    return rows


def _table(rows: Sequence[MaskRow], metadata: Mapping[str, str]):
    """The five columns as an Arrow table, with the generator's provenance in the schema.

    The column types are stated once, here, and :data:`COLUMNS` is read off :class:`MaskRow` —
    #146's lesson applied to a schema rather than a metric roster: a second home for a list is
    what lets the two copies drift.
    """
    import pyarrow as pa

    schema = pa.schema(
        [(name, pa.int64() if name == INTEGER_COLUMN else pa.string()) for name in COLUMNS],
        metadata={key: str(value) for key, value in metadata.items()},
    )
    return pa.table(
        {name: [getattr(row, name) for row in rows] for name in COLUMNS}, schema=schema
    )


def _serialize(rows: Sequence[MaskRow], metadata: Mapping[str, str]) -> bytes:
    """The parquet as bytes, ready to commit.

    zstd, as the corpus's own catalog writes it: the RLE column is decimal digits and compresses
    7x, which takes the committed artifact from 22 MB to 3.1.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    sink = pa.BufferOutputStream()
    pq.write_table(_table(rows, metadata), sink, compression="zstd")
    return sink.getvalue().to_pybytes()


def _is_the_same_draw(path: Path, rows: Sequence[MaskRow], metadata: Mapping[str, str]) -> bool:
    """Does the committed file hold this draw? Read off its rows, never off its bytes.

    This is the predicate :func:`benchmark.artifacts.write_committed` takes, and the reason it
    takes one. A parquet's footer carries the writer's own version string, so byte equality —
    that guard's default, and the right answer for every artifact whose serialization is a
    function of its content — would here answer "is this the same draw, written by this
    pyarrow?", and a library upgrade would then refuse a freeze that had re-derived the identical
    30,000 masks. The draw is the five columns plus the generator and the config hash that
    produced them, and that is what is compared.
    """
    import pyarrow.parquet as pq

    committed = pq.read_table(path)
    stored = committed.schema.metadata or {}
    if {key: stored.get(key.encode(), b"").decode() for key in metadata} != dict(metadata):
        return False
    return committed.to_pydict() == {
        name: [getattr(row, name) for row in rows] for name in COLUMNS
    }


def write_eval_masks(
    rows: Sequence[MaskRow], path: Path | str, *, artifact_hash: str
) -> tuple[Path, str]:
    """Write the eval mask list, idempotently, refusing to overwrite a moved one (§3.4).

    Returns ``(path, md5)``. The guard is :func:`benchmark.artifacts.write_committed`, which
    every committed input goes through; what is local to this one is *what counts as the same
    artifact* — :func:`_is_the_same_draw` rather than the byte comparison, for the reason stated
    there.
    """
    metadata = {
        "generator": GENERATOR,
        "artifact_hash": artifact_hash,
        "rows": str(len(rows)),
        "columns": ",".join(COLUMNS),
    }
    return write_committed(
        path,
        lambda: _serialize(rows, metadata),
        what="draw",
        holds=lambda committed: _is_the_same_draw(committed, rows, metadata),
    )


@dataclass(frozen=True)
class EvalMaskList:
    """What `freeze` committed, by content — and what it cost, since #151 re-derives that line."""

    path: Path
    rows: int
    md5: str
    artifact_hash: str
    tiles: int
    seconds: float


def freeze(cfg: BenchmarkConfig) -> EvalMaskList:
    """`freeze`'s mask stage: draw §3.2's 30,000 synthetic rows off the committed test list.

    Reads the **committed file** rather than re-drawing the tiles, on the contract family's one
    rule (ADR-0005 §5): the committed file is what later phases re-read. So a `freeze` that failed
    after the tile lists landed leaves behind exactly the input this stage reproduces from.
    """
    directory = Path(cfg.paths.artifacts_dir)
    test_list = directory / TEST_TILES_NAME
    if not test_list.is_file():
        raise FileNotFoundError(
            f"{test_list} is not on this box, and §3.2 draws one eval mask per test tile. "
            f"`python -m benchmark.launch freeze` writes §3.1's lists before it reaches this "
            f"stage, so this is a partly-frozen box rather than a missing step"
        )
    tile_ids = [tile_id for tile_id, _ in read_tile_strata(test_list, what="test list")]

    started = time.perf_counter()
    rows = eval_rows(tile_ids, cfg.masks)
    path, md5 = write_eval_masks(
        rows, directory / EVAL_MASKS_NAME, artifact_hash=cfg.artifact_hash
    )
    return EvalMaskList(
        path=path,
        rows=len(rows),
        md5=md5,
        artifact_hash=cfg.artifact_hash,
        tiles=len(tile_ids),
        seconds=time.perf_counter() - started,
    )


def format_written(written: EvalMaskList, geometry: Masks) -> list[str]:
    """What `freeze` says it committed — by content, since that is what the run record quotes."""
    return [
        f"{'committed':<16} {written.rows:,} eval mask rows into {written.path}",
        f"{'':<16} {written.path.name:<21} md5 {written.md5}",
        f"{'':<16} {written.tiles:,} test tiles x {len(geometry.synthetic_families)} synthetic "
        f"families x {len(geometry.buckets)} buckets, drawn in {written.seconds:.1f} s",
        f"{'':<16} generator {GENERATOR}, artifacts {written.artifact_hash} — §3.4's one "
        f"determinism contract, and the hash the train mask bank carries",
    ]
