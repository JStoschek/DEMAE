"""§3.2's void-mask generator, the RLE codec and §3.4's two mask artifacts (#150, #151).

One generator, two committed artifacts, one determinism contract:

- **the eval mask list** — one row per scored instance over the test draw, committed as bytes;
- **the train mask bank** — K = 50 draws per head-train tile, committed as a *hash*, because
  1,000,000 rows is 104 MB and cannot go in git.

They carry the same :data:`GENERATOR` and the same generator config hash, which is what makes
that contract one contract rather than two (§3.4). What keeps their draws apart is
:data:`BANK_TAG`, and the bank half starts at :class:`BankRow`.

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
same set rather than similar ones.

That factor is worth more than it looked. Over §3.2's six cells `freeze` measures **0.947 ms**
a draw, against p121's 2.00 — which is half of why plan §7's on-the-fly line was re-derived
from 19.1 CPU-h to **8.2** (#151), the other half being a presentation count that had gone
stale for unrelated reasons. And it is what makes the bank **21.1 min** rather than 33.

The eval mask list is 27,138 draws, ~40 s, and one file: the **4,523 clean** test tiles x §3.2's
two synthetic families x three buckets, one void per tile per instance, every eval tile appearing
once per bucket in both families so bucket scores are computed over identical tile sets.

## Why 4,523 and not §3.1's 5,000

A synthetic void is placed by geometry alone — :func:`draw_from_seed` is a function of a seed and
never opens the tile — so nothing in the draw can notice that a void has landed on painted ocean
fill rather than on terrain. On the #150 draw it did: **400** of 30,000 instances held some ocean,
39 were mostly ocean, and **8** were entirely ocean. So the tile set is filtered through
:func:`benchmark.voids.eligible` (:func:`scoreable_tiles`), and the 477 part-ocean test tiles are
dropped whole.

The filter is on the **tile** and not on the placement, which is the design and not a shortcut —
:func:`scoreable_tiles` records the measurement that settled it.

## §3.3's real family, in the same five columns

#153 appends **9,273** more rows: every void SRTM actually failed to see in #127's 24 granules,
cropped out of the granule and pasted onto a stratum-matched clean test tile (:func:`transplant`,
:func:`real_rows`). The protocol is **Reuter et al. 2007's**, down to the stratum-matched
relocation, and is credited as such — the plan presented it as an invention until [#124].

**Nothing is resampled.** Below 50 degrees SRTM 1 arc-second and GLO-30 are the same lattice, so
there is no resampling step even available to get wrong; the pasted pixels are the measured
pixels, elementwise, and `tests/test_benchmark_masks.py` asserts both that identity and that no
interpolation kernel is reached for at all. What the transplant preserves is the void's **shape**
and its **band**, and explicitly not its native ruggedness — 34% of these voids are steeper than
the ruggedest tile in the whole eligible pool, so matching tighter than the band is not on offer.

**9,273 and not 9,978.** Those are §3.3's two counts and they answer different questions:
:attr:`RealFamily.rows` is the published instances the sub-rows carry (8,824 + 438 + 11), and
:attr:`RealFamily.scored_components` is the wider 9,978, which also holds the 705 components that
placed cleanly and fell in the *gaps between* sub-rows. The gaps are excluded from the rows by
the same sentence of §3.3 that names them — widening the sub-rows to cover them would destroy the
size-comparability with synthetic small that the boundaries exist for — so they are committed as
**a count** instead. Plan §3.4 and the build spec both said "9,978" for the row count until the
build wrote the file; the amendment is recorded in both, and this is the arithmetic.

Every excluded component is counted, never dropped: the 705 in the gaps, and the 182 above the
160 px top. That second number matters most, because the ceiling on large real voids is **this
benchmark's tile, not nature** — giants over 400 px exist in abundance and cannot fit a 224 px
tile under §3.2's 40% area cap. A histogram that simply stopped would imply a scarcity that is
not there.

Both families land in one artifact under **one generator and one config hash**, which is what
makes C4's ordering claim confound-free: a real row and a synthetic row differ in the shape of
the void and in nothing else about how they were produced.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final

import numpy as np
from scipy.ndimage import distance_transform_cdt, distance_transform_edt, label

from dataset.config import TILE_PX
from pretrain.artifact import read_tile_strata

from . import voids
from .artifacts import (
    EVAL_MASKS_NAME,
    MASK_BANK_SIDECAR_NAME,
    TEST_TILES_NAME,
    TILE_LIST_NAMES,
    file_md5,
    write_committed,
)
from .config import BenchmarkConfig, Masks

# The types §3.3 hands across this seam are named bare; everything that *does* something stays
# ``voids.``-qualified at the call site, so a reader can see which module is acting.
from .voids import PLAN_PREDICATE, Bands, Component, Predicate, SizeRow, VoidError

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


#: What keeps the bank's draws apart from the eval list's. The two artifacts share a generator
#: and a config hash by design (§3.4), so the *only* thing separating their streams is what goes
#: into the seed. §3.1's splits already put the two tile sets in different geocells, which means
#: a shared draw is unreachable today — the tag is here so that it stays unreachable for a
#: structural reason rather than as a consequence of a check living in another module.
BANK_TAG: Final = "bank"


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


def _check_placement(mask: np.ndarray, *, bucket: str, geometry: Masks) -> int:
    """§3.2's two *placement* constraints, and the void's area. Raises, or returns the area.

    The half of :func:`check` that is about where a void sits on a tile rather than about which
    size row it files under — which is the half #153's real family shares. It is one function
    because it is one rule: a transplanted void is held to the same margin and the same cap as a
    drawn one, and a second copy of them for the real family would be a second place for §3.2's
    numbers to move (the module docstring's whole argument, applied to a constraint instead of a
    bucket boundary).
    """
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
    return area


def check(mask: np.ndarray, *, bucket: str, geometry: Masks | None = None) -> np.ndarray:
    """Raise unless ``mask`` satisfies all three of §3.2's constraints. Returns it if it does.

    Read off the mask's own shape rather than off a configured tile size, so the answer is about
    the array in hand. Called on **every** draw (:func:`draw_from_seed`), and public so that a
    reader of the committed column can make the *other* claim — that the artifact every arm
    scores against satisfies them — against the decoded rows rather than against the generator's
    word. `tests/test_benchmark_masks.py` makes exactly that pass over all 27,138.
    """
    geometry = Masks() if geometry is None else geometry
    area = _check_placement(mask, bucket=bucket, geometry=geometry)

    eqdiam = equivalent_diameter(area)
    if bucket_of(eqdiam, geometry) != bucket:
        lo, hi = dict(geometry.buckets)[bucket]
        raise OutOfBucket(
            f"a void of {area:,} px has equivalent diameter {eqdiam:.2f} px and is filed under "
            f"{bucket} ({lo}-{hi} px). §3.2's buckets are cut on the achieved diameter, so the "
            f"draw is redrawn rather than trimmed to fit"
        )
    return mask


def check_real(mask: np.ndarray, *, row: SizeRow, geometry: Masks | None = None) -> np.ndarray:
    """The same check for a transplanted void, whose size axis is §3.3's sub-row (#153).

    Not :func:`check` with a different string: the real family carries **no bucket axis** (§3.2),
    and its size rows are cut on a top of 160 px where the synthetic large bucket stops at 158.
    Handing a real void to :func:`check` would file it against boundaries it was never drawn
    inside — the size rows are a *measurement* of what real voids are, and the 40% cap that
    lowered the synthetic top is not a constraint on a measurement.

    What is shared is what is genuinely the same rule: :func:`_check_placement`'s margin and cap.
    A transplant is held to those exactly as a draw is, because they are about the tile.
    """
    geometry = Masks() if geometry is None else geometry
    area = _check_placement(mask, bucket=row.name, geometry=geometry)

    eqdiam = equivalent_diameter(area)
    if not row.contains(eqdiam):
        raise OutOfBucket(
            f"a real void of {area:,} px has equivalent diameter {eqdiam:.2f} px and is filed "
            f"under {row.name} ({row.lo_px:g}-{row.hi_px:g} px). A transplant is cropped and "
            f"pasted, so a shape that has left its own sub-row has been resampled somewhere"
        )
    return mask


# --- Family (i): random-walk blobs dilated to target area ---------------------


def _paste_interior(
    shape: np.ndarray, rng: np.random.Generator, *, margin: int, tile_px: int
) -> np.ndarray:
    """``shape`` on an empty tile, at a seeded position inside §3.2's inset interior.

    The one place a void is put onto a tile, for all three families. It is a **paste** — an
    array written into a slice — and nothing else: no scale, no rotation, no kernel. §3.3's
    transplant rests on that being literally true (the shapes are SRTM's own pixels), and one
    home for it is what keeps the claim checkable in one place rather than in three.

    The caller has already established that ``shape`` fits; this draws where it lands.
    """
    height, width = shape.shape
    interior = tile_px - 2 * margin
    row0 = margin + int(rng.integers(0, interior - height + 1))
    col0 = margin + int(rng.integers(0, interior - width + 1))
    canvas = np.zeros((tile_px, tile_px), dtype=bool)
    canvas[row0 : row0 + height, col0 : col0 + width] = shape
    return canvas


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
            return _paste_interior(blob, rng, margin=margin, tile_px=tile_px)
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
            return _paste_interior(
                np.ones((height, width), dtype=bool), rng, margin=margin, tile_px=tile_px
            )
    raise NotInterior(
        f"a rectangle of d={d:.2f} px will not fit the {interior} px interior after "
        f"{MAX_ATTEMPTS} draws"
    )


#: §3.2's two synthetic families, by the names :class:`benchmark.config.Masks` registers them
#: under. A table rather than a branch — and §3.3's real family is deliberately **not** in it:
#: a synthetic family is a function of a diameter and an RNG, while a transplant is a function
#: of a measured shape, so the real family's entry point is :func:`transplant` rather than a
#: third callable behind a signature it cannot honour.
FAMILIES: Final[dict[str, Callable[..., np.ndarray]]] = {
    "blob": blob_mask,
    "rectangle": rectangle_mask,
}


# --- Family (iii): §3.3's real voids, cropped and pasted ----------------------


@dataclass(frozen=True)
class HostPool:
    """The clean test tiles a real void may be transplanted onto, grouped by ruggedness band.

    Three conditions, and each one is in the ticket for a reason. Hosts come from **the
    committed test list**, so a real row is scored on the same ground every synthetic row is.
    They are **clean** — :func:`benchmark.voids.eligible`, the same predicate §5 cut its band
    edges over — because a host outside that pool carries a band index meaning something
    slightly different from every other band index in the corpus. And they are **band-matched**
    to the void's own source, which is the stratum-matched relocation Reuter et al. 2007
    specifies and the plan presented as an invention until [#124].

    Matching tighter than the band is not available and the plan says so: the voids come from
    terrain with a median `slope_mean` of 31.7 degrees, 34% of them steeper than the ruggedest
    tile in the whole pool, and 156 pool tiles at or above 25 degrees would be 57x reuse.
    So the transplant preserves the void's shape and its band, and **not** its native
    ruggedness — a distinction plan §3.3 states in those words after an earlier draft claimed
    otherwise.
    """

    #: Band index (:meth:`benchmark.voids.Bands.of`) to that band's host tile ids, sorted.
    by_band: Mapping[int, tuple[str, ...]]
    #: How many test-list tiles were looked at, clean or not. Recorded so the drop is visible,
    #: and required rather than defaulted — it is quoted in :meth:`pick`'s refusal, where a
    #: silent 0 would read as an empty test list rather than as an unbanded one.
    considered: int

    @property
    def tiles(self) -> int:
        return sum(len(hosts) for hosts in self.by_band.values())

    def pick(self, band: int, rng: np.random.Generator) -> str:
        """One host from ``band``, off the row's own stream."""
        hosts = self.by_band.get(band, ())
        if not hosts:
            raise MaskError(
                f"no clean test tile sits in band R{band + 1}, and §3.3 transplants a void only "
                f"onto ground of its own stratum. The pool holds {self.tiles:,} of "
                f"{self.considered:,} listed tiles across bands "
                f"{sorted('R%d' % (b + 1) for b in self.by_band)}"
            )
        return hosts[int(rng.integers(0, len(hosts)))]

    @classmethod
    def from_test_list(
        cls, test_list: Path | str, corpus_dir: Path | str, *, bands: Bands
    ) -> HostPool:
        """Read §3.1's committed test list and band every clean tile in it off the manifest.

        The **committed file**, not the draw — the contract family's one rule (ADR-0005 §5),
        and the same read :func:`freeze` makes for the synthetic half one stage earlier.
        """
        listed = [tile_id for tile_id, _ in read_tile_strata(test_list, what="test list")]
        column = voids.read_manifest_columns(
            corpus_dir, ["tile_id", "land_frac", "water_affected", "slope_mean"]
        )
        keep = voids.eligible(column["land_frac"], column["water_affected"])
        slope = dict(zip(column["tile_id"][keep], column["slope_mean"][keep]))

        by_band: dict[int, list[str]] = {}
        for tile_id in listed:
            if tile_id in slope:
                by_band.setdefault(bands.of(float(slope[tile_id])), []).append(tile_id)
        if not by_band:
            raise ValueError(
                f"no clean tile in {test_list} — §3.3 transplants onto clean GLO-30 test tiles, "
                f"so a list whose every tile is water-affected or part-ocean has no host at all"
            )
        return cls(
            by_band={band: tuple(sorted(hosts)) for band, hosts in sorted(by_band.items())},
            considered=len(listed),
        )


@dataclass(frozen=True)
class Transplanted:
    """One real void, on the tile it was pasted onto, and the seed that put it there."""

    mask: np.ndarray
    tile_id: str
    seed: int
    row: SizeRow
    component: Component
    band: int

    @property
    def area_px(self) -> int:
        return int(self.mask.sum())

    @property
    def mask_rle(self) -> str:
        return encode_rle(self.mask)


def real_seed_for(cell: str, label: int, family: str, geometry: Masks) -> int:
    """The seed a real row is committed under — keyed on the *component*, not on a host tile.

    The synthetic seed keys on ``(tile_id, family, bucket)`` because the draw is a function of
    those three. Here the host tile is an **output**: the void exists before any tile is chosen,
    and what identifies it is the granule it was measured in and its label within that granule.
    Keying on the host instead would make the seed a function of the thing it decides.
    """
    return stable_seed(GENERATOR, geometry.seed, family, cell, label)


def transplant(
    component: Component,
    shape: np.ndarray,
    *,
    row: SizeRow,
    band: int,
    pool: HostPool,
    geometry: Masks | None = None,
    tile_px: int = TILE_PX,
) -> Transplanted:
    """§3.3's transplant: crop and paste one real void onto a stratum-matched clean test tile.

    **Nothing here scales, rotates or interpolates.** ``shape`` is written into an empty tile at
    an offset and that is the whole operation — the pasted pixels are the measured pixels,
    elementwise, which is what makes the family's claim ("the shapes are real") true rather than
    approximately true. Below 50 degrees there is no resampling step even available to get
    wrong: SRTM 1 arc-second and GLO-30 are the same lattice, which is the premise
    :func:`benchmark.voids.assert_same_lattice` refuses a source cell without.

    A shape that will not fit the interior is **refused**, not trimmed. There is no redraw
    available — the shape is a measurement, not a draw — so the two honest answers are place it
    or refuse it, and a clipped void filed under the sub-row its whole diameter fell in is the
    one repair this module exists to never make. In practice the refusal does not fire on
    §3.3's own catalog: :attr:`benchmark.voids.Component.placeable` has already been evaluated
    on every published instance, and this is the check standing behind that flag rather than
    beside it.
    """
    geometry = Masks() if geometry is None else geometry
    shape = np.asarray(shape, dtype=bool)
    height, width = shape.shape
    margin = geometry.edge_margin_px
    interior = tile_px - 2 * margin
    if height > interior or width > interior:
        raise NotInterior(
            f"{component.cell} label {component.label} is {height}x{width} px and will not fit "
            f"the {interior} px interior a void has to stay inside (§3.2). It is refused rather "
            f"than cropped: a clipped void filed under {row.name} is a void of a size it is not"
        )

    seed = real_seed_for(component.cell, component.label, geometry.real_family, geometry)
    rng = np.random.default_rng(seed)
    tile_id = pool.pick(band, rng)
    canvas = _paste_interior(shape, rng, margin=margin, tile_px=tile_px)
    check_real(canvas, row=row, geometry=geometry)
    return Transplanted(
        mask=canvas, tile_id=tile_id, seed=seed, row=row, component=component, band=band
    )


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


@dataclass(frozen=True)
class RealFamily:
    """§3.3's half of the eval mask list, and everything the transplant did not carry into it.

    The attrition fields are the ticket's fourth acceptance criterion — *counts, not silence*.
    They are here rather than in a log line because the published size histogram is drawn from
    the committed artifact, and a histogram that simply stopped at 160 px would imply a scarcity
    that is not there: giants over 400 px exist in abundance and every void-rich cell holds
    several. What excludes them is a 224 px tile under §3.2's 40% area cap — **this benchmark's
    tile geometry, not nature** — and a reader can only tell those apart if the number is stated.

    Two counts, two meanings, under §3.3's own names (:class:`benchmark.voids.Instance` makes the
    same distinction and for the same reason). :attr:`rows` is the published instances, the
    8,824 + 438 + 11 the sub-rows carry. :attr:`scored_components` is the wider 9,978, which
    also counts the components that placed cleanly and fell in the gaps *between* sub-rows.
    """

    rows: tuple[MaskRow, ...]
    #: Components at or above the floor but above the published 160 px top.
    excluded_above_top: int
    #: Components that placed cleanly and fell in one of the two sub-row gaps — §3.3's 705.
    excluded_in_gaps: int
    #: Components inside the published range that fail §3.2's placement tests.
    failed_placement: int
    #: Everything at or above the 16 px floor, before any exclusion.
    components_at_or_above_floor: int
    #: Source granules read.
    cells: int
    #: Distinct host tiles the rows landed on, and the most rows any one of them carries.
    hosts: int
    max_reuse: int
    #: What the transplant cost. Required rather than defaulted: `format_written` derives the
    #: synthetic half's time by subtracting it, so a default would silently inflate that line.
    seconds: float

    @classmethod
    def none(cls) -> RealFamily:
        """No real family in this list — every count zero, and *said* rather than defaulted.

        For an eval mask list that deliberately holds §3.2's half alone: the fixtures that drive
        `run` and the scoring seam against a synthetic-only artifact. The zeros are accurate
        there — such a list transplanted nothing and therefore excluded nothing.

        A constructor rather than a default on :func:`write_eval_masks`, because those are two
        different claims. "This list has no real family" is a thing a caller can be right about;
        "I forgot to say" is not, and a defaulted argument cannot tell them apart. `freeze`
        never reaches this — a box without the granules is refused, not written empty.
        """
        return cls(
            rows=(),
            excluded_above_top=0,
            excluded_in_gaps=0,
            failed_placement=0,
            components_at_or_above_floor=0,
            cells=0,
            hosts=0,
            max_reuse=0,
            seconds=0.0,
        )

    @property
    def scored_components(self) -> int:
        """§3.3's registered 9,978: the published instances plus the two excluded gap rows."""
        return len(self.rows) + self.excluded_in_gaps

    @property
    def counts(self) -> dict[str, str]:
        """What the committed artifact records about the family, as schema metadata."""
        return {
            "real_rows": str(len(self.rows)),
            "real_scored_components": str(self.scored_components),
            "real_excluded_above_top": str(self.excluded_above_top),
            "real_excluded_in_gaps": str(self.excluded_in_gaps),
            "real_failed_placement": str(self.failed_placement),
            "real_components_at_or_above_floor": str(self.components_at_or_above_floor),
            "real_source_cells": str(self.cells),
            "real_host_tiles": str(self.hosts),
        }


class _KeepTheGranule:
    """A ``.num`` reader that hands the array on and keeps it, so the caller can read it twice.

    :func:`benchmark.voids.extract_cell` owns the order the granule is touched in — the lattice
    assertion *before* the 13 MB read, the surface *after* the components — and that order is
    load-bearing enough that #152 wrote it down. So the read stays inside ``extract_cell`` and
    this rides along on the seam it already has, rather than the transplant reading the granule
    itself and handing in an array (which would pay the read before the assertion) or reading
    it a second time afterwards.

    Deliberately a small class and not a closure over a dict: what the transplant needs back is
    *this cell's* array, and asking for it by name — :meth:`granule` — raises when the reader
    was called for something else, where a bare dict would ``KeyError`` from a line that looks
    like a lookup rather than like a contract.
    """

    def __init__(self, read_num_fn: Callable[[str], np.ndarray]) -> None:
        self._read = read_num_fn
        self._kept: dict[str, np.ndarray] = {}

    def __call__(self, cell: str) -> np.ndarray:
        self._kept[cell] = np.asarray(self._read(cell))
        return self._kept[cell]

    def granule(self, cell: str) -> np.ndarray:
        """The array ``extract_cell`` was handed for ``cell``."""
        if cell not in self._kept:
            raise VoidError(
                f"nothing was read for {cell} — the transplant re-derives the void mask from "
                f"the array `extract_cell` was handed, so a reader that was never called for "
                f"this cell means the two are looking at different granules"
            )
        return self._kept[cell]


def real_rows(
    cells: Sequence[str],
    *,
    pool: HostPool,
    bands: Bands,
    geometry: Masks | None = None,
    predicate: Predicate = PLAN_PREDICATE,
    read_num_fn: Callable[[str], np.ndarray] | None = None,
    read_surface_fn: Callable[[str], np.ndarray] | None = None,
) -> RealFamily:
    """§3.3's real family: every published instance in ``cells``, transplanted onto a host.

    One granule at a time, and the granule is read **once** — `benchmark.voids.extract_cell`
    owns the ordering (the lattice assertion before the 13 MB read, the surface after the
    components), so the read is injected back into it through the seam it already has rather
    than done here and handed in. What this function adds is the half that needs the pixels:
    :func:`benchmark.voids.shapes` crops each published component, and :func:`transplant` pastes
    it.

    The order is cell, then label within the cell — a total order over the components, so two
    runs of the same catalog produce the same bytes. Row *n* of the file is not a function of
    row *n-1*: each host and each offset comes off that component's own keyed seed, so a cell
    that gains a void does not move every row after it.
    """
    geometry = Masks() if geometry is None else geometry
    read_num_fn = voids.read_num if read_num_fn is None else read_num_fn
    started = time.perf_counter()

    rows: list[MaskRow] = []
    used: dict[str, int] = {}
    above_top = in_gaps = failed = at_or_above_floor = 0

    for cell in sorted(cells):
        keep = _KeepTheGranule(read_num_fn)
        catalog = voids.extract_cell(
            cell,
            bands=bands,
            predicate=predicate,
            masks=geometry,
            read_num_fn=keep,
            read_surface_fn=read_surface_fn,
        )
        above_top += catalog.excluded_above_top
        in_gaps += len(catalog.gaps)
        failed += catalog.failed_placement
        at_or_above_floor += catalog.components_at_or_above_floor

        # The predicate again rather than a mask carried out of `extract_cell`: it is a
        # comparison over the array already in hand, and re-deriving it keeps the catalog's
        # return shape the one #152 committed to.
        mask = predicate.mask(keep.granule(cell), where=cell)
        shapes = voids.shapes(mask, [void.component for void in catalog.instances])
        for void in catalog.instances:
            placed = transplant(
                void.component,
                shapes[void.component.label],
                row=void.row,
                band=void.band,
                pool=pool,
                geometry=geometry,
            )
            used[placed.tile_id] = used.get(placed.tile_id, 0) + 1
            rows.append(
                MaskRow(
                    tile_id=placed.tile_id,
                    family=geometry.real_family,
                    bucket=void.size_row,
                    mask_rle=placed.mask_rle,
                    seed=placed.seed,
                )
            )

    return RealFamily(
        rows=tuple(rows),
        excluded_above_top=above_top,
        excluded_in_gaps=in_gaps,
        failed_placement=failed,
        components_at_or_above_floor=at_or_above_floor,
        cells=len(cells),
        hosts=len(used),
        max_reuse=max(used.values(), default=0),
        seconds=time.perf_counter() - started,
    )


def _schema(
    columns: Sequence[str], integers: Sequence[str], metadata: Mapping[str, str]
):
    """An Arrow schema for one of §3.4's two mask artifacts, with its provenance in the metadata.

    Both go through here. They are different shapes — the eval list's five columns and the
    bank's six — but "text unless it is one of these, plus the generator's provenance" is one
    rule, and #146's lesson is that a rule with two homes is a rule two homes can drift on.
    """
    import pyarrow as pa

    return pa.schema(
        [(name, pa.int64() if name in integers else pa.string()) for name in columns],
        metadata={key: str(value) for key, value in metadata.items()},
    )


def _columns_of(rows: Sequence, columns: Sequence[str]) -> dict[str, list]:
    """The rows as Arrow's column-per-key mapping, read off the row class rather than retyped."""
    return {name: [getattr(row, name) for row in rows] for name in columns}


def _table(rows: Sequence[MaskRow], metadata: Mapping[str, str]):
    """The eval list's five columns as an Arrow table."""
    import pyarrow as pa

    return pa.table(
        _columns_of(rows, COLUMNS), schema=_schema(COLUMNS, (INTEGER_COLUMN,), metadata)
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
    27,138 masks. The draw is the five columns plus the generator and the config hash that
    produced them, and that is what is compared.
    """
    import pyarrow.parquet as pq

    committed = pq.read_table(path)
    stored = committed.schema.metadata or {}
    if {key: stored.get(key.encode(), b"").decode() for key in metadata} != dict(metadata):
        return False
    return committed.to_pydict() == _columns_of(rows, COLUMNS)


def write_eval_masks(
    rows: Sequence[MaskRow], path: Path | str, *, artifact_hash: str, real: RealFamily
) -> tuple[Path, str]:
    """Write the eval mask list, idempotently, refusing to overwrite a moved one (§3.4).

    Returns ``(path, md5)``. The guard is :func:`benchmark.artifacts.write_committed`, which
    every committed input goes through; what is local to this one is *what counts as the same
    artifact* — :func:`_is_the_same_draw` rather than the byte comparison, for the reason stated
    there.

    §3.3's excluded counts ride in the schema metadata beside the generator, which is where the
    published histogram reads them from. They are part of the artifact's identity, not a note
    about it: a file holding the same rows under a *different* attrition is a different draw,
    and :func:`_is_the_same_draw` compares whatever is in this dict.
    """
    metadata = {
        "generator": GENERATOR,
        "artifact_hash": artifact_hash,
        "rows": str(len(rows)),
        "columns": ",".join(COLUMNS),
        "synthetic_rows": str(len(rows) - len(real.rows)),
        **real.counts,
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
    #: Test tiles a synthetic void was actually drawn on — :func:`scoreable_tiles`' output.
    tiles: int
    #: Test tiles the committed list named, clean or not. Recorded rather than differenced at
    #: the print, on §3.3's convention: what a stage excluded is a count, never a silence.
    listed_tiles: int
    seconds: float
    real: RealFamily


def real_family(cfg: BenchmarkConfig, pool: HostPool, bands: Bands) -> RealFamily:
    """§3.3's half of the list, off this box's fetched granules.

    Refused rather than skipped when the granules are not here. A `freeze` that wrote the
    synthetic 27,138 and quietly left the real family out would produce a perfectly well-formed
    eval mask list, of a benchmark missing the arm C4 is claimed on — the same failure
    `benchmark.launch` refuses one stage up when it will not let a partial freeze exit 0.
    """
    granule_dir = voids.default_granule_dir(cfg.paths)
    source_cells = voids.cells(granule_dir)
    if not source_cells:
        raise FileNotFoundError(
            f"no SRTMGL1N granule in {granule_dir}, and §3.3's real family is transplanted from "
            f"their void shapes. `python scripts/p127_fetch_srtm_voids.py` fetches the 24 this "
            f"box needs; a freeze that skipped the family would commit a well-formed eval mask "
            f"list of a benchmark with no C4 arm in it"
        )
    return real_rows(
        source_cells,
        pool=pool,
        bands=bands,
        geometry=cfg.masks,
        read_num_fn=lambda cell: voids.read_num(cell, granule_dir=granule_dir),
    )


def scoreable_tiles(tile_ids: Iterable[str], corpus_dir: Path | str) -> list[str]:
    """The listed test tiles §3.2 may draw a synthetic void on: the ones that are clean ground.

    :func:`benchmark.voids.eligible` — all-land, water flag clear — which is the predicate §5
    cuts its band edges over and the one §3.3's :class:`HostPool` already transplants onto. This
    is the synthetic half applying it too, and until it did the two halves of one artifact were
    drawn over two different populations: 5,000 tiles for the synthetic rows, the 4,523 clean
    ones for the real rows. The module docstring's claim that a real row and a synthetic row
    "differ in the shape of the void and in nothing else about how they were produced" is what
    that divergence quietly cost, and what this restores.

    ## What it is fixing

    A synthetic void is placed by geometry alone. :func:`draw_from_seed` takes a seed and returns
    a mask; it never opens the tile, and neither :func:`check` nor anything downstream of it can
    see that the pixels underneath are Copernicus's painted ocean fill rather than terrain. On
    the #150 draw, **400** of 30,000 instances held some ocean, **39** were mostly ocean, and
    **8** were entirely ocean.

    The 8 are the ones that break something. ADR-0001 §3 masks the fill *before* differencing, so
    a void with no land under it leaves ``slope_rmse``, ``slope_rmse_eroded``, ``seam_gap``,
    ``aspect_error`` and ``profile_curvature`` undefined — and one NaN is enough to take a whole
    cell of the results table with it, because the table means these columns. The elevation
    columns are quieter and not worse: they stay finite and read ~0 m of error for having
    interpolated ocean fill across ocean fill. Measured with A1 the 8 score 0.025 m against a
    population mean of 14.4 m, which moves a cell mean by ~0.01 m. That is a rounding error on
    the number, and the reason to care is not its size — those instances are not measuring
    anything, and every arm is flattered by them about equally, so what they cost is meaning
    rather than rank.

    ## Why the tile and not the placement

    A land predicate inside :func:`draw_from_seed`'s retry loop is the other repair, and it was
    measured before this one was written. Over the affected tiles' 672 cells it refuses **152**
    (22.6%) after :data:`MAX_ATTEMPTS`, **114** of them in the large bucket, because a 158 px
    void does not fit the land of a 26%-land tile at any placement at all. That does not fail
    loudly — it removes the large bucket from precisely the coastal tiles it keeps in the other
    two, which is a terrain-composition difference between buckets, in the axis C1 and C4 read,
    introduced by the repair. It also breaks :func:`eval_rows`' one-row-per-tile-per-bucket
    balance, ends the property that a row is reproducible from its own seed (the mask would
    become a function of the corpus), and moves the attempt counts :data:`MAX_ATTEMPTS`
    documents from 1.000 to 2.898.

    Dropping the tile costs 477 of 5,000 and keeps every one of those properties. It removes 389
    of the 400, **all 8** entirely-ocean and **all 39** mostly-ocean instances; the 11 that
    remain carry between 0.005% and 0.084% ocean, one to forty-two pixels, and produce no NaN in
    any column.
    """
    listed = [str(tile_id) for tile_id in tile_ids]
    column = voids.read_manifest_columns(
        corpus_dir, ["tile_id", "land_frac", "water_affected"]
    )
    keep = voids.eligible(column["land_frac"], column["water_affected"])
    clean = set(column["tile_id"][keep].astype(str))
    scoreable = [tile_id for tile_id in listed if tile_id in clean]
    if not scoreable:
        raise ValueError(
            f"none of the {len(listed):,} listed test tiles is clean ground, and §3.2 draws a "
            f"synthetic void only where the fill it is scored against is terrain. A list whose "
            f"every tile is water-affected or part-ocean has nothing to draw on — the same "
            f"refusal `HostPool.from_test_list` makes about the same population"
        )
    return scoreable


def freeze(cfg: BenchmarkConfig, *, real: RealFamily | None = None) -> EvalMaskList:
    """`freeze`'s mask stage: §3.2's 27,138 synthetic rows and §3.3's real family, in one file.

    Reads the **committed file** rather than re-drawing the tiles, on the contract family's one
    rule (ADR-0005 §5): the committed file is what later phases re-read. So a `freeze` that failed
    after the tile lists landed leaves behind exactly the input this stage reproduces from. The
    real family's host pool is read off that same committed list, for the same reason.

    Both families land in **one artifact under one generator and one config hash** (#153's fifth
    acceptance criterion), which is what makes C4's ordering claim confound-free by construction:
    a real row and a synthetic row differ in the shape of the void and in nothing else about how
    they were produced.

    Both halves are drawn over the **same** population, which is :func:`scoreable_tiles`'
    subject: the committed test list filtered through :func:`benchmark.voids.eligible`. That is
    what makes the confound-free claim above true of the tile as well as of the generator — the
    real family was already restricted to clean hosts by :meth:`HostPool.from_test_list`, and
    for a stretch the synthetic half was not.

    ``real`` is the seam the artifact-mechanics tests drive: passing a family skips the granule
    read, which is the only part of this stage that needs 24 granules and a COG per cell.
    """
    directory = Path(cfg.paths.artifacts_dir)
    test_list = directory / TEST_TILES_NAME
    if not test_list.is_file():
        raise FileNotFoundError(
            f"{test_list} is not on this box, and §3.2 draws one eval mask per test tile. "
            f"`python -m benchmark.launch freeze` writes §3.1's lists before it reaches this "
            f"stage, so this is a partly-frozen box rather than a missing step"
        )
    listed = [tile_id for tile_id, _ in read_tile_strata(test_list, what="test list")]
    tile_ids = scoreable_tiles(listed, cfg.paths.corpus_dir)

    started = time.perf_counter()
    rows = eval_rows(tile_ids, cfg.masks)
    if real is None:
        bands = Bands.from_manifest(cfg.paths.corpus_dir)
        pool = HostPool.from_test_list(test_list, cfg.paths.corpus_dir, bands=bands)
        real = real_family(cfg, pool, bands)
    path, md5 = write_eval_masks(
        [*rows, *real.rows],
        directory / EVAL_MASKS_NAME,
        artifact_hash=cfg.artifact_hash,
        real=real,
    )
    return EvalMaskList(
        path=path,
        rows=len(rows) + len(real.rows),
        md5=md5,
        artifact_hash=cfg.artifact_hash,
        tiles=len(tile_ids),
        listed_tiles=len(listed),
        seconds=time.perf_counter() - started,
        real=real,
    )


def format_written(written: EvalMaskList, geometry: Masks) -> list[str]:
    """What `freeze` says it committed — by content, since that is what the run record quotes.

    The excluded counts are printed, not just committed. §3.3's histogram is only readable
    beside them: what the 224 px tile cut off at the top, and what the sub-row boundaries cut
    out of the middle to keep real-small size-comparable to synthetic small. The synthetic
    half's own exclusion is on the same line for the same reason — a reader who sees 27,138
    where §3.1 drew 5,000 test tiles should not have to divide to find out why.
    """
    real = written.real
    synthetic = written.rows - len(real.rows)
    return [
        f"{'committed':<16} {written.rows:,} eval mask rows into {written.path}",
        f"{'':<16} {written.path.name:<21} md5 {written.md5}",
        f"{'':<16} {synthetic:,} synthetic — {written.tiles:,} test tiles x "
        f"{len(geometry.synthetic_families)} families x {len(geometry.buckets)} buckets, drawn "
        f"in {written.seconds - real.seconds:.1f} s",
        f"{'':<16} {written.listed_tiles - written.tiles:,} of {written.listed_tiles:,} listed "
        f"test tiles excluded as part-ocean or water-affected — a synthetic void is drawn on "
        f"the clean ground §3.3's hosts come from, never on painted fill",
        f"{'':<16} {len(real.rows):,} real — transplanted from {real.cells} granules onto "
        f"{real.hosts:,} stratum-matched clean test tiles (at most {real.max_reuse} each), "
        f"cropped and pasted in {real.seconds:.1f} s",
        f"{'':<16} {real.scored_components:,} scored components, of which "
        f"{real.excluded_in_gaps:,} fall in the sub-row gaps and are excluded; "
        f"{real.excluded_above_top:,} more sit above the {geometry.real_band_top_px} px top, "
        f"cut by this benchmark's tile geometry rather than by scarcity",
        f"{'':<16} generator {GENERATOR}, artifacts {written.artifact_hash} — §3.4's one "
        f"determinism contract, carried by both families and by the train mask bank",
    ]


# --- The train mask bank ------------------------------------------------------
#
# §3.4's second mask artifact, and the only committed input that is **not committed as bytes**.
# K = 50 draws per head-train tile is 1,000,000 rows and 104 MB, which cannot go in git on the
# same rule the corpus's shards follow — so what git carries is a sidecar holding the bank's md5
# and the generator config hash, and `run` refuses a bank whose md5 has moved rather than reading
# it. The determinism contract survives intact: the mask sequence a run sees is still a committed
# input, its *hash* rather than its bytes.
#
# Everything here is streamed. The eval list materializes its 27,138 rows and compares them to
# what is committed (:func:`_is_the_same_draw`); at 1,000,000 rows of RLE text that would be
# several GB of Python objects, so the bank is written a row group at a time and identified by a
# rolling hash over the rows instead.


@dataclass(frozen=True)
class BankRow:
    """One banked training void — §3.4's six columns, in their committed order."""

    tile_id: str
    #: Which of the K draws this is, 0-based. §4.1 reads epoch ``e`` of 50 and §4.2 the first 20,
    #: so the column is an index into the sequence rather than a label on it.
    epoch: int
    family: str
    bucket: str
    mask_rle: str
    seed: int


#: The six columns, in the order §3.4 states them — read off :class:`BankRow` for the reason
#: :data:`COLUMNS` is read off :class:`MaskRow`.
BANK_COLUMNS: Final = tuple(f.name for f in fields(BankRow))

#: The two that are not text. ``epoch`` is an index and ``seed`` is the 63-bit draw key.
BANK_INTEGER_COLUMNS: Final = ("epoch", "seed")

#: Rows per parquet row group. 20,000 rows is ~15 MB of RLE — big enough that the footer's
#: per-group overhead is noise over 50 groups, small enough that the writer's working set stays
#: a fraction of the 104 MB artifact.
BANK_ROW_GROUP: Final = 20_000


def bank_seed_for(tile_id: str, epoch: int, family: str, bucket: str, geometry: Masks) -> int:
    """The seed one banked draw is committed under — the row's sixth column.

    ``epoch`` is in the key, which is what makes the K rows for a tile K *different* voids
    rather than one void written K times. :data:`BANK_TAG` is in it for the reason stated there.
    """
    return stable_seed(GENERATOR, geometry.seed, BANK_TAG, tile_id, epoch, family, bucket)


def bank_cells(tile_id: str, geometry: Masks) -> list[tuple[str, str]]:
    """Which (family, bucket) each of this tile's K epochs draws from.

    **Apportioned, not sampled.** The K draws are dealt over §3.2's six cells as evenly as they
    divide — eight or nine each at K = 50 — and only then ordered. Sampling instead would leave
    a tile whose fifty draws happened to hold fourteen large blobs and two small rectangles, and
    a head trained on that tile has seen a different curriculum from its neighbour for no reason
    anything downstream records. §3.1's tile lists are apportioned for the same reason.

    The order is per **tile**, not per epoch. A bank that dealt the cells in a fixed order would
    make epoch 0 blob/small for every tile at once, so each epoch would train on one geometry —
    a curriculum nobody chose, and one the epoch column makes invisible.

    Which cells carry the remainder is drawn per tile too, so the imbalance K leaves over does
    not land on the same two cells 20,000 times.
    """
    cells = [
        (family, bucket)
        for family in geometry.synthetic_families
        for bucket, _ in geometry.buckets
    ]
    draws = geometry.bank_draws_per_tile
    rng = np.random.default_rng(
        stable_seed(GENERATOR, geometry.seed, BANK_TAG, "cells", tile_id)
    )
    base, extra = divmod(draws, len(cells))
    counts = np.full(len(cells), base, dtype=int)
    counts[rng.permutation(len(cells))[:extra]] += 1
    dealt = [cell for cell, count in zip(cells, counts) for _ in range(count)]
    return [dealt[position] for position in rng.permutation(draws)]


def bank_rows(tile_ids: Iterable[str], geometry: Masks | None = None) -> Iterator[BankRow]:
    """§3.2's K draws per head-train tile, streamed in the order they are committed in.

    A generator rather than a list: the real bank is 1,000,000 rows of RLE text, and the caller
    that writes it never holds more than one row group of them (:func:`write_bank`). The order
    is tile then epoch — a total order, so two runs that agree on the draw produce the same rows
    and the same hash whatever they were iterated on.
    """
    geometry = Masks() if geometry is None else geometry
    for tile_id in sorted(tile_ids):
        for epoch, (family, bucket) in enumerate(bank_cells(tile_id, geometry)):
            seed = bank_seed_for(tile_id, epoch, family, bucket, geometry)
            drawn = draw_from_seed(seed, family=family, bucket=bucket, geometry=geometry)
            yield BankRow(
                tile_id=tile_id,
                epoch=epoch,
                family=family,
                bucket=bucket,
                mask_rle=drawn.mask_rle,
                seed=seed,
            )


def _batched(rows: Iterator[BankRow], size: int) -> Iterator[list[BankRow]]:
    """``itertools.batched``, which this project's floor of Python 3.11 does not have."""
    batch: list[BankRow] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


#: Bytes of BLAKE2b in the draw hash. 16 is the width :mod:`benchmark.config` uses for a config
#: hash, and this is the same kind of claim about the same kind of thing.
DRAW_HASH_BYTES: Final = 16


def _digest_line(row: BankRow) -> bytes:
    """One row, as the draw hash sees it: every committed column, in the committed order."""
    return ("|".join(str(getattr(row, name)) for name in BANK_COLUMNS) + "\n").encode()


@dataclass(frozen=True)
class WrittenBank:
    """A bank on disk, by content and by cost."""

    path: Path
    rows: int
    #: What it takes on this box. Not a property of the draw — see :class:`BankSidecar`.
    size_bytes: int
    md5: str
    #: BLAKE2b over the rows themselves, and therefore the same number under any parquet writer.
    draw_hash: str
    seconds: float


def write_bank(
    rows: Iterator[BankRow],
    path: Path | str,
    *,
    metadata: Mapping[str, str],
    expect_draw_hash: str | None = None,
) -> WrittenBank:
    """Stream the bank to ``path``, hashing the rows as they go.

    Written to a neighbouring ``.partial`` and moved into place at the end, so a freeze that is
    interrupted — or one that draws something ``expect_draw_hash`` refuses — leaves the bank a
    box already has exactly where it was. Twenty minutes of generation is long enough that being
    interrupted partway is a real state rather than a theoretical one, and a half-written parquet
    at ``paths.mask_bank_path`` is a file `run` would resolve, hash and refuse for the wrong
    reason.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    schema = _schema(BANK_COLUMNS, BANK_INTEGER_COLUMNS, metadata)

    digest = hashlib.blake2b(digest_size=DRAW_HASH_BYTES)
    written = 0
    started = time.perf_counter()
    try:
        with pq.ParquetWriter(partial, schema, compression="zstd") as writer:
            for batch in _batched(rows, BANK_ROW_GROUP):
                for row in batch:
                    digest.update(_digest_line(row))
                writer.write_table(pa.table(_columns_of(batch, BANK_COLUMNS), schema=schema))
                written += len(batch)
        seconds = time.perf_counter() - started
        draw_hash = digest.hexdigest()
        if expect_draw_hash is not None and draw_hash != expect_draw_hash:
            raise FileExistsError(
                f"{MASK_BANK_SIDECAR_NAME} commits draw hash {expect_draw_hash} and this config "
                f"draws {draw_hash} over the same {written:,} rows. The shape of the draw "
                f"agreed, so what moved is the generator itself rather than a knob — which §3.4 "
                f"does not let a `freeze` overwrite. Delete the sidecar deliberately if the draw "
                f"really has changed, remembering that every result already trained against it "
                f"quotes the old md5"
            )
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, path)
    return WrittenBank(
        path=path,
        rows=written,
        size_bytes=path.stat().st_size,
        md5=file_md5(path),
        draw_hash=draw_hash,
        seconds=seconds,
    )


# --- The committed sidecar ----------------------------------------------------


@dataclass(frozen=True)
class BankDraw:
    """What the bank *is*, before a single mask of it exists.

    Every field here is a function of the config and the committed head-train list, so two boxes
    freezing one config agree on all six without either having drawn anything. That is what makes
    it the thing §3.4's guard reads: the refusal has to arrive before the twenty minutes, not
    after (:func:`_refuse_a_moved_draw`).

    Deliberately **not** here: ``md5`` and ``size_bytes``, which describe *this box's encoding* of
    the draw rather than the draw. A parquet footer carries the writing library's own version
    string, so a guard that compared those would refuse a re-derivation of the identical
    1,000,000 masks the day pyarrow was upgraded — #150's lesson one level up. Nor ``draw_hash``,
    which is the draw exactly but is only knowable once it has been paid for.
    """

    generator: str
    artifact_hash: str
    rows: int
    tiles: int
    draws_per_tile: int
    columns: str

    @classmethod
    def of(cls, cfg: BenchmarkConfig, *, tiles: int) -> BankDraw:
        """What this config's bank would be, over a head-train list of ``tiles`` tiles."""
        return cls(
            generator=GENERATOR,
            artifact_hash=cfg.artifact_hash,
            rows=tiles * cfg.masks.bank_draws_per_tile,
            tiles=tiles,
            draws_per_tile=cfg.masks.bank_draws_per_tile,
            columns=",".join(BANK_COLUMNS),
        )

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class BankSidecar:
    """What git carries about a bank it does not carry: the draw, and this box's encoding of it.

    The split is the point. :class:`BankDraw` and ``draw_hash`` are the draw — identical on every
    box that freezes this config, and what §3.4's guard is about. ``md5`` and ``size_bytes`` are
    what the draw came out as *here*, which is what `run` checks the file against and what a run
    record quotes.

    **Why this is not written through** :func:`benchmark.artifacts.write_committed`, which every
    other committed input goes through: that function's contract is *this file already holds what
    I would write, or it is a refusal*, and two of these fields are allowed to move without the
    artifact having changed. A re-freeze under a newer pyarrow re-derives the same 1,000,000
    masks at a different md5, and the right outcome is to record the new md5, not to refuse and
    not to leave the old one pointing at a bank that no longer hashes to it. So the refusal is
    kept — in §3.4's own words, and reading the draw rather than the file — while the write is
    local. It is the same split ``write_committed`` documents, at the one artifact where *what
    counts as the same artifact* also decides what counts as a legitimate rewrite.
    """

    draw: BankDraw
    draw_hash: str
    md5: str
    size_bytes: int

    @property
    def megabytes(self) -> float:
        """What `freeze` prints. MB rather than bytes because 103,653,977 is not a size."""
        return self.size_bytes / 1e6

    def to_dict(self) -> dict:
        """Flat JSON: the draw's six fields beside the three this box measured.

        Flat rather than nested because a human comparing a sidecar to what `freeze` printed
        should not have to know which half a field lives in.
        """
        return {
            **self.draw.to_dict(),
            "draw_hash": self.draw_hash,
            "md5": self.md5,
            "size_bytes": self.size_bytes,
        }

    def render(self) -> str:
        """The committed bytes: sorted keys and a trailing newline, as the extractor spec is."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"


def read_sidecar(path: Path | str) -> BankSidecar:
    """Read the committed sidecar, refusing anything that is not one.

    Every way the file can fail to be a sidecar is a refusal rather than a partial read, on the
    rule the RLE codec follows: a bank resolved against half a sidecar is a bank nobody checked.

    Absence is separated out from unreadability because they call for opposite things. No
    sidecar is the ordinary pre-`freeze` state and the fix is to freeze; a sidecar that will not
    parse is a committed file that has been damaged, and re-freezing over it is the one thing
    §3.4 does not want done by reflex.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not on this box, and it is what says which bank this is. The bank "
            f"itself is deliberately not in git (§3.4) but this file is — "
            f"`python -m benchmark.launch freeze` writes both"
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not a readable {MASK_BANK_SIDECAR_NAME}: {exc}") from None
    drawn = [f.name for f in fields(BankDraw)]
    measured = ["draw_hash", "md5", "size_bytes"]
    missing = sorted(set(drawn + measured) - set(payload))
    if missing:
        raise ValueError(f"{path} is missing sidecar field(s) {missing}")
    return BankSidecar(
        draw=BankDraw(**{name: payload[name] for name in drawn}),
        **{name: payload[name] for name in measured},
    )


def _refuse_a_moved_draw(path: Path, committed: BankSidecar, drawn: BankDraw) -> None:
    """§3.4's refusal, before the generation rather than after it.

    Named field by field, because the person reading this is being told to delete a committed
    file and whether that is the right thing to do is a question about *what* moved: a bank
    redrawn under a new geometry is a different benchmark, and a bank whose tile count moved is
    usually a head-train list that was redrawn underneath it.
    """
    was = committed.draw.to_dict()
    detail = ", ".join(
        f"{name} {was[name]!r} -> {value!r}"
        for name, value in drawn.to_dict().items()
        if was[name] != value
    )
    if not detail:
        return
    raise FileExistsError(
        f"{path} is already frozen at md5 {committed.md5}, describing a different draw from the "
        f"one this config produces ({detail}). `freeze` does not overwrite a committed input "
        f"whose content hash has moved (§3.4); delete it deliberately if the draw really has "
        f"changed, remembering that every result already trained against it quotes the old md5"
    )


@dataclass(frozen=True)
class TrainMaskBank:
    """What `freeze` left on this box, and whether it had to draw it.

    Holds the sidecar rather than restating it, so there is one description of the bank and the
    committed file is it — the alternative was this class copying nine fields out of that one and
    the two drifting on the tenth.
    """

    path: Path
    sidecar_path: Path
    #: What the sidecar says — the draw, its hash, and this box's md5 and size. Held rather than
    #: copied out field by field, so there is one description of the bank and the committed file
    #: is it.
    committed: BankSidecar
    seconds: float
    #: ``False`` when the committed sidecar already described the bank on disk. Idempotence here
    #: has to mean *not spending the twenty minutes again*, so it is reported rather than implied.
    generated: bool


def freeze_bank(cfg: BenchmarkConfig) -> TrainMaskBank:
    """`freeze`'s bank stage: K draws per head-train tile, and the sidecar git keeps instead.

    Reads the **committed 20k list** rather than the one ``--size`` names, and rather than
    re-drawing the tiles. Both for the same reason: the bank is an input every cell shares
    (§3.4), the label-efficiency rungs are nested prefixes of the one committed draw (§3.1), and
    a bank that moved with ``--size`` would be a second determinism contract.

    Three outcomes, and the order they are decided in is what keeps the expensive one last:

    *The committed sidecar describes the bank on disk.* Nothing is drawn and nothing is written.

    *The committed sidecar describes a different draw.* Refused, having read one small JSON file
    — not after twenty minutes of generation (:func:`_refuse_a_moved_draw`).

    *There is no sidecar, or the bank is gone or is not the one the sidecar describes.* Drawn,
    checked against the committed draw hash if there is one, and the sidecar written with the
    md5 the bank came out as here.
    """
    directory = Path(cfg.paths.artifacts_dir)
    head_train = directory / TILE_LIST_NAMES["20k"]
    if not head_train.is_file():
        raise FileNotFoundError(
            f"{head_train} is not on this box, and §3.2 banks K draws per head-train tile. "
            f"`python -m benchmark.launch freeze` writes §3.1's lists before it reaches this "
            f"stage, so this is a partly-frozen box rather than a missing step"
        )
    tile_ids = [tile_id for tile_id, _ in read_tile_strata(head_train, what="head-train list")]

    bank_path = Path(cfg.paths.mask_bank_path)
    sidecar_path = directory / MASK_BANK_SIDECAR_NAME
    drawing = BankDraw.of(cfg, tiles=len(tile_ids))

    committed = read_sidecar(sidecar_path) if sidecar_path.is_file() else None
    if committed is not None:
        _refuse_a_moved_draw(sidecar_path, committed, drawing)
        if bank_path.is_file() and file_md5(bank_path) == committed.md5:
            return TrainMaskBank(
                path=bank_path,
                sidecar_path=sidecar_path,
                committed=committed,
                seconds=0.0,
                generated=False,
            )

    written = write_bank(
        bank_rows(tile_ids, cfg.masks),
        bank_path,
        metadata=drawing.to_dict(),
        expect_draw_hash=None if committed is None else committed.draw_hash,
    )
    sidecar = BankSidecar(
        draw=drawing,
        draw_hash=written.draw_hash,
        md5=written.md5,
        size_bytes=written.size_bytes,
    )
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(sidecar.render())
    return TrainMaskBank(
        path=written.path,
        sidecar_path=sidecar_path,
        committed=sidecar,
        seconds=written.seconds,
        generated=True,
    )


def assert_bank_matches(cfg: BenchmarkConfig) -> str:
    """What `run` does before it spends anything: refuse a bank whose md5 has moved (§3.4).

    Returns the bank's md5, which is what the run record quotes it by.

    A **refusal** rather than a warning, and before the encoder is loaded, because every way
    this can be wrong is silent: a head trained against a re-drawn bank converges normally and
    prints a table exactly as wide as a good one. The generator config hash is checked beside
    the md5 so that the mid-change state — a sidecar that has moved with the old bank still on
    disk — is caught by the thing that moved rather than by waiting for the redraw.

    It hashes the bank itself rather than reading the md5
    :func:`benchmark.artifacts.resolve` computed moments earlier, which is a second pass over
    104 MB and deliberate. A safety check whose correctness depends on being handed the right
    snapshot is one a future caller can hold wrong; this one takes a config and answers about
    the disk. The cost is ~0.2 s, once, ahead of a run measured in minutes.
    """
    sidecar_path = Path(cfg.paths.artifacts_dir) / MASK_BANK_SIDECAR_NAME
    bank_path = Path(cfg.paths.mask_bank_path)
    committed = read_sidecar(sidecar_path)

    drawn = committed.draw
    if drawn.generator != GENERATOR or drawn.artifact_hash != cfg.artifact_hash:
        raise ValueError(
            f"{sidecar_path} describes a bank drawn by {drawn.generator} under artifact_hash "
            f"{drawn.artifact_hash}; this run is {GENERATOR} under {cfg.artifact_hash}. The "
            f"two mask artifacts carry one generator config hash (§3.4), so a run cannot be "
            f"trained against this bank and scored against this eval mask list"
        )

    observed = file_md5(bank_path)
    if observed is None:
        raise FileNotFoundError(
            f"{bank_path} is not on this box. The bank is {drawn.rows:,} rows and is "
            f"deliberately not in git (§3.4) — `python -m benchmark.launch freeze` regenerates "
            f"it from the committed sidecar's draw, which is what the sidecar is for"
        )
    if observed != committed.md5:
        raise ValueError(
            f"{bank_path} has md5 {observed}; {sidecar_path.name} commits {committed.md5}. "
            f"§3.4 refuses a bank whose md5 has moved rather than reading it: the mask sequence "
            f"a run sees is a committed input, and a head trained against a re-drawn bank "
            f"converges normally and prints a table exactly as wide as a good one. Delete the "
            f"bank and re-freeze to redraw it from the committed sidecar"
        )
    return observed


def format_bank_written(bank: TrainMaskBank) -> list[str]:
    """What `freeze` says it committed — and what it cost, since #151 re-derives that line.

    The row count, the byte size and the generation time are printed because plan §7's 1M /
    33 min / 740 MB were derived on the p121 prototype against a pass count §2.3 has since
    rewritten. A figure a command prints every time it runs does not go stale in a document.
    """
    committed, draw = bank.committed, bank.committed.draw
    # Minutes past two of them: the real artifact takes a third of an hour, and "1,265.4 s" is a
    # number a reader has to divide before it means anything.
    took = (
        f"{bank.seconds:.1f} s" if bank.seconds < 120 else f"{bank.seconds / 60:.1f} min"
    )
    drawn = f"drawn in {took}" if bank.generated else "already frozen — nothing was redrawn"
    return [
        f"{'committed':<16} {draw.rows:,} train mask bank rows into {bank.path}",
        f"{'':<16} {bank.path.name:<21} md5 {committed.md5}",
        f"{'':<16} {draw.tiles:,} head-train tiles x K = {draw.draws_per_tile}, "
        f"{committed.megabytes:,.0f} MB, {drawn}",
        f"{'':<16} {bank.sidecar_path.name:<21} draw {committed.draw_hash} — the bank is not in "
        f"git (§3.4); this sidecar is, and `run` refuses a bank whose md5 has moved",
        f"{'':<16} generator {GENERATOR}, artifacts {draw.artifact_hash} — the same contract "
        f"the eval mask list carries",
    ]
