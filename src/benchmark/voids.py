"""Real SRTM void geometry, and the band it is transplanted under (plan §3.3, #152).

The real-void family's whole claim is that the *shapes* are real. This module is where that
claim is made good: it reads the 24 fetched SRTMGL1N granules, selects the void pixels, labels
the connected components, buckets them into §3.3's size sub-rows, and reads each one's
ruggedness band off **GLO-30 at the void's own coordinates**. Promoted from
``prototypes/p135_void_ruggedness_proto.py``, which measured all of it, and from
``scripts/p127_fetch_srtm_voids.py``, which put the granules on disk and stays a script.

Four things here are load-bearing, and each has a way of coming back wrong.

**The predicate is `num > 10 AND num < 201`, and the rule it replaced yields nothing.**
SRTM-observed posts carry the swath count 201-224, water is 1/2/5, and land values above 10
but below 201 are fill from some other source, i.e. void. Plan §3.3 originally carried "pixels
with scene count 0 are true voids", which is a rule about a *different product* —
:data:`REFUTED_ZERO_SCENE_PREDICATE` — and **there is no value 0 in SRTMGL1N's class table at
all** (SRTM Collection User Guide V3, Table 2, p. 6). Shipping it produces an empty mask, and
an empty mask is not something anything downstream would notice: a family with no instances
prints as a family with no instances. So :meth:`Predicate.mask` refuses one, by name.

**The band comes from GLO-30 and never from the SRTM surround.** Matching a void to a
ruggedness stratum needs the terrain *under* the void, which SRTM by definition never observed
there. Inferring it from the observed pixels around the hole measures the wrong terrain and
biases systematically flat, because radar shadow occurs exactly where the surroundings are
tamer than the void. #126 found that circularity and fixed it; the defence that keeps it fixed
is structural rather than a comment — :func:`band_for` takes the GLO-30 surface and the
component's own geometry, and there is no parameter a ``.num`` raster could arrive through.
The SRTM granule contributes the void's shape and nothing else.

**The same-lattice premise is asserted, not discovered.** The transplant crops and pastes with
no resampling, which is true only where SRTM 1 arc-second and GLO-30 sample the same lattice.
GLO-30 widens its longitude spacing poleward, so the premise fails *silently* at 50 degrees —
the arrays still have shapes that broadcast. :func:`assert_same_lattice` reads the repo's own
band table (:data:`dataset.geometry.COLS_PER_DEGREE`) rather than a second copy of the rule,
and it runs before a single byte is read.

**The size sub-rows do not tile the range.** Real-small (16-48 px) is the headline row and its
honest comparison partner is synthetic *small* in the same band; real-medium+ (64-96 px) is
reported separately so the headline stays size-comparable; 128-160 px is **an instance count
with no metric attached**, because a mean over eleven instances invites a significance the
number cannot carry. The two gaps between those rows hold 7.1% of a real draw and are
**excluded, not silently dropped** — :func:`summarise` states their counts beside the scored
ones, and widening the rows to cover them would destroy the comparability they exist for.

Reading is injected throughout, so the tests drive this without a network or an Earthdata
account — ``tests/test_p127_fetch_srtm_voids.py``'s rule for the fetch path it tests without
one.
"""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Iterable, Sequence

import numpy as np

from dataset import config as dataset_config
from dataset.geometry import (
    COLS_PER_DEGREE,
    ROWS_PER_DEGREE,
    lat_band,
    pixel_spacing_m,
    window_center_lonlat,
)
from dataset.metrics import slope_mean as slope_mean_of

from .config import Masks, Paths

# --- The product, as its user guide describes it ------------------------------

#: SRTMGL1N granules are 3601x3601 uint8, with a one-post overlap onto each neighbour.
SRTM_PX: Final = 3601
#: GLO-30 below 50 degrees is 3600x3600 on the *same* lattice (P125 §10.1), so SRTM post
#: ``(r, c)`` for ``r, c < 3600`` is GLO-30 post ``(r, c)``. The relation — one fewer post per
#: degree, from the overlap — is what :func:`extract_cell` checks; the constants are what the
#: readers check.
GLO_PX: Final = 3600
#: One byte per post. The whole of ``verify_zip``'s assertion in ``scripts/p127``, re-derived
#: rather than retyped.
NUM_BYTES: Final = SRTM_PX * SRTM_PX

#: The swath-count range an SRTM-observed post carries (User Guide V3, Table 2).
OBSERVED_SCENE_COUNTS: Final = (201, 224)
#: Water bodies, in the same table.
WATER_CLASSES: Final = (1, 2, 5)
#: §3.3's two arguable values: 11 is very small SRTMv1 voids NGA interpolated, 25 is GDEM fill
#: that was itself SRTM-derived. Excluding them is the strict "never observed by radar"
#: reading — a documented flag on the extractor, not a silent choice. p135 measured the
#: difference at **one** component of 9,978.
ARGUABLE_CLASSES: Final = (11, 25)

#: §5's four ruggedness bands, cut as quartiles of the eligible test pool.
N_BANDS: Final = 4

#: 8-connected, per P125 §10.2's extraction path.
_STRUCTURE: Final = np.ones((3, 3), dtype=bool)

#: How §3.3 publishes a size sub-row.
REPORTINGS: Final = ("metric", "count", "excluded")


class VoidError(ValueError):
    """Base for everything this module refuses. All of them are silent failures otherwise."""


class EmptyVoidMask(VoidError):
    """A predicate selected nothing — including the one plan §3.3 originally carried."""


class OffLattice(VoidError):
    """A source cell where SRTM 1 arc-second and GLO-30 stop being the same lattice."""


class LatticeMismatch(VoidError):
    """A granule and a surface that do not stand in the 3601/3600 relation."""


class NotAMetricRow(VoidError):
    """A metric was asked of a size sub-row §3.3 publishes as a bare count."""


# --- §3.3's predicate ---------------------------------------------------------


@dataclass(frozen=True)
class Predicate:
    """The void rule, as data — so the rule that was wrong can be *refused* rather than absent.

    ``low`` and ``high`` are both exclusive, which is how §3.3 writes it: ``num > 10 AND
    num < 201``. ``strict`` additionally drops :data:`ARGUABLE_CLASSES`.
    """

    low: int = 10
    high: int = 201
    strict: bool = False

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(
                f"the void range is exclusive on both sides; ({self.low}, {self.high}) is empty"
            )

    @property
    def excluded(self) -> tuple[int, ...]:
        """The class values ``strict`` drops from the range."""
        return ARGUABLE_CLASSES if self.strict else ()

    @property
    def described(self) -> str:
        """The rule as §3.3 writes it, for the message that has to teach."""
        rule = f"num > {self.low} AND num < {self.high}"
        if self.strict:
            rule += f" excluding {list(ARGUABLE_CLASSES)} (strict)"
        return rule

    def selects(self, value: int) -> bool:
        """Does this rule call class ``value`` a void?"""
        return self.low < value < self.high and value not in self.excluded

    def mask(self, num: np.ndarray, *, where: str = "") -> np.ndarray:
        """The void mask, refusing to return an empty one.

        Emptiness is the failure mode this whole module was rewritten around, and it is not an
        error anything downstream raises on: a family with no instances is a family with no
        instances. The 24 granules were drawn for being void-rich (3.7-22.5% over the Himalaya
        cells), so no void at all means the wrong rule or a bad read, and both are said here.
        """
        num = np.asarray(num)
        mask = (num > self.low) & (num < self.high)
        for value in self.excluded:
            mask &= num != value
        if not mask.any():
            raise EmptyVoidMask(self._empty_note(num, where=where))
        return mask

    def _empty_note(self, num: np.ndarray, *, where: str) -> str:
        prefix = f"{where}: " if where else ""
        note = (
            f"{prefix}the void predicate `{self.described}` selected 0 of {num.size:,} posts. "
        )
        if self.selects(0):
            return note + (
                "This is plan §3.3's *original* rule, and it is refused rather than shipped: "
                "there is no value 0 in SRTMGL1N's class table (SRTM Collection User Guide V3, "
                "Table 2, p. 6). 0 means void in SRTMIMGM's differently-encoded `.num`, which "
                "is a different product. The rule that works on this one is "
                "`num > 10 AND num < 201` — `benchmark.voids.PLAN_PREDICATE`."
            )
        return note + (
            "The 24 granules were drawn for being void-rich (P125: 3.7-22.5% over the Himalaya "
            "cells), so an empty mask here is the wrong granule or a bad read, not a result."
        )


#: §3.3's shipped rule.
PLAN_PREDICATE: Final = Predicate()
#: The same rule under the strict "never observed by radar" reading.
STRICT_PREDICATE: Final = Predicate(strict=True)
#: The rule plan §3.3 originally carried, kept **as a named refusal** rather than deleted. It
#: belongs to SRTMIMGM's differently-encoded ``.num``; on this product it selects nothing, and
#: :meth:`Predicate.mask` says so. A rule that is merely absent is a rule that comes back.
REFUTED_ZERO_SCENE_PREDICATE: Final = Predicate(low=-1, high=1)


# --- §3.3's size sub-rows -----------------------------------------------------


@dataclass(frozen=True)
class SizeRow:
    """One published size sub-row: its equivalent-diameter span, and how §3.3 publishes it."""

    name: str
    lo_px: float
    hi_px: float
    #: ``"metric"`` — scored and reported with numbers. ``"count"`` — an instance count and
    #: nothing else. ``"excluded"`` — outside the scored rows, and stated rather than dropped.
    reporting: str

    def __post_init__(self) -> None:
        if self.reporting not in REPORTINGS:
            raise ValueError(f"{self.name}: reporting {self.reporting!r} not in {REPORTINGS}")
        if self.lo_px >= self.hi_px:
            raise ValueError(f"{self.name}: ({self.lo_px}, {self.hi_px}) is not ascending")

    @property
    def carries_metric(self) -> bool:
        return self.reporting == "metric"

    @property
    def scored(self) -> bool:
        """Is an instance in this row one the family ships? Gaps are counted, never scored."""
        return self.reporting != "excluded"

    def contains(self, eqdiam_px: float) -> bool:
        """Closed on both ends; the rows are walked in order, so a boundary lands in the lower."""
        return self.lo_px <= eqdiam_px <= self.hi_px


def size_rows(masks: Masks | None = None) -> tuple[SizeRow, ...]:
    """§3.3's sub-rows, **derived** from §3.2's buckets rather than retyped beside them.

    The boundaries are the same boundaries: real-small has to stay size-comparable to synthetic
    *small* and real-medium+ to synthetic *medium*, which is the entire reason the rows do not
    tile the 16-160 px range. A second copy of them here is a copy that drifts, and #146 is
    what a second home for a roster costs.

    The top is ``real_band_top_px`` (160) rather than the synthetic large bucket's 158: those
    bands describe sizes real voids **have**, measured on SRTM, and the 40% area cap that
    lowered the synthetic bucket is not a constraint on a measurement.
    """
    masks = Masks() if masks is None else masks
    small_lo, small_hi = masks.small_px
    medium_lo, medium_hi = masks.medium_px
    top_lo, top_hi = masks.large_px[0], masks.real_band_top_px

    def gap(lo: float, hi: float) -> SizeRow:
        return SizeRow(f"gap-{lo:g}-{hi:g}", lo, hi, "excluded")

    return (
        SizeRow("real-small", small_lo, small_hi, "metric"),
        gap(small_hi, medium_lo),
        SizeRow("real-medium+", medium_lo, medium_hi, "metric"),
        gap(medium_hi, top_lo),
        SizeRow(f"count-only-{top_lo:g}-{top_hi:g}", top_lo, top_hi, "count"),
    )


def size_row_of(eqdiam_px: float, rows: Sequence[SizeRow]) -> SizeRow | None:
    """The first row holding ``eqdiam_px``, or ``None`` outside the published range."""
    for row in rows:
        if row.contains(eqdiam_px):
            return row
    return None


def require_metric_row(row: SizeRow) -> SizeRow:
    """``row`` back if it carries a metric, or a refusal naming what it carries instead.

    §3.3 publishes 128-160 px as an instance count *with no metric attached* — eleven instances
    cannot carry a mean, and the way that rule is broken is not by arguing with it but by a
    later stage pooling every row it finds. This is the guard that stage calls.
    """
    if not row.carries_metric:
        raise NotAMetricRow(
            f"§3.3 publishes `{row.name}` ({row.lo_px:g}-{row.hi_px:g} px) as a bare "
            f"{row.reporting}, not as a metric: a mean over a row this thin would invite a "
            f"significance the number cannot carry. Use its instance count."
        )
    return row


# --- The cell, and the premise the transplant rests on ------------------------


def cell_origin(cell: str) -> tuple[int, int]:
    """``(origin_lat, origin_lon)`` — the 1-degree cell's south-west corner, signed."""
    lat = int(cell[1:3]) * (1 if cell[0].upper() == "N" else -1)
    lon = int(cell[4:7]) * (1 if cell[3].upper() == "E" else -1)
    return lat, lon


def cog_stem(cell: str) -> str:
    """``N28E086`` -> the Copernicus COG stem for the same 1-degree cell."""
    return f"Copernicus_DSM_COG_10_{cell[0]}{cell[1:3]}_00_{cell[3]}{cell[4:7]}_00_DEM"


def assert_same_lattice(cell: str, *, cols_per_degree: int = GLO_PX) -> None:
    """Refuse a source cell where SRTM 1 arc-second and GLO-30 stop being the same lattice.

    Asserted rather than discovered, because the failure is silent: poleward of 50 degrees
    GLO-30 samples 2400 columns per degree instead of 3600, so a crop-and-paste transplant
    would still produce arrays that broadcast — and a mask stretched across 1.5x the ground it
    was measured on. The band table is :data:`dataset.geometry.COLS_PER_DEGREE`, read here
    rather than copied, so this check and the corpus's own readers cannot disagree.

    Both extreme pixel *centres* are tested, not the cell's nominal edges: N49 and S50 stop
    exactly at the band edge and their posts all sit inside it.
    """
    origin_lat, _ = cell_origin(cell)
    inset = 0.5 / ROWS_PER_DEGREE
    for lat in (origin_lat + inset, origin_lat + 1 - inset):
        cols = COLS_PER_DEGREE[lat_band(lat)]
        if cols != cols_per_degree:
            raise OffLattice(
                f"{cell} reaches |lat| >= 50 (band {lat_band(lat)}), where GLO-30 samples "
                f"{cols} columns per degree against SRTM 1 arc-second's {cols_per_degree}. "
                f"§3.3's transplant crops and pastes rather than interpolating, and the "
                f"premise it rests on is that the two are the *same lattice* (P125 §10.1) — "
                f"which holds only equatorward of 50 degrees. Poleward the premise fails "
                f"silently: the arrays still broadcast."
            )


# --- Reading ------------------------------------------------------------------


def default_granule_dir(paths: Paths | None = None) -> Path:
    """Where ``scripts/p127_fetch_srtm_voids.py`` put the 24 granules."""
    return Path((paths or Paths()).srtm_dir) / "num"


def cells(directory: Path | str | None = None) -> tuple[str, ...]:
    """The source cells on this box, by granule filename."""
    directory = default_granule_dir() if directory is None else Path(directory)
    return tuple(sorted(path.name.split(".")[0] for path in directory.glob("*.num.zip")))


def read_num(cell: str, *, granule_dir: Path | str | None = None) -> np.ndarray:
    """The 3601x3601 uint8 class raster out of its zip, no temp file.

    The byte count is re-asserted here even though ``scripts/p127`` asserted it at fetch time,
    for P125 §13.2's reason: the failure mode is not an error but a login page of plausible
    size, saved to disk under a ``.zip`` name.
    """
    directory = default_granule_dir() if granule_dir is None else Path(granule_dir)
    path = directory / f"{cell}.SRTMGL1N.num.zip"
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise ValueError(f"{cell}: {len(members)} members in {path.name}, expected 1")
        raw = archive.read(members[0])
    if len(raw) != NUM_BYTES:
        raise ValueError(
            f"{cell}: {len(raw):,} bytes in {members[0]}, expected {NUM_BYTES:,} "
            f"({SRTM_PX} x {SRTM_PX} posts at one byte each)"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(SRTM_PX, SRTM_PX)


def read_surface(cell: str, *, open_fn: Callable | None = None) -> np.ndarray:
    """The whole GLO-30 cell as metres — the terrain the void's band is read from.

    The lattice claim is asserted rather than trusted, since everything downstream reads SRTM
    post ``(r, c)`` as GLO-30 post ``(r, c)``: 3600x3600, exactly one arc-second, and pixel
    *centres* on whole arc-second multiples. A COG that is any of those things by a half pixel
    would still slice, and every mask would sit half a post off the ground it was cut from.
    """
    if open_fn is None:  # pragma: no cover - the network path
        import rasterio

        open_fn = rasterio.open

    stem = cog_stem(cell)
    origin_lat, origin_lon = cell_origin(cell)
    step = 1.0 / GLO_PX
    with open_fn(f"/vsis3/copernicus-dem-30m/{stem}/{stem}.tif") as dataset:
        if (dataset.width, dataset.height) != (GLO_PX, GLO_PX):
            raise LatticeMismatch(
                f"{cell}: GLO-30 COG is {dataset.width}x{dataset.height}, not {GLO_PX}^2 — "
                f"§3.3's transplant needs SRTM's lattice, one post per arc-second"
            )
        transform = dataset.transform
        want_x = origin_lon - 0.5 * step
        want_y = origin_lat + 1 + 0.5 * step
        if abs(transform.c - want_x) > 1e-9 or abs(transform.f - want_y) > 1e-9:
            raise LatticeMismatch(
                f"{cell}: COG origin {transform.c},{transform.f} != {want_x},{want_y} — its "
                f"pixel centres are not on whole arc-second multiples, so SRTM post (r, c) is "
                f"not GLO-30 post (r, c)"
            )
        if abs(transform.a - step) > 1e-12 or abs(transform.e + step) > 1e-12:
            raise LatticeMismatch(
                f"{cell}: COG pixel size {transform.a},{transform.e} != {step} (one arc-second)"
            )
        return dataset.read(1)


# --- §5's ruggedness bands ----------------------------------------------------


@dataclass(frozen=True)
class Bands:
    """§5's ruggedness bands: quartile edges of ``slope_mean`` over the eligible test pool.

    Cut on the *pool*, not on the voids — the bands describe the ground a void is transplanted
    onto, so a band index means the same thing here as it does for every other tile in the
    corpus.
    """

    edges: tuple[float, ...]
    #: How many eligible test tiles the edges were cut over. Recorded, not used.
    pool_n: int = 0

    def __post_init__(self) -> None:
        if len(self.edges) < 3:
            raise ValueError(f"{len(self.edges)} edges describe fewer than two bands")
        if any(b < a for a, b in zip(self.edges, self.edges[1:])):
            raise ValueError(f"band edges must ascend; got {list(self.edges)}")

    @property
    def count(self) -> int:
        return len(self.edges) - 1

    def of(self, slope: float) -> int:
        """0-based band index, digitized on the interior edges exactly as p119 does."""
        return int(np.digitize(slope, np.asarray(self.edges[1:-1])))

    def label(self, band: int) -> str:
        """``R1``..``R4`` — §5's own names for them."""
        return f"R{band + 1}"

    @classmethod
    def from_manifest(
        cls, corpus_dir: Path | str, *, n_bands: int = N_BANDS, split: str = "test"
    ) -> Bands:
        """p119's ``eligible_pool``, recomputed: the test split, all-land, water-flag clear."""
        import pyarrow.parquet as pq

        table = pq.read_table(
            Path(corpus_dir) / "manifest.parquet",
            columns=["split", "land_frac", "water_affected", "slope_mean"],
        )
        column = {
            name: table.column(name).to_numpy(zero_copy_only=False)
            for name in table.column_names
        }
        keep = (
            (column["split"] == split)
            & (column["land_frac"] >= 0.999)
            & (~column["water_affected"].astype(bool))
        )
        pool = column["slope_mean"][keep].astype(float)
        if pool.size == 0:
            raise ValueError(
                f"no eligible {split} tiles in {corpus_dir} — §5's bands are cut on the pool "
                f"a void is transplanted onto, so an empty pool has no bands"
            )
        edges = np.quantile(pool, np.linspace(0.0, 1.0, n_bands + 1))
        return cls(edges=tuple(float(e) for e in edges), pool_n=int(pool.size))


# --- Component extraction -----------------------------------------------------


@dataclass(frozen=True)
class Component:
    """One connected void, in the granule's own pixel space. Shape and position, nothing else."""

    cell: str
    label: int
    area_px: int
    eqdiam_px: float
    #: Centroid, SRTM post space.
    row: float
    col: float
    bbox_h: int
    bbox_w: int
    #: §3.2's two placement tests, evaluated rather than assumed.
    fits_interior: bool
    fits_area: bool

    @property
    def placeable(self) -> bool:
        return self.fits_interior and self.fits_area


def components(
    cell: str,
    mask: np.ndarray,
    *,
    masks: Masks | None = None,
    min_eqdiam_px: float | None = None,
) -> tuple[Component, ...]:
    """Every connected void at or above the scored range's floor, with §3.2's tests evaluated.

    8-connected, per P125 §10.2. Components above the range's top come back too — the >160 px
    tail is real and abundant, it is *this benchmark's tile geometry* that excludes it, and
    §3.3's published histogram shows it beside the scored range so a reader can see what the
    tile size cut.
    """
    # scipy is a dev-group dependency, imported inside the function exactly as
    # `dataset.select` imports cKDTree: the freeze and scoring paths must stay importable on a
    # box that installed the runtime alone.
    from scipy import ndimage

    masks = Masks() if masks is None else masks
    rows = size_rows(masks)
    floor = rows[0].lo_px if min_eqdiam_px is None else min_eqdiam_px

    labelled, found = ndimage.label(mask, structure=_STRUCTURE)
    if found == 0:
        return ()
    areas = np.bincount(labelled.ravel())[1:]
    eqdiam = 2.0 * np.sqrt(areas / math.pi)
    # Only components that could ever be published need their geometry resolved — the
    # centroid pass over a Himalaya granule's ~100k specks is the expensive half.
    wanted = np.nonzero(eqdiam >= floor)[0]
    if wanted.size == 0:
        return ()

    labels = (wanted + 1).tolist()
    boxes = ndimage.find_objects(labelled)
    centroids = ndimage.center_of_mass(mask, labelled, labels)
    interior = dataset_config.TILE_PX - 2 * masks.edge_margin_px
    max_area = masks.max_area_frac * dataset_config.TILE_PX**2

    out: list[Component] = []
    for label, (centre_row, centre_col) in zip(labels, centroids):
        row_slice, col_slice = boxes[label - 1]
        height = row_slice.stop - row_slice.start
        width = col_slice.stop - col_slice.start
        out.append(
            Component(
                cell=cell,
                label=int(label),
                area_px=int(areas[label - 1]),
                eqdiam_px=float(eqdiam[label - 1]),
                row=float(centre_row),
                col=float(centre_col),
                bbox_h=int(height),
                bbox_w=int(width),
                fits_interior=bool(height <= interior and width <= interior),
                fits_area=bool(areas[label - 1] <= max_area),
            )
        )
    return tuple(out)


# --- The band, from GLO-30 at the void's own coordinates ----------------------


@dataclass(frozen=True)
class BandReading:
    """What GLO-30 said about the ground one void actually sits on."""

    slope_mean: float
    band: int
    lon: float
    lat: float
    window_row0: int
    window_col0: int
    #: A void within half a tile of the cell edge cannot have a centred window without reading
    #: the neighbouring COG. The window is shifted and the shift is recorded, never hidden.
    clamped: bool


def _window_origin(centroid: float, *, extent: int, tile: int) -> tuple[int, bool]:
    """Top-left origin of the ``tile`` px window centred on ``centroid``, clamped in-cell."""
    wanted = int(round(centroid)) - tile // 2
    origin = int(np.clip(wanted, 0, extent - tile))
    return origin, origin != wanted


def band_for(
    surface: np.ndarray,
    component: Component,
    bands: Bands,
    *,
    tile_px: int | None = None,
) -> BandReading:
    """The void's ruggedness band, read off GLO-30 at the void's own coordinates.

    **There is no ``num`` parameter here, and that is the point.** #126's circularity was
    inferring the stratum from the SRTM posts *surrounding* a void, which measures the wrong
    terrain and biases systematically flat — radar shadow occurs precisely where the
    surroundings are less extreme than the hole. The only elevation this function can see is
    ``surface``: GLO-30, void-free at these coordinates, on the same lattice below 50 degrees.
    The component contributes its shape and its position and nothing else.

    The window is 224 px centred on the component — the ground a transplanted void actually
    came from — scored with :func:`dataset.metrics.slope_mean` at
    :func:`dataset.geometry.pixel_spacing_m`, which is the same call ``dataset.catalog`` makes
    for every other window in the corpus. p135 §3.3 measured the answer indifferent to that
    choice: the corpus's own grid window and a 448 px window move six components of 9,978.
    """
    tile = dataset_config.TILE_PX if tile_px is None else tile_px
    height, width = surface.shape
    origin_lat, origin_lon = cell_origin(component.cell)

    row0, row_clamped = _window_origin(component.row, extent=height, tile=tile)
    col0, col_clamped = _window_origin(component.col, extent=width, tile=tile)
    lon, lat = window_center_lonlat(
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        width=width,
        height=height,
        win_row0=row0,
        win_col0=col0,
        tile=tile,
    )
    window = surface[row0 : row0 + tile, col0 : col0 + tile]
    spacing = pixel_spacing_m(lat, rows_per_degree=height, cols_per_degree=width)
    slope = float(slope_mean_of(window, spacing))
    return BandReading(
        slope_mean=slope,
        band=bands.of(slope),
        lon=float(lon),
        lat=float(lat),
        window_row0=row0,
        window_col0=col0,
        clamped=bool(row_clamped or col_clamped),
    )


# --- The catalog --------------------------------------------------------------


@dataclass(frozen=True)
class ScoredVoid:
    """One published instance: a real shape, and the band of the ground it will be pasted onto."""

    component: Component
    size_row: str
    reporting: str
    reading: BandReading

    @property
    def cell(self) -> str:
        return self.component.cell

    @property
    def label(self) -> int:
        return self.component.label

    @property
    def area_px(self) -> int:
        return self.component.area_px

    @property
    def eqdiam_px(self) -> float:
        return self.component.eqdiam_px

    @property
    def slope_mean(self) -> float:
        return self.reading.slope_mean

    @property
    def band(self) -> int:
        return self.reading.band

    @property
    def lon(self) -> float:
        return self.reading.lon

    @property
    def lat(self) -> float:
        return self.reading.lat

    @property
    def window_row0(self) -> int:
        return self.reading.window_row0

    @property
    def window_col0(self) -> int:
        return self.reading.window_col0

    @property
    def clamped(self) -> bool:
        return self.reading.clamped


@dataclass(frozen=True)
class CellCatalog:
    """One granule's whole answer — what it yielded, and everything it lost on the way."""

    cell: str
    predicate: Predicate
    bands: Bands
    rows: tuple[SizeRow, ...]
    #: Instances in a *published* row: real-small, real-medium+, and the count-only band.
    scored: tuple[ScoredVoid, ...]
    #: Components in one of the two gap rows. Counted, never scored — §3.3 excludes them, and
    #: excluding is not the same as dropping.
    gaps: tuple[Component, ...]
    #: Components at or above the floor but above the published top. Excluded by this
    #: benchmark's 224 px tile under §3.2's 40% cap, not by scarcity.
    excluded_above_top: int
    #: Components inside the published range that fail §3.2's placement tests.
    failed_placement: int
    #: Everything at or above the floor, before any of the above.
    components_at_or_above_floor: int
    #: What fraction of the granule the predicate called void.
    void_frac: float

    @property
    def scored_instances(self) -> int:
        return len(self.scored)

    def instances_by_row(self) -> dict[str, int]:
        """Every published row's instance count, including the gaps and including zeros."""
        counts = {row.name: 0 for row in self.rows}
        for void in self.scored:
            counts[void.size_row] += 1
        for component in self.gaps:
            row = size_row_of(component.eqdiam_px, self.rows)
            counts[row.name] += 1
        return counts


def extract_cell(
    cell: str,
    *,
    bands: Bands,
    predicate: Predicate = PLAN_PREDICATE,
    masks: Masks | None = None,
    read_num_fn: Callable[[str], np.ndarray] | None = None,
    read_surface_fn: Callable[[str], np.ndarray] | None = None,
) -> CellCatalog:
    """One granule, end to end: the predicate, the components, the rows, and the bands.

    The order is load-bearing at both ends. :func:`assert_same_lattice` comes **first**, before
    a 13 MB read and long before a COG read, because the premise it checks fails silently and
    discovering that afterwards is discovering it having already paid. The surface is read
    **once**, after the components are known — a per-component COG read is 24 cells x hundreds
    of voids for the same answer.
    """
    masks = Masks() if masks is None else masks
    rows = size_rows(masks)
    assert_same_lattice(cell)

    read_num_fn = read_num if read_num_fn is None else read_num_fn
    read_surface_fn = read_surface if read_surface_fn is None else read_surface_fn

    num = np.asarray(read_num_fn(cell))
    mask = predicate.mask(num, where=cell)
    found = components(cell, mask, masks=masks)

    top = rows[-1].hi_px
    above_top = sum(1 for component in found if component.eqdiam_px > top)
    inside = [c for c in found if c.eqdiam_px <= top]
    placeable = [c for c in inside if c.placeable]

    surface = np.asarray(read_surface_fn(cell))
    _assert_shared_lattice_shape(cell, num, surface)

    scored: list[ScoredVoid] = []
    gaps: list[Component] = []
    for component in placeable:
        row = size_row_of(component.eqdiam_px, rows)
        if row is None:  # pragma: no cover - `inside` already bounds the range
            continue
        if not row.scored:
            gaps.append(component)
            continue
        scored.append(
            ScoredVoid(
                component=component,
                size_row=row.name,
                reporting=row.reporting,
                reading=band_for(surface, component, bands),
            )
        )

    return CellCatalog(
        cell=cell,
        predicate=predicate,
        bands=bands,
        rows=rows,
        scored=tuple(scored),
        gaps=tuple(gaps),
        excluded_above_top=above_top,
        failed_placement=len(inside) - len(placeable),
        components_at_or_above_floor=len(found),
        void_frac=float(mask.mean()),
    )


def _assert_shared_lattice_shape(cell: str, num: np.ndarray, surface: np.ndarray) -> None:
    """The 3601/3600 relation, as a relation rather than as two constants.

    SRTM's granule overlaps each neighbour by one post and GLO-30's cell does not, so the
    surface is exactly one post narrower per degree. Anything else and post ``(r, c)`` in one
    raster is not post ``(r, c)`` in the other — which no shape check downstream would notice,
    because the slice still succeeds.
    """
    expected = (num.shape[0] - 1, num.shape[1] - 1)
    if surface.shape != expected:
        raise LatticeMismatch(
            f"{cell}: the GLO-30 surface is {surface.shape[0]}x{surface.shape[1]} against the "
            f"granule's {num.shape[0]}x{num.shape[1]}, so it is not the same lattice one post "
            f"narrower (expected {expected[0]}x{expected[1]}). SRTM overlaps its neighbours by "
            f"a post and GLO-30 does not; §3.3 reads SRTM post (r, c) as GLO-30 post (r, c)."
        )


# --- The published summary ----------------------------------------------------


def summarise(catalogs: Iterable[CellCatalog]) -> dict:
    """The histogram §3.3 publishes: what was scored, and everything excluded beside it.

    The count-only row comes back as **a bare count** — no mean, no band breakdown — because
    those are the two shapes that would give eleven instances an authority they do not have.
    The gap rows come back the same way, stated rather than dropped: they hold 7.1% of a real
    draw, and the boundaries that create them exist to keep real-small size-comparable to
    synthetic small.
    """
    catalogs = list(catalogs)
    if not catalogs:
        raise ValueError("nothing to summarise — no granule was extracted")
    rows = catalogs[0].rows
    bands = catalogs[0].bands
    predicate = catalogs[0].predicate
    for catalog in catalogs:
        if catalog.rows != rows or catalog.bands != bands or catalog.predicate != predicate:
            raise ValueError(
                f"{catalog.cell} was extracted under a different predicate, band cut or size "
                f"row set — one histogram cannot be pooled over two of them"
            )

    counts: dict[str, int] = {row.name: 0 for row in rows}
    per_row_bands = {row.name: [0] * bands.count for row in rows if row.carries_metric}
    for catalog in catalogs:
        for name, n in catalog.instances_by_row().items():
            counts[name] += n
        for void in catalog.scored:
            if void.size_row in per_row_bands:
                per_row_bands[void.size_row][void.band] += 1

    published: dict[str, dict] = {}
    for row in rows:
        payload = {"instances": counts[row.name], "reporting": row.reporting}
        if row.carries_metric:
            payload["bands"] = per_row_bands[row.name]
        published[row.name] = payload

    band_totals = [0] * bands.count
    for catalog in catalogs:
        for void in catalog.scored:
            band_totals[void.band] += 1

    return {
        "granules": len(catalogs),
        "predicate": {
            "rule": predicate.described,
            "low": predicate.low,
            "high": predicate.high,
            "strict": predicate.strict,
        },
        "bands": {
            "edges": list(bands.edges),
            "labels": [bands.label(b) for b in range(bands.count)],
            "pool_n": bands.pool_n,
        },
        "rows": published,
        "band_totals": band_totals,
        "attrition": {
            "components_at_or_above_floor": sum(
                c.components_at_or_above_floor for c in catalogs
            ),
            "excluded_above_top": sum(c.excluded_above_top for c in catalogs),
            "excluded_in_gaps": sum(len(c.gaps) for c in catalogs),
            "failed_placement": sum(c.failed_placement for c in catalogs),
            "scored": sum(c.scored_instances for c in catalogs),
        },
        "cells": [
            {
                "cell": c.cell,
                "void_frac": round(c.void_frac, 5),
                "scored": c.scored_instances,
            }
            for c in catalogs
        ],
    }
