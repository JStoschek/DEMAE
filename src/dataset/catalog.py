"""The production global scorer and the raw candidate catalog — Phase C3 (issue #35).

Where :mod:`dataset.pilot` scored a 15,000-window *sample* on all nine candidate
metrics to let Phase B prune them, this scores **every window of every COG** on the
**six metrics ADR-0004 froze**, and lands the result on disk as Parquet. That frame
is the reusable sampling frame: Phase D's stratifier, the debug tier, and the scale
tier all draw from it without re-reading a byte of Copernicus.

**What is scored, and where the numbers come from.** The vector is frozen (ADR-0004
§1, Gate B) and this module does not reopen it — it only routes each metric to the
resolution the ADR assigned:

- **full-res 30 m** — ``slope_mean``, ``texture``, ``geomorphon_entropy``;
- **×8 overview (~240 m)** — ``local_relief``, ``mean_elevation``,
  ``elevation_relief_ratio``.

The three metrics Phase B pruned (``slope_std`` and ``normals_dispersion`` on
redundancy, ``tpi_proxy`` on B3's empty re-entry trigger) are **not scored**. They
still exist in :data:`dataset.metrics.METRICS`, so the split here is deliberately
*not* a partition of that registry the way :mod:`dataset.pilot`'s was — it is a
partition of the frozen six, and :func:`_resolution_of` raises for anything else.

**The read strategy is C1's locked always-full-res** (issue #33): every COG is read
twice — once at ×8, once at native 30 m — in every latitude band. The measured ≥50°
hybrid saving (59.6% of scoring wall-clock) was declined because this catalog is a
frozen frame every tier reuses, and the hybrid's sub-50° fidelity is established for
``slope_mean`` only.

C1 handed two open questions to this ticket. Both are answered by measurement, not
by assumption:

1. **Is the second S3 open (the overview read) removable?** Kept. It could be
   replaced by decimating the full-res array in memory, but the pilot measured
   overview fidelity against GDAL's *stored*, mode-/average-resampled overview, and
   an in-memory decimation is a different resampling — so removing it would put the
   three overview metrics on an unmeasured ruler. The prize is small either way:
   C1's own table prices the overview read at 1.35 s against 9.75 s of compute in the
   pixel-heavy 0–50° band, so the whole second open is ~12% of wall-clock. Not worth
   an unmeasured ruler on a frozen frame.
2. **Is ``geomorphon_entropy`` the compute hotspot?** It was, by a wide margin, and
   the lever has since been pulled. Per 224² window, at this module's resolutions,
   the original measurement read: ``geomorphon_entropy`` 25.8 ms, ``texture``
   13.3 ms, ``slope_mean`` 0.54 ms, the three overview metrics together under
   0.02 ms, and the ``exact_zero_frac``/``land_std`` pair 0.24 ms. So ~98% of
   scoring cost was the two full-res neighbourhood metrics, and C1's ~46 ms/window
   reproduced.

   The lever was :func:`dataset.metrics._sight_line_angles`, which called ``arctan``
   once per line-of-sight step — eighty times per cell. It now walks the ray in raw
   slopes and converts the two extremes once, which is **bit-exact** rather than
   merely close: ``arctan`` is monotone, so it commutes with the running max and
   min. Re-measured as a matched pair on one machine, median over 30 real corpus
   windows: ``geomorphon_entropy`` **30.2 ms → 14.1 ms** (2.1x), against ``texture``
   12.9 ms and ``slope_mean`` 0.75 ms unchanged. (The pre-change number benches
   higher here than the 25.8 ms on record while ``texture`` reproduces, so trust the
   ratio and the matched pair, not the difference against the older figure.) Full-res
   compute per window falls ~37%, from ~43.8 ms to ~27.7 ms.

   The same bar was then applied to ``texture``, the metric the hoist promoted to
   #1: :func:`dataset.metrics._median_filter` now takes a plain ``median`` on a
   window with no ocean fill, which is the common case and **bit-exact** there (an
   odd, all-finite neighbourhood's median is the middle element either way) — 3.3x
   on the filter. What neither of them fixed was the *parallelism*, which is the
   next section and was the larger error by far.

   What was *not* done: deleting the ``arctan`` rather than hoisting it. Every
   comparison downstream is monotone in the angle, so raw slopes look equivalent and
   are ~1.4x faster again — but ``arctan`` is monotone without being injective on
   float64, and above a slope of ~2 (63°) neighbouring slopes share an angle. Raw
   slopes therefore break ties the angles hold, and a broken tie is a different
   landform: ``tests/test_metrics.py`` carries a window where one cell moves from
   ``flat`` to ``shoulder`` and the entropy with it. ADR-0004 froze this vector and
   every downstream tier reads these values, so the last 1.4x is declined.

**The pass runs on a process pool** (:func:`main`, issue #38). C1's 4.19 h
projection divided a *serial* per-COG cost by 16 threads, and the dry run came in at
1.87 s/COG — a ~14 h job. The gap is not per-window cost, it is the GIL: scoring is
~98% numpy neighbourhood compute, and threads measured 2.4x against one process at
best, however many more you add. Worker *processes* remove the ceiling, and
:func:`~dataset.concurrency.ordered_futures` supplies the same bounded, ordered
scheduling either way, so the frame is identical and only the wall-clock moves. The
measured scaling curve, the corrected corpus projection, and the bit-identity check
against the pre-change catalog are in
[`docs/research/c3-scoring-throughput.md`](../../docs/research/c3-scoring-throughput.md).

**Nothing is gated.** Every window of every COG gets a row, ocean-only windows
included, and every fraction is stored raw. The quality gates (zero-fill, degenerate,
water, edited/filled, coastline) are C4/C5's business and are expressed as filters
over these columns — which only works if the frame carries the *denominator* too. A
window with no land pixels is emitted with NaN metrics rather than dropped.

**WBM is stored per class** (:data:`WBM_CLASS_FRACTIONS`), the amendment C2
([#34](https://github.com/JStoschek/DEMAE/issues/34), ADR-0001 Amendment 2) made to
this ticket. C2 measured that 141 of 245 candidate coastline windows contain no ocean
pixel at all — they are sea-level river and lake, median ``land_std`` 2.23 m against
69.99 m for genuine shoreline — so the coastline stratum is
``(0.05 <= exact_zero_frac < 0.75) & (wbm1_frac > 0)``. Collapsing WBM into one
aggregate column would make that stratum uncomputable without a global re-score.

**The run is resumable.** 26,450 COGs is a multi-hour pass; a pass that cannot resume
is a pass that never finishes. Rows are flushed to numbered ``part-*.parquet`` files
under :data:`PARTS_DIR` and each flush appends a line to :data:`PROGRESS_FILE` naming
the COGs it covered. A re-run skips those COGs and discards any part file the ledger
never recorded — the orphan a crash between the write and the ledger append would
leave. The plan's §C3 asks for "one Parquet"; the parts share a declared schema and
:func:`read_catalog` hands back exactly that one table.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

if TYPE_CHECKING:  # pyarrow is imported lazily — only the writer needs it
    from concurrent.futures import Executor

    import pyarrow as pa

from . import config
from .calibration import _window_slices
from .geometry import CogRef, grid_windows, lat_band, pixel_spacing_m, window_center_lonlat
from .masks import MaskNotFound, MaskReader
from .metrics import METRICS, land_mask
from .normalize import land_stats
from .prevalence import EDITED_EDM_CODES, FLM_CODES_FILL, WBM_WATER_CODES
from .sampling import geocell_of
from .screen import zero_fraction
from .tiling import CogReader

# --- The frozen vector and its resolutions (ADR-0004 §1) ----------------------

#: Scored on the full-resolution 30 m window. ``slope_mean`` forces the read (its
#: rank collapses on the ×8 overview above 50°); ``texture`` and
#: ``geomorphon_entropy`` ride it at ~zero marginal cost (ADR-0004 §3).
FULL_RES_METRICS: Final = ("slope_mean", "texture", "geomorphon_entropy")
#: Scored on the ×8 overview — they clear the ρ ≥ 0.9 gate in every latitude band.
OVERVIEW_METRICS: Final = ("local_relief", "mean_elevation", "elevation_relief_ratio")

#: The frozen six, in ADR-0004 §1 table order (the order the catalog columns use).
FROZEN_METRICS: Final = (
    "slope_mean",
    "local_relief",
    "texture",
    "elevation_relief_ratio",
    "mean_elevation",
    "geomorphon_entropy",
)

#: The overview decimation the overview metrics are scored on (ADR-0001 §4).
OVERVIEW_FACTOR: Final = config.TRI_OVERVIEW_FACTOR

# --- Mask fraction columns (per class, C2 / #34) ------------------------------

#: WBM stored one column per water class — Ocean, Lake, River. The per-class split
#: is what keeps C2's sea-touching coastline term (``wbm1_frac > 0``) computable.
WBM_CLASS_FRACTIONS: Final = ("wbm1_frac", "wbm2_frac", "wbm3_frac")

#: Every mask column: EDM terrain editing, FLM non-TanDEM fill, the three WBM
#: classes, and the WBM aggregate over {1,2,3} (kept for the water cap, which does
#: not care which kind of water it is).
MASK_FRACTIONS: Final = (
    "edm_edited_frac",
    "flm_fill_frac",
    *WBM_CLASS_FRACTIONS,
    "wbm_water_frac",
)

#: The catalog's column order, id and geography first, then the measurements.
COLUMNS: Final = (
    "window_id",
    "stem",
    "band",
    "geocell",
    "lon",
    "lat",
    "win_row0",
    "win_col0",
    "exact_zero_frac",
    "land_std",
    *FROZEN_METRICS,
    *MASK_FRACTIONS,
)

#: Columns carrying a string; everything else is numeric (see :func:`_schema`).
_STRING_COLUMNS: Final = ("window_id", "stem", "band", "geocell")
#: Columns carrying a pixel offset — small non-negative integers.
_INT_COLUMNS: Final = ("win_row0", "win_col0")


def _resolution_of(metric: str) -> str:
    """Which read a frozen metric is scored from: ``"full"`` or ``"overview"``.

    Raises :class:`KeyError` for anything outside the frozen six — including the
    metrics Phase B pruned. Unlike :func:`dataset.pilot._resolution_of`, this is
    *not* a total function over :data:`dataset.metrics.METRICS`: the catalog scores
    ADR-0004's vector, not the registry, and a pruned metric arriving here is a bug
    rather than a metric that lost its home.
    """
    if metric in FULL_RES_METRICS:
        return "full"
    if metric in OVERVIEW_METRICS:
        return "overview"
    raise KeyError(f"{metric!r} is not in the ADR-0004 frozen vector {FROZEN_METRICS}")


# --- Per-window measurements --------------------------------------------------


def exact_zero_frac(window: np.ndarray) -> float:
    """Fraction of pixels at Copernicus's exact ocean fill, ``0.0``.

    Measured on the **full-resolution** window, which is what C2's per-pixel
    cross-check ran on and what :func:`dataset.normalize.prepare_tile` re-derives at
    materialization. The column keeps its accurate name: it measures exact zeros,
    and the *gate* built on it is named ``zero_fill`` (ADR-0001 Amendment 2), because
    a 75%-zero window may be sea-level river or lake rather than ocean.

    The arithmetic is :func:`dataset.screen.zero_fraction`'s — that module owns the
    zero-fill screen and :mod:`dataset.normalize` already defers to it, so the catalog
    measures the gate's own input rather than a look-alike of it. All this adds is the
    empty-window NaN, which the screen never needs because it is only ever handed a
    real window.
    """
    a = np.asarray(window)
    if a.size == 0:
        return float("nan")
    return zero_fraction(a)


def land_std(window: np.ndarray) -> float:
    """Population standard deviation over the window's land pixels, in metres.

    The input to ADR-0002 §2's degenerate gate (land-std < 3 m), and — per C2 — the
    column that separates a genuine shoreline (median 69.99 m) from flat delta river
    (2.23 m) once ``wbm1_frac`` has split them. NaN when no land pixel survives.

    The statistic is :func:`dataset.normalize.land_stats`'s second return value —
    the same land-only population std the degenerate gate is defined against, not a
    re-derivation of it. What differs is *what it is measured on*: the float32 read
    here, the stored int16 array there. The two part company by sub-metre rounding,
    two orders below the 3 m threshold; the catalog measures terrain, and the
    materialization step re-derives its own gate from the bytes it writes.
    """
    a = np.asarray(window, dtype=np.float64)
    mask = land_mask(a)
    if not mask.any():
        return float("nan")
    return land_stats(a, mask)[1]


def _code_frac(mask: np.ndarray, codes: Sequence[int]) -> float:
    """Fraction of a mask window's pixels whose code is in ``codes`` (NaN if empty).

    Over *all* window pixels, matching the prevalence table's per-tile fractions;
    ``np.isin`` handles FLM's non-contiguous 100–102 fill codes, which a ``range``
    test would drop.
    """
    a = np.asarray(mask).ravel()
    if a.size == 0:
        return float("nan")
    return float(np.isin(a, list(codes)).sum() / a.size)


def window_mask_fractions(
    edm: np.ndarray, flm: np.ndarray, wbm: np.ndarray
) -> dict[str, float]:
    """The six EDM/FLM/WBM fractions for one window, keyed by :data:`MASK_FRACTIONS`.

    ``edm_edited_frac`` is terrain editing (:data:`~dataset.prevalence.EDITED_EDM_CODES`
    — water excluded, the WBM columns own it); ``flm_fill_frac`` is external
    non-TanDEM fill (FLM ≥ 3); then **one column per WBM water class** plus their
    aggregate. The per-class split is C2's amendment to this ticket: the coastline
    stratum needs ``wbm1_frac`` (Ocean) on its own, and no aggregate can supply it.
    """
    fracs = {
        "edm_edited_frac": _code_frac(edm, EDITED_EDM_CODES),
        "flm_fill_frac": _code_frac(flm, FLM_CODES_FILL),
    }
    for code, name in zip(WBM_WATER_CODES, WBM_CLASS_FRACTIONS):
        fracs[name] = _code_frac(wbm, (code,))
    fracs["wbm_water_frac"] = _code_frac(wbm, WBM_WATER_CODES)
    return fracs


def _nan_mask_fractions() -> dict[str, float]:
    """The mask columns for a window whose masks could not be read — all NaN."""
    return {name: float("nan") for name in MASK_FRACTIONS}


def _nan_scores() -> dict[str, float]:
    """The metric columns for a window with no land to measure — all NaN."""
    return {name: float("nan") for name in FROZEN_METRICS}


# --- Scoring: one read per resolution, six metrics ----------------------------


def score_window(
    full_arr: np.ndarray,
    ov_arr: np.ndarray,
    *,
    win_row0: int,
    win_col0: int,
    full_h: int,
    full_w: int,
    ov_h: int,
    ov_w: int,
    lat: float,
    tile: int = config.TILE_PX,
) -> dict[str, float]:
    """Score the frozen six, each at its ADR-0004 resolution.

    ``full_arr``/``ov_arr`` are the *whole* COG at native and ×factor resolution; the
    window is sliced from each, so one pair of reads serves every window of the tile.
    The overview slice covers the identical ground at its own resolution
    (:func:`~dataset.calibration._window_slices`), and each read is handed the ground
    pixel size of the raster it came from — ADR-0001 §1's anisotropy is per-resolution,
    not per-tile.

    A window with no land pixel short-circuits to all-NaN. That is not a gate: every
    metric masks the ocean fill before it measures anything, so each would return NaN
    on its own — this only declines to spend ~46 ms of full-res compute proving it.
    """
    fwin = full_arr[win_row0 : win_row0 + tile, win_col0 : win_col0 + tile]
    if not land_mask(fwin).any():
        return _nan_scores()

    _, ov_row = _window_slices(win_row0, tile, full_h, ov_h)
    _, ov_col = _window_slices(win_col0, tile, full_w, ov_w)
    owin = ov_arr[ov_row, ov_col]

    fpx = pixel_spacing_m(lat, rows_per_degree=full_h, cols_per_degree=full_w)
    opx = pixel_spacing_m(lat, rows_per_degree=ov_h, cols_per_degree=ov_w)

    scores: dict[str, float] = {}
    for name in FROZEN_METRICS:
        fn = METRICS[name]
        if _resolution_of(name) == "full":
            scores[name] = fn(fwin, fpx)
        else:
            scores[name] = fn(owin, opx)
    return scores


# --- One row per window -------------------------------------------------------


@dataclass(frozen=True)
class CatalogRow:
    """One scored candidate window — the frozen six, the fractions, the geography.

    ``metrics`` is keyed by :data:`FROZEN_METRICS`, ``masks`` by
    :data:`MASK_FRACTIONS`. Every field is a raw measurement: no gate has been
    applied, and none is implied by a row's presence. ``band`` and ``geocell`` join
    a row to the strata Phase D will build; :attr:`window_id` and the window origin
    make it traceable back to the pixels it came from.
    """

    stem: str
    band: str
    geocell: str
    lon: float
    lat: float
    win_row0: int
    win_col0: int
    exact_zero_frac: float
    land_std: float
    metrics: dict[str, float]
    masks: dict[str, float]

    @property
    def window_id(self) -> str:
        """A stable id for this window: ``<stem>:<row0>:<col0>``."""
        return f"{self.stem}:{self.win_row0}:{self.win_col0}"


# --- Per-COG scoring (the parallel read unit) ---------------------------------


def _read_mask_windows(
    mask_reader: MaskReader,
    cog: CogRef,
    origins: list[tuple[int, int]],
    tile: int,
) -> tuple[list[dict[str, float]], tuple[str, str] | None]:
    """Mask fractions per window; NaN columns + an anomaly if the mask is *absent*.

    Absence and failure are handled differently here, which is the whole reason
    :mod:`dataset.masks` distinguishes them. A genuinely-absent mask
    (:class:`~dataset.masks.MaskNotFound`) is permanent, so its windows keep their
    terrain columns and take NaN mask columns — never read as "no edits" (mask spec
    fog note 4) — and the absence is surfaced as ``(stem, reason)``.

    Any *other* read failure propagates. It is transient by definition (network,
    throttling), and :func:`scan` answers a raised exception by discarding the COG
    and leaving it unmarked, so the next run retries it. Swallowing it into NaN would
    bake a five-second S3 hiccup into a permanently unknowable ``wbm1_frac``, and C2's
    coastline stratum cannot be computed over an unknowable ``wbm1_frac``.
    """
    try:
        edm = mask_reader.read_windows(cog, "EDM", origins, tile)
        flm = mask_reader.read_windows(cog, "FLM", origins, tile)
        wbm = mask_reader.read_windows(cog, "WBM", origins, tile)
    except MaskNotFound as exc:
        return [_nan_mask_fractions() for _ in origins], (cog.stem, f"mask absent: {exc}")
    return [window_mask_fractions(e, f, w) for e, f, w in zip(edm, flm, wbm)], None


def tile_rows(
    cog_reader: CogReader,
    mask_reader: MaskReader,
    cog: CogRef,
    *,
    tile: int = config.TILE_PX,
    factor: int = OVERVIEW_FACTOR,
) -> tuple[list[CatalogRow], tuple[str, str] | None]:
    """Score **every** window of one COG — the unit of parallelism in :func:`scan`.

    Two DEM reads (full-res + ×``factor``, C1's locked always-full-res path) amortise
    the COG open across all of that tile's windows, then the three masks are read once
    each at the same origins. The window grid is ADR-0001 §2's unchanged one:
    non-overlapping, edge-flushed (:func:`~dataset.geometry.grid_windows`).

    Unlike :func:`dataset.pilot.tile_rows`, no window is skipped. An ocean-only window
    lands with ``exact_zero_frac`` 1.0 and NaN terrain columns, because the zero-fill
    gate is a C4 filter over this frame and its attrition count needs a denominator.
    """
    full = cog_reader.read_overview(cog, 1)
    ov = cog_reader.read_overview(cog, factor)
    full_arr, ov_arr = full.array, ov.array
    full_h, full_w = full.full_h, full.full_w
    ov_h, ov_w = ov.ov_h, ov.ov_w

    origins = grid_windows(full_w, full_h, tile)
    if not origins:
        return [], None

    mask_fracs, anomaly = _read_mask_windows(mask_reader, cog, origins, tile)
    geocell = geocell_of(cog)

    rows: list[CatalogRow] = []
    for (win_row0, win_col0), masks in zip(origins, mask_fracs):
        lon, lat = window_center_lonlat(
            origin_lon=cog.origin_lon,
            origin_lat=cog.origin_lat,
            width=full_w,
            height=full_h,
            win_row0=win_row0,
            win_col0=win_col0,
            tile=tile,
        )
        fwin = full_arr[win_row0 : win_row0 + tile, win_col0 : win_col0 + tile]
        rows.append(
            CatalogRow(
                stem=cog.stem,
                band=lat_band(lat),
                geocell=geocell,
                lon=lon,
                lat=lat,
                win_row0=win_row0,
                win_col0=win_col0,
                exact_zero_frac=exact_zero_frac(fwin),
                land_std=land_std(fwin),
                metrics=score_window(
                    full_arr,
                    ov_arr,
                    win_row0=win_row0,
                    win_col0=win_col0,
                    full_h=full_h,
                    full_w=full_w,
                    ov_h=ov_h,
                    ov_w=ov_w,
                    lat=lat,
                    tile=tile,
                ),
                masks=masks,
            )
        )
    return rows, anomaly


# --- Flat records + Parquet ---------------------------------------------------


def to_records(rows: Sequence[CatalogRow]) -> list[dict[str, Any]]:
    """Flat per-window records in :data:`COLUMNS` order — what a part file holds."""
    out: list[dict[str, Any]] = []
    for r in rows:
        rec: dict[str, Any] = {
            "window_id": r.window_id,
            "stem": r.stem,
            "band": r.band,
            "geocell": r.geocell,
            "lon": r.lon,
            "lat": r.lat,
            "win_row0": r.win_row0,
            "win_col0": r.win_col0,
            "exact_zero_frac": r.exact_zero_frac,
            "land_std": r.land_std,
        }
        for metric in FROZEN_METRICS:
            rec[metric] = r.metrics[metric]
        for frac in MASK_FRACTIONS:
            rec[frac] = r.masks[frac]
        out.append(rec)
    return out


def _schema() -> pa.Schema:
    """The Parquet schema, declared rather than inferred.

    Inference would type a column from whatever the first part file happened to hold
    — an all-NaN ``land_std`` in a polar batch could land as null and refuse to
    concatenate with a later float column. Declaring it once means every part file is
    readable as one dataset.
    """
    import pyarrow as pa

    fields = []
    for name in COLUMNS:
        if name in _STRING_COLUMNS:
            fields.append(pa.field(name, pa.string()))
        elif name in _INT_COLUMNS:
            fields.append(pa.field(name, pa.int32()))
        else:
            fields.append(pa.field(name, pa.float64()))
    return pa.schema(fields)


def write_part(records: Sequence[dict[str, Any]], path: Path) -> None:
    """Write one part file atomically (temp file + rename).

    A half-written Parquet that a later run reads as data would corrupt the frame
    silently, so the file only appears at its final name once it is complete.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _schema()
    columns = {
        name: pa.array([rec[name] for rec in records], type=schema.field(name).type)
        for name in COLUMNS
    }
    table = pa.table(columns, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# --- The resumable ledger -----------------------------------------------------

#: Where the part files live. A subdirectory, not the catalog root, so the whole
#: catalog reads as one Parquet dataset — ``pyarrow`` infers a dataset's schema from
#: every file it finds, and a sidecar ``catalog_meta.json`` sitting alongside the
#: parts makes that inference fail outright.
PARTS_DIR: Final = "parts"
#: One JSON line per flushed part: the part index, the COGs it covers, its row count.
PROGRESS_FILE: Final = "progress.jsonl"
#: One JSON line per COG that could not be fully read — surfaced, never swallowed.
ANOMALY_FILE: Final = "anomalies.jsonl"
#: The catalog fingerprint: provenance, the scored vector, totals.
META_FILE: Final = "catalog_meta.json"


def part_path(out_dir: Path, index: int) -> Path:
    """Path of part ``index`` — zero-padded so the parts sort in write order."""
    return out_dir / PARTS_DIR / f"part-{index:05d}.parquet"


def read_catalog(out_dir: Path | str):
    """Read the whole catalog as one :class:`pyarrow.Table` — the consumer entry point.

    The frame is written as many part files so a multi-hour pass can resume, but
    every consumer (C4's threshold tuning, the summary notebook, Phase D's
    stratifier) wants the single table the plan's §C3 describes. This is that table:
    the parts share a declared schema (:func:`_schema`), so concatenating them is
    total, and the sidecar ledger/meta files sit outside :data:`PARTS_DIR` where the
    dataset reader will not try to parse them as Parquet.
    """
    import pyarrow.parquet as pq

    return pq.read_table(Path(out_dir) / PARTS_DIR)


@dataclass
class Ledger:
    """What a previous run of :func:`scan` already finished, read back from disk."""

    done_stems: set[str] = field(default_factory=set)
    next_part: int = 0
    n_rows: int = 0
    n_parts: int = 0


def read_ledger(out_dir: Path) -> Ledger:
    """Replay :data:`PROGRESS_FILE` into the state a resumed scan starts from.

    A truncated final line (the crash case: a partial ``write`` of the append) is
    dropped rather than raising — the COGs it named are simply re-scanned. A record
    with no ``part`` covered only empty COGs and wrote no file (see
    :func:`scan`'s ``flush``), so it advances the done set without claiming an index.
    """
    ledger = Ledger()
    path = out_dir / PROGRESS_FILE
    if not path.exists():
        return ledger
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn tail from a crashed append; those COGs re-scan
        ledger.done_stems.update(rec["stems"])
        if rec.get("part") is not None:
            ledger.next_part = max(ledger.next_part, int(rec["part"]) + 1)
            ledger.n_parts += 1
        ledger.n_rows += int(rec["rows"])
    return ledger


def _append_json_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _discard_orphan_parts(out_dir: Path, next_part: int) -> list[Path]:
    """Delete part files at or beyond ``next_part`` — ones the ledger never recorded.

    A crash between :func:`write_part` and the ledger append leaves a complete part
    file that no line claims. Its COGs will be re-scanned, so keeping it would
    double-count those windows in the frame.
    """
    dropped = []
    for path in sorted((out_dir / PARTS_DIR).glob("part-*.parquet")):
        try:
            index = int(path.stem.split("-")[1])
        except (IndexError, ValueError):
            continue
        if index >= next_part:
            path.unlink()
            dropped.append(path)
    return dropped


# --- The scan driver ----------------------------------------------------------


#: How a :func:`scan` fans its per-COG work out. ``"process"`` is what the global
#: pass runs on (#38); ``"thread"`` is the in-process path — see :func:`_make_executor`.
ExecutorKind = Literal["thread", "process"]

#: Mild oversubscription: a worker blocked on an S3 read leaves its core idle, and
#: another worker's compute fills it. Measured, not assumed — see :func:`default_workers`.
WORKERS_PER_CORE: Final = 2
#: The top of the measured curve on the dev box, and where its memory stops being
#: free: a worker peaks at 337 MB (measured RSS), so 16 is ~5.4 GB resident. A
#: *default*, not a limit — an explicit ``workers=`` is taken at its word, and a
#: bigger or same-region box should re-measure before raising it.
MAX_DEFAULT_WORKERS: Final = 16
#: Where thread scaling flattens: measured 2.56 → 2.27 → 2.37 s/COG at 4, 8 and 16
#: threads, so 8 is the top and 16 is already the far side of it (issue #38).
THREAD_GIL_KNEE: Final = 8


def _check_executor(kind: str) -> ExecutorKind:
    """Reject an unknown executor at the first gate rather than the last.

    Both :func:`default_workers` and :func:`_make_executor` branch on the kind, and
    :func:`scan` calls them in that order — so without this the bad value would get a
    plausible worker count before anything complained.
    """
    if kind not in ("thread", "process"):
        raise ValueError(f"unknown executor {kind!r}; expected 'thread' or 'process'")
    return kind  # type: ignore[return-value]


def default_workers(kind: ExecutorKind = "process") -> int:
    """How many COGs to score in parallel — measured on this box, not assumed (#38).

    C1's projection divided a *serial* per-COG cost by 16 workers, which assumed
    linear thread scaling. Scoring is dominated by numpy neighbourhood compute under
    the GIL, so the two executors want very different counts, and neither wants 16.
    One sweep, 24 spread COGs per config on the dev box (8 logical cores), against a
    measured serial baseline of 5.51 s/COG (``docs/research/c3-throughput.json``):

    - **thread** — 2.56 @4, 2.27 @8, 2.37 @16. Best case **2.4x**, reached at 8 and
      given back by 16. Extra threads past :data:`THREAD_GIL_KNEE` buy contention.
    - **process** — 1.89 @4, 1.22 @6, 1.12 @8, 1.11 @12, 0.87 @16, 0.94 @20. Best
      case **6.3x**, and it keeps improving *past* the core count because a worker
      waiting on a COG read is not using one. It turns over at ~2.5x cores, hence
      :data:`WORKERS_PER_CORE` capped at :data:`MAX_DEFAULT_WORKERS`.

    The oversubscription is transport-dependent: this box is in the US and the bucket
    is in ``eu-central-1``, so there is a lot of read latency to hide. A same-region
    run has less, and its knee will sit nearer the core count — re-measure with
    ``notebooks/c3_throughput.py`` rather than porting this number.

    ``os.process_cpu_count()`` (3.13+) respects an affinity mask or a container CPU
    limit; ``os.cpu_count()`` is the 3.12 fallback. Both count *logical* cores, so on
    an asymmetric box (this dev machine: 6P + 2E) the E-cores are included — they
    contribute, just less, and mild oversubscription absorbs the difference anyway.
    """
    cores = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    if _check_executor(kind) == "thread":
        return min(THREAD_GIL_KNEE, cores)
    return max(1, min(WORKERS_PER_CORE * cores, MAX_DEFAULT_WORKERS))


def _make_executor(kind: ExecutorKind, workers: int) -> Executor:
    """Build the pool ``scan`` fans across — the one place the two paths differ.

    Processes are forced onto the **spawn** start method rather than taking the
    platform default. macOS spawns anyway, but the production run is a rented Linux
    box where the default is fork, and forking a parent that has already opened GDAL
    handles and a boto3 client inherits both into a child that must not share them.
    Spawning everywhere means the box that runs the pass behaves like the box the
    pass was measured on.
    """
    if _check_executor(kind) == "thread":
        from concurrent.futures import ThreadPoolExecutor

        return ThreadPoolExecutor(max_workers=workers)

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    )


@dataclass(frozen=True)
class ScanResult:
    """What one :func:`scan` invocation did, and what the catalog now holds."""

    out_dir: Path
    n_cogs: int  # COGs scored by *this* run
    n_skipped: int  # COGs the ledger already had
    n_anomalies: int  # COGs this run could not fully read
    n_rows: int  # cumulative rows in the catalog, across every run
    n_parts: int  # cumulative part *files* written (not the next part index)
    n_cogs_total: int  # cumulative COGs the ledger has marked done


def scan(
    cog_reader: CogReader,
    mask_reader: MaskReader,
    cogs: Sequence[CogRef],
    out_dir: Path | str,
    *,
    tile: int = config.TILE_PX,
    factor: int = OVERVIEW_FACTOR,
    rows_per_part: int = 500_000,
    workers: int | None = None,
    executor: ExecutorKind = "thread",
    code_version: str = "unknown",
    snapshot_date: str = "unknown",
    progress: Callable[[int, str], Any] | None = None,
) -> ScanResult:
    """Score every window of every COG in ``cogs`` into a Parquet catalog at ``out_dir``.

    Work is fanned out **one COG per task** and consumed in input order. Rows
    accumulate until ``rows_per_part`` and are then flushed as a part file, after
    which a ledger line names the COGs that part covers — so a re-run resumes at the
    last completed flush rather than the start.

    **``executor`` picks the pool, and it is the difference between a 14-hour pass and
    a practical one** (#38). This work is not network-bound the way a full-res
    materialize is: most of it is numpy neighbourhood compute holding the GIL, so
    *threads* measured 2.4x against one process at best and gave some of it back at 16 —
    a ceiling, not a slope. ``"process"`` gives each
    worker its own interpreter and reached 6.3x on the same draw; the global pass runs
    on it (:func:`main` defaults
    to it), and :func:`ordered_futures` provides the same bounded, ordered scheduling
    either way, so the frame is byte-identical and only the wall-clock moves.

    The default here is ``"thread"`` because the library entry point takes *injected*
    readers, and a process pool constrains them in two ways an in-process caller has
    no reason to expect: they must pickle, and any state they mutate while scoring
    (a counter, a cache, a fault injector) mutates in the worker where the parent
    cannot see it. The live readers satisfy both — :class:`RasterioMaskReader` was
    made picklable for exactly this — so the driver opts in, and a caller passing its
    own reader opts in when it knows the reader can take it.

    **A COG is marked done only when its rows are complete.** A read that raises —
    a transient S3 failure, a mask fetch that fell over — is recorded in
    :data:`ANOMALY_FILE` and its rows are discarded, so a later run retries it. The
    one anomaly that *is* kept is a genuinely-absent mask
    (:class:`~dataset.masks.MaskNotFound`), which no retry will conjure: those rows
    land with NaN mask columns. That split is exactly the one :mod:`dataset.masks`
    exists to make, and it matters here because a transient failure baked in as a
    permanent NaN ``wbm1_frac`` would make C2's coastline stratum uncomputable for
    those windows without the global re-score the amendment exists to avoid.

    ``ScanResult.n_rows``, ``n_parts`` and ``n_cogs_total`` are cumulative over every
    run that has written into ``out_dir`` — they describe the catalog, while
    ``n_cogs``/``n_skipped``/``n_anomalies`` describe this invocation.
    """
    from concurrent.futures.process import BrokenProcessPool

    from .concurrency import ordered_futures

    workers = default_workers(executor) if workers is None else workers
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = read_ledger(out_dir)
    _discard_orphan_parts(out_dir, ledger.next_part)

    todo = [c for c in cogs if c.stem not in ledger.done_stems]
    n_skipped = len(cogs) - len(todo)
    part = ledger.next_part
    total_rows = ledger.n_rows
    total_parts = ledger.n_parts
    total_cogs = len(ledger.done_stems)

    buffer: list[CatalogRow] = []
    covered: list[str] = []
    n_cogs = 0
    n_anomalies = 0
    bar = progress(len(todo), "catalog") if progress else None

    def flush() -> None:
        nonlocal part, buffer, covered, total_rows, total_parts, total_cogs
        if not covered:
            return
        # A flush covering only empty COGs writes no file, so it claims no part index
        # — otherwise the index would run ahead of the files and leave a hole that
        # `_discard_orphan_parts` reads as an orphan on the next run.
        record: dict[str, Any] = {"stems": covered, "rows": len(buffer)}
        if buffer:
            write_part(to_records(buffer), part_path(out_dir, part))
            record["part"] = part
            part += 1
            total_parts += 1
        _append_json_line(out_dir / PROGRESS_FILE, record)
        total_rows += len(buffer)
        total_cogs += len(covered)
        buffer = []
        covered = []

    # A module-level function bound by `partial` rather than a closure: a process
    # pool pickles the task, and a closure does not pickle.
    score = partial(tile_rows, cog_reader, mask_reader, tile=tile, factor=factor)

    try:
        with _make_executor(executor, workers) as ex:
            for cog, fut in ordered_futures(
                ex=ex, fn=score, items=todo, in_flight=max(1, workers * 2)
            ):
                if bar:
                    bar.update(1)
                try:
                    rows, anomaly = fut.result()
                except BrokenProcessPool:
                    # Every remaining future is already dead, so stepping over this
                    # one would book 26,000 anomalies in a tight loop. Bail out and
                    # let the `finally` flush; the ledger resumes the rest.
                    raise
                except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
                    n_anomalies += 1
                    _append_json_line(
                        out_dir / ANOMALY_FILE,
                        {"stem": cog.stem, "reason": f"{type(exc).__name__}: {exc}"},
                    )
                    continue  # rows discarded, not marked done -> a later run retries
                if anomaly is not None:
                    n_anomalies += 1
                    _append_json_line(
                        out_dir / ANOMALY_FILE, {"stem": anomaly[0], "reason": anomaly[1]}
                    )
                buffer.extend(rows)
                covered.append(cog.stem)
                n_cogs += 1
                if len(buffer) >= rows_per_part:
                    flush()
    finally:
        flush()
        if bar:
            bar.close()

    result = ScanResult(
        out_dir=out_dir,
        n_cogs=n_cogs,
        n_skipped=n_skipped,
        n_anomalies=n_anomalies,
        n_rows=total_rows,
        n_parts=total_parts,
        n_cogs_total=total_cogs,
    )
    write_meta(
        out_dir,
        result,
        factor=factor,
        tile=tile,
        code_version=code_version,
        snapshot_date=snapshot_date,
    )
    return result


def write_meta(
    out_dir: Path,
    result: ScanResult,
    *,
    factor: int,
    tile: int,
    code_version: str,
    snapshot_date: str,
) -> Path:
    """Write :data:`META_FILE` — the fingerprint that makes the catalog auditable.

    Records what was scored and at which resolution alongside the provenance, so a
    reader of the frame never has to infer the recipe from the column names. Mirrors
    ADR-0002 §3's ``corpus_meta.json`` convention.
    """
    from datetime import datetime, timezone

    meta = {
        "schema_version": config.MANIFEST_SCHEMA_VERSION,
        "code_version": code_version,
        "snapshot_date": snapshot_date,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tile_px": tile,
        "overview_factor": factor,
        "read_strategy": "always-full-res",  # C1 / #33
        "columns": list(COLUMNS),
        "metrics": list(FROZEN_METRICS),
        "resolutions": {m: _resolution_of(m) for m in FROZEN_METRICS},
        "gated": False,  # every window scored; gates are C4/C5 filter columns
        "n_rows": result.n_rows,
        "n_cogs": result.n_cogs_total,
        "n_parts": result.n_parts,
    }
    path = out_dir / META_FILE
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


# --- Live driver --------------------------------------------------------------


def _load_frame(frame_cache: Path | None) -> list[CogRef]:
    """The full COG frame — from a cached stem list if present, else list S3 once.

    The fourth copy of the prevalence/calibration/pilot loader, kept for the reason
    :func:`dataset.pilot._load_frame` records: the four drivers share nothing else,
    and importing one driver into another to save five lines would couple every live
    entry point to whichever one happened to define it first.
    """
    from .geometry import parse_cog_key

    if frame_cache and frame_cache.exists():
        stems = json.loads(frame_cache.read_text())
        return [parse_cog_key(s) for s in stems]
    from .discovery import list_cog_refs

    refs = list_cog_refs()
    if frame_cache:
        frame_cache.parent.mkdir(parents=True, exist_ok=True)
        frame_cache.write_text(json.dumps([r.stem for r in refs]))
    return refs


def select_cogs(
    refs: Sequence[CogRef],
    *,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = config.RANDOM_SEED,
) -> list[CogRef]:
    """Which COGs to score: all of them, a prefix, or a geographically spread draw.

    The full pass takes the frame as it stands. The two narrowing options are not
    interchangeable, and picking the wrong one quietly wastes the run:

    - ``limit`` takes a **prefix**, which is the bucket's key order — the first few
      hundred stems are all one corner of equatorial Africa. That is right for a
      smoke check (does the live path work at all?) and useless for anything that
      looks at a distribution.
    - ``sample`` draws through :func:`dataset.sampling.sample_cogs`, spread across
      latitude bands and 10° geo-cells, so a few hundred COGs actually resemble the
      corpus. That is what a dry run before the full pass wants.

    ``sample`` wins if both are given, since asking for a spread draw and getting a
    prefix is the failure worth ruling out.
    """
    from .sampling import sample_cogs

    if sample is not None:
        return sample_cogs(refs, sample, allocation="proportional", seed=seed)
    if limit is not None:
        return list(refs[:limit])
    return list(refs)


def _tqdm(total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="cog", dynamic_ncols=True, smoothing=0.02)
    except ImportError:  # pragma: no cover
        return None


def main(argv: list[str] | None = None) -> int:
    """``python -m dataset.catalog`` — run the global scoring pass.

    Re-running the same command after an interruption resumes: finished COGs are
    skipped and only the tail is scored. ``--limit-cogs`` runs a short prefix (the
    live-path smoke check) and ``--sample-cogs`` a band-spread draw (the dry run whose
    distributions mean something) — see :func:`select_cogs`.
    """
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log = logging.getLogger("catalog")

    p = argparse.ArgumentParser(
        prog="python -m dataset.catalog",
        description="Score every window of every Copernicus COG into the raw candidate "
        "catalog (Phase C, issue #35). Resumable: re-run to continue.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/catalog"))
    p.add_argument("--limit-cogs", type=int, default=None,
                   help="score only the first N COGs in bucket order — a smoke check, "
                        "not a representative sample (the prefix is one corner of Africa)")
    p.add_argument("--sample-cogs", type=int, default=None,
                   help="score a band/geo-cell-spread draw of N COGs; use this, not "
                        "--limit-cogs, for a dry run whose distributions mean something")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                   help="seed for --sample-cogs")
    p.add_argument("--executor", choices=("process", "thread"), default="process",
                   help="score COGs in worker processes (default) or threads; the "
                        "work is GIL-bound numpy, so threads cap out near 2.1x")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel whole-COG scorers (default: measured per executor "
                        "— see catalog.default_workers)")
    p.add_argument("--factor", type=int, default=OVERVIEW_FACTOR, help="overview decimation")
    p.add_argument("--rows-per-part", type=int, default=500_000,
                   help="rows per Parquet part file (also the resume granularity)")
    p.add_argument("--cache-dir", type=Path, default=Path(config.MASK_CACHE_DIR))
    p.add_argument("--frame-cache", type=Path, default=Path("data/cog_frame.json"),
                   help="cache the S3 prefix listing here so re-runs skip it")
    p.add_argument("--snapshot-date", default=None)
    args = p.parse_args(argv)

    frame = _load_frame(args.frame_cache)
    refs = select_cogs(
        frame, limit=args.limit_cogs, sample=args.sample_cogs, seed=args.seed
    )
    workers = args.workers if args.workers is not None else default_workers(args.executor)
    log.info(
        "frame: %d COGs, scoring %d -> %s (%d %s workers)",
        len(frame), len(refs), args.out_dir, workers, args.executor,
    )

    from .provenance import git_sha, today
    from .masks import RasterioMaskReader
    from .tiling import RasterioCogReader

    result = scan(
        RasterioCogReader(),
        RasterioMaskReader(cache_dir=args.cache_dir),
        refs,
        args.out_dir,
        factor=args.factor,
        rows_per_part=args.rows_per_part,
        workers=workers,
        executor=args.executor,
        code_version=git_sha(),
        snapshot_date=args.snapshot_date or today(),
        progress=_tqdm,
    )

    print(
        f"\nCatalog at {result.out_dir}: {result.n_rows:,} rows in {result.n_parts} part(s).\n"
        f"  this run: {result.n_cogs} COGs scored, {result.n_skipped} already done, "
        f"{result.n_anomalies} anomalies\n"
        f"  meta: {result.out_dir / META_FILE}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
