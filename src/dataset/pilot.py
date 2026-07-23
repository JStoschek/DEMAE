"""The window-level pilot-scoring frame — Phase B1 (issue #24).

The table the whole Phase-B pruning study runs on. Where A3's calibration
(:mod:`dataset.calibration`) scored every metric *twice per window* — full-res and
×8 — to measure whether the overview survives, this scores every metric **once, at
the resolution production will use** (the Gate A3 verdict, #21), and adds the three
EDM/FLM/WBM mask fractions per window. B2's correlation/PCA matrix must measure the
metrics *as they will be scored*, or it prunes against the wrong ruler.

**The resolution split (#21).** Five metrics read **full-res** — ``slope_mean``,
``slope_std``, ``normals_dispersion``, ``texture``, ``geomorphon_entropy`` — because
their rank collapses on the ×8 overview above 50° (where #15's pole-density
inversion puts the corpus majority). Three stay on the **×8 overview** —
``local_relief``, ``mean_elevation``, ``elevation_relief_ratio`` — clearing the gate
in every band. ``tpi_proxy`` is scored on the **2× box** (window + a half-window
ring, ``TPI_CORE_FRAC = 0.5``) read from the overview (#22); an outer-ring window
whose box runs off the tile returns NaN **for that column alone**. The assignment
partitions :data:`dataset.metrics.METRICS` exactly, and :func:`_resolution_of`
raises on any metric added without a home — a new metric cannot silently drop out.

**The draw is cell-balanced, not band-balanced (recorded per #24, #15).** The frame
wraps :func:`dataset.sampling.sample_cogs` with ``allocation="proportional"`` — the
corpus's own shape by cells, the same principle as
:func:`dataset.sampling.prevalence_sample`'s pixel-pooling. Band-*balanced* would
put ~83% of the draw above 50° (five of the six bands are ≥50°), a polar distortion
right for A3's band-local artifact hunt but wrong for a representativeness
correlation study. Taking *every* land window of each proportionally-drawn COG also
weights the frame by pixels — polar cells yield fewer windows — so it matches what
Phase C production will actually score. See :func:`build_report` §0 for the recorded
reasoning.

**ADR-0001 constraints ride the metrics unchanged.** Exact-zero pixels are excluded
and the mask goes on *before* every neighbourhood op inside
:mod:`dataset.metrics`; the anisotropic ``dx/dy`` per latitude band comes from
:func:`dataset.geometry.pixel_spacing_m`. This module adds no new masking — it
routes each metric to its resolution and nothing more.

The read path mirrors :mod:`dataset.calibration` and :mod:`dataset.prevalence`: a
pure, unit-tested core (scoring, mask fractions, the frame) and a thin live driver
(:func:`main`) that fans whole-COG reads across a thread pool and accumulates
windows in priority order until the target is met.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from . import config
from .calibration import _scale, _window_slices, box_span
from .geometry import (
    CogRef,
    grid_windows,
    lat_band,
    make_window_id,
    pixel_spacing_m,
    window_center_lonlat,
)
from .masks import MaskNotFound, MaskReader
from .metrics import METRICS, land_mask
from .prevalence import EDITED_EDM_CODES, FLM_CODES_FILL, WBM_WATER_CODES
from .sampling import Allocation, band_of, geocell_of
from .tiling import CogReader

# --- The Gate A3 resolution assignment (#21) ----------------------------------

#: Scored on the full-resolution window — their rank collapses on the ×8 overview
#: above 50° (#21), where #15's inversion puts the corpus majority.
FULL_RES_METRICS: Final = (
    "slope_mean",
    "slope_std",
    "normals_dispersion",
    "texture",
    "geomorphon_entropy",
)
#: Scored on the ×8 overview window — they clear the ρ ≥ 0.9 gate in every band.
OVERVIEW_METRICS: Final = ("local_relief", "mean_elevation", "elevation_relief_ratio")
#: The one metric scored on the 2× box (window + half-window ring), read from the
#: overview (#22). ≈1.0 by construction across resolutions; it needs no full-res read.
BOX_METRIC: Final = "tpi_proxy"

#: The overview decimation production scores the overview metrics on (ADR-0001 §4).
OVERVIEW_FACTOR: Final = config.TRI_OVERVIEW_FACTOR

#: The three per-window mask-fraction columns, using the semantic code groupings the
#: prevalence table already fixed (spec §2-4): terrain-edited (EDM), external fill
#: (FLM ≥ 3, non-contiguous), and water (WBM ∈ {1,2,3}).
MASK_FRACTIONS: Final = ("edm_edited_frac", "flm_fill_frac", "wbm_water_frac")


def _resolution_of(metric: str) -> str:
    """Which read a metric is scored from: ``"full"``, ``"overview"`` or ``"box"``.

    Raises :class:`KeyError` for any metric with no Gate A3 home — the guard that
    stops a metric added to :data:`dataset.metrics.METRICS` from silently dropping
    out of the frame at whatever resolution happened to be handy.
    """
    if metric in FULL_RES_METRICS:
        return "full"
    if metric in OVERVIEW_METRICS:
        return "overview"
    if metric == BOX_METRIC:
        return "box"
    raise KeyError(f"metric {metric!r} has no Gate A3 resolution assignment (#21)")


# --- Per-window mask fractions (prevalence groupings, per window) --------------


def _code_frac(mask: np.ndarray, codes: Sequence[int]) -> float:
    """Fraction of a mask window's pixels whose code is in ``codes`` (NaN if empty).

    Over *all* window pixels, matching the prevalence table's per-tile fractions
    (each pixel carries its own weight); ``np.isin`` handles FLM's non-contiguous
    100–102 fill codes, which a ``range`` test would drop.
    """
    a = np.asarray(mask).ravel()
    if a.size == 0:
        return float("nan")
    return float(np.isin(a, list(codes)).sum() / a.size)


def window_mask_fractions(
    edm: np.ndarray, flm: np.ndarray, wbm: np.ndarray
) -> dict[str, float]:
    """The three EDM/FLM/WBM fractions for one window, keyed by :data:`MASK_FRACTIONS`.

    ``edm_edited_frac`` is terrain editing (:data:`~dataset.prevalence.EDITED_EDM_CODES`,
    water excluded — the WBM column owns it); ``flm_fill_frac`` is external
    non-TanDEM fill (FLM ≥ 3); ``wbm_water_frac`` is Ocean/Lake/River (WBM ∈ {1,2,3}).
    """
    return {
        "edm_edited_frac": _code_frac(edm, EDITED_EDM_CODES),
        "flm_fill_frac": _code_frac(flm, FLM_CODES_FILL),
        "wbm_water_frac": _code_frac(wbm, WBM_WATER_CODES),
    }


def _nan_mask_fractions() -> dict[str, float]:
    """The mask columns for a window whose masks could not be read — all NaN."""
    return {name: float("nan") for name in MASK_FRACTIONS}


# --- Production scoring: one read per metric, at its A3 resolution -------------


def _overview_box(
    ov_arr: np.ndarray,
    win_row0: int,
    win_col0: int,
    tile: int,
    ring: int,
    full_h: int,
    full_w: int,
    ov_h: int,
    ov_w: int,
) -> np.ndarray | None:
    """The ``tpi_proxy`` 2× box on the overview, or ``None`` if it runs off the tile.

    Bounds are decided at full res (:func:`~dataset.calibration.box_span`, so the
    tile-edge test is in the same pixels #22 pinned) and mapped onto the overview
    grid (:func:`~dataset.calibration._scale`), so the box frames the identical
    ground either way. ``None`` on either axis means the box left the tile → the
    caller scores ``tpi_proxy`` NaN there, ADR-0001 §4's one-read budget forbidding
    the neighbour tile.
    """
    rows = box_span(win_row0, tile, ring, full_h)
    cols = box_span(win_col0, tile, ring, full_w)
    if rows is None or cols is None:
        return None
    (rlo, rhi), (clo, chi) = rows, cols
    return ov_arr[
        _scale(rlo, full_h, ov_h) : _scale(rhi, full_h, ov_h),
        _scale(clo, full_w, ov_w) : _scale(chi, full_w, ov_w),
    ]


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
    ring: int | None = None,
) -> dict[str, float]:
    """Score every metric once, each at its Gate A3 production resolution.

    ``full_arr``/``ov_arr`` are the *whole* tile at native and ×``factor`` resolution;
    the window is sliced from each so one pair of reads serves every window. Full-res
    metrics read the bare full-res window; overview metrics read the same ground on
    the overview grid (:func:`~dataset.calibration._window_slices`); ``tpi_proxy``
    reads the overview 2× box (:func:`_overview_box`), NaN where the box leaves the
    tile. Each read is handed its own ground pixel size at this latitude, the input
    ADR-0001 §1's anisotropy needs.
    """
    if ring is None:
        ring = tile // 2

    fwin = full_arr[win_row0 : win_row0 + tile, win_col0 : win_col0 + tile]
    _, ov_row = _window_slices(win_row0, tile, full_h, ov_h)
    _, ov_col = _window_slices(win_col0, tile, full_w, ov_w)
    owin = ov_arr[ov_row, ov_col]

    fpx = pixel_spacing_m(lat, rows_per_degree=full_h, cols_per_degree=full_w)
    opx = pixel_spacing_m(lat, rows_per_degree=ov_h, cols_per_degree=ov_w)

    ov_box = _overview_box(
        ov_arr, win_row0, win_col0, tile, ring, full_h, full_w, ov_h, ov_w
    )

    scores: dict[str, float] = {}
    for name, fn in METRICS.items():
        res = _resolution_of(name)
        if res == "full":
            scores[name] = fn(fwin, fpx)
        elif res == "overview":
            scores[name] = fn(owin, opx)
        else:  # box — tpi_proxy on the overview 2× box
            scores[name] = float("nan") if ov_box is None else fn(ov_box, opx)
    return scores


# --- One row per window -------------------------------------------------------


@dataclass(frozen=True)
class WindowRow:
    """One scored pilot window: the nine metrics, three mask fractions, geography.

    ``metrics`` is keyed by :data:`dataset.metrics.METRICS` name (each at its
    production resolution), ``masks`` by :data:`MASK_FRACTIONS`. ``band`` and
    ``geocell`` join a row back to the draw's strata; :attr:`window_id` and the
    window origin make it traceable to the pixels it came from.
    """

    stem: str
    band: str
    geocell: str
    lon: float
    lat: float
    win_row0: int
    win_col0: int
    metrics: dict[str, float]
    masks: dict[str, float]

    @property
    def window_id(self) -> str:
        """A stable id for this window: ``<stem>:<row0>:<col0>`` (see `geometry`)."""
        return make_window_id(self.stem, self.win_row0, self.win_col0)


# --- Per-COG scoring (the parallel read unit) ---------------------------------


def _read_mask_windows(
    mask_reader: MaskReader,
    cog: CogRef,
    origins: list[tuple[int, int]],
    tile: int,
) -> tuple[list[dict[str, float]], tuple[str, str] | None]:
    """Mask fractions for each land window, or NaN columns + an anomaly on absence.

    A genuinely-absent mask (:class:`~dataset.masks.MaskNotFound`) or any other read
    failure is surfaced as ``(stem, reason)`` and every window's mask columns go NaN
    — never read as "no edits" (mask spec fog note 4). The metric columns are
    untouched: a COG with a missing mask still contributes usable terrain rows.
    """
    try:
        edm = mask_reader.read_windows(cog, "EDM", origins, tile)
        flm = mask_reader.read_windows(cog, "FLM", origins, tile)
        wbm = mask_reader.read_windows(cog, "WBM", origins, tile)
    except MaskNotFound as exc:
        return [_nan_mask_fractions() for _ in origins], (cog.stem, f"mask absent: {exc}")
    except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed as "no edits"
        return (
            [_nan_mask_fractions() for _ in origins],
            (cog.stem, f"mask read failed: {type(exc).__name__}: {exc}"),
        )
    return [window_mask_fractions(e, f, w) for e, f, w in zip(edm, flm, wbm)], None


def tile_rows(
    cog_reader: CogReader,
    mask_reader: MaskReader,
    cog: CogRef,
    *,
    tile: int = config.TILE_PX,
    factor: int = OVERVIEW_FACTOR,
    land_only: bool = True,
) -> tuple[list[WindowRow], tuple[str, str] | None]:
    """Score every land-bearing window of one COG, with its three mask fractions.

    Two DEM reads (full-res + ×``factor``) amortise the COG open across all windows,
    then the three masks are read once each at ``origins`` and served per window.
    Returns ``(rows, mask_anomaly)`` — ``mask_anomaly`` is ``None`` unless a mask was
    absent/unreadable, in which case the rows still carry metrics with NaN mask
    columns. Ocean-only windows are skipped (``land_only``); coastal windows are
    kept, where resolution-dependent masking drift lives (ADR-0001 §3).
    """
    full = cog_reader.read_overview(cog, 1)
    ov = cog_reader.read_overview(cog, factor)
    full_arr, ov_arr = full.array, ov.array
    full_h, full_w = full.full_h, full.full_w
    ov_h, ov_w = ov.ov_h, ov.ov_w

    origins = [
        (r, c)
        for r, c in grid_windows(full_w, full_h, tile)
        if not land_only or land_mask(full_arr[r : r + tile, c : c + tile]).any()
    ]
    if not origins:
        return [], None

    mask_fracs, anomaly = _read_mask_windows(mask_reader, cog, origins, tile)
    geocell = geocell_of(cog)

    rows: list[WindowRow] = []
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
        scores = score_window(
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
        )
        rows.append(
            WindowRow(
                stem=cog.stem,
                band=lat_band(lat),
                geocell=geocell,
                lon=lon,
                lat=lat,
                win_row0=win_row0,
                win_col0=win_col0,
                metrics=scores,
                masks=masks,
            )
        )
    return rows, anomaly


# --- The frame: priority-order accumulation to a window target -----------------


def measure(
    cog_reader: CogReader,
    mask_reader: MaskReader,
    cogs: Sequence[CogRef],
    *,
    target_windows: int,
    tile: int = config.TILE_PX,
    factor: int = OVERVIEW_FACTOR,
    workers: int = 8,
    progress: Callable[[int, str], object] | None = None,
) -> tuple[list[WindowRow], list[tuple[str, str]]]:
    """Read ``cogs`` in priority order, accumulating windows until ``target_windows``.

    ``cogs`` is the priority-ordered :func:`~dataset.sampling.sample_cogs` draw — any
    prefix is band-spread — so reading it in order and stopping once the target is met
    keeps the frame geographically balanced while bounding I/O. Reads are batched
    across a thread pool (network-bound); within a batch
    results are re-ordered back to priority order before accumulation, so the frame is
    deterministic regardless of which read finishes first. A COG that cannot be read
    (or whose mask is absent) is surfaced in ``anomalies`` and never folded in as
    zeros; the readable COGs still score.
    """
    from concurrent.futures import ThreadPoolExecutor

    rows: list[WindowRow] = []
    anomalies: list[tuple[str, str]] = []
    bar = progress(target_windows, "pilot") if progress else None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        idx = 0
        while idx < len(cogs) and len(rows) < target_windows:
            batch = list(cogs[idx : idx + workers])
            idx += len(batch)
            results: list[tuple[list[WindowRow], tuple[str, str] | None] | None]
            results = [None] * len(batch)
            futs = {
                ex.submit(
                    tile_rows, cog_reader, mask_reader, cog, tile=tile, factor=factor
                ): k
                for k, cog in enumerate(batch)
            }
            for fut in futs:
                k = futs[fut]
                cog = batch[k]
                try:
                    results[k] = fut.result()
                except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
                    anomalies.append((cog.stem, f"{type(exc).__name__}: {exc}"))
                    results[k] = ([], None)
            for res in results:  # priority order within the batch
                assert res is not None
                batch_rows, anomaly = res
                if anomaly is not None:
                    anomalies.append(anomaly)
                if bar:
                    bar.update(len(batch_rows))
                rows.extend(batch_rows)
    if bar:
        bar.close()
    return rows[:target_windows], anomalies


# --- Report + rows ------------------------------------------------------------


def to_rows(rows: Sequence[WindowRow]) -> list[dict[str, float | str | int]]:
    """Flat per-window records for the committed pilot-frame CSV.

    One row per window: id + geography, then the nine metrics (each at its production
    resolution), then the three mask fractions — the frame the pruning study reads.
    """
    out: list[dict[str, float | str | int]] = []
    for r in rows:
        rec: dict[str, float | str | int] = {
            "window_id": r.window_id,
            "stem": r.stem,
            "band": r.band,
            "geocell": r.geocell,
            "lon": r.lon,
            "lat": r.lat,
            "win_row0": r.win_row0,
            "win_col0": r.win_col0,
        }
        for metric in METRICS:
            rec[metric] = r.metrics[metric]
        for frac in MASK_FRACTIONS:
            rec[frac] = r.masks[frac]
        out.append(rec)
    return out


def _bands_present(rows: Sequence[WindowRow]) -> list[str]:
    """The latitude bands the frame covers, equatorward-first."""
    return sorted({r.band for r in rows}, key=lambda b: float(b.split("-")[0]))


def _finite(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    return a[np.isfinite(a)]


def _stat_cells(values: Sequence[float]) -> str:
    """The seven trailing distribution cells ``n | mean | std | min | p50 | p90 | max``.

    Shared by the metric table (§1) and the mask-fraction table (§2) so the "describe
    a column" shape lives in one place; each caller prepends its own lead cells.
    """
    a = _finite(values)
    if a.size == 0:
        return "0 | — | — | — | — | — | —"
    q = np.quantile(a, [0.5, 0.9])
    return (
        f"{a.size:,} | {a.mean():.4g} | {a.std():.4g} | "
        f"{a.min():.4g} | {q[0]:.4g} | {q[1]:.4g} | {a.max():.4g}"
    )


def build_report(
    rows: Sequence[WindowRow],
    anomalies: Sequence[tuple[str, str]],
    *,
    seed: int,
    allocation: Allocation,
    target_windows: int,
    cogs_read: int,
    factor: int,
    code_version: str,
    snapshot_date: str,
) -> str:
    """Assemble the committed markdown describing the pilot frame (CSV holds the data)."""
    bands = _bands_present(rows)
    total = len(rows)
    tpi_undefined = sum(1 for r in rows if not np.isfinite(r.metrics[BOX_METRIC]))
    mask_nan = sum(1 for r in rows if not np.isfinite(r.masks["edm_edited_frac"]))
    allocation_word = "cell-balanced" if allocation == "proportional" else "band-balanced"

    out: list[str] = []
    w = out.append
    w(f"# Phase-B pilot scoring frame — {total:,} windows, nine metrics + mask fractions")
    w("")
    w("<!-- GENERATED by `python -m dataset.pilot`. Do not hand-edit numbers; "
      "re-run to refresh. The per-window CSV is the artifact; this is its cover sheet. -->")
    w("")
    w(f"Resolves issue #24 (part of the Phase-B map, #23). Builds the window-level "
      f"frame the whole pruning study (B2/B3) runs on: every window scored on **all "
      f"nine `dataset.metrics.METRICS` functions at their Gate A3 production "
      f"resolutions** (#21) plus the three EDM/FLM/WBM mask fractions. Unlike A3's "
      f"calibration, which scored each metric twice to *measure* the overview, this "
      f"scores each **once, as production will** — so B2's correlation matrix measures "
      f"the metrics as they will actually be clustered.")
    w("")
    w(f"- Windows: **{total:,}** land-bearing windows (target {target_windows:,}) from "
      f"**{cogs_read}** COGs"
      f"{'' if not anomalies else f' ({len(anomalies)} anomalies — see below)'}. "
      f"Ocean-only windows are skipped; coastal windows are kept (ADR-0001 §3).")
    w(f"- `tpi_proxy` NaN on **{tpi_undefined:,}** windows (2× box ran off the tile — "
      f"NaN for that column only, #22).")
    w(f"- Mask columns NaN on **{mask_nan:,}** windows (a mask was absent/unreadable — "
      f"surfaced below, never read as 'no edits').")
    w(f"- Provenance: code `{code_version}`, snapshot `{snapshot_date}`, seed `{seed:#x}`.")
    w("")
    w("Copyright / attribution (mask spec §5, Art. 6(b)): *produced using Copernicus "
      "WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
      "2014-2018 provided under COPERNICUS by the European Union and ESA; all rights "
      "reserved.* Art. 6(c): *The organisations in charge of the Copernicus programme "
      "by law or by delegation do not incur any liability for any use of the "
      "Copernicus WorldDEM-30.*")
    w("")

    ge50 = sum(1 for r in rows if abs(r.lat) >= 50.0)
    ge50_frac = ge50 / total if total else float("nan")
    w("## 0. The draw is cell-balanced, not band-balanced (#24, #15)")
    w("")
    w(f"The frame wraps `dataset.sampling.sample_cogs` with "
      f"`allocation=\"{allocation}\"` (**{allocation_word}**) and takes *every* land "
      f"window of each drawn COG, accumulating in the sampler's priority order so any "
      f"prefix stays band-spread. The distinction that matters — and the one #24 asks "
      f"to record — is that a **cell** draw and the **window** frame it produces are "
      f"not the same shape:")
    w("")
    w("- **The COG draw is uniform over cells.** `allocation=\"proportional\"` gives "
      "each 1° cell equal probability — the same principle as `prevalence_sample`. By "
      "*cell count* the corpus is pole-dense: 55.2% of cells sit above 50° (#15).")
    w(f"- **The window frame is then naturally pixel/area-weighted, not cell-weighted.** "
      f"Taking every land window means a cell contributes windows in proportion to its "
      f"pixels, and polar cells hold fewer columns — so the pole-density inversion "
      f"*discounts itself*. This frame lands at **{ge50_frac:.1%}** of windows above "
      f"50° (§3), close to the corpus's **29.8% by land area** (#15) and far from its "
      f"55.2% by cell. That land-area shape is the honest corpus weighting for a window "
      f"population, and it is exactly what Phase C production will score — every land "
      f"window of every kept tile.")
    w("- **Band-balanced would have gone the wrong way.** The band edges are "
      "`(0, 50, 60, 70, 80, 85, 90)` — five of the six bands sit ≥50°, so an "
      "equal-per-band draw puts ~83% of its windows above 50°, *more* polar than either "
      "the corpus's cell count or its land area. Equal power per band is right for A3's "
      "band-*local* artifact hunt (`calibration_sample`); for a representativeness "
      "correlation study it is a 5:1 polar distortion, so it is declined here.")
    w("")

    w(f"## 1. Per-metric distribution across the frame (×{factor} where overview)")
    w("")
    w("Shape of each metric column over the whole frame — the raw material B2 will "
      "correlate. `n` counts windows where the metric is defined (NaN dropped).")
    w("")
    w("| Metric | resolution | n | mean | std | min | p50 | p90 | max |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    _res_label = {"full": "full-res", "overview": f"×{factor} overview", "box": "2× box (ov)"}
    for metric in METRICS:
        res = _res_label[_resolution_of(metric)]
        cells = _stat_cells([r.metrics[metric] for r in rows])
        w(f"| `{metric}` | {res} | {cells} |")
    w("")

    w("## 2. Mask fractions across the frame")
    w("")
    w("| Mask fraction | n | mean | std | min | p50 | p90 | max |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for frac in MASK_FRACTIONS:
        w(f"| `{frac}` | {_stat_cells([r.masks[frac] for r in rows])} |")
    w("")

    w("## 3. Geographic spread (windows per band and geo-cell)")
    w("")
    geocells = {r.geocell for r in rows}
    w(f"Windows span **{len(bands)}** latitude bands and **{len(geocells)}** 10° "
      f"geo-cells. Cell-balanced, so the band mix is the corpus's land-area shape, not "
      f"its cell count (§0).")
    w("")
    w("| Band | windows | fraction |")
    w("|---|---:|---:|")
    for b in bands:
        n = sum(1 for r in rows if r.band == b)
        w(f"| {b}° | {n:,} | {n / total:.4f} |" if total else f"| {b}° | 0 | — |")
    w("")

    if anomalies:
        w("## Anomalies (surfaced, not swallowed)")
        w("")
        w("A COG that could not be read is excluded; an absent mask NaNs that COG's "
          "mask columns but keeps its metric rows (mask spec fog note 4).")
        w("")
        for stem, reason in anomalies:
            w(f"- `{stem}` — {reason}")
        w("")

    return "\n".join(out) + "\n"


# --- Live driver --------------------------------------------------------------


def _load_frame(frame_cache: Path | None) -> list[CogRef]:
    """The full COG frame — from a cached stem list if present, else list S3 once.

    A deliberate third copy of the prevalence/calibration loader (they share nothing
    else); a one-function import coupling for five lines is not worth the dependency.
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


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def _tqdm(total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="win", dynamic_ncols=True)
    except ImportError:  # pragma: no cover
        return None


def main(argv: list[str] | None = None) -> int:
    """``python -m dataset.pilot`` — build and write the committed pilot frame."""
    import argparse
    import csv
    import logging

    from .sampling import sample_cogs

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("pilot")

    p = argparse.ArgumentParser(prog="python -m dataset.pilot")
    p.add_argument("--target-windows", type=int, default=15_000,
                   help="stop once this many land windows are scored (plan ~10-20k)")
    p.add_argument("--allocation", choices=("proportional", "balanced"),
                   default="proportional",
                   help="latitude-band allocation (default proportional = cell-balanced, #24)")
    p.add_argument("--draw-cogs", type=int, default=600,
                   help="priority-ordered COGs to draw before the window target stops the read")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--workers", type=int, default=8, help="parallel whole-COG reads")
    p.add_argument("--factor", type=int, default=OVERVIEW_FACTOR, help="overview decimation")
    p.add_argument("--out", type=Path, default=Path("docs/research/pilot-frame.md"))
    p.add_argument("--csv", type=Path,
                   default=Path("docs/research/pilot-frame-per-window.csv"))
    p.add_argument("--cache-dir", type=Path, default=Path(config.MASK_CACHE_DIR))
    p.add_argument("--frame-cache", type=Path, default=Path("data/cog_frame.json"),
                   help="cache the S3 prefix listing here so re-runs skip it (blank to list)")
    args = p.parse_args(argv)

    log.info("loading COG frame ...")
    refs = _load_frame(args.frame_cache)
    log.info("frame: %d COGs", len(refs))

    sample = sample_cogs(refs, args.draw_cogs, allocation=args.allocation, seed=args.seed)
    log.info("drew %d COGs (seed %#x, %s); reading until %d windows",
             len(sample), args.seed, args.allocation, args.target_windows)

    from .masks import RasterioMaskReader
    from .tiling import RasterioCogReader

    cog_reader = RasterioCogReader()
    mask_reader = RasterioMaskReader(cache_dir=args.cache_dir)
    rows, anomalies = measure(
        cog_reader,
        mask_reader,
        sample,
        target_windows=args.target_windows,
        factor=args.factor,
        workers=args.workers,
        progress=_tqdm,
    )
    cogs_read = len({r.stem for r in rows})
    log.info("scored %d windows from %d COGs, %d anomalies",
             len(rows), cogs_read, len(anomalies))

    from .provenance import git_sha

    report = build_report(
        rows,
        anomalies,
        seed=args.seed,
        allocation=args.allocation,
        target_windows=args.target_windows,
        cogs_read=cogs_read,
        factor=args.factor,
        code_version=git_sha(),
        snapshot_date=_today(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    log.info("wrote %s", args.out)

    flat = to_rows(rows)
    if flat:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            wr.writeheader()
            wr.writerows(flat)
        log.info("wrote %s", args.csv)

    print(
        f"\nScored {len(rows)} windows from {cogs_read} COGs "
        f"({args.allocation}, {len(anomalies)} anomalies).\n"
        f"  -> {args.out}\n  -> {args.csv}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
