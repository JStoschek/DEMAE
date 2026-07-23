"""Overview-vs-full-res metric calibration — Gate A3 (issue #20).

The one measurement the whole ``metric(array, pixel_size)`` contract was built to
make possible. Every metric in :data:`dataset.metrics.METRICS` is a pure function of
an array and that array's ground pixel size, with no reach into a latitude or a
manifest — so the *same* function scores a full-resolution 224² window and the ×8
overview of the same ground, no special-casing. This module runs exactly that:
per window, score all nine metric functions twice, and report the Spearman rank
correlation per metric between the two reads.

**Why rank correlation, and why 0.9.** The metrics feed *k-means clustering* into
terrain strata (plan §D), so what a stratum needs is that windows keep their
*relative order* between the cheap ×8 read and the expensive full-res one — not that
the absolute numbers agree. Spearman ρ measures exactly that ordering agreement, and
the plan's gate is ρ ≥ :data:`RHO_GATE`. This module **reports** ρ; it does not decide
drop-or-full-res per metric. That call is Gate A3's own ticket (#21), an explicit
human decision, and nothing here pre-empts it.

**Why per band, not just pooled.** The artifact this calibration exists to catch is
band-*local*: Copernicus's columns go anisotropic and its sampling steps down at the
band edges above 50° (plan §11.4). A metric can pass pooled and fail above 50°, and a
pooled-only table would hide it — so :func:`rho_table` reports ρ per latitude band and
the calibration sample (:func:`dataset.sampling.calibration_sample`) is band-*balanced*
to give every band equal power, the poles included.

**Two suspects are flagged, not judged, in the committed table.**

* ``tpi_proxy`` is scored on the **2× box** (window + a half-window ring), per #22 —
  ``TPI_CORE_FRAC = 0.5`` puts the window itself at the core, so the reading is the
  plan's "window mean − large-neighbourhood mean". The box is sized in *window* units,
  so it covers the same ground at both resolutions, and its ρ is **≈1.0 by
  construction** — the least-stressed column, not a real survival test. Outer-ring
  windows whose box runs off the tile return **NaN for ``tpi_proxy`` only**; the other
  eight still score them.
* ``geomorphon_entropy``'s line-of-sight radius (:data:`~dataset.metrics.GEOMORPHON_LOS_PX`)
  is a fixed *cell* count, so it spans 300 m at full res and ~2.4 km at ×8 — a
  different-sized landform at each resolution. A low ρ there reads "the ruler moved",
  not "geomorphons don't survive the overview"; the reach is B2's free parameter.

The read path mirrors :mod:`dataset.prevalence`: a pure, unit-tested core (Spearman,
window/box extraction, the ρ table) and a thin live driver (:func:`main`) that fans
~100 whole-COG reads across a thread pool — the same I/O-bound parallel-read shape
the rest of the pipeline uses.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from . import config
from .geometry import (
    CogRef,
    lat_band,
    pixel_spacing_m,
    window_center_lonlat,
)
from .metrics import METRICS, land_mask
from .tiling import CogReader

#: The metric scored on the 2× box rather than the bare window (issue #22).
TPI_METRIC: Final = "tpi_proxy"

#: The plan's Gate A3 threshold: ρ at or above this is "survives the overview" for
#: *ranking* purposes. Reported against, never acted on here (#21 owns the call).
RHO_GATE: Final = 0.9

#: The overview decimation A3 calibrates against — the ×8 the plan scores on and the
#: pipeline already reads (:data:`config.TRI_OVERVIEW_FACTOR`, ADR-0001 §4).
OVERVIEW_FACTOR: Final = config.TRI_OVERVIEW_FACTOR

#: Pooled-band key for the ρ table, alongside the per-band rows.
POOLED: Final = "pooled"
#: Convenience aggregate over every band at/above 50° — the risk region A3 is drawn
#: to interrogate. A separate row so a reader sees the polar verdict at a glance.
POOLED_GE50: Final = ">=50"


# --- Spearman rank correlation (numpy; no scipy) ------------------------------


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average (fractional) ranks of a 1-D array, ties shared — scipy's algorithm.

    Tied values each receive the mean of the ranks they span, which is what makes
    the Pearson-of-ranks below a *tie-corrected* Spearman rather than one that
    depends on the arbitrary order ties were sorted in. Input must be finite; the
    caller drops NaNs first.
    """
    a = np.asarray(a, dtype=np.float64)
    sorter = np.argsort(a, kind="mergesort")  # stable, so ties keep input order
    inv = np.empty(a.size, dtype=np.intp)
    inv[sorter] = np.arange(a.size)
    a_sorted = a[sorter]
    # Dense rank of each element: number of distinct values at or below it.
    obs = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    dense = obs.cumsum()[inv]
    # Boundaries of each run of equal values, in sorted order.
    count = np.concatenate((np.nonzero(obs)[0], [a.size]))
    # Average of the 1-based ranks a run spans: (first + last + 1) / 2.
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation of ``x`` and ``y`` over their complete pairs.

    Pairs where either value is non-finite (NaN metrics — no land, degenerate ratio,
    a ``tpi_proxy`` box off the tile) are dropped first; this is the pairwise-complete
    convention the ρ table needs, so a metric that is undefined on a handful of windows
    is still scored on the rest. Returns NaN with fewer than two complete pairs, or
    when either ranked series has zero variance (every value tied — no ordering to
    correlate).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _n_complete(x: Sequence[float], y: Sequence[float]) -> int:
    """How many ``(x, y)`` pairs are finite in both — the ρ's sample size."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return int((np.isfinite(x) & np.isfinite(y)).sum())


# --- Two-resolution window / box extraction -----------------------------------


def _scale(coord: int, full_dim: int, ov_dim: int) -> int:
    """Map a full-res pixel coordinate onto the overview grid, proportionally.

    Both rasters span the same one degree, so a pixel boundary at ``coord/full_dim``
    of the way across sits at the same fraction of the overview. Rounding keeps the
    ×8 window ~28 px and the ×8 box ~56 px whatever the parity of the band's width,
    and clamps inside ``[0, ov_dim]`` because the full-res coordinate is in range.
    """
    return int(min(ov_dim, max(0, round(coord * ov_dim / full_dim))))


def _window_slices(
    origin: int, tile: int, full_dim: int, ov_dim: int
) -> tuple[slice, slice]:
    """The full-res and overview slices for one window axis.

    The full-res slice is the exact ``tile`` pixels; the overview slice maps the same
    ground span (:func:`_scale` of both ends), so the two arrays cover the identical
    footprint at their own resolutions — the whole premise of the comparison.
    """
    full = slice(origin, origin + tile)
    ov = slice(_scale(origin, full_dim, ov_dim), _scale(origin + tile, full_dim, ov_dim))
    return full, ov


def box_span(origin: int, tile: int, ring: int, dim: int) -> tuple[int, int] | None:
    """Full-res ``[lo, hi)`` of the ``tpi_proxy`` box on one axis, or ``None`` off-tile.

    The box is the window plus a ``ring``-pixel margin each side (a half-window ring
    → a 2× box), so ``tpi_proxy`` measures the window against the terrain framing it
    (#22). A box that would reach past the tile edge cannot be filled from a
    single-tile read — ADR-0001 §4's one-read budget forbids the neighbour tile — so
    it returns ``None`` and the caller scores ``tpi_proxy`` as NaN there, that metric
    alone.
    """
    lo, hi = origin - ring, origin + tile + ring
    if lo < 0 or hi > dim:
        return None
    return lo, hi


@dataclass(frozen=True)
class WindowScore:
    """One window's paired readings: every metric scored full-res and at ×8.

    ``full`` and ``overview`` are keyed by :data:`dataset.metrics.METRICS` name, so a
    per-metric ρ is a column-wise correlation across many of these. ``band``/``lat``
    join a row back to its latitude band for the per-band table; ``stem`` and the
    window origin make every row traceable to the pixels it came from.
    """

    stem: str
    band: str
    lon: float
    lat: float
    win_row0: int
    win_col0: int
    full: dict[str, float]
    overview: dict[str, float]


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
) -> tuple[dict[str, float], dict[str, float]]:
    """Score every metric on one window at both resolutions; return ``(full, ov)`` dicts.

    ``full_arr``/``ov_arr`` are the *whole* tile at native and ×8 resolution; the
    window is sliced from each so a single pair of reads serves every window (and
    every ``tpi_proxy`` box). The pixel size handed to each metric is that read's own
    ground spacing at this window's latitude — full-res rows/cols per degree for one,
    the overview's for the other — which is the input that lets one function fairly
    score both.

    The eight window-only metrics get the bare window; ``tpi_proxy`` gets the 2× box
    (:func:`box_span`), NaN where the box runs off the tile. ``ring`` defaults to a
    half-window (``tile // 2``), the #22 box.
    """
    if ring is None:
        ring = tile // 2

    full_row, ov_row = _window_slices(win_row0, tile, full_h, ov_h)
    full_col, ov_col = _window_slices(win_col0, tile, full_w, ov_w)
    fwin = full_arr[full_row, full_col]
    owin = ov_arr[ov_row, ov_col]

    fpx = pixel_spacing_m(lat, rows_per_degree=full_h, cols_per_degree=full_w)
    opx = pixel_spacing_m(lat, rows_per_degree=ov_h, cols_per_degree=ov_w)

    # tpi_proxy's box, built once at full res and mapped to the overview so both cover
    # the same ground. None on either axis means the box left the tile -> NaN, tpi only.
    fbox = _tpi_arrays(
        full_arr, ov_arr, win_row0, win_col0, tile, ring, full_h, full_w, ov_h, ov_w
    )

    full_scores: dict[str, float] = {}
    ov_scores: dict[str, float] = {}
    for name, fn in METRICS.items():
        if name == TPI_METRIC:
            if fbox is None:
                full_scores[name] = float("nan")
                ov_scores[name] = float("nan")
            else:
                full_box, ov_box = fbox
                full_scores[name] = fn(full_box, fpx)
                ov_scores[name] = fn(ov_box, opx)
        else:
            full_scores[name] = fn(fwin, fpx)
            ov_scores[name] = fn(owin, opx)
    return full_scores, ov_scores


def _tpi_arrays(
    full_arr: np.ndarray,
    ov_arr: np.ndarray,
    win_row0: int,
    win_col0: int,
    tile: int,
    ring: int,
    full_h: int,
    full_w: int,
    ov_h: int,
    ov_w: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The ``tpi_proxy`` box at both resolutions, or ``None`` if it runs off the tile.

    The box bounds are decided at full res (:func:`box_span`); the overview box maps
    the identical ground span, so ``tpi_proxy``'s core-vs-frame split falls on the same
    footprint either way — which is why this column's ρ is ≈1.0 by construction.
    """
    rows = box_span(win_row0, tile, ring, full_h)
    cols = box_span(win_col0, tile, ring, full_w)
    if rows is None or cols is None:
        return None
    (rlo, rhi), (clo, chi) = rows, cols
    full_box = full_arr[rlo:rhi, clo:chi]
    ov_box = ov_arr[
        _scale(rlo, full_h, ov_h) : _scale(rhi, full_h, ov_h),
        _scale(clo, full_w, ov_w) : _scale(chi, full_w, ov_w),
    ]
    return full_box, ov_box


# --- Per-COG scoring (the parallel read unit) ---------------------------------


def tile_scores(
    reader: CogReader,
    cog: CogRef,
    *,
    tile: int = config.TILE_PX,
    factor: int = OVERVIEW_FACTOR,
    land_only: bool = True,
) -> list[WindowScore]:
    """Read one COG full-res and at ×``factor``, and score every land-bearing window.

    Two reads per COG — the whole tile at native resolution and its stored ×8 overview
    — amortise the (dominant) COG open across all of its windows, the same batching
    :mod:`dataset.tiling` uses. Windows with no land in the full-res read are skipped
    when ``land_only`` (the default): they score NaN on every metric and would only
    pad the CSV. Coastal windows *are* kept — they are where resolution-dependent
    masking drift lives (ADR-0001 §3), exactly what A3 must see.
    """
    from .geometry import grid_windows

    full = reader.read_overview(cog, 1)
    ov = reader.read_overview(cog, factor)
    full_arr, ov_arr = full.array, ov.array
    full_h, full_w = full.full_h, full.full_w
    ov_h, ov_w = ov.ov_h, ov.ov_w

    out: list[WindowScore] = []
    for win_row0, win_col0 in grid_windows(full_w, full_h, tile):
        if land_only and not land_mask(
            full_arr[win_row0 : win_row0 + tile, win_col0 : win_col0 + tile]
        ).any():
            continue
        lon, lat = window_center_lonlat(
            origin_lon=cog.origin_lon,
            origin_lat=cog.origin_lat,
            width=full_w,
            height=full_h,
            win_row0=win_row0,
            win_col0=win_col0,
            tile=tile,
        )
        full_scores, ov_scores = score_window(
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
        out.append(
            WindowScore(
                stem=cog.stem,
                band=lat_band(lat),
                lon=lon,
                lat=lat,
                win_row0=win_row0,
                win_col0=win_col0,
                full=full_scores,
                overview=ov_scores,
            )
        )
    return out


# --- The ρ table (per metric, per band + pooled) ------------------------------


@dataclass(frozen=True)
class Rho:
    """One cell of the calibration table: ρ over ``n`` complete window pairs."""

    rho: float
    n: int

    @property
    def passes(self) -> bool:
        """Whether ρ clears the gate. NaN (undefined) is not a pass."""
        return np.isfinite(self.rho) and self.rho >= RHO_GATE


def _column(scores: Sequence[WindowScore], metric: str) -> tuple[np.ndarray, np.ndarray]:
    full = np.array([s.full[metric] for s in scores], dtype=np.float64)
    ov = np.array([s.overview[metric] for s in scores], dtype=np.float64)
    return full, ov


def metric_rho(scores: Sequence[WindowScore], metric: str) -> Rho:
    """Spearman ρ between the full-res and ×8 reads of one metric across ``scores``."""
    full, ov = _column(scores, metric)
    return Rho(rho=spearman(full, ov), n=_n_complete(full, ov))


def bands_present(scores: Sequence[WindowScore]) -> list[str]:
    """The latitude bands the sample actually covers, equatorward-first."""
    return sorted({s.band for s in scores}, key=lambda b: float(b.split("-")[0]))


def rho_table(scores: Sequence[WindowScore]) -> dict[str, dict[str, Rho]]:
    """The full calibration table: ``{metric: {band-or-pooled: Rho}}``.

    Every metric gets a ``pooled`` cell (all windows), a ``>=50`` cell (the polar
    risk region), and one cell per latitude band present — the per-band split that
    would otherwise let a metric pass globally while failing above 50°.
    """
    bands = bands_present(scores)
    ge50 = [s for s in scores if abs(s.lat) >= 50.0]
    table: dict[str, dict[str, Rho]] = {}
    for metric in METRICS:
        row: dict[str, Rho] = {POOLED: metric_rho(scores, metric)}
        if ge50:
            row[POOLED_GE50] = metric_rho(ge50, metric)
        for band in bands:
            row[band] = metric_rho([s for s in scores if s.band == band], metric)
        table[metric] = row
    return table


# --- Live driver --------------------------------------------------------------


def _load_frame(frame_cache: Path | None) -> list[CogRef]:
    """The full COG frame — from a cached stem list if present, else list S3 once.

    A deliberate second copy of :func:`dataset.prevalence._load_frame`: the two
    drivers share nothing else, and a one-function import coupling for a five-line
    helper is not worth the module dependency.
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


def measure(
    reader: CogReader,
    cogs: Sequence[CogRef],
    *,
    tile: int = config.TILE_PX,
    factor: int = OVERVIEW_FACTOR,
    workers: int = 8,
    progress: Callable[[int, str], object] | None = None,
) -> tuple[list[WindowScore], list[tuple[str, str]]]:
    """Score every COG in ``cogs`` (parallel reads), collecting per-COG anomalies.

    Returns ``(scores, anomalies)``, ``scores`` flattened across all COGs. An anomaly
    is ``(stem, reason)`` for a COG that could not be read; it is surfaced and its
    windows are left out rather than folded in as zeros. The thread pool overlaps the
    network-bound reads — the reads are I/O-bound, so the result is identical regardless
    of ``workers``.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    scores: list[WindowScore] = []
    anomalies: list[tuple[str, str]] = []
    bar = progress(len(cogs), "calibration") if progress else None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(tile_scores, reader, cog, tile=tile, factor=factor): cog
            for cog in cogs
        }
        for fut in as_completed(futs):
            cog = futs[fut]
            try:
                scores.extend(fut.result())
            except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed as data
                anomalies.append((cog.stem, f"{type(exc).__name__}: {exc}"))
            if bar:
                bar.update(1)
    if bar:
        bar.close()
    scores.sort(key=lambda s: (s.stem, s.win_row0, s.win_col0))
    return scores, anomalies


# --- Report assembly ----------------------------------------------------------

#: Human-readable note per metric for the committed table — the two suspects, and the
#: one column that is ≈1.0 by construction, are flagged inline so a reader is not left
#: to infer why a ρ is what it is.
_METRIC_NOTES: Final = {
    TPI_METRIC: "scored on the 2× box (#22); ρ≈1.0 by construction — same ground both reads",
    "geomorphon_entropy": "LOS radius is a fixed cell count → different ground reach per read; a low ρ is the ruler moving, not the metric failing (#17)",
    "normals_dispersion": "VRM piles up near 0 on smooth terrain (#17); watch the tail, not the mean",
}


def _rho_cell(r: Rho) -> str:
    if not np.isfinite(r.rho):
        return f"— (n={r.n})"
    mark = "✓" if r.passes else "✗"
    return f"{r.rho:.4f} {mark} (n={r.n})"


def build_report(
    scores: Sequence[WindowScore],
    anomalies: Sequence[tuple[str, str]],
    *,
    seed: int,
    requested_n: int,
    cogs_read: int,
    factor: int,
    code_version: str,
    snapshot_date: str,
) -> str:
    """Assemble the committed markdown calibration table from scored windows."""
    table = rho_table(scores)
    bands = bands_present(scores)
    columns = [POOLED, POOLED_GE50, *bands]
    total_windows = len(scores)
    tpi_undefined = sum(
        1
        for s in scores
        if not np.isfinite(s.full[TPI_METRIC]) or not np.isfinite(s.overview[TPI_METRIC])
    )

    out: list[str] = []
    w = out.append
    w(f"# Overview-vs-full-res metric calibration — ×{factor} vs full res, Spearman ρ")
    w("")
    w("<!-- GENERATED by `python -m dataset.calibration`. Do not hand-edit numbers; "
      "re-run to refresh. -->")
    w("")
    w(f"Resolves issue #20 (part of the Phase-A map, #13; Gate A3). Every metric in "
      f"`dataset.metrics.METRICS` is scored **twice per window** — once from the "
      f"full-resolution read, once from the ×{factor} overview of the same ground — and "
      f"the table below is the **Spearman rank correlation** per metric between the two "
      f"reads. Rank, not absolute agreement: the metrics feed k-means clustering into "
      f"terrain strata, so what matters is that windows keep their relative order "
      f"(plan §A3). The gate is **ρ ≥ {RHO_GATE}**. (The plan's eight metric rows are "
      f"nine functions — slope's mean and σ share a row — so nine rows appear below.)")
    w("")
    w(f"**This table reports the numbers; it does not make the call.** Deciding "
      f"drop-or-full-res per metric is Gate A3's own ticket (#21) — an explicit human "
      f"decision. The ✓/✗ marks are ρ against the {RHO_GATE} gate, nothing more.")
    w("")
    w(f"- Sample: the deterministic {requested_n}-tile calibration draw "
      f"(`dataset.sampling.calibration_sample`, **balanced** across latitude bands, "
      f"seed `{seed:#x}`), so every band — the poles included — has equal power. "
      f"COGs read: **{cogs_read}**"
      f"{'' if not anomalies else f' ({len(anomalies)} anomalies — see below)'}.")
    w(f"- Windows scored: **{total_windows:,}** land-bearing windows (ADR-0001 §2 grid, "
      f"tile {config.TILE_PX}²). Ocean-only windows are skipped; coastal windows are "
      f"kept — they are where resolution-dependent masking drift lives (ADR-0001 §3).")
    w(f"- `tpi_proxy` undefined on **{tpi_undefined:,}** windows (box ran past the tile edge, "
      f"or its core/frame is all ocean → NaN for that column only, #22); they drop from "
      f"its pairwise-complete ρ and are scored by the other eight.")
    w(f"- Provenance: code `{code_version}`, snapshot `{snapshot_date}`.")
    w("")
    w("Copyright / attribution (mask spec §5, Art. 6(b)): *produced using Copernicus "
      "WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
      "2014-2018 provided under COPERNICUS by the European Union and ESA; all rights "
      "reserved.* Art. 6(c): *The organisations in charge of the Copernicus programme "
      "by law or by delegation do not incur any liability for any use of the "
      "Copernicus WorldDEM-30.*")
    w("")

    w("## 1. Spearman ρ per metric — pooled, ≥50°, and per band")
    w("")
    w("Each cell is `ρ ✓/✗ (n)`: the rank correlation over `n` windows where both reads "
      "are defined (pairwise-complete). `—` means too few defined pairs to correlate.")
    w("")
    header = "| Metric | " + " | ".join(
        {POOLED: "pooled", POOLED_GE50: "≥50°"}.get(c, f"{c}°") for c in columns
    ) + " |"
    w(header)
    w("|" + "---|" * (len(columns) + 1))
    for metric in METRICS:
        row = table[metric]
        cells = " | ".join(_rho_cell(row[c]) for c in columns if c in row)
        w(f"| `{metric}` | {cells} |")
    w("")

    w("## 2. What each column means")
    w("")
    w("- **pooled** — every land window, all bands. The headline agreement.")
    w("- **≥50°** — every window at or above 50° latitude, pooled. The band-local "
      "artifact A3 hunts (anisotropic pixels, the sampling step-down at the band edges, "
      "plan §11.4) shows here first; a metric that passes pooled and fails this column "
      "is a finding, not a pass.")
    w("- **per band** — one column per resolution band Copernicus defines. Read the "
      "poleward bands for where a metric starts to drift.")
    w("")

    w("## 3. Metric notes (flagged, not judged)")
    w("")
    w("Three columns carry a known caveat that shapes how their ρ should be read. None "
      "of this is a drop-or-full-res call — that is #21.")
    w("")
    for metric, note in _METRIC_NOTES.items():
        w(f"- **`{metric}`** — {note}.")
    w("")

    if anomalies:
        w("## Anomalies (surfaced, not swallowed)")
        w("")
        w("COGs that could not be read. Their windows are excluded from the table above "
          "rather than folded in as zeros.")
        w("")
        for stem, reason in anomalies:
            w(f"- `{stem}` — {reason}")
        w("")

    return "\n".join(out) + "\n"


def to_rows(scores: Sequence[WindowScore]) -> list[dict[str, float | str | int]]:
    """Flat per-window records for the auditable CSV behind the ρ table."""
    rows: list[dict[str, float | str | int]] = []
    for s in scores:
        rec: dict[str, float | str | int] = {
            "stem": s.stem,
            "band": s.band,
            "lon": s.lon,
            "lat": s.lat,
            "win_row0": s.win_row0,
            "win_col0": s.win_col0,
        }
        for metric in METRICS:
            rec[f"{metric}_full"] = s.full[metric]
            rec[f"{metric}_ov"] = s.overview[metric]
        rows.append(rec)
    return rows


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def _tqdm(total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="cog", dynamic_ncols=True)
    except ImportError:  # pragma: no cover
        return None


def main(argv: list[str] | None = None) -> int:
    """``python -m dataset.calibration`` — measure and write the committed table."""
    import argparse
    import csv
    import logging

    from .sampling import CALIBRATION_SAMPLE_N, calibration_sample

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("calibration")

    p = argparse.ArgumentParser(prog="python -m dataset.calibration")
    p.add_argument("--n", type=int, default=None, help="sample size (default: plan's 100)")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--workers", type=int, default=8, help="parallel whole-COG S3 reads")
    p.add_argument("--factor", type=int, default=OVERVIEW_FACTOR, help="overview decimation")
    p.add_argument("--out", type=Path, default=Path("docs/research/overview-calibration.md"))
    p.add_argument(
        "--csv", type=Path, default=Path("docs/research/overview-calibration-per-window.csv")
    )
    p.add_argument(
        "--frame-cache",
        type=Path,
        default=Path("data/cog_frame.json"),
        help="cache the S3 prefix listing here so re-runs skip it (blank to always list)",
    )
    p.add_argument("--limit", type=int, default=None, help="cap COGs read (debugging)")
    args = p.parse_args(argv)

    requested_n = args.n or CALIBRATION_SAMPLE_N

    log.info("loading COG frame ...")
    refs = _load_frame(args.frame_cache)
    log.info("frame: %d COGs", len(refs))

    sample = calibration_sample(refs, requested_n, seed=args.seed)
    if args.limit:
        sample = sample[: args.limit]
    log.info("sample: %d COGs (seed %#x, balanced)", len(sample), args.seed)

    from .tiling import RasterioCogReader

    reader = RasterioCogReader()
    scores, anomalies = measure(
        reader, sample, factor=args.factor, workers=args.workers, progress=_tqdm
    )
    cogs_read = len({s.stem for s in scores})
    log.info("scored %d windows from %d COGs, %d anomalies",
             len(scores), cogs_read, len(anomalies))

    from .provenance import git_sha

    report = build_report(
        scores,
        anomalies,
        seed=args.seed,
        requested_n=requested_n,
        cogs_read=cogs_read,
        factor=args.factor,
        code_version=git_sha(),
        snapshot_date=_today(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    log.info("wrote %s", args.out)

    rows = to_rows(scores)
    if rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        log.info("wrote %s", args.csv)

    table = rho_table(scores)
    fails = [
        m for m in METRICS
        if not table[m][POOLED].passes and np.isfinite(table[m][POOLED].rho)
    ]
    print(
        f"\nScored {len(scores)} windows from {cogs_read} COGs.\n"
        f"  pooled ρ below {RHO_GATE}: {fails or 'none'}\n"
        f"  -> {args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
