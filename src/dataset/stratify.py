"""Terrain screening, ruggedness scoring, and stratified quota selection.

This is the heart of ADR-0001 §3-5: a single x8-overview read per COG yields both
the ocean screen (exact-zero fraction) and the TRI ruggedness score for every
candidate window; the global TRI distribution is then cut into quintile strata,
and each stratum's quota is filled round-robin across 10-degree geo-cells with a
per-COG cap so rugged terrain is oversampled without clustering in one range.

All functions are pure (numpy in, numpy/objects out) so the recipe is testable
without reading a byte from S3.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from . import config


# --- Zero-fill screen (ADR-0001 §3, Amendment 2) ------------------------------


def zero_fraction(window: np.ndarray) -> float:
    """Fraction of pixels equal to Copernicus's exact ocean fill (``0.0``)."""
    return float(np.mean(window == config.OCEAN_FILL_VALUE))


def is_zero_fill(window: np.ndarray) -> bool:
    """Whether a window is mostly zero-fill and should be dropped (§3).

    True iff the exact-zero fraction reaches :data:`config.LAND_ZERO_FRAC_DROP`
    (0.75). Coastlines (well under 75% sea) are kept; genuine sub-sea-level land
    is negative, not zero, so it is never mistaken for fill.

    **Not named ``is_ocean``** (ADR-0001 §3 Amendment 2, issue #34). A *pixel*
    value of ``0.0`` really is Copernicus's ocean fill — ``WBM == 1`` was measured
    to be a strict subset of exact-zero, 0 counter-examples in 144.5 M pixels. But
    a *window* that is 75% zeros is not necessarily ocean: Copernicus paints
    sea-level rivers and lakes (WBM 2/3) to exactly ``0.0`` too, so this predicate
    fires on Bangladesh delta and Scheldt polder water. It screens zero-fill, and
    the catalog books its attrition under ``zero_fill`` — not ``ocean``.
    """
    return zero_fraction(window) >= config.LAND_ZERO_FRAC_DROP


# --- TRI ruggedness (ADR-0001 §4) ---------------------------------------------


def tri_riley(a: np.ndarray) -> np.ndarray:
    """Per-pixel Terrain Ruggedness Index (Riley): mean absolute elevation
    difference to the 8 neighbours.

    Computed on the x8 overview by the caller. Edge pixels use replicate padding
    (out-of-frame neighbours contribute a zero difference), which biases the very
    border slightly low but is immaterial once averaged over a ~28x28 window.
    """
    a = a.astype(np.float64, copy=False)
    pad = np.pad(a, 1, mode="edge")
    total = np.zeros(a.shape, dtype=np.float64)
    h, w = a.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neigh = pad[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
            total += np.abs(a - neigh)
    return total / 8.0


def window_overview_slice(
    *,
    win_row0: int,
    win_col0: int,
    tile: int,
    full_h: int,
    full_w: int,
    ov_h: int,
    ov_w: int,
) -> tuple[int, int, int, int]:
    """Overview-pixel bounds ``(o_r0, o_r1, o_c0, o_c1)`` for a full-res window.

    Maps the full-resolution window rectangle onto the overview by the exact
    row/column downsample ratios (the x8 overview of a 3600 COG is 450 px, so the
    ratio is 1/8). Bounds are clamped and kept non-empty.
    """
    sr = ov_h / full_h
    sc = ov_w / full_w
    o_r0 = int(round(win_row0 * sr))
    o_r1 = int(round((win_row0 + tile) * sr))
    o_c0 = int(round(win_col0 * sc))
    o_c1 = int(round((win_col0 + tile) * sc))
    o_r0 = max(0, min(o_r0, ov_h - 1))
    o_c0 = max(0, min(o_c0, ov_w - 1))
    o_r1 = max(o_r0 + 1, min(o_r1, ov_h))
    o_c1 = max(o_c0 + 1, min(o_c1, ov_w))
    return o_r0, o_r1, o_c0, o_c1


def window_tri(tri_map: np.ndarray, *, o_r0: int, o_c0: int, o_r1: int, o_c1: int) -> float:
    """Mean TRI over a window's overview footprint."""
    return float(tri_map[o_r0:o_r1, o_c0:o_c1].mean())


# --- Quantile strata (ADR-0001 §4) --------------------------------------------


def quantile_edges(tri_values: np.ndarray, n_strata: int = config.N_STRATA) -> np.ndarray:
    """Internal quantile cut points splitting the TRI distribution into
    ``n_strata`` equal-population strata (data-driven, not fixed thresholds)."""
    qs = np.linspace(0.0, 1.0, n_strata + 1)[1:-1]  # e.g. .2 .4 .6 .8
    return np.quantile(np.asarray(tri_values, dtype=np.float64), qs)


def assign_strata(tri_values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign each TRI value its stratum index ``0..n_strata-1`` (monotone)."""
    strata = np.searchsorted(edges, np.asarray(tri_values, dtype=np.float64), side="right")
    return np.clip(strata, 0, len(edges)).astype(np.int64)


# --- Quota selection (ADR-0001 §5) --------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A scored candidate window awaiting selection into the corpus."""

    index: int  # caller's opaque handle back to the full candidate record
    stratum: int
    geocell: str
    cog_stem: str
    tri: float
    tile_id: str


def select_quota(
    candidates: list[Candidate],
    *,
    per_stratum_quota: int = config.PER_STRATUM_QUOTA,
    per_tile_cap: int = config.PER_TILE_WINDOW_CAP,
    seed: int = config.RANDOM_SEED,
) -> list[Candidate]:
    """Select up to ``per_stratum_quota`` windows per stratum, spread across
    geo-cells round-robin, with a per-COG window cap (ADR-0001 §5).

    Within each stratum the candidates are bucketed by 10-degree geo-cell and one
    is drawn from each cell in turn until the quota is met or the stratum is
    exhausted; a COG that has already contributed ``per_tile_cap`` windows is
    skipped. Order within a bucket is a seeded shuffle, so the spread is
    reproducible but not biased by tile-id ordering. A stratum with too few
    candidates is simply under-filled (the caller logs the shortfall).
    """
    rng = np.random.default_rng(seed)
    per_cog_count: dict[str, int] = defaultdict(int)
    selected: list[Candidate] = []

    by_stratum: dict[int, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_stratum[c.stratum].append(c)

    for stratum in sorted(by_stratum):
        buckets: dict[str, list[Candidate]] = defaultdict(list)
        for c in by_stratum[stratum]:
            buckets[c.geocell].append(c)
        # Deterministic-but-unbiased order within each geo-cell.
        for gc in buckets:
            order = rng.permutation(len(buckets[gc]))
            buckets[gc] = [buckets[gc][i] for i in order]

        cursor = {gc: 0 for gc in buckets}
        active = deque(sorted(buckets))  # deterministic geo-cell visit order
        picked = 0
        while picked < per_stratum_quota and active:
            gc = active.popleft()
            lst = buckets[gc]
            chosen: Candidate | None = None
            while cursor[gc] < len(lst):
                cand = lst[cursor[gc]]
                cursor[gc] += 1
                if per_cog_count[cand.cog_stem] < per_tile_cap:
                    chosen = cand
                    break
            if chosen is not None:
                selected.append(chosen)
                per_cog_count[chosen.cog_stem] += 1
                picked += 1
                if cursor[gc] < len(lst):
                    active.append(gc)  # cell still has candidates -> back of queue
            # bucket exhausted -> drop it from the rotation
    return selected
