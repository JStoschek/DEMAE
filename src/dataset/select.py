"""The sampler — draw the debug-tier selection list (Phase D4, issue #44).

This is the destination the whole Phase-D map has been walking toward. Everything it
needs arrives settled from the tickets that block it, and this module only makes them
real, reproducibly:

- **the frame** (:mod:`dataset.frame`) — which 3,052,735 kept rows k-means sees, and the
  committed scaler that standardizes them;
- **the strata** (:mod:`dataset.centroids`) — the k=20 centroids, and :func:`centroids.assign`,
  the deterministic ``argmin`` that labels every row;
- **the allocation** ([`docs/research/d2-allocation.md`](../../docs/research/d2-allocation.md)) —
  pure Neyman ``n_h ∝ N_h·σ_h`` with a within-stratum coastal floor, apportioned to an
  exact budget by largest-remainder;
- **the within-stratum rule** ([`docs/research/d3-fscs-cost.md`](../../docs/research/d3-fscs-cost.md)) —
  FSCS sub-k-means at ``k = n_h``, the window nearest each sub-centroid, geo-cell
  round-robin with fall-through as the tie-break.

## Budget is an argument; the fitted artifacts are inputs

The 50,000-window debug tier is one call; Phase F re-runs at 200–500k against the *same*
committed centroids and scaler (plan §F). So :func:`select_windows` takes ``budget`` as a
parameter and re-derives the allocation from the frame at that budget — it never
re-derives the strata. The committed [`d2-allocation-nh.json`](../../docs/research/d2-allocation-nh.json)
is the frozen 50k reference this code reproduces, not a table it reads.

## Determinism is the acceptance criterion

Two runs from the committed artifacts must produce byte-identical selections, and must
keep doing so across library upgrades. Two things make that hold:

1. **The persisted selection list is the contract, not the seed.** #43 measured that
   multithreaded Lloyd is *not* reproducible even at a fixed seed (same-seed Jaccard
   0.42), because scikit-learn sums centroids in a thread-scheduling-dependent float
   order and "nearest window" amplifies the ULP drift into different picks. The fit is
   pinned single-threaded (:func:`_fit_subkmeans` under ``threadpool_limits(1)``, the
   lever #43 found restores same-seed Jaccard 1.0), but the guarantee the map rests on
   is that the emitted window ids are shipped and re-loaded — exactly as #40 concluded
   for the centroids and #43 for the selection.
2. **Every tie-break is a keyed BLAKE2b hash, never a NumPy ``Generator``.** NumPy
   guarantees its bit-generator streams across releases but not the distributions layered
   on them (``sampling.py``'s house rule); the round-robin's within-cell order and the
   fill pass both order by :func:`dataset.sampling._shuffle_key` so the draw is stable for
   the life of the project and independent of the order rows arrive from the catalog.

## What lands on disk

The selection list (``window_id``, ``stratum_id``, ``provenance``) — the columns Phase
E1's manifest addendum consumes. ``provenance`` is one of exactly two labels,
:data:`PROV_BASE` and :data:`PROV_COASTLINE`: the coastal floor is a within-stratum
reservation, so each stratum runs FSCS twice — once over its coastal rows (tagged
``floor:coastline``) for ``c_h`` picks, once over the rest (tagged ``base``) for
``base_h`` — and a pick's provenance is fixed by *which* draw selected it, not by
selection history (#42 §4). The realized windows-per-COG distribution is reported beside
it: #42 retired the per-COG cap as inert at the debug tier (~2/COG) but requires the
sampler to measure it at every tier, so Phase F can decide on evidence whether the
denser scale tier needs a cap reintroduced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np

from . import catalog, centroids as centroids_mod, config, frame as frame_mod
from .sampling import _shuffle_key

# --- The two-value provenance vocabulary (#42 §4) -----------------------------

#: A window bought by its stratum's Neyman base draw (FSCS over the non-coastal rows).
PROV_BASE: Final = "base"
#: A window bought by the within-stratum coastal reservation (FSCS over the coastal rows).
PROV_COASTLINE: Final = "floor:coastline"

# --- Locked allocation constants (#42) ----------------------------------------

#: The debug-tier budget. Not hard-wired anywhere below — it is the default of a
#: parameter, because Phase F re-runs the same code at 200–500k (plan §F).
DEBUG_BUDGET: Final = 50_000

#: Coastal floor as a fraction of budget — the *frame rate* of coastline in the kept
#: frame (40,815 / 3,052,735 = 1.34%), rounded to the value #42 committed so that
#: ``round(COAST_FLOOR_RATE · 50,000) = 670`` reproduces the shipped table exactly. It
#: is a frame-rate floor, not an over-boost: coast is a location tag, not a descriptor
#: shape, so the honest minimum guarantees coast is never *under*-represented relative to
#: reality without front-running E2's validation (#42 §4).
COAST_FLOOR_RATE: Final = 0.0134

#: The §D2 ``max(500, 1%·budget)`` per-stratum floor. Measured **inert** at every budget
#: (smallest Neyman share 1.47% > 1%; #42 §2) and retained only as a non-binding
#: raise-to-floor clamp — free insurance against a Phase-F re-fit changing stratum
#: proportions. :func:`allocate` records how many strata it fires on (0 here).
FLOOR_CLAMP_MIN: Final = 500
FLOOR_CLAMP_FRAC: Final = 0.01

#: Where the shipped debug-tier selection list lives. Committed — ``data/`` is gitignored
#: and this file *is* the determinism contract, re-loaded by Phase E1 rather than
#: regenerated. CSV, sorted, so a re-run diffs byte-for-byte in git.
SELECTION_PATH: Final = Path("artifacts/phase_d/selection_debug_50k.csv")

#: Lloyd (full batch), matching the centroid fit and #43's pricing — MiniBatch is the
#: named fallback only if a Phase-F budget ever makes Lloyd's cost bite (it does not
#: here: ~8 min for the whole 50k tier).
ALGORITHM: Final = "lloyd"


# --- Largest-remainder apportionment ------------------------------------------


def largest_remainder(
    weights: np.ndarray, total: int, *, cap: np.ndarray | None = None
) -> np.ndarray:
    """Integer apportionment of ``total`` in proportion to ``weights``, summing *exactly*.

    Floor the exact shares, then hand the leftover slots one each to the largest
    fractional remainders — deterministic, ties broken by lowest index, **no RNG** (the
    map's determinism contract bans NumPy's random distributions from any committed
    number). An optional per-cell ``cap`` clamps and re-spills the overflow in
    descending-weight order, so the coastal split can never ask a stratum for more
    coastal windows than it holds.

    This is the one apportionment primitive both the Neyman base and the coastal floor go
    through; #42's ``prototypes/d2_allocation_proto.py`` is its throwaway ancestor.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1-D; got shape {weights.shape}")
    if total < 0:
        raise ValueError(f"total must be non-negative; got {total}")

    if weights.sum() <= 0:
        base = np.zeros(len(weights), dtype=np.int64)
    else:
        exact = weights / weights.sum() * total
        base = np.floor(exact).astype(np.int64)
        remainder = exact - base
        leftover = int(round(total - base.sum()))
        order = sorted(range(len(weights)), key=lambda i: (-remainder[i], i))
        for i in order[:leftover]:
            base[i] += 1

    if cap is not None:
        cap = np.asarray(cap, dtype=np.int64)
        over = base > cap
        if over.any():
            spill = int((base[over] - cap[over]).sum())
            base[over] = cap[over]
            room = cap - base
            for i in sorted(range(len(weights)), key=lambda i: (-weights[i], i)):
                if spill <= 0:
                    break
                give = min(spill, int(room[i]))
                base[i] += give
                spill -= give
    return base


# --- Allocation ---------------------------------------------------------------


@dataclass(frozen=True)
class Allocation:
    """The per-stratum quota table — what each stratum draws, split base vs coastal.

    Every array is length ``k`` and indexed by ``stratum_id``. ``n_h = base + coastal``
    exactly; ``sum(n_h) == budget`` exactly; ``sum(coastal) == coast_floor`` exactly.
    """

    N_h: np.ndarray  # kept rows per stratum (occupancy)
    sigma_h: np.ndarray  # mean within-stratum distance to centroid, standardized space
    coast_h: np.ndarray  # coastal rows per stratum
    n_h: np.ndarray  # total quota per stratum (Neyman, exact budget)
    coastal: np.ndarray  # c_h — reserved coastal picks per stratum
    base: np.ndarray  # base_h = n_h - c_h
    budget: int
    coast_floor: int
    floor_clamp: int
    floor_clamp_strata_below: int  # how many strata the (inert) clamp would raise

    @property
    def k(self) -> int:
        return int(self.n_h.shape[0])

    def rows(self) -> list[dict[str, int | float]]:
        """One JSON-ready dict per stratum, in the shape of ``d2-allocation-nh.json``."""
        return [
            {
                "stratum_id": h,
                "N_h": int(self.N_h[h]),
                "sigma_h": round(float(self.sigma_h[h]), 4),
                "coast_h": int(self.coast_h[h]),
                "n_h": int(self.n_h[h]),
                "coastal": int(self.coastal[h]),
                "base": int(self.base[h]),
            }
            for h in range(self.k)
        ]


def stratum_sigma(Z: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Mean within-stratum distance to the centroid, in standardized space — the σ_h Neyman weights.

    ADR-0004's "standardize, don't whiten" makes this readable as sd-units: a stratum's
    σ_h is how many standard deviations its rows sit, on average, from their centroid.
    Computed in float32 to match :func:`centroids.assign` and the committed centroids,
    so the resulting ``N_h·σ_h`` weights reproduce #42's table bit-for-bit.
    """
    Z = np.asarray(Z, dtype=np.float32)
    C = np.asarray(centers, dtype=np.float32)
    k = C.shape[0]
    out = np.empty(k, dtype=np.float64)
    for h in range(k):
        rows = Z[labels == h]
        out[h] = float(np.linalg.norm(rows - C[h], axis=1).mean()) if rows.size else 0.0
    return out


def allocate(
    N_h: np.ndarray,
    sigma_h: np.ndarray,
    coast_h: np.ndarray,
    budget: int,
    *,
    coast_floor_rate: float = COAST_FLOOR_RATE,
    floor_clamp_min: int = FLOOR_CLAMP_MIN,
    floor_clamp_frac: float = FLOOR_CLAMP_FRAC,
) -> Allocation:
    """Compose #42's ordered rule into a per-stratum quota table for any ``budget``.

    1. **Neyman base.** ``w_h = N_h·σ_h``; largest-remainder to exactly ``budget`` → ``n_h``.
    2. **Coastal reservation.** ``F = round(coast_floor_rate·budget)`` apportioned by
       largest-remainder on ``coast_h``, capped at ``min(n_h, coast_h)`` → ``c_h``.
    3. **Base split.** ``base_h = n_h − c_h``.

    The ``max(500, 1%·budget)`` clamp is checked and *reported*, never applied — it is
    measured inert at every budget (#42 §2). ``n_h < N_h`` is asserted: impossible to
    violate at k=20 (max ``n_h`` 3,701 ≪ min ``N_h`` 32,229), the guard is for a Phase-F
    re-fit that changes stratum proportions.
    """
    N_h = np.asarray(N_h, dtype=np.int64)
    sigma_h = np.asarray(sigma_h, dtype=np.float64)
    coast_h = np.asarray(coast_h, dtype=np.int64)
    if budget <= 0:
        raise ValueError(f"budget must be positive; got {budget}")

    n_h = largest_remainder(N_h.astype(np.float64) * sigma_h, budget)
    if not (n_h < N_h).all():
        offending = [int(h) for h in np.flatnonzero(n_h >= N_h)]
        raise ValueError(
            f"Neyman asked strata {offending} for more windows than they hold — a "
            "Phase-F re-fit changed stratum proportions; reduce the budget or re-fit"
        )

    floor_clamp = max(floor_clamp_min, int(floor_clamp_frac * budget))
    floor_clamp_strata_below = int((n_h < floor_clamp).sum())

    coast_floor = int(round(coast_floor_rate * budget))
    c_h = largest_remainder(
        coast_h.astype(np.float64), coast_floor, cap=np.minimum(n_h, coast_h)
    )
    base_h = n_h - c_h

    return Allocation(
        N_h=N_h,
        sigma_h=sigma_h,
        coast_h=coast_h,
        n_h=n_h,
        coastal=c_h,
        base=base_h,
        budget=budget,
        coast_floor=coast_floor,
        floor_clamp=floor_clamp,
        floor_clamp_strata_below=floor_clamp_strata_below,
    )


# --- FSCS within-stratum selection --------------------------------------------


def _fit_subkmeans(S: np.ndarray, quota: int, seed: int) -> np.ndarray:
    """Full-Lloyd sub-k-means at ``k = quota``; return the sub-centroids.

    Pinned single-threaded via ``threadpool_limits(1)`` — #43's lever: multithreaded
    Lloyd sums centroids in a thread-scheduling-dependent float order, and "nearest
    window" amplifies the ULP drift into a 58%-different selection. sklearn (and
    threadpoolctl) are imported lazily: only this fit needs them, and the reproducible
    read path (re-loading the persisted list) stays pure-NumPy.
    """
    from sklearn.cluster import KMeans

    # threadpoolctl ships with scikit-learn; import it directly rather than degrading to a
    # no-op on ImportError — a missing thread pin silently breaks the determinism the whole
    # module rests on (#43), so its absence must fail loud, not quietly.
    from threadpoolctl import threadpool_limits

    S = np.ascontiguousarray(S, dtype=np.float32)
    with threadpool_limits(limits=1):
        km = KMeans(n_clusters=quota, n_init=1, random_state=seed, algorithm=ALGORITHM)
        km.fit(S)
    return km.cluster_centers_.astype(np.float32)


#: How many nearest rows per sub-centroid the round-robin considers before falling
#: through. Generous: two sub-centroids in one dense pocket must both find a distinct,
#: geographically spread representative, and #43 warns that a rule that just drops the
#: collision under-fills the quota. The rare shortfall past this is closed exactly by
#: :func:`_fill_shortfall`.
_NEIGHBOURS: Final = 24


def _nearest_representatives(
    S: np.ndarray,
    centers: np.ndarray,
    geocell: np.ndarray,
    window_id: np.ndarray,
    seed: int,
) -> list[int]:
    """One representative row per sub-centroid: nearest window, geo-cell round-robin ties.

    §D3's rule, as #43 measured it. For each sub-centroid, walk its nearest rows and take
    the first *unused* one that sits in the least-used geo-cell so far — a row whose cell
    is already over-used yields to a next-nearest row from a less-used cell. Without this
    tie-break two sub-centroids in the same dense pocket resolve to the same over-
    represented cell and the stratum quietly re-concentrates geographically while looking
    well spread in descriptor space. Ties in the ``(cell use, rank)`` key are broken by a
    keyed BLAKE2b of the window id, the house determinism rule.
    """
    from scipy.spatial import cKDTree

    k_query = min(_NEIGHBOURS, S.shape[0])
    _, candidates = cKDTree(S).query(centers, k=k_query, workers=-1)
    candidates = np.atleast_2d(candidates)  # always (n_centres, k_query) here

    taken: set[int] = set()
    cell_use: dict[str, int] = {}
    picked: list[int] = []
    for r in range(candidates.shape[0]):
        best: int | None = None
        best_key: tuple[int, int, bytes] | None = None
        for rank, row in enumerate(candidates[r]):
            row = int(row)
            if row in taken:
                continue
            cell = str(geocell[row])
            key = (cell_use.get(cell, 0), rank, _shuffle_key(str(window_id[row]), seed))
            if best_key is None or key < best_key:
                best, best_key = row, key
        if best is None:
            continue  # all neighbours taken — closed by the fill pass
        taken.add(best)
        cell_use[str(geocell[best])] = cell_use.get(str(geocell[best]), 0) + 1
        picked.append(best)
    return picked


def _fill_shortfall(
    S: np.ndarray,
    centers: np.ndarray,
    picked: list[int],
    quota: int,
    window_id: np.ndarray,
    seed: int,
) -> list[int]:
    """Top a short draw up to exactly ``quota`` with the unused rows nearest any centroid.

    The round-robin can fall short of ``quota`` only when a dense pocket exhausts every
    one of a sub-centroid's :data:`_NEIGHBOURS` candidates (rare, and never at the debug
    tier). #43 is explicit that the fix is *fall-through to the next-nearest row*, not a
    dedup that ships fewer than ``n_h`` windows — so this closes the gap deterministically
    rather than accepting a shortfall. Unused rows are ranked by distance to their nearest
    sub-centroid, ties by keyed BLAKE2b of the window id.
    """
    if len(picked) >= quota:
        return picked
    from scipy.spatial import cKDTree

    taken = set(picked)
    unused = np.array([i for i in range(S.shape[0]) if i not in taken], dtype=np.int64)
    if unused.size == 0:
        return picked
    # Distance from each unused row to its *nearest* sub-centroid — the true nearest-
    # centroid distance (a KDTree over the centres, so no (N·quota) matrix is formed).
    dist, _ = cKDTree(centers).query(S[unused], k=1, workers=-1)
    order = sorted(
        range(unused.size),
        key=lambda j: (float(dist[j]), _shuffle_key(str(window_id[unused[j]]), seed)),
    )
    for j in order:
        if len(picked) >= quota:
            break
        picked.append(int(unused[j]))
    return picked


def fscs_select(
    S: np.ndarray,
    quota: int,
    geocell: np.ndarray,
    window_id: np.ndarray,
    *,
    seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    """Select exactly ``quota`` distinct rows of ``S`` by FSCS — local indices, sorted.

    ``S`` is one stratum's rows in standardized space (already one *tag* group — coastal
    or base). Sub-k-means at ``k = quota`` clusters the group into as many sub-clusters as
    it has quota; each sub-centroid contributes its nearest window under the geo-cell
    round-robin (:func:`_nearest_representatives`), and any shortfall is filled exactly
    (:func:`_fill_shortfall`). Returns ``≤ len(S)`` local indices — when ``quota`` meets or
    exceeds the group size every row is taken.
    """
    if quota <= 0:
        return np.empty(0, dtype=np.int64)
    S = np.ascontiguousarray(S, dtype=np.float32)
    if quota >= S.shape[0]:
        return np.arange(S.shape[0], dtype=np.int64)

    centers = _fit_subkmeans(S, quota, seed)
    picked = _nearest_representatives(S, centers, geocell, window_id, seed)
    picked = _fill_shortfall(S, centers, picked, quota, window_id, seed)
    return np.array(sorted(picked[:quota]), dtype=np.int64)


# --- Per-stratum work, fanned across processes --------------------------------


class _StratumJob(NamedTuple):
    """One stratum's self-contained FSCS work — everything a worker needs, no shared state."""

    stratum_id: int
    Z: np.ndarray  # (N_h, 6) standardized rows of this stratum
    coastal: np.ndarray  # (N_h,) bool — which rows are coastal
    geocell: np.ndarray  # (N_h,) geo-cell ids
    window_id: np.ndarray  # (N_h,) window ids
    c_h: int  # coastal reservation
    base_h: int  # base quota
    seed: int


def _select_stratum(job: _StratumJob) -> list[tuple[str, int, str]]:
    """Draw one stratum's picks — two FSCS runs, coastal then base — as ``(wid, h, prov)``.

    The unit of parallelism: a stratum owns its rows and quotas and touches nothing else,
    so this runs unchanged in-process or across a process pool. The fit inside
    :func:`fscs_select` is pinned single-threaded, so a worker is bit-reproducible on its
    own core; the parent re-sorts the merged result, so the selection is identical whether
    the strata ran sequentially or fanned out.
    """
    out: list[tuple[str, int, str]] = []
    for mask, quota, tag in (
        (job.coastal, job.c_h, PROV_COASTLINE),
        (~job.coastal, job.base_h, PROV_BASE),
    ):
        if quota <= 0:
            continue
        idx = np.flatnonzero(mask)
        picks = fscs_select(
            job.Z[idx], quota, job.geocell[idx], job.window_id[idx], seed=job.seed
        )
        for w in job.window_id[idx[picks]]:
            out.append((str(w), job.stratum_id, tag))
    return out


def _run_stratum_jobs(
    jobs: list[_StratumJob], workers: int
) -> list[list[tuple[str, int, str]]]:
    """Run every stratum's FSCS, sequentially (``workers <= 1``) or across a process pool.

    A process pool, not a thread pool: the fit holds no GIL-friendly sleep and each fit is
    already single-threaded for determinism, so real parallelism needs separate
    interpreters. Results are collected out-of-order-safe — the caller sorts — so the
    default ``ProcessPoolExecutor.map`` ordering is immaterial.
    """
    if workers <= 1 or len(jobs) <= 1:
        return [_select_stratum(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_select_stratum, jobs))


def _default_workers(k: int) -> int:
    """Fan across the cores available, one per stratum at most, leaving two for the OS."""
    import os

    return max(1, min(k, (os.cpu_count() or 2) - 2))


# --- The selection ------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """The emitted selection list plus the allocation and per-COG report behind it.

    ``window_id``, ``stratum_id`` and ``provenance`` are row-aligned and sorted by
    ``(stratum_id, window_id)`` — the columns Phase E1 consumes, in a byte-stable order.
    """

    window_id: np.ndarray
    stratum_id: np.ndarray
    provenance: np.ndarray
    allocation: Allocation
    budget: int
    fingerprint: dict[str, Any]

    @property
    def n(self) -> int:
        return int(self.window_id.shape[0])

    def cog_stems(self) -> np.ndarray:
        """The source COG stem of each selected window (``window_id`` is ``stem:row:col``)."""
        return np.array([w.rsplit(":", 2)[0] for w in self.window_id], dtype=object)

    def per_cog_distribution(self) -> dict[str, Any]:
        """Realized windows-per-COG distribution — the metric #42 requires at every tier.

        #42 retired ``PER_TILE_WINDOW_CAP`` as inert at the debug tier (~2/COG) but binds
        the sampler to *measure* it, so Phase F can decide on evidence — not the untested
        64 — whether the denser scale tier needs a cap. Reported as the summary a cap
        threshold would be read off: counts, the tail, and the densest COGs.
        """
        stems, counts = np.unique(self.cog_stems(), return_counts=True)
        order = np.argsort(counts)[::-1]
        return {
            "n_windows": self.n,
            "n_cogs": int(stems.size),
            "mean_per_cog": round(float(counts.mean()), 4),
            "median_per_cog": int(np.median(counts)),
            "max_per_cog": int(counts.max()),
            "p99_per_cog": int(np.percentile(counts, 99)),
            "histogram": {int(v): int(c) for v, c in zip(*np.unique(counts, return_counts=True))},
            "densest_cogs": [
                {"stem": str(stems[i]), "windows": int(counts[i])} for i in order[:10]
            ],
        }

    def provenance_counts(self) -> dict[str, int]:
        vals, counts = np.unique(self.provenance, return_counts=True)
        return {str(v): int(c) for v, c in zip(vals, counts)}

    def summary(self) -> dict[str, Any]:
        """Everything the d4 write-up and Phase E need that is not the list itself."""
        return {
            "budget": self.budget,
            "n_selected": self.n,
            "k": self.allocation.k,
            "coast_floor": self.allocation.coast_floor,
            "floor_clamp": self.allocation.floor_clamp,
            "floor_clamp_strata_below": self.allocation.floor_clamp_strata_below,
            "provenance_counts": self.provenance_counts(),
            "per_cog": self.per_cog_distribution(),
            "allocation": self.allocation.rows(),
            "frame_fingerprint": dict(self.fingerprint),
            "determinism": (
                "the persisted selection list is the contract; the sub-k-means fit is "
                "pinned single-threaded but not relied on for reproducibility (#43)"
            ),
        }


def select_windows(
    frame: frame_mod.Frame,
    centroids: centroids_mod.Centroids,
    scaler: frame_mod.Scaler,
    geocell: np.ndarray,
    *,
    budget: int = DEBUG_BUDGET,
    seed: int = config.RANDOM_SEED,
    workers: int = 1,
) -> Selection:
    """Draw ``budget`` windows from a loaded frame — the pure core, no I/O.

    Steps, all deterministic: standardize with the committed scaler → assign strata via
    :func:`centroids.assign` → :func:`allocate` at this budget → for each stratum run
    :func:`fscs_select` twice (coastal rows for ``c_h``, base rows for ``base_h``) and tag
    each pick's provenance by *which* draw took it. ``geocell`` is the kept frame's
    geo-cell column, row-aligned with ``frame.window_id`` — the round-robin's spatial axis,
    read off the catalog rather than re-derived (one definition of a window's cell).

    ``workers`` fans the independent per-stratum FSCS across that many *processes* (>1),
    each fit still pinned single-threaded. The strata are independent and the merged list
    is re-sorted, so **the selection is byte-identical to the ``workers=1`` path** — the
    parallelism is a pure speedup, mattering at the Phase-F scale tier where a serial
    single-threaded pass runs into the hour.
    """
    if geocell.shape[0] != frame.n:
        raise ValueError(
            f"geocell ({geocell.shape[0]}) is not row-aligned with the frame ({frame.n})"
        )

    Z = np.ascontiguousarray(frame.standardized(scaler).astype(np.float32))
    labels = centroids_mod.assign(Z, centroids)
    k = centroids.k

    sigma = stratum_sigma(Z, labels, centroids.centers)
    N_h = np.bincount(labels, minlength=k).astype(np.int64)
    coast_h = np.array(
        [int(frame.coastline[labels == h].sum()) for h in range(k)], dtype=np.int64
    )
    alloc = allocate(N_h, sigma, coast_h, budget)

    jobs = []
    for h in range(k):
        rows_h = np.flatnonzero(labels == h)
        jobs.append(
            _StratumJob(
                stratum_id=h,
                Z=Z[rows_h],
                coastal=frame.coastline[rows_h],
                geocell=geocell[rows_h],
                window_id=frame.window_id[rows_h],
                c_h=int(alloc.coastal[h]),
                base_h=int(alloc.base[h]),
                seed=seed,
            )
        )
    triples = [t for stratum in _run_stratum_jobs(jobs, workers) for t in stratum]

    # Sort by (stratum, window_id) so the emitted list is byte-stable and identical
    # regardless of the order the strata finished in.
    triples.sort(key=lambda t: (t[1], t[0]))
    return Selection(
        window_id=np.array([t[0] for t in triples], dtype=object),
        stratum_id=np.array([t[1] for t in triples], dtype=np.int64),
        provenance=np.array([t[2] for t in triples], dtype=object),
        allocation=alloc,
        budget=budget,
        fingerprint=dict(frame.fingerprint),
    )


# --- Reading the geo-cell axis off the catalog --------------------------------


def kept_geocell(catalog_dir: Path | str = Path("data/catalog_gated")) -> np.ndarray:
    """The kept frame's ``geocell`` column, row-aligned with :func:`frame.kept_frame`.

    Read straight off the gated catalog rather than re-derived from ``window_id``: the
    catalog already owns the one definition of a window's geo-cell (``dataset.geometry``
    via ``dataset.catalog``), and re-parsing it here would be a second one that could
    drift. The ``keep`` mask is applied identically to :func:`frame.kept_frame`, so the
    two line up row-for-row.
    """
    import pyarrow.parquet as pq

    catalog_dir = Path(catalog_dir)
    table = pq.read_table(catalog_dir / catalog.PARTS_DIR, columns=["keep", "geocell"])
    keep = table.column("keep").to_numpy(zero_copy_only=False).astype(bool)
    return table.column("geocell").to_numpy(zero_copy_only=False)[np.flatnonzero(keep)]


def run(
    catalog_dir: Path | str = Path("data/catalog_gated"),
    *,
    budget: int = DEBUG_BUDGET,
    seed: int = config.RANDOM_SEED,
    workers: int | None = None,
) -> Selection:
    """Load every committed artifact and draw the selection — the live entry point.

    Refuses a moved frame the same way #40/#41 do: :func:`frame.read_scaler` and
    :func:`centroids.read_centroids` are both handed the frame fingerprint, so a catalog
    that drifted from the one the artifacts were fit over raises rather than silently
    producing a plausible, wrong corpus. ``workers`` defaults to one process per core
    (leaving two for the OS); the drawn selection is identical for any worker count.
    """
    fr = frame_mod.kept_frame(catalog_dir)
    scaler = frame_mod.read_scaler(fingerprint=fr.fingerprint)
    cen = centroids_mod.read_centroids(fingerprint=fr.fingerprint)
    geocell = kept_geocell(catalog_dir)
    if workers is None:
        workers = _default_workers(cen.k)
    return select_windows(fr, cen, scaler, geocell, budget=budget, seed=seed, workers=workers)


# --- Persist ------------------------------------------------------------------


def write_selection(selection: Selection, path: Path | str = SELECTION_PATH) -> Path:
    """Write the selection list as sorted CSV — the committed determinism contract."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["window_id,stratum_id,provenance"]
    lines += [
        f"{w},{s},{p}"
        for w, s, p in zip(selection.window_id, selection.stratum_id, selection.provenance)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_summary(selection: Selection, path: Path | str) -> Path:
    """Write the JSON summary the d4 write-up reads (provenance + per-COG + allocation)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection.summary(), indent=2) + "\n")
    return path


# --- Live driver --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python -m dataset.select`` — draw the debug tier and persist the list.

    Reads the committed scaler and centroids (never re-fits — that is what keeps Phase F
    comparable), assigns, allocates at ``--budget``, runs FSCS across ``--workers``
    processes, and writes the sorted CSV plus a summary JSON. It touches no COG.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m dataset.select",
        description="Draw the Phase-D selection list from the committed artifacts "
        f"(budget-parameterized; debug tier {DEBUG_BUDGET:,}, issue #44).",
    )
    p.add_argument("--catalog-dir", type=Path, default=Path("data/catalog_gated"))
    p.add_argument("--budget", type=int, default=DEBUG_BUDGET)
    p.add_argument("--out", type=Path, default=SELECTION_PATH)
    p.add_argument("--summary", type=Path, default=None, help="optional summary JSON path")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--workers", type=int, default=None,
                   help="processes to fan the per-stratum FSCS across (default: one per "
                   "core; the drawn selection is identical for any worker count)")
    args = p.parse_args(argv)

    selection = run(args.catalog_dir, budget=args.budget, seed=args.seed, workers=args.workers)
    out = write_selection(selection, args.out)
    prov = selection.provenance_counts()
    per_cog = selection.per_cog_distribution()

    print(f"Selected {selection.n:,} windows at budget {args.budget:,} (k={selection.allocation.k}).")
    print(f"  provenance: base {prov.get(PROV_BASE, 0):,}, "
          f"coastline {prov.get(PROV_COASTLINE, 0):,} (floor {selection.allocation.coast_floor})")
    print(f"  windows/COG: mean {per_cog['mean_per_cog']}, median {per_cog['median_per_cog']}, "
          f"max {per_cog['max_per_cog']} over {per_cog['n_cogs']:,} COGs")
    print(f"  floor clamp {selection.allocation.floor_clamp} fires on "
          f"{selection.allocation.floor_clamp_strata_below}/{selection.allocation.k} strata (inert)")
    print(f"  selection: {out}")
    if args.summary is not None:
        print(f"  summary:   {write_summary(selection, args.summary)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
