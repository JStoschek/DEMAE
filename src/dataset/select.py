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

The 50,000-window debug tier is one call; Phase F re-runs at 500k against the *same*
committed centroids and scaler (plan §F). So :func:`select_windows` takes ``budget`` as a
parameter and re-derives the allocation from the frame at that budget — it never
re-derives the strata. The committed [`d2-allocation-nh.json`](../../docs/research/d2-allocation-nh.json)
is the frozen 50k reference this code reproduces, not a table it reads.

**#71 spent that design and it held**: the 500k scale tier is this module at
``--budget 500000``, drawn from the same two artifacts, with no new sampler and no
hierarchical rule (#45 ruled the tiers independent draws). Plan §F's *re-clustering* knob
was recorded **structurally inert** rather than merely declined: the centroids were fit on
the entire gated catalog (``fit_rows`` 3,052,735), not on the debug tier, so a Phase-F
refit would run on identical input and could only lose the cross-tier comparability the
artifacts exist to provide — #41 having established that the *fit* is not bitwise
reproducible across threads while ``assign`` is. See
[`f-scale-draw.md`](../../docs/research/f-scale-draw.md).

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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np

from . import catalog, centroids as centroids_mod, config, frame as frame_mod
from .geometry import window_id_stem
from .sampling import _shuffle_key

# --- The two-value provenance vocabulary (#42 §4) -----------------------------

#: A window bought by its stratum's Neyman base draw (FSCS over the non-coastal rows).
PROV_BASE: Final = "base"
#: A window bought by the within-stratum coastal reservation (FSCS over the coastal rows).
PROV_COASTLINE: Final = "floor:coastline"

#: The emitted list's columns — what Phase E1's manifest addendum consumes, and what
#: :meth:`Selection.from_csv` refuses to read a file without.
SELECTION_COLUMNS: Final = ("window_id", "stratum_id", "provenance")

# --- Locked allocation constants (#42) ----------------------------------------

#: The debug-tier budget. Not hard-wired anywhere below — it is the default of a
#: parameter, because Phase F re-runs the same code at 500k (plan §F, #71).
DEBUG_BUDGET: Final = 50_000

#: The Phase-F scale-tier budget (#71) — the plan's upper bound, ten times the debug
#: tier, drawn by *this* code from *these* artifacts. It is a second default, not a
#: second sampler: #44 built budget as an argument precisely so this line is all the
#: 10× costs.
SCALE_BUDGET: Final = 500_000

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
SELECTION_PATH: Final = Path(config.SELECTION_PATH)

#: Where the shipped scale-tier selection list lives (#71). Same contract as the debug
#: tier's: committed, re-loaded rather than regenerated, and the thing two runs are
#: compared byte-for-byte against.
SCALE_SELECTION_PATH: Final = Path(config.SCALE_SELECTION_PATH)

#: How the quota table was derived — copied verbatim into every emitted allocation
#: table so the JSON states its own rule (the string ``d2-allocation-nh.json`` shipped).
ALLOCATION_METHOD: Final = (
    "neyman n_h proportional to N_h*sigma_h; largest-remainder to exact budget; "
    "coastal floor reserved within-stratum proportional to coast_h, capped at "
    "min(n_h, coast_h)"
)

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
    coast_cap_strata_binding: int  # strata whose coastal share hit min(n_h, coast_h)
    coast_floor_rate: float  # the rate the reservation was taken at

    @property
    def k(self) -> int:
        return int(self.n_h.shape[0])

    @property
    def headroom(self) -> np.ndarray:
        """``N_h / n_h`` — how many kept windows each stratum holds per window it must yield.

        The number that actually moves with the budget. Neyman shares are budget-invariant,
        so a 10× budget is a 10× *shallower* pool: over the four flat strata ADR-0007 leans
        on (2/5/7/17) the debug tier's ≥53× becomes ≥5.3× at 500k, and FSCS reaches
        proportionally further from each sub-centroid. ADR-0007's Phase-F revisit trigger is
        written against exactly this ratio on strata 2/17.

        A stratum allocated nothing is ``inf``, not ``N_h``: an unallocated pool is
        infinitely deep, and rounding it down to its own size would make it look like the
        *thinnest* stratum to any consumer taking a minimum.
        """
        n_h = self.n_h.astype(np.float64)
        return np.divide(
            self.N_h.astype(np.float64), n_h, out=np.full(self.k, np.inf), where=n_h > 0
        )

    def rows(self) -> list[dict[str, int | float | None]]:
        """One JSON-ready dict per stratum — ``d2-allocation-nh.json``'s columns plus headroom.

        The extra column is additive on purpose: a reader written for the shipped 50k table
        still finds every key it knows, and the two tiers' tables diff row-for-row.
        ``headroom`` is ``null`` for an unallocated stratum, JSON having no infinity.
        """
        headroom = self.headroom
        return [
            {
                "stratum_id": h,
                "N_h": int(self.N_h[h]),
                "sigma_h": round(float(self.sigma_h[h]), 4),
                "coast_h": int(self.coast_h[h]),
                "n_h": int(self.n_h[h]),
                "coastal": int(self.coastal[h]),
                "base": int(self.base[h]),
                "headroom": (
                    round(float(headroom[h]), 2) if np.isfinite(headroom[h]) else None
                ),
            }
            for h in range(self.k)
        ]

    def binding(self) -> dict[str, Any]:
        """Which of the allocation's two guards actually *bind* at this budget.

        Both were measured inert at 50k, and #71 asks the scale tier to measure rather than
        assume it — Neyman's shares are budget-invariant, so they *probably* stay inert, but
        probably is not measured. Two distinct guards are reported:

        - the ``max(500, 1%·budget)`` per-stratum clamp, which is checked and never applied
          (:func:`allocate`), so "binds" means *it would have raised a stratum*;
        - the coastal reservation's ``min(n_h, coast_h)`` cap, which **is** applied, so
          "binds" means a coast-heavy but Neyman-poor stratum could not hold its share and
          the remainder spilled elsewhere.

        ``realized_coast_share`` against ``frame_coast_rate`` is the third thing to read,
        with one precision: the reservation is a within-stratum *split* (#42 §4), so a
        ``base`` pick is never coastal and the reservation **sets** the corpus's coastal
        share rather than raising a floor under whatever the draw would otherwise produce.
        A realized share at the frame's own coastal rate therefore says the split is doing
        what it was specified to do — it is not evidence about the unreserved draw, which
        only a second full draw could supply.
        """
        pool = int(self.N_h.sum())
        headroom = self.headroom
        thinnest = int(np.argmin(headroom))
        return {
            "budget": self.budget,
            "floor_clamp": self.floor_clamp,
            "floor_clamp_strata_below": self.floor_clamp_strata_below,
            "min_n_h": int(self.n_h.min()),
            "coast_floor": self.coast_floor,
            "coast_floor_rate": self.coast_floor_rate,
            "frame_coast_rate": round(int(self.coast_h.sum()) / pool, 6) if pool else 0.0,
            "realized_coast_share": round(int(self.coastal.sum()) / self.budget, 6),
            "coast_cap_strata_binding": self.coast_cap_strata_binding,
            "min_headroom": (
                round(float(headroom[thinnest]), 2)
                if np.isfinite(headroom[thinnest])
                else None
            ),
            "min_headroom_stratum": thinnest,
            "anything_binds": bool(
                self.floor_clamp_strata_below or self.coast_cap_strata_binding
            ),
        }


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
    coast_cap = np.minimum(n_h, coast_h)
    # The uncapped apportionment is computed only to *report* whether the cap bound —
    # comparing the capped result against the cap itself cannot tell a stratum that was
    # clamped from one that happens to sit exactly at its ceiling (every coast-free
    # stratum does, at zero).
    c_h_uncapped = largest_remainder(coast_h.astype(np.float64), coast_floor)
    c_h = largest_remainder(coast_h.astype(np.float64), coast_floor, cap=coast_cap)
    base_h = n_h - c_h

    return Allocation(
        coast_cap_strata_binding=int((c_h_uncapped > coast_cap).sum()),
        coast_floor_rate=coast_floor_rate,
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


#: Inventory below which a COG's *fraction* drawn stops being informative. A COG holding
#: three kept windows is "fully drawn" the moment the sampler takes three, which says
#: nothing about depth; #69's F6c baseline reports its headline maximum over COGs holding
#: at least this many, and the fully-drawn count is split at the same line.
FRACTION_INVENTORY_FLOOR: Final = 32


def _held_by(stems: np.ndarray, inventory: Mapping[str, int]) -> np.ndarray:
    """The kept inventory of each drawn COG, in ``stems`` order — refusing an unknown COG."""
    missing = [str(s) for s in stems if str(s) not in inventory]
    if missing:
        raise ValueError(
            f"{len(missing)} drawn COG(s) absent from the kept inventory, e.g. "
            f"{missing[:3]} — the draw produced a source the kept pool does not hold"
        )
    return np.array([int(inventory[str(s)]) for s in stems], dtype=np.float64)


def _inventory_fraction_summary(
    frac: np.ndarray,
    counts: np.ndarray,
    held: np.ndarray,
    *,
    n_in_pool: int,
    n_drawn: int,
) -> dict[str, Any]:
    """Summarize how deeply the draw reached into each COG — #69's ``F6c`` shape."""
    sizeable = held >= FRACTION_INVENTORY_FLOOR
    full = frac >= 1.0
    return {
        "mean": round(float(frac.mean()), 4),
        "median": round(float(np.median(frac)), 4),
        "p99": round(float(np.percentile(frac, 99)), 4),
        "max": round(float(frac.max()), 4),
        "inventory_floor": FRACTION_INVENTORY_FLOOR,
        "max_among_sizeable_cogs": (
            round(float(frac[sizeable].max()), 4) if sizeable.any() else None
        ),
        "n_cogs_at_or_over_half": int((frac >= 0.5).sum()),
        "n_cogs_fully_drawn": int(full.sum()),
        "n_cogs_fully_drawn_sizeable": int((full & sizeable).sum()),
        "max_windows_from_a_fully_drawn_cog": (
            int(counts[full].max()) if full.any() else 0
        ),
        "n_cogs_in_pool": n_in_pool,
        "n_cogs_undrawn": n_in_pool - n_drawn,
    }


def read_selection_rows(path: Path | str) -> list[list[str]]:
    """The rows of a persisted selection list, header validated.

    Deliberately not `csv.reader`/pandas: the file this reads is the determinism contract
    and its three columns never contain a quote, a comma or an escape, so a split is both
    sufficient and impossible to mis-configure. The header check is the guard that matters
    — it refuses a file of the wrong *kind* rather than silently mis-parsing one.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    header = lines[0].split(",") if lines else []
    if header != list(SELECTION_COLUMNS):
        raise ValueError(
            f"{path} has header {header!r}; expected {list(SELECTION_COLUMNS)!r}"
        )
    return [ln.split(",") for ln in lines[1:] if ln]


def read_window_ids(path: Path | str) -> np.ndarray:
    """Just the ``window_id`` column of a persisted selection list.

    What a cross-tier comparison needs: the other tier's ids carry no useful stratum or
    provenance here, and pairing them with *this* tier's allocation to build a full
    :class:`Selection` would state a relationship (#45 ruled out) that does not exist.
    """
    return np.array([r[0] for r in read_selection_rows(path)], dtype=object)


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

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        *,
        allocation: Allocation,
        fingerprint: dict[str, Any],
    ) -> "Selection":
        """Re-load a persisted selection list — the artifact *is* the contract (#43/#44).

        Every report about a tier is then a function of the committed CSV rather than of
        an in-memory run, so it can be regenerated at any time without paying for FSCS
        again. The ``allocation`` is handed in because the quota table is derived from the
        frame, not stored in the list; :func:`allocation_at` re-derives it in seconds.
        """
        rows = read_selection_rows(path)
        return cls(
            window_id=np.array([r[0] for r in rows], dtype=object),
            stratum_id=np.array([int(r[1]) for r in rows], dtype=np.int64),
            provenance=np.array([r[2] for r in rows], dtype=object),
            allocation=allocation,
            budget=allocation.budget,
            fingerprint=dict(fingerprint),
        )

    @property
    def n(self) -> int:
        return int(self.window_id.shape[0])

    def cog_stems(self) -> np.ndarray:
        """The source COG stem of each selected window (``window_id`` is ``stem:row:col``)."""
        return np.array([window_id_stem(w) for w in self.window_id], dtype=object)

    def per_cog_distribution(
        self, inventory: Mapping[str, int] | None = None
    ) -> dict[str, Any]:
        """Realized windows-per-COG distribution — the metric #42 requires at every tier.

        #42 retired ``PER_TILE_WINDOW_CAP`` as inert at the debug tier (~2/COG) but binds
        the sampler to *measure* it, so Phase F can decide on evidence — not the untested
        64 — whether the denser scale tier needs a cap. Reported as the summary a cap
        threshold would be read off: counts, the tail, and the densest COGs.

        Pass ``inventory`` (:func:`kept_cog_inventory`) to add the half of the picture a
        raw count cannot give. #69 retired the cap of 64 as *unfixable* rather than merely
        stale, because the ceiling on a COG is its own kept inventory — at most 256 windows
        — so 44-from-a-COG means nothing until you know whether that COG held 50 or 250.
        The fraction-of-inventory distribution is the form #69's ``F6c`` registered, and a
        drawn COG missing from ``inventory`` raises: it would mean the draw produced a
        source the kept pool does not contain.
        """
        stems, counts = np.unique(self.cog_stems(), return_counts=True)
        held = _held_by(stems, inventory) if inventory is not None else None
        order = np.argsort(counts)[::-1]

        def cog_entry(i: int) -> dict[str, Any]:
            entry: dict[str, Any] = {"stem": str(stems[i]), "windows": int(counts[i])}
            if held is not None:
                entry["inventory"] = int(held[i])
                entry["fraction"] = round(float(counts[i] / held[i]), 4)
            return entry

        out: dict[str, Any] = {
            "n_windows": self.n,
            "n_cogs": int(stems.size),
            "mean_per_cog": round(float(counts.mean()), 4),
            "median_per_cog": int(np.median(counts)),
            "max_per_cog": int(counts.max()),
            "p99_per_cog": int(np.percentile(counts, 99)),
            "histogram": {int(v): int(c) for v, c in zip(*np.unique(counts, return_counts=True))},
            "densest_cogs": [cog_entry(i) for i in order[:10]],
        }
        if held is not None:
            frac = counts / held
            out["fraction_of_inventory"] = _inventory_fraction_summary(
                frac, counts, held, n_in_pool=len(inventory or {}), n_drawn=int(stems.size)
            )
            out["fraction_of_inventory"]["highest_fraction_cogs"] = [
                cog_entry(i) for i in np.argsort(frac)[::-1][:10]
            ]
        return out

    def provenance_counts(self) -> dict[str, int]:
        vals, counts = np.unique(self.provenance, return_counts=True)
        return {str(v): int(c) for v, c in zip(vals, counts)}

    def summary(self, inventory: Mapping[str, int] | None = None) -> dict[str, Any]:
        """Everything the d4/f-scale write-ups and Phase E/F need that is not the list itself."""
        return {
            "budget": self.budget,
            "n_selected": self.n,
            "k": self.allocation.k,
            "coast_floor": self.allocation.coast_floor,
            "floor_clamp": self.allocation.floor_clamp,
            "floor_clamp_strata_below": self.allocation.floor_clamp_strata_below,
            "provenance_counts": self.provenance_counts(),
            "per_cog": self.per_cog_distribution(inventory),
            "binding": self.allocation.binding(),
            "allocation": self.allocation.rows(),
            "frame_fingerprint": dict(self.fingerprint),
            "determinism": (
                "the persisted selection list is the contract; the sub-k-means fit is "
                "pinned single-threaded but not relied on for reproducibility (#43)"
            ),
        }


def _allocate_labelled(
    Z: np.ndarray,
    labels: np.ndarray,
    coastline: np.ndarray,
    centroids: centroids_mod.Centroids,
    budget: int,
) -> Allocation:
    """The quota table for an already-standardized, already-labelled frame."""
    k = centroids.k
    sigma = stratum_sigma(Z, labels, centroids.centers)
    N_h = np.bincount(labels, minlength=k).astype(np.int64)
    coast_h = np.array([int(coastline[labels == h].sum()) for h in range(k)], dtype=np.int64)
    return allocate(N_h, sigma, coast_h, budget)


def allocation_at(
    frame: frame_mod.Frame,
    centroids: centroids_mod.Centroids,
    scaler: frame_mod.Scaler,
    *,
    budget: int,
) -> Allocation:
    """The per-stratum quota table at ``budget`` — everything :func:`select_windows` does
    except the draw itself.

    Standardize, assign, allocate: seconds, and no sub-k-means anywhere. This is the seam
    the tier reports are built on. A report that re-ran the draw to recover its own quota
    table would cost an hour at the scale tier and — worse — would be describing a *fresh*
    draw rather than the committed one; pairing this with :meth:`Selection.from_csv` keeps
    every number a function of the artifact on disk.
    """
    Z = np.ascontiguousarray(frame.standardized(scaler).astype(np.float32))
    labels = centroids_mod.assign(Z, centroids)
    return _allocate_labelled(Z, labels, frame.coastline, centroids, budget)


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
    alloc = _allocate_labelled(Z, labels, frame.coastline, centroids, budget)

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


def _kept_column(catalog_dir: Path | str, column: str) -> np.ndarray:
    """One catalog column, narrowed to the kept rows and row-aligned with the kept frame.

    The ``keep`` mask is applied exactly as :func:`frame.kept_frame` applies it, which is
    what makes every column read through here line up row-for-row with the frame.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(
        Path(catalog_dir) / catalog.PARTS_DIR, columns=["keep", column]
    )
    keep = table.column("keep").to_numpy(zero_copy_only=False).astype(bool)
    return table.column(column).to_numpy(zero_copy_only=False)[np.flatnonzero(keep)]


def kept_geocell(catalog_dir: Path | str = Path("data/catalog_gated")) -> np.ndarray:
    """The kept frame's ``geocell`` column, row-aligned with :func:`frame.kept_frame`.

    Read straight off the gated catalog rather than re-derived from ``window_id``: the
    catalog already owns the one definition of a window's geo-cell (``dataset.geometry``
    via ``dataset.catalog``), and re-parsing it here would be a second one that could drift.
    """
    return _kept_column(catalog_dir, "geocell")


def kept_cog_inventory(
    catalog_dir: Path | str = Path("data/catalog_gated"),
) -> dict[str, int]:
    """How many kept windows each source COG holds — the denominator a per-COG count needs.

    Budget-invariant: it is a property of the catalog and the Phase-C gates, so the same
    dict serves every tier. #69's ``F6b`` bounds each COG's realized count by it, and
    ``F6c`` reports the realized count as a fraction of it — which is the only way to read
    "44 windows from one COG" as either alarming or unremarkable.
    """
    vals, counts = np.unique(_kept_column(catalog_dir, "stem"), return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, counts)}


# --- Overlap between tiers (#45: incidental, reported, never engineered) -------


def overlap_report(
    window_id: Iterable[str],
    other: Iterable[str],
    *,
    n_pool: int,
    label: str,
) -> dict[str, Any]:
    """How much this selection shares with another tier's — a *finding*, not a target.

    #45 ruled the scale draw **independent**: the debug tier does not nest inside it and no
    hierarchical rule exists. Whatever the two lists share is therefore incidental, and
    reporting it carries no obligation to grow or shrink it. It is still worth knowing,
    because it is not a coin flip — FSCS is deterministic and coverage-seeking, so both
    tiers keep resolving to the same nearest-the-sub-centroid representatives.
    ``expected_if_uniform`` (``n_this·n_other/n_pool``, the overlap two *uniform* draws
    from the kept pool would average) makes that readable: a lift far above 1.0 is the
    sampler agreeing with itself, not a nesting rule leaking in.
    """
    this = {str(w) for w in window_id}
    that = {str(w) for w in other}
    shared = this & that
    expected = len(this) * len(that) / n_pool if n_pool > 0 else 0.0
    return {
        "label": label,
        "n_this": len(this),
        "n_other": len(that),
        "n_pool": int(n_pool),
        "n_overlap": len(shared),
        "share_of_this": round(len(shared) / len(this), 6) if this else 0.0,
        "share_of_other": round(len(shared) / len(that), 6) if that else 0.0,
        "expected_if_uniform": round(expected, 4),
        "lift_over_uniform": round(len(shared) / expected, 3) if expected > 0 else 0.0,
    }


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
    lines = [",".join(SELECTION_COLUMNS)]
    lines += [
        f"{w},{s},{p}"
        for w, s, p in zip(selection.window_id, selection.stratum_id, selection.provenance)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_summary(
    selection: Selection,
    path: Path | str,
    *,
    inventory: Mapping[str, int] | None = None,
) -> Path:
    """Write the JSON summary the d4 write-up reads (provenance + per-COG + allocation)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection.summary(inventory), indent=2) + "\n")
    return path


def write_allocation_table(
    allocation: Allocation, path: Path | str, *, fingerprint: Mapping[str, Any]
) -> Path:
    """Write the per-stratum quota table in the shape of ``d2-allocation-nh.json``.

    One shape per tier, so the 50k and 500k tables diff column-for-column rather than
    needing a reader that knows two formats. ``fingerprint`` rides along because the table
    is only meaningful against the strata it was allocated over — the same
    refusal-on-drift discipline #40/#41 built into the artifacts themselves.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "k": allocation.k,
        "budget": allocation.budget,
        "coast_floor": allocation.coast_floor,
        "floor_clamp": allocation.floor_clamp,
        "floor_clamp_strata_below": allocation.floor_clamp_strata_below,
        "method": ALLOCATION_METHOD,
        "centroids_fingerprint": dict(fingerprint),
        "rows": allocation.rows(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# --- Live driver --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python -m dataset.select`` — draw a tier and persist the list.

    Reads the committed scaler and centroids (never re-fits — that is what keeps Phase F
    comparable), assigns, allocates at ``--budget``, runs FSCS across ``--workers``
    processes, and writes the sorted CSV plus a summary JSON. It touches no COG.

    Both shipped tiers are this one command at two budgets — the debug tier (#44) at
    ``--budget 50000`` and the scale tier (#71) at ``--budget 500000 --out
    artifacts/phase_f/selection_scale_500k.csv``. That was the design's claim and #71 is
    where it is spent, so ``--out`` is deliberately *not* derived from ``--budget``: the
    two lists are separate committed contracts and neither may silently overwrite the other.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m dataset.select",
        description="Draw a selection list from the committed artifacts "
        f"(budget-parameterized; debug tier {DEBUG_BUDGET:,} #44, "
        f"scale tier {SCALE_BUDGET:,} #71).",
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
