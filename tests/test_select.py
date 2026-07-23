"""The sampler: largest-remainder allocation, FSCS selection, and the emitted list (#44).

Unit tests run against synthetic frames — a deterministic sampler is unusually well
suited to them. One integration test reproduces the committed 50k allocation table
against the real catalog when it is present (skipped in CI, where ``data/`` is absent).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset import centroids as centroids_mod
from dataset import frame as frame_mod
from dataset import select

_CATALOG = Path("data/catalog_gated")
_ALLOC_TABLE = Path("docs/research/d2-allocation-nh.json")


# --- largest_remainder --------------------------------------------------------


def test_largest_remainder_sums_exactly_and_is_proportional():
    out = select.largest_remainder(np.array([1.0, 1.0, 2.0]), 100)
    assert out.sum() == 100
    assert list(out) == [25, 25, 50]


def test_largest_remainder_leftover_goes_to_largest_fraction_then_lowest_index():
    # ideal shares 3.333 each -> floor 3,3,3, one leftover to the largest remainder;
    # all remainders equal (.333), so the tie breaks to the lowest index.
    out = select.largest_remainder(np.array([1.0, 1.0, 1.0]), 10)
    assert out.sum() == 10
    assert list(out) == [4, 3, 3]


def test_largest_remainder_zero_weights_gives_zeros():
    out = select.largest_remainder(np.array([0.0, 0.0]), 5)
    assert list(out) == [0, 0]


def test_largest_remainder_cap_clamps_and_respills():
    # weights want [50, 50] but the first is capped at 10 -> 40 spills to the second.
    out = select.largest_remainder(np.array([1.0, 1.0]), 100, cap=np.array([10, 200]))
    assert list(out) == [10, 90]
    assert out.sum() == 100


def test_largest_remainder_is_deterministic():
    w = np.array([3.1, 2.9, 1.7, 8.3, 0.4])
    a = select.largest_remainder(w, 137)
    b = select.largest_remainder(w, 137)
    assert np.array_equal(a, b)


# --- allocate -----------------------------------------------------------------


def test_allocate_hits_budget_and_coast_floor_exactly():
    rng = np.random.default_rng(0)
    N_h = rng.integers(30_000, 300_000, size=20)
    sigma_h = rng.uniform(0.8, 1.4, size=20)
    coast_h = rng.integers(0, 6000, size=20)
    alloc = select.allocate(N_h, sigma_h, coast_h, 50_000)

    assert int(alloc.n_h.sum()) == 50_000
    assert int(alloc.coastal.sum()) == alloc.coast_floor == round(0.0134 * 50_000)
    assert np.array_equal(alloc.base, alloc.n_h - alloc.coastal)
    assert (alloc.coastal <= np.minimum(alloc.n_h, coast_h)).all()  # capped both ways


def test_allocate_budget_parameterized_shares_stay_stable():
    rng = np.random.default_rng(1)
    N_h = rng.integers(30_000, 300_000, size=20)
    sigma_h = rng.uniform(0.8, 1.4, size=20)
    coast_h = rng.integers(0, 6000, size=20)

    a50 = select.allocate(N_h, sigma_h, coast_h, 50_000)
    a500 = select.allocate(N_h, sigma_h, coast_h, 500_000)
    assert int(a500.n_h.sum()) == 500_000
    # Neyman shares are budget-invariant (#45): the 10x budget scales each n_h ~10x.
    ratio = a500.n_h / a50.n_h
    assert np.allclose(ratio, 10.0, atol=0.05)


def test_allocate_rejects_overflow_stratum():
    # A budget larger than the strata can hold (a Phase-F re-fit failure mode): two
    # strata of 100 rows asked to yield 125 each.
    with pytest.raises(ValueError, match="more windows than they hold"):
        select.allocate(
            np.array([100, 100]),
            np.array([1.0, 1.0]),
            np.array([0, 0]),
            250,
        )


def test_allocate_floor_clamp_is_reported_not_applied():
    # Two equal strata, budget 1000 -> n_h ~500 each, clamp max(500, 10) = 500: inert.
    alloc = select.allocate(np.array([100_000, 100_000]), np.array([1.0, 1.0]),
                            np.array([0, 0]), 1000)
    assert alloc.floor_clamp == 500
    assert alloc.floor_clamp_strata_below == 0
    assert int(alloc.n_h.sum()) == 1000  # nothing was raised to the floor


# --- fscs_select --------------------------------------------------------------


def _blob(n: int, center: np.ndarray, spread: float, rng) -> np.ndarray:
    return center + rng.normal(0, spread, size=(n, center.size))


def test_fscs_select_fills_quota_exactly_with_distinct_rows():
    rng = np.random.default_rng(0)
    S = _blob(500, np.zeros(6), 1.0, rng).astype(np.float32)
    geocell = np.array([f"g{i % 8}" for i in range(500)], dtype=object)
    wid = np.array([f"w{i}" for i in range(500)], dtype=object)
    picks = select.fscs_select(S, 40, geocell, wid, seed=0)
    assert picks.size == 40
    assert np.unique(picks).size == 40  # distinct
    assert (picks == np.sort(picks)).all()  # returned sorted


def test_fscs_select_is_deterministic():
    rng = np.random.default_rng(0)
    S = _blob(400, np.zeros(6), 1.0, rng).astype(np.float32)
    geocell = np.array([f"g{i % 5}" for i in range(400)], dtype=object)
    wid = np.array([f"w{i}" for i in range(400)], dtype=object)
    a = select.fscs_select(S, 30, geocell, wid, seed=0)
    b = select.fscs_select(S, 30, geocell, wid, seed=0)
    assert np.array_equal(a, b)


def test_fscs_select_spreads_across_geocells():
    # 6 geocells, quota 6: the round-robin should touch several cells, not pile into one.
    rng = np.random.default_rng(2)
    S = _blob(600, np.zeros(6), 1.0, rng).astype(np.float32)
    geocell = np.array([f"g{i % 6}" for i in range(600)], dtype=object)
    wid = np.array([f"w{i}" for i in range(600)], dtype=object)
    picks = select.fscs_select(S, 6, geocell, wid, seed=0)
    assert np.unique([geocell[i] for i in picks]).size >= 3


def test_fscs_select_quota_at_or_above_size_takes_all():
    rng = np.random.default_rng(0)
    S = _blob(10, np.zeros(6), 1.0, rng).astype(np.float32)
    geocell = np.array([f"g{i}" for i in range(10)], dtype=object)
    wid = np.array([f"w{i}" for i in range(10)], dtype=object)
    assert np.array_equal(select.fscs_select(S, 10, geocell, wid), np.arange(10))
    assert np.array_equal(select.fscs_select(S, 99, geocell, wid), np.arange(10))


def test_fscs_select_zero_quota_is_empty():
    S = np.zeros((5, 6), dtype=np.float32)
    g = np.array(["g0"] * 5, dtype=object)
    w = np.array([f"w{i}" for i in range(5)], dtype=object)
    assert select.fscs_select(S, 0, g, w).size == 0


# --- select_windows (full plumbing on a synthetic frame) ----------------------


def _synthetic_inputs(n_per: int = 400, k: int = 4, seed: int = 0):
    """A well-separated synthetic frame + identity scaler + its own cluster centroids."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((k, 6), dtype=np.float64)
    centers[:, 0] = np.arange(k) * 10.0  # separate the clusters along dim 0
    X = np.vstack([_blob(n_per, centers[h], 0.6, rng) for h in range(k)])
    n = X.shape[0]
    window_id = np.array(
        [f"Copernicus_DSM_COG_10_N{h:02d}_00_E000_00_DEM:{i}:0"
         for h in range(k) for i in range(n_per)],
        dtype=object,
    )
    coastline = np.zeros(n, dtype=bool)
    coastline[::10] = True  # 10% coastal, spread across strata
    geocell = np.array([f"g{i % 12}" for i in range(n)], dtype=object)

    scaler = frame_mod.Scaler(
        mean=np.zeros(6), std=np.ones(6), fingerprint={"synthetic": True}
    )
    frame = frame_mod.Frame(
        X=X,
        window_id=window_id,
        water_affected=np.zeros(n, dtype=bool),
        coastline=coastline,
        fingerprint={"synthetic": True},
    )
    centroids = centroids_mod.Centroids(
        centers=centers, fingerprint={"synthetic": True}, provenance={"synthetic": True}
    )
    return frame, centroids, scaler, geocell


def test_select_windows_emits_budget_windows_with_valid_provenance():
    frame, centroids, scaler, geocell = _synthetic_inputs()
    sel = select.select_windows(frame, centroids, scaler, geocell, budget=200, seed=0)

    assert sel.n == 200
    assert np.unique(sel.window_id).size == 200  # distinct windows
    assert set(sel.provenance) <= {select.PROV_BASE, select.PROV_COASTLINE}
    # coastal picks equal the allocation's reserved total.
    n_coastal = int((sel.provenance == select.PROV_COASTLINE).sum())
    assert n_coastal == int(sel.allocation.coastal.sum()) == sel.allocation.coast_floor


def test_select_windows_is_byte_stable_and_sorted():
    frame, centroids, scaler, geocell = _synthetic_inputs()
    a = select.select_windows(frame, centroids, scaler, geocell, budget=200, seed=0)
    b = select.select_windows(frame, centroids, scaler, geocell, budget=200, seed=0)
    assert np.array_equal(a.window_id, b.window_id)
    assert np.array_equal(a.provenance, b.provenance)
    # sorted by (stratum_id, window_id)
    keys = list(zip(a.stratum_id.tolist(), a.window_id.tolist()))
    assert keys == sorted(keys)


def test_select_windows_parallel_matches_sequential():
    # Fanning the per-stratum FSCS across processes must give a byte-identical selection.
    frame, centroids, scaler, geocell = _synthetic_inputs()
    serial = select.select_windows(frame, centroids, scaler, geocell, budget=200, seed=0, workers=1)
    parallel = select.select_windows(frame, centroids, scaler, geocell, budget=200, seed=0, workers=2)
    assert np.array_equal(serial.window_id, parallel.window_id)
    assert np.array_equal(serial.stratum_id, parallel.stratum_id)
    assert np.array_equal(serial.provenance, parallel.provenance)


def test_select_windows_provenance_matches_assigned_stratum():
    frame, centroids, scaler, geocell = _synthetic_inputs()
    sel = select.select_windows(frame, centroids, scaler, geocell, budget=160, seed=0)
    # Every coastal-tagged pick must actually be a coastal window in the frame.
    coastal_ids = set(frame.window_id[frame.coastline].tolist())
    for wid, prov in zip(sel.window_id, sel.provenance):
        if prov == select.PROV_COASTLINE:
            assert wid in coastal_ids


def test_select_windows_rejects_misaligned_geocell():
    frame, centroids, scaler, geocell = _synthetic_inputs()
    with pytest.raises(ValueError, match="row-aligned"):
        select.select_windows(frame, centroids, scaler, geocell[:-1], budget=100)


# --- Selection reporting ------------------------------------------------------


def test_per_cog_distribution_counts_windows_per_cog():
    frame, centroids, scaler, geocell = _synthetic_inputs()
    sel = select.select_windows(frame, centroids, scaler, geocell, budget=120, seed=0)
    dist = sel.per_cog_distribution()
    assert dist["n_windows"] == sel.n
    assert dist["max_per_cog"] >= dist["median_per_cog"] >= 1
    assert sum(dist["histogram"].values()) == dist["n_cogs"]


# --- Integration: reproduce the committed 50k allocation on the real catalog ---


@pytest.mark.skipif(not _CATALOG.exists(), reason="gated catalog not present")
def test_allocation_reproduces_committed_table_on_real_catalog():
    """Full-precision σ_h from the real frame must reproduce d2-allocation-nh.json exactly."""
    fr = frame_mod.kept_frame(_CATALOG)
    scaler = frame_mod.read_scaler(fingerprint=fr.fingerprint)
    cen = centroids_mod.read_centroids(fingerprint=fr.fingerprint)
    Z = np.ascontiguousarray(fr.standardized(scaler).astype(np.float32))
    labels = centroids_mod.assign(Z, cen)

    sigma = select.stratum_sigma(Z, labels, cen.centers)
    N_h = np.bincount(labels, minlength=cen.k).astype(np.int64)
    coast_h = np.array([int(fr.coastline[labels == h].sum()) for h in range(cen.k)])
    alloc = select.allocate(N_h, sigma, coast_h, 50_000)

    committed = json.loads(_ALLOC_TABLE.read_text())["rows"]
    assert [int(v) for v in alloc.n_h] == [r["n_h"] for r in committed]
    assert [int(v) for v in alloc.coastal] == [r["coastal"] for r in committed]
    assert [int(v) for v in alloc.base] == [r["base"] for r in committed]
    assert [int(v) for v in alloc.N_h] == [r["N_h"] for r in committed]
