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


def test_allocate_headroom_is_the_kept_pool_over_the_quota():
    alloc = select.allocate(
        np.array([100_000, 100_000]), np.array([1.0, 1.0]), np.array([0, 0]), 1000
    )
    assert np.allclose(alloc.headroom, 100_000 / alloc.n_h)
    assert all(r["headroom"] == pytest.approx(200.0, abs=0.5) for r in alloc.rows())


def test_allocate_headroom_of_an_unallocated_stratum_is_infinite_not_its_own_size():
    # A stratum Neyman gives nothing has an infinitely deep pool. Reporting N_h there would
    # make the *emptiest* draw look like the thinnest one to anything taking a minimum.
    alloc = select.allocate(
        np.array([1_000_000, 500]), np.array([1.0, 0.0001]), np.array([0, 0]), 1000
    )
    assert int(alloc.n_h[1]) == 0
    assert np.isinf(alloc.headroom[1])
    assert alloc.rows()[1]["headroom"] is None
    assert alloc.binding()["min_headroom_stratum"] == 0  # not the empty stratum


def test_allocate_reports_the_coastal_cap_as_non_binding_when_coast_is_plentiful():
    alloc = select.allocate(
        np.array([100_000, 100_000]), np.array([1.0, 1.0]), np.array([5_000, 5_000]),
        50_000,
    )
    binding = alloc.binding()
    assert binding["coast_cap_strata_binding"] == 0
    assert binding["floor_clamp_strata_below"] == 0
    assert binding["coast_floor"] == alloc.coast_floor == round(0.0134 * 50_000)
    assert binding["realized_coast_share"] == pytest.approx(0.0134)
    assert binding["anything_binds"] is False


def test_allocate_reports_the_coastal_cap_when_a_stratum_cannot_hold_its_share():
    # Stratum 1 is coast-heavy (5,000 of the 6,000 coastal rows) but Neyman-poor
    # (n_h = 100), so min(n_h, coast_h) clamps its 558-window share to 100 and the
    # remainder spills to stratum 0 — the cap doing the job it exists for.
    alloc = select.allocate(
        np.array([1_000_000, 20_000]), np.array([1.0, 0.1]), np.array([1_000, 5_000]),
        50_000,
    )
    binding = alloc.binding()
    assert int(alloc.coastal[1]) == int(alloc.n_h[1]) == 100
    assert int(alloc.coastal.sum()) == alloc.coast_floor
    assert binding["coast_cap_strata_binding"] == 1
    assert binding["anything_binds"] is True


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


def test_per_cog_distribution_with_inventory_reports_fraction_drawn():
    """#69's F6c wants each COG's draw as a fraction of the inventory it actually holds."""
    frame, centroids, scaler, geocell = _synthetic_inputs()
    sel = select.select_windows(frame, centroids, scaler, geocell, budget=120, seed=0)
    stems, counts = np.unique(sel.cog_stems(), return_counts=True)
    # Give every drawn COG exactly twice the inventory it yielded -> every fraction is 0.5.
    inventory = {str(s): int(c) * 2 for s, c in zip(stems, counts)}

    dist = sel.per_cog_distribution(inventory=inventory)
    frac = dist["fraction_of_inventory"]
    assert frac["max"] == pytest.approx(0.5)
    assert frac["median"] == pytest.approx(0.5)
    assert frac["n_cogs_at_or_over_half"] == dist["n_cogs"]


def test_per_cog_distribution_separates_sizeable_cogs_from_tiny_ones():
    """A COG holding 3 windows hits fraction 1.0 without meaning anything (#69's F6c)."""
    frame, centroids, scaler, geocell = _synthetic_inputs()
    sel = select.select_windows(frame, centroids, scaler, geocell, budget=120, seed=0)
    stems, counts = np.unique(sel.cog_stems(), return_counts=True)
    # Every COG is drawn in full, but none of them holds the sizeable floor's worth.
    inventory = {str(s): int(c) for s, c in zip(stems, counts)}

    frac = sel.per_cog_distribution(inventory=inventory)["fraction_of_inventory"]
    assert frac["inventory_floor"] == select.FRACTION_INVENTORY_FLOOR
    assert frac["n_cogs_fully_drawn"] == len(inventory)
    assert frac["n_cogs_fully_drawn_sizeable"] == sum(
        c >= select.FRACTION_INVENTORY_FLOOR for c in inventory.values()
    )


def test_per_cog_distribution_rejects_a_cog_missing_from_the_inventory():
    """A drawn COG absent from the kept pool is a bug, not a zero-denominator to paper over."""
    frame, centroids, scaler, geocell = _synthetic_inputs()
    sel = select.select_windows(frame, centroids, scaler, geocell, budget=120, seed=0)
    with pytest.raises(ValueError, match="absent from the kept inventory"):
        sel.per_cog_distribution(inventory={})


# --- Overlap with another tier (#45: incidental, reported, never engineered) ---


def test_overlap_report_counts_shared_windows_both_ways():
    rep = select.overlap_report(
        np.array(["a", "b", "c", "d"], dtype=object),
        ["c", "d", "e"],
        n_pool=1000,
        label="debug 50k",
    )
    assert rep["label"] == "debug 50k"
    assert rep["n_overlap"] == 2
    assert rep["share_of_this"] == pytest.approx(0.5)
    assert rep["share_of_other"] == pytest.approx(2 / 3)
    # Under two independent uniform draws from a 1,000-row pool: 4*3/1000.
    assert rep["expected_if_uniform"] == pytest.approx(0.012)


def test_overlap_report_disjoint_lists_report_zero():
    rep = select.overlap_report(
        np.array(["a", "b"], dtype=object), ["x"], n_pool=10, label="other"
    )
    assert rep["n_overlap"] == 0
    assert rep["share_of_this"] == 0.0
    assert rep["lift_over_uniform"] == 0.0


# --- Re-loading the persisted list (the contract, not the seed) ---------------


def test_selection_from_csv_round_trips_the_written_list(tmp_path):
    frame, centroids, scaler, geocell = _synthetic_inputs()
    sel = select.select_windows(frame, centroids, scaler, geocell, budget=120, seed=0)
    path = select.write_selection(sel, tmp_path / "sel.csv")

    reloaded = select.Selection.from_csv(
        path, allocation=sel.allocation, fingerprint=sel.fingerprint
    )
    assert np.array_equal(reloaded.window_id, sel.window_id)
    assert np.array_equal(reloaded.stratum_id, sel.stratum_id)
    assert np.array_equal(reloaded.provenance, sel.provenance)
    assert reloaded.budget == sel.budget
    assert reloaded.summary()["per_cog"] == sel.summary()["per_cog"]


def test_read_window_ids_returns_only_the_id_column(tmp_path):
    frame, centroids, scaler, geocell = _synthetic_inputs()
    sel = select.select_windows(frame, centroids, scaler, geocell, budget=120, seed=0)
    path = select.write_selection(sel, tmp_path / "sel.csv")
    assert np.array_equal(select.read_window_ids(path), sel.window_id)


def test_selection_from_csv_rejects_a_foreign_header(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("window_id,stratum\nw:0:0,1\n")
    frame, centroids, scaler, geocell = _synthetic_inputs()
    alloc = select.allocate(
        np.array([100, 100]), np.array([1.0, 1.0]), np.array([0, 0]), 10
    )
    with pytest.raises(ValueError, match="header"):
        select.Selection.from_csv(bad, allocation=alloc, fingerprint={})


def test_allocation_at_matches_the_allocation_a_draw_derives():
    """The report driver re-derives the quota table without paying for FSCS again."""
    frame, centroids, scaler, geocell = _synthetic_inputs()
    sel = select.select_windows(frame, centroids, scaler, geocell, budget=200, seed=0)
    alloc = select.allocation_at(frame, centroids, scaler, budget=200)
    assert np.array_equal(alloc.n_h, sel.allocation.n_h)
    assert np.array_equal(alloc.coastal, sel.allocation.coastal)
    assert np.allclose(alloc.sigma_h, sel.allocation.sigma_h)


# --- The committed allocation table -------------------------------------------


def test_write_allocation_table_matches_the_committed_d2_shape(tmp_path):
    frame, centroids, scaler, _ = _synthetic_inputs()
    alloc = select.allocation_at(frame, centroids, scaler, budget=200)
    path = select.write_allocation_table(
        alloc, tmp_path / "alloc.json", fingerprint=frame.fingerprint
    )

    got = json.loads(path.read_text())
    committed = json.loads(_ALLOC_TABLE.read_text())
    assert set(got) == set(committed)
    assert set(got["rows"][0]) >= set(committed["rows"][0])
    assert got["budget"] == 200
    assert got["k"] == alloc.k
    assert sum(r["n_h"] for r in got["rows"]) == 200


# --- Kept per-COG inventory ---------------------------------------------------


def _tiny_catalog(path: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from dataset import catalog as catalog_mod

    stems = ["cogA", "cogA", "cogA", "cogB", "cogB"]
    keep = [True, True, False, True, False]
    parts = path / catalog_mod.PARTS_DIR
    parts.mkdir(parents=True, exist_ok=True)
    table = pa.table({"stem": stems, "keep": keep, "geocell": ["g0"] * 5})
    pq.write_table(table.slice(0, 3), parts / "part-00000.parquet")
    pq.write_table(table.slice(3), parts / "part-00001.parquet")
    return path


def test_kept_cog_inventory_counts_only_kept_rows(tmp_path):
    inv = select.kept_cog_inventory(_tiny_catalog(tmp_path))
    assert inv == {"cogA": 2, "cogB": 1}


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
