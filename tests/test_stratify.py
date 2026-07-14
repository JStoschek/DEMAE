"""Land-ocean screen, TRI ruggedness, quantile strata, and quota selection."""

from __future__ import annotations

import numpy as np
import pytest

from dataset import config, stratify


# --- Ocean screen (ADR-0001 §3) -----------------------------------------------


def test_zero_fraction():
    a = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    assert stratify.zero_fraction(a) == pytest.approx(0.5)


def test_is_ocean_threshold():
    # exactly at the 0.75 threshold -> dropped
    a = np.zeros((4, 4), dtype=np.float32)
    a[0] = 5.0  # 4/16 land -> 0.75 zero-fraction
    assert stratify.zero_fraction(a) == pytest.approx(0.75)
    assert stratify.is_ocean(a) is True
    a[1] = 5.0  # now 0.5 zero-fraction -> kept (coastline)
    assert stratify.is_ocean(a) is False


# --- TRI (Riley, ADR-0001 §4) -------------------------------------------------


def test_tri_flat_is_zero():
    a = np.full((10, 10), 42.0, dtype=np.float32)
    assert np.allclose(stratify.tri_riley(a), 0.0)


def test_tri_rugged_beats_tilted_plane():
    # A smooth tilted plane has a constant gradient; rough noise should score higher.
    yy, xx = np.mgrid[0:32, 0:32]
    plane = (xx + yy).astype(np.float32)  # constant slope
    rng = np.random.default_rng(0)
    rough = plane + rng.normal(0, 20, size=plane.shape).astype(np.float32)
    assert stratify.tri_riley(rough).mean() > stratify.tri_riley(plane).mean()


def test_tri_matches_hand_computation_at_center():
    a = np.array(
        [[0, 0, 0], [0, 8, 0], [0, 0, 0]], dtype=np.float32
    )
    tri = stratify.tri_riley(a)
    # centre pixel: |8-0| to all 8 neighbours = 8 each, mean = 8
    assert tri[1, 1] == pytest.approx(8.0)


def test_aggregate_tri_over_window_maps_to_overview():
    ov = np.zeros((28, 28), dtype=np.float32)
    ov[:14] = 100.0  # top half rugged when TRI computed
    tri = stratify.tri_riley(ov)
    # a window covering the whole overview
    val = stratify.window_tri(tri, o_r0=0, o_c0=0, o_r1=28, o_c1=28)
    assert val == pytest.approx(tri.mean())


def test_overview_slice_for_window():
    # full-res 3600, x8 overview 450; window at (224, 448) size 224
    sl = stratify.window_overview_slice(
        win_row0=224, win_col0=448, tile=224,
        full_h=3600, full_w=3600, ov_h=450, ov_w=450,
    )
    assert sl == (28, 56, 56, 84)  # 224/8=28, 448/8=56, (448+224)/8=84


# --- Quantile strata (ADR-0001 §4) --------------------------------------------


def test_quantile_strata_split_into_quintiles():
    tri = np.arange(1000, dtype=np.float64)
    edges = stratify.quantile_edges(tri, config.N_STRATA)
    assert len(edges) == config.N_STRATA - 1  # 4 internal cut points
    strata = stratify.assign_strata(tri, edges)
    # roughly equal populations across 5 strata for a uniform distribution
    counts = np.bincount(strata, minlength=config.N_STRATA)
    assert counts.min() >= 180 and counts.max() <= 220
    assert strata.min() == 0 and strata.max() == config.N_STRATA - 1


def test_assign_strata_is_monotone():
    edges = np.array([10.0, 20.0, 30.0, 40.0])
    assert stratify.assign_strata(np.array([5.0]), edges)[0] == 0
    assert stratify.assign_strata(np.array([25.0]), edges)[0] == 2
    assert stratify.assign_strata(np.array([99.0]), edges)[0] == 4


# --- Quota selection (ADR-0001 §5) --------------------------------------------


def _cand(idx, stratum, geocell, cog):
    return stratify.Candidate(
        index=idx, stratum=stratum, geocell=geocell, cog_stem=cog,
        tri=float(idx), tile_id=f"{cog}_{idx}",
    )


def test_select_quota_equal_per_stratum():
    cands = []
    k = 0
    for s in range(config.N_STRATA):
        for g in range(4):
            for _ in range(20):
                cands.append(_cand(k, s, f"g{g}", f"cog{g}"))
                k += 1
    picked = stratify.select_quota(
        cands, per_stratum_quota=10, per_tile_cap=1000, seed=0
    )
    by_stratum = np.bincount([c.stratum for c in picked], minlength=config.N_STRATA)
    assert list(by_stratum) == [10] * config.N_STRATA


def test_select_quota_spreads_across_geocells_round_robin():
    # one stratum, 3 geocells with plenty each; a quota of 6 should take 2 from each
    cands = []
    k = 0
    for g in range(3):
        for _ in range(10):
            cands.append(_cand(k, 0, f"g{g}", f"cog{g}"))
            k += 1
    picked = stratify.select_quota(cands, per_stratum_quota=6, per_tile_cap=1000, seed=0)
    by_cell = {}
    for c in picked:
        by_cell[c.geocell] = by_cell.get(c.geocell, 0) + 1
    assert sorted(by_cell.values()) == [2, 2, 2]


def test_select_quota_respects_per_tile_cap():
    # a single COG dominates one geocell; cap must limit how many of its windows land
    cands = [_cand(i, 0, "g0", "hog") for i in range(100)]
    picked = stratify.select_quota(cands, per_stratum_quota=50, per_tile_cap=5, seed=0)
    assert len(picked) == 5  # capped, quota under-filled (not enough distinct tiles)


def test_select_quota_is_deterministic():
    cands = [_cand(i, i % 5, f"g{i % 7}", f"cog{i % 11}") for i in range(400)]
    a = stratify.select_quota(cands, per_stratum_quota=8, per_tile_cap=3, seed=42)
    b = stratify.select_quota(cands, per_stratum_quota=8, per_tile_cap=3, seed=42)
    assert [c.index for c in a] == [c.index for c in b]
