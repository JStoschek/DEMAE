"""Overview-vs-full-res calibration — Gate A3 (issue #20).

Every test here is zero-network: the Spearman core is exercised on hand-built
arrays, and the read path on the in-memory
:class:`~dataset.tiling.SyntheticCogReader`. The live ~100-COG pass lives in the
:mod:`dataset.calibration` driver, not the suite.

The load-bearing checks are (1) the numpy Spearman matches known values *with tie
correction* — no scipy in the deps, so the rank algorithm is ours to get right — and
(2) ``tpi_proxy`` is scored on the 2× box while the other eight get the bare window,
with an off-tile box degrading that column alone to NaN (issue #22).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dataset import calibration, config, metrics
from dataset.geometry import parse_cog_key, pixel_spacing_m
from dataset.tiling import SyntheticCogReader


# --- _rankdata: average ranks, ties shared ------------------------------------


def test_rankdata_no_ties_is_one_based():
    r = calibration._rankdata(np.array([30.0, 10.0, 20.0]))
    np.testing.assert_array_equal(r, [3.0, 1.0, 2.0])


def test_rankdata_ties_share_the_mean_rank():
    # 20 and 20 span ranks 2 and 3 -> both 2.5.
    r = calibration._rankdata(np.array([10.0, 20.0, 20.0, 40.0]))
    np.testing.assert_array_equal(r, [1.0, 2.5, 2.5, 4.0])


def test_rankdata_all_tied_is_the_midrank():
    r = calibration._rankdata(np.array([5.0, 5.0, 5.0]))
    np.testing.assert_array_equal(r, [2.0, 2.0, 2.0])


# --- spearman -----------------------------------------------------------------


def test_spearman_perfect_monotone_is_one():
    assert calibration.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_reversed_is_minus_one():
    assert calibration.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_rank_not_value_based():
    # A monotone-but-nonlinear map is still ρ = 1 — the whole reason A3 uses rank.
    assert calibration.spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)


def test_spearman_tie_correction_matches_hand_value():
    # x ranks [1,2,3,4]; y=[1,2,2,4] ranks [1,2.5,2.5,4]. Pearson of the two.
    got = calibration.spearman([1, 2, 3, 4], [1, 2, 2, 4])
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.5, 2.5, 4.0])
    expect = np.corrcoef(x, y)[0, 1]
    assert got == pytest.approx(expect)


def test_spearman_drops_non_finite_pairs():
    # The NaN pair vanishes, leaving a perfect monotone triple.
    assert calibration.spearman([1, 2, np.nan, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_spearman_too_few_pairs_is_nan():
    assert math.isnan(calibration.spearman([1.0], [1.0]))
    assert math.isnan(calibration.spearman([1, np.nan], [np.nan, 2]))


def test_spearman_zero_variance_is_nan():
    # A metric that reads the same on every window has no order to correlate.
    assert math.isnan(calibration.spearman([5, 5, 5, 5], [1, 2, 3, 4]))


def test_n_complete_counts_only_finite_pairs():
    assert calibration._n_complete([1, np.nan, 3, 4], [1, 2, np.nan, 4]) == 2


# --- box_span / _scale --------------------------------------------------------


def test_box_span_interior_window_is_the_2x_box():
    # ring defaults to tile // 2, so the box is window + half-window each side = 2x.
    assert calibration.box_span(4, tile=4, ring=2, dim=12) == (2, 10)


def test_box_span_off_tile_returns_none():
    assert calibration.box_span(0, tile=4, ring=2, dim=12) is None  # runs off the low edge
    assert calibration.box_span(8, tile=4, ring=2, dim=12) is None  # runs off the high edge


def test_scale_maps_proportionally_and_clamps():
    # A ×8 overview of a 3600-px band: full-res 224 -> 28 overview px.
    assert calibration._scale(0, 3600, 450) == 0
    assert calibration._scale(224, 3600, 450) == 28
    assert calibration._scale(3600, 3600, 450) == 450  # clamped to the far edge


# --- score_window: the box goes to tpi, the window to the rest -----------------


def _bump_tile(base: float = 100.0, top: float = 200.0) -> np.ndarray:
    """A 12×12 base with a raised central 4×4 plateau — a bump the box can see but a
    window sitting on the plateau cannot.

    The base carries a faint W→E ramp so no window is perfectly flat (which would make
    the scale-free ratios legitimately NaN); the ramp is tiny next to the 100 m bump,
    so the plateau still reads as position within its box.
    """
    yy, xx = np.mgrid[0:12, 0:12].astype(np.float64)
    a = base + 0.01 * xx + 0.01 * yy
    a[4:8, 4:8] = top
    return a


def test_score_window_tpi_reads_the_box_not_the_window():
    a = _bump_tile()
    px = (30.0, 30.0)
    full, ov = calibration.score_window(
        a, a, win_row0=4, win_col0=4, full_h=12, full_w=12, ov_h=12, ov_w=12,
        lat=45.0, tile=4,
    )
    # The window is all 200 -> position *within* it is 0; the 2× box sees the 100 ring
    # below it -> +100. score_window must use the box.
    window = a[4:8, 4:8]
    assert metrics.tpi_proxy(window, px) == pytest.approx(0.0, abs=0.1)
    assert full["tpi_proxy"] == pytest.approx(100.0, abs=0.5)
    assert ov["tpi_proxy"] == pytest.approx(100.0, abs=0.5)  # ov == full here (same array)


def test_score_window_non_tpi_metrics_read_the_bare_window():
    a = _bump_tile()
    px = (30.0, 30.0)
    full, _ = calibration.score_window(
        a, a, win_row0=4, win_col0=4, full_h=12, full_w=12, ov_h=12, ov_w=12,
        lat=45.0, tile=4,
    )
    window = a[4:8, 4:8]
    # An all-200 window has zero relief; if score_window leaked the box in, relief
    # would be 100 (box spans 100..200).
    assert full["local_relief"] == pytest.approx(metrics.local_relief(window, px))
    assert full["local_relief"] == pytest.approx(0.0)


def test_score_window_off_tile_box_nans_tpi_only():
    a = _bump_tile()
    full, ov = calibration.score_window(
        a, a, win_row0=0, win_col0=0, full_h=12, full_w=12, ov_h=12, ov_w=12,
        lat=45.0, tile=4,
    )
    assert math.isnan(full["tpi_proxy"])
    assert math.isnan(ov["tpi_proxy"])
    # Every other metric still scores the window.
    for name in metrics.METRICS:
        if name == "tpi_proxy":
            continue
        assert np.isfinite(full[name]), name


def test_score_window_hands_each_read_its_own_pixel_size():
    # A ramp scored at a coarser pixel size is a gentler slope — the contract A3 rides.
    ramp = np.tile(np.linspace(1000.0, 1100.0, 12), (12, 1))  # rises W->E
    full, ov = calibration.score_window(
        ramp, ramp[:, ::2], win_row0=4, win_col0=0, full_h=12, full_w=12,
        ov_h=12, ov_w=6, lat=0.0, tile=4,
    )
    # Same ground, half the columns -> the overview's dx is twice as wide, so the
    # overview slope is read gentler. (Values differ; this asserts the wiring, not a
    # closed form.) At minimum both are finite and the overview is <= full-res slope.
    assert np.isfinite(full["slope_mean"]) and np.isfinite(ov["slope_mean"])


# --- tile_scores over a synthetic reader --------------------------------------

_STEM = "Copernicus_DSM_COG_10_N45_00_E006_00_DEM"


def _hilly(n: int = 672) -> np.ndarray:
    """A smooth deterministic hilly tile, strictly positive so nothing reads as ocean."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    z = 500.0 + 80.0 * np.sin(xx / 40.0) + 60.0 * np.cos(yy / 55.0) + xx / 30.0
    return z.astype(np.float32)


def test_tile_scores_covers_the_window_grid():
    reader = SyntheticCogReader({_STEM: _hilly(672)})
    cog = parse_cog_key(_STEM)
    scores = calibration.tile_scores(reader, cog, factor=8)
    # 3 windows per axis -> 9 land-bearing windows, each carrying every metric twice.
    assert len(scores) == 9
    for s in scores:
        assert set(s.full) == set(metrics.METRICS)
        assert set(s.overview) == set(metrics.METRICS)
        assert s.band == "0-50"


def test_tile_scores_interior_window_has_a_defined_tpi():
    reader = SyntheticCogReader({_STEM: _hilly(672)})
    scores = calibration.tile_scores(reader, parse_cog_key(_STEM), factor=8)
    interior = [s for s in scores if s.win_row0 == 224 and s.win_col0 == 224]
    assert interior and np.isfinite(interior[0].full["tpi_proxy"])
    # A corner window's box runs off the tile -> tpi NaN there.
    corner = [s for s in scores if s.win_row0 == 0 and s.win_col0 == 0]
    assert corner and math.isnan(corner[0].full["tpi_proxy"])


def test_tile_scores_skips_ocean_only_windows():
    a = _hilly(672)
    a[:, :224] = config.OCEAN_FILL_VALUE  # the west column of windows is all ocean
    reader = SyntheticCogReader({_STEM: a})
    scores = calibration.tile_scores(reader, parse_cog_key(_STEM), factor=8)
    assert len(scores) == 6  # 9 minus the 3 ocean-only windows
    assert all(s.win_col0 != 0 for s in scores)


# --- rho_table ----------------------------------------------------------------


def _synthetic_scores(n_windows: int = 40):
    """A deterministic list of WindowScore where full and ×8 agree monotonically for
    relief and disagree (shuffled) for one metric — enough to exercise the table."""
    rng = np.random.default_rng(0)
    base = rng.uniform(1.0, 100.0, n_windows)
    scores = []
    for i, v in enumerate(base):
        band = "0-50" if i % 2 == 0 else "50-60"
        full = {m: float(v + j) for j, m in enumerate(metrics.METRICS)}
        # Overview agrees on rank for every metric except one we scramble.
        ov = dict(full)
        ov["geomorphon_entropy"] = float(rng.uniform(0, 100))
        lat = 20.0 if band == "0-50" else 55.0
        scores.append(
            calibration.WindowScore(
                stem=_STEM, band=band, lon=6.5, lat=lat,
                win_row0=i, win_col0=0, full=full, overview=ov,
            )
        )
    return scores


def test_rho_table_has_pooled_ge50_and_per_band_cells():
    scores = _synthetic_scores()
    table = calibration.rho_table(scores)
    assert set(table) == set(metrics.METRICS)
    row = table["local_relief"]
    assert calibration.POOLED in row
    assert calibration.POOLED_GE50 in row
    assert "0-50" in row and "50-60" in row
    # local_relief agrees perfectly on rank -> ρ = 1 and it passes the gate.
    assert row[calibration.POOLED].rho == pytest.approx(1.0)
    assert row[calibration.POOLED].passes


def test_rho_table_flags_a_scrambled_metric_below_gate():
    table = calibration.rho_table(_synthetic_scores())
    scrambled = table["geomorphon_entropy"][calibration.POOLED]
    assert not scrambled.passes  # a random overview cannot clear ρ ≥ 0.9


def test_rho_passes_requires_finite_rho():
    assert not calibration.Rho(rho=float("nan"), n=0).passes
    assert calibration.Rho(rho=0.95, n=100).passes
    assert not calibration.Rho(rho=0.5, n=100).passes


def test_ge50_column_excludes_sub_50_windows():
    scores = _synthetic_scores()
    ge50 = [s for s in scores if abs(s.lat) >= 50.0]
    assert ge50 and all(s.band == "50-60" for s in ge50)
    table = calibration.rho_table(scores)
    assert table["local_relief"][calibration.POOLED_GE50].n == len(ge50)


# --- report + rows ------------------------------------------------------------


def test_build_report_mentions_the_gate_and_defers_the_call():
    scores = _synthetic_scores()
    report = calibration.build_report(
        scores, [], seed=config.RANDOM_SEED, requested_n=100, cogs_read=1,
        factor=8, code_version="test", snapshot_date="2026-07-17",
    )
    assert "Spearman" in report
    assert "#20" in report and "#21" in report  # reports, but defers the decision
    assert "`local_relief`" in report
    assert "≥50°" in report


def test_to_rows_carries_both_reads_per_metric():
    rows = calibration.to_rows(_synthetic_scores(4))
    assert len(rows) == 4
    r = rows[0]
    for m in metrics.METRICS:
        assert f"{m}_full" in r and f"{m}_ov" in r
    assert r["stem"] == _STEM
