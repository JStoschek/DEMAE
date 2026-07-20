"""Window-level pilot scoring frame — Phase B1 (issue #24).

Every test here is zero-network: metrics are scored on hand-built arrays and the
read path runs against the in-memory :class:`~dataset.tiling.SyntheticCogReader`
and :class:`~dataset.masks.SyntheticMaskReader`. The live ~15k-window pass lives in
the :mod:`dataset.pilot` driver, not the suite.

The load-bearing checks are (1) each metric is scored at its **Gate A3 production
resolution** (#21) — the five full-res, the three overview, ``tpi_proxy`` on the
overview 2× box — not twice like A3's calibration; (2) the resolution assignment
partitions the whole ``METRICS`` registry, so a metric added without a home fails
loud; (3) the three mask fractions ride each window; and (4) the frame wraps
``sample_cogs`` (cell-balanced) and accumulates windows in priority order.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dataset import config, metrics, pilot
from dataset.geometry import parse_cog_key, pixel_spacing_m
from dataset.masks import SyntheticMaskReader
from dataset.tiling import SyntheticCogReader

_STEM = "Copernicus_DSM_COG_10_N45_00_E006_00_DEM"
_STEM2 = "Copernicus_DSM_COG_10_N48_00_E009_00_DEM"


# --- Resolution assignment partitions the registry ----------------------------


def test_resolution_assignment_partitions_the_registry():
    # Every metric has exactly one Gate A3 home, and together they cover the registry
    # — a metric added to METRICS without a resolution must not silently vanish.
    assigned = set(pilot.FULL_RES_METRICS) | set(pilot.OVERVIEW_METRICS) | {pilot.BOX_METRIC}
    assert assigned == set(metrics.METRICS)
    # Disjoint: no metric scored at two resolutions.
    total = len(pilot.FULL_RES_METRICS) + len(pilot.OVERVIEW_METRICS) + 1
    assert total == len(metrics.METRICS)


def test_resolution_of_the_a3_verdict():
    # The #21 verdict, pinned: five full-res, three overview, tpi on the box.
    assert pilot._resolution_of("slope_mean") == "full"
    assert pilot._resolution_of("geomorphon_entropy") == "full"
    assert pilot._resolution_of("local_relief") == "overview"
    assert pilot._resolution_of("mean_elevation") == "overview"
    assert pilot._resolution_of("tpi_proxy") == "box"


def test_resolution_of_unassigned_metric_raises():
    with pytest.raises(KeyError):
        pilot._resolution_of("not_a_metric")


# --- window mask fractions ----------------------------------------------------


def test_window_mask_fractions_over_all_window_pixels():
    # EDM: half code 4 (Smoothed, edited), half code 1 (Not edited) -> 0.5 edited.
    edm = np.array([[4, 4], [1, 1]], dtype=np.uint8)
    # FLM: one ASTER fill (3), rest native (2) -> 0.25 fill.
    flm = np.array([[3, 2], [2, 2]], dtype=np.uint8)
    # WBM: one Lake (2), rest No-water (0) -> 0.25 water.
    wbm = np.array([[0, 2], [0, 0]], dtype=np.uint8)
    fr = pilot.window_mask_fractions(edm, flm, wbm)
    assert fr["edm_edited_frac"] == pytest.approx(0.5)
    assert fr["flm_fill_frac"] == pytest.approx(0.25)
    assert fr["wbm_water_frac"] == pytest.approx(0.25)


def test_window_mask_fractions_non_contiguous_flm_fill_counts():
    # FLM 100-102 are national-DEM fill and must count despite the 10..99 gap.
    flm = np.array([[100, 101], [102, 2]], dtype=np.uint8)
    fr = pilot.window_mask_fractions(
        np.ones_like(flm), flm, np.zeros_like(flm)
    )
    assert fr["flm_fill_frac"] == pytest.approx(0.75)


def test_window_mask_fractions_empty_is_nan():
    empty = np.empty((0, 0), dtype=np.uint8)
    fr = pilot.window_mask_fractions(empty, empty, empty)
    assert all(math.isnan(v) for v in fr.values())


# --- score_window: one score per metric, at its production resolution ---------


def _bump_tile(base: float = 100.0, top: float = 200.0) -> np.ndarray:
    """A 12×12 base with a raised central 4×4 plateau (mirrors the calibration fixture).

    The faint W→E+N→S ramp keeps every window from being perfectly flat, so the
    scale-free ratios stay defined; the ramp is tiny next to the 100 m bump.
    """
    yy, xx = np.mgrid[0:12, 0:12].astype(np.float64)
    a = base + 0.01 * xx + 0.01 * yy
    a[4:8, 4:8] = top
    return a


def test_score_window_scores_each_metric_exactly_once():
    a = _bump_tile()
    scores = pilot.score_window(
        a, a, win_row0=4, win_col0=4, full_h=12, full_w=12, ov_h=12, ov_w=12,
        lat=45.0, tile=4,
    )
    assert set(scores) == set(metrics.METRICS)  # nine values, not eighteen


def test_score_window_tpi_reads_the_overview_box_not_the_window():
    a = _bump_tile()
    px = (30.0, 30.0)
    scores = pilot.score_window(
        a, a, win_row0=4, win_col0=4, full_h=12, full_w=12, ov_h=12, ov_w=12,
        lat=45.0, tile=4,
    )
    window = a[4:8, 4:8]
    # The all-200 window scores 0 for within-window position; the 2× box sees the
    # 100 m ring below and scores +100. Production hands tpi the box.
    assert metrics.tpi_proxy(window, px) == pytest.approx(0.0, abs=0.1)
    assert scores["tpi_proxy"] == pytest.approx(100.0, abs=0.5)


def test_overview_box_maps_onto_a_decimated_overview():
    # A genuinely decimated overview (factor 2): the tpi box bounds are decided at
    # full res, then mapped onto the ×2 grid — not read from the full-res one.
    full = np.zeros((24, 24), dtype=np.float64)
    ov = full[::2, ::2]  # 12×12, a real decimation (ov_h < full_h)
    box = pilot._overview_box(
        ov, win_row0=8, win_col0=8, tile=8, ring=4, full_h=24, full_w=24, ov_h=12, ov_w=12
    )
    # box_span at full res is [8-4, 8+8+4] = [4, 20] (16 px); on the ×2 overview that
    # is [2, 10] = 8 px each axis. Reading it off the full grid would give 16.
    assert box is not None and box.shape == (8, 8)
    # A corner window's box runs off the low edge -> None (NaN for tpi only).
    assert pilot._overview_box(
        ov, win_row0=0, win_col0=0, tile=8, ring=4, full_h=24, full_w=24, ov_h=12, ov_w=12
    ) is None


def test_score_window_tpi_reads_the_decimated_overview_box():
    # End-to-end with a decimated overview: an interior window's tpi is finite and
    # scored from the ×2 box, a corner window's is NaN.
    yy, xx = np.mgrid[0:24, 0:24].astype(np.float64)
    full = 100.0 + 0.01 * (xx + yy)
    full[8:16, 8:16] = 200.0  # a central plateau the box can see
    ov = full[::2, ::2]
    interior = pilot.score_window(
        full, ov, win_row0=8, win_col0=8, full_h=24, full_w=24, ov_h=12, ov_w=12,
        lat=45.0, tile=8,
    )
    corner = pilot.score_window(
        full, ov, win_row0=0, win_col0=0, full_h=24, full_w=24, ov_h=12, ov_w=12,
        lat=45.0, tile=8,
    )
    assert np.isfinite(interior["tpi_proxy"]) and interior["tpi_proxy"] > 0
    assert math.isnan(corner["tpi_proxy"])


def test_score_window_off_tile_box_nans_tpi_only():
    a = _bump_tile()
    scores = pilot.score_window(
        a, a, win_row0=0, win_col0=0, full_h=12, full_w=12, ov_h=12, ov_w=12,
        lat=45.0, tile=4,
    )
    assert math.isnan(scores["tpi_proxy"])
    for name in metrics.METRICS:
        if name != "tpi_proxy":
            assert np.isfinite(scores[name]), name


def test_score_window_full_res_metric_reads_full_not_overview():
    # slope_mean is a full-res metric. Make full-res a tilted ramp (real slope) and
    # the overview a flat constant; a non-zero slope proves it read the full array.
    full = np.tile(np.linspace(1000.0, 1100.0, 16), (16, 1))  # rises W->E
    ov = np.full((8, 8), 1050.0)  # the decimated read is flat -> zero slope
    scores = pilot.score_window(
        full, ov, win_row0=0, win_col0=0, full_h=16, full_w=16, ov_h=8, ov_w=8,
        lat=0.0, tile=8,
    )
    assert scores["slope_mean"] > 0.01  # read full-res; the flat overview would give 0


def test_score_window_overview_metric_reads_overview_not_full():
    # mean_elevation is an overview metric. Make full-res and overview disagree on
    # mean, and confirm the score matches the overview read.
    full = np.full((16, 16), 500.0)
    ov = np.full((8, 8), 900.0)  # the decimated read carries a different mean
    scores = pilot.score_window(
        full, ov, win_row0=0, win_col0=0, full_h=16, full_w=16, ov_h=8, ov_w=8,
        lat=0.0, tile=8,
    )
    assert scores["mean_elevation"] == pytest.approx(900.0)


def test_score_window_hands_each_read_its_own_pixel_size():
    ramp = np.tile(np.linspace(1000.0, 1100.0, 12), (12, 1))  # rises W->E
    scores = pilot.score_window(
        ramp, ramp[:, ::2], win_row0=4, win_col0=0, full_h=12, full_w=12,
        ov_h=12, ov_w=6, lat=0.0, tile=4,
    )
    assert np.isfinite(scores["slope_mean"])  # full-res slope, defined


# --- tile_rows over synthetic readers -----------------------------------------


def _hilly(n: int = 672) -> np.ndarray:
    """A smooth deterministic hilly tile, strictly positive so nothing reads as ocean."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    z = 500.0 + 80.0 * np.sin(xx / 40.0) + 60.0 * np.cos(yy / 55.0) + xx / 30.0
    return z.astype(np.float32)


def _mask_reader(stem: str, n: int = 672, *, edit: int = 0) -> SyntheticMaskReader:
    edm = np.ones((n, n), dtype=np.uint8)  # code 1 = Not edited
    if edit:
        edm[:, :edit] = 4  # a strip of Smoothed (edited) pixels
    flm = np.full((n, n), 2, dtype=np.uint8)  # code 2 = native / not filled
    wbm = np.zeros((n, n), dtype=np.uint8)  # code 0 = No water
    return SyntheticMaskReader({stem: {"EDM": edm, "FLM": flm, "WBM": wbm}})


def test_tile_rows_covers_land_windows_with_metrics_and_masks():
    cog = parse_cog_key(_STEM)
    rows, anomaly = pilot.tile_rows(
        SyntheticCogReader({_STEM: _hilly(672)}), _mask_reader(_STEM), cog, factor=8
    )
    assert anomaly is None
    assert len(rows) == 9  # 3×3 land-bearing windows
    for r in rows:
        assert set(r.metrics) == set(metrics.METRICS)
        assert set(r.masks) == set(pilot.MASK_FRACTIONS)
        assert r.band == "0-50"
        assert r.geocell == "+000_+040"
        assert r.stem == _STEM


def test_tile_rows_window_id_is_traceable():
    cog = parse_cog_key(_STEM)
    rows, _ = pilot.tile_rows(
        SyntheticCogReader({_STEM: _hilly(672)}), _mask_reader(_STEM), cog, factor=8
    )
    r = rows[0]
    assert r.window_id == f"{_STEM}:{r.win_row0}:{r.win_col0}"


def test_tile_rows_mask_fraction_tracks_the_edited_strip():
    # A 224-px edited strip on the west edge -> the westmost window column is fully
    # edited (frac 1.0), the rest untouched (frac 0.0).
    cog = parse_cog_key(_STEM)
    rows, _ = pilot.tile_rows(
        SyntheticCogReader({_STEM: _hilly(672)}), _mask_reader(_STEM, edit=224), cog,
        factor=8,
    )
    west = [r for r in rows if r.win_col0 == 0]
    east = [r for r in rows if r.win_col0 != 0]
    assert west and all(r.masks["edm_edited_frac"] == pytest.approx(1.0) for r in west)
    assert east and all(r.masks["edm_edited_frac"] == pytest.approx(0.0) for r in east)


def test_tile_rows_skips_ocean_only_windows():
    a = _hilly(672)
    a[:, :224] = config.OCEAN_FILL_VALUE  # the west column of windows is all ocean
    cog = parse_cog_key(_STEM)
    rows, _ = pilot.tile_rows(
        SyntheticCogReader({_STEM: a}), _mask_reader(_STEM), cog, factor=8
    )
    assert len(rows) == 6  # 9 minus the 3 ocean-only windows
    assert all(r.win_col0 != 0 for r in rows)


def test_tile_rows_absent_mask_nans_the_columns_and_surfaces_anomaly():
    # A COG whose masks are absent still yields metric rows; the mask columns go NaN
    # and the absence is surfaced, never read as "no edits".
    cog = parse_cog_key(_STEM)
    empty_masks = SyntheticMaskReader({})  # no masks registered -> MaskNotFound
    rows, anomaly = pilot.tile_rows(
        SyntheticCogReader({_STEM: _hilly(672)}), empty_masks, cog, factor=8
    )
    assert anomaly is not None and _STEM in anomaly[0]
    assert len(rows) == 9
    for r in rows:
        assert all(math.isnan(v) for v in r.masks.values())
        assert np.isfinite(r.metrics["slope_mean"])  # metrics unaffected


# --- measure: priority-order accumulation to a window target ------------------


def _reader_two_tiles() -> tuple[SyntheticCogReader, SyntheticMaskReader]:
    cog_reader = SyntheticCogReader({_STEM: _hilly(672), _STEM2: _hilly(672)})
    masks = {
        s: {
            "EDM": np.ones((672, 672), dtype=np.uint8),
            "FLM": np.full((672, 672), 2, dtype=np.uint8),
            "WBM": np.zeros((672, 672), dtype=np.uint8),
        }
        for s in (_STEM, _STEM2)
    }
    return cog_reader, SyntheticMaskReader(masks)


def test_measure_accumulates_to_the_window_target():
    cog_reader, mask_reader = _reader_two_tiles()
    cogs = [parse_cog_key(_STEM), parse_cog_key(_STEM2)]
    rows, anomalies = pilot.measure(
        cog_reader, mask_reader, cogs, target_windows=9, factor=8, workers=2
    )
    assert anomalies == []
    # Nine windows per tile; the target stops accumulation after the first tile.
    assert len(rows) == 9


def test_measure_reads_cogs_in_priority_order():
    cog_reader, mask_reader = _reader_two_tiles()
    cogs = [parse_cog_key(_STEM), parse_cog_key(_STEM2)]
    rows, _ = pilot.measure(
        cog_reader, mask_reader, cogs, target_windows=5, factor=8, workers=2
    )
    # With target 5 < 9, only the first COG's windows should appear.
    assert {r.stem for r in rows} == {_STEM}


def test_measure_surfaces_a_dead_cog_without_dropping_the_rest():
    cog_reader, mask_reader = _reader_two_tiles()
    missing = parse_cog_key("Copernicus_DSM_COG_10_N10_00_E010_00_DEM")
    cogs = [missing, parse_cog_key(_STEM)]
    rows, anomalies = pilot.measure(
        cog_reader, mask_reader, cogs, target_windows=100, factor=8, workers=2
    )
    assert len(anomalies) == 1 and "N10" in anomalies[0][0]
    assert {r.stem for r in rows} == {_STEM}  # the readable COG still scores


# --- report + rows ------------------------------------------------------------


def _rows(n: int = 4) -> list[pilot.WindowRow]:
    cog_reader, mask_reader = _reader_two_tiles()
    rows, _ = pilot.tile_rows(cog_reader, mask_reader, parse_cog_key(_STEM), factor=8)
    return rows[:n]


def test_to_rows_carries_metrics_masks_and_geography():
    flat = pilot.to_rows(_rows(3))
    assert len(flat) == 3
    r = flat[0]
    for m in metrics.METRICS:
        assert m in r
    for f in pilot.MASK_FRACTIONS:
        assert f in r
    assert r["window_id"] and r["band"] == "0-50" and r["geocell"] == "+000_+040"


def test_build_report_records_the_cell_balanced_decision_and_a3_split():
    report = pilot.build_report(
        _rows(9), [], seed=config.RANDOM_SEED, allocation="proportional",
        target_windows=9, cogs_read=1, factor=8, code_version="test",
        snapshot_date="2026-07-20",
    )
    assert "#24" in report
    assert "cell-balanced" in report.lower() or "proportional" in report.lower()
    assert "slope_mean" in report  # the full-res metrics are named
    assert "tpi_proxy" in report
