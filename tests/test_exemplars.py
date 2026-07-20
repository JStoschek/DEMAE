"""The flat-wet / flat-dry exemplar set — Phase B3 (issue #25).

Zero-network, like :mod:`tests.test_pilot`: every test scores hand-built arrays
through the in-memory :class:`~dataset.tiling.SyntheticCogReader` /
:class:`~dataset.masks.SyntheticMaskReader`. The live 8-exemplar pass lives in the
:func:`dataset.exemplars.main` driver, not the suite.

The load-bearing checks are (1) each exemplar resolves to the production grid
window nearest its point; (2) the window is scored by the pilot's own core, so its
metrics/masks/window_id are the B1 frame's; (3) the ground-truth class rides each
row into the CSV and the separation report; and (4) a dead COG is surfaced, not
swallowed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dataset import config, exemplars, metrics
from dataset.exemplars import Exemplar
from dataset.geometry import cog_at, parse_cog_key, window_center_lonlat
from dataset.masks import SyntheticMaskReader
from dataset.tiling import Overview, SyntheticCogReader

# A COG at a known origin so window centres are predictable: N45 E006 spans
# lon [6, 7], lat [45, 46].
_STEM = "Copernicus_DSM_COG_10_N45_00_E006_00_DEM"


# --- The curated set is well-formed -------------------------------------------


def test_exemplar_set_is_balanced_wet_and_dry():
    wet = [e for e in exemplars.EXEMPLARS if e.klass == exemplars.FLAT_WET]
    dry = [e for e in exemplars.EXEMPLARS if e.klass == exemplars.FLAT_DRY]
    # A contrast set needs both classes, and a balanced one is the honest control.
    assert len(wet) == len(dry) >= 3
    # Every exemplar is one of the two known classes, named, and annotated.
    for e in exemplars.EXEMPLARS:
        assert e.klass in {exemplars.FLAT_WET, exemplars.FLAT_DRY}
        assert e.name and e.note
        assert -180.0 <= e.lon <= 180.0 and -90.0 <= e.lat <= 90.0


def test_exemplar_cog_floors_the_point_to_its_cell():
    ex = Exemplar("test", exemplars.FLAT_DRY, lon=-102.5, lat=34.2)
    assert ex.cog == cog_at(-102.5, 34.2)
    assert ex.cog.origin_lon == -103 and ex.cog.origin_lat == 34


# --- nearest_window: snap a point to the production grid ----------------------


def _overview(full: int = 672, factor: int = 8) -> Overview:
    ov_dim = full // factor
    return Overview(array=np.zeros((ov_dim, ov_dim), np.float32), full_h=full, full_w=full)


def test_nearest_window_returns_a_real_grid_origin():
    cog = parse_cog_key(_STEM)
    ov = _overview(672)
    # Target the centre of the cell (lon 6.5, lat 45.5); with a 672-px COG and
    # 224-px tiles the grid is 3×3, whose centre window origin is (224, 224).
    origin = exemplars.nearest_window(ov, cog, lon=6.5, lat=45.5, tile=224)
    assert origin == (224, 224)


def test_nearest_window_picks_the_closest_centre():
    cog = parse_cog_key(_STEM)
    ov = _overview(672)
    # A point near the NW corner (lon just east of 6, lat just south of 46) must land
    # in the NW window, origin (0, 0).
    origin = exemplars.nearest_window(ov, cog, lon=6.05, lat=45.95, tile=224)
    assert origin == (0, 0)
    # And the resolved window's centre really is the closest of the nine.
    wlon, wlat = window_center_lonlat(
        origin_lon=cog.origin_lon, origin_lat=cog.origin_lat,
        width=672, height=672, win_row0=0, win_col0=0, tile=224,
    )
    assert abs(wlon - 6.05) < 0.4 and abs(wlat - 45.95) < 0.4


# --- score_exemplar: the pilot core, one window -------------------------------


def _hilly(n: int = 672) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    z = 500.0 + 80.0 * np.sin(xx / 40.0) + 60.0 * np.cos(yy / 55.0) + xx / 30.0
    return z.astype(np.float32)


def _masks(stem: str, n: int = 672, *, water: int = 0) -> SyntheticMaskReader:
    edm = np.ones((n, n), dtype=np.uint8)      # 1 = Not edited
    flm = np.full((n, n), 2, dtype=np.uint8)   # 2 = native
    wbm = np.zeros((n, n), dtype=np.uint8)     # 0 = No water
    if water:
        wbm[:, :water] = 1  # a strip of Ocean (WBM 1) on the west edge
    return SyntheticMaskReader({stem: {"EDM": edm, "FLM": flm, "WBM": wbm}})


def test_score_exemplar_scores_the_resolved_window_through_the_b1_frame():
    ex = Exemplar("cell centre", exemplars.FLAT_DRY, lon=6.5, lat=45.5)
    reader = SyntheticCogReader({_STEM: _hilly(672)})
    row, anomaly = exemplars.score_exemplar(reader, _masks(_STEM), ex, factor=8)
    assert anomaly is None
    # Nine metrics, three masks — the same frame shape the pilot writes.
    assert set(row.window.metrics) == set(metrics.METRICS)
    assert set(row.window.masks) == set(exemplars.MASK_FRACTIONS)
    # The window id is a real, traceable grid origin on the exemplar's COG.
    assert row.window.window_id == f"{_STEM}:{row.window.win_row0}:{row.window.win_col0}"
    assert row.window.stem == _STEM
    assert row.exemplar is ex


def test_score_exemplar_window_centre_is_near_the_target_point():
    ex = Exemplar("near NW", exemplars.FLAT_WET, lon=6.1, lat=45.9)
    reader = SyntheticCogReader({_STEM: _hilly(672)})
    row, _ = exemplars.score_exemplar(reader, _masks(_STEM), ex, factor=8)
    # The resolved window centre should be within half a tile (~0.33°) of the target.
    assert abs(row.window.lon - ex.lon) < 0.34
    assert abs(row.window.lat - ex.lat) < 0.34


def test_score_exemplar_reads_water_into_the_wbm_fraction():
    # A wet exemplar whose resolved NW window sits over a full-width water strip
    # should carry wbm_water_frac == 1 there — the label reaching the masks.
    ex = Exemplar("water NW", exemplars.FLAT_WET, lon=6.05, lat=45.95)
    reader = SyntheticCogReader({_STEM: _hilly(672)})
    row, _ = exemplars.score_exemplar(reader, _masks(_STEM, water=224), ex, factor=8)
    assert row.window.win_col0 == 0
    assert row.window.masks["wbm_water_frac"] == pytest.approx(1.0)


def test_score_exemplar_absent_mask_nans_columns_and_surfaces_anomaly():
    ex = Exemplar("no masks", exemplars.FLAT_DRY, lon=6.5, lat=45.5)
    reader = SyntheticCogReader({_STEM: _hilly(672)})
    row, anomaly = exemplars.score_exemplar(reader, SyntheticMaskReader({}), ex, factor=8)
    assert anomaly is not None and _STEM in anomaly[0]
    assert all(math.isnan(v) for v in row.window.masks.values())
    assert np.isfinite(row.window.metrics["slope_mean"])  # metrics unaffected


# --- measure: score the set, surface failures ---------------------------------


def _two_cog_readers():
    stem2 = "Copernicus_DSM_COG_10_S18_00_W057_00_DEM"  # Pantanal cell
    reader = SyntheticCogReader({_STEM: _hilly(672), stem2: _hilly(672)})
    masks = SyntheticMaskReader({
        s: {
            "EDM": np.ones((672, 672), np.uint8),
            "FLM": np.full((672, 672), 2, np.uint8),
            "WBM": np.zeros((672, 672), np.uint8),
        }
        for s in (_STEM, stem2)
    })
    return reader, masks


def test_measure_scores_every_exemplar_in_order():
    reader, masks = _two_cog_readers()
    exs = [
        Exemplar("A", exemplars.FLAT_WET, lon=-56.5, lat=-17.6),  # S18 W057 cell
        Exemplar("B", exemplars.FLAT_DRY, lon=6.5, lat=45.5),     # N45 E006 cell
    ]
    rows, anomalies = exemplars.measure(reader, masks, exs, factor=8)
    assert anomalies == []
    assert [r.exemplar.name for r in rows] == ["A", "B"]


def test_measure_surfaces_a_dead_cog_without_dropping_the_rest():
    reader, masks = _two_cog_readers()
    exs = [
        Exemplar("missing", exemplars.FLAT_WET, lon=10.5, lat=10.5),  # no such tile
        Exemplar("present", exemplars.FLAT_DRY, lon=6.5, lat=45.5),
    ]
    rows, anomalies = exemplars.measure(reader, masks, exs, factor=8)
    assert len(anomalies) == 1 and "missing" in anomalies[0][0]
    assert [r.exemplar.name for r in rows] == ["present"]


# --- to_rows + report ---------------------------------------------------------


def _scored_pair():
    reader, masks = _two_cog_readers()
    exs = [
        Exemplar("Wetland", exemplars.FLAT_WET, lon=-56.5, lat=-17.6, note="wet note"),
        Exemplar("Plateau", exemplars.FLAT_DRY, lon=6.5, lat=45.5, note="dry note"),
    ]
    return exemplars.measure(reader, masks, exs, factor=8)


def test_to_rows_carries_label_window_id_scores_and_masks():
    rows, _ = _scored_pair()
    flat = exemplars.to_rows(rows)
    assert len(flat) == 2
    r = flat[0]
    assert r["name"] == "Wetland" and r["klass"] == exemplars.FLAT_WET
    assert r["window_id"] and r["note"] == "wet note"
    for m in metrics.METRICS:
        assert m in r
    for f in exemplars.MASK_FRACTIONS:
        assert f in r
    # Both the target point and the resolved window centre are recorded.
    assert r["target_lon"] == pytest.approx(-56.5)
    assert "win_lon" in r and "win_lat" in r


def test_build_report_shows_the_wet_dry_separation_and_class_labels():
    rows, anomalies = _scored_pair()
    report = exemplars.build_report(
        rows, anomalies, factor=8, code_version="test", snapshot_date="2026-07-20",
    )
    assert "#25" in report
    assert exemplars.FLAT_WET in report and exemplars.FLAT_DRY in report
    assert "Wetland" in report and "Plateau" in report
    assert "slope_mean" in report and "tpi_proxy" in report
    assert "wbm_water_frac" in report  # the mask sanity check is present
