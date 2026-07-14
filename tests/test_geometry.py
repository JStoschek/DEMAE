"""Geometry: COG-key parsing, lat bands, geocells, window grid, centre coords."""

from __future__ import annotations

import math

import pytest

from dataset import config, geometry


# --- COG key parsing ----------------------------------------------------------


def test_parse_cog_key_northeast():
    cog = geometry.parse_cog_key(
        "Copernicus_DSM_COG_10_N45_00_E006_00_DEM/"
        "Copernicus_DSM_COG_10_N45_00_E006_00_DEM.tif"
    )
    assert cog.origin_lat == 45
    assert cog.origin_lon == 6
    assert cog.stem == "Copernicus_DSM_COG_10_N45_00_E006_00_DEM"


def test_parse_cog_key_southwest_from_prefix():
    cog = geometry.parse_cog_key("Copernicus_DSM_COG_10_S01_00_W123_00_DEM")
    assert cog.origin_lat == -1
    assert cog.origin_lon == -123


def test_parse_cog_key_rejects_junk():
    with pytest.raises(ValueError):
        geometry.parse_cog_key("not-a-copernicus-key")


# --- Latitude bands (resolution bands, recon §6 / ADR-0002) -------------------


@pytest.mark.parametrize(
    "lat,band",
    [
        (0.5, "0-50"),
        (49.9, "0-50"),
        (50.0, "50-60"),  # edge is the lower bound of its band
        (-52.0, "50-60"),  # symmetric poleward
        (65.0, "60-70"),
        (78.0, "70-80"),
        (82.0, "80-85"),
        (88.0, "85-90"),
        (90.0, "85-90"),  # inclusive top
    ],
)
def test_lat_band(lat, band):
    assert geometry.lat_band(lat) == band


# --- Geocells (10-degree lon/lat cell, ADR-0001 §5) ---------------------------


def test_geocell_id_is_sw_corner_multiple_of_ten():
    # lon 6.5, lat 45.2 -> SW corner (0, 40)
    assert geometry.geocell_id(6.5, 45.2) == geometry.geocell_id(3.0, 41.0)
    # a different cell differs
    assert geometry.geocell_id(6.5, 45.2) != geometry.geocell_id(16.0, 45.2)


def test_geocell_id_negative():
    # lon -123.4 -> floor to -130; lat -1.2 -> floor to -10
    cid = geometry.geocell_id(-123.4, -1.2)
    assert cid == geometry.geocell_id(-121.0, -9.9)
    assert cid != geometry.geocell_id(-123.4, 1.2)


# --- Window grid (ADR-0001 §2) ------------------------------------------------


def test_window_grid_divisible_axis_has_no_overlap():
    # 3600 wide/high @ 224 -> floor = 16 windows per axis
    starts = geometry.axis_starts(3600)
    assert len(starts) == 3600 // config.TILE_PX  # 16
    assert starts[0] == 0
    # last window is flushed to the edge
    assert starts[-1] == 3600 - config.TILE_PX
    # strictly increasing
    assert all(b > a for a, b in zip(starts, starts[1:]))


def test_window_grid_count_matches_floor():
    for dim, n in [(3600, 16), (2400, 10), (1800, 8), (1200, 5), (720, 3), (360, 1)]:
        assert len(geometry.axis_starts(dim)) == n


def test_window_grid_last_flushed_to_edge_when_not_divisible():
    starts = geometry.axis_starts(2400)  # 2400/224 = 10.7 -> 10 windows
    assert starts[-1] == 2400 - config.TILE_PX  # 2176, flush to edge
    # the penultimate is on the regular stride
    assert starts[-2] == 8 * config.TILE_PX


def test_grid_windows_is_row_major_product():
    wins = geometry.grid_windows(width=448, height=448)  # 2x2
    assert wins == [(0, 0), (0, 224), (224, 0), (224, 224)]


# --- Window-centre coordinates ------------------------------------------------


def test_window_centre_near_nw_corner():
    lon, lat = geometry.window_center_lonlat(
        origin_lon=6, origin_lat=45, width=3600, height=3600, win_row0=0, win_col0=0
    )
    # centre of the top-left 224 window sits just inside the NW corner (6, 46)
    assert lon == pytest.approx(6 + 112 / 3600)
    assert lat == pytest.approx(46 - 112 / 3600)


def test_window_centre_uses_band_width_for_lon_res():
    # a poleward tile is only 1800 px wide -> coarser lon spacing
    lon, _ = geometry.window_center_lonlat(
        origin_lon=10, origin_lat=62, width=1800, height=3600, win_row0=0, win_col0=0
    )
    assert lon == pytest.approx(10 + 112 / 1800)


def test_window_centre_stays_inside_the_cell():
    for r0 in geometry.axis_starts(3600):
        for c0 in geometry.axis_starts(3600):
            lon, lat = geometry.window_center_lonlat(
                origin_lon=6, origin_lat=45, width=3600, height=3600,
                win_row0=r0, win_col0=c0,
            )
            assert 6 <= lon <= 7
            assert 45 <= lat <= 46
            assert not math.isnan(lon) and not math.isnan(lat)
