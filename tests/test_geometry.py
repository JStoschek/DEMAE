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


@pytest.mark.parametrize(
    "origin_lon, origin_lat, stem",
    [
        (6, 45, "Copernicus_DSM_COG_10_N45_00_E006_00_DEM"),
        (-123, -1, "Copernicus_DSM_COG_10_S01_00_W123_00_DEM"),
        (0, 0, "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"),
    ],
)
def test_format_cog_stem_is_the_inverse_of_parse(origin_lon, origin_lat, stem):
    assert geometry.format_cog_stem(origin_lon, origin_lat) == stem
    cog = geometry.parse_cog_key(stem)
    assert (cog.origin_lon, cog.origin_lat) == (origin_lon, origin_lat)


@pytest.mark.parametrize(
    "lon, lat, origin_lon, origin_lat",
    [
        (6.5, 45.2, 6, 45),      # north-east interior
        (-91.05, 33.4, -92, 33), # floor pushes W of the given lon
        (-57.0, -17.6, -57, -18),# south: lat floors poleward
        (129.0, -31.0, 129, -31),
    ],
)
def test_cog_at_floors_to_the_sw_corner(lon, lat, origin_lon, origin_lat):
    cog = geometry.cog_at(lon, lat)
    assert (cog.origin_lon, cog.origin_lat) == (origin_lon, origin_lat)


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


# --- Ground pixel spacing (feeds the metric library's (dy, dx) contract) -------


def test_pixel_spacing_is_square_ish_at_the_equator():
    # A 3600x3600 COG over a 1-degree cell at the equator: ~31 m either way.
    dy, dx = geometry.pixel_spacing_m(0.5, rows_per_degree=3600, cols_per_degree=3600)
    assert dy == pytest.approx(30.92, abs=0.01)
    assert dx == pytest.approx(dy, rel=1e-3)


def test_pixel_spacing_narrows_east_west_with_latitude():
    # Same grid, higher latitude: a degree of longitude is cos(lat) as wide.
    dy, dx = geometry.pixel_spacing_m(60.0, rows_per_degree=3600, cols_per_degree=3600)
    assert dx == pytest.approx(dy * 0.5, rel=1e-6)


def test_pixel_spacing_is_why_copernicus_coarsens_columns_poleward():
    # Copernicus drops to 1800 columns above 60 deg precisely to hold the ground
    # sample distance near 31 m instead of letting it halve (recon §6, ADR-0001 §1).
    _, dx_naive = geometry.pixel_spacing_m(65.0, rows_per_degree=3600, cols_per_degree=3600)
    _, dx_actual = geometry.pixel_spacing_m(65.0, rows_per_degree=3600, cols_per_degree=1800)
    assert dx_naive == pytest.approx(13.07, abs=0.01)
    assert dx_actual == pytest.approx(26.14, abs=0.01)


def test_pixel_spacing_anisotropy_peaks_just_below_fifty_degrees():
    # ADR-0001 §1's worst case, ~1.55:1, and it is not at the pole: it is at the
    # top of the 3600-column band, right before Copernicus coarsens to 2400 and
    # hands most of the width back.
    def ratio(lat, *, cols_per_degree):
        dy, dx = geometry.pixel_spacing_m(
            lat, rows_per_degree=3600, cols_per_degree=cols_per_degree
        )
        return dy / dx

    worst = ratio(49.9, cols_per_degree=3600)
    assert worst == pytest.approx(1.55, abs=0.02)
    assert ratio(50.1, cols_per_degree=2400) < worst  # the coarsening step
    assert ratio(85.5, cols_per_degree=360) < worst  # and the pole is milder still
