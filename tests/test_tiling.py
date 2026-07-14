"""COG-reader interface: synthetic reads, and an opt-in real-S3 smoke test."""

from __future__ import annotations

import os

import numpy as np
import pytest

from dataset import config, tiling
from dataset.geometry import grid_windows, parse_cog_key

STEM = "Copernicus_DSM_COG_10_N45_00_E006_00_DEM"


def _make_reader(h=448, w=448):
    rng = np.random.default_rng(0)
    arr = rng.normal(1000, 200, size=(h, w)).astype(np.float32)
    return tiling.SyntheticCogReader({STEM: arr}), arr


def test_synthetic_lists_parsed_refs():
    reader, _ = _make_reader()
    refs = reader.list_cogs()
    assert len(refs) == 1
    assert refs[0] == parse_cog_key(STEM)


def test_synthetic_overview_is_decimated_with_full_dims():
    reader, arr = _make_reader(448, 448)
    ov = reader.read_overview(reader.list_cogs()[0], factor=8)
    assert ov.full_h == 448 and ov.full_w == 448
    assert ov.array.shape == (56, 56)  # 448 / 8
    assert np.array_equal(ov.array, arr[::8, ::8])


def test_synthetic_windows_match_grid_slices():
    reader, arr = _make_reader(448, 448)
    cog = reader.list_cogs()[0]
    origins = grid_windows(width=448, height=448)
    wins = reader.read_windows(cog, origins, config.TILE_PX)
    assert len(wins) == 4
    for (r0, c0), win in zip(origins, wins):
        assert win.shape == (config.TILE_PX, config.TILE_PX)
        assert np.array_equal(win, arr[r0 : r0 + 224, c0 : c0 + 224])


@pytest.mark.skipif(
    os.environ.get("DEMAE_S3_TEST") != "1",
    reason="set DEMAE_S3_TEST=1 to hit the live Copernicus bucket",
)
def test_rasterio_reader_reads_a_real_cog():
    reader = tiling.RasterioCogReader()
    cog = parse_cog_key(STEM)
    ov = reader.read_overview(cog, factor=8)
    assert ov.full_h == 3600 and ov.full_w == 3600
    win = reader.read_windows(cog, [(0, 0)], config.TILE_PX)[0]
    assert win.shape == (config.TILE_PX, config.TILE_PX)
