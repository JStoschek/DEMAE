"""Full-res tile preparation: clip, land-only stats, degenerate drop, int16 raw.

Storage is raw int16 metres (ADR-0002); the per-tile mean/std computed here are
stored in the manifest for the loader to normalize with, not baked into the pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset import config, normalize


def test_plausibility_clip_bounds():
    a = np.array([-9999.0, -500.0, 0.0, 9000.0, 12345.0], dtype=np.float32)
    clipped = normalize.plausibility_clip(a)
    assert clipped.min() == config.CLIP_MIN_M
    assert clipped.max() == config.CLIP_MAX_M
    # real terrain in-range is untouched (no percentile clip, ADR-0001 §6)
    assert clipped[2] == 0.0


def test_land_mask_excludes_exact_zero_only():
    a = np.array([[0.0, -12.0], [5.0, 0.0]], dtype=np.float32)
    m = normalize.land_mask(a)
    assert m.tolist() == [[False, True], [True, False]]  # -12 m polder is land


def test_land_stats_ignore_ocean_pixels():
    a = np.array([0.0, 0.0, 10.0, 20.0, 30.0], dtype=np.float32)
    mask = normalize.land_mask(a)
    mean, std = normalize.land_stats(a, mask)
    assert mean == pytest.approx(20.0)  # mean of 10,20,30 — zeros excluded
    assert std == pytest.approx(np.std([10.0, 20.0, 30.0]))


def test_to_int16_rounds_to_nearest_metre():
    a = np.array([1.4, 1.6, -2.5, 8999.9], dtype=np.float32)
    i16 = normalize.to_int16(a)
    assert i16.dtype == np.int16
    assert i16.tolist() == [1, 2, -2, 9000]  # banker's rounding on -2.5 -> -2


def _rugged_window(fill=1000.0, noise=50.0, seed=0):
    rng = np.random.default_rng(seed)
    return (fill + rng.normal(0, noise, size=(config.TILE_PX, config.TILE_PX))).astype(
        np.float32
    )


def test_prepare_tile_happy_path():
    win = _rugged_window()
    tile, reason = normalize.prepare_tile(win)
    assert reason == "ok"
    assert tile is not None
    assert tile.raw_i16.shape == (config.TILE_PX, config.TILE_PX)
    assert tile.raw_i16.dtype == np.int16
    assert tile.land_frac == pytest.approx(1.0)
    assert tile.std > config.DEGENERATE_STD_EPS_M


def test_prepare_tile_drops_degenerate_flat():
    # near-constant terrain (std < 3 m) -> dropped so int16 quantisation can't band it
    win = np.full((config.TILE_PX, config.TILE_PX), 3040.0, dtype=np.float32)
    win += np.random.default_rng(0).normal(0, 0.5, size=win.shape).astype(np.float32)
    tile, reason = normalize.prepare_tile(win)
    assert tile is None
    assert reason == "degenerate"


def test_prepare_tile_drops_mostly_ocean():
    win = np.zeros((config.TILE_PX, config.TILE_PX), dtype=np.float32)
    win[:40] = 100.0  # ~18% land -> >75% ocean
    tile, reason = normalize.prepare_tile(win)
    assert tile is None
    assert reason == "ocean"


def test_prepare_tile_keeps_coastline():
    win = np.zeros((config.TILE_PX, config.TILE_PX), dtype=np.float32)
    win[:120] = _rugged_window()[:120]  # ~54% land coastline
    tile, reason = normalize.prepare_tile(win)
    assert reason == "ok"
    assert tile is not None
    assert 0.4 < tile.land_frac < 0.6


def test_prepare_tile_stats_are_land_only_on_coastline():
    win = np.zeros((config.TILE_PX, config.TILE_PX), dtype=np.float32)
    win[:120] = 800.0 + np.random.default_rng(1).normal(
        0, 40, size=(120, config.TILE_PX)
    ).astype(np.float32)
    tile, reason = normalize.prepare_tile(win)
    assert reason == "ok"
    # mean reflects only the ~800 m land band, not the sea of zeros
    assert tile.mean == pytest.approx(800.0, abs=15.0)
