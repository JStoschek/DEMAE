"""A fabricated Copernicus-shaped bucket for the zero-network sample corpus.

The Phase-0 debug corpus (~50k tiles) is built from the live ``eu-central-1``
bucket and is far too large — and too slow to fetch transatlantic — to commit or
to rebuild in CI. This module fabricates a small, deterministic, terrain-diverse
stand-in so a **fresh reader can reproduce a sample end-to-end with zero network**
(``python -m dataset.build --source synthetic``) and the visualization notebook
has something real to render. It is a *sample*, never the shipped corpus.

The COGs are placed at real-world origins spanning continents and latitude bands
(so the geo round-robin of ADR-0001 §5 and the poleward resolution bands of
ADR-0002 are exercised) with per-COG ruggedness spanning the §4 TRI quintile
strata and a few ocean-carrying cells (so the §3 ocean screen and §6 land mask
fire). Everything derives from a seed, so two readers built with the same seed are
byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tiling import SyntheticCogReader


@dataclass(frozen=True)
class _Site:
    """One synthetic source cell — a Copernicus origin plus its terrain knobs."""

    ns: str  # "N" | "S"
    lat: int  # origin latitude, unsigned degrees
    ew: str  # "E" | "W"
    lon: int  # origin longitude, unsigned degrees
    ruggedness: float  # metres of fine detail (drives the TRI stratum)
    ocean_frac: float  # fraction of top rows set to the exact-zero ocean fill


# Real-world origins chosen to spread 10-degree geocells and hit the
# 0-50/50-60/60-70/70-80 bands, with a range of ruggedness so the quintile TRI
# strata are all populated.
_SITES: list[_Site] = [
    _Site("N", 45, "E", 6, 260.0, 0.0),    # Alps — high relief
    _Site("N", 28, "E", 86, 380.0, 0.0),   # Himalaya — extreme relief
    _Site("S", 23, "W", 67, 300.0, 0.0),   # Andes — high relief
    _Site("N", 35, "W", 118, 150.0, 0.0),  # California ranges — moderate
    _Site("S", 1, "E", 36, 90.0, 0.0),     # Kenya highlands — moderate
    _Site("N", 10, "E", 77, 70.0, 0.0),    # South India — low-moderate
    _Site("N", 52, "E", 5, 18.0, 0.35),    # Netherlands — flat + coast/sea
    _Site("N", 55, "E", 10, 40.0, 0.30),   # Denmark — low relief + sea
    _Site("N", 60, "E", 24, 45.0, 0.0),    # Finland — 60-70 band
    _Site("N", 64, "W", 21, 200.0, 0.20),  # Iceland — 60-70 band + coast
    _Site("N", 72, "W", 42, 120.0, 0.25),  # Greenland — 70-80 band + sea
    _Site("S", 34, "E", 18, 130.0, 0.25),  # Cape coast — 0-50 band + sea
]

# 5 windows per axis => 25 candidate windows/COG => ~300 candidates before the
# ocean screen. Enough to fill several hundred sample tiles across all strata.
_WINDOWS_PER_AXIS = 5


def _stem(ns: str, lat: int, ew: str, lon: int) -> str:
    return f"Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM"


def _terrain(
    size: int, ruggedness: float, ocean_frac: float, seed: int
) -> np.ndarray:
    """A plausible DEM: smooth low-frequency base + ruggedness-scaled detail.

    The base is a coarse random field bilinearly upsampled (so tiles look like
    terrain, not white noise), the detail is fine Gaussian noise scaled by the
    per-COG ruggedness (so the TRI stratification has something to separate), and
    an optional top ocean band is set to the exact-zero fill.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.normal(1200.0, 500.0, size=(9, 9)).astype(np.float32)
    # Bilinear upsample the coarse field to full size via 1-D interpolation.
    idx = np.linspace(0, coarse.shape[0] - 1, size)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, coarse.shape[0] - 1)
    w = (idx - lo).astype(np.float32)
    rows = coarse[lo] * (1 - w)[:, None] + coarse[hi] * w[:, None]
    base = rows[:, lo] * (1 - w)[None, :] + rows[:, hi] * w[None, :]
    detail = rng.normal(0.0, ruggedness, size=(size, size)).astype(np.float32)
    arr = np.clip(base + detail, 1.0, None).astype(np.float32)  # keep land != 0
    if ocean_frac > 0.0:
        arr[: int(size * ocean_frac)] = 0.0
    return arr


def synthetic_reader(seed: int = 0, *, windows_per_axis: int = _WINDOWS_PER_AXIS):
    """Build the deterministic synthetic sample bucket as a ``SyntheticCogReader``."""
    from . import config

    size = windows_per_axis * config.TILE_PX
    tiles: dict[str, np.ndarray] = {}
    for i, s in enumerate(_SITES):
        stem = _stem(s.ns, s.lat, s.ew, s.lon)
        tiles[stem] = _terrain(size, s.ruggedness, s.ocean_frac, seed=seed * 1000 + i)
    return SyntheticCogReader(tiles)
