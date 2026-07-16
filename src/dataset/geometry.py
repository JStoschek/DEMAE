"""Pure tiling geometry — no I/O.

Everything here is a deterministic function of a COG's name and pixel dimensions:
parsing the Copernicus key into its 1-degree origin, mapping a latitude to its
resolution band, binning a coordinate into a 10-degree geo-cell, laying the
non-overlapping window grid (ADR-0001 §2), and turning a window's pixel origin
into its centre lon/lat. Keeping this free of rasterio makes the whole sampling
recipe unit-testable without touching S3.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final

from . import config

# Copernicus GLO-30 key: ..._10_{N|S}<dd>_00_{E|W}<ddd>_00_DEM (recon §1).
_KEY_RE = re.compile(
    r"Copernicus_DSM_COG_10_([NS])(\d{2})_00_([EW])(\d{3})_00_DEM"
)

# Mean metres per degree of latitude on a spherical earth.
_M_PER_DEG_LAT: Final = 111_320.0

#: Ground metres per pixel, ``(dy, dx)`` — north-south first, then east-west. The
#: pair travels together everywhere (:func:`pixel_spacing_m` makes it,
#: :mod:`dataset.metrics` consumes it) and the two are never interchangeable:
#: Copernicus's columns are up to ~1.55x wider on the ground than its rows.
PixelSize = tuple[float, float]


@dataclass(frozen=True)
class CogRef:
    """A source COG identified by its 1-degree south-west (origin) corner."""

    stem: str  # e.g. "Copernicus_DSM_COG_10_N45_00_E006_00_DEM"
    origin_lat: int  # south edge, signed (N positive, S negative)
    origin_lon: int  # west edge, signed (E positive, W negative)


def parse_cog_key(key: str) -> CogRef:
    """Parse a bucket key or prefix into a :class:`CogRef`.

    Accepts a bare prefix, a ``.tif`` object key, or a ``prefix/`` — anything that
    contains the Copernicus stem. The name encodes the tile's south-west corner
    (SRTM convention): ``N45..E006`` -> origin (lat=45, lon=6), spanning
    [45, 46] x [6, 7]; ``S01..W123`` -> origin (lat=-1, lon=-123).
    """
    m = _KEY_RE.search(key)
    if m is None:
        raise ValueError(f"not a Copernicus GLO-30 key: {key!r}")
    ns, lat, ew, lon = m.groups()
    origin_lat = int(lat) * (1 if ns == "N" else -1)
    origin_lon = int(lon) * (1 if ew == "E" else -1)
    return CogRef(stem=m.group(0), origin_lat=origin_lat, origin_lon=origin_lon)


def lat_band(lat: float) -> str:
    """Resolution band label for a latitude (poleward-symmetric, recon §6).

    Bands come from :data:`config.LAT_BAND_EDGES`; the label is ``"<lo>-<hi>"``.
    Each band is closed on its lower edge (``50.0`` -> ``"50-60"``); ``90`` is
    inclusive at the top.
    """
    a = abs(lat)
    edges = config.LAT_BAND_EDGES
    for lo, hi in zip(edges, edges[1:]):
        if lo <= a < hi:
            return f"{lo}-{hi}"
    # a == top edge (90) or numerical overshoot -> the last band
    return f"{edges[-2]}-{edges[-1]}"


def geocell_id(lon: float, lat: float) -> str:
    """Stable id of the 10x10-degree cell containing (lon, lat), ADR-0001 §5.

    Identified by the cell's south-west corner (floor to a multiple of
    :data:`config.GEOCELL_SIZE_DEG`), formatted signed and zero-padded so ids
    sort and round-trip: ``lon 6.5, lat 45.2 -> "+000_+040"``.
    """
    size = config.GEOCELL_SIZE_DEG
    lon_cell = _floor_to(lon, size)
    lat_cell = _floor_to(lat, size)
    return f"{lon_cell:+04d}_{lat_cell:+04d}"


def _floor_to(value: float, step: int) -> int:
    return int(math.floor(value / step) * step)


def axis_starts(dim: int, tile: int = config.TILE_PX) -> list[int]:
    """Window start offsets along one axis: ``floor(dim/tile)`` non-overlapping
    windows with the last flushed to the edge (ADR-0001 §2).

    For a divisible axis this is a clean stride; otherwise the trailing window is
    pulled back to ``dim - tile`` so there is no coverage gap (it may overlap its
    predecessor). All Copernicus band widths are >= tile, so at least one window
    always fits.
    """
    n = dim // tile
    if n <= 0:
        return []
    starts = [i * tile for i in range(n)]
    starts[-1] = dim - tile  # flush the last window to the far edge
    return starts


def grid_windows(width: int, height: int, tile: int = config.TILE_PX) -> list[tuple[int, int]]:
    """Row-major list of ``(win_row0, win_col0)`` origins tiling a COG."""
    rows = axis_starts(height, tile)
    cols = axis_starts(width, tile)
    return [(r, c) for r in rows for c in cols]


def window_center_lonlat(
    *,
    origin_lon: int,
    origin_lat: int,
    width: int,
    height: int,
    win_row0: int,
    win_col0: int,
    tile: int = config.TILE_PX,
) -> tuple[float, float]:
    """Centre (lon, lat) of a pixel window in a 1-degree EPSG:4326 COG.

    The COG spans exactly 1 degree in each direction; rows are always sampled
    N->S and columns W->E, so pixel resolution is ``1/height`` (lat) and
    ``1/width`` (lon). ``width`` is band-dependent (columns shrink poleward),
    which is why it must be passed in rather than assumed 3600.
    """
    north_lat = origin_lat + 1.0
    west_lon = origin_lon
    lon = west_lon + (win_col0 + tile / 2) * (1.0 / width)
    lat = north_lat - (win_row0 + tile / 2) * (1.0 / height)
    return lon, lat


def pixel_spacing_m(
    lat: float, *, rows_per_degree: int, cols_per_degree: int
) -> tuple[float, float]:
    """Ground metres per pixel ``(dy, dx)`` for a raster sampling a 1-degree cell.

    This is what :mod:`dataset.metrics` wants: the metrics take a pixel size and
    refuse to know about latitude, so somebody has to do the conversion, and this
    is the only place that does. Pass the dimensions of the raster that spans the
    degree — the full COG, or its x8 overview — and the spacing holds for any
    window cut from it.

    ``dy`` is latitude-independent; ``dx`` shrinks with ``cos(lat)``, which is why
    the two must be applied to their own axes rather than averaged into one
    resolution (ADR-0001 §1). Copernicus's poleward column coarsening cancels most
    of that shrink, but not all of it: the anisotropy peaks around 1.55:1 near
    50 deg, just inside the 3600-column band.

    A spherical earth is close enough here. The metrics feed a *ranking* — the
    stratifier's quantile cuts — and the ~0.3% ellipsoidal error is far below the
    resolution at which a rank flips.
    """
    dy = _M_PER_DEG_LAT / rows_per_degree
    dx = _M_PER_DEG_LAT * math.cos(math.radians(lat)) / cols_per_degree
    return dy, dx
