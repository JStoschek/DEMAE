"""Per-window terrain metrics — pure numpy, no I/O, no global state.

The metric library behind the plan's A2. One function per metric, each with the
same binding contract:

    metric(z, pixel_size) -> float

where ``z`` is a 2-D elevation array in metres and ``pixel_size`` is the
``(dy, dx)`` ground metres per pixel of *that* array. Passing the pixel size in
(rather than a latitude, or a lookup into the manifest) is what lets an identical
call score a full-resolution window and its x8 overview — the comparison A3's
calibration is built on. Metrics that are scale-free still take the pair, so a
caller can loop over :data:`METRICS` without special cases.

Pixels equal to Copernicus's exact ocean fill (``0.0``) are excluded from every
statistic (ADR-0001 §3). A window with no land left is not an error: the metric
returns NaN, and so does any ratio whose denominator degenerates. The caller
decides what an undefined window means.

The gradient mechanics come from ``prototypes/terrain_stratify_proto.py``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import config
from .geometry import PixelSize


# --- Land mask (ADR-0001 §3) --------------------------------------------------


def land_mask(z: np.ndarray) -> np.ndarray:
    """Boolean mask of the pixels that count: everything but the exact ocean fill.

    Exactness is the whole trick — genuine land is essentially never exactly
    ``0.0``, and sub-sea-level land is negative (ADR-0001 §3).
    """
    return np.asarray(z) != config.OCEAN_FILL_VALUE


def _land_values(z: np.ndarray) -> np.ndarray:
    """The land pixels of ``z`` as a flat float64 array (possibly empty)."""
    a = np.asarray(z, dtype=np.float64)
    return a[land_mask(a)]


# --- Local relief -------------------------------------------------------------


def local_relief(z: np.ndarray, pixel_size: PixelSize) -> float:
    """Land elevation range, ``max - min``, in metres.

    ``pixel_size`` is unused — relief is scale-free — but is part of the uniform
    metric contract.
    """
    del pixel_size
    land = _land_values(z)
    if land.size == 0:
        return float("nan")
    return float(land.max() - land.min())


# --- Slope --------------------------------------------------------------------


def slope_deg(z: np.ndarray, pixel_size: PixelSize) -> np.ndarray:
    """Per-pixel slope in degrees: the gradient magnitude, arctan'd. NaN where
    the slope is not defined on land.

    ``dy`` and ``dx`` are applied to their own axes rather than assuming square
    ground pixels, because Copernicus's columns are up to ~1.55x wider than its
    rows near 50 deg (ADR-0001 §1). Edge pixels get numpy's one-sided difference,
    which is exact for a plane and the same first-order approximation elsewhere.

    The mask goes on *before* the gradient, not after. Ocean is a painted fill,
    not a measurement, so the step from a coastal plateau down to it is a
    digitization artifact rather than terrain — and a one-pixel step is steeper
    read at 30 m than at 240 m, so letting it through would inject
    resolution-dependent noise into exactly the coastal windows ADR-0001 §3 sets
    out to keep, and A3's calibration would read it as drift. NaN-ing the fill
    first costs the one land pixel either side of the shoreline, whose gradient
    cannot be computed without it; that is the honest price of not inventing a
    cliff.
    """
    dy, dx = pixel_size
    a = np.asarray(z, dtype=np.float64)
    land = np.where(land_mask(a), a, np.nan)
    gy, gx = np.gradient(land, dy, dx)
    return np.degrees(np.arctan(np.hypot(gy, gx)))


def _defined_slopes(z: np.ndarray, pixel_size: PixelSize) -> np.ndarray:
    """The land slopes of ``z`` that are actually defined (see :func:`slope_deg`)."""
    slopes = slope_deg(z, pixel_size)
    return slopes[np.isfinite(slopes)]


def slope_mean(z: np.ndarray, pixel_size: PixelSize) -> float:
    """Mean slope in degrees over the land pixels."""
    slopes = _defined_slopes(z, pixel_size)
    if slopes.size == 0:
        return float("nan")
    return float(slopes.mean())


def slope_std(z: np.ndarray, pixel_size: PixelSize) -> float:
    """Standard deviation of slope in degrees over the land pixels.

    Separates a rough surface from a merely tilted one: any plane scores zero.
    """
    slopes = _defined_slopes(z, pixel_size)
    if slopes.size == 0:
        return float("nan")
    return float(slopes.std())


# --- Elevation-relief ratio ---------------------------------------------------


def elevation_relief_ratio(z: np.ndarray, pixel_size: PixelSize) -> float:
    """Pike & Wilson's elevation-relief ratio, ``(mean - min) / (max - min)``.

    Identical to the hypsometric integral, and the cheap discriminator between
    terrain that is mostly-high-with-incisions (canyon country, ratio near 1) and
    mostly-low-with-peaks (ratio near 0); any plane sits at 0.5.

    Returns NaN where ``max == min``, since a surface with no relief has no
    hypsometry to integrate. ``pixel_size`` is unused — the ratio is scale-free
    and dimensionless — but is part of the uniform metric contract.
    """
    del pixel_size
    land = _land_values(z)
    if land.size == 0:
        return float("nan")
    lo, hi = land.min(), land.max()
    if hi == lo:
        return float("nan")
    return float((land.mean() - lo) / (hi - lo))


# --- Mean elevation -----------------------------------------------------------


def mean_elevation(z: np.ndarray, pixel_size: PixelSize) -> float:
    """Mean land elevation in metres.

    ``pixel_size`` is unused — an average of elevations does not care how far
    apart they are — but is part of the uniform metric contract.
    """
    del pixel_size
    land = _land_values(z)
    if land.size == 0:
        return float("nan")
    return float(land.mean())


# --- Registry -----------------------------------------------------------------

#: The metrics with exact analytic oracles, keyed by manifest column name.
METRICS: dict[str, Callable[[np.ndarray, PixelSize], float]] = {
    "local_relief": local_relief,
    "slope_mean": slope_mean,
    "slope_std": slope_std,
    "elevation_relief_ratio": elevation_relief_ratio,
    "mean_elevation": mean_elevation,
}
