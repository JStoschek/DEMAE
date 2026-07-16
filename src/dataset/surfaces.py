"""Synthetic terrain surfaces whose metric values are known in closed form.

These are the oracles the metric library is tested against (plan A2): each
generator is parameterized so the expected value of every metric follows from the
parameters by hand, not from re-running the metric. A ramp's relief is its range;
an inclined plane's slope is its tilt at any aspect, and its hypsometric integral
is 0.5 by symmetry; a plateau with a pit of area fraction ``f`` has an
elevation-relief ratio of exactly ``1 - f``.

Not to be confused with :mod:`dataset.synthetic`, which fabricates a
Copernicus-shaped S3 bucket for zero-network pipeline tests. This module makes
bare arrays and knows nothing about COGs.

Every surface avoids the exact value ``0.0``, which the pipeline reserves for
Copernicus's ocean fill (ADR-0001 §3) and every metric masks out. Callers that
want ocean must paint it in themselves.
"""

from __future__ import annotations

import math

import numpy as np

Shape = tuple[int, int]
PixelSize = tuple[float, float]  # (dy, dx) ground metres per pixel

_DEFAULT_SHAPE: Shape = (64, 64)


def ramp(
    shape: Shape = _DEFAULT_SHAPE,
    *,
    low: float = 0.0,
    high: float = 100.0,
    axis: int = 0,
) -> np.ndarray:
    """A linear ramp from ``low`` to ``high`` across ``axis``.

    Oracles: local relief is exactly ``high - low``; the mean is the midpoint
    ``(low + high) / 2``; the elevation-relief ratio is exactly 0.5.
    """
    n = shape[axis]
    line = np.linspace(low, high, n, dtype=np.float64)
    z = np.broadcast_to(
        line.reshape((-1, 1) if axis == 0 else (1, -1)), shape
    )
    return np.ascontiguousarray(z, dtype=np.float64)


def inclined_plane(
    shape: Shape = _DEFAULT_SHAPE,
    *,
    pixel_size: PixelSize,
    slope_deg: float,
    aspect_deg: float = 0.0,
    base: float = 1000.0,
) -> np.ndarray:
    """A plane tilted ``slope_deg`` from horizontal, falling towards ``aspect_deg``.

    Aspect is the downslope azimuth in degrees clockwise from north, the GIS
    convention; the raster is north-up, so rows increase southward and columns
    eastward. ``base`` is the elevation at the north-west corner pixel and is
    lifted well clear of ``0.0`` so no pixel is mistaken for ocean.

    Oracles: the gradient magnitude is exactly ``tan(slope_deg)`` everywhere, so
    slope mean is the tilt and slope std is zero, whatever the aspect and whatever
    the pixel anisotropy. The hypsometric integral of any plane is 0.5.
    """
    dy, dx = pixel_size
    a = math.radians(aspect_deg)
    s = math.tan(math.radians(slope_deg))
    rows = np.arange(shape[0], dtype=np.float64).reshape(-1, 1)
    cols = np.arange(shape[1], dtype=np.float64).reshape(1, -1)
    # Downslope unit vector is (east, north) = (sin a, cos a); north = -row * dy.
    z = base + s * (rows * dy * math.cos(a) - cols * dx * math.sin(a))
    return np.ascontiguousarray(np.broadcast_to(z, shape), dtype=np.float64)


def plateau(shape: Shape = _DEFAULT_SHAPE, *, height: float = 250.0) -> np.ndarray:
    """A flat plateau at ``height``.

    The zero-slope, zero-relief baseline, and the base that :func:`plateau_with_pit`
    carves into. The same surface as :func:`constant` under a different name: the
    two are kept apart because they anchor different oracles — the plateau is a
    *terrain* whose slope and relief are zero, while the constant surface is the
    *degenerate guard* where ``max == min`` leaves a ratio undefined.
    """
    return constant(shape, value=height)


def plateau_with_pit(
    shape: Shape = _DEFAULT_SHAPE,
    *,
    height: float = 250.0,
    pit_depth: float = 100.0,
    pit_frac: float = 0.0625,
) -> np.ndarray:
    """A flat plateau at ``height`` with a square pit ``pit_depth`` metres deep
    covering ``pit_frac`` of the area, centred.

    The high-hypsometric-integral counterpart to the plane: nearly every pixel
    sits at the maximum, so the mean is dragged only ``pit_depth * f`` below it
    and the elevation-relief ratio is exactly ``1 - f``.

    ``f`` is the *achieved* fraction, ``side**2 / (rows * cols)`` for an integer
    ``side``, which equals ``pit_frac`` only when that lands on a whole number of
    pixels — pick a ``pit_frac`` that squares cleanly (0.0625 of a 64x64 grid is a
    16x16 pit) when an exact oracle matters. :func:`pit_fraction` reports it.
    """
    rows, cols = shape
    side = _pit_side(shape, pit_frac)
    z = plateau(shape, height=height)
    r0 = (rows - side) // 2
    c0 = (cols - side) // 2
    z[r0 : r0 + side, c0 : c0 + side] = height - pit_depth
    return z


def pit_fraction(shape: Shape = _DEFAULT_SHAPE, *, pit_frac: float = 0.0625) -> float:
    """The area fraction :func:`plateau_with_pit` actually achieves for ``pit_frac``.

    Exact when ``pit_frac`` squares to a whole number of pixels; rounded down to
    the nearest whole pit side otherwise.
    """
    side = _pit_side(shape, pit_frac)
    return side * side / (shape[0] * shape[1])


def _pit_side(shape: Shape, pit_frac: float) -> int:
    if not 0.0 <= pit_frac <= 1.0:
        raise ValueError(f"pit_frac must be in [0, 1]: {pit_frac!r}")
    side = int(round(math.sqrt(pit_frac * shape[0] * shape[1])))
    return min(side, min(shape))


def sinusoid(
    shape: Shape = _DEFAULT_SHAPE,
    *,
    pixel_size: PixelSize,
    wavelength_m: float = 480.0,
    amplitude: float = 50.0,
    base: float = 1000.0,
    axis: int = 1,
) -> np.ndarray:
    """Corrugations of a given *physical* wavelength across ``axis``.

    The wavelength is in ground metres, converted to pixels through ``pixel_size``,
    so this is the surface for a frequency sweep — the same terrain read at x8
    overview is the same wave sampled 8x coarser.

    Oracles: over a whole number of periods the mean is ``base`` and the
    hypsometric integral is 0.5 by symmetry; relief is ``2 * amplitude`` once a
    crest and a trough are actually sampled, which is exact when the wavelength is
    a multiple of four pixels. The steepest slope of the continuous wave is
    ``arctan(2 * pi * amplitude / wavelength_m)``; a sampled gradient approaches it
    from below as the sampling fines up.
    """
    if wavelength_m <= 0:
        raise ValueError(f"wavelength_m must be positive: {wavelength_m!r}")
    spacing = pixel_size[axis]
    n = shape[axis]
    distance_m = np.arange(n, dtype=np.float64) * spacing
    wave = base + amplitude * np.sin(2.0 * math.pi * distance_m / wavelength_m)
    z = np.broadcast_to(wave.reshape((-1, 1) if axis == 0 else (1, -1)), shape)
    return np.ascontiguousarray(z, dtype=np.float64)


def constant(shape: Shape = _DEFAULT_SHAPE, *, value: float = 250.0) -> np.ndarray:
    """A constant surface — the degenerate guard (``max == min``).

    Pass ``value=0.0`` for the other degeneracy: a window that is entirely ocean
    fill, so nothing survives the land mask.
    """
    return np.full(shape, float(value), dtype=np.float64)
