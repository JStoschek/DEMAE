"""Synthetic terrain surfaces whose metric values are known in closed form.

These are the oracles the metric library is tested against (plan A2): each
generator is parameterized so the expected value of every metric follows from the
parameters by hand, not from re-running the metric. A ramp's relief is its range;
an inclined plane's slope is its tilt at any aspect, and its hypsometric integral
is 0.5 by symmetry; a plateau with a pit of area fraction ``f`` has an
elevation-relief ratio of exactly ``1 - f``.

This module makes bare arrays and knows nothing about COGs — an analytic surface,
not a Copernicus-shaped fixture.

Every surface avoids the exact value ``0.0``, which the pipeline reserves for
Copernicus's ocean fill (ADR-0001 §3) and every metric masks out. Callers that
want ocean must paint it in themselves.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import PixelSize

Shape = tuple[int, int]

_DEFAULT_SHAPE: Shape = (64, 64)


def ramp(
    shape: Shape = _DEFAULT_SHAPE,
    *,
    low: float = 100.0,
    high: float = 200.0,
    axis: int = 0,
) -> np.ndarray:
    """A linear ramp from ``low`` to ``high`` across ``axis``.

    Oracles: local relief is exactly ``high - low``; the mean is the midpoint
    ``(low + high) / 2``; the elevation-relief ratio is exactly 0.5.

    ``low`` defaults clear of ``0.0`` rather than to it: a ramp starting at sea
    level would have its floor row eaten by the land mask, and every oracle above
    would quietly be a little wrong.
    """
    line = np.linspace(low, high, shape[axis], dtype=np.float64)
    return _spread_along_axis(line, shape, axis)


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
    eastward. ``base`` is the elevation at the north-west corner pixel; a steep
    enough tilt over a big enough grid will still run the far corner below sea
    level, which is harmless — only exactly ``0.0`` reads as ocean, and a plane
    hits that on at most one contour line.

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
    return np.broadcast_to(z, shape).astype(np.float64, copy=True)


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
    return _spread_along_axis(wave, shape, axis)


def ridge(
    shape: Shape = _DEFAULT_SHAPE,
    *,
    pixel_size: PixelSize,
    flank_slope_deg: float = 20.0,
    crest_height: float = 1000.0,
    axis: int = 0,
) -> np.ndarray:
    """A tent: a straight crest at ``crest_height``, flanks falling at
    ``flank_slope_deg`` to either side across ``axis``.

    The convex counterpart to :func:`valley`, and the surface the geomorphon
    classes were named for. Oracles: the crest cell sees terrain falling in the
    six directions that leave the crest line and level ground along the other two,
    which is the canonical geomorphon *ridge*; every flank cell is a plane tilted
    at ``flank_slope_deg``. With an even ``shape[axis]`` the crest falls between
    two cells and both sit half a pixel below the nominal height, so pass an odd
    length when the crest itself is what's under test.

    ``crest_height`` is the top of the surface and the flanks fall away from it, so
    a steep enough tilt over a wide enough grid runs them down through ``0.0`` and
    the mask eats a contour line of the terrain. The defaults leave the flanks
    clear; a shallower ``crest_height`` needs checking against them.
    """
    return _tent(shape, pixel_size, flank_slope_deg, crest_height, axis, falls_away=True)


def valley(
    shape: Shape = _DEFAULT_SHAPE,
    *,
    pixel_size: PixelSize,
    flank_slope_deg: float = 20.0,
    floor_height: float = 1000.0,
    axis: int = 0,
) -> np.ndarray:
    """A trough: a straight floor at ``floor_height``, walls rising at
    ``flank_slope_deg`` to either side across ``axis``.

    :func:`ridge` upside down, and the concave half of the geomorphon oracle: the
    floor cell sees terrain rising in six directions, the canonical *valley*.

    ``floor_height`` is the bottom of the surface and the walls only climb from it,
    so nothing here can reach ``0.0`` unless the floor is put there.
    """
    return _tent(shape, pixel_size, flank_slope_deg, floor_height, axis, falls_away=False)


def _tent(
    shape: Shape,
    pixel_size: PixelSize,
    flank_slope_deg: float,
    height: float,
    axis: int,
    *,
    falls_away: bool,
) -> np.ndarray:
    """A vee across ``axis``, hinged at ``height`` on the centre line: pointing up
    into a ridge when it ``falls_away`` from the hinge, down into a valley when not.
    """
    spacing = pixel_size[axis]
    n = shape[axis]
    centre = (n - 1) / 2.0
    distance_m = np.abs(np.arange(n, dtype=np.float64) - centre) * spacing
    rise = -1.0 if falls_away else 1.0
    line = height + rise * math.tan(math.radians(flank_slope_deg)) * distance_m
    return _spread_along_axis(line, shape, axis)


def basin(
    shape: Shape = _DEFAULT_SHAPE,
    *,
    pixel_size: PixelSize,
    depth: float = 100.0,
    radius_m: float = 960.0,
    floor: float = 1000.0,
) -> np.ndarray:
    """A paraboloid bowl centred on the array: ``floor + depth * (r / radius_m)**2``,
    where ``r`` is the ground distance in metres from the centre.

    A negative ``depth`` turns the bowl over into a dome — the same paraboloid, so
    the same oracles with the signs flipped, which is why there is no separate
    ``dome``. The centre cell is a geomorphon *pit* (a dome's is a *peak*).

    The quadratic is what makes this the TPI oracle: the mean of ``r**2`` over any
    kernel symmetric about a cell is that cell's ``r**2`` plus the kernel's own
    mean squared radius, so cell-minus-neighbourhood is the same negative constant
    at every interior cell, ``-depth / radius_m**2`` times that kernel constant. A
    bowl reads below its surroundings everywhere, however the window is placed.

    ``r`` is measured in metres through ``pixel_size``, so the contours are round
    on the ground rather than round in pixels.

    A dome deep enough to bring its skirt down through ``0.0`` loses that contour
    to the mask; the defaults keep a 64-cell window's corners well clear of it.
    """
    if radius_m <= 0:
        raise ValueError(f"radius_m must be positive: {radius_m!r}")
    dy, dx = pixel_size
    rows = (np.arange(shape[0], dtype=np.float64) - (shape[0] - 1) / 2.0).reshape(-1, 1) * dy
    cols = (np.arange(shape[1], dtype=np.float64) - (shape[1] - 1) / 2.0).reshape(1, -1) * dx
    r2 = rows**2 + cols**2
    return floor + depth * r2 / radius_m**2


def _spread_along_axis(line: np.ndarray, shape: Shape, axis: int) -> np.ndarray:
    """Repeat a 1-D profile across the other axis into a writable ``shape`` grid.

    Writable matters: callers paint ocean into these surfaces to build the coastal
    cases, and ``np.broadcast_to`` alone hands back a read-only view.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1: {axis!r}")
    profile = line.reshape((-1, 1) if axis == 0 else (1, -1))
    return np.broadcast_to(profile, shape).astype(np.float64, copy=True)


def constant(shape: Shape = _DEFAULT_SHAPE, *, value: float = 250.0) -> np.ndarray:
    """A constant surface — the degenerate guard (``max == min``).

    Pass ``value=0.0`` for the other degeneracy: a window that is entirely ocean
    fill, so nothing survives the land mask.
    """
    return np.full(shape, float(value), dtype=np.float64)
