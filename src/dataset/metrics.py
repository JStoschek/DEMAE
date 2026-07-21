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

The mask goes on *before* any neighbourhood operation — gradient, median, sight
line, average — and never after. The fill is painted, not measured, so the step
down to it is an artifact of the file format; worse, it is a resolution-dependent
one, which lands as fake drift in A3's calibration for exactly the coastal windows
ADR-0001 §3 sets out to keep. Each metric's docstring says what its own ordering
costs.

The gradient, median and sight-line mechanics come from
``prototypes/terrain_stratify_proto.py`` — but only the mechanics. Two of the
prototype's decisions do not survive contact with a test and are documented where
they are overturned: its geomorphon lookup is read with the axes transposed (see
:data:`_LANDFORM_TABLE`), and its per-cell TPI relies on tile context a window does
not have (see :func:`tpi_proxy`).
"""

from __future__ import annotations

import math
from typing import Callable, Final

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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


def _masked_land(z: np.ndarray) -> np.ndarray:
    """``z`` as float64 with the ocean fill blanked to NaN, shape preserved.

    What :func:`_land_values` is for the statistics, this is for the neighbourhood
    operations: every one of them starts here, so none of them can quietly start
    reading the fill as terrain (ADR-0001 §3).
    """
    a = np.asarray(z, dtype=np.float64)
    return np.where(land_mask(a), a, np.nan)


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
    gy, gx = _masked_gradient(z, pixel_size)
    return np.degrees(np.arctan(np.hypot(gy, gx)))


def _masked_gradient(z: np.ndarray, pixel_size: PixelSize) -> tuple[np.ndarray, np.ndarray]:
    """``(dz/dy, dz/dx)`` in metres per metre, NaN wherever the ocean fill was.

    The mask goes on before the difference, not after — see :func:`slope_deg` for
    why that ordering is the whole point. Both gradient metrics take their rise
    from here, so neither can drift from the rule.
    """
    dy, dx = pixel_size
    return np.gradient(_masked_land(z), dy, dx)


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


# --- Iwahashi-Pike surface texture --------------------------------------------

#: Median-filter kernel for :func:`texture`, in cells (Iwahashi & Pike's 3x3).
TEXTURE_MEDIAN_PX: Final = 3
#: How far a cell must sit off its local median to count as a pit or a peak, in
#: metres. Iwahashi & Pike's 1 m: below it, terrain is under the DEM's noise floor.
TEXTURE_THRESHOLD_M: Final = 1.0


def texture(z: np.ndarray, pixel_size: PixelSize) -> float:
    """Iwahashi & Pike's surface texture: the fraction of land cells that are a pit
    or a peak, in ``[0, 1]``.

    A cell is a pit or a peak where it departs from the median of its 3x3
    neighbourhood by more than :data:`TEXTURE_THRESHOLD_M`. Fine, busy terrain
    marks a large fraction of its cells; smooth terrain marks almost none, however
    steep it is — the median of a monotone neighbourhood is the cell in the middle
    of it, so a plane has no texture at any tilt. That indifference to steepness is
    the point: texture is the axis slope cannot see.

    ``pixel_size`` is unused. Texture counts cells, not metres, so the same array
    scores the same at any pixel size — but the *terrain* it reads is coarser at
    ×8, where eight cells of fine corrugation become one. Whether that survives the
    overview is Gate A3's question, not a reason to rescale the kernel here.

    The published algorithm counts this density in a moving ~10-cell-radius circle
    and emits a map. A window's metric is one number, and the mean of that map is
    the same density up to how the circle weights the edges, so the counting window
    collapses into the window itself and only the marking survives.

    The median goes on the masked array, per ADR-0001 §3: a coastal cell takes the
    median of the land it can see. Unlike :func:`slope_deg`, which needs the one
    specific neighbour the fill took away, a median over a partial neighbourhood is
    still a median — so texture keeps the shoreline cells that slope has to drop.
    """
    del pixel_size
    a = np.asarray(z, dtype=np.float64)
    land = land_mask(a)
    if not land.any():
        return float("nan")
    departure = _median_filter(_masked_land(a), TEXTURE_MEDIAN_PX) - a
    marked = np.abs(departure[land]) > TEXTURE_THRESHOLD_M
    return float(marked.mean())


def _median_filter(a: np.ndarray, size: int) -> np.ndarray:
    """Median of each cell's ``size``x``size`` neighbourhood, ignoring NaN.

    Edge-replicated at the array border, the median filter's usual convention and
    the one that keeps a plane's median on the plane: reflecting or zero-padding
    would bend the surface at the rim and mark the whole border as texture. Cells
    that are themselves NaN come back NaN; the caller drops them anyway.

    **A window with no ocean fill skips ``nanmedian`` entirely** (#38). ``nanmedian``
    pays for a masked sort on every cell; over an odd, all-finite neighbourhood the
    median *is* the middle element by partition, which is the element ``median``
    returns — so on a NaN-free window the branch is bit-exact and ~3x faster. Most
    windows are NaN-free (the fill only appears at the coast), and this filter is
    ``texture``'s whole cost and the pass's #2 hotspot after the geomorphon hoist.
    ``tests/test_metrics.py`` holds both implementations against each other.
    """
    pad = size // 2
    padded = np.pad(a, pad, mode="edge")
    windows = sliding_window_view(padded, (size, size)).reshape(*a.shape, size * size)
    real = np.isfinite(a)
    if real.all():
        # No fill anywhere, so `padded` is finite too and every neighbourhood is a
        # plain odd-length median. The whole output is defined; no scatter needed.
        return np.median(windows, axis=-1)
    out = np.full(a.shape, np.nan)
    if real.any():
        out[real] = np.nanmedian(windows[real], axis=-1)
    return out


# --- Normals dispersion -------------------------------------------------------


def normals_dispersion(z: np.ndarray, pixel_size: PixelSize) -> float:
    """Spherical variance of the unit surface normals, ``1 - ‖mean normal‖``, in
    ``[0, 1)``.

    Sauer's vector ruggedness measure. Every cell's normal is a unit vector; where
    the terrain is smooth they all point the same way and their mean is itself a
    unit vector, so the dispersion is zero — **on any plane, at any tilt**. Only
    terrain that turns the normals against each other scores. That is what
    separates this from :func:`slope_std`, which a curved-but-smooth hillside can
    still light up, and from slope, which reads a cliff as the roughest thing in
    the window.

    Zero is the floor, not the typical value, and that is the thing to watch: VRM's
    documented failure mode is a distribution piled up against zero, ~97% of
    windows reading "level". A metric can be exactly right and still not
    discriminate; whether this one earns its column is Phase B2's call.

    The normals come from :func:`_masked_gradient`, so ADR-0001 §3 binds here as it
    does on slope: the fill is NaN before the difference, and the shoreline cells
    whose gradient needs it are dropped rather than tilted against a painted cliff.
    """
    gy, gx = _masked_gradient(z, pixel_size)
    # Upward unit normal of the surface z(x, y): (-dz/dx, -dz/dy, 1), normalized.
    scale = np.sqrt(1.0 + gx**2 + gy**2)
    defined = np.isfinite(scale)
    if not defined.any():
        return float("nan")
    components = [-gx[defined] / scale[defined], -gy[defined] / scale[defined], 1.0 / scale[defined]]
    resultant = math.hypot(*(float(c.mean()) for c in components))
    # A mean of unit vectors cannot be longer than one, so a negative dispersion is
    # rounding, not terrain -- and a smooth window is where it lands, which is most
    # of them. Report the floor rather than -2e-16.
    return max(0.0, 1.0 - resultant)


# --- Geomorphons --------------------------------------------------------------

#: The ten geomorphon landforms (Jasiewicz & Stepinski 2013), in code order: a
#: :func:`geomorphons` code indexes this tuple.
GEOMORPHON_CLASSES: Final = (
    "flat",
    "peak",
    "ridge",
    "shoulder",
    "spur",
    "slope",
    "hollow",
    "footslope",
    "valley",
    "pit",
)
#: The code for a cell with no landform — the ocean fill (ADR-0001 §3).
GEOMORPHON_UNCLASSED: Final = -1
#: Line-of-sight radius, in cells — the prototype's 10, which it picked as an
#: *overview* radius meaning ~2.4 km of ground. Ten cells of a full-resolution
#: window is 300 m, so the same call asks a question about a different-sized
#: landform at each resolution, and A3 will see that as drift. Read a low ρ on this
#: metric as "the reach moved", not as "geomorphons don't survive the overview" —
#: the reach is a free parameter with no canonical cell value, and B2 owns it. Left
#: at the prototype's value because Phase A does not pre-prune, and changing it
#: here would silently rewrite the question B2 is meant to ask.
GEOMORPHON_LOS_PX: Final = 10
#: Below this elevation angle a direction reads as level, in degrees.
GEOMORPHON_FLAT_DEG: Final = 1.0

# The canonical ternary lookup, keyed ``(directions that fall away, directions
# that rise)``. Read the key the other way round and the table still returns a
# plausible landform — the wrong one, mirrored: a peak comes back "pit", a ridge
# "valley". The orientation is pinned by tests, not by trust; the constraint
# ``fall + rise <= 8`` leaves the rest of the 9x9 grid unreachable.
_FL, _PK, _RI, _SH, _SP, _SL, _HL, _FS, _VL, _PT = GEOMORPHON_CLASSES
_LANDFORM_TABLE: Final = {
    (0, 0): _FL, (0, 1): _FL, (0, 2): _FL, (0, 3): _FS, (0, 4): _FS, (0, 5): _FS,
    (0, 6): _VL, (0, 7): _VL, (0, 8): _PT,
    (1, 0): _FL, (1, 1): _FL, (1, 2): _FS, (1, 3): _FS, (1, 4): _FS, (1, 5): _VL,
    (1, 6): _VL, (1, 7): _VL,
    (2, 0): _FL, (2, 1): _SH, (2, 2): _SL, (2, 3): _SL, (2, 4): _HL, (2, 5): _HL,
    (2, 6): _VL,
    (3, 0): _SH, (3, 1): _SH, (3, 2): _SL, (3, 3): _SL, (3, 4): _SL, (3, 5): _HL,
    (4, 0): _SH, (4, 1): _SH, (4, 2): _SP, (4, 3): _SL, (4, 4): _HL,
    (5, 0): _RI, (5, 1): _RI, (5, 2): _SP, (5, 3): _SL,
    (6, 0): _RI, (6, 1): _RI, (6, 2): _SP,
    (7, 0): _RI, (7, 1): _RI,
    (8, 0): _PK,
}


def _landform_lookup() -> np.ndarray:
    lut = np.full((9, 9), GEOMORPHON_UNCLASSED, dtype=np.int8)
    for (falls, rises), name in _LANDFORM_TABLE.items():
        lut[falls, rises] = GEOMORPHON_CLASSES.index(name)
    return lut


_LANDFORM_LOOKUP: Final = _landform_lookup()


def geomorphons(z: np.ndarray, pixel_size: PixelSize) -> np.ndarray:
    """The 10-class landform map: an ``int8`` code per cell, indexing
    :data:`GEOMORPHON_CLASSES`, or :data:`GEOMORPHON_UNCLASSED` on the ocean fill.

    Each cell looks out along the eight compass directions to
    :data:`GEOMORPHON_LOS_PX` cells. A direction *rises* if the highest thing in
    sight along it clears :data:`GEOMORPHON_FLAT_DEG` and out-climbs whatever the
    direction falls to; it *falls* on the mirror condition; otherwise it is level.
    Counting the two gives the ternary pattern that names the landform — eight
    falling and none rising is a peak, six rising is a valley floor.

    Comparing the two angles, rather than counting a direction as both rising and
    falling when its line of sight does both, is what keeps the two counts summing
    to at most eight and so keeps the lookup on its canonical domain. The line of
    sight itself is the prototype's simplification of Jasiewicz & Stepinski: the
    extreme elevation angle along the ray, not their zenith/nadir openness.

    All four comparisons below are strict, so the *ties* matter as much as the
    order: a direction whose steepest rise and steepest fall come out equal counts
    as neither. That is why :func:`_sight_line_angles` still returns angles rather
    than the slopes it now works in — the two orders agree, but the tie sets do not.

    The mask goes on before the sight lines, per ADR-0001 §3, so a ray stops at the
    shore and reads the land behind it rather than the painted cliff. A direction
    with no land in sight is level, which makes a lone coastal cell flat — the
    honest reading when there is nothing to see. Rays are cut at the window edge
    the same way, so the border cells of a window see less than its middle does.
    """
    dy, dx = pixel_size
    a = np.asarray(z, dtype=np.float64)
    land = land_mask(a)
    masked = _masked_land(a)
    diagonal = math.hypot(dy, dx)
    directions = (
        (-1, 0, dy), (1, 0, dy), (0, -1, dx), (0, 1, dx),
        (-1, -1, diagonal), (-1, 1, diagonal), (1, -1, diagonal), (1, 1, diagonal),
    )
    level = math.radians(GEOMORPHON_FLAT_DEG)
    n_falls = np.zeros(a.shape, dtype=np.int8)
    n_rises = np.zeros(a.shape, dtype=np.int8)
    for dr, dc, step in directions:
        highest, lowest = _sight_line_angles(masked, dr, dc, step)
        drop = -lowest  # positive where the direction falls away
        n_rises += ((highest > drop) & (highest > level)).astype(np.int8)
        n_falls += ((drop > highest) & (drop > level)).astype(np.int8)
    return np.where(land, _LANDFORM_LOOKUP[n_falls, n_rises], GEOMORPHON_UNCLASSED).astype(np.int8)


def _sight_line_angles(
    z: np.ndarray, dr: int, dc: int, step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Highest and lowest elevation angle in radians along one direction, over
    :data:`GEOMORPHON_LOS_PX` cells; ``-pi/2`` / ``+pi/2`` where nothing is in sight.

    ``step`` is the ground distance of one cell in this direction, so the diagonals
    reach further than the axes and the anisotropy of ``dy`` against ``dx`` lands on
    the angle rather than being averaged away. NaN neighbours (fill, or off the
    edge) are skipped by ``fmax`` / ``fmin`` rather than poisoning the ray. A cell
    that saw nothing at all keeps its ``∓inf`` seed, which ``arctan`` carries to
    ``∓pi/2`` — level under every comparison in :func:`geomorphons`, exactly as the
    infinity it replaces.

    The ray is walked in raw slopes and converted to angles once at the end, not
    converted at every step, which is where this metric's cost used to sit (see the
    :mod:`dataset.catalog` module docstring). The rewrite is bit-exact, not merely
    close: ``arctan`` is monotone, and a monotone map commutes with ``max`` and
    ``min``, so ``max(arctan(s_k)) == arctan(max(s_k))`` for the floats too — the
    ``arctan`` of the winning slope *is* the winning angle, whatever the rounding.
    Two calls per direction therefore do the work of ten.

    Converting at the end rather than not at all is the deliberate part. Every use
    of these angles downstream is a comparison, and comparisons are monotone, so raw
    slopes look like they would do — but ``arctan`` is monotone without being
    *injective* on float64: above a slope of about 2 (~63°) neighbouring floats share
    an angle. Raw slopes would break ties that the angles hold, and a broken tie is a
    different landform. ``tests/test_metrics.py`` pins the window where it is.
    """
    h, w = z.shape
    highest = np.full(z.shape, -np.inf)
    lowest = np.full(z.shape, np.inf)
    for k in range(1, GEOMORPHON_LOS_PX + 1):
        r0, r1 = max(0, -dr * k), h - max(0, dr * k)
        c0, c1 = max(0, -dc * k), w - max(0, dc * k)
        if r1 <= r0 or c1 <= c0:
            break  # the whole array is off the end of this ray
        ahead = z[r0 + dr * k : r1 + dr * k, c0 + dc * k : c1 + dc * k]
        here = z[r0:r1, c0:c1]
        slope = (ahead - here) / (step * k)
        np.fmax(highest[r0:r1, c0:c1], slope, out=highest[r0:r1, c0:c1])
        np.fmin(lowest[r0:r1, c0:c1], slope, out=lowest[r0:r1, c0:c1])
    return np.arctan(highest), np.arctan(lowest)


def geomorphon_entropy(z: np.ndarray, pixel_size: PixelSize) -> float:
    """Shannon entropy in bits of the window's 10-class geomorphon histogram, from
    0 (every cell the same landform) to ``log2(10)`` ≈ 3.32 (all ten, equally).

    How *mixed* a window's terrain is, rather than how rough: a plateau and a
    uniform hillside both score 0, one because it is flat and the other because it
    is all one slope, while a window holding crests, flanks and hollows at once
    scores high. Only the land cells are counted (ADR-0001 §3).

    Demoted to a candidate by the curation plan, and carrying the obvious
    degeneracy risk: gentle terrain is all "flat", so the score piles up on 0 for
    exactly the windows the flat tail is made of. Phase B2 decides whether it
    discriminates; this is not the place to pre-prune it.
    """
    codes = geomorphons(z, pixel_size)
    classed = codes[codes != GEOMORPHON_UNCLASSED]
    if classed.size == 0:
        return float("nan")
    counts = np.bincount(classed, minlength=len(GEOMORPHON_CLASSES))
    share = counts[counts > 0] / classed.size
    # The single-class case negates a clean zero into -0.0; entropy has no sign.
    return float(-(share * np.log2(share)).sum()) + 0.0


# --- TPI-proxy ----------------------------------------------------------------

#: The share of the array's width taken as the TPI core; the frame left around it
#: is the neighbourhood the core is measured against. A half-width core is a
#: quarter of the area, so it is read against the three quarters surrounding it.
TPI_CORE_FRAC: Final = 0.5


def tpi_proxy(z: np.ndarray, pixel_size: PixelSize) -> float:
    """Topographic position in metres: how far the middle of the window stands
    above (positive) or below (negative) the ground around it.

    The plan's flat-wet axis, and the one metric here that reads *position* rather
    than shape. A basin floor scores negative, a dome or a ridge crest positive,
    and any plane scores exactly 0 — its middle is as far below the uphill half as
    it is above the downhill half, whatever the tilt. Flat terrain is not the same
    terrain twice: a floodplain and a plateau are both featureless inside the
    window, and only their position tells them apart. That is what B3's check needs
    and why this is held in reserve rather than dropped.

    **This is not the prototype's per-cell TPI, and deliberately so.** Measuring
    every cell against a 15-cell annulus and averaging over the window is a
    high-pass filter followed by its own mean: the interior cancels to nothing by
    construction, and what survives is the annulus being clipped at the window's
    rim. A bowl scored that way comes back *positive* — the exact inverse of the
    reading it is named for. The prototype gets away with it because it filters the
    whole tile and only then cuts windows out, so a window's annulus reaches into
    the tile around it.

    **The scale is the caller's to choose, and it is the whole metric.** The core
    is a share of the array, not a count of cells, so what "surroundings" means is
    whatever array is handed in:

    * a window, and the frame is the window's own margin — position *within* the
      window, which is what A3's calibration will loop over;
    * the window plus a ring of the terrain around it (an array twice the window's
      width puts the window itself at the core), and the frame is that ring — which
      is the plan's "window mean minus large-neighbourhood mean", exactly.

    So the reading has no fixed reach in metres; it inherits the reach of its input.
    Handed a window alone it cannot see anything wider than that window: a
    floodplain broader than 6.7 km reads flat, because from inside it, it is, and a
    5 m floodplain and a 2 km plateau both score 0. **B3's flat-wet check therefore
    needs the second call, not the first** — a null from a window-only score means
    "not visible at window scale", not "not there". That the plan's own contract
    hides this behind an innocuous-looking array argument is the finding, not a
    property of the terrain.

    ``pixel_size`` is unused — the core is a share of the array and the elevations
    already carry the geometry — but is part of the uniform metric contract. Only
    land is averaged, on either side of the split (ADR-0001 §3); a window whose
    core or frame is all sea has nothing to compare and returns NaN.
    """
    del pixel_size
    a = np.asarray(z, dtype=np.float64)
    core = np.zeros(a.shape, dtype=bool)
    core[_core_slice(a.shape[0]), _core_slice(a.shape[1])] = True
    land = land_mask(a)
    middle, frame = a[core & land], a[~core & land]
    if middle.size == 0 or frame.size == 0:
        return float("nan")
    return float(middle.mean() - frame.mean())


def _core_slice(length: int) -> slice:
    """The central :data:`TPI_CORE_FRAC` of an axis, centred exactly.

    Trimming the same margin off both ends is what keeps the core concentric with
    the window whatever the parity of ``length``, and concentric is the whole
    oracle: an off-centre core would read a plane's tilt as topographic position.
    """
    margin = (length - round(length * TPI_CORE_FRAC)) // 2
    return slice(margin, length - margin)


# --- Registry -----------------------------------------------------------------

#: Every metric candidate, keyed by manifest column name — the curation plan's
#: eight table rows, as nine functions, since slope's mean and standard deviation
#: share a row. The list is deliberately unpruned: it holds the one metric the plan
#: marks "candidate only" (geomorphon entropy) and the one it holds in reserve
#: (TPI-proxy), because Phase B2 prunes against measured terrain and cannot prune
#: what was never built. A3's calibration loops over it as it stands.
METRICS: dict[str, Callable[[np.ndarray, PixelSize], float]] = {
    "local_relief": local_relief,
    "slope_mean": slope_mean,
    "slope_std": slope_std,
    "elevation_relief_ratio": elevation_relief_ratio,
    "mean_elevation": mean_elevation,
    "texture": texture,
    "normals_dispersion": normals_dispersion,
    "geomorphon_entropy": geomorphon_entropy,
    "tpi_proxy": tpi_proxy,
}
