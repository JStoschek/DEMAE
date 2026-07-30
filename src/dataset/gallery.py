"""E2.4 stratum gallery — the pure half (Phase E, issue #53).

The gallery is the plan's **human backstop** (§186) that k=20 strata are terrain-
meaningful: 12 tiles per stratum, drawn as hillshade contact sheets, judged for
*internal coherence* against the protocol pre-registered in
``docs/research/e2-gallery-protocol.md``. You cannot ``assert gallery_looks_right``,
so what code can own here is everything *up to* the judgement — which 12 tiles, and
how a 224² int16 elevation array becomes an image a human can read terrain shape off.

That is what this module is. The rendering *driver* lives in
``notebooks/e2_stratum_gallery.py``; matplotlib is a dev-only dependency and no
``src/dataset`` module imports it, so the shading here is plain NumPy — which also
makes it unit-testable, unlike a figure.

## Why the rendering choices are choices

A badly-drawn DEM makes any stratum look like any other, so each knob below was
prototyped against real tiles before it was fixed (#53's HITL react-then-commit):

* **Grey hillshade, magnitudes printed — never colour-coded.** A shared elevation
  ramp renders every low-relief stratum as one flat block, destroying exactly the
  strata that are hardest to judge. A *per-tile* hypsometric ramp is worse: it gave a
  10 m-relief tile the full green-valley-to-white-peak treatment, manufacturing
  drama a coherence review would then be judging. Grey shading carries shape; the
  caption carries magnitude.
* **A low-pass before the shade (**:data:`SMOOTH_SIGMA_PX`**).** Hillshading is a
  derivative, and at the exaggeration a 20 m-relief tile needs, the stored int16's
  1 m quantum becomes visible stipple — two flat strata can differ in apparent
  "texture" for no reason but noise. One pixel of Gaussian recovers the real
  landform (dune fields, patterned ground, drainage nets emerge); σ ≥ 2 starts
  smearing genuine detail off rugged tiles.
* **Per-tile vertical exaggeration** (:func:`auto_vert_exag`). Fixed exaggeration
  cannot serve both a 2,600 m Andean tile and a 14 m coastal plain.
* **True anisotropic pixel spacing** (:func:`~dataset.geometry.band_pixel_spacing_m`).
  Copernicus columns are up to 1.55x wider on the ground than its rows; shading a
  polar tile as if its pixels were square draws it far too steep across.

Every one of these is a *presentation* transform. None of them touches a stored
tile, and the caption on each sheet states them, because a reviewer marking a
stratum `incoherent` is entitled to know what was done to the pixels first.

## The draw

Two draws, and the distinction between them is the point (#53):

* :func:`uniform_draw` is the **gating** one — it is the "fixed random sample with a
  committed seed" the protocol pre-registered, and it estimates the *typical* member
  of a stratum.
* :func:`spanning_draw` walks each stratum interior → edge. #41 found the descriptor
  space is a continuum rather than clumps, so a stratum's boundary tiles are where
  "is this a real family" actually gets tested — but over-sampling the boundary makes
  the coherence gate *stricter* than what was registered. So it ships as a published
  diagnostic and gates nothing, the same standing #47 gave the pool-ECDF and the
  coverage map.

Both order by :func:`dataset.sampling._shuffle_key`, the house determinism rule — a
keyed BLAKE2b, never a NumPy ``Generator`` — so the 240 tiles the verdict was rendered
against are reproducible from the committed manifest and the seed alone. That matters
more here than anywhere else in Phase E: the shards are gitignored, so once they are
gone the seed plus the manifest is the *only* thing that can reconstitute the exact
sheets a Gate E mark was written against.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .centroids import Centroids
from .frame import METRICS, Scaler, metric_matrix
from .geometry import band_pixel_spacing_m
from .sampling import _shuffle_key  # the house keyed-BLAKE2b (as `select` imports it)

#: The gallery's committed draw seed. Fixed here rather than passed in: the protocol's
#: procedure rule 2 requires the exact 240 tiles to be recoverable after the fact, and a
#: seed that lives in a shell invocation is not committed. The driver writes it into the
#: sidecar JSON beside the tile ids it produced.
GALLERY_SEED: int = 20260723

#: Tiles per sheet — plan §186's twelve, and the number the protocol's mark table assumes.
TILES_PER_SHEET: int = 12

#: Gaussian σ in **pixels** applied before shading. See the module docstring: 1 px (30 m
#: at the equator) is where quantization stipple stops dominating the derivative and
#: real landform reappears, without visibly softening high-relief tiles.
SMOOTH_SIGMA_PX: float = 1.0

#: The land-std, in metres, :func:`auto_vert_exag` exaggerates each tile *towards*.
TARGET_LAND_STD_M: float = 60.0

#: Exaggeration bounds. The floor keeps the highest-relief tiles from being flattened
#: into mush; the ceiling stops a near-degenerate tile (land std just over ADR-0002's
#: 3 m ``degenerate`` screen) from being amplified into pure noise.
VE_MIN: float = 0.2
VE_MAX: float = 400.0

#: Illumination. North-west at 45° is the cartographic convention, and it is a
#: convention worth keeping: lighting a DEM from the south-east makes most readers see
#: ridges as valleys (the relief-inversion illusion), which would corrupt a review whose
#: whole question is "what landform is this".
AZIMUTH_DEG: float = 315.0
ALTITUDE_DEG: float = 45.0

#: Ocean fill (ADR-0001 §3's exact zero) is painted, not shaded — a flat plane returns a
#: uniform grey that reads as land. Pale blue instead, so a coastline is unmistakable.
OCEAN_RGB: tuple[float, float, float] = (0.62, 0.73, 0.82)


# --- Which twelve -------------------------------------------------------------


def uniform_draw(
    tile_ids: "np.ndarray | list[str]",
    *,
    n: int = TILES_PER_SHEET,
    seed: int = GALLERY_SEED,
) -> list[int]:
    """Positions of a keyed-random ``n`` of ``tile_ids`` — **the gating draw**.

    Returns *positions into the input*, not ids, so the caller keeps its join to the
    manifest row (and so a stratum's tiles can be pulled from a slice without carrying
    a second lookup). Order is the hash order, which is arbitrary but fixed — the sheet
    lays them out in it, so the grid position of a tile is itself reproducible.
    """
    return sorted(range(len(tile_ids)), key=lambda i: _shuffle_key(str(tile_ids[i]), seed))[:n]


def spanning_draw(
    tile_ids: "np.ndarray | list[str]",
    radii: np.ndarray,
    *,
    n: int = TILES_PER_SHEET,
    seed: int = GALLERY_SEED,
) -> list[int]:
    """Positions of ``n`` tiles spanning interior → edge — **the diagnostic draw**.

    Splits the stratum into ``n`` equal-count bins by distance to its own centroid and
    takes one keyed-random tile from each, so the sheet walks from the most typical
    member to the most marginal one. Equal-*count* bins rather than equal-width: radius
    is densely packed near the centroid and long-tailed outward, so width bins would
    spend most of the sheet on the tail.

    Deliberately **not** the gating draw. Over-representing the boundary raises the
    chance of a `mixed` mark relative to the uniform sample #47 registered, and a gate
    that is quietly stricter than its pre-registration is no longer pre-registered.
    """
    if len(tile_ids) != len(radii):
        raise ValueError(f"{len(tile_ids)} tile ids but {len(radii)} radii")
    order = sorted(
        range(len(tile_ids)),
        key=lambda i: (float(radii[i]), _shuffle_key(str(tile_ids[i]), seed)),
    )
    if not order:
        return []
    return [
        min(b.tolist(), key=lambda i: _shuffle_key(str(tile_ids[i]), seed))
        for b in np.array_split(np.asarray(order), min(n, len(order)))
    ]


def centroid_radius(
    columns: "Any", stratum_id: np.ndarray, *, scaler: Scaler, centroids: Centroids
) -> np.ndarray:
    """Each row's Euclidean distance to **its own** stratum centroid, standardized space.

    "Interior vs edge" only means anything in the space the strata were actually cut in,
    so this goes through :func:`~dataset.frame.metric_matrix` (ADR-0004's ``ln(x+1)`` on
    two of the six) and the committed scaler rather than re-deriving either — the same
    reason :func:`~dataset.centroids.assign` does. Reimplementing the transform here is
    the one bug in this module that would produce plausible, wrong numbers rather than
    an error: a plain ``log`` for ``ln(x+1)`` shifts every radius by a smooth amount and
    nothing downstream would notice.
    """
    Z = scaler.standardize(metric_matrix(columns))
    centers = np.asarray(centroids.centers, dtype=np.float64)
    return np.linalg.norm(Z - centers[np.asarray(stratum_id, dtype=np.intp)], axis=1)


def radius_percentile(radii: np.ndarray, member: np.ndarray) -> np.ndarray:
    """Percentile rank of each member's radius **within its group** — 0 = the centre.

    Annotated on every tile so a reviewer weighing a `mixed` mark can see whether the
    odd tile is a marginal member or a core one, without the draw itself having been
    changed to chase boundaries.
    """
    radii = np.asarray(radii, dtype=np.float64)
    out = np.zeros(radii.size, dtype=np.float64)
    for g in np.unique(member):
        sel = np.flatnonzero(member == g)
        if sel.size == 1:
            continue
        r = radii[sel]
        out[sel] = 100.0 * (r[:, None] > r[None, :]).sum(axis=1) / (sel.size - 1)
    return out


# --- Turning a stored tile into something a human can read --------------------


def gaussian_kernel1d(sigma: float, *, truncate: float = 3.0) -> np.ndarray:
    """A normalized 1-D Gaussian, radius ``ceil(truncate * sigma)`` — SciPy's convention.

    Hand-rolled because SciPy is not a declared dependency of this project (it arrives
    only under scikit-learn, and :mod:`dataset.select` imports it lazily for that
    reason). A separable 7-tap blur is not worth a dependency, and this way the smoothing
    the sheets were rendered with is covered by the test suite.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive; got {sigma}")
    r = int(math.ceil(truncate * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def smooth(a: np.ndarray, sigma: float = SMOOTH_SIGMA_PX) -> np.ndarray:
    """Separable Gaussian blur with edge reflection.

    Reflected rather than zero-padded: a tile is a window cut out of a continuous
    surface, and zero-padding would fabricate a sea-level cliff around all four sides
    that the shading would then draw as a bright rim.
    """
    k = gaussian_kernel1d(sigma)
    r = k.size // 2
    out = np.asarray(a, dtype=np.float64)
    for axis in (0, 1):
        padded = np.pad(out, [(r, r) if ax == axis else (0, 0) for ax in (0, 1)], mode="reflect")
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis, padded)
    return out


def auto_vert_exag(
    land_std: float,
    *,
    target: float = TARGET_LAND_STD_M,
    lo: float = VE_MIN,
    hi: float = VE_MAX,
) -> float:
    """Vertical exaggeration that brings a tile's land-std to ``target`` metres.

    The corpus spans ~3 m to ~613 m of land-std by design — that range *is* the
    stratification — so no single exaggeration renders both ends. Scaling each tile
    towards one target is what makes a flat stratum's landform visible at all, and it
    is also why the sheets say nothing about relative magnitude: the caption's printed
    ``relief`` and ``elev`` carry that, and the factor applied is printed beside them so
    the transform is never silent.
    """
    return float(np.clip(target / max(float(land_std), 1e-3), lo, hi))


def hillshade(
    z: np.ndarray,
    *,
    dy: float,
    dx: float,
    vert_exag: float = 1.0,
    azimuth_deg: float = AZIMUTH_DEG,
    altitude_deg: float = ALTITUDE_DEG,
    stretch: bool = True,
) -> np.ndarray:
    """Shaded relief in ``[0, 1]`` — surface normals dotted with the light vector.

    ``dy``/``dx`` are ground metres per pixel and are applied to their own axes (never
    averaged), so the anisotropy Copernicus's poleward column coarsening leaves behind
    is drawn as the shape it is.

    By default the result is contrast-stretched across the tile, matching
    :class:`matplotlib.colors.LightSource` — whose formulation this reproduces, so the
    committed sheets look like the prototype they were approved from. The stretch means
    *apparent contrast carries no information*: it is full-range on every tile,
    exaggerated or not. That is a feature for a coherence review (each tile is legible
    on its own terms) and a trap for anything else, which is why nothing quantitative in
    E2 reads a rendered image.

    ``stretch=False`` returns the plain clipped Lambertian intensity, and it exists for the
    one caller that must *not* have the trap: plan §10.2's reconstruction grids (#98) shade
    a prediction beside its own truth, and the failure they are built to catch is a
    prediction that came back **smeared**. Stretched per panel, a smeared reconstruction has
    its contrast pushed straight back to full range and looks fine. Unstretched, less relief
    draws as less contrast, which is the whole point.
    """
    # Row 0 is the *north* edge, so a step down the row axis is a step south: the row
    # spacing enters the gradient negated. Dropping this sign mirrors the illumination
    # north-south, which inverts every ridge into a valley — the precise illusion the
    # north-west convention above exists to avoid, and it is invisible unless you look
    # for it, because the picture stays a perfectly plausible piece of terrain.
    ey, ex = np.gradient(np.asarray(z, dtype=np.float64) * float(vert_exag), -dy, dx)
    normal = np.dstack([-ex, -ey, np.ones_like(ex)])
    normal /= np.linalg.norm(normal, axis=2)[..., None]

    az, alt = math.radians(90.0 - azimuth_deg), math.radians(altitude_deg)
    light = np.array([math.cos(az) * math.cos(alt), math.sin(az) * math.cos(alt), math.sin(alt)])

    intensity = normal.dot(light)
    if stretch:
        lo, hi = float(intensity.min()), float(intensity.max())
        if hi - lo > 1e-6:
            intensity = (intensity - lo) / (hi - lo)
    return np.clip(intensity, 0.0, 1.0)


def shade_tile(
    raw_i16: np.ndarray,
    *,
    lat: float,
    sigma: float = SMOOTH_SIGMA_PX,
    target: float = TARGET_LAND_STD_M,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """A stored tile → ``(rgb, vert_exag, (dy, dx))`` ready for ``imshow``.

    Ocean (ADR-0001 §3's exact zero) is filled with the land median *before* smoothing
    and shading, then painted :data:`OCEAN_RGB` afterwards. Filling first stops the
    coastline from being shaded as a several-hundred-metre cliff whose glare swamps the
    land beside it — which on a coastal tile is the terrain actually under review.

    The exaggeration is measured on the **smoothed** field, so the factor reported is
    the one that produced the picture rather than one computed against detail the blur
    already removed.
    """
    z = np.asarray(raw_i16, dtype=np.float64)
    land = z != 0
    filled = z.copy()
    if land.any():
        filled[~land] = float(np.median(z[land]))
    filled = smooth(filled, sigma) if sigma > 0 else filled

    land_std = float(filled[land].std()) if land.any() else 1.0
    ve = auto_vert_exag(land_std, target=target)
    dy, dx = band_pixel_spacing_m(lat)

    rgb = np.repeat(hillshade(filled, dy=dy, dx=dx, vert_exag=ve)[..., None], 3, axis=2)
    rgb[~land] = np.asarray(OCEAN_RGB)
    return rgb, ve, (dy, dx)


__all__ = [
    "ALTITUDE_DEG",
    "AZIMUTH_DEG",
    "GALLERY_SEED",
    "METRICS",
    "OCEAN_RGB",
    "SMOOTH_SIGMA_PX",
    "TARGET_LAND_STD_M",
    "TILES_PER_SHEET",
    "VE_MAX",
    "VE_MIN",
    "auto_vert_exag",
    "centroid_radius",
    "gaussian_kernel1d",
    "hillshade",
    "radius_percentile",
    "shade_tile",
    "smooth",
    "spanning_draw",
    "uniform_draw",
]
