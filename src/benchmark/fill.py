"""§2.4's classical fill — A1, and the base every learned arm is a residual over (#158).

A1 runs first and its numbers gate everything else, because the classical baseline is what
every learned arm improves on. That makes this module load-bearing twice over: it produces the
**A1 row**, and it produces the **input** three of the frozen arms read — §1's
``a1_fill_plus_mask`` convention is literally this function's output with the void mask beside
it, and §2.4 has every learned arm predict a residual over it so that *a head that learns
nothing scores exactly A1*.

## Two variants, not three

The ticket names A1's variants as "the ``gdal_fillnodata``-class fill, rasterio ``fillnodata``,
and the TPS variant", and the spec's registry line says the same. **The first two are one
implementation.** :func:`rasterio.fill.fillnodata` is a binding of GDAL's own
``GDALFillNodata``, so "the ``gdal_fillnodata``-class fill" names the *class* of method and
"rasterio ``fillnodata``" names this project's way of calling it. Registering both would put two
names on one code path and invite a results table with an ``a1_variant`` column whose two values
are bit-identical. So :data:`VARIANTS` holds ``fillnodata`` and ``tps``, and this paragraph is
the resolution rather than a third entry — the disposition the spec asks for when the build
finds the plan wrong.

## Writing inside the void and nowhere else

Every variant returns the tile it was handed with the void written over, and **nothing else
moved**. That is enforced here (:func:`fill` restores the outside from the input) rather than
trusted of each interpolator, and it buys three separate things:

* :func:`benchmark.metrics.score` refuses a fill that differs from the truth outside the void —
  its check that an arm was scored in metres. A variant that shifted an untouched pixel by a
  float32 ulp would trip that refusal for a reason that has nothing to do with units.
* ``fill(identity-mask) == input`` holds exactly for **every** variant, present and future,
  rather than for the ones that happen to be identity-preserving on their own — GDAL's is only
  identity-preserving because a tile is int16 metres, which is a fact about the corpus rather
  than about the method.
* §2.4's residual convention is well-defined: a learned arm adds its residual to this array, so
  "outside the void" has to mean the original pixels exactly.

## Why the interpolator never sees the answer

The void's true pixels are punched out before any variant runs. GDAL is measured to ignore
masked values entirely — filling with 0, −9999, 99999 or NaN under the same mask produces
bit-identical output — so the punch is belt-and-braces there, but it is the *contract* for the
variant table, and :func:`_tps` genuinely reads the array it is handed. A variant that could
see the truth inside the void would score a perfect fill and raise nothing, which is the class
of failure §8's risk row is about.

## Float32, and why the identity gate is exact

GDAL's fill computes in float32: handed float64 it returns exactly ``a.astype(np.float32)``.
Rather than hide that behind a float64 signature, this module works in float32 throughout. It
costs nothing — a tile is int16 metres, which float32 represents exactly (the corpus tops out
four orders of magnitude below float32's 24-bit integer range) — and it is what makes the
identity gate an equality rather than a tolerance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Final

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - the annotation only; see the config note below
    from .config import Fill

#: What GDAL is handed in the void. Any value does — the fill is measured to be independent of
#: it — so this is the one that makes a leak *visible* rather than plausible: an interpolator
#: that read it would produce a fill sitting at zero metres, which no scored column could
#: mistake for terrain.
PUNCH_VALUE: Final = 0.0

#: The dtype every variant computes and returns in. GDAL's own — see the module docstring.
DTYPE: Final = np.float32


class FillError(ValueError):
    """A fill that could not be produced — never a fill that was quietly repaired."""


def _punched(tile: np.ndarray, void: np.ndarray) -> np.ndarray:
    """The tile as an arm is handed it: float32 metres, with the void's true pixels gone."""
    punched = np.asarray(tile, dtype=DTYPE).copy()
    punched[void] = PUNCH_VALUE
    return punched


def _fillnodata(tile: np.ndarray, void: np.ndarray, recipe: Fill) -> np.ndarray:
    """GDAL's ``GDALFillNodata`` through rasterio — §2.4's ``fillnodata``-class fill.

    Inverse-distance-weighted interpolation from the void's edge inwards, which is the method
    §2.4 names and the one every published DEM void-fill baseline is a variation of. Imported
    inside the function on :mod:`benchmark`'s torch-optional rule one layer down: ``freeze`` is
    CPU-only and never fills anything, so it should not pay for a GDAL binding at import.

    ``max_search_distance`` defaults to the whole tile rather than to a radius, because §3.2
    caps a void at 40% of the tile and guarantees it interior — so every void pixel *has* a
    reachable edge, and a search radius short enough to leave one unfilled would be a silent
    hole in a scored instance rather than a tuning choice.
    """
    from rasterio.fill import fillnodata

    return fillnodata(
        _punched(tile, void),
        mask=(~void).astype(np.uint8),
        max_search_distance=float(recipe.max_search_distance_px),
        smoothing_iterations=int(recipe.smoothing_iterations),
    )


def _thin_plate_kernel(r2: np.ndarray) -> np.ndarray:
    """``U(r) = r² · log r``, written on ``r²`` so the log is one call and ``U(0) = 0`` is exact.

    ``r² log r`` is ``½ r² log r²``, and taking it that way avoids a square root whose only use
    is to be squared again. The ``r² = 0`` case is the self-distance on the diagonal of the
    control-point matrix; the limit there is 0, and ``np.where`` on a pre-masked log keeps it
    from being ``0 · -inf``.
    """
    return np.where(r2 > 0, 0.5 * r2 * np.log(np.where(r2 > 0, r2, 1.0)), 0.0)


def _tps(tile: np.ndarray, void: np.ndarray, recipe: Fill) -> np.ndarray:
    """The thin-plate-spline variant — §2.4's smooth-surface alternative to the IDW fill.

    Fitted to the void's **boundary ring** (the valid pixels 8-connected to it) rather than to
    the whole tile: the ring is what a fill has to be continuous with, and it is what makes the
    seam-gap column mean something. A TPS over the whole tile would be a different estimator
    that happened to also fill the void.

    The exact solve is ``O(N³)`` in control points, and §7 prices eval scoring as this
    benchmark's largest line (2.0–5.2 CPU-hours). A large void's ring runs to ~500 px, so the
    ring is **subsampled to** :attr:`~benchmark.config.Fill.tps_control_points` on an even
    stride — deterministic, so the variant is as reproducible as the committed masks it fills,
    and stated in the config rather than hidden here, because a control-point count is a
    property of the baseline a results row is scored against.

    Written in NumPy rather than reached for from ``scipy.interpolate``: :mod:`benchmark` runs
    the scoring path on a box with no CUDA stack, scipy is in the environment only transitively
    through scikit-learn, and the solve is one linear system — the same argument
    :func:`benchmark.metrics.erode` makes about ``scipy.ndimage``.
    """
    out = _punched(tile, void)
    rows, cols = np.nonzero(void)
    if rows.size == 0:
        # Nothing to predict, so there is nothing to fit — and a void with no pixels has no
        # boundary ring either, which is the check below. This is the identity-mask case
        # reaching the one variant that has to say something about it.
        return out

    ring = _boundary_ring(void)
    if not ring.any():
        raise FillError(
            "the void has no boundary ring to interpolate from: every pixel 8-connected to it "
            "is itself void, which means the void reaches the tile edge. §3.2 draws every void "
            "fully interior at >= 8 px from the edge, so this is a mask that should not have "
            "been committed rather than a fill to repair"
        )
    ring_rows, ring_cols = _subsampled(ring, recipe.tps_control_points)
    values = np.asarray(tile, dtype=np.float64)[ring_rows, ring_cols]

    # The control points centred, so the affine part is solved about the void rather than about
    # the tile's corner. Purely conditioning: `U` grows like r² log r, and an origin 200 px away
    # makes the polynomial block's columns differ by two orders of magnitude from the kernel's.
    origin_r, origin_c = float(ring_rows.mean()), float(ring_cols.mean())
    cr, cc = ring_rows - origin_r, ring_cols - origin_c
    n = cr.size
    if n < 3:
        raise FillError(
            f"the void's boundary ring holds {n} pixel(s), and a thin-plate spline's affine "
            f"part needs three. §3.2 draws no void that thin, so this is a committed mask to "
            f"look at rather than a fill to repair"
        )

    r2 = (cr[:, None] - cr[None, :]) ** 2 + (cc[:, None] - cc[None, :]) ** 2
    k = _thin_plate_kernel(r2)
    if recipe.tps_regularization:
        k = k + recipe.tps_regularization * np.eye(n)
    p = np.column_stack([np.ones(n), cr, cc])

    a = np.zeros((n + 3, n + 3), dtype=np.float64)
    a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T
    b = np.concatenate([values, np.zeros(3)])
    # `lstsq` rather than `solve`: the system is singular whenever the ring's control points are
    # collinear — a thin rectangular void's is — and the minimum-norm solution is the right
    # reading of that, where a `LinAlgError` would refuse an instance §3.2 permits.
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    w, affine = solution[:n], solution[n:]

    vr, vc = rows - origin_r, cols - origin_c
    d2 = (vr[:, None] - cr[None, :]) ** 2 + (vc[:, None] - cc[None, :]) ** 2
    predicted = (
        affine[0] + affine[1] * vr + affine[2] * vc + _thin_plate_kernel(d2) @ w
    )
    out[rows, cols] = predicted.astype(DTYPE)
    return out


def _boundary_ring(void: np.ndarray) -> np.ndarray:
    """The valid pixels 8-connected to the void — what a fill has to be continuous with.

    Nine ORed slices, the dilation :func:`benchmark.metrics.erode` is the mirror of, for the
    same reason that one is written out: it is the whole of the dependency and it is six lines.
    """
    h, w = void.shape
    padded = np.pad(void, 1, constant_values=False)
    dilated = np.zeros_like(void, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            dilated |= padded[dr : dr + h, dc : dc + w]
    return dilated & ~void


def _subsampled(ring: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """At most ``budget`` ring pixels as ``(rows, cols)``, on an even stride over raster order.

    An even stride rather than a random sample: the draw has to be a function of the mask alone,
    since the committed eval mask list is what makes every arm's inputs byte-identical, and a
    control set drawn off a `Generator` would make A1's own baseline depend on call order.
    """
    rows, cols = np.nonzero(ring)
    if rows.size > budget:
        picked = np.unique(np.linspace(0, rows.size - 1, budget).round().astype(int))
        rows, cols = rows[picked], cols[picked]
    return rows, cols


#: A1's variants, by the name a config knob carries. A table rather than a branch, on
#: :data:`benchmark.masks.FAMILIES`' rule: a fourth variant is an entry, and an unregistered one
#: fails the way every other unknown name in this benchmark does.
VARIANTS: Final[dict[str, Callable[[np.ndarray, np.ndarray, "Fill"], np.ndarray]]] = {
    "fillnodata": _fillnodata,
    "tps": _tps,
}


def fill(tile_m: np.ndarray, void: np.ndarray, *, recipe: Fill | None = None) -> np.ndarray:
    """A1's fill: the tile in metres with the void interpolated over, and nothing else moved.

    ``tile_m`` is a whole tile in **metres** — the corpus stores int16 metres, so for A1 there is
    no transform to invert and :func:`benchmark.metrics.to_metres` does not enter this path.
    ``void`` is the boolean mask of the pixels this arm is asked to predict, decoded from the
    committed eval mask list.

    An empty ``void`` returns the input unchanged — the ``fill(identity-mask) == input`` gate's
    subject. Worth being exact about what that gate proves, since it is one of the two §8 fires
    at ``run``: it is not evidence that an interpolator is correct, because there is nothing to
    interpolate. What it holds shut is the surrounding machinery — that a variant handed no work
    returns the array it was handed, at the same shape and dtype, in metres, with the void mask
    decoded to the right polarity. A pipeline that had the mask inverted fills the whole tile
    here and fails loudly, which is exactly the bug that otherwise reaches the parquet as a
    plausible column of numbers.
    """
    from .config import Fill as FillRecipe

    recipe = FillRecipe() if recipe is None else recipe
    tile_m = np.asarray(tile_m)
    void = np.asarray(void, dtype=bool)
    if tile_m.shape != void.shape:
        raise ValueError(f"tile {tile_m.shape} and void {void.shape} are one tile")
    if tile_m.ndim != 2:
        raise ValueError(f"fill takes one tile at a time; got shape {tile_m.shape}")
    if recipe.variant not in VARIANTS:
        raise ValueError(
            f"unknown A1 variant {recipe.variant!r}; registered: {sorted(VARIANTS)}. §2.4's "
            f"classical fill is the `gdal_fillnodata`-class method (`fillnodata`) and the "
            f"thin-plate spline (`tps`) — see this module on why those are two names and not "
            f"three"
        )

    filled = VARIANTS[recipe.variant](tile_m, void, recipe)

    # Inside the void, a non-finite value is a hole in a scored instance — every metric in §5 is
    # computed there, so it would print as NaN in a column rather than as a failure.
    if not np.isfinite(filled[void]).all():
        raise FillError(
            f"the {recipe.variant} fill left {int((~np.isfinite(filled[void])).sum())} of "
            f"{int(void.sum())} void pixels non-finite. Every §5 metric is computed inside the "
            f"void, so this would reach the results parquet as a NaN column rather than as a "
            f"refusal"
        )
    # The outside, restored from the input rather than trusted of the interpolator — see the
    # module docstring on why this is the enforcement point.
    outside = ~void
    filled[outside] = np.asarray(tile_m, dtype=DTYPE)[outside]
    return filled
