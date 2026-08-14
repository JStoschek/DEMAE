"""§2.4's classical fill — A1's two variants and the invariant every arm leans on (#158).

Seam 3-shaped: these are the claims that are equalities or refusals rather than magnitudes, and
they read badly through a CLI. What is *not* asserted here is how good either variant is — that
is a claim about a result, and the pre-registration exists to leave it open.

The load-bearing one is `writes inside the void and nowhere else`. Three separate things rest on
it: `benchmark.metrics.score`'s units check, §2.4's residual convention, and the
`fill(identity-mask) == input` gate — so it is asserted per variant rather than once.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark import fill
from benchmark.config import Fill

VARIANTS = sorted(fill.VARIANTS)

#: A plane rising 4 m per row and 3 m per column off a 1,200 m base — the golden tile's own
#: surface (`benchmark.metrics.GOLDEN_RECIPE`), and int16 metres exactly as a shard stores one.
ROWS, COLS = np.mgrid[0:224, 0:224]
PLANE = (1200 + 4 * ROWS + 3 * COLS).astype(np.int16)


def rectangle(row0=60, row1=100, col0=90, col1=130) -> np.ndarray:
    void = np.zeros((224, 224), dtype=bool)
    void[row0:row1, col0:col1] = True
    return void


def rough(seed: int = 0) -> np.ndarray:
    """Terrain a smooth interpolator cannot reproduce exactly — so a fill has to be a fill."""
    rng = np.random.default_rng(seed)
    return (PLANE + rng.normal(0, 40, PLANE.shape)).astype(np.int16)


# --- The identity gate --------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_fill_of_the_identity_mask_is_the_input(variant):
    """§8's first gate, at the function seam. Exact, not approximate — see the module docstring.

    A tile is int16 metres and float32 represents those exactly, so there is no round trip to
    absorb here and a tolerance would only hide one.
    """
    tile = rough()
    filled = fill.fill(tile, np.zeros(tile.shape, dtype=bool), recipe=Fill(variant=variant))

    assert np.array_equal(filled, tile.astype(np.float32))
    assert filled.dtype == fill.DTYPE


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_fill_writes_inside_the_void_and_nowhere_else(variant):
    """The invariant three separate things rest on — see the module docstring."""
    tile, void = rough(), rectangle()
    filled = fill.fill(tile, void, recipe=Fill(variant=variant))

    assert np.array_equal(filled[~void], tile[~void].astype(np.float32))
    assert not np.array_equal(filled[void], tile[void].astype(np.float32)), (
        "the fill reproduced the truth exactly inside the void, which means it read it"
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_fill_never_reads_the_truth_inside_the_void(variant):
    """The answer is punched out before any variant runs, so the void's contents cannot matter.

    Two tiles identical outside the void and wildly different inside it must fill identically.
    That is the property that makes A1 a *fill* rather than a copy, and it is not a property of
    the interpolator — GDAL's is measured to ignore masked values, but the punch is the contract
    the variant table is written against.
    """
    void = rectangle()
    a, b = rough(), rough()
    b[void] = -12_000  # nothing like the terrain around it

    recipe = Fill(variant=variant)
    assert np.array_equal(fill.fill(a, void, recipe=recipe), fill.fill(b, void, recipe=recipe))


# --- The variants -------------------------------------------------------------


def test_the_registry_holds_the_two_variants_that_are_two_methods():
    """The ticket names three and two of them are one implementation — see `benchmark.fill`.

    `rasterio.fill.fillnodata` *is* a binding of GDAL's `GDALFillNodata`, so registering both
    "the gdal_fillnodata-class fill" and "rasterio fillnodata" would put two names on one code
    path and give the results parquet a variant column with two bit-identical values.
    """
    assert VARIANTS == ["fillnodata", "tps"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_void_pixel_is_filled_finite(variant):
    tile, void = rough(), rectangle()
    filled = fill.fill(tile, void, recipe=Fill(variant=variant))

    assert np.isfinite(filled[void]).all()
    assert void.sum() == 1600


def test_the_thin_plate_spline_reproduces_a_plane_exactly():
    """TPS's own closed form, as the check that the solve is a solve.

    A thin-plate spline's affine part spans any plane, and its bending energy on one is zero — so
    over a planar tile the fill is the truth to float32. That is an identity about the estimator
    rather than a claim about void filling, which is exactly what makes it a good test of the
    linear system: an error in the kernel, the centring or the constraint block breaks it.
    """
    void = rectangle()
    filled = fill.fill(PLANE, void, recipe=Fill(variant="tps"))

    assert filled[void] == pytest.approx(PLANE[void].astype(np.float64), abs=1e-2)


def test_the_thin_plate_spline_is_a_function_of_the_mask_alone():
    """No `Generator` anywhere in it: the control points are an even stride over the ring.

    The committed eval mask list is what makes every arm's inputs byte-identical, so a control
    set drawn at random would make A1's own baseline depend on call order.
    """
    tile, void = rough(), rectangle()
    recipe = Fill(variant="tps")
    first = fill.fill(tile, void, recipe=recipe)

    for _ in range(3):
        assert np.array_equal(fill.fill(tile, void, recipe=recipe), first)


def test_the_control_point_budget_bounds_the_solve_rather_than_the_ring():
    """A large void's ring runs past the budget, and §7 prices eval scoring as the largest line.

    Asserted as a *difference*: a budget below the ring's size has to change the answer, or the
    subsampling is not happening and the O(N³) bound is not being enforced.
    """
    tile = rough()
    void = rectangle(40, 140, 40, 140)  # a ring of ~400 px
    tight = fill.fill(tile, void, recipe=Fill(variant="tps", tps_control_points=16))
    loose = fill.fill(tile, void, recipe=Fill(variant="tps", tps_control_points=512))

    assert not np.array_equal(tight[void], loose[void])


# --- The refusals -------------------------------------------------------------


def test_an_unregistered_variant_is_refused_by_the_config():
    """A typo'd override stops before the corpus is touched — the rule `--arm A8` follows."""
    with pytest.raises(ValueError, match="unknown fill variant 'tsp'"):
        Fill(variant="tsp")


def test_a_variant_the_registry_lost_is_refused_at_the_fill():
    """The config refuses first, so this is the second line: `fill` states the registry too."""
    recipe = Fill()
    object.__setattr__(recipe, "variant", "nearest")  # past the frozen dataclass's guard
    with pytest.raises(ValueError, match="unknown A1 variant 'nearest'"):
        fill.fill(rough(), rectangle(), recipe=recipe)


def test_a_mask_that_is_not_the_tiles_shape_is_refused():
    with pytest.raises(ValueError, match="are one tile"):
        fill.fill(rough(), np.zeros((100, 100), dtype=bool))


def test_a_void_touching_the_tile_edge_is_refused_rather_than_filled():
    """§3.2 draws every void interior at >= 8 px, so this is a mask nobody should have committed.

    The TPS variant is the one that can see it — a void reaching the edge on every side has no
    boundary ring — and it says so in those terms rather than raising a linear-algebra error.
    """
    void = np.ones((224, 224), dtype=bool)
    with pytest.raises(fill.FillError, match="no boundary ring"):
        fill.fill(rough(), void, recipe=Fill(variant="tps"))
