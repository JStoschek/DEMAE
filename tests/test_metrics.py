"""Analytic-oracle metrics — each asserted against a synthetic surface whose
expected value is known in closed form (plan A2, issue #16)."""

from __future__ import annotations

import numpy as np
import pytest

from dataset import metrics, surfaces

#: Square 30 m pixels -- Copernicus at the equator, and the default the oracles
#: below are hand-computed against. Cases that turn on the pixel size (anisotropy,
#: the x8 overview) spell theirs out locally instead.
PX = (30.0, 30.0)


# --- local relief: ramp of known range ----------------------------------------


def test_local_relief_equals_ramp_range():
    z = surfaces.ramp((64, 64), low=100.0, high=350.0)
    assert metrics.local_relief(z, (30.0, 30.0)) == pytest.approx(250.0)


def test_local_relief_of_flat_plateau_is_zero():
    z = surfaces.plateau((32, 32), height=250.0)
    assert metrics.local_relief(z, (30.0, 30.0)) == pytest.approx(0.0)


# --- slope: inclined plane -> exact slope, zero std ----------------------------


@pytest.mark.parametrize("tilt", [0.5, 5.0, 30.0, 60.0])
@pytest.mark.parametrize("aspect", [0.0, 45.0, 90.0, 205.0, 330.0])
def test_slope_of_inclined_plane_is_its_tilt_at_any_aspect(tilt, aspect):
    px = (30.0, 30.0)
    z = surfaces.inclined_plane((48, 48), pixel_size=px, slope_deg=tilt, aspect_deg=aspect)
    assert metrics.slope_mean(z, px) == pytest.approx(tilt, abs=1e-9)
    assert metrics.slope_std(z, px) == pytest.approx(0.0, abs=1e-9)


def test_slope_of_inclined_plane_is_exact_under_anisotropic_pixels():
    # ~50 deg N: Copernicus columns are ~1.55x wider on the ground than rows.
    px = (30.9, 19.9)
    z = surfaces.inclined_plane((48, 48), pixel_size=px, slope_deg=12.0, aspect_deg=70.0)
    assert metrics.slope_mean(z, px) == pytest.approx(12.0, abs=1e-9)
    assert metrics.slope_std(z, px) == pytest.approx(0.0, abs=1e-9)


def test_slope_of_flat_plateau_is_zero():
    z = surfaces.plateau((32, 32), height=250.0)
    assert metrics.slope_mean(z, (30.0, 30.0)) == pytest.approx(0.0)
    assert metrics.slope_std(z, (30.0, 30.0)) == pytest.approx(0.0)


def test_slope_reads_the_pixel_size_it_is_given():
    # The contract that makes A3 possible: the same array scored at a coarser
    # pixel size is a gentler slope. A ramp of 100 m over 31 gaps of dy metres.
    z = surfaces.ramp((32, 32), low=1000.0, high=1100.0, axis=0)
    fine = metrics.slope_mean(z, (30.0, 30.0))
    coarse = metrics.slope_mean(z, (240.0, 240.0))
    assert fine == pytest.approx(np.degrees(np.arctan(100.0 / (31 * 30.0))), abs=1e-9)
    assert coarse == pytest.approx(np.degrees(np.arctan(100.0 / (31 * 240.0))), abs=1e-9)


# --- elevation-relief ratio (hypsometric integral) ----------------------------


@pytest.mark.parametrize("aspect", [0.0, 45.0, 90.0, 205.0])
def test_err_of_any_plane_is_one_half(aspect):
    # A plane's mean sits exactly midway between its min and max, whatever the tilt.
    px = (30.0, 30.0)
    z = surfaces.inclined_plane((48, 48), pixel_size=px, slope_deg=20.0, aspect_deg=aspect)
    assert metrics.elevation_relief_ratio(z, px) == pytest.approx(0.5, abs=1e-12)


def test_err_of_ramp_is_one_half():
    z = surfaces.ramp((64, 64), low=100.0, high=350.0)
    assert metrics.elevation_relief_ratio(z, (30.0, 30.0)) == pytest.approx(0.5, abs=1e-12)


def test_err_of_plateau_with_pit_is_high():
    # A plateau holed by a pit over 1/16 of its area: almost every pixel sits at
    # the max, so the hypsometric integral is 1 - 1/16 by hand.
    z = surfaces.plateau_with_pit((64, 64), height=250.0, pit_depth=100.0, pit_frac=0.0625)
    err = metrics.elevation_relief_ratio(z, (30.0, 30.0))
    assert err == pytest.approx(0.9375, abs=1e-12)
    assert err > 0.9


def test_err_of_constant_surface_is_undefined():
    # max == min: the ratio's denominator vanishes. Guard, don't divide.
    z = surfaces.constant((16, 16), value=250.0)
    assert np.isnan(metrics.elevation_relief_ratio(z, (30.0, 30.0)))


# --- sinusoid: a frequency sweep ----------------------------------------------


def test_sinusoid_relief_is_twice_its_amplitude():
    px = (30.0, 30.0)
    z = surfaces.sinusoid((32, 32), pixel_size=px, wavelength_m=480.0, amplitude=50.0)
    assert metrics.local_relief(z, px) == pytest.approx(100.0)
    assert metrics.mean_elevation(z, px) == pytest.approx(1000.0)
    assert metrics.elevation_relief_ratio(z, px) == pytest.approx(0.5, abs=1e-12)


def test_sinusoid_slope_steepens_as_the_wavelength_shortens():
    # Same amplitude, tighter corrugations -> steeper terrain. This is the signal
    # the texture metric will ride, and it only exists if pixel_size is honoured.
    px = (30.0, 30.0)
    slopes = [
        metrics.slope_mean(
            surfaces.sinusoid((64, 64), pixel_size=px, wavelength_m=w, amplitude=50.0), px
        )
        for w in (1920.0, 960.0, 480.0, 240.0)
    ]
    assert slopes == sorted(slopes)
    assert slopes[0] < slopes[-1]


def test_sinusoid_steepest_slope_converges_on_the_analytic_value():
    # Continuous wave: max |dz/dx| = 2*pi*A/lambda, at the zero crossings. A
    # finite-difference gradient always reads that low, by less and less as the
    # wave is sampled more finely -- the discretization error a coarse overview
    # pays and A3's calibration measures.
    px = (30.0, 30.0)
    amplitude = 50.0
    errors = []
    for wavelength in (240.0, 480.0, 1920.0):  # 8, 16, 64 pixels per period
        z = surfaces.sinusoid((64, 64), pixel_size=px, wavelength_m=wavelength, amplitude=amplitude)
        analytic = np.degrees(np.arctan(2 * np.pi * amplitude / wavelength))
        steepest = metrics.slope_deg(z, px).max()
        assert steepest < analytic
        errors.append((analytic - steepest) / analytic)
    assert errors == sorted(errors, reverse=True)  # finer sampling, smaller error
    assert errors[-1] < 0.01  # at 64 px per period, within a percent


def test_sinusoid_has_a_nonzero_slope_spread_unlike_a_plane():
    px = (30.0, 30.0)
    wave = surfaces.sinusoid((64, 64), pixel_size=px, wavelength_m=480.0, amplitude=50.0)
    plane = surfaces.inclined_plane((64, 64), pixel_size=px, slope_deg=15.0)
    assert metrics.slope_std(wave, px) > 1.0
    assert metrics.slope_std(plane, px) == pytest.approx(0.0, abs=1e-9)


# --- mean elevation -----------------------------------------------------------


def test_mean_elevation_of_plateau_is_its_height():
    z = surfaces.plateau((32, 32), height=250.0)
    assert metrics.mean_elevation(z, (30.0, 30.0)) == pytest.approx(250.0)


def test_mean_elevation_of_ramp_is_the_midpoint():
    z = surfaces.ramp((64, 64), low=100.0, high=350.0)
    assert metrics.mean_elevation(z, (30.0, 30.0)) == pytest.approx(225.0)


# --- Iwahashi-Pike texture: the sinusoid frequency sweep ----------------------


def test_texture_of_a_plateau_is_zero():
    z = surfaces.plateau((32, 32), height=250.0)
    assert metrics.texture(z, (30.0, 30.0)) == pytest.approx(0.0)


@pytest.mark.parametrize("aspect", [0.0, 90.0, 180.0, 270.0])
def test_texture_of_a_plane_is_zero_however_steep_it_is(aspect):
    # Texture is pits and peaks, not steepness: the median of a monotone
    # neighbourhood is the cell in the middle of it, so nothing is ever marked.
    px = (30.0, 30.0)
    z = surfaces.inclined_plane((48, 48), pixel_size=px, slope_deg=40.0, aspect_deg=aspect)
    assert metrics.texture(z, px) == pytest.approx(0.0)


def test_texture_rises_as_the_corrugations_tighten():
    # The oracle: same amplitude, shorter wavelength -> more pits and peaks per
    # cell. A 1-D wave marks exactly its crest and trough columns, so the density
    # is 2 cells per period -- 2/16, 2/8, 2/4 at 16, 8 and 4 pixels per period.
    px = (30.0, 30.0)
    textures = [
        metrics.texture(
            surfaces.sinusoid((64, 64), pixel_size=px, wavelength_m=w, amplitude=50.0), px
        )
        for w in (480.0, 240.0, 120.0)
    ]
    assert textures[0] < textures[1] < textures[2]
    assert textures[0] == pytest.approx(0.125, abs=0.02)
    assert textures[1] == pytest.approx(0.25, abs=0.02)
    assert textures[2] == pytest.approx(0.5, abs=0.02)


def test_texture_ignores_relief_too_small_for_its_threshold():
    # Iwahashi & Pike count a cell only where it departs from the local median by
    # more than a metre. A wave stretched over 64 pixels bends by 0.24 m per cell
    # at its crest: real relief, but below the noise floor the threshold is there
    # to reject.
    px = (30.0, 30.0)
    z = surfaces.sinusoid((64, 64), pixel_size=px, wavelength_m=1920.0, amplitude=50.0)
    assert metrics.texture(z, px) == pytest.approx(0.0)


def test_texture_is_indifferent_to_the_pixel_size_it_is_given():
    # Same array, coarser pixels: the wave is 4 cells per period either way, so
    # the pit/peak density does not move. Texture counts cells, not metres --
    # which is exactly the scale sensitivity Gate A3 measures.
    px_full, px_ov = (30.0, 30.0), (240.0, 240.0)
    z = surfaces.sinusoid((64, 64), pixel_size=px_full, wavelength_m=120.0, amplitude=50.0)
    assert metrics.texture(z, px_ov) == pytest.approx(metrics.texture(z, px_full))


def test_texture_does_not_invent_pits_at_the_shoreline():
    # The fill is painted, not measured: a flat coastal plateau has no texture.
    # An unmasked median would read the 250 m step down to the sea as a peak
    # along the whole waterline.
    z = surfaces.plateau((32, 32), height=250.0)
    z[:16] = 0.0
    assert metrics.texture(z, (30.0, 30.0)) == pytest.approx(0.0)


# --- normals dispersion: the metric that must not be slope --------------------


@pytest.mark.parametrize("tilt", [0.0, 5.0, 30.0, 60.0, 85.0])
@pytest.mark.parametrize("aspect", [0.0, 45.0, 90.0, 205.0, 330.0])
def test_normals_dispersion_of_any_plane_is_zero_regardless_of_tilt(tilt, aspect):
    # The sharp one. Every normal on a plane points the same way, so they sum to
    # a unit vector and disperse by nothing -- at 5 degrees and at 85. An
    # implementation that has quietly reinvented slope fails here.
    px = (30.0, 30.0)
    z = surfaces.inclined_plane((48, 48), pixel_size=px, slope_deg=tilt, aspect_deg=aspect)
    assert metrics.normals_dispersion(z, px) == pytest.approx(0.0, abs=1e-12)


def test_normals_dispersion_of_a_plane_is_zero_under_anisotropic_pixels():
    px = (30.9, 19.9)
    z = surfaces.inclined_plane((48, 48), pixel_size=px, slope_deg=12.0, aspect_deg=70.0)
    assert metrics.normals_dispersion(z, px) == pytest.approx(0.0, abs=1e-12)


def test_normals_dispersion_separates_rough_from_merely_steep():
    # A cliff face is steep and smooth; corrugations are gentle and rough. Slope
    # ranks these the wrong way round, which is the whole reason this metric is a
    # candidate.
    px = (30.0, 30.0)
    cliff = surfaces.inclined_plane((64, 64), pixel_size=px, slope_deg=60.0)
    ripples = surfaces.sinusoid((64, 64), pixel_size=px, wavelength_m=240.0, amplitude=20.0)
    assert metrics.slope_mean(cliff, px) > metrics.slope_mean(ripples, px)
    assert metrics.normals_dispersion(cliff, px) < metrics.normals_dispersion(ripples, px)


def test_normals_dispersion_rises_as_the_corrugations_tighten():
    px = (30.0, 30.0)
    values = [
        metrics.normals_dispersion(
            surfaces.sinusoid((64, 64), pixel_size=px, wavelength_m=w, amplitude=50.0), px
        )
        for w in (1920.0, 960.0, 480.0, 240.0)
    ]
    assert values == sorted(values)
    assert 0.0 < values[0] < values[-1] < 1.0


def test_normals_dispersion_never_reports_a_negative_spread():
    # Rounding puts the resultant a hair over unit length on a smooth window, and
    # smooth windows are most of them: a manifest column of -2e-16 is a column of
    # noise where the answer is zero.
    px = (30.0, 30.0)
    for tilt in (0.0, 1.0, 10.0, 45.0):
        z = surfaces.inclined_plane((48, 48), pixel_size=px, slope_deg=tilt)
        assert metrics.normals_dispersion(z, px) >= 0.0


def test_normals_dispersion_ignores_the_cliff_against_the_ocean_fill():
    z = surfaces.plateau((32, 32), height=250.0)
    z[:16] = 0.0
    assert metrics.normals_dispersion(z, (30.0, 30.0)) == pytest.approx(0.0, abs=1e-12)


# --- geomorphons: the class map, then the entropy over it ---------------------

_CENTRE = 24  # of the (49, 49) surfaces below: an interior cell, its rays uncut


def _class_at_the_centre(z):
    return metrics.GEOMORPHON_CLASSES[metrics.geomorphons(z, PX)[_CENTRE, _CENTRE]]


def test_geomorphons_call_a_peak_a_peak():
    # The orientation test. The canonical lookup is indexed by (directions that
    # fall away, directions that rise): an apex with all eight falling is
    # (8, 0) -> peak, and reading the table the other way round returns "pit" --
    # numerically plausible, exactly wrong, and what the prototype does.
    dome = surfaces.basin((49, 49), pixel_size=PX, depth=-100.0, radius_m=480.0)
    assert _class_at_the_centre(dome) == "peak"


def test_geomorphons_call_a_pit_a_pit():
    bowl = surfaces.basin((49, 49), pixel_size=PX, depth=100.0, radius_m=480.0)
    assert _class_at_the_centre(bowl) == "pit"


def test_geomorphons_call_a_ridge_crest_a_ridge():
    # Six directions leave the crest downhill; the two along it are level.
    z = surfaces.ridge((49, 49), pixel_size=PX, flank_slope_deg=20.0)
    assert _class_at_the_centre(z) == "ridge"


def test_geomorphons_call_a_valley_floor_a_valley():
    z = surfaces.valley((49, 49), pixel_size=PX, flank_slope_deg=20.0)
    assert _class_at_the_centre(z) == "valley"


def test_geomorphons_call_a_plane_a_slope():
    z = surfaces.inclined_plane((49, 49), pixel_size=PX, slope_deg=20.0, aspect_deg=0.0)
    assert _class_at_the_centre(z) == "slope"


def test_geomorphons_call_a_plateau_flat():
    z = surfaces.plateau((49, 49), height=250.0)
    assert set(np.unique(metrics.geomorphons(z, PX))) == {metrics.GEOMORPHON_CLASSES.index("flat")}


def test_geomorphons_give_the_ocean_fill_no_class_at_all():
    z = surfaces.plateau((32, 32), height=250.0)
    z[:16] = 0.0
    codes = metrics.geomorphons(z, PX)
    assert np.all(codes[:16] == metrics.GEOMORPHON_UNCLASSED)
    assert np.all(codes[16:] == metrics.GEOMORPHON_CLASSES.index("flat"))


def test_geomorphon_entropy_of_a_plateau_is_zero():
    # One class for every cell: no uncertainty, no entropy.
    z = surfaces.plateau((49, 49), height=250.0)
    assert metrics.geomorphon_entropy(z, PX) == pytest.approx(0.0)


def test_geomorphon_entropy_of_a_plateau_is_not_negative_zero():
    # One class is a clean zero, and the negation in Shannon's formula turns it
    # into -0.0. Equal to zero, and still wrong on the face of a table.
    z = surfaces.plateau((49, 49), height=250.0)
    assert not np.signbit(metrics.geomorphon_entropy(z, PX))


def test_geomorphon_entropy_of_a_ridge_is_higher_than_of_a_plateau():
    # The oracle: a ridge is a crest, two flanks and their breaks -- several
    # classes, so a spread-out histogram.
    ridge = surfaces.ridge((49, 49), pixel_size=PX, flank_slope_deg=20.0)
    plateau = surfaces.plateau((49, 49), height=250.0)
    assert metrics.geomorphon_entropy(ridge, PX) > metrics.geomorphon_entropy(plateau, PX)
    assert metrics.geomorphon_entropy(ridge, PX) > 0.0


def test_geomorphon_entropy_cannot_exceed_a_ten_class_histogram():
    px = PX
    for z in (
        surfaces.ridge((49, 49), pixel_size=px, flank_slope_deg=20.0),
        surfaces.valley((49, 49), pixel_size=px, flank_slope_deg=20.0),
        surfaces.basin((49, 49), pixel_size=px, radius_m=480.0),
        surfaces.sinusoid((49, 49), pixel_size=px, wavelength_m=240.0),
    ):
        assert 0.0 <= metrics.geomorphon_entropy(z, px) <= np.log2(10)


def test_geomorphon_entropy_ignores_the_ocean_fill():
    # The land is a flat plateau, so its entropy is zero. Letting the fill in
    # would carve a coastline of spurious ridges and shoulders out of a painted
    # cliff and report a busy, high-entropy window.
    z = surfaces.plateau((32, 32), height=250.0)
    z[:16] = 0.0
    assert metrics.geomorphon_entropy(z, PX) == pytest.approx(0.0)


# --- geomorphons: the arctan hoist is bit-exact, and has to be -----------------
#
# `_sight_line_angles` used to call arctan once per line-of-sight step; it now
# walks the ray in raw slopes and converts the two extremes at the end. That is
# ~2.7x faster on a 224^2 window and was the dominant cost of the whole Phase-C
# scoring pass, so the bar is identity of the class map, not closeness -- ADR-0004
# froze this vector and the catalog's values feed every downstream tier.


def _angles_via_per_step_arctan(z, dr, dc, step):
    """The pre-hoist `_sight_line_angles`: arctan inside the k loop."""
    h, w = z.shape
    highest = np.full(z.shape, -np.inf)
    lowest = np.full(z.shape, np.inf)
    for k in range(1, metrics.GEOMORPHON_LOS_PX + 1):
        r0, r1 = max(0, -dr * k), h - max(0, dr * k)
        c0, c1 = max(0, -dc * k), w - max(0, dc * k)
        if r1 <= r0 or c1 <= c0:
            break
        ahead = z[r0 + dr * k : r1 + dr * k, c0 + dc * k : c1 + dc * k]
        here = z[r0:r1, c0:c1]
        angle = np.arctan((ahead - here) / (step * k))
        np.fmax(highest[r0:r1, c0:c1], angle, out=highest[r0:r1, c0:c1])
        np.fmin(lowest[r0:r1, c0:c1], angle, out=lowest[r0:r1, c0:c1])
    return highest, lowest


def _geomorphon_windows():
    """Windows chosen so that only an exact tie at a comparison boundary could
    separate the two implementations: random, terrain-shaped, flat, coastal (with
    the exact-0.0 ocean fill), single-cell-land, anisotropic, and cliff-steep --
    the last because arctan stops being injective above a slope of about 2."""
    rng = np.random.default_rng(20260720)
    fill = np.zeros((32, 32))
    lone = np.zeros((32, 32))
    lone[16, 16] = 300.0
    coast = surfaces.sinusoid((49, 49), pixel_size=PX, wavelength_m=240.0)
    coast[:, :20] = 0.0
    # Terrain-shaped: summed octaves, then the sub-metre detail a float32 COG
    # carries -- integer-metre test surfaces tie far more often than real data.
    terrain = sum(
        surfaces.sinusoid((64, 64), pixel_size=PX, wavelength_m=wl) * (wl / 240.0)
        for wl in (960.0, 480.0, 240.0, 120.0)
    ) + rng.normal(0.0, 0.4, (64, 64))
    cases = {
        "random": (rng.normal(0.0, 200.0, (64, 64)), PX),
        "terrain": (terrain, PX),
        "terrain-anisotropic": (terrain, (30.9, 19.9)),
        "cliffs": (rng.normal(0.0, 5000.0, (49, 49)), PX),
        "quantised": (np.round(rng.normal(0.0, 3.0, (49, 49))), PX),
        "plateau": (surfaces.plateau((49, 49)), PX),
        "ridge": (surfaces.ridge((49, 49), pixel_size=PX, flank_slope_deg=20.0), PX),
        "valley": (surfaces.valley((49, 49), pixel_size=PX, flank_slope_deg=20.0), PX),
        "basin": (surfaces.basin((49, 49), pixel_size=PX, radius_m=480.0), PX),
        "steep-plane": (
            surfaces.inclined_plane((49, 49), pixel_size=PX, slope_deg=80.0, aspect_deg=17.0),
            PX,
        ),
        "coastal": (coast, PX),
        "single-cell-land": (lone, PX),
        "all-ocean": (fill, PX),
    }
    return cases


_GEOMORPHON_CASES = _geomorphon_windows()


@pytest.mark.parametrize("case", sorted(_GEOMORPHON_CASES))
def test_sight_line_angles_survive_the_arctan_hoist_bit_for_bit(case):
    # arctan is monotone, so it commutes with the running max/min: the arctan of
    # the winning slope is the winning angle, whatever the rounding does. The one
    # difference is the seed for a cell that saw nothing -- arctan carries the
    # +-inf to +-pi/2, which is level under every comparison the same way.
    z, px = _GEOMORPHON_CASES[case]
    masked = metrics._masked_land(np.asarray(z, dtype=np.float64))
    dy, dx = px
    diagonal = np.hypot(dy, dx)
    for dr, dc, step in (
        (-1, 0, dy), (1, 0, dy), (0, -1, dx), (0, 1, dx),
        (-1, -1, diagonal), (-1, 1, diagonal), (1, -1, diagonal), (1, 1, diagonal),
    ):
        new = metrics._sight_line_angles(masked, dr, dc, step)
        old = _angles_via_per_step_arctan(masked, dr, dc, step)
        for got, want in zip(new, old):
            unseen = np.isinf(want)
            assert np.array_equal(got[unseen], np.arctan(want[unseen]))
            assert np.array_equal(got[~unseen], want[~unseen], equal_nan=True)


@pytest.mark.parametrize("case", sorted(_GEOMORPHON_CASES))
def test_geomorphon_class_map_is_identical_after_the_hoist(case):
    # Identical, not close: a single reclassified cell moves the entropy, and the
    # entropy is a frozen catalog column.
    z, px = _GEOMORPHON_CASES[case]
    got = metrics.geomorphons(z, px)

    a = np.asarray(z, dtype=np.float64)
    masked = metrics._masked_land(a)
    dy, dx = px
    diagonal = np.hypot(dy, dx)
    level = np.radians(metrics.GEOMORPHON_FLAT_DEG)
    n_falls = np.zeros(a.shape, dtype=np.int8)
    n_rises = np.zeros(a.shape, dtype=np.int8)
    for dr, dc, step in (
        (-1, 0, dy), (1, 0, dy), (0, -1, dx), (0, 1, dx),
        (-1, -1, diagonal), (-1, 1, diagonal), (1, -1, diagonal), (1, 1, diagonal),
    ):
        highest, lowest = _angles_via_per_step_arctan(masked, dr, dc, step)
        drop = -lowest
        n_rises += ((highest > drop) & (highest > level)).astype(np.int8)
        n_falls += ((drop > highest) & (drop > level)).astype(np.int8)
    want = np.where(
        metrics.land_mask(a),
        metrics._LANDFORM_LOOKUP[n_falls, n_rises],
        metrics.GEOMORPHON_UNCLASSED,
    ).astype(np.int8)

    assert np.array_equal(got, want)


def test_arctan_commutes_with_the_running_extreme():
    # The property the hoist rests on, asserted on the implementation rather than
    # assumed of it: np.arctan must never reorder two floats. IEEE does not
    # promise a monotone arctan, so if a numpy or libm build ever breaks it, this
    # fails here rather than silently in a catalog column.
    rng = np.random.default_rng(7)
    for scale in (1e-3, 1e-1, 1.0, 3.0, 1e2, 1e6):
        s = rng.normal(0.0, scale, (2048, 24))
        assert np.array_equal(np.arctan(s.max(axis=1)), np.arctan(s).max(axis=1))
        assert np.array_equal(np.arctan(s.min(axis=1)), np.arctan(s).min(axis=1))


def test_raw_slopes_would_reclassify_a_cell_so_the_arctan_stays():
    # Why the conversion is hoisted and not deleted. Every comparison downstream
    # is monotone in the angle, which makes raw slopes look equivalent and ~1.4x
    # faster again -- but arctan is not *injective* on float64: above a slope of
    # ~2 (63 deg) neighbouring floats share an angle. Here the steepest rise and
    # the steepest fall along the southward ray land one ulp apart, so the angles
    # tie and the slopes do not, and the tie is the difference between "neither
    # rises nor falls" and "falls".
    dy, rise, drop_at_k3 = 30.9, 63.0, 189.0
    z = np.full((40, 40), 1000.0)
    r = c = 20
    for k in range(1, metrics.GEOMORPHON_LOS_PX + 1):
        z[r + k, c] = 1000.0 + rise
        z[r, c + k] = z[r, c - k] = 500.0  # two clean falls, onto a lookup boundary
    z[r + 3, c] = 1000.0 - drop_at_k3

    step = dy
    steepest_rise = rise / step
    steepest_fall = drop_at_k3 / (step * 3)
    assert steepest_rise != steepest_fall
    assert np.arctan(steepest_rise) == np.arctan(steepest_fall)

    codes = metrics.geomorphons(z, (dy, dy))
    assert metrics.GEOMORPHON_CLASSES[codes[r, c]] == "flat"  # the tie holds
    # and on raw slopes the same cell reads (3, 0) -> shoulder.
    assert metrics._LANDFORM_LOOKUP[3, 0] == metrics.GEOMORPHON_CLASSES.index("shoulder")


# --- TPI-proxy: the reserve metric --------------------------------------------


@pytest.mark.parametrize("aspect", [0.0, 45.0, 90.0, 205.0])
@pytest.mark.parametrize("shape", [(64, 64), (49, 49)])
def test_tpi_proxy_of_any_plane_is_zero(aspect, shape):
    # The sharp one, and what a concentric core buys: a plane's middle sits as far
    # under its uphill half as it stands over its downhill half, so the two cancel
    # at any tilt and any aspect. A core off by half a cell would read the tilt as
    # position, so the odd-sided window is here on purpose.
    z = surfaces.inclined_plane(shape, pixel_size=PX, slope_deg=20.0, aspect_deg=aspect)
    assert metrics.tpi_proxy(z, PX) == pytest.approx(0.0, abs=1e-9)


def test_tpi_proxy_of_a_plateau_is_zero():
    z = surfaces.plateau((64, 64), height=250.0)
    assert metrics.tpi_proxy(z, PX) == pytest.approx(0.0, abs=1e-9)


def test_tpi_proxy_reads_a_basin_as_below_its_surroundings():
    # The oracle: the floor of a bowl sits under the rim around it, so the reading
    # is negative; turned over into a dome, the same surface reads positive.
    bowl = surfaces.basin((64, 64), pixel_size=PX, depth=100.0, radius_m=960.0)
    dome = surfaces.basin((64, 64), pixel_size=PX, depth=-100.0, radius_m=960.0)
    assert metrics.tpi_proxy(bowl, PX) < 0.0
    assert metrics.tpi_proxy(dome, PX) > 0.0
    assert metrics.tpi_proxy(dome, PX) == pytest.approx(-metrics.tpi_proxy(bowl, PX))


def test_tpi_proxy_tells_a_basin_floor_from_a_ridge_crest():
    # The distinction B3's flat-wet check is made of: the low ground and the high
    # ground, told apart by position rather than by shape.
    valley = surfaces.valley((64, 64), pixel_size=PX, flank_slope_deg=20.0)
    ridge = surfaces.ridge((64, 64), pixel_size=PX, flank_slope_deg=20.0)
    assert metrics.tpi_proxy(valley, PX) < 0.0 < metrics.tpi_proxy(ridge, PX)


def test_tpi_proxy_deepens_with_the_basin():
    # TPI is linear in elevation, so twice the bowl is exactly twice the reading.
    shallow = surfaces.basin((64, 64), pixel_size=PX, depth=100.0, radius_m=960.0)
    deep = surfaces.basin((64, 64), pixel_size=PX, depth=200.0, radius_m=960.0)
    assert metrics.tpi_proxy(deep, PX) == pytest.approx(2.0 * metrics.tpi_proxy(shallow, PX))


def test_tpi_proxy_cannot_see_a_basin_wider_than_the_window():
    # The documented blind spot: from inside a basin ten times the width of the
    # window, the window is nearly flat, and the metric says so. B3 has to read a
    # null here as "not visible at this scale", not as "not there".
    broad = surfaces.basin((64, 64), pixel_size=PX, depth=100.0, radius_m=9600.0)
    tight = surfaces.basin((64, 64), pixel_size=PX, depth=100.0, radius_m=960.0)
    assert abs(metrics.tpi_proxy(broad, PX)) < 0.01 * abs(metrics.tpi_proxy(tight, PX))


def test_tpi_proxy_tells_a_flat_floor_from_a_plateau_only_with_the_surroundings():
    # The plan's actual formula, and B3's flat-wet check in miniature. Flat is not
    # one terrain: the low wet ground and the high dry ground are both featureless
    # inside the window, and only what surrounds them tells them apart.
    #
    # A 128-cell array holding a flat-floored basin exactly one 64-cell window
    # wide, and a plateau. Scored on the window alone, the two are the same
    # reading -- not a bug, and not the terrain's fault. Scored on the array that
    # holds the rim as well, the window *is* the core, the frame is the ground
    # around it, and the floor sits exactly its 100 m below. Same metric, same
    # contract; the reach came in through the array argument.
    floor = surfaces.plateau_with_pit((128, 128), height=250.0, pit_depth=100.0, pit_frac=0.25)
    plateau = surfaces.plateau((128, 128), height=250.0)
    window = (slice(32, 96), slice(32, 96))

    assert metrics.tpi_proxy(floor[window], PX) == pytest.approx(0.0)
    assert metrics.tpi_proxy(plateau[window], PX) == pytest.approx(0.0)

    assert metrics.tpi_proxy(floor, PX) == pytest.approx(-100.0)
    assert metrics.tpi_proxy(plateau, PX) == pytest.approx(0.0)


def test_tpi_proxy_ignores_the_ocean_fill():
    # The land is flat and level with itself; the fill is 250 m below it. Averaged
    # in, it would report the whole coast as a plateau standing above the sea --
    # true of the painted file, meaningless as terrain.
    z = surfaces.plateau((64, 64), height=250.0)
    z[:32] = 0.0
    assert metrics.tpi_proxy(z, PX) == pytest.approx(0.0, abs=1e-9)


def test_tpi_proxy_is_undefined_when_the_window_has_no_frame_to_compare_against():
    z = surfaces.basin((64, 64), pixel_size=PX, depth=100.0, radius_m=960.0)
    z[:16] = z[48:] = z[:, :16] = z[:, 48:] = 0.0  # sea everywhere but the core
    assert np.isnan(metrics.tpi_proxy(z, PX))


# --- the mask contract, binding on every metric (ADR-0001 §3) -----------------


def test_mean_elevation_ignores_ocean_fill():
    # Half plateau, half ocean fill: the land-only mean is the plateau height,
    # not the 125.0 an unmasked mean would report.
    z = surfaces.plateau((32, 32), height=250.0)
    z[:16] = 0.0
    assert metrics.mean_elevation(z, (30.0, 30.0)) == pytest.approx(250.0)


def test_relief_and_err_ignore_ocean_fill():
    # Ocean fill sits below the ramp's floor; unmasked it would stretch the range
    # from 250 m to 350 m and drag the ratio off 0.5.
    z = surfaces.ramp((64, 64), low=100.0, high=350.0)
    z[:8] = 0.0
    assert metrics.local_relief(z, (30.0, 30.0)) == pytest.approx(250.0 * (55 / 63), rel=1e-9)
    assert metrics.elevation_relief_ratio(z, (30.0, 30.0)) == pytest.approx(0.5, abs=1e-12)


def test_slope_ignores_the_cliff_against_the_ocean_fill():
    # A flat coastal plateau is flat. The 250 m step down to the fill is a
    # digitization artifact, not terrain: the fill is painted, not measured.
    z = surfaces.plateau((32, 32), height=250.0)
    z[:16] = 0.0
    assert metrics.slope_mean(z, (30.0, 30.0)) == pytest.approx(0.0, abs=1e-9)
    assert metrics.slope_std(z, (30.0, 30.0)) == pytest.approx(0.0, abs=1e-9)


def test_coastal_slope_does_not_depend_on_the_resolution_it_is_read_at():
    # The artifact this guards against is resolution-dependent -- a one-pixel
    # step is steeper at 30 m than at 240 m -- so leaving it in would show up as
    # drift in A3's overview-vs-full-res calibration and be read as real.
    z = surfaces.inclined_plane((32, 32), pixel_size=(30.0, 30.0), slope_deg=8.0)
    z[:8] = 0.0  # sea in the north
    full = metrics.slope_mean(z, (30.0, 30.0))
    assert full == pytest.approx(8.0, abs=1e-9)


def test_sub_sea_level_land_is_not_mistaken_for_ocean():
    # ADR-0001 §3: genuine land can be negative (the Netherlands at -12 m); only
    # exactly 0.0 is ocean.
    z = surfaces.constant((16, 16), value=-12.0)
    assert metrics.mean_elevation(z, (30.0, 30.0)) == pytest.approx(-12.0)


def test_err_guard_fires_when_masking_is_what_flattens_the_window():
    # The window is not constant, but its *land* is: the guard has to look at
    # what survives the mask, not at the raw array.
    z = surfaces.plateau((32, 32), height=250.0)
    z[:16] = 0.0
    assert np.isnan(metrics.elevation_relief_ratio(z, (30.0, 30.0)))
    assert metrics.local_relief(z, (30.0, 30.0)) == pytest.approx(0.0)


@pytest.mark.parametrize("name", sorted(metrics.METRICS))
def test_every_metric_returns_nan_for_an_all_ocean_window(name):
    z = surfaces.constant((16, 16), value=0.0)
    assert np.isnan(metrics.METRICS[name](z, (30.0, 30.0)))


@pytest.mark.parametrize("name", sorted(metrics.METRICS))
def test_every_metric_honours_the_array_plus_pixel_size_contract(name):
    # A3's calibration loops over the registry: same call, full-res and overview.
    px_full, px_ov = (30.0, 30.0), (240.0, 240.0)
    z = surfaces.inclined_plane((32, 32), pixel_size=px_full, slope_deg=10.0)
    for px in (px_full, px_ov):
        assert isinstance(metrics.METRICS[name](z, px), float)
