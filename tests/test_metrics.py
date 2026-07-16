"""Analytic-oracle metrics — each asserted against a synthetic surface whose
expected value is known in closed form (plan A2, issue #16)."""

from __future__ import annotations

import numpy as np
import pytest

from dataset import metrics, surfaces


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


def test_sub_sea_level_land_is_not_mistaken_for_ocean():
    # ADR-0001 §3: genuine land can be negative (the Netherlands at -12 m); only
    # exactly 0.0 is ocean.
    z = surfaces.constant((16, 16), value=-12.0)
    assert metrics.mean_elevation(z, (30.0, 30.0)) == pytest.approx(-12.0)


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
