"""Synthetic oracle surfaces — that each one really is the shape it claims.

The metric tests lean on these surfaces as their source of truth, so the closed
forms are pinned here against hand-computed literals rather than against the
metric library that consumes them.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset import surfaces


def test_ramp_spans_its_range_along_the_named_axis():
    z = surfaces.ramp((4, 3), low=10.0, high=40.0, axis=0)
    assert z.shape == (4, 3)
    assert z[:, 0] == pytest.approx([10.0, 20.0, 30.0, 40.0])
    assert np.allclose(z[0], 10.0)  # constant across the other axis

    z = surfaces.ramp((3, 4), low=10.0, high=40.0, axis=1)
    assert z[0] == pytest.approx([10.0, 20.0, 30.0, 40.0])


def test_inclined_plane_falls_towards_its_aspect():
    px = (30.0, 30.0)
    # Aspect 180 (due south) -> elevation drops as rows increase southward.
    south = surfaces.inclined_plane((8, 8), pixel_size=px, slope_deg=10.0, aspect_deg=180.0)
    assert south[-1, 0] < south[0, 0]
    assert south[0, 0] == pytest.approx(south[0, -1])  # flat across the strike

    # Aspect 90 (due east) -> elevation drops as columns increase eastward.
    east = surfaces.inclined_plane((8, 8), pixel_size=px, slope_deg=10.0, aspect_deg=90.0)
    assert east[0, -1] < east[0, 0]
    assert east[0, 0] == pytest.approx(east[-1, 0])


def test_inclined_plane_rise_over_run_is_the_tangent_of_its_tilt():
    px = (30.0, 30.0)
    z = surfaces.inclined_plane((8, 8), pixel_size=px, slope_deg=30.0, aspect_deg=180.0)
    drop_per_row = z[0, 0] - z[1, 0]
    assert drop_per_row == pytest.approx(np.tan(np.radians(30.0)) * 30.0)


def test_plateau_is_flat_at_its_height():
    z = surfaces.plateau((8, 8), height=250.0)
    assert np.all(z == 250.0)


def test_plateau_with_pit_carves_the_named_area_fraction():
    z = surfaces.plateau_with_pit((64, 64), height=250.0, pit_depth=100.0, pit_frac=0.0625)
    assert z.max() == pytest.approx(250.0)
    assert z.min() == pytest.approx(150.0)
    # 0.0625 of 4096 px is a 16x16 pit -- exact, no rounding.
    assert surfaces.pit_fraction((64, 64), pit_frac=0.0625) == pytest.approx(0.0625)
    assert np.mean(z == 150.0) == pytest.approx(0.0625)


def test_pit_fraction_reports_the_rounding_when_the_pit_is_not_a_whole_square():
    # 0.1 of 4096 px wants a 20.24-px side; the achieved pit is 20x20.
    assert surfaces.pit_fraction((64, 64), pit_frac=0.1) == pytest.approx(400 / 4096)


def test_pit_fraction_rejects_a_fraction_outside_the_unit_interval():
    with pytest.raises(ValueError, match="pit_frac"):
        surfaces.plateau_with_pit((8, 8), pit_frac=1.5)


def test_sinusoid_has_the_wavelength_it_was_given():
    px = (30.0, 30.0)
    # 480 m wavelength at 30 m pixels = one full period every 16 columns.
    z = surfaces.sinusoid((32, 32), pixel_size=px, wavelength_m=480.0, amplitude=50.0)
    assert z.shape == (32, 32)
    assert np.allclose(z[:, 0], z[:, 16])  # exactly one period apart
    assert z.max() == pytest.approx(1050.0)  # base + amplitude, crest sampled
    assert z.min() == pytest.approx(950.0)
    assert z.mean() == pytest.approx(1000.0)  # whole periods integrate to base


def test_ridge_is_a_tent_falling_away_from_its_crest():
    px = (30.0, 30.0)
    z = surfaces.ridge((5, 4), pixel_size=px, flank_slope_deg=45.0, crest_height=1000.0, axis=0)
    # 5 rows: the crest is row 2, and 45 deg falls exactly one pixel-width per pixel.
    assert z[:, 0] == pytest.approx([940.0, 970.0, 1000.0, 970.0, 940.0])
    assert np.allclose(z[2], 1000.0)  # the crest runs across the other axis
    assert z.max() == pytest.approx(1000.0)


def test_valley_is_a_ridge_turned_upside_down():
    px = (30.0, 30.0)
    z = surfaces.valley((5, 4), pixel_size=px, flank_slope_deg=45.0, floor_height=1000.0, axis=0)
    assert z[:, 0] == pytest.approx([1060.0, 1030.0, 1000.0, 1030.0, 1060.0])
    assert z.min() == pytest.approx(1000.0)


def test_ridge_reads_the_pixel_size_it_is_given():
    # The flank tilt is a *ground* angle, so a coarser pixel means a taller ridge.
    fine = surfaces.ridge((9, 9), pixel_size=(30.0, 30.0), flank_slope_deg=30.0)
    coarse = surfaces.ridge((9, 9), pixel_size=(240.0, 240.0), flank_slope_deg=30.0)
    assert (coarse.max() - coarse.min()) == pytest.approx(8 * (fine.max() - fine.min()))


def test_basin_is_a_paraboloid_bowl_around_its_floor():
    px = (30.0, 30.0)
    z = surfaces.basin((65, 65), pixel_size=px, depth=100.0, radius_m=960.0, floor=1000.0)
    assert z[32, 32] == pytest.approx(1000.0)  # the floor is the centre cell
    # 32 columns east of centre is 960 m, exactly one radius: floor + depth * 1**2.
    assert z[32, 64] == pytest.approx(1100.0)
    assert z[64, 32] == pytest.approx(1100.0)
    assert z.min() == pytest.approx(1000.0)


def test_basin_with_a_negative_depth_is_a_dome():
    px = (30.0, 30.0)
    z = surfaces.basin((65, 65), pixel_size=px, depth=-100.0, radius_m=960.0, floor=1000.0)
    assert z[32, 32] == pytest.approx(1000.0)  # now the apex
    assert z[32, 64] == pytest.approx(900.0)
    assert z.max() == pytest.approx(1000.0)


def test_basin_is_round_on_the_ground_not_in_pixels():
    # Anisotropic pixels: the bowl's contours are circles in metres, so a column
    # step of 19.9 m climbs less than a row step of 30.9 m.
    z = surfaces.basin((65, 65), pixel_size=(30.9, 19.9), depth=100.0, radius_m=960.0)
    assert z[33, 32] > z[32, 33]


def test_constant_surface_is_degenerate():
    z = surfaces.constant((8, 8), value=250.0)
    assert z.max() == z.min() == 250.0


def test_every_surface_is_writable_so_callers_can_paint_ocean_into_it():
    # Painting sea into a surface is how the coastal cases get built; a surface
    # handed back as a read-only broadcast view would make that impossible.
    px = (30.0, 30.0)
    for z in _every_default_surface(px):
        z[0, 0] = 0.0  # must not raise


def test_no_surface_accidentally_paints_ocean_fill():
    # Every surface must stay clear on *its own defaults* of the exact 0.0 the
    # mask reserves for ocean, or the oracles silently lose pixels (ADR-0001 §3):
    # a ramp defaulting to low=0.0 would report 98.4 m of relief, not 100.
    for z in _every_default_surface((30.0, 30.0)):
        assert not np.any(z == 0.0)


def _every_default_surface(px):
    """Each surface built on its own defaults, so the defaults are what's tested."""
    return [
        surfaces.ramp(),
        surfaces.inclined_plane(pixel_size=px, slope_deg=20.0, aspect_deg=45.0),
        surfaces.plateau(),
        surfaces.plateau_with_pit(),
        surfaces.sinusoid(pixel_size=px),
        surfaces.ridge(pixel_size=px),
        surfaces.valley(pixel_size=px),
        surfaces.basin(pixel_size=px),
        surfaces.constant(),
    ]
