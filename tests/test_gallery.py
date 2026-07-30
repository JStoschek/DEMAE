"""E2.4 gallery: the draws, the radius, and the shading transform (issue #53).

The sheet itself is a human artifact and untestable by construction — what is tested
here is everything the reviewer's judgement rests *on top of*: that the 240 tiles are
recoverable from the seed, that "interior vs edge" is measured in the space the strata
were cut in, and that the shading transform does what its caption claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset import gallery, geometry
from dataset.centroids import Centroids
from dataset.frame import METRICS, Scaler


# --- The draws ----------------------------------------------------------------


def _ids(n: int) -> list[str]:
    return [f"tile_{i:04d}" for i in range(n)]


def test_uniform_draw_is_reproducible_from_the_seed():
    ids = _ids(500)
    assert gallery.uniform_draw(ids) == gallery.uniform_draw(ids)


def test_uniform_draw_is_independent_of_input_order():
    """The protocol's reproducibility rule has to survive a re-sorted manifest."""
    ids = _ids(300)
    forward = {ids[i] for i in gallery.uniform_draw(ids)}
    shuffled = list(reversed(ids))
    backward = {shuffled[i] for i in gallery.uniform_draw(shuffled)}
    assert forward == backward


def test_uniform_draw_changes_with_the_seed():
    ids = _ids(500)
    a = {ids[i] for i in gallery.uniform_draw(ids, seed=1)}
    b = {ids[i] for i in gallery.uniform_draw(ids, seed=2)}
    assert a != b


def test_uniform_draw_takes_twelve_by_default():
    assert len(gallery.uniform_draw(_ids(500))) == gallery.TILES_PER_SHEET


def test_uniform_draw_of_a_short_stratum_returns_everything():
    """n_h below the sheet size must not raise — it just makes a shorter sheet."""
    assert sorted(gallery.uniform_draw(_ids(5))) == [0, 1, 2, 3, 4]


def test_spanning_draw_walks_interior_to_edge():
    ids = _ids(120)
    radii = np.linspace(0.0, 10.0, 120)
    picks = gallery.spanning_draw(ids, radii)
    chosen = radii[picks]
    assert len(picks) == gallery.TILES_PER_SHEET
    assert np.all(np.diff(chosen) > 0), "sheet must read centre-outward"
    assert chosen[0] < 1.0 and chosen[-1] > 9.0, "both ends of the stratum sampled"


def test_spanning_draw_uses_equal_count_bins_not_equal_width():
    """A long radius tail must not eat most of the sheet."""
    ids = _ids(100)
    radii = np.concatenate([np.full(90, 1.0) + np.arange(90) * 1e-3, np.linspace(5.0, 50.0, 10)])
    chosen = radii[gallery.spanning_draw(ids, radii)]
    # 90% of the stratum sits near r=1, so most bins — and most tiles — must land there.
    assert (chosen < 2.0).sum() >= 9


def test_spanning_draw_is_reproducible_and_rejects_a_length_mismatch():
    ids = _ids(60)
    radii = np.linspace(0, 3, 60)
    assert gallery.spanning_draw(ids, radii) == gallery.spanning_draw(ids, radii)
    with pytest.raises(ValueError):
        gallery.spanning_draw(ids, radii[:-1])


# --- Radius in the space the strata were cut in -------------------------------


def _scaler(mean=None, std=None) -> Scaler:
    n = len(METRICS)
    return Scaler(
        mean=np.zeros(n) if mean is None else np.asarray(mean, float),
        std=np.ones(n) if std is None else np.asarray(std, float),
        fingerprint={},
    )


def test_centroid_radius_is_zero_at_the_centroid():
    """A row sitting exactly on its centroid has radius 0 — through the real transform."""
    cols = {m: np.array([1.0]) for m in METRICS}
    scaler = _scaler()
    # metric_matrix applies ln(x+1) to the two log metrics; the centroid must be built
    # in that same space, which is the whole point of going through the module.
    z = np.array([[np.log(2.0), np.log(2.0), 1.0, 1.0, 1.0, 1.0]])
    centroids = Centroids(centers=z, provenance={}, fingerprint={})
    r = gallery.centroid_radius(cols, np.array([0]), scaler=scaler, centroids=centroids)
    assert r[0] == pytest.approx(0.0, abs=1e-12)


def test_centroid_radius_uses_log1p_not_log():
    """The bug that would produce plausible wrong numbers instead of an error.

    ``local_relief`` is one of ADR-0004's ``ln(x + 1)`` metrics. A centroid placed at
    ``ln(x)`` rather than ``ln(x + 1)`` still yields a finite, ordinary-looking radius —
    so this pins the transform rather than trusting it.
    """
    x = 99.0
    cols = {m: np.array([x if m == "local_relief" else 0.0]) for m in METRICS}
    centre = np.zeros((1, len(METRICS)))
    centre[0, METRICS.index("local_relief")] = np.log(x + 1.0)
    centroids = Centroids(centers=centre, provenance={}, fingerprint={})
    r = gallery.centroid_radius(cols, np.array([0]), scaler=_scaler(), centroids=centroids)
    assert r[0] == pytest.approx(0.0, abs=1e-12)
    # ln(x) would have left a radius of ln(100) - ln(99) ~ 0.01 — small, and silent.
    assert r[0] < 1e-9


def test_centroid_radius_measures_against_each_rows_own_stratum():
    cols = {m: np.zeros(2) for m in METRICS}
    centers = np.zeros((2, len(METRICS)))
    centers[1, 2] = 5.0  # stratum 1's centroid is far from the origin
    centroids = Centroids(centers=centers, provenance={}, fingerprint={})
    r = gallery.centroid_radius(cols, np.array([0, 1]), scaler=_scaler(), centroids=centroids)
    assert r[0] == pytest.approx(0.0)
    assert r[1] == pytest.approx(5.0)


def test_radius_percentile_is_per_group_and_centre_is_zero():
    radii = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    member = np.array([0, 0, 0, 1, 1])
    pct = gallery.radius_percentile(radii, member)
    assert pct[0] == pytest.approx(0.0)  # innermost of stratum 0
    assert pct[2] == pytest.approx(100.0)  # outermost of stratum 0
    assert pct[3] == pytest.approx(0.0)  # innermost of stratum 1, not of everything
    assert pct[4] == pytest.approx(100.0)


def test_radius_percentile_handles_a_singleton_group():
    pct = gallery.radius_percentile(np.array([7.0]), np.array([3]))
    assert pct.tolist() == [0.0]


# --- Shading ------------------------------------------------------------------


def test_gaussian_kernel_is_normalized_and_symmetric():
    k = gallery.gaussian_kernel1d(1.0)
    assert k.sum() == pytest.approx(1.0)
    assert k == pytest.approx(k[::-1])
    assert k.size == 7  # radius ceil(3 * 1)


def test_gaussian_kernel_rejects_a_nonpositive_sigma():
    with pytest.raises(ValueError):
        gallery.gaussian_kernel1d(0.0)


def test_smooth_preserves_a_constant_field():
    """Reflected padding, not zero padding — a flat tile must not grow a rim."""
    a = np.full((32, 32), 500.0)
    out = gallery.smooth(a, 1.0)
    assert out == pytest.approx(a)


def test_smooth_attenuates_single_pixel_noise_more_than_broad_structure():
    """The claim the sheets rest on: the blur eats the quantum, not the landform."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64]
    broad = 50.0 * np.sin(2 * np.pi * xx / 64.0)
    noise = rng.normal(0.0, 1.0, (64, 64))  # ~ the int16 metre quantum
    s_broad = gallery.smooth(broad, 1.0)
    s_noise = gallery.smooth(noise, 1.0)
    assert s_noise.std() < 0.45 * noise.std()
    assert s_broad.std() > 0.98 * broad.std()


@pytest.mark.parametrize(
    "land_std, expected",
    [
        (60.0, 1.0),
        (6.0, 10.0),
        (0.0, gallery.VE_MAX),
        (1e9, gallery.VE_MIN),
        # The corpus's ruggedest tile has a land std of ~613 m, so the floor is live at
        # the top of the range: the rule alone would ask for ~0.1 and gets 0.2 instead.
        (600.0, gallery.VE_MIN),
    ],
)
def test_auto_vert_exag_targets_the_land_std_and_clamps(land_std, expected):
    assert gallery.auto_vert_exag(land_std) == pytest.approx(expected)


def test_hillshade_is_bounded_and_flat_terrain_is_uniform():
    flat = np.zeros((16, 16))
    hs = gallery.hillshade(flat, dy=30.0, dx=30.0)
    assert hs.min() >= 0.0 and hs.max() <= 1.0
    assert hs.std() == pytest.approx(0.0)


def test_hillshade_matches_matplotlib_lightsource():
    """The sheets were approved from a LightSource prototype; this pins the match.

    Not a redundant test of someone else's library: the shading had to be reimplemented
    in NumPy because matplotlib is dev-only and no ``src/dataset`` module may import it,
    so "same picture" is a claim this module makes and must be held to. It caught the
    real bug — a missing row-axis sign flip that mirrored the lighting north-south.
    """
    mpl_colors = pytest.importorskip("matplotlib.colors")
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 50.0, (24, 24)).cumsum(0).cumsum(1)
    ls = mpl_colors.LightSource(azdeg=gallery.AZIMUTH_DEG, altdeg=gallery.ALTITUDE_DEG)
    for dy, dx, ve in [(30.0, 30.0, 1.0), (31.0, 20.0, 2.5), (30.0, 30.0, 0.3)]:
        assert gallery.hillshade(z, dy=dy, dx=dx, vert_exag=ve) == pytest.approx(
            ls.hillshade(z, vert_exag=ve, dx=dx, dy=dy), abs=1e-12
        )


def test_unstretched_hillshade_keeps_a_flatter_surface_flatter():
    """What ``stretch=False`` buys #98: relief and apparent contrast stop being independent.

    Stretched, both fields below come back full-range and look alike — which is right for a
    Gate-E coherence sheet and wrong for plan §10.2's grids, where a *smeared* reconstruction
    beside its own truth is the failure being looked for.
    """
    _, xx = np.mgrid[0:32, 0:32].astype(float)
    rough, smooth_ = 20.0 * np.sin(xx), 1.0 * np.sin(xx)
    kwargs = {"dy": 30.0, "dx": 30.0}
    assert gallery.hillshade(smooth_, **kwargs, stretch=False).std() < 0.2 * gallery.hillshade(
        rough, **kwargs, stretch=False
    ).std()
    # ... where the stretched pair are all but indistinguishable in spread.
    stretched = [gallery.hillshade(z, **kwargs).std() for z in (rough, smooth_)]
    assert stretched[1] == pytest.approx(stretched[0], rel=0.1)


def test_hillshade_lights_north_facing_slopes_brightest():
    """The sign flip that keeps ridges from reading as valleys, in its own right.

    Row 0 is north, so a surface *descending* northward faces the light at azimuth 315
    and must come out brighter than one descending southward. Tilts on a single axis:
    a north-west diagonal cancels against the light vector and shades both ways alike.
    """
    yy, _ = np.mgrid[0:32, 0:32].astype(float)
    descends_north, descends_south = yy, -yy
    assert (
        gallery.hillshade(descends_north, dy=30.0, dx=30.0).mean()
        > gallery.hillshade(descends_south, dy=30.0, dx=30.0).mean()
    )


def test_hillshade_respects_pixel_anisotropy():
    """A ramp running east is shaded differently when columns are narrower on the ground."""
    _, xx = np.mgrid[0:32, 0:32].astype(float)
    ramp = 10.0 * xx
    wide = gallery.hillshade(ramp, dy=30.0, dx=30.0)
    narrow = gallery.hillshade(ramp, dy=30.0, dx=20.0)
    assert not np.allclose(wide.mean(), narrow.mean())


def test_shade_tile_paints_ocean_and_reports_its_transform():
    tile = np.zeros((32, 32), dtype=np.int16)
    tile[:, 16:] = 200  # half ocean fill, half a 200 m plateau
    rgb, ve, (dy, dx) = gallery.shade_tile(tile, lat=0.0)

    assert rgb.shape == (32, 32, 3)
    assert rgb[0, 0] == pytest.approx(np.asarray(gallery.OCEAN_RGB))
    assert not np.allclose(rgb[0, 31], np.asarray(gallery.OCEAN_RGB))
    assert gallery.VE_MIN <= ve <= gallery.VE_MAX
    assert (dy, dx) == pytest.approx(geometry.band_pixel_spacing_m(0.0))


def test_shade_tile_fills_ocean_before_shading_so_the_coast_is_not_a_cliff():
    """Land beside a coastline must stay legible rather than being blown out by glare."""
    yy, xx = np.mgrid[0:64, 0:64]
    land = (800.0 + 20.0 * np.sin(2 * np.pi * xx / 16.0)).astype(np.int16)
    coastal, inland = land.copy(), land.copy()
    coastal[:, :16] = 0  # ADR-0001 §3 ocean fill

    rgb_coastal, _, _ = gallery.shade_tile(coastal, lat=0.0)
    rgb_inland, _, _ = gallery.shade_tile(inland, lat=0.0)
    # The corrugation on the land half must survive at close to its inland contrast; an
    # unfilled 800 m coastal step would dominate the stretch and flatten it.
    contrast = rgb_coastal[:, 32:, 0].std() / rgb_inland[:, 32:, 0].std()
    assert contrast > 0.8


def test_shade_tile_makes_a_flat_stratum_legible():
    """Why the module exists: 20 m of relief has to render as landform, not grey."""
    yy, xx = np.mgrid[0:64, 0:64]
    faint = (100.0 + 10.0 * np.sin(2 * np.pi * xx / 32.0)).astype(np.int16)
    rgb, ve, _ = gallery.shade_tile(faint, lat=0.0)
    assert ve > 1.0, "a low-relief tile must be exaggerated, not drawn 1:1"
    assert rgb[..., 0].std() > 0.1, "and the result must actually have visible structure"


# --- The band table the gallery introduced ------------------------------------


def test_band_pixel_spacing_matches_an_explicit_lookup():
    for lat, cols in [(10.0, 3600), (55.0, 2400), (65.0, 1800), (75.0, 1200), (82.0, 720), (87.0, 360)]:
        assert geometry.band_pixel_spacing_m(lat) == pytest.approx(
            geometry.pixel_spacing_m(lat, rows_per_degree=3600, cols_per_degree=cols)
        )


def test_band_pixel_spacing_covers_every_band_label():
    assert set(geometry.COLS_PER_DEGREE) == {
        geometry.lat_band(lat) for lat in (0, 25, 55, 65, 75, 82, 87, 90)
    }


def test_columns_are_wider_on_the_ground_than_rows_everywhere():
    """The anisotropy that makes square-pixel shading wrong (ADR-0001 §1)."""
    for lat in (0.0, 30.0, 49.0, 55.0, 65.0, 75.0, 82.0, 87.0):
        dy, dx = geometry.band_pixel_spacing_m(lat)
        assert dx <= dy + 1e-9, f"columns should not be finer than rows at {lat}"
