"""Deterministic global COG sampler — bands, geocells, quotas, reproducibility."""

from __future__ import annotations

from collections import Counter

import pytest

from dataset import config, sampling
from dataset.geometry import CogRef, parse_cog_key


def _ref(lat: int, lon: int) -> CogRef:
    """A CogRef from a real Copernicus stem, by 1-degree south-west origin."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return parse_cog_key(
        f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
    )


# Band populations of the real released corpus (26,450 COGs, measured against
# s3://copernicus-dem-30m). The poles are cell-dense, not cell-thin: above 50 degrees
# is 55% of the cells and under a third of the land area. Pinned here so the sampler's
# behaviour on the shape it actually ships against is a test, not a claim.
REAL_BAND_COUNTS = {
    "0-50": 11_861, "50-60": 2_459, "60-70": 3_582,
    "70-80": 4_552, "80-85": 2_196, "85-90": 1_800,
}


def _frame() -> list[CogRef]:
    """A deliberately pole-thin frame (360 / 40 / 20 / 8 / 4 / 2 = 434 cells).

    Inverted from the real corpus on purpose: thin tail bands are what exercise the
    allocator's rounding, ceiling, and floor edges. ``REAL_BAND_COUNTS`` covers the
    real shape.
    """
    refs: list[CogRef] = []
    for lat in range(0, 40, 2):  # 20 lats x 18 lons = 360
        for lon in range(-180, 180, 20):
            refs.append(_ref(lat, lon))
    for lat in range(50, 58, 2):  # 4 x 10 = 40
        for lon in range(-180, 180, 36):
            refs.append(_ref(lat, lon))
    for lat in range(60, 64, 2):  # 2 x 10 = 20
        for lon in range(-180, 180, 36):
            refs.append(_ref(lat, lon))
    for lon in range(-180, 180, 45):  # 8
        refs.append(_ref(70, lon))
    for lon in range(-180, 180, 90):  # 4
        refs.append(_ref(80, lon))
    for lon in (-90, 90):  # 2
        refs.append(_ref(85, lon))
    return refs


_BAND_SPANS = {
    "0-50": (0, 50), "50-60": (50, 60), "60-70": (60, 70),
    "70-80": (70, 80), "80-85": (80, 85), "85-90": (85, 90),
}


def _real_shaped_frame(divisor: int = 10) -> list[CogRef]:
    """A frame carrying the real corpus's band proportions, scaled down by ``divisor``.

    Lets the consumer samples be tested against the shape they actually ship against,
    without standing up 26,450 cells.
    """
    refs: list[CogRef] = []
    for band, (lo, hi) in _BAND_SPANS.items():
        # lon-major, so each band spreads over geo-cells before it deepens
        pairs = [(lat, lon) for lon in range(-180, 180) for lat in range(lo, hi)]
        refs += [_ref(lat, lon) for lat, lon in pairs[: round(REAL_BAND_COUNTS[band] / divisor)]]
    return refs


def _bands(refs) -> Counter:
    return Counter(sampling.band_of(r) for r in refs)


def _above_50(counts) -> int:
    """Draws landing in a band where the anisotropic-pixel risk lives (plan §11.4)."""
    return sum(v for b, v in counts.items() if b != "0-50")


# --- Cell -> band / geocell keys ----------------------------------------------


def test_band_of_uses_the_cell_center_not_its_origin():
    # origin 49 spans [49, 50] -> centre 49.5 stays in the 0-50 band
    assert sampling.band_of(_ref(49, 0)) == "0-50"
    assert sampling.band_of(_ref(50, 0)) == "50-60"


def test_band_of_is_poleward_symmetric():
    # bands are labelled on |lat| (recon §6), so S55 and N55 share a band
    assert sampling.band_of(_ref(-55, 0)) == sampling.band_of(_ref(55, 0)) == "50-60"


def test_geocell_of_uses_the_cell_center():
    # N45E006 spans [45,46]x[6,7] -> centre (6.5, 45.5) -> cell (+000, +040)
    assert sampling.geocell_of(_ref(45, 6)) == "+000_+040"


# --- Band quota allocation ----------------------------------------------------


def test_proportional_quota_tracks_band_population():
    counts = {"0-50": 360, "50-60": 40, "60-70": 20, "70-80": 8, "80-85": 4, "85-90": 2}
    q = sampling.band_quotas(counts, 100, allocation="proportional")
    assert sum(q.values()) == 100
    assert q["0-50"] == 83  # 360/434 * 100, largest-remainder rounded
    assert q["50-60"] == 9
    assert q["85-90"] == 0  # a band this thin rounds away entirely


def test_every_band_survives_proportional_on_the_real_corpus_shape():
    # The real frame is pole-DENSE, so nothing rounds away and every band is reachable
    # without a floor -- the poles are not where a corpus-shaped draw goes wrong.
    q = sampling.band_quotas(REAL_BAND_COUNTS, 100, allocation="proportional")
    assert sum(q.values()) == 100
    assert all(v > 0 for v in q.values())
    assert sum(v for b, v in q.items() if b != "0-50") == 55  # 55% of cells are >50


def test_balanced_buys_equal_power_per_band_on_the_real_corpus_shape():
    q = sampling.band_quotas(REAL_BAND_COUNTS, 100, allocation="balanced")
    assert sum(q.values()) == 100
    assert max(q.values()) - min(q.values()) <= 1  # every band gets ~n/6


def test_balanced_quota_lifts_the_high_latitude_tail():
    counts = {"0-50": 360, "50-60": 40, "60-70": 20, "70-80": 8, "80-85": 4, "85-90": 2}
    prop = sampling.band_quotas(counts, 100, allocation="proportional")
    bal = sampling.band_quotas(counts, 100, allocation="balanced")
    assert sum(bal.values()) == 100
    assert _above_50(bal) > _above_50(prop)
    assert all(v > 0 for v in bal.values())  # every populated band represented


def test_quota_never_exceeds_the_available_cells():
    counts = {"0-50": 360, "50-60": 40, "60-70": 20, "70-80": 8, "80-85": 4, "85-90": 2}
    q = sampling.band_quotas(counts, 100, allocation="balanced")
    # an equal share would be ~17 each, but these bands hold fewer cells than that
    assert q["85-90"] == 2
    assert q["80-85"] == 4
    assert q["70-80"] == 8
    assert sum(q.values()) == 100  # the surplus redistributes to bands with headroom


def test_quota_underfills_when_the_frame_is_too_small():
    q = sampling.band_quotas({"0-50": 5, "50-60": 2}, 100, allocation="proportional")
    assert q == {"0-50": 5, "50-60": 2}  # everything, and no more


# --- Sampling: size, spread, determinism --------------------------------------


def test_sample_returns_n_distinct_cogs():
    picked = sampling.sample_cogs(_frame(), 100)
    assert len(picked) == 100
    assert len({r.stem for r in picked}) == 100


def test_sample_is_deterministic():
    a = sampling.sample_cogs(_frame(), 100, seed=7)
    b = sampling.sample_cogs(_frame(), 100, seed=7)
    assert [r.stem for r in a] == [r.stem for r in b]


def test_sample_is_independent_of_frame_order():
    # discovery lists S3 prefixes; the sampler must not inherit that ordering
    frame = _frame()
    shuffled = list(reversed(frame))
    a = sampling.sample_cogs(frame, 100, seed=7)
    b = sampling.sample_cogs(shuffled, 100, seed=7)
    assert [r.stem for r in a] == [r.stem for r in b]


def test_sample_is_stable_against_a_hostile_stem_ordering():
    # sorted() input is the pathological case for any order-dependent shuffle
    frame = _frame()
    a = sampling.sample_cogs(frame, 60, seed=7)
    b = sampling.sample_cogs(sorted(frame, key=lambda r: r.stem), 60, seed=7)
    assert [r.stem for r in a] == [r.stem for r in b]


def test_seed_changes_the_draw():
    a = sampling.sample_cogs(_frame(), 100, seed=1)
    b = sampling.sample_cogs(_frame(), 100, seed=2)
    assert [r.stem for r in a] != [r.stem for r in b]


def test_default_seed_is_the_locked_project_seed():
    a = sampling.sample_cogs(_frame(), 50)
    b = sampling.sample_cogs(_frame(), 50, seed=config.RANDOM_SEED)
    assert [r.stem for r in a] == [r.stem for r in b]


def test_balanced_sample_reaches_above_50_degrees():
    # plan §11.4: the calibration sample MUST carry high-latitude cells, where the
    # anisotropic-pixel artifact lives.
    picked = sampling.sample_cogs(_frame(), 100, allocation="balanced")
    bands = _bands(picked)
    for band in ("50-60", "60-70", "70-80", "80-85", "85-90"):
        assert bands[band] > 0, f"band {band} missing from the calibration sample"


# --- The Phase-A consumers ----------------------------------------------------


def test_calibration_sample_carries_every_high_latitude_band():
    # The issue's hard requirement, pinned as a property of the sampler rather than of
    # the caller's arguments: A3 cannot be decided without cells above 50 degrees.
    for frame in (_frame(), _real_shaped_frame()):
        bands = _bands(sampling.calibration_sample(frame))
        for band in ("50-60", "60-70", "70-80", "80-85", "85-90"):
            assert bands[band] > 0, f"band {band} missing from the calibration sample"


def test_calibration_sample_beats_a_corpus_shaped_draw_where_the_risk_is():
    # The reason A3 does not just take the prevalence sample: on the real frame shape a
    # corpus-shaped draw spends most of itself in 0-50, where the artifact is not.
    frame = _real_shaped_frame()
    assert _above_50(_bands(sampling.calibration_sample(frame))) > _above_50(
        _bands(sampling.prevalence_sample(frame, sampling.CALIBRATION_SAMPLE_N))
    )


def test_calibration_sample_spreads_power_evenly_across_bands():
    bands = _bands(sampling.calibration_sample(_real_shaped_frame()))
    assert max(bands.values()) - min(bands.values()) <= 1


def test_prevalence_sample_is_corpus_shaped():
    frame = _real_shaped_frame()
    picked = sampling.prevalence_sample(frame, 500)
    assert len(picked) == 500
    # each band lands within a cell or two of its share of the frame
    for band, count in _bands(frame).items():
        assert abs(_bands(picked)[band] - 500 * count / len(frame)) <= 2


def test_consumer_samples_default_to_the_plan_sizes():
    frame = _real_shaped_frame()
    assert len(sampling.prevalence_sample(frame)) == sampling.PREVALENCE_SAMPLE_N == 500
    assert len(sampling.calibration_sample(frame)) == sampling.CALIBRATION_SAMPLE_N == 100


def test_consumer_samples_are_deterministic():
    frame = _real_shaped_frame()
    for draw in (sampling.prevalence_sample, sampling.calibration_sample):
        assert [r.stem for r in draw(frame)] == [r.stem for r in draw(frame)]


def test_sample_honours_the_band_quotas():
    frame = _frame()
    counts = dict(_bands(frame))
    quotas = sampling.band_quotas(counts, 100, allocation="balanced")
    picked = sampling.sample_cogs(frame, 100, allocation="balanced")
    assert dict(_bands(picked)) == {b: q for b, q in quotas.items() if q}


def test_sample_spreads_round_robin_across_geocells():
    # 40 cells across 10 geocells in one band: a quota of 20 takes 2 from each,
    # never 20 from one.
    frame = [_ref(50 + (i % 4) * 2, -180 + (i // 4) * 36) for i in range(40)]
    picked = sampling.sample_cogs(frame, 20, allocation="balanced")
    by_cell = Counter(sampling.geocell_of(r) for r in picked)
    assert sorted(by_cell.values()) == [2] * 10


def test_sample_takes_everything_when_the_frame_is_smaller_than_n():
    frame = _frame()
    picked = sampling.sample_cogs(frame, 10_000)
    assert {r.stem for r in picked} == {r.stem for r in frame}


def test_sample_prefix_stays_band_balanced():
    # the over-draw contract: read down the list, stop early, still have a spread
    # sample -- this is what lets a caller filter mostly-ocean cells post-read.
    n = 60
    picked = sampling.sample_cogs(_frame(), n, allocation="balanced")
    full = _bands(picked)
    half = _bands(picked[: n // 2])
    for band, quota in full.items():
        assert abs(half[band] - quota / 2) <= 1, f"band {band} clusters in the prefix"


def test_rejects_a_nonpositive_n():
    with pytest.raises(ValueError, match="n"):
        sampling.sample_cogs(_frame(), 0)


def test_rejects_an_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        sampling.sample_cogs([], 10)


def test_rejects_an_unknown_allocation():
    with pytest.raises(ValueError, match="allocation"):
        sampling.sample_cogs(_frame(), 10, allocation="nonsense")
