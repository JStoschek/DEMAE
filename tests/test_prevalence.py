"""Mask-prevalence counting and aggregation — pure, zero-network (issue #19).

Every test here runs against fabricated masks (in-memory
:class:`~dataset.masks.SyntheticMaskReader` arrays) so the whole measurement
recipe is exercised without touching S3. The live 500-tile pass lives in the
:mod:`dataset.prevalence` driver, not the suite.

The load-bearing checks are the three §6.1 cross-mask identities (a mis-keyed bin
would corrupt a committed table) and the pixel-*pooled* aggregation (the plan's ⚠️:
averaging per-cell fractions inherits a polar over-weight; pooling raw counts does
not).
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset import config, masks, prevalence
from dataset.geometry import parse_cog_key

STEM_A = "Copernicus_DSM_COG_10_N45_00_E007_00_DEM"  # mid-latitude
STEM_B = "Copernicus_DSM_COG_10_N65_00_W019_00_DEM"  # high-latitude, half-width


# --- count_codes: the vocabulary tripwire (spec §3, §6.1) ---------------------


def test_count_codes_bins_every_pixel_and_sums_to_size():
    arr = np.array([[1, 1, 2], [3, 8, 8]], dtype=np.uint8)
    counts = prevalence.count_codes(arr, masks.EDM_CODES)
    assert counts[1] == 2 and counts[2] == 1 and counts[3] == 1 and counts[8] == 2
    assert sum(counts.values()) == arr.size


def test_count_codes_keeps_the_full_vocabulary_including_absent_codes():
    # A code the raster never shows still appears, at zero — a table row is never
    # silently missing just because this tile lacks that class.
    arr = np.array([[1, 1], [2, 2]], dtype=np.uint8)
    counts = prevalence.count_codes(arr, masks.EDM_CODES)
    assert set(counts) == set(masks.EDM_CODES)
    assert counts[0] == 0 and counts[13] == 0


def test_count_codes_handles_flm_non_contiguous_high_codes():
    # 100..102 must not be dropped or folded into a contiguous range.
    arr = np.array([[2, 3, 100], [101, 102, 9]], dtype=np.uint8)
    counts = prevalence.count_codes(arr, masks.FLM_CODES)
    assert counts[100] == 1 and counts[101] == 1 and counts[102] == 1
    assert set(counts) == set(masks.FLM_CODES)


def test_count_codes_rejects_out_of_vocabulary_value():
    # A real mask never carries a code outside its Handbook vocabulary; if one
    # appears it is a read bug, and must be loud, not silently pooled.
    arr = np.array([[1, 2], [3, 200]], dtype=np.uint8)
    with pytest.raises(ValueError):
        prevalence.count_codes(arr, masks.EDM_CODES)


# --- tile_counts: full-res read of all three masks ----------------------------


def _reader(masks_by_stem):
    return masks.SyntheticMaskReader(masks_by_stem)


def _consistent_tile(h=32, w=32):
    """A fabricated tile whose three masks satisfy the §6.1 identities exactly.

    Layout by pixel: a native-edited block (EDM 1 / FLM 2 / WBM 0), an
    external-infill block (EDM 2 / FLM 5=SRTM30 / WBM 0), an ocean block
    (EDM 8 / FLM 2 / WBM 1), and a lake block (EDM 9 / FLM 2 / WBM 2).
    """
    edm = np.full((h, w), 1, np.uint8)
    flm = np.full((h, w), 2, np.uint8)
    wbm = np.full((h, w), 0, np.uint8)
    # external infill — EDM 2 ≡ FLM ≥ 3
    edm[:, :8] = 2
    flm[:, :8] = 5
    # ocean — EDM 8 ≡ WBM 1; water is edited-not-filled, so FLM 1 (not 2), which is
    # what keeps EDM 1 ≡ FLM 2 exact.
    edm[:8, 8:16] = 8
    flm[:8, 8:16] = 1
    wbm[:8, 8:16] = 1
    # lake — EDM 9 ≡ WBM 2
    edm[8:16, 8:16] = 9
    flm[8:16, 8:16] = 1
    wbm[8:16, 8:16] = 2
    return {"EDM": edm, "FLM": flm, "WBM": wbm}


def test_tile_counts_reads_full_res_and_totals_match():
    stem = STEM_A
    reader = _reader({stem: _consistent_tile(32, 32)})
    tc = prevalence.tile_counts(reader, parse_cog_key(stem))
    assert tc.total == 32 * 32
    assert sum(tc.edm.values()) == tc.total
    assert sum(tc.flm.values()) == tc.total
    assert sum(tc.wbm.values()) == tc.total
    # geo attributes come from the stem, not a read.
    assert tc.band == "0-50"
    assert tc.stem == stem


def test_tile_counts_full_res_not_decimated():
    # A 1-pixel rare class must survive: the read is full-res, never an ×8 overview
    # (spec §7 — mode resampling erases never-dominant codes).
    h = w = 32
    t = _consistent_tile(h, w)
    t["EDM"][0, 0] = 11  # a single shoreline pixel
    reader = _reader({STEM_A: t})
    tc = prevalence.tile_counts(reader, parse_cog_key(STEM_A))
    assert tc.edm[11] == 1


# --- §6.1 identities: the committed-table tripwire ----------------------------


def test_identity_residuals_are_zero_on_a_consistent_tile():
    reader = _reader({STEM_A: _consistent_tile()})
    tc = prevalence.tile_counts(reader, parse_cog_key(STEM_A))
    res = prevalence.identity_residuals(tc)
    assert res["edm1_eq_flm2"] == 0
    assert res["edm2_eq_flm_fill"] == 0
    assert res["wbm_water_eq_edm_water"] == 0
    assert prevalence.identities_hold(tc)


def test_identity_residuals_flag_a_miskeyed_tile():
    t = _consistent_tile()
    t["FLM"][:, :8] = 2  # break EDM 2 ≡ FLM ≥ 3 (call the infill "not filled")
    reader = _reader({STEM_A: t})
    tc = prevalence.tile_counts(reader, parse_cog_key(STEM_A))
    res = prevalence.identity_residuals(tc)
    assert res["edm2_eq_flm_fill"] > 0
    assert not prevalence.identities_hold(tc)


# --- derived per-tile fractions (the Phase-C gate inputs) ---------------------


def test_derived_fractions_match_the_hand_counted_blocks():
    reader = _reader({STEM_A: _consistent_tile(32, 32)})
    tc = prevalence.tile_counts(reader, parse_cog_key(STEM_A))
    # water: ocean 8x8 + lake 8x8 = 128 of 1024
    assert prevalence.wbm_water_frac(tc) == pytest.approx(128 / 1024)
    # external fill: FLM≥3 block is 32x8 = 256 of 1024
    assert prevalence.flm_fill_frac(tc) == pytest.approx(256 / 1024)
    # EDM-edited (terrain edits, water excluded) == the infill block, 256 of 1024;
    # equals the union with FLM≥3 by the §6.1 identity.
    assert prevalence.edm_edited_frac(tc) == pytest.approx(256 / 1024)


def test_edm_edited_excludes_void_pristine_and_water():
    # Only {2,3,4,5,6,7,12,13} count as terrain editing; 0/1 and water 8-11 do not.
    assert 0 not in prevalence.EDITED_EDM_CODES
    assert 1 not in prevalence.EDITED_EDM_CODES
    for water in (8, 9, 10, 11):
        assert water not in prevalence.EDITED_EDM_CODES
    assert prevalence.EDITED_EDM_CODES == frozenset({2, 3, 4, 5, 6, 7, 12, 13})


# --- pixel-pooled aggregation (the plan's ⚠️ polar-weight guard) --------------


def test_pooled_fraction_weights_by_pixels_not_by_tile():
    # A big all-'external-fill' tile and a small all-'not-edited' tile: pooling by
    # pixels must land far from the per-tile average of the two fractions.
    big = {
        "EDM": np.full((100, 100), 2, np.uint8),
        "FLM": np.full((100, 100), 5, np.uint8),
        "WBM": np.zeros((100, 100), np.uint8),
    }
    small = {
        "EDM": np.full((10, 10), 1, np.uint8),
        "FLM": np.full((10, 10), 2, np.uint8),
        "WBM": np.zeros((10, 10), np.uint8),
    }
    reader = _reader({STEM_A: big, STEM_B: small})
    tiles = [
        prevalence.tile_counts(reader, parse_cog_key(STEM_A)),
        prevalence.tile_counts(reader, parse_cog_key(STEM_B)),
    ]
    pooled = prevalence.pooled_fractions(tiles, "EDM")
    # 10000 external-fill px + 100 not-edited px over 10100 total.
    assert pooled[2] == pytest.approx(10000 / 10100)
    assert pooled[1] == pytest.approx(100 / 10100)
    # per-tile mean of the code-2 fraction would be (1.0 + 0.0)/2 = 0.5 — pooling
    # must not report that.
    assert abs(pooled[2] - 0.5) > 0.4


def test_pooled_fractions_sum_to_one_over_the_vocabulary():
    reader = _reader({STEM_A: _consistent_tile()})
    tc = prevalence.tile_counts(reader, parse_cog_key(STEM_A))
    for layer in ("EDM", "FLM", "WBM"):
        pooled = prevalence.pooled_fractions([tc], layer)
        assert sum(pooled.values()) == pytest.approx(1.0)


def test_pooled_subset_frac_pools_a_code_group_by_pixels():
    reader = _reader({STEM_A: _consistent_tile(32, 32)})
    tc = prevalence.tile_counts(reader, parse_cog_key(STEM_A))
    # WBM water {1,2,3} over the whole tile == wbm_water_frac, and the non-contiguous
    # FLM fill list is honoured.
    assert prevalence.pooled_subset_frac([tc], "WBM", (1, 2, 3)) == pytest.approx(
        prevalence.wbm_water_frac(tc)
    )
    assert prevalence.pooled_subset_frac(
        [tc], "FLM", prevalence.FLM_CODES_FILL
    ) == pytest.approx(prevalence.flm_fill_frac(tc))
    assert set(prevalence.FLM_CODES_FILL) == {3, 4, 5, 6, 7, 8, 9, 100, 101, 102}


# --- measure(): the §6.1 identity tripwire actually guards the pool --------------


def test_measure_excludes_identity_violators_from_the_pool():
    good = _consistent_tile()  # identities hold
    bad = _consistent_tile()
    bad["FLM"][:, :8] = 2  # break EDM 2 ≡ FLM ≥ 3
    good_stem = "Copernicus_DSM_COG_10_N45_00_E007_00_DEM"
    bad_stem = "Copernicus_DSM_COG_10_N10_00_E010_00_DEM"
    reader = _reader({good_stem: good, bad_stem: bad})
    cogs = [parse_cog_key(good_stem), parse_cog_key(bad_stem)]

    tiles, anomalies = prevalence.measure(reader, cogs, workers=2)
    kept = {tc.stem for tc in tiles}
    assert good_stem in kept
    assert bad_stem not in kept  # a mis-keyed tile never reaches a pooled number
    assert any(bad_stem in stem and "identity" in reason for stem, reason in anomalies)


def test_measure_records_absent_masks_as_anomalies_not_zeros():
    good_stem = "Copernicus_DSM_COG_10_N45_00_E007_00_DEM"
    reader = _reader({good_stem: _consistent_tile()})
    absent = parse_cog_key("Copernicus_DSM_COG_10_N00_00_E000_00_DEM")
    tiles, anomalies = prevalence.measure(reader, [parse_cog_key(good_stem), absent], workers=2)
    assert {tc.stem for tc in tiles} == {good_stem}
    assert any("absent" in reason for _, reason in anomalies)


# --- distribution + tail ------------------------------------------------------


def test_describe_reports_mean_quantiles_and_tail():
    vals = np.linspace(0.0, 1.0, 101)  # 0.00, 0.01, ..., 1.00
    d = prevalence.describe(vals)
    assert d["n"] == 101
    assert d["mean"] == pytest.approx(0.5)
    assert d["p50"] == pytest.approx(0.5, abs=0.01)
    assert d["p90"] == pytest.approx(0.9, abs=0.01)
    assert d["max"] == pytest.approx(1.0)


# --- threshold attrition (apply the plan's defaults; do not re-tune) ----------


def test_attrition_counts_tiles_dropped_by_each_gate():
    # Three tiles: one wet (water 0.5), one edited (fill 0.4), one clean.
    wet = {
        "EDM": np.full((10, 10), 8, np.uint8),
        "FLM": np.full((10, 10), 2, np.uint8),
        "WBM": np.full((10, 10), 1, np.uint8),
    }
    wet["EDM"][:5] = 1  # half ocean -> water frac 0.5
    wet["WBM"][:5] = 0
    edited = {
        "EDM": np.full((10, 10), 1, np.uint8),
        "FLM": np.full((10, 10), 2, np.uint8),
        "WBM": np.zeros((10, 10), np.uint8),
    }
    edited["EDM"][:4] = 2  # 0.4 external infill
    edited["FLM"][:4] = 5
    clean = {
        "EDM": np.full((10, 10), 1, np.uint8),
        "FLM": np.full((10, 10), 2, np.uint8),
        "WBM": np.zeros((10, 10), np.uint8),
    }
    stems = [
        "Copernicus_DSM_COG_10_N45_00_E007_00_DEM",
        "Copernicus_DSM_COG_10_N10_00_E010_00_DEM",
        "Copernicus_DSM_COG_10_N20_00_E020_00_DEM",
    ]
    reader = _reader(dict(zip(stems, [wet, edited, clean])))
    tiles = [prevalence.tile_counts(reader, parse_cog_key(s)) for s in stems]

    att = prevalence.attrition(tiles, water_cap=0.30, edit_cap=0.25)
    assert att["n_tiles"] == 3
    assert att["dropped_water"] == 1  # the wet tile
    assert att["dropped_edit"] == 1  # the edited tile
    assert att["dropped_either"] == 2  # union
    assert att["frac_dropped_either"] == pytest.approx(2 / 3)
