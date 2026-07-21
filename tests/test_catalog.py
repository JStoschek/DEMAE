"""The production global scorer and the raw candidate catalog — Phase C3 (issue #35).

Zero-network, like every other suite here: metrics are scored on hand-built arrays
and the read path runs against :class:`~dataset.tiling.SyntheticCogReader` /
:class:`~dataset.masks.SyntheticMaskReader`. The live 26,450-COG pass lives in the
:mod:`dataset.catalog` driver, not the suite.

The load-bearing checks are the ones a wrong catalog would silently pass:

1. **The scored vector is exactly ADR-0004's frozen six**, split full-res / ×8 as
   the ADR's §1 table says — and the three metrics B2 pruned are *not* scored.
2. **Nothing is gated away.** Every window of every COG lands a row, ocean-only
   windows included, because C4 tunes thresholds against this frame and needs the
   denominator (issue #35: "store all columns raw — no destructive gating").
3. **WBM is stored per class** (`wbm1_frac` … `wbm3_frac` + the aggregate), the
   amendment C2/#34 made to this ticket: the coastline stratum is uncomputable
   without `wbm1_frac`.
4. The part-file ledger is **resumable** — a 4-hour global pass that cannot resume
   is a pass that never finishes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from dataset import catalog, config, metrics
from dataset.geometry import parse_cog_key
from dataset.masks import SyntheticMaskReader
from dataset.tiling import SyntheticCogReader

_STEM = "Copernicus_DSM_COG_10_N45_00_E006_00_DEM"
_STEM2 = "Copernicus_DSM_COG_10_N48_00_E009_00_DEM"


# --- The frozen vector, and only the frozen vector ----------------------------


def test_frozen_metrics_are_adr0004_six():
    assert set(catalog.FROZEN_METRICS) == {
        "slope_mean",
        "local_relief",
        "texture",
        "elevation_relief_ratio",
        "mean_elevation",
        "geomorphon_entropy",
    }
    assert len(catalog.FROZEN_METRICS) == 6


def test_pruned_metrics_are_not_scored():
    # B2 pruned slope_std / normals_dispersion (redundant) and B3 pruned tpi_proxy.
    # They still live in dataset.metrics; the catalog must not pay to compute them.
    for pruned in ("slope_std", "normals_dispersion", "tpi_proxy"):
        assert pruned in metrics.METRICS
        assert pruned not in catalog.FROZEN_METRICS


def test_resolution_split_partitions_the_frozen_vector():
    full, ov = set(catalog.FULL_RES_METRICS), set(catalog.OVERVIEW_METRICS)
    assert full | ov == set(catalog.FROZEN_METRICS)
    assert full & ov == set()


def test_resolution_of_pins_the_adr0004_table():
    for m in ("slope_mean", "texture", "geomorphon_entropy"):
        assert catalog._resolution_of(m) == "full"
    for m in ("local_relief", "mean_elevation", "elevation_relief_ratio"):
        assert catalog._resolution_of(m) == "overview"


def test_resolution_of_unassigned_metric_raises():
    with pytest.raises(KeyError):
        catalog._resolution_of("tpi_proxy")


# --- score_window -------------------------------------------------------------


def _ramp(n: int, lo: float = 1000.0, hi: float = 1100.0) -> np.ndarray:
    return np.tile(np.linspace(lo, hi, n), (n, 1))


def test_score_window_scores_the_six_once_each():
    a = _ramp(16)
    scores = catalog.score_window(
        a, a, win_row0=0, win_col0=0, full_h=16, full_w=16, ov_h=16, ov_w=16,
        lat=45.0, tile=8,
    )
    assert set(scores) == set(catalog.FROZEN_METRICS)


def test_score_window_full_res_metric_reads_full_not_overview():
    # slope_mean is full-res: a tilted full array against a flat overview must
    # score non-zero slope.
    full = _ramp(16)
    ov = np.full((8, 8), 1050.0)
    scores = catalog.score_window(
        full, ov, win_row0=0, win_col0=0, full_h=16, full_w=16, ov_h=8, ov_w=8,
        lat=0.0, tile=8,
    )
    assert scores["slope_mean"] > 0.01


def test_score_window_overview_metric_reads_overview_not_full():
    full = np.full((16, 16), 500.0)
    ov = np.full((8, 8), 900.0)
    scores = catalog.score_window(
        full, ov, win_row0=0, win_col0=0, full_h=16, full_w=16, ov_h=8, ov_w=8,
        lat=0.0, tile=8,
    )
    assert scores["mean_elevation"] == pytest.approx(900.0)


def test_score_window_land_free_window_is_all_nan():
    # An all-ocean window has no land to measure: every metric NaN, no exception,
    # and the row is still emitted (the gate is C4's, not the scorer's).
    a = np.zeros((16, 16), dtype=np.float64)
    scores = catalog.score_window(
        a, a, win_row0=0, win_col0=0, full_h=16, full_w=16, ov_h=16, ov_w=16,
        lat=0.0, tile=8,
    )
    assert all(math.isnan(v) for v in scores.values())


# --- exact-zero fraction and land-std -----------------------------------------


def test_exact_zero_frac_counts_the_ocean_fill():
    a = np.array([[0.0, 0.0], [10.0, -5.0]])
    assert catalog.exact_zero_frac(a) == pytest.approx(0.5)


def test_land_std_excludes_the_ocean_fill():
    a = np.array([[0.0, 0.0], [10.0, 20.0]])
    # Land pixels are {10, 20}: population std 5, not the 8.29 the zeros would give.
    assert catalog.land_std(a) == pytest.approx(5.0)


def test_land_std_of_a_land_free_window_is_nan():
    assert math.isnan(catalog.land_std(np.zeros((4, 4))))


# --- mask fractions, WBM per class (C2 / #34 amendment) -----------------------


def test_window_mask_fractions_split_wbm_by_class():
    edm = np.array([[4, 4], [1, 1]], dtype=np.uint8)  # half Smoothed (edited)
    flm = np.array([[3, 2], [2, 2]], dtype=np.uint8)  # one ASTER fill
    wbm = np.array([[1, 2], [3, 0]], dtype=np.uint8)  # Ocean, Lake, River, none
    fr = catalog.window_mask_fractions(edm, flm, wbm)
    assert fr["edm_edited_frac"] == pytest.approx(0.5)
    assert fr["flm_fill_frac"] == pytest.approx(0.25)
    assert fr["wbm1_frac"] == pytest.approx(0.25)
    assert fr["wbm2_frac"] == pytest.approx(0.25)
    assert fr["wbm3_frac"] == pytest.approx(0.25)
    assert fr["wbm_water_frac"] == pytest.approx(0.75)


def test_wbm_class_fractions_sum_to_the_aggregate():
    rng = np.random.default_rng(0)
    wbm = rng.integers(0, 4, size=(16, 16), dtype=np.uint8)
    fr = catalog.window_mask_fractions(np.ones_like(wbm), np.full_like(wbm, 2), wbm)
    total = fr["wbm1_frac"] + fr["wbm2_frac"] + fr["wbm3_frac"]
    assert total == pytest.approx(fr["wbm_water_frac"])


def test_coastline_stratum_is_computable_from_stored_columns():
    # The C2 verdict expressed as a filter over the catalog's own columns: an inland
    # sea-level river window (no ocean pixel) is NOT coastline; a sea-touching one is.
    river = catalog.window_mask_fractions(
        np.ones((4, 4), np.uint8), np.full((4, 4), 2, np.uint8),
        np.full((4, 4), 3, np.uint8),  # all River, no Ocean
    )
    shore = catalog.window_mask_fractions(
        np.ones((4, 4), np.uint8), np.full((4, 4), 2, np.uint8),
        np.array([[1, 1], [0, 0]], np.uint8).repeat(2, 0).repeat(2, 1),
    )
    assert river["wbm1_frac"] == 0.0  # aggregate alone could not tell these apart
    assert river["wbm_water_frac"] == pytest.approx(1.0)
    assert shore["wbm1_frac"] > 0.0


def test_window_mask_fractions_empty_is_nan():
    empty = np.empty((0, 0), dtype=np.uint8)
    fr = catalog.window_mask_fractions(empty, empty, empty)
    assert set(fr) == set(catalog.MASK_FRACTIONS)
    assert all(math.isnan(v) for v in fr.values())


# --- tile_rows: every window, nothing gated -----------------------------------


def _hilly(n: int = 672) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    z = 500.0 + 80.0 * np.sin(xx / 40.0) + 60.0 * np.cos(yy / 55.0) + xx / 30.0
    return z.astype(np.float32)


def _masks_for(*stems: str, n: int = 672) -> SyntheticMaskReader:
    return SyntheticMaskReader(
        {
            s: {
                "EDM": np.ones((n, n), dtype=np.uint8),
                "FLM": np.full((n, n), 2, dtype=np.uint8),
                "WBM": np.zeros((n, n), dtype=np.uint8),
            }
            for s in stems
        }
    )


def test_tile_rows_emits_every_window_including_ocean_only():
    # The west column is pure ocean fill. pilot.py skipped such windows; the catalog
    # keeps them — the zero_fill gate is a C4 filter column, so its denominator must
    # be in the frame.
    a = _hilly(672)
    a[:, :224] = config.OCEAN_FILL_VALUE
    rows, anomaly = catalog.tile_rows(
        SyntheticCogReader({_STEM: a}), _masks_for(_STEM), parse_cog_key(_STEM), factor=8
    )
    assert anomaly is None
    assert len(rows) == 9  # the full 3×3 grid, not the 6 land-bearing windows
    ocean = [r for r in rows if r.win_col0 == 0]
    assert len(ocean) == 3
    for r in ocean:
        assert r.exact_zero_frac == pytest.approx(1.0)
        assert math.isnan(r.land_std)
        assert all(math.isnan(v) for v in r.metrics.values())


def test_tile_rows_carries_geography_and_a_traceable_id():
    rows, _ = catalog.tile_rows(
        SyntheticCogReader({_STEM: _hilly(672)}), _masks_for(_STEM),
        parse_cog_key(_STEM), factor=8,
    )
    r = rows[0]
    assert r.window_id == f"{_STEM}:{r.win_row0}:{r.win_col0}"
    assert r.band == "0-50" and r.geocell == "+000_+040" and r.stem == _STEM
    assert set(r.metrics) == set(catalog.FROZEN_METRICS)
    assert set(r.masks) == set(catalog.MASK_FRACTIONS)


def test_tile_rows_absent_mask_nans_mask_columns_and_surfaces_anomaly():
    rows, anomaly = catalog.tile_rows(
        SyntheticCogReader({_STEM: _hilly(672)}), SyntheticMaskReader({}),
        parse_cog_key(_STEM), factor=8,
    )
    assert anomaly is not None and _STEM in anomaly[0]
    assert len(rows) == 9
    for r in rows:
        assert all(math.isnan(v) for v in r.masks.values())
        assert np.isfinite(r.metrics["slope_mean"])  # terrain columns unaffected


# --- records + parquet --------------------------------------------------------


def _rows(stem: str = _STEM) -> list[catalog.CatalogRow]:
    rows, _ = catalog.tile_rows(
        SyntheticCogReader({stem: _hilly(672)}), _masks_for(stem),
        parse_cog_key(stem), factor=8,
    )
    return rows


def test_to_records_emits_the_declared_columns_in_order():
    recs = catalog.to_records(_rows()[:2])
    assert list(recs[0]) == list(catalog.COLUMNS)


def test_catalog_columns_cover_every_gate_input_c4_needs():
    # Each C4/C5 gate must be expressible as a filter over stored columns.
    for column in (
        "exact_zero_frac",  # zero_fill (ADR-0001 §3 Amendment 2)
        "land_std",  # degenerate (ADR-0002 §2)
        "wbm_water_frac",  # water cap / water_affected
        "wbm1_frac",  # coastline stratum, sea-touching (C2 / #34)
        "edm_edited_frac",  # edited cap
        "flm_fill_frac",  # non-TanDEM fill cap
    ):
        assert column in catalog.COLUMNS


def test_write_part_round_trips_through_parquet(tmp_path: Path):
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "part-00000.parquet"
    recs = catalog.to_records(_rows()[:4])
    catalog.write_part(recs, path)
    table = pq.read_table(path)
    assert table.num_rows == 4
    assert table.column_names == list(catalog.COLUMNS)
    assert table.column("window_id")[0].as_py() == recs[0]["window_id"]
    assert table.column("slope_mean")[0].as_py() == pytest.approx(recs[0]["slope_mean"])


# --- the scan driver: parts, ledger, resume -----------------------------------


def _readers(*stems: str):
    return (
        SyntheticCogReader({s: _hilly(672) for s in stems}),
        _masks_for(*stems),
    )


def test_scan_writes_parts_and_a_ledger(tmp_path: Path):
    pytest.importorskip("pyarrow.parquet")
    cog_reader, mask_reader = _readers(_STEM, _STEM2)
    result = catalog.scan(
        cog_reader, mask_reader,
        [parse_cog_key(_STEM), parse_cog_key(_STEM2)],
        tmp_path, rows_per_part=9, factor=8, workers=2,
    )
    assert result.n_rows == 18 and result.n_cogs == 2 and result.n_skipped == 0
    assert sorted(p.name for p in (tmp_path / catalog.PARTS_DIR).glob("part-*.parquet")) == [
        "part-00000.parquet", "part-00001.parquet"
    ]
    ledger = [json.loads(x) for x in (tmp_path / catalog.PROGRESS_FILE).read_text().splitlines()]
    assert [r["part"] for r in ledger] == [0, 1]
    assert {s for r in ledger for s in r["stems"]} == {_STEM, _STEM2}


def test_scan_resumes_without_rescoring_finished_cogs(tmp_path: Path):
    pytest.importorskip("pyarrow.parquet")
    cog_reader, mask_reader = _readers(_STEM, _STEM2)
    catalog.scan(
        cog_reader, mask_reader, [parse_cog_key(_STEM)], tmp_path,
        rows_per_part=9, factor=8, workers=1,
    )
    again = catalog.scan(
        cog_reader, mask_reader,
        [parse_cog_key(_STEM), parse_cog_key(_STEM2)],
        tmp_path, rows_per_part=9, factor=8, workers=1,
    )
    assert again.n_skipped == 1 and again.n_cogs == 1
    assert again.n_rows == 18  # cumulative across both runs, from the ledger
    assert len(list((tmp_path / catalog.PARTS_DIR).glob("part-*.parquet"))) == 2


def test_scan_discards_a_part_the_ledger_never_recorded(tmp_path: Path):
    # A crash between writing a part and appending its ledger line leaves an orphan.
    # Trusting it would double-count those windows, so the next run drops it.
    pytest.importorskip("pyarrow.parquet")
    cog_reader, mask_reader = _readers(_STEM, _STEM2)
    catalog.scan(
        cog_reader, mask_reader, [parse_cog_key(_STEM)], tmp_path,
        rows_per_part=9, factor=8, workers=1,
    )
    orphan = tmp_path / catalog.PARTS_DIR / "part-00001.parquet"
    orphan.write_bytes(b"garbage")
    catalog.scan(
        cog_reader, mask_reader,
        [parse_cog_key(_STEM), parse_cog_key(_STEM2)],
        tmp_path, rows_per_part=9, factor=8, workers=1,
    )
    assert orphan.exists()  # rewritten, not the garbage
    import pyarrow.parquet as pq

    assert pq.read_table(orphan).num_rows == 9


def test_scan_surfaces_a_dead_cog_without_losing_the_rest(tmp_path: Path):
    pytest.importorskip("pyarrow.parquet")
    cog_reader, mask_reader = _readers(_STEM)
    missing = parse_cog_key("Copernicus_DSM_COG_10_N10_00_E010_00_DEM")
    result = catalog.scan(
        cog_reader, mask_reader, [missing, parse_cog_key(_STEM)], tmp_path,
        rows_per_part=100, factor=8, workers=2,
    )
    assert result.n_rows == 9
    assert result.n_anomalies == 1
    logged = (tmp_path / catalog.ANOMALY_FILE).read_text().splitlines()
    assert len(logged) == 1 and "N10" in logged[0]


class _FlakyMaskReader:
    """A mask reader whose reads fail transiently — the S3 hiccup, not an absence."""

    def __init__(self, inner: SyntheticMaskReader, fail_stems: set[str]) -> None:
        self._inner = inner
        self._fail = fail_stems

    def read_overview(self, cog, layer, factor):  # pragma: no cover - unused here
        return self._inner.read_overview(cog, layer, factor)

    def read_windows(self, cog, layer, origins, tile):
        if cog.stem in self._fail:
            raise OSError("connection reset by peer")
        return self._inner.read_windows(cog, layer, origins, tile)


def test_scan_retries_a_transient_mask_failure_instead_of_baking_in_nan(tmp_path: Path):
    # A transient mask read failure must not land rows with permanently-unknowable
    # wbm1_frac: the COG is dropped unmarked so the next run rescores it.
    pytest.importorskip("pyarrow.parquet")
    cog_reader, good_masks = _readers(_STEM)
    flaky = _FlakyMaskReader(good_masks, {_STEM})

    first = catalog.scan(
        cog_reader, flaky, [parse_cog_key(_STEM)], tmp_path,
        rows_per_part=100, factor=8, workers=1,
    )
    assert first.n_rows == 0 and first.n_cogs == 0 and first.n_anomalies == 1

    healed = catalog.scan(
        cog_reader, good_masks, [parse_cog_key(_STEM)], tmp_path,
        rows_per_part=100, factor=8, workers=1,
    )
    assert healed.n_cogs == 1 and healed.n_skipped == 0 and healed.n_rows == 9
    table = catalog.read_catalog(tmp_path)
    assert all(v is not None for v in table.column("wbm1_frac").to_pylist())


def test_scan_marks_an_absent_mask_done_rather_than_retrying_forever(tmp_path: Path):
    # A genuinely-absent mask is permanent (masks.MaskNotFound). Those rows land with
    # NaN mask columns and the COG is marked done — retrying could never help.
    pytest.importorskip("pyarrow.parquet")
    cog_reader, _ = _readers(_STEM)
    first = catalog.scan(
        cog_reader, SyntheticMaskReader({}), [parse_cog_key(_STEM)], tmp_path,
        rows_per_part=100, factor=8, workers=1,
    )
    assert first.n_rows == 9 and first.n_anomalies == 1
    again = catalog.scan(
        cog_reader, SyntheticMaskReader({}), [parse_cog_key(_STEM)], tmp_path,
        rows_per_part=100, factor=8, workers=1,
    )
    assert again.n_skipped == 1 and again.n_cogs == 0


def test_read_catalog_returns_one_table_over_every_part(tmp_path: Path):
    # The plan's §C3 asks for "one Parquet"; the parts are an implementation detail of
    # resumability, and the ledger/meta sidecars must not break the dataset read.
    pytest.importorskip("pyarrow.parquet")
    cog_reader, mask_reader = _readers(_STEM, _STEM2)
    catalog.scan(
        cog_reader, mask_reader, [parse_cog_key(_STEM), parse_cog_key(_STEM2)],
        tmp_path, rows_per_part=9, factor=8, workers=2,
    )
    table = catalog.read_catalog(tmp_path)
    assert table.num_rows == 18
    assert table.column_names == list(catalog.COLUMNS)


def test_scan_writes_meta_with_the_scored_vector_and_resolutions(tmp_path: Path):
    pytest.importorskip("pyarrow.parquet")
    cog_reader, mask_reader = _readers(_STEM)
    catalog.scan(
        cog_reader, mask_reader, [parse_cog_key(_STEM)], tmp_path,
        rows_per_part=100, factor=8, workers=1, code_version="test",
        snapshot_date="2026-07-20",
    )
    meta = json.loads((tmp_path / catalog.META_FILE).read_text())
    assert meta["n_rows"] == 9 and meta["n_cogs"] == 1
    assert meta["columns"] == list(catalog.COLUMNS)
    assert meta["resolutions"]["slope_mean"] == "full"
    assert meta["resolutions"]["mean_elevation"] == "overview"
    assert meta["overview_factor"] == 8
    assert meta["code_version"] == "test"
