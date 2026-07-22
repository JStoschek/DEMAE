"""The quality gates as filter columns over the raw catalog — Phase C5 (issue #37).

Zero-network and zero-catalog: every case here is a hand-built frame of a few rows,
so the suite pins the *predicates* C4 locked
([`docs/research/gate-attrition.md`](../docs/research/gate-attrition.md)) rather than
re-measuring the 4.39 M-row corpus, which lives in ``notebooks/c5_catalog_summary.py``.

The load-bearing checks are the ones a wrong gating pass would silently pass:

1. **The thresholds are C4's locked values**, and each predicate's *edge* is on the
   side the doc puts it (``>=`` vs ``>``, ``<`` vs ``<=``).
2. **Nothing is destroyed.** Gating adds boolean columns; every raw row and every raw
   column survives, so a threshold can move without touching S3.
3. **NaN is not a pass.** ``land_std`` is NaN on a window with no land pixel and the
   mask fractions are NaN when a mask is absent — neither may fall through into
   ``keep`` as if the gate had been evaluated and cleared.
4. The tags (``water_affected``, ``coastline``) are **not drops** — they are computed
   over the whole frame and only intersected with ``keep`` when reported.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from dataset import catalog, gates

# --- Frame construction -------------------------------------------------------

#: A window that passes every gate: land, rugged, dry, unedited.
_CLEAN = {
    "exact_zero_frac": 0.0,
    "land_std": 40.0,
    "edm_edited_frac": 0.0,
    "flm_fill_frac": 0.0,
    "wbm1_frac": 0.0,
    "wbm2_frac": 0.0,
    "wbm3_frac": 0.0,
    "wbm_water_frac": 0.0,
}


def _frame(*overrides: dict[str, float]) -> dict[str, np.ndarray]:
    """A column frame of ``len(overrides)`` rows, each a ``_CLEAN`` row plus edits."""
    rows = [{**_CLEAN, **o} for o in overrides]
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in _CLEAN}


def _flags(*overrides: dict[str, float]) -> dict[str, np.ndarray]:
    return gates.gate_flags(_frame(*overrides))


def _one(name: str, override: dict[str, float]) -> bool:
    return bool(_flags(override)[name][0])


# --- The locked thresholds ----------------------------------------------------


def test_thresholds_are_c4s_locked_values():
    # docs/research/gate-attrition.md §3. Changing any of these is an amendment to
    # that document, not an edit to a constant.
    assert gates.ZERO_FILL_MIN == 0.75
    assert gates.DEGENERATE_LAND_STD_M == 3.0
    assert gates.WATER_DROP_MAX == 0.30
    assert gates.WATER_AFFECTED_MIN == 0.05
    assert gates.EDITED_MAX == 0.25


def test_drop_gates_are_the_four_c4_locked():
    assert gates.DROP_GATES == ("zero_fill", "degenerate", "water", "edited")


def test_flag_columns_cover_the_gates_the_tags_and_keep():
    assert set(gates.FLAG_COLUMNS) == {
        *gates.DROP_GATES,
        "masks_missing",
        "keep",
        "water_affected",
        "coastline",
    }


def test_every_gate_input_is_a_stored_catalog_column():
    # A gate that needs a column the catalog does not store is a gate that needs a
    # global re-score — the thing this frame exists to make unnecessary.
    for column in gates.GATE_INPUTS:
        assert column in catalog.COLUMNS


# --- Predicate edges ----------------------------------------------------------


def test_zero_fill_fires_at_the_threshold_not_below():
    assert _one("zero_fill", {"exact_zero_frac": 0.75}) is True
    assert _one("zero_fill", {"exact_zero_frac": 0.7499}) is False


def test_degenerate_is_strict_below_three_metres():
    assert _one("degenerate", {"land_std": 2.999}) is True
    assert _one("degenerate", {"land_std": 3.0}) is False


def test_degenerate_does_not_fire_on_a_window_with_no_land():
    # land_std is NaN when no land pixel survives (catalog.land_std). C4 §1 turns on
    # this: `degenerate` cannot be the cover for an all-ocean window.
    assert _one("degenerate", {"land_std": float("nan")}) is False


def test_water_drops_above_the_cap_and_flags_above_the_tag():
    assert _one("water", {"wbm_water_frac": 0.3001}) is True
    assert _one("water", {"wbm_water_frac": 0.30}) is False
    assert _one("water_affected", {"wbm_water_frac": 0.0501}) is True
    assert _one("water_affected", {"wbm_water_frac": 0.05}) is False


def test_water_affected_is_a_tag_not_a_drop():
    flags = _flags({"wbm_water_frac": 0.10})
    assert bool(flags["water_affected"][0]) is True
    assert bool(flags["water"][0]) is False
    assert bool(flags["keep"][0]) is True


def test_edited_fires_on_either_edm_or_flm():
    assert _one("edited", {"edm_edited_frac": 0.2501}) is True
    assert _one("edited", {"flm_fill_frac": 0.2501}) is True
    assert _one("edited", {"edm_edited_frac": 0.25, "flm_fill_frac": 0.25}) is False


def test_coastline_needs_a_sea_pixel_not_just_water():
    # C2 (#34): 141 of 245 candidate windows were flat sea-level river with no ocean
    # pixel. The sea-touching term is what excludes them.
    assert _one("coastline", {"exact_zero_frac": 0.3, "wbm1_frac": 0.1}) is True
    assert _one("coastline", {"exact_zero_frac": 0.3, "wbm3_frac": 0.9}) is False


def test_coastline_has_no_lower_edge():
    # C4 §3.5 dropped `lo`: any exact-zero fraction below 0.75 qualifies, and the
    # `wbm1_frac > 0` term already implies exact_zero_frac > 0 at pixel level.
    assert _one("coastline", {"exact_zero_frac": 0.001, "wbm1_frac": 0.001}) is True
    assert _one("coastline", {"exact_zero_frac": 0.75, "wbm1_frac": 0.5}) is False


def test_coastline_is_a_tag_computed_over_the_whole_frame():
    # A coastline window that a drop gate also catches keeps its tag; the notebook
    # intersects with `keep` when it reports the stratum (C4 §3, "over the kept frame").
    flags = _flags({"exact_zero_frac": 0.5, "wbm1_frac": 0.9, "wbm_water_frac": 0.9})
    assert bool(flags["coastline"][0]) is True
    assert bool(flags["water"][0]) is True
    assert bool(flags["keep"][0]) is False


# --- keep, and what NaN does to it --------------------------------------------


def test_keep_is_the_complement_of_the_drop_gates():
    flags = _flags(
        {},  # clean
        {"exact_zero_frac": 0.9, "wbm_water_frac": 1.0, "land_std": float("nan")},
        {"land_std": 1.0},
        {"wbm_water_frac": 0.9},
        {"edm_edited_frac": 0.9},
    )
    assert flags["keep"].tolist() == [True, False, False, False, False]


def test_masks_missing_is_flagged_and_not_kept():
    # A mask this run could not read leaves NaN fractions (catalog._nan_mask_fractions).
    # `NaN > 0.30` is False, so without this the window would sail through the water
    # and edited caps unevaluated and land in Phase D's sampling frame.
    flags = _flags({"wbm_water_frac": float("nan"), "wbm1_frac": float("nan")})
    assert bool(flags["masks_missing"][0]) is True
    assert bool(flags["water"][0]) is False  # the predicate itself did not fire
    assert bool(flags["keep"][0]) is False  # but the window is not kept


def test_masks_missing_is_false_when_every_mask_column_is_known():
    assert _one("masks_missing", {}) is False


def test_gate_flags_are_boolean_arrays_of_the_frame_length():
    flags = _flags({}, {"land_std": 1.0}, {})
    assert set(flags) == set(gates.FLAG_COLUMNS)
    for name, arr in flags.items():
        assert arr.dtype == np.bool_, name
        assert arr.shape == (3,), name


def test_gate_flags_rejects_a_frame_missing_a_gate_input():
    frame = _frame({})
    del frame["wbm1_frac"]
    with pytest.raises(KeyError, match="wbm1_frac"):
        gates.gate_flags(frame)


# --- Applying the gates to a table: additive, idempotent ----------------------


def _table(*overrides: dict[str, float]):
    """A catalog-shaped :class:`pyarrow.Table` — the raw columns, no flags."""
    pa = pytest.importorskip("pyarrow")
    frame = _frame(*overrides)
    n = len(next(iter(frame.values())))
    cols: dict[str, object] = {}
    for name in catalog.COLUMNS:
        if name in frame:
            cols[name] = pa.array(frame[name], type=pa.float64())
        elif name in ("window_id", "stem", "band", "geocell"):
            cols[name] = pa.array([f"{name}-{i}" for i in range(n)], type=pa.string())
        elif name in ("win_row0", "win_col0"):
            cols[name] = pa.array(list(range(n)), type=pa.int32())
        else:
            cols[name] = pa.array([1.0] * n, type=pa.float64())
    return pa.table(cols, schema=catalog._schema())


def test_apply_gates_appends_flags_and_keeps_every_raw_column():
    table = _table({}, {"land_std": 1.0})
    gated = gates.apply_gates(table)
    assert gated.num_rows == table.num_rows  # nothing dropped — the whole point
    assert gated.column_names == list(gates.GATED_COLUMNS)
    for name in catalog.COLUMNS:
        assert gated.column(name).to_pylist() == table.column(name).to_pylist()
    assert gated.column("keep").to_pylist() == [True, False]


def test_apply_gates_is_idempotent():
    once = gates.apply_gates(_table({}, {"land_std": 1.0}))
    twice = gates.apply_gates(once)
    assert twice.column_names == once.column_names
    assert twice.to_pydict() == once.to_pydict()


def test_apply_gates_flag_columns_are_boolean():
    pa = pytest.importorskip("pyarrow")
    gated = gates.apply_gates(_table({}))
    for name in gates.FLAG_COLUMNS:
        assert gated.schema.field(name).type == pa.bool_()


# --- Attrition accounting -----------------------------------------------------


def test_attrition_separates_fires_from_sole_cause():
    # Row 0 clean; row 1 water only; rows 2-3 water AND degenerate. So `water` fires
    # on 3 and is the sole cause of 1; `degenerate` fires on 2 and is sole cause of 0.
    flags = _flags(
        {},
        {"wbm_water_frac": 0.9},
        {"wbm_water_frac": 0.9, "land_std": 1.0},
        {"wbm_water_frac": 0.9, "land_std": 1.0},
    )
    report = gates.attrition(flags)
    assert report.n_rows == 4
    assert report.n_kept == 1
    by_name = {g.gate: g for g in report.gates}
    assert (by_name["water"].fires, by_name["water"].sole_cause) == (3, 1)
    assert (by_name["degenerate"].fires, by_name["degenerate"].sole_cause) == (2, 0)
    assert (by_name["edited"].fires, by_name["edited"].sole_cause) == (0, 0)


def test_attrition_counts_the_tags_over_the_kept_frame():
    flags = _flags(
        {"wbm_water_frac": 0.10},  # kept, water_affected
        {"wbm_water_frac": 0.9},  # dropped, water_affected too
        {"exact_zero_frac": 0.2, "wbm1_frac": 0.1},  # kept coastline
        {"exact_zero_frac": 0.8, "wbm1_frac": 0.1, "wbm_water_frac": 0.9},  # dropped
    )
    report = gates.attrition(flags)
    assert report.n_water_affected == 1
    assert report.n_coastline == 1


def test_attrition_keep_rate_is_a_fraction_of_the_whole_frame():
    report = gates.attrition(_flags({}, {"land_std": 1.0}, {"land_std": 1.0}, {}))
    assert report.keep_rate == pytest.approx(0.5)


# --- The persisted gated catalog ----------------------------------------------


def _raw_catalog(tmp_path: Path, n_parts: int = 2) -> Path:
    """A tiny raw catalog on disk: ``n_parts`` part files plus a meta sidecar."""
    pytest.importorskip("pyarrow.parquet")
    out = tmp_path / "raw"
    table = _table({}, {"land_std": 1.0}, {"wbm_water_frac": 0.9})
    records = table.to_pylist()
    for i in range(n_parts):
        catalog.write_part(records, catalog.part_path(out, i))
    (out / catalog.META_FILE).write_text(
        json.dumps({"code_version": "deadbee", "snapshot_date": "2026-07-22", "gated": False})
    )
    return out


def test_write_gated_mirrors_the_parts_and_keeps_every_row(tmp_path: Path):
    src = _raw_catalog(tmp_path, n_parts=2)
    result = gates.write_gated(src, tmp_path / "gated")
    assert result.n_rows == 6 and result.n_parts == 2
    assert result.n_kept == 2  # one clean row per part
    table = catalog.read_catalog(tmp_path / "gated")
    assert table.num_rows == 6
    assert table.column_names == list(gates.GATED_COLUMNS)


def test_write_gated_writes_a_meta_recording_the_thresholds_and_counts(tmp_path: Path):
    src = _raw_catalog(tmp_path)
    gates.write_gated(src, tmp_path / "gated")
    meta = json.loads((tmp_path / "gated" / catalog.META_FILE).read_text())
    assert meta["gated"] is True
    assert meta["thresholds"]["zero_fill_min"] == gates.ZERO_FILL_MIN
    assert meta["thresholds"]["degenerate_land_std_m"] == gates.DEGENERATE_LAND_STD_M
    assert meta["source_catalog"]["code_version"] == "deadbee"
    assert meta["n_rows"] == 6 and meta["n_kept"] == 2  # 2 of the 3 rows drop, per part
    assert meta["columns"] == list(gates.GATED_COLUMNS)
    assert meta["attrition"]["gates"][0]["gate"] in gates.DROP_GATES


def test_write_gated_refuses_to_overwrite_the_raw_catalog(tmp_path: Path):
    # The raw frame is the thing that cost six hours of S3 reads. Writing the gated
    # parts over it would destroy the denominator every future threshold needs.
    src = _raw_catalog(tmp_path)
    with pytest.raises(ValueError, match="in place"):
        gates.write_gated(src, src)


def test_write_gated_is_rerunnable_over_an_existing_output(tmp_path: Path):
    # Re-gating after a threshold moves must not leave the old parts behind to be
    # concatenated with the new ones.
    src = _raw_catalog(tmp_path, n_parts=2)
    out = tmp_path / "gated"
    gates.write_gated(src, out)
    stale = catalog.part_path(out, 99)
    stale.write_bytes(b"not parquet")
    result = gates.write_gated(src, out)
    assert not stale.exists()
    assert result.n_parts == 2
    assert catalog.read_catalog(out).num_rows == 6


# --- The invariant Phase D depends on -----------------------------------------


def test_a_kept_window_always_has_a_finite_metric_vector(tmp_path: Path):
    # Phase D stratifies on the frozen six; a kept row with a NaN metric would break
    # k-means. The catalog NaNs the vector only when a window has no land pixel, and
    # such a window is 100% exact-zero, so `zero_fill` always catches it first.
    from dataset.geometry import parse_cog_key
    from dataset.masks import SyntheticMaskReader
    from dataset.tiling import SyntheticCogReader

    stem = "Copernicus_DSM_COG_10_N45_00_E006_00_DEM"
    n = 672
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    arr = (500.0 + 80.0 * np.sin(xx / 40.0) + 60.0 * np.cos(yy / 55.0)).astype(np.float32)
    arr[:, :224] = 0.0  # an ocean-only column: NaN metrics, NaN land_std
    masks = SyntheticMaskReader(
        {
            stem: {
                "EDM": np.ones((n, n), np.uint8),
                "FLM": np.full((n, n), 2, np.uint8),
                "WBM": np.zeros((n, n), np.uint8),
            }
        }
    )
    rows, _ = catalog.tile_rows(
        SyntheticCogReader({stem: arr}), masks, parse_cog_key(stem), factor=8
    )
    records = catalog.to_records(rows)
    frame = {
        name: np.array([r[name] for r in records], dtype=np.float64)
        for name in gates.GATE_INPUTS
    }
    keep = gates.gate_flags(frame)["keep"]
    assert keep.sum() == 6 and (~keep).sum() == 3
    for rec, kept in zip(records, keep):
        if kept:
            assert all(not math.isnan(rec[m]) for m in catalog.FROZEN_METRICS)
