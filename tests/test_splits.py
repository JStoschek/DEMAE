"""The geographic split: shortfall objective, block selection, and the committed artifact (#77).

Unit tests run against synthetic blocks — the selection is a deterministic function of
counts and centroids, so a hand-built frame exercises every rule. Two integration tests
read the shipped 500k manifest when it is present (skipped in CI, where ``data/`` is absent).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from dataset import config
from dataset import materialize
from dataset import splits


# --- relative_shortfall -------------------------------------------------------


def test_relative_shortfall_is_zero_on_an_exact_match():
    q = np.array([0.5, 0.3, 0.2])
    assert splits.relative_shortfall(np.array([50, 30, 20]), q) == 0.0


def test_relative_shortfall_counts_a_missing_stratum_as_one_whole_unit():
    # the rare stratum is absent: its share is short by 100% of itself, so it costs 1.0
    # however small it is — that is the property that makes the objective coverage-aware.
    q = np.array([0.9, 0.1])
    assert splits.relative_shortfall(np.array([100, 0]), q) == 1.0


def test_relative_shortfall_weighs_a_halved_common_stratum_like_a_halved_rare_one():
    q = np.array([0.8, 0.2])
    # p = (0.9, 0.1): the rare one is at half its share -> 0.5
    assert splits.relative_shortfall(np.array([90, 10]), q) == 0.5
    q2 = np.array([0.2, 0.8])
    # p = (0.1, 0.9): the common one is at half its share -> also 0.5
    assert splits.relative_shortfall(np.array([10, 90]), q2) == 0.5


# --- blocks -------------------------------------------------------------------


def _tiles(spec: dict[str, list[tuple[int, float, float]]]):
    """Expand ``{block_id: [(stratum, lon, lat), ...]}`` into flat tile columns."""
    ids, strata, lons, lats = [], [], [], []
    for bid, rows in spec.items():
        for s, lon, lat in rows:
            ids.append(bid)
            strata.append(s)
            lons.append(lon)
            lats.append(lat)
    return (
        np.array(ids, dtype=object),
        np.array(strata, dtype=np.int32),
        np.array(lons, dtype=np.float64),
        np.array(lats, dtype=np.float64),
    )


def test_blocks_group_tiles_by_id_and_count_every_stratum():
    ids, strata, lon, lat = _tiles({"a": [(0, 0.0, 0.0), (2, 2.0, 0.0)], "b": [(0, 10.0, 0.0)]})
    b = splits.blocks_from(ids, strata, lon, lat, k=3)
    assert list(b.ids) == ["a", "b"]
    assert b.counts.tolist() == [[1, 0, 1], [1, 0, 0]]
    assert b.n.tolist() == [2, 1]
    assert b.lon.tolist() == [1.0, 10.0]  # the block's own tile centroid, not the cell centre


def test_blocks_are_independent_of_the_order_rows_arrive_in():
    ids, strata, lon, lat = _tiles({"a": [(0, 0.0, 0.0), (1, 2.0, 1.0)], "b": [(1, 10.0, 0.0)]})
    order = np.array([2, 0, 1])
    a = splits.blocks_from(ids, strata, lon, lat, k=2)
    c = splits.blocks_from(ids[order], strata[order], lon[order], lat[order], k=2)
    assert list(a.ids) == list(c.ids)
    assert np.array_equal(a.counts, c.counts)
    assert np.allclose(a.lon, c.lon)


# --- select_test_geocells -----------------------------------------------------


def _grid_blocks(profiles: list[tuple[int, int]], *, spacing_deg: float = 30.0) -> splits.Blocks:
    """Blocks laid along the equator ``spacing_deg`` apart, each ``(n_s0, n_s1)`` tiles."""
    spec = {}
    for i, (a, b) in enumerate(profiles):
        lon = -180.0 + i * spacing_deg
        spec[f"c{i:02d}"] = [(0, lon, 0.0)] * a + [(1, lon, 0.0)] * b
    ids, strata, lon, lat = _tiles(spec)
    return splits.blocks_from(ids, strata, lon, lat, k=2)


def test_test_geocells_prefer_the_pool_that_matches_the_corpus_composition():
    # balanced cells and one-sided ones against a 50/50 reference: any two balanced cells
    # score a perfect zero and no pair containing a one-sided cell can.
    blocks = _grid_blocks([(50, 50), (100, 0), (50, 50), (0, 100), (50, 50), (100, 0)])
    sel = splits.select_test_geocells(
        blocks, budget=200, n_cells=2, reference=np.array([0.5, 0.5]), min_share=0.1, max_share=0.6
    )
    assert len(sel.ids) == 2
    assert sel.shortfall == 0.0
    assert set(sel.ids) <= {"c00", "c02", "c04"}


def test_test_geocells_stay_inside_the_tile_budget_and_the_per_cell_size_band():
    blocks = _grid_blocks([(50, 50), (5, 5), (60, 60), (40, 40), (55, 55), (45, 45)])
    sel = splits.select_test_geocells(blocks, budget=300, n_cells=3, min_share=0.05, max_share=0.4)
    assert len(sel.ids) == 3
    assert sel.n_tiles <= 300
    for bid in sel.ids:
        n = blocks.n[list(blocks.ids).index(bid)]
        assert 0.05 * 300 <= n <= 0.4 * 300, f"{bid} holds {n}, outside the band"
    assert "c01" not in sel.ids  # 10 tiles is below the floor: it would waste a slot


def test_test_geocells_keep_the_separation_floor_between_the_regions_they_pick():
    # 5-degree spacing at the equator is ~556 km, under a 1000 km floor, so no two
    # neighbours may both be picked.
    blocks = _grid_blocks([(50, 50)] * 6, spacing_deg=5.0)
    sel = splits.select_test_geocells(blocks, budget=400, n_cells=3, separation_km=1000.0)
    picked = sorted(int(b[1:]) for b in sel.ids)
    assert len(picked) == 3
    assert all(b - a >= 2 for a, b in zip(picked, picked[1:])), picked


def test_test_geocells_are_deterministic_and_independent_of_row_order():
    profiles = [(50, 50), (30, 70), (70, 30), (45, 55), (55, 45), (20, 80)]
    band = {"min_share": 0.1, "max_share": 0.5}
    a = splits.select_test_geocells(_grid_blocks(profiles), budget=250, n_cells=3, **band)
    b = splits.select_test_geocells(_grid_blocks(list(reversed(profiles))), budget=250, n_cells=3, **band)
    again = splits.select_test_geocells(_grid_blocks(profiles), budget=250, n_cells=3, **band)
    assert list(a.ids) == list(again.ids)
    # the same six cells under different ids: the *composition* chosen must match
    assert sorted(a.counts.tolist()) == sorted(b.counts.tolist())


def test_test_geocells_swap_pass_never_leaves_the_greedy_seed_worse():
    blocks = _grid_blocks([(90, 10), (10, 90), (50, 50), (80, 20), (20, 80), (60, 40), (40, 60)])
    sel = splits.select_test_geocells(blocks, budget=300, n_cells=3, min_share=0.1, max_share=0.4)
    assert sel.shortfall <= sel.seed_shortfall


# --- select_val_cogs ----------------------------------------------------------


def _cog_blocks(profiles: list[tuple[int, int]]) -> splits.Blocks:
    """COG-shaped blocks: ``(n_s0, n_s1)`` per COG, laid 1 degree apart."""
    spec = {}
    for i, (a, b) in enumerate(profiles):
        spec[f"cog{i:03d}"] = [(0, i * 1.0, 0.0)] * a + [(1, i * 1.0, 0.0)] * b
    ids, strata, lon, lat = _tiles(spec)
    return splits.blocks_from(ids, strata, lon, lat, k=2)


def test_val_quota_apportions_the_whole_budget_across_strata():
    blocks = _cog_blocks([(10, 0), (0, 10)] * 10)
    sel = splits.select_val_cogs(blocks, budget=100, reference=np.array([0.4, 0.6]))
    assert sel.quota.tolist() == [40, 60]
    assert sel.n_tiles <= 100


def test_val_takes_whole_cogs_and_never_overshoots_a_stratum_quota():
    blocks = _cog_blocks([(7, 1)] * 10 + [(1, 7)] * 10)
    sel = splits.select_val_cogs(blocks, budget=80, reference=np.array([0.5, 0.5]))
    for bid in sel.ids:
        i = list(blocks.ids).index(bid)
        assert sel.counts_by_block[bid].tolist() == blocks.counts[i].tolist()
    assert (sel.spent <= sel.quota).all()


def test_val_never_draws_from_an_excluded_block():
    blocks = _cog_blocks([(5, 5)] * 12)
    eligible = np.ones(len(blocks.ids), dtype=bool)
    eligible[:6] = False
    sel = splits.select_val_cogs(blocks, budget=40, eligible=eligible)
    assert set(sel.ids).isdisjoint({f"cog{i:03d}" for i in range(6)})


def test_val_draw_is_size_blind_and_does_not_collect_the_shallowest_cogs():
    # ten deep COGs and ten shallow ones per stratum. A shortfall-greedy draw would take
    # the 2-tile COGs because small steps track a target better; the hash-order draw must
    # not, because a COG the gates nearly emptied is not a representative COG (#77 §2).
    blocks = _cog_blocks([(40, 0)] * 50 + [(2, 0)] * 50 + [(0, 40)] * 50 + [(0, 2)] * 50)
    sel = splits.select_val_cogs(blocks, budget=2000, reference=np.array([0.5, 0.5]))
    pool_depth = float(blocks.n.mean())
    assert sel.mean_depth >= 0.5 * pool_depth, f"drew {len(sel.ids)} COGs of mean depth {sel.mean_depth}"


def test_val_draw_is_deterministic():
    blocks = _cog_blocks([(6, 2), (2, 6), (4, 4)] * 8)
    a = splits.select_val_cogs(blocks, budget=60)
    b = splits.select_val_cogs(blocks, budget=60)
    assert a.ids == b.ids


# --- assign_splits ------------------------------------------------------------

_CELLS = np.array(["+000_+000", "+000_+000", "+010_+000", "+010_+000", "+020_+000"])
_COGS = np.array(["A", "B", "C", "D", "E"])


def test_assign_labels_every_tile_exactly_once():
    out = splits.assign_splits(_CELLS, _COGS, test_cells=["+010_+000"], val_cogs=["B"])
    assert out.tolist() == ["train", "val", "test", "test", "train"]


def test_assign_holds_out_whole_blocks_so_no_block_straddles_the_line():
    cells = np.array(["+000_+000"] * 4 + ["+010_+000"] * 3)
    cogs = np.array(["A", "A", "B", "B", "C", "C", "D"])
    out = splits.assign_splits(cells, cogs, test_cells=["+010_+000"], val_cogs=["B"])
    # every COG carries exactly one label: the finest blocking unit is never cut
    for cog in np.unique(cogs):
        assert len(set(out[cogs == cog])) == 1, f"COG {cog} straddles the split"
    # a test geocell is wholly test, and no test tile lives outside one
    assert set(out[cells == "+010_+000"]) == {"test"}
    assert "test" not in set(out[cells != "+010_+000"])
    # val does straddle a geocell — that is the design, not a defect: val blocks at the COG
    assert set(out[cells == "+000_+000"]) == {"train", "val"}


def test_assign_gives_test_precedence_over_val():
    # a COG inside a test geocell cannot be drawn for val, but the labelling must still
    # be total and unambiguous if one ever is: the untouched-region guarantee wins.
    out = splits.assign_splits(_CELLS, _COGS, test_cells=["+010_+000"], val_cogs=["C"])
    assert out[2] == "test"


def test_assign_rejects_a_test_cell_the_corpus_does_not_hold():
    with pytest.raises(ValueError, match="not present"):
        splits.assign_splits(_CELLS, _COGS, test_cells=["+090_+080"], val_cogs=[])


# --- the committed artifact ---------------------------------------------------


def test_assignment_round_trips_through_the_committed_csv(tmp_path):
    tile_ids = np.array(["t0", "t1", "t2"])
    labels = np.array(["train", "val", "test"])
    path = splits.write_assignment(tile_ids, labels, tmp_path / "split.csv")
    back = splits.read_assignment(path)
    assert back == {"t0": "train", "t1": "val", "t2": "test"}


def test_assignment_csv_is_byte_stable_across_writes(tmp_path):
    tile_ids = np.array(["t2", "t0", "t1"])
    labels = np.array(["test", "train", "val"])
    a = splits.write_assignment(tile_ids, labels, tmp_path / "a.csv")
    b = splits.write_assignment(tile_ids, labels, tmp_path / "b.csv")
    assert a.read_bytes() == b.read_bytes()
    assert a.read_text().splitlines()[0] == "tile_id,split"


def test_assignment_write_refuses_a_label_outside_the_vocabulary(tmp_path):
    with pytest.raises(ValueError, match="unknown split"):
        splits.write_assignment(np.array(["t0"]), np.array(["holdout"]), tmp_path / "x.csv")


# --- filling the manifest column ----------------------------------------------


def _tiny_manifest(path):
    from dataset import manifest as manifest_mod

    rows = [
        manifest_mod.TileRecord(
            tile_id=f"t{i}", shard_id=0, row_index=i, source_cog=f"COG{i//2}",
            win_row0=0, win_col0=224 * i, lon=float(i), lat=0.0, lat_band="0-50",
            geocell_id="+000_+000" if i < 2 else "+010_+000", mean=1.0, std=2.0,
            land_frac=1.0, stratum_id=i % 2, provenance="base", slope_mean=1.0,
            local_relief=2.0, texture=0.1, elevation_relief_ratio=0.4, mean_elevation=100.0,
            geomorphon_entropy=2.0, edm_edited_frac=0.0, flm_fill_frac=0.0, wbm_water_frac=0.0,
            water_affected=False, coastline=False, content_hash=i + 1,
        )
        for i in range(4)
    ]
    manifest_mod.write_manifest(rows, path)
    return path


def test_filling_the_manifest_writes_the_column_and_leaves_everything_else_alone(tmp_path):
    import pyarrow.parquet as pq

    from dataset import manifest as manifest_mod

    path = _tiny_manifest(tmp_path / "manifest.parquet")
    before = pq.read_table(path)
    splits.fill_manifest_split(path, {"t0": "train", "t1": "val", "t2": "test", "t3": "test"})
    after = pq.read_table(path)

    assert after.schema == manifest_mod.MANIFEST_SCHEMA
    assert after.column_names == before.column_names
    assert after["tile_id"].to_pylist() == before["tile_id"].to_pylist()  # row order preserved
    assert after["split"].to_pylist() == ["train", "val", "test", "test"]
    for col in before.column_names:
        if col != "split":
            assert after[col].to_pylist() == before[col].to_pylist()


def test_filling_the_manifest_twice_is_a_no_op(tmp_path):
    path = _tiny_manifest(tmp_path / "manifest.parquet")
    assignment = {"t0": "train", "t1": "val", "t2": "test", "t3": "test"}
    splits.fill_manifest_split(path, assignment)
    first = path.read_bytes()
    splits.fill_manifest_split(path, assignment)
    assert path.read_bytes() == first


def test_filling_the_manifest_refuses_an_assignment_that_misses_a_tile(tmp_path):
    path = _tiny_manifest(tmp_path / "manifest.parquet")
    with pytest.raises(ValueError, match="does not cover"):
        splits.fill_manifest_split(path, {"t0": "train", "t1": "val"})


# --- plan_splits, end to end --------------------------------------------------


def _synthetic_corpus(n_cells=10, cogs_per_cell=4, tiles_per_cog=10, k=4):
    """A toy corpus: ``n_cells`` geocells 20 degrees apart, each holding whole COGs."""
    cell, cog, stratum, lon, lat, tid = [], [], [], [], [], []
    for c in range(n_cells):
        base_lon = -180.0 + c * 20.0
        for g in range(cogs_per_cell):
            for t in range(tiles_per_cog):
                cell.append(f"{int(base_lon):+04d}_+000")
                cog.append(f"C{c:02d}G{g}")
                stratum.append((c + g + t) % k)
                lon.append(base_lon + g * 0.5)
                lat.append(0.0)
                tid.append(f"C{c:02d}G{g}T{t}")
    return (
        np.array(tid),
        np.array(cell),
        np.array(cog),
        np.array(stratum, dtype=np.int32),
        np.array(lon),
        np.array(lat),
    )


def test_plan_splits_holds_out_whole_cells_and_whole_cogs_and_labels_the_rest_train():
    tid, cell, cog, stratum, lon, lat = _synthetic_corpus()
    plan = splits.plan_splits(
        tile_id=tid, geocell_id=cell, source_cog=cog, stratum_id=stratum, lon=lon, lat=lat,
        k=4, n_cells=2, test_fraction=0.2, val_fraction=0.1,
        min_share=0.1, max_share=0.6, separation_km=1000.0,
    )
    assert set(plan.labels) == {"train", "val", "test"}
    assert len(plan.test.ids) == 2
    for c in plan.test.ids:
        assert set(plan.labels[cell == c]) == {"test"}
    for g in np.unique(cog):
        assert len(set(plan.labels[cog == g])) == 1, f"COG {g} straddles the split"


def test_plan_splits_refuses_a_hold_out_that_misses_a_stratum():
    # plan §9 requires all 20 strata in the held-out pool. The shortfall objective prices a
    # missing stratum at a whole unit, which makes that *likely* — the check makes it true,
    # so a re-draw at other parameters cannot quietly ship a pool with a terrain family
    # nothing evaluates. Here stratum 3 lives in a single cell the budget cannot afford.
    tid, cell, cog, stratum, lon, lat = _synthetic_corpus(k=3)
    stratum = stratum.copy()
    stratum[cog == "C09G3"] = 3  # a fourth stratum, in one COG of the largest-index cell
    with pytest.raises(ValueError, match="misses stratum"):
        splits.plan_splits(
            tile_id=tid, geocell_id=cell, source_cog=cog, stratum_id=stratum, lon=lon, lat=lat,
            k=4, n_cells=2, test_fraction=0.2, val_fraction=0.1,
            min_share=0.1, max_share=0.6, separation_km=1000.0,
        )


def test_plan_splits_never_draws_val_from_a_test_geocell():
    tid, cell, cog, stratum, lon, lat = _synthetic_corpus()
    plan = splits.plan_splits(
        tile_id=tid, geocell_id=cell, source_cog=cog, stratum_id=stratum, lon=lon, lat=lat,
        k=4, n_cells=2, test_fraction=0.2, val_fraction=0.1,
        min_share=0.1, max_share=0.6, separation_km=1000.0,
    )
    val_cells = {cell[list(cog).index(c)] for c in plan.val.ids}
    assert val_cells.isdisjoint(plan.test.ids)


# --- against the shipped corpus (skipped where data/ is absent) ---------------

_MANIFEST = Path("data/corpus_500k/manifest.parquet")
_ASSIGNMENT = Path(config.SPLIT_ASSIGNMENT_PATH)

_needs_corpus = pytest.mark.skipif(not _MANIFEST.exists(), reason="500k corpus not present")
_needs_artifact = pytest.mark.skipif(not _ASSIGNMENT.exists(), reason="split artifact not committed")


@_needs_corpus
@_needs_artifact
def test_the_draw_reproduces_the_committed_assignment_byte_for_byte(tmp_path):
    """The determinism contract: re-running the draw must re-emit the shipped file."""
    plan = splits.plan_from_manifest(_MANIFEST)
    again = splits.write_assignment(plan.tile_id, plan.labels, tmp_path / "again.csv")
    assert materialize.file_md5(again) == materialize.file_md5(_ASSIGNMENT)


@_needs_corpus
@_needs_artifact
def test_the_committed_split_leaks_no_block_across_the_line():
    table = pq.read_table(_MANIFEST, columns=["tile_id", "geocell_id", "source_cog", "stratum_id"])
    assignment = splits.read_assignment(_ASSIGNMENT)
    label = np.array([assignment[t] for t in table["tile_id"].to_pylist()])
    cells = table["geocell_id"].to_numpy(zero_copy_only=False)
    cogs = table["source_cog"].to_numpy(zero_copy_only=False)
    stratum = table["stratum_id"].to_numpy()

    blocks = json.loads(Path(config.SPLIT_BLOCKS_PATH).read_text())
    test_cells = set(blocks["test"]["cells"])

    # no source COG is split — the finest blocking unit the corpus card allows
    order = np.argsort(cogs, kind="stable")
    boundaries = np.flatnonzero(cogs[order][1:] != cogs[order][:-1])
    for start, stop in zip(np.r_[0, boundaries + 1], np.r_[boundaries + 1, len(order)]):
        assert len(set(label[order][start:stop])) == 1, f"COG {cogs[order][start]} straddles"

    # every test geocell is wholly test, and no test tile lives outside one
    assert set(label[np.isin(cells, list(test_cells))]) == {"test"}
    assert "test" not in set(label[~np.isin(cells, list(test_cells))])

    # all 20 strata appear in both held-out pools (plan §9)
    for s in ("val", "test"):
        assert len(np.unique(stratum[label == s])) == 20, f"{s} misses a stratum"

    shares = {s: float((label == s).mean()) for s in splits.SPLITS}
    assert shares["train"] == pytest.approx(0.95, abs=0.005)
    assert shares["val"] == pytest.approx(0.02, abs=0.002)
    assert shares["test"] == pytest.approx(0.03, abs=0.003)
