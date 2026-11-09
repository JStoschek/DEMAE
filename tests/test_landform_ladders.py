"""§3.3's three geocell ladders, through ``python -m benchmark.launch --task landform`` (#200).

Plan §3.3 is the whole specification: each ladder is one walk over Task 1's head-train list in
``tie_rank`` order keyed by that ladder's draw seed (geocells first, then tiles within a cell),
and a rung is a prefix of the walk: an exact tile budget, whole cells plus at most one partial
cell whose id is recorded. ``freeze`` commits the three walks and a coverage report beside each;
``run`` binds seed *i* to ladder *i* and refuses any other pairing.

The fixture corpus walks the same rule at budgets it holds (``ladders.budgets``). The other
``freeze`` stages belong to other tickets and need their own fakes, so these tests run the
ladder stage alone (:func:`only_the_ladder_stage`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmark import landform, launch
from benchmark.landform_config import LandformConfig
from dataset.splits import tie_rank
from pretrain.artifact import read_tile_strata, write_tile_strata

#: The fixture's train geocells and how many COGs each holds. Two tiles a COG, so cells of
#: 2, 6, 4, 10, 2 and 8 tiles: 32 in all, uneven on purpose.
TRAIN_CELLS = {
    "-010_+000": 1,
    "+000_+000": 3,
    "+010_+000": 2,
    "+020_+000": 5,
    "+030_+000": 1,
    "+040_+000": 4,
}
TILES_PER_COG = 2
TRAIN_TILES = 2 * sum(TRAIN_CELLS.values())
#: Scaled-down rungs over the fixture's 32 train tiles; the top one is the whole list.
BUDGETS = (5, 13, TRAIN_TILES)

#: Ladder *i*'s draw seed under the declared rule: the project seed (0xDEAE) plus one, plus *i*.
DRAW_SEEDS = (57007, 57008, 57009)


def _cogs():
    cogs, lon = [], 0
    for cell, n in TRAIN_CELLS.items():
        for _ in range(n):
            cogs.append((f"Copernicus_DSM_COG_10_N10_00_E{lon:03d}_00_DEM", cell, "train"))
            lon += 1
    # A test geocell, which is what the ruggedness bands are cut over.
    for _ in range(2):
        cogs.append((f"Copernicus_DSM_COG_10_N10_00_E{lon:03d}_00_DEM", "+050_+000", "test"))
        lon += 1
    return cogs


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from tests.conftest import build_corpus

    root = tmp_path_factory.mktemp("ladder_corpus")
    return build_corpus(
        root,
        _cogs(),
        tiles_per_cog=TILES_PER_COG,
        metrics_fn=lambda c, j: {"stratum_id": (2 * c + j) % 20, "slope_mean": 0.03 * c + 0.01 * j},
    )


def _manifest(corpus) -> dict[str, dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(
        corpus / "manifest.parquet", columns=["tile_id", "geocell_id", "stratum_id", "split"]
    ).to_pylist()
    return {row["tile_id"]: row for row in rows}


@pytest.fixture
def box(tmp_path, corpus):
    """A box holding Task 1's frozen head-train list: every train tile of the fixture corpus."""
    train = sorted(
        (t, row["stratum_id"]) for t, row in _manifest(corpus).items() if row["split"] == "train"
    )
    assert len(train) == TRAIN_TILES
    write_tile_strata(train, tmp_path / "task1" / "head_train_tiles.csv")
    (tmp_path / "task2").mkdir()
    return tmp_path


@pytest.fixture
def flags(box, corpus):
    return [
        "--task", "landform",
        "--set", f'paths.artifacts_dir="{box / "task2"}"',
        "--set", f'paths.task1_artifacts_dir="{box / "task1"}"',
        "--set", f'paths.results_path="{box / "results.parquet"}"',
        "--set", f'paths.corpus_dir="{corpus}"',
        "--set", f"ladders.budgets={list(BUDGETS)}",
    ]


@pytest.fixture
def only_the_ladder_stage(monkeypatch):
    stages = tuple(stage for stage in landform.FREEZE_STAGES if stage.label == "ladders")
    assert len(stages) == 1 and stages[0].write is not None, "the ladder stage is not built"
    monkeypatch.setattr(landform, "FREEZE_STAGES", stages)


@pytest.fixture
def frozen(box, flags, only_the_ladder_stage, capsys):
    """``freeze --task landform`` run once; returns what it printed."""
    assert launch.main(["freeze", *flags]) == 0
    return capsys.readouterr().out


def _ladder(box, i) -> list[str]:
    return [t for t, _ in read_tile_strata(box / "task2" / f"ladder_{i}.csv", what="ladder")]


def _coverage(box, i) -> dict:
    return json.loads((box / "task2" / f"ladder_{i}_coverage.json").read_text())


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _snapshot(directory: Path) -> dict[str, str]:
    return {p.name: _md5(p) for p in sorted(directory.iterdir())}


# --- freeze writes three ladders and their coverage reports -------------------


def test_freeze_writes_three_ladders_and_a_coverage_report_beside_each(box, frozen):
    assert sorted(p.name for p in (box / "task2").iterdir()) == [
        "ladder_0.csv", "ladder_0_coverage.json",
        "ladder_1.csv", "ladder_1_coverage.json",
        "ladder_2.csv", "ladder_2_coverage.json",
    ]
    for i in range(3):
        assert sorted(_ladder(box, i)) == sorted(
            t for t, _ in read_tile_strata(box / "task1" / "head_train_tiles.csv", what="x")
        ), "at 100% every ladder is the whole head-train list"
        report = _coverage(box, i)
        assert report["ladder"] == i
        assert report["draw_seed"] == DRAW_SEEDS[i]
        assert report["ladder_md5"] == _md5(box / "task2" / f"ladder_{i}.csv")
        assert report["head_train_md5"] == _md5(box / "task1" / "head_train_tiles.csv")
        assert f"ladder {i}" in frozen and report["ladder_md5"] in frozen


def test_the_three_ladders_are_three_different_walks(box, frozen):
    walks = [_ladder(box, i) for i in range(3)]
    assert walks[0] != walks[1] != walks[2] != walks[0]


def test_each_walk_is_cells_in_keyed_order_then_tiles_in_keyed_order(box, corpus, frozen):
    """Story 27, checked against the plan's sentence rather than the module: ``tie_rank`` keyed
    with the ladder's draw seed over geocell ids, then over tile ids within a cell."""
    manifest = _manifest(corpus)
    for i, seed in enumerate(DRAW_SEEDS):
        walk = _ladder(box, i)
        cells = sorted(TRAIN_CELLS)
        cell_order = [cells[k] for k in np.argsort(tie_rank(np.array(cells), seed))]
        want = []
        for cell in cell_order:
            ids = sorted(t for t, row in manifest.items() if row["geocell_id"] == cell)
            want += [ids[k] for k in np.argsort(tie_rank(np.array(ids), seed))]
        assert walk == want, i


def test_every_rung_lands_on_its_budget_with_at_most_one_partial_cell_whose_id_is_recorded(
    box, corpus, frozen
):
    """Story 28 on the fixture: counted from the committed walk and the manifest."""
    manifest = _manifest(corpus)
    size = {cell: 2 * n for cell, n in TRAIN_CELLS.items()}
    for i in range(3):
        walk, report = _ladder(box, i), _coverage(box, i)
        assert [r["rung"] for r in report["rungs"]] == ["1%", "10%", "100%"]
        for budget, rung in zip(BUDGETS, report["rungs"]):
            prefix = walk[:budget]
            held: dict[str, int] = {}
            for t in prefix:
                held[manifest[t]["geocell_id"]] = held.get(manifest[t]["geocell_id"], 0) + 1
            partial = [cell for cell, n in held.items() if n < size[cell]]
            assert rung["tiles"] == budget == len(prefix)
            assert len(partial) <= 1
            assert rung["partial_cell"] == (
                None if not partial else {"geocell_id": partial[0], "tiles": held[partial[0]],
                                          "of": size[partial[0]]}
            )
            assert sorted(rung["whole_cells"]) == sorted(c for c in held if c not in partial)


def test_each_rung_reports_its_strata_and_band_coverage(box, corpus, frozen):
    """Story 30's coverage half. The per-class train-pixel counts need labels (#201, #202)."""
    manifest = _manifest(corpus)
    for i in range(3):
        walk, report = _ladder(box, i), _coverage(box, i)
        assert report["strata_k"] == 20
        assert len(report["bands"]["edges"]) == 5
        for budget, rung in zip(BUDGETS, report["rungs"]):
            strata = sorted({manifest[t]["stratum_id"] for t in walk[:budget]})
            assert rung["strata"] == {"present": len(strata), "ids": strata}
            assert sorted(rung["bands"]["tiles"]) == ["R1", "R2", "R3", "R4"]
            assert sum(rung["bands"]["tiles"].values()) == budget
            assert rung["bands"]["present"] == sum(
                1 for n in rung["bands"]["tiles"].values() if n
            )
            assert rung["class_pixels"] is None
        assert "#201" in report["class_pixels_note"]


def test_freeze_prints_each_rung_s_coverage(frozen):
    lines = frozen.splitlines()
    for rung in ("1%", "10%", "100%"):
        assert any(rung in line and "strata" in line and "bands" in line for line in lines)


def test_a_second_freeze_is_byte_identical(box, flags, frozen, capsys):
    before = _snapshot(box / "task2")
    assert launch.main(["freeze", *flags]) == 0
    assert _snapshot(box / "task2") == before


def test_freeze_refuses_to_overwrite_a_ladder_whose_walk_moved(box, flags, frozen):
    before = _snapshot(box / "task2")
    with pytest.raises(FileExistsError, match="ladder"):
        launch.main(["freeze", *flags, "--set", "ladders.first_draw_seed=1"])
    assert _snapshot(box / "task2") == before


def test_freeze_refuses_to_overwrite_a_hand_edited_ladder(box, flags, frozen):
    path = box / "task2" / "ladder_1.csv"
    rows = read_tile_strata(path, what="ladder")
    write_tile_strata([rows[1], rows[0], *rows[2:]], path)
    with pytest.raises(FileExistsError, match="ladder"):
        launch.main(["freeze", *flags])


def test_freeze_refuses_a_top_rung_that_is_not_the_whole_list(box, flags, only_the_ladder_stage):
    with pytest.raises(ValueError, match="whole head-train list"):
        launch.main(["freeze", *flags, "--set", f"ladders.budgets=[5, 13, {TRAIN_TILES - 1}]"])
    assert list((box / "task2").iterdir()) == []


def test_freeze_refuses_a_head_train_list_naming_a_tile_outside_train(
    box, corpus, flags, only_the_ladder_stage
):
    """The list was containment-checked when Task 1 drew it; a hand edit since is caught."""
    manifest = _manifest(corpus)
    stray = next(t for t, row in manifest.items() if row["split"] == "test")
    path = box / "task1" / "head_train_tiles.csv"
    rows = read_tile_strata(path, what="head-train list")
    write_tile_strata(sorted([*rows[1:], (stray, manifest[stray]["stratum_id"])]), path)
    with pytest.raises(ValueError, match="not a train tile"):
        launch.main(["freeze", *flags])


@pytest.mark.parametrize(
    "budgets, refusal",
    [("[5, 5, 32]", "must ascend"), ("[5, 32]", "one tile budget per rung")],
)
def test_budgets_that_are_not_three_nested_rungs_are_refused(budgets, refusal, flags):
    with pytest.raises(ValueError, match=refusal):
        launch.main(["freeze", "--dry-run", *flags, "--set", f"ladders.budgets={budgets}"])


# --- the walk itself, below the seam ------------------------------------------


def _walk(cells: dict[str, int], budgets, seed=DRAW_SEEDS[0]):
    from benchmark import ladders

    tile_id = [f"{cell}_t{k}" for cell, n in cells.items() for k in range(n)]
    geocell = [cell for cell, n in cells.items() for _ in range(n)]
    return ladders.walk(
        tile_id, [0] * len(tile_id), geocell, index=0, draw_seed=seed, budgets=budgets
    )


def _cell_order(cells, seed=DRAW_SEEDS[0]):
    names = sorted(cells)
    return [names[k] for k in np.argsort(tie_rank(np.array(names), seed))]


CELLS = {"a": 7, "b": 3, "c": 5, "d": 9, "e": 4}


def test_a_rung_ending_on_a_cell_boundary_has_no_partial_cell():
    first, second = _cell_order(CELLS)[:2]
    exact = CELLS[first] + CELLS[second]
    ladder = _walk(CELLS, (exact, exact + 1, sum(CELLS.values())))
    rung = ladder.rungs[0]
    assert rung.partial_cell is None
    assert sorted(rung.whole_cells) == sorted([first, second])


def test_the_cell_that_would_overshoot_contributes_just_enough_tiles():
    first, second = _cell_order(CELLS)[:2]
    ladder = _walk(CELLS, (CELLS[first] + 1, CELLS[first] + CELLS[second] + 1, 28))
    rung = ladder.rungs[0]
    assert len(rung.tile_id) == CELLS[first] + 1
    assert rung.whole_cells == (first,)
    assert rung.partial_cell == second and rung.partial_tiles == 1


def test_a_first_cell_larger_than_the_whole_budget_is_the_rung_s_one_partial_cell():
    first = _cell_order(CELLS)[0]
    ladder = _walk(CELLS, (1, 2, 28))
    assert ladder.rungs[0].whole_cells == ()
    assert ladder.rungs[0].partial_cell == first
    assert ladder.rungs[1].partial_cell == first


def test_nesting_is_asserted_on_every_ladder_in_code(monkeypatch):
    """Story 29: the walk checks its own rungs, so a later edit cannot break nesting quietly."""
    from benchmark import ladders

    real_rung = ladders.Ladder.rung_at

    def scrambled(self, k):
        rung = real_rung(self, k)
        if k == 0:  # the smallest rung names a tile the next one does not hold
            return rung.__class__(**{**rung.__dict__, "tile_id": rung.tile_id[::-1]})
        return rung

    monkeypatch.setattr(ladders.Ladder, "rung_at", scrambled)
    with pytest.raises(AssertionError, match="not a prefix"):
        _walk(CELLS, (8, 15, 28))


def test_a_redraw_from_the_committed_seed_is_byte_identical():
    from benchmark import ladders

    a, b = _walk(CELLS, (8, 15, 28)), _walk(CELLS, (8, 15, 28))
    assert ladders.render(a) == ladders.render(b)
    assert ladders.render(a) != ladders.render(_walk(CELLS, (8, 15, 28), seed=DRAW_SEEDS[1]))


# --- the resolver binds seed i to ladder i ------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_dry_run_with_seed_i_resolves_ladder_i_and_prints_its_md5(seed, box, flags, frozen,
                                                                     capsys):
    assert launch.main(["run", "--dry-run", "--rung", "10%", "--seed", str(seed), *flags]) == 0
    out = capsys.readouterr().out
    path = box / "task2" / f"ladder_{seed}.csv"
    assert f"{path}  md5 {_md5(path)}" in out
    assert f"the first 13 tiles of ladder {seed} (draw seed {DRAW_SEEDS[seed]})" in out
    for other in {0, 1, 2} - {seed}:
        assert f"ladder_{other}.csv" not in out


def test_the_ladder_md5_is_in_the_resolved_cell(box, flags, frozen):
    args = launch.build_parser().parse_args(["run", "--seed", "1", *flags])
    resolution = landform.resolve(args)
    ladder = next(item for item in resolution.inputs if item.key == "ladder")
    assert ladder.to_dict() == {
        "path": str(box / "task2" / "ladder_1.csv"),
        "md5": _md5(box / "task2" / "ladder_1.csv"),
        "required": True,
    }


def test_a_dry_run_before_freeze_says_the_ladder_is_not_frozen(box, flags, capsys):
    assert launch.main(["run", "--dry-run", "--seed", "1", *flags]) == 0
    out = capsys.readouterr().out
    assert any(
        line.startswith("ladder") and "ladder_1.csv" in line and "not frozen yet" in line
        for line in out.splitlines()
    )


def test_a_seed_with_no_ladder_is_refused(flags, capsys):
    with pytest.raises(ValueError, match="no ladder 3"):
        launch.main(["run", "--dry-run", "--seed", "3", *flags])
    assert capsys.readouterr().out == ""


def test_tuning_on_a_ladder_other_than_its_seed_is_refused(flags, capsys):
    with pytest.raises(ValueError, match="seed 0 trains on ladder 0, not ladder 1"):
        launch.main(["run", "--dry-run", *flags, "--set", "schedule.tune_ladder=1"])
    assert capsys.readouterr().out == ""


def test_a_ladder_file_swapped_for_another_ladder_s_is_refused(box, flags, frozen, capsys):
    """Seed 0 paired with ladder 1's walk, by a copy over the committed file."""
    task2 = box / "task2"
    (task2 / "ladder_0.csv").write_bytes((task2 / "ladder_1.csv").read_bytes())
    with pytest.raises(ValueError, match="ladder_0.csv.*does not hold ladder 0"):
        launch.main(["run", "--dry-run", "--seed", "0", *flags])
    (task2 / "ladder_0_coverage.json").write_bytes((task2 / "ladder_1_coverage.json").read_bytes())
    with pytest.raises(ValueError, match="records ladder 1 under draw seed 57008"):
        launch.main(["run", "--dry-run", "--seed", "0", *flags])
    assert capsys.readouterr().out == ""


# --- the real head-train list -------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
REAL_MANIFEST = REPO / "data" / "corpus_500k" / "manifest.parquet"
REAL_HEAD_TRAIN = REPO / "artifacts" / "task1" / "head_train_tiles.csv"


@pytest.mark.skipif(not REAL_MANIFEST.is_file(), reason="the 500k manifest is not on this box")
def test_the_real_ladders_land_on_200_2000_and_20000_with_at_most_one_partial_cell():
    """§3.3 on the committed head-train list: 376 train cells, three exact budgets."""
    from benchmark import ladders
    from benchmark.voids import read_manifest_columns

    rows = read_tile_strata(REAL_HEAD_TRAIN, what="head-train list")
    cols = read_manifest_columns(REAL_MANIFEST.parent, ["tile_id", "geocell_id"])
    cell_of = dict(zip(cols["tile_id"].astype(str), cols["geocell_id"].astype(str)))
    cfg = LandformConfig().ladders
    for i in range(cfg.count):
        ladder = ladders.walk(
            [t for t, _ in rows], [s for _, s in rows], [cell_of[t] for t, _ in rows],
            index=i, draw_seed=cfg.draw_seed(i), budgets=cfg.budgets,
        )
        assert [len(r.tile_id) for r in ladder.rungs] == [200, 2_000, 20_000]
        top = ladder.rungs[-1]
        assert top.partial_cell is None and len(top.whole_cells) == 376


def test_a_ladder_whose_coverage_report_is_missing_is_refused_by_name(box, flags, frozen, capsys):
    """Seed 0 paired with ladder 1's walk, its report deleted so nothing can contradict it."""
    task2 = box / "task2"
    (task2 / "ladder_0.csv").write_bytes((task2 / "ladder_1.csv").read_bytes())
    (task2 / "ladder_0_coverage.json").unlink()
    with pytest.raises(ValueError, match="ladder_0_coverage.json.*absent"):
        launch.main(["run", "--dry-run", "--seed", "0", *flags])
    assert capsys.readouterr().out == ""


def test_the_ladder_count_is_l3_s_and_not_a_knob(flags, capsys):
    """ADR-0015 L3 registers three seeds; a `--set` cannot make seed 3 resolvable."""
    from benchmark import prereg

    assert LandformConfig().ladders.count == prereg.TASK2.rule("L3").value["seeds"] == 3
    with pytest.raises(ValueError, match="count"):
        launch.main(["run", "--dry-run", "--seed", "3", *flags, "--set", "ladders.count=4"])
    assert capsys.readouterr().out == ""
