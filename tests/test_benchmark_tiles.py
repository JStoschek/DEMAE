"""`benchmark.tiles` — the four committed tile lists and split containment (#149).

Driven through seam 1 (`python -m benchmark.launch freeze`) wherever a claim survives the trip,
which is the build spec's rule for this module: *assert the committed lists' composition and
that a contaminated split is refused*. The composition claims are asserted at their real budget
against the shipped corpus and skipped elsewhere — §3.1's ladder is fixed at 1k/5k/20k, so
nothing smaller than a 475,030-tile train pool can express the draw the benchmark commits.

The refusal tests matter more than the composition ones. A benchmark that scored on ground the
encoder pretrained on would print a perfectly normal table — the whole failure is invisible in
the numbers — and it is unrepairable without re-pretraining, so every way the draw can reach a
held-out block is asserted to raise rather than warn.
"""

from __future__ import annotations

import json
from dataclasses import replace
from inspect import signature
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from benchmark import launch, tiles
from benchmark.config import LABEL_EFFICIENCY_SIZES, SIZE_TILES, BenchmarkConfig, Tiles
from dataset import config as dataset_config
from dataset.materialize import file_md5
from pretrain.artifact import HEADER, read_tile_strata
from tests.conftest import build_corpus

CORPUS_500K = Path("data/corpus_500k")

#: Six COGs over three geocells: four train, one val COG beside them, and one geocell held out
#: whole as test. Twenty tiles each, so the conftest builder's ``stratum_id = j % 20`` gives
#: every pool all twenty strata and a proportional draw has something to apportion.
TILE_COGS = [
    ("Copernicus_DSM_COG_10_N45_00_E006_00_DEM", "+000_+040", "train"),
    ("Copernicus_DSM_COG_10_N45_00_E007_00_DEM", "+000_+040", "train"),
    ("Copernicus_DSM_COG_10_N46_00_E006_00_DEM", "+000_+040", "val"),
    ("Copernicus_DSM_COG_10_S01_00_E036_00_DEM", "+030_+000", "train"),
    ("Copernicus_DSM_COG_10_S02_00_E036_00_DEM", "+030_+000", "train"),
    ("Copernicus_DSM_COG_10_N55_00_E010_00_DEM", "+010_+050", "test"),
]

#: A draw this corpus can supply: 40 head-train of 80, 10 val of 20, 10 test of 20.
SMALL = Tiles(head_train=40, val=10, test=10)

#: §3.1's ladder at this corpus's scale — a twentieth, a quarter, and the whole draw.
SMALL_SIZES = {"1k": 2, "5k": 10, "20k": 40}


@pytest.fixture
def corpus(tmp_path):
    """A corpus with both blocking units expressed: a val COG, and a whole test geocell."""
    root = tmp_path / "corpus"
    build_corpus(root, TILE_COGS, tiles_per_cog=20)
    return root


@pytest.fixture
def blocks_path(tmp_path):
    """#77's committed block list for :data:`TILE_COGS` — one test geocell, one val COG."""
    path = tmp_path / "split_blocks.json"
    path.write_text(
        json.dumps(
            {
                "test": {"block": "geocell_id", "cells": ["+010_+050"]},
                "val": {
                    "block": "source_cog",
                    "cogs": ["Copernicus_DSM_COG_10_N46_00_E006_00_DEM"],
                },
            }
        )
    )
    return path


def draw(corpus, blocks_path, *, tiles_config=SMALL, head_train_sizes=SMALL_SIZES, **kwargs):
    """The draw under test, at :data:`SMALL`'s budget unless a test moves one.

    ``assignment_path`` defaults to unset, as `test_pretrain_data.py` defaults it for a
    `SplitDataset`: the committed CSV on this box describes the shipped corpus, not this one.
    """
    return tiles.draw(
        corpus / "manifest.parquet",
        tiles_config,
        blocks_path=blocks_path,
        head_train_sizes=head_train_sizes,
        **kwargs,
    )


def labels_of(corpus):
    """The corpus's ``split`` column, as a list ready to be edited and written back."""
    return pq.read_table(corpus / "manifest.parquet", columns=["split"])["split"].to_pylist()


def relabel(corpus, set_split, fn):
    """Rewrite the corpus's ``split`` column through ``fn(label) -> label``."""
    set_split(corpus, [fn(label) for label in labels_of(corpus)])
    return corpus


def relabel_one_test_tile(corpus, set_split):
    """Move a single tile out of the held-out geocell — the common accident."""
    labels = labels_of(corpus)
    labels[labels.index("test")] = "train"
    return set_split(corpus, labels)


# --- What the four draws are --------------------------------------------------


def test_the_draw_commits_a_list_per_size_plus_val_and_test(corpus, blocks_path):
    drawn = draw(corpus, blocks_path)
    assert [lst.name for lst in drawn.lists] == [
        "head_train_1k.csv",
        "head_train_5k.csv",
        "head_train_tiles.csv",
        "val_tiles.csv",
        "test_tiles.csv",
    ]
    assert [len(lst) for lst in drawn.lists] == [2, 10, 40, 10, 10]


def test_each_list_comes_only_from_the_geocells_of_its_own_split(corpus, blocks_path):
    drawn = draw(corpus, blocks_path)
    cols = pq.read_table(corpus / "manifest.parquet", columns=["tile_id", "split"])
    split_of = dict(zip(cols["tile_id"].to_pylist(), cols["split"].to_pylist()))
    for lst in drawn.lists:
        assert {split_of[t] for t in lst.tile_id} == {lst.split}, lst.name


def test_the_head_train_draw_is_proportional_over_the_strata(corpus, blocks_path):
    # 80 train tiles, four per stratum, 40 drawn: an even two from each of the twenty. A draw
    # that over-weighted a stratum would train the head on a terrain mix the corpus does not
    # have, and nothing in the table would say so.
    drawn = draw(corpus, blocks_path)
    per_stratum = {s: drawn.head_train["20k"].stratum_id.count(s) for s in range(20)}
    assert set(per_stratum.values()) == {2}


def test_the_test_draw_is_stratified_too_so_flat_and_rugged_are_both_present(corpus, blocks_path):
    # §3.1: "stratified over strata so flat and rugged terrain are both represented". Ten tiles
    # cannot cover twenty strata, but no stratum may take two while another takes none — that
    # is what largest-remainder apportionment buys at a budget below k.
    drawn = draw(corpus, blocks_path)
    assert len(set(drawn.test.stratum_id)) == 10


def test_a_list_carries_the_stratum_each_tile_was_committed_under(corpus, blocks_path):
    # The redundancy is the point, exactly as it is for the probe set: it is what lets a later
    # phase *catch* a manifest re-fit underneath a committed list instead of silently comparing
    # arms under two different label vectors.
    drawn = draw(corpus, blocks_path)
    cols = pq.read_table(corpus / "manifest.parquet", columns=["tile_id", "stratum_id"])
    stratum_of = dict(zip(cols["tile_id"].to_pylist(), cols["stratum_id"].to_pylist()))
    lst = drawn.head_train["20k"]
    assert [stratum_of[t] for t in lst.tile_id] == list(lst.stratum_id)


# --- Nesting ------------------------------------------------------------------


def test_the_label_efficiency_subsets_are_nested(corpus, blocks_path):
    drawn = draw(corpus, blocks_path)
    small, medium, large = (set(drawn.head_train[s].tile_id) for s in LABEL_EFFICIENCY_SIZES)
    assert small < medium < large


def test_nesting_is_asserted_rather_than_assumed(corpus, blocks_path):
    """The ladder is checked as set containment after the draw, not inferred from the quotas.

    Largest-remainder apportionment is not monotone in the house size (the Alabama paradox), so
    "the smaller draw is a prefix of the larger" is a property of the numbers rather than of the
    construction. C3's sweep would then measure draw luck instead of data volume, which is the
    one thing §3.1 nests the subsets to rule out.
    """
    drawn = draw(corpus, blocks_path)
    ladder = [drawn.head_train[s] for s in LABEL_EFFICIENCY_SIZES]
    ladder[0] = replace(ladder[0], tile_id=("nobody-drew-this",), stratum_id=(0,))
    with pytest.raises(ValueError, match="not nested"):
        tiles.assert_nested(ladder)


def test_a_ladder_that_does_not_end_at_the_committed_draw_is_refused(corpus, blocks_path):
    with pytest.raises(ValueError, match="nested inside the one committed draw"):
        draw(corpus, blocks_path, head_train_sizes={"1k": 2, "5k": 40, "20k": 10})


# --- Determinism --------------------------------------------------------------


def test_a_redraw_is_byte_identical(corpus, blocks_path, tmp_path):
    first = tiles.write(draw(corpus, blocks_path), tmp_path / "a")
    second = tiles.write(draw(corpus, blocks_path), tmp_path / "b")
    assert first == second
    assert set(first) == {lst.name for lst in draw(corpus, blocks_path).lists}


def test_the_committed_file_is_the_pretrain_artifact_format(corpus, blocks_path, tmp_path):
    tiles.write(draw(corpus, blocks_path), tmp_path)
    path = tmp_path / "head_train_tiles.csv"
    assert path.read_text().splitlines()[0] == HEADER
    rows = read_tile_strata(path, what="head-train list")
    assert len(rows) == 40
    # Sorted by tile_id, like the probe set: two runs that agree on the set produce the same
    # bytes whatever order the rows arrived in, which is what makes the md5 a contract.
    assert rows == sorted(rows)


def test_a_different_seed_is_a_different_draw(corpus, blocks_path):
    moved = draw(corpus, blocks_path, tiles_config=SMALL.replace(draw_seed=SMALL.draw_seed + 1))
    assert set(moved.head_train["20k"].tile_id) != set(
        draw(corpus, blocks_path).head_train["20k"].tile_id
    )


# --- Split containment --------------------------------------------------------


def test_a_train_tile_inside_a_test_geocell_is_refused(corpus, blocks_path, set_split):
    """The common accident: one tile relabelled inside a region the rest of which is held out.

    Caught by the blocks *derived* from the manifest's own column — the other nineteen test
    tiles still say the geocell is held out.
    """
    relabel_one_test_tile(corpus, set_split)
    with pytest.raises(ValueError, match="split containment violated"):
        draw(corpus, blocks_path)


def test_the_derived_blocks_catch_it_with_no_committed_list_at_all(corpus, set_split):
    # Half of "neither subsumes the other", asserted rather than asserted-about.
    relabel_one_test_tile(corpus, set_split)
    with pytest.raises(ValueError, match="split containment violated"):
        draw(corpus, None)


def test_a_whole_test_geocell_relabelled_train_is_refused(corpus, blocks_path, set_split):
    """The other half: nothing left in the manifest looks held out.

    Only the *committed* block list still names the geocell, which is why the check is against
    the union of the two and why neither is dropped as redundant.
    """
    relabel(corpus, set_split, lambda s: "train" if s == "test" else s)
    with pytest.raises(ValueError, match="split containment violated"):
        draw(corpus, blocks_path)


def test_the_derived_blocks_alone_would_not_have_caught_that_one(corpus, set_split):
    # The converse of the test above: with the committed list gone, a wholesale relabelling
    # leaves a corpus that looks perfectly clean. This is the check that would have passed.
    relabel(corpus, set_split, lambda s: "train" if s == "test" else s)
    with pytest.raises(ValueError, match="no tile carries split 'test'"):
        draw(corpus, None)


def test_a_block_list_from_another_corpus_is_refused_rather_than_matching_nothing(
    corpus, tmp_path
):
    stale = tmp_path / "stale_blocks.json"
    stale.write_text(
        json.dumps({"test": {"cells": ["+090_+090"]}, "val": {"cogs": ["not_a_cog"]}})
    )
    with pytest.raises(ValueError, match="blocks this corpus does not contain"):
        draw(corpus, stale)


def test_a_partly_split_corpus_is_refused_rather_than_drawn_from(corpus, blocks_path, set_split):
    # Its nulls read as "not yet split" while the rows they mark are perfectly drawable, so a
    # draw would quietly sample tiles nothing decided the fate of. The wording is
    # `pretrain.fingerprint.split_state`'s, so the fingerprint and the draw cannot disagree
    # about whether a corpus is split.
    labels = labels_of(corpus)
    labels[0] = None
    set_split(corpus, labels)
    with pytest.raises(ValueError, match="split column is filled on all but 1 of 120 rows"):
        draw(corpus, blocks_path)


def test_a_manifest_that_disagrees_with_the_committed_assignment_is_refused(
    corpus, blocks_path, set_split, write_assignment, tmp_path
):
    """The fifth refusal a Phase-1 Dataset makes, made here for the same reason (#89).

    The blocks defend the *regions*; this defends the *claim*. A results table says which split
    it was scored against by quoting #77's assignment, and a manifest column edited out from
    under that file would make the claim a lie no reader of the table could catch.
    """
    path = write_assignment(tmp_path / "split.csv", corpus)
    relabel(corpus, set_split, lambda s: "train" if s == "val" else s)
    with pytest.raises(ValueError, match="disagrees with"):
        draw(corpus, blocks_path, assignment_path=path)


def test_a_pool_too_small_for_its_budget_is_refused_rather_than_short(corpus, blocks_path):
    # The probe set returns fewer than asked when the split is thin, deliberately. A benchmark
    # cannot: a "20,000-tile head-train set" holding 12,000 is a different benchmark reported
    # under the same name, and every label-efficiency number would be off its own axis.
    with pytest.raises(ValueError, match="holds 80"):
        draw(
            corpus,
            blocks_path,
            tiles_config=SMALL.replace(head_train=200),
            head_train_sizes={**SMALL_SIZES, "20k": 200},
        )


# --- Through seam 1 -----------------------------------------------------------


@pytest.fixture
def flags(corpus, blocks_path, tmp_path):
    # The assignment is pointed away from the committed one, which describes the shipped corpus
    # and not this six-COG one — an absent assignment is skipped, exactly as `SplitDataset`
    # skips it. That the check *fires* when the file is there is asserted above, and against
    # the shipped corpus below, where the real committed CSV is what `freeze` reads.
    return [
        "--set", f'paths.corpus_dir="{corpus}"',
        "--set", f'paths.split_blocks_path="{blocks_path}"',
        "--set", f'paths.split_assignment_path="{tmp_path / "no-assignment.csv"}"',
        "--artifacts-dir", str(tmp_path / "artifacts"),
    ]


def test_freeze_refuses_a_contaminated_split_before_it_writes_anything(
    corpus, set_split, flags, tmp_path
):
    # The containment check runs before the budget check, which is why this raises on a corpus
    # far too small for §3.1's 20,000 — a leak is refused first, whatever else is wrong.
    relabel(corpus, set_split, lambda s: "train" if s == "test" else s)
    with pytest.raises(ValueError, match="split containment violated"):
        launch.main(["freeze", *flags])
    assert not list((tmp_path / "artifacts").glob("*.csv"))


def test_a_dry_run_freeze_still_writes_nothing(corpus, flags, tmp_path):
    assert launch.main(["freeze", "--dry-run", *flags]) == 0
    assert not (tmp_path / "artifacts").exists()


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """One real `freeze` of the shipped corpus, shared by the class below.

    The `NotImplementedError` is the expected outcome and not a failure to write: `freeze`
    commits §3.1's lists and then names the stage the last ticket builds.

    ``masks.bank_draws_per_tile`` is held to 1 and the bank is pointed at a temporary path,
    because `freeze` runs every stage before the one that raises and #151's bank is 20,000
    tiles x K = 50 — 1,000,000 draws and twenty minutes, written to ``paths.mask_bank_path``,
    which defaults to a real location under ``data/``. What this file is about is §3.1's tile
    lists; `tests/test_benchmark_bank.py` is where the bank's own shape is asserted.
    """
    directory = tmp_path_factory.mktemp("task1")
    bank = tmp_path_factory.mktemp("bank") / "train_mask_bank.parquet"
    with pytest.raises(NotImplementedError, match="dataset card"):
        launch.main(
            [
                "freeze",
                "--artifacts-dir", str(directory),
                "--set", "masks.bank_draws_per_tile=1",
                "--set", f'paths.mask_bank_path="{bank}"',
            ]
        )
    return directory


@pytest.mark.skipif(
    not (CORPUS_500K / "manifest.parquet").exists()
    or not Path(dataset_config.SPLIT_BLOCKS_PATH).exists(),
    reason="the shipped corpus or the committed split (#77) is not present",
)
class TestAgainstTheShippedCorpus:
    """The composition claims at their real budget — the only corpus that can carry them.

    §3.1's numbers are 20,000 / 2,000 / 5,000 against a 475,030-tile train pool, and the ladder
    is fixed at 1k/5k/20k. Nothing smaller can express the draw the benchmark actually commits.
    """

    def test_freeze_writes_the_five_committed_lists_at_section_three_ones_sizes(self, frozen):
        sizes = {
            path.name: len(read_tile_strata(path, what="tile list"))
            for path in frozen.glob("*.csv")
        }
        assert sizes == {
            "head_train_1k.csv": 1_000,
            "head_train_5k.csv": 5_000,
            "head_train_tiles.csv": 20_000,
            "val_tiles.csv": 2_000,
            "test_tiles.csv": 5_000,
        }

    def test_the_nested_subsets_are_nested_at_the_committed_budget(self, frozen):
        sets = [
            {t for t, _ in read_tile_strata(frozen / name, what="tile list")}
            for name in ("head_train_1k.csv", "head_train_5k.csv", "head_train_tiles.csv")
        ]
        assert sets[0] < sets[1] < sets[2]

    def test_the_head_train_draw_is_proportional_over_the_twenty_strata(self, frozen):
        pool = pq.read_table(CORPUS_500K / "manifest.parquet", columns=["split", "stratum_id"])
        strata = pool["stratum_id"].to_numpy(zero_copy_only=False)
        train = strata[pool["split"].to_numpy(zero_copy_only=False) == "train"]
        rows = read_tile_strata(frozen / "head_train_tiles.csv", what="head-train list")
        drawn = np.bincount([s for _, s in rows], minlength=BenchmarkConfig().tiles.strata)

        assert (drawn > 0).all(), "a stratum the corpus holds went undrawn"
        # Every stratum within one tile of its proportional share — what largest-remainder
        # guarantees, and what makes the head-train set a sample of the corpus rather than of
        # the sampler.
        ideal = 20_000 * np.bincount(train, minlength=len(drawn)) / train.size
        assert np.abs(drawn - ideal).max() < 1.0

    def test_the_test_list_comes_only_from_the_thirteen_test_geocells(self, frozen):
        committed = json.loads(Path(dataset_config.SPLIT_BLOCKS_PATH).read_text())
        held_out = set(committed["test"]["cells"])
        cols = pq.read_table(
            CORPUS_500K / "manifest.parquet", columns=["tile_id", "geocell_id", "split"]
        )
        ids = cols["tile_id"].to_pylist()
        cell_of = dict(zip(ids, cols["geocell_id"].to_pylist()))
        split_of = dict(zip(ids, cols["split"].to_pylist()))
        drawn = [t for t, _ in read_tile_strata(frozen / "test_tiles.csv", what="test list")]
        assert {cell_of[t] for t in drawn} <= held_out
        assert {split_of[t] for t in drawn} == {"test"}

    def test_a_redraw_reproduces_the_committed_bytes_and_says_what_it_wrote(
        self, frozen, tmp_path, capsys
    ):
        # K = 1 and a temporary bank path for the reason the `frozen` fixture gives: `freeze`
        # runs every stage before the one that raises, and the bank's default path is a real
        # location under `data/`.
        with pytest.raises(NotImplementedError):
            launch.main(
                [
                    "freeze",
                    "--artifacts-dir", str(tmp_path),
                    "--set", "masks.bank_draws_per_tile=1",
                    "--set", f'paths.mask_bank_path="{tmp_path / "bank.parquet"}"',
                ]
            )
        out = capsys.readouterr().out
        for path in frozen.glob("*.csv"):
            md5 = file_md5(path)
            assert file_md5(tmp_path / path.name) == md5, path.name
            # And the md5 is on screen, which is what the run record quotes the list by.
            assert md5 in out, path.name


def test_the_default_ladder_is_the_one_the_config_registers():
    # One home for the numbers: `benchmark.tiles` apportions whatever ladder it is handed, and
    # the one it defaults to is §3.1's, off the config that already registers it.
    assert signature(tiles.draw).parameters["head_train_sizes"].default is SIZE_TILES
    assert BenchmarkConfig().tiles.head_train == SIZE_TILES[LABEL_EFFICIENCY_SIZES[-1]]
