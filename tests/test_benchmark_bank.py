"""`benchmark.masks`'s train mask bank, its md5 sidecar and §3.4's guard over both (#151).

The bank is the one committed input that is **not committed as bytes**. 1,000,000 RLE rows and
104 MB cannot go in git, on the same rule the corpus's shards follow, so what git carries is a
sidecar holding the bank's md5, its draw hash and the generator config hash — and `run` refuses a
bank whose md5 has moved rather than reading it. That indirection is the whole ticket, so it is
what most of this file asserts.

Three altitudes, for the reason `tests/test_benchmark_masks.py` gives:

**The draw** attaches at the module. K rows per tile, the K differing from one another, each row
re-drawable from its own seed, each mask satisfying §3.2's constraints — none of it reads a CLI,
and all of it is cheap because the tile list is handed in rather than drawn.

**The sidecar and its guard** attach at :func:`benchmark.masks.freeze_bank`, against a
hand-written head-train list, because what they are about is a *file* and a refusal.

**The committed artifact** attaches at seam 1, `python -m benchmark.launch freeze`, against the
shipped corpus. The real bank is 20,000 tiles x K = 50 and twenty minutes, which is not a thing
a pre-push gate can run, so the class below freezes the same shape at K = 2 — what it asserts is
that the stage lands, is a function of the *head-train* list, and says what it committed. The
1,000,000-row figures in plan §7 are measured by running `freeze` for real, not here.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from benchmark import launch, masks
from benchmark.artifacts import MASK_BANK_SIDECAR_NAME, TILE_LIST_NAMES
from benchmark.config import BenchmarkConfig, Masks
from dataset import config as dataset_config
from dataset.materialize import file_md5
from pretrain.artifact import write_tile_strata

CORPUS_500K = Path("data/corpus_500k")

#: The six (family, bucket) cells the bank draws over — §3.2's two synthetic families times its
#: three buckets, exactly as the eval mask list does.
CELLS = [
    (family, bucket)
    for family in Masks().synthetic_families
    for bucket, _ in Masks().buckets
]


def tile_ids(n: int) -> list[str]:
    """``n`` plausible tile ids — the generator keys on the string and reads no corpus."""
    return [f"N45_E007_{i:05d}" for i in range(n)]


def small(**knobs) -> Masks:
    """§3.2's geometry at a bank size a test can afford. 12 draws is two per cell."""
    return Masks(bank_draws_per_tile=12).replace(**knobs)


# --- The draw -----------------------------------------------------------------


def test_the_bank_is_k_rows_per_tile_numbered_by_epoch():
    geometry = small()
    rows = list(masks.bank_rows(tile_ids(3), geometry))

    assert len(rows) == 3 * geometry.bank_draws_per_tile
    for tile in tile_ids(3):
        epochs = [row.epoch for row in rows if row.tile_id == tile]
        assert epochs == list(range(geometry.bank_draws_per_tile))


def test_the_fifty_rows_for_one_tile_differ_from_one_another():
    """The ticket's own sentence, at the K the benchmark actually runs.

    50 committed masks per tile is what 50 on-the-fly epochs would have produced (§3.2), so a
    bank whose rows repeated would be a *frozen* training mask wearing the words of a varying
    one — the exact failure the deleted feature cache was rejected for (§7). Nothing downstream
    raises on it: a head trained against one mask fifty times converges perfectly normally.
    """
    rows = [row for row in masks.bank_rows(["N45_E007_00000"], Masks())]

    assert len(rows) == 50 == Masks().bank_draws_per_tile
    assert len({row.mask_rle for row in rows}) == 50
    assert len({row.seed for row in rows}) == 50


def test_each_tile_draws_over_every_family_and_bucket_rather_than_one_cell():
    """§3.2's geometry axis is the eval axis, so training must see all six cells.

    Balanced per *tile* rather than per epoch: a bank that gave epoch 0 to blob/small for every
    tile would train each epoch on one geometry, which is a curriculum nobody chose and which
    the epoch axis makes invisible.
    """
    geometry = small()
    for tile in tile_ids(4):
        drawn = [(row.family, row.bucket) for row in masks.bank_rows([tile], geometry)]
        assert sorted(set(drawn)) == sorted(CELLS)
        assert {drawn.count(cell) for cell in CELLS} == {2}

    first_epoch = {
        row.tile_id: (row.family, row.bucket)
        for row in masks.bank_rows(tile_ids(12), geometry)
        if row.epoch == 0
    }
    assert len(set(first_epoch.values())) > 1


def test_every_bank_row_satisfies_the_constraints_it_was_drawn_under():
    # The same pass `tests/test_benchmark_masks.py` makes over the committed eval rows: the
    # bank is drawn by the same generator, so it is subject to the same three constraints.
    for row in masks.bank_rows(tile_ids(3), small()):
        mask = masks.decode_rle(row.mask_rle)
        masks.check(mask, bucket=row.bucket)
        assert masks.bucket_of(masks.equivalent_diameter(int(mask.sum()))) == row.bucket


def test_a_bank_row_is_re_drawable_from_the_seed_it_carries():
    # What the seed column is for, in the eval list and here for the same reason: a row whose
    # seed no longer produces its mask is the one corruption an md5 over the file cannot find.
    row = next(row for row in masks.bank_rows(["N45_E007_00007"], small()) if row.epoch == 5)
    again = masks.draw_from_seed(row.seed, family=row.family, bucket=row.bucket)
    assert masks.encode_rle(again.mask) == row.mask_rle


def test_a_bank_row_and_an_eval_row_for_the_same_cell_are_different_draws():
    """The two artifacts share a generator and a hash, and share no mask.

    A bank row that reproduced the eval row for its (tile, family, bucket) would train a head on
    the instance it is scored against — for whichever tiles the two draws happened to share.
    They are drawn under one contract and keyed apart by construction, which is what the
    :data:`~benchmark.masks.BANK_TAG` in the seed is for.
    """
    tile = "N45_E007_00000"
    evaluated = masks.draw(tile, "blob", "small")
    banked = [
        row
        for row in masks.bank_rows([tile], Masks())
        if row.family == "blob" and row.bucket == "small"
    ]
    assert banked, "the balanced draw gives every cell at least one epoch at K = 50"
    assert all(row.seed != evaluated.seed for row in banked)
    assert all(row.mask_rle != evaluated.mask_rle for row in banked)


def test_moving_the_generator_seed_moves_the_whole_bank():
    moved = small(seed=Masks().seed + 1)
    assert [row.mask_rle for row in masks.bank_rows(tile_ids(2), moved)] != [
        row.mask_rle for row in masks.bank_rows(tile_ids(2), small())
    ]


# --- The committed sidecar ----------------------------------------------------


@pytest.fixture
def frozen_tiles(tmp_path):
    """An artifacts directory holding the one thing the bank stage reads: §3.1's 20k list."""
    write_tile_strata(
        [(tile, i % 20) for i, tile in enumerate(tile_ids(4))],
        tmp_path / TILE_LIST_NAMES["20k"],
    )
    return tmp_path


@pytest.fixture
def cfg(frozen_tiles, tmp_path):
    base = BenchmarkConfig()
    return base.replace(
        masks=small(),
        paths=base.paths.replace(
            artifacts_dir=str(frozen_tiles),
            mask_bank_path=str(tmp_path / "bank" / "train_mask_bank.parquet"),
        ),
    )


def sidecar_of(cfg) -> Path:
    return Path(cfg.paths.artifacts_dir) / MASK_BANK_SIDECAR_NAME


def test_the_stage_writes_the_six_column_artifact(cfg):
    written = masks.freeze_bank(cfg)
    table = pq.read_table(cfg.paths.mask_bank_path)

    assert table.column_names == ["tile_id", "epoch", "family", "bucket", "mask_rle", "seed"]
    assert table.num_rows == written.committed.draw.rows == 4 * cfg.masks.bank_draws_per_tile
    assert written.committed.md5 == file_md5(cfg.paths.mask_bank_path)
    assert written.committed.size_bytes == Path(cfg.paths.mask_bank_path).stat().st_size


def test_the_bank_is_drawn_off_the_head_train_list_and_not_the_size_the_cell_names(cfg):
    """§3.4's bank is an input *every* cell shares, so `--size 1k` must not redraw it.

    The label-efficiency rungs are nested prefixes of the one committed 20k draw (§3.1), so one
    bank over the whole draw serves all three — and a bank that moved with `--size` would be a
    second determinism contract, which is what :attr:`BenchmarkConfig.artifact_hash` exists to
    keep from happening.

    Staged as a *different* 1k list rather than as two freezes of one directory, so that a bank
    which read `TILE_LIST_NAMES[cfg.size]` fails here instead of passing on the tiles happening
    to be nested.
    """
    write_tile_strata(
        [(f"S01_E036_{i:05d}", i) for i in range(2)],
        Path(cfg.paths.artifacts_dir) / TILE_LIST_NAMES["1k"],
    )
    banked = masks.freeze_bank(cfg.replace(size="1k"))
    tiles = set(pq.read_table(cfg.paths.mask_bank_path, columns=["tile_id"])["tile_id"].to_pylist())

    assert tiles == set(tile_ids(4))
    assert banked.committed.draw.rows == 4 * cfg.masks.bank_draws_per_tile
    assert banked.committed.draw.artifact_hash == cfg.artifact_hash


def test_the_sidecar_carries_the_bank_md5_and_the_generator_config_hash(cfg):
    written = masks.freeze_bank(cfg)
    committed = json.loads(sidecar_of(cfg).read_text())

    assert committed["md5"] == file_md5(cfg.paths.mask_bank_path) == written.committed.md5
    assert committed["artifact_hash"] == cfg.artifact_hash
    assert committed["generator"] == masks.GENERATOR
    assert committed["rows"] == written.committed.draw.rows
    assert committed["draws_per_tile"] == cfg.masks.bank_draws_per_tile
    assert committed["tiles"] == 4


def test_both_mask_artifacts_carry_one_contract_rather_than_two(cfg, frozen_tiles):
    """§3.4's sentence, as a check: the same generator and the same config hash, in both files.

    Asserted on the artifacts rather than on the config that made them, because the claim is
    about what a *reader* of the two files can tell — the eval list and the bank were generated
    from one recipe, so a run cannot be scored against masks from one draw and trained against
    masks from another.
    """
    from benchmark.artifacts import EVAL_MASKS_NAME, TEST_TILES_NAME

    write_tile_strata(
        [(tile, i % 20) for i, tile in enumerate(tile_ids(4))], frozen_tiles / TEST_TILES_NAME
    )
    masks.freeze(cfg)
    masks.freeze_bank(cfg)

    evaluated = pq.read_schema(frozen_tiles / EVAL_MASKS_NAME).metadata
    banked = pq.read_schema(cfg.paths.mask_bank_path).metadata
    assert evaluated[b"generator"] == banked[b"generator"] == masks.GENERATOR.encode()
    assert evaluated[b"artifact_hash"] == banked[b"artifact_hash"] == cfg.artifact_hash.encode()
    assert json.loads(sidecar_of(cfg).read_text())["artifact_hash"] == cfg.artifact_hash


# --- §3.4's guard -------------------------------------------------------------


def test_re_freezing_an_unchanged_bank_regenerates_nothing(cfg):
    """Idempotence has to mean *not spending the twenty minutes again*, not merely 'same bytes'.

    So it is asserted as a fact about the file rather than about the return value: the bank on
    disk is untouched, which is also what keeps the md5 every run record quotes from moving
    underneath it.
    """
    first = masks.freeze_bank(cfg)
    stamp = Path(cfg.paths.mask_bank_path).stat().st_mtime_ns

    again = masks.freeze_bank(cfg)
    assert again.committed.md5 == first.committed.md5
    assert again.generated is False
    assert Path(cfg.paths.mask_bank_path).stat().st_mtime_ns == stamp


def test_a_bank_whose_draw_has_moved_is_refused_rather_than_overwritten(cfg):
    masks.freeze_bank(cfg)
    moved = cfg.replace(masks=cfg.masks.replace(seed=cfg.masks.seed + 1))

    with pytest.raises(FileExistsError, match="already frozen"):
        masks.freeze_bank(moved)
    assert json.loads(sidecar_of(cfg).read_text())["artifact_hash"] == cfg.artifact_hash


def test_the_refusal_comes_before_the_twenty_minutes_rather_than_after_it(cfg):
    """A guard that spent the generation first and refused afterwards is a guard nobody keeps.

    The shape of the draw — the generator, the config hash, the row count, K — is knowable
    before a single mask exists, so it is what the refusal reads. The bank on disk is the
    evidence: a refused freeze leaves the committed one exactly where it was.
    """
    masks.freeze_bank(cfg)
    stamp = Path(cfg.paths.mask_bank_path).stat().st_mtime_ns
    moved = cfg.replace(masks=cfg.masks.replace(bank_draws_per_tile=6))

    with pytest.raises(FileExistsError, match="draws_per_tile"):
        masks.freeze_bank(moved)
    assert Path(cfg.paths.mask_bank_path).stat().st_mtime_ns == stamp


def test_a_bank_that_went_missing_is_regenerated_against_the_committed_sidecar(cfg):
    """The case the sidecar exists *for*: git has the hash, the box has lost the 104 MB.

    Regenerating must reproduce the committed draw exactly, and the check is the draw hash
    rather than the file's — see the next test for why those are two questions.
    """
    first = masks.freeze_bank(cfg)
    Path(cfg.paths.mask_bank_path).unlink()

    again = masks.freeze_bank(cfg)
    assert again.generated is True
    assert again.committed.draw_hash == first.committed.draw_hash
    assert again.committed.md5 == first.committed.md5


def test_the_guard_reads_the_draw_and_not_the_bytes_it_arrived_in(cfg):
    """#150's lesson, one level up: the sidecar's md5 is a fact about *this box's* encoding.

    A parquet footer carries the writing library's own version string, so a sidecar compared
    whole would refuse a re-freeze that had re-derived the identical 1,000,000 masks, the day
    pyarrow was upgraded — a refusal about the environment wearing the words of one about the
    draw. A sidecar carrying a stale md5 stands in for that here: the draw is unchanged, so the
    bank is regenerated and the sidecar's md5 is brought up to what is on disk.
    """
    first = masks.freeze_bank(cfg)
    committed = json.loads(sidecar_of(cfg).read_text())
    committed["md5"] = "0" * 32
    sidecar_of(cfg).write_text(json.dumps(committed, sort_keys=True, indent=2) + "\n")

    again = masks.freeze_bank(cfg)
    assert again.generated is True
    assert again.committed.draw_hash == first.committed.draw_hash
    assert json.loads(sidecar_of(cfg).read_text())["md5"] == file_md5(cfg.paths.mask_bank_path)


def test_the_stage_refuses_a_box_whose_head_train_list_is_not_frozen_yet(tmp_path):
    base = BenchmarkConfig()
    cfg = base.replace(paths=base.paths.replace(artifacts_dir=str(tmp_path)))
    with pytest.raises(FileNotFoundError, match="freeze"):
        masks.freeze_bank(cfg)


# --- What `run` does about it -------------------------------------------------


def test_a_run_refuses_a_bank_whose_md5_has_moved_rather_than_reading_it(cfg):
    """The sidecar's whole purpose, and the failure it closes is silent.

    A head trained against a re-drawn mask bank converges normally and prints a table exactly as
    wide as a good one (§15). So the mismatch is named before the encoder is loaded, and it is
    named as a *refusal* — reading the bank anyway and mentioning it would be the same run.
    """
    masks.freeze_bank(cfg)
    masks.assert_bank_matches(cfg)  # the frozen box: no exception, and it returns the md5

    Path(cfg.paths.mask_bank_path).write_bytes(b"not the committed draw")
    with pytest.raises(ValueError, match="md5"):
        masks.assert_bank_matches(cfg)


def test_a_run_refuses_a_bank_frozen_under_a_different_generator_config(cfg):
    # The md5 would catch this too, once the bank is redrawn. This catches it while the *old*
    # bank is still on disk, which is the state a box is actually in mid-change.
    masks.freeze_bank(cfg)
    moved = cfg.replace(masks=cfg.masks.replace(seed=cfg.masks.seed + 1))
    with pytest.raises(ValueError, match="artifact_hash|generator config hash"):
        masks.assert_bank_matches(moved)


def test_a_run_against_a_box_with_no_bank_says_so_rather_than_reading_a_stale_hash(cfg):
    masks.freeze_bank(cfg)
    Path(cfg.paths.mask_bank_path).unlink()
    with pytest.raises(FileNotFoundError, match="freeze"):
        masks.assert_bank_matches(cfg)


# --- Through seam 1 -----------------------------------------------------------


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """One real `freeze` of the shipped corpus, and what it said.

    At K = 2 rather than §3.2's 50, and 100 test tiles rather than 5,000: the real artifact is
    1,000,000 draws and twenty minutes, and this suite is the gate before every push. What the
    shape at K = 2 still proves is everything the stage is responsible for — that it lands, that
    it is a function of the committed 20,000-tile head-train list, and that it says what it
    committed. Plan §7's measured figures come from running `freeze` for real, not from here.
    """
    directory = tmp_path_factory.mktemp("task1")
    bank = tmp_path_factory.mktemp("bank") / "train_mask_bank.parquet"
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        with pytest.raises(NotImplementedError, match="dataset card"):
            launch.main(
                [
                    "freeze",
                    "--artifacts-dir", str(directory),
                    "--set", "masks.bank_draws_per_tile=2",
                    "--set", "tiles.test=100",
                    "--set", f'paths.mask_bank_path="{bank}"',
                ]
            )
    return directory, bank, said.getvalue()


@pytest.mark.skipif(
    not (CORPUS_500K / "manifest.parquet").exists()
    or not Path(dataset_config.SPLIT_BLOCKS_PATH).exists(),
    reason="the shipped corpus or the committed split (#77) is not present",
)
class TestAgainstTheShippedCorpus:
    """§3.4's bank at its real tile budget — 20,000 head-train tiles, K held down to 2."""

    def test_freeze_commits_k_draws_for_every_head_train_tile(self, frozen):
        directory, bank, _ = frozen
        from pretrain.artifact import read_tile_strata

        committed = {
            t for t, _ in read_tile_strata(directory / TILE_LIST_NAMES["20k"], what="20k list")
        }
        table = pq.read_table(bank, columns=["tile_id", "epoch"])
        assert len(committed) == 20_000
        assert table.num_rows == 40_000
        assert set(table["tile_id"].to_pylist()) == committed

    def test_the_sidecar_is_committed_beside_the_lists_and_the_bank_is_not(self, frozen):
        # §3.4's one resolved implicit: 104 MB cannot go in git, so what git carries is the
        # hash. The sidecar lands in `artifacts_dir` with the tile lists; the bank lands
        # wherever this box put it, and the two are related by an md5 and nothing else.
        directory, bank, _ = frozen
        committed = json.loads((directory / MASK_BANK_SIDECAR_NAME).read_text())
        assert committed["md5"] == file_md5(bank)
        assert committed["rows"] == 40_000
        assert not (directory / bank.name).exists()

    def test_the_freeze_says_what_it_committed_and_what_it_cost(self, frozen):
        # The ticket's own acceptance line: the row count, the byte size and the generation
        # time, printed — because they are what §7's stale 740 MB / 33 min are replaced by.
        _, bank, said = frozen
        assert "40,000" in said
        assert file_md5(bank) in said
        assert "MB" in said and " s" in said
        assert masks.GENERATOR in said
