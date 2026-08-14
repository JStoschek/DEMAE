"""`benchmark.masks` — the generator, the RLE codec and the eval mask list (#150).

Three kinds of claim, at the altitude each one survives at.

**Numeric identities** attach at the module, `tests/test_masks.py`-style: the codec is an exact
inverse, `equivalent_diameter` inverts `bucket_area`, and the fast dilation is the same set the
iterated 3x3 dilation produces. None of these read through a CLI.

**The constraint checks** are the ticket's own sentence — *checked on every draw, never assumed*
— so they are asserted twice: once over a draw sample the size of the one plan §3.2 published
(300 per (family, bucket) cell, zero violations of either kind), and once as refusals, by handing
:func:`benchmark.masks.check` a mask that breaks each constraint on purpose. A draw that would
violate one is asserted to be **retried or refused, never trimmed**, which is the only one of the
three that a silently-trimming generator would still pass.

**The committed artifact** attaches at seam 1, `python -m benchmark.launch freeze`, against the
shipped corpus — 30,000 rows is 5,000 test tiles x 2 families x 3 buckets, and nothing smaller
can express it. The stage is also driven directly off a hand-written tile list, so the round-trip
through the parquet is asserted on a box with no corpus at all.
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.ndimage import binary_dilation

from benchmark import launch, masks
from benchmark.artifacts import EVAL_MASKS_NAME, TEST_TILES_NAME
from benchmark.config import BenchmarkConfig, Masks
from dataset import config as dataset_config
from dataset.materialize import file_md5
from pretrain.artifact import read_tile_strata, write_tile_strata

CORPUS_500K = Path("data/corpus_500k")

TILE_PX = dataset_config.TILE_PX

#: The six (family, bucket) cells §3.2 draws over.
CELLS = [
    (family, bucket)
    for family in Masks().synthetic_families
    for bucket, _ in Masks().buckets
]

#: Plan §3.2 published its constraint measurement over 300 draws per cell ([#121]); the sample
#: here is the same size, so "zero cap violations, zero interiority violations" is asserted over
#: the population it was measured on rather than over a token handful.
SAMPLE_PER_CELL = 300


def tile_ids(n: int) -> list[str]:
    """``n`` plausible tile ids — the generator keys on the string and reads no corpus."""
    return [f"N45_E007_{i:05d}" for i in range(n)]


def draw(tile_id: str = "N45_E007_00000", family: str = "blob", bucket: str = "small", **kw):
    return masks.draw(tile_id, family, bucket, **kw)


# --- The RLE codec ------------------------------------------------------------


def test_the_codec_round_trips_every_cell_exactly():
    for family, bucket in CELLS:
        for tile in tile_ids(5):
            mask = draw(tile, family, bucket).mask
            assert np.array_equal(masks.decode_rle(masks.encode_rle(mask)), mask), (
                family,
                bucket,
                tile,
            )


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((4, 5), dtype=bool),
        np.ones((4, 5), dtype=bool),
        np.eye(6, dtype=bool),
        # Foreground at pixel 0 — the case a "runs start with background" encoding gets wrong
        # unless it is allowed to open with a zero-length background run.
        np.array([[True, False], [False, True]]),
    ],
)
def test_the_codec_round_trips_the_degenerate_masks(mask):
    assert np.array_equal(masks.decode_rle(masks.encode_rle(mask)), mask)


def test_the_encoding_carries_its_own_shape():
    """So decoding needs no out-of-band width.

    A mask decoded at the wrong width is not a crash — it is a plausible mask, scored normally,
    in a column nothing reads back. That is the failure this repo refuses to leave available.
    """
    text = masks.encode_rle(np.zeros((7, 3), dtype=bool))
    assert text.startswith("7x3:")
    assert masks.decode_rle(text).shape == (7, 3)


def test_the_encoding_is_stable_across_runs():
    first = masks.encode_rle(draw().mask)
    assert first == masks.encode_rle(draw().mask)


def test_an_rle_whose_runs_do_not_fill_its_shape_is_refused():
    with pytest.raises(ValueError, match="4x4 needs 16 pixels"):
        masks.decode_rle("4x4:3,4")


def test_an_rle_that_is_not_one_is_refused_rather_than_half_read():
    for text in ("nonsense", "4:1,2", "4x4", "4x-4:16"):
        with pytest.raises(ValueError, match="not an RLE"):
            masks.decode_rle(text)


# --- The geometry -------------------------------------------------------------


def test_equivalent_diameter_inverts_the_bucket_area():
    # The unit is load-bearing (§3.2): both families are area-matched through `d`, so a bucket
    # boundary in diameter has to mean one pixel count whichever family is drawing.
    for d in (16.0, 48.0, 64.0, 96.0, 128.0, 158.0):
        assert masks.equivalent_diameter(masks.bucket_area(d)) == pytest.approx(d, abs=0.02)


def test_both_families_cover_the_same_pixel_count_at_one_diameter():
    rng = np.random.default_rng(0)
    margin = Masks().edge_margin_px
    for d in (32.0, 80.0, 150.0):
        target = masks.bucket_area(d)
        blob = masks.blob_mask(d, rng, margin=margin)
        rect = masks.rectangle_mask(d, rng, margin=margin)
        assert int(blob.sum()) == target
        # The rectangle floors its second side, so it lands at or just under the blob's area —
        # never over, which is what keeps it inside the 40% cap by construction (§3.2).
        assert target - int(rect.sum()) < max(rect.sum(axis=1).max(), rect.sum(axis=0).max())
        assert int(rect.sum()) <= target


def test_the_bucket_of_a_mask_is_read_from_its_equivalent_diameter():
    for bucket, (lo, hi) in Masks().buckets:
        assert masks.bucket_of(lo) == bucket
        assert masks.bucket_of(hi) == bucket
        assert masks.bucket_of((lo + hi) / 2) == bucket
    # The gaps between §3.2's buckets are nobody's bucket, and are answered as such.
    assert masks.bucket_of(56.0) is None


def test_the_fast_dilation_is_the_iterated_dilation_it_replaces():
    """A numeric identity, not a resemblance — the promoted generator's one implementation change.

    p119 grew the trace by one 3x3 dilation at a time and stopped at the first radius reaching
    the target area. Iterated dilation by scipy's default (4-connected) structure *is* the
    taxicab distance transform thresholded at that radius, so the same set is available in one
    pass. The speed-up is 3.5x on the large bucket and worth having twice over — this generator
    also writes #151's 1,000,000-row bank — but only if the sets are identical, which is what
    this asserts rather than assumes.
    """
    rng = np.random.default_rng(0)
    for d in (20.0, 70.0, 150.0):
        target = masks.bucket_area(d)
        for _ in range(5):
            trace = masks._rasterize_walk(masks._random_walk(max(4, target // 3), rng))
            grown, padded, radius = masks._dilate_to_area(trace, target)

            reference, ref_trace, steps = trace, trace, 0
            while reference.sum() < target:
                reference = binary_dilation(np.pad(reference, 1))
                ref_trace = np.pad(ref_trace, 1)
                steps += 1
            assert (radius, grown.shape) == (steps, reference.shape)
            assert np.array_equal(grown, reference)
            assert np.array_equal(padded, ref_trace)


# --- The constraints, checked on every draw -----------------------------------


def test_the_constraint_checks_fire_over_a_draw_sample():
    """§3.2's three constraints over the sample size #121 measured them on.

    Zero area-cap violations, zero interiority violations, and every mask's own equivalent
    diameter inside the bucket the row files it under. The third is the one the generator can
    fail quietly: `d` is drawn inside the bucket, but the mask a family builds from it lands on
    an integer pixel count, and the *achieved* diameter is what the bucket axis means.
    """
    geometry = Masks()
    interior = TILE_PX - 2 * geometry.edge_margin_px
    cap = geometry.max_area_frac * TILE_PX * TILE_PX
    over_cap = not_interior = out_of_bucket = 0
    widest = 0

    for family, bucket in CELLS:
        for tile in tile_ids(SAMPLE_PER_CELL):
            mask = masks.draw(tile, family, bucket, geometry=geometry).mask
            area = int(mask.sum())
            over_cap += area > cap
            not_interior += bool(
                mask[: geometry.edge_margin_px].any()
                or mask[-geometry.edge_margin_px :].any()
                or mask[:, : geometry.edge_margin_px].any()
                or mask[:, -geometry.edge_margin_px :].any()
            )
            out_of_bucket += masks.bucket_of(masks.equivalent_diameter(area)) != bucket
            rows, cols = np.nonzero(mask)
            widest = max(widest, rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)

    assert (over_cap, not_interior, out_of_bucket) == (0, 0, 0)
    assert widest <= interior


def test_a_mask_reaching_the_edge_margin_is_refused():
    mask = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    mask[7:60, 40:100] = True  # one row inside the 8 px margin
    with pytest.raises(masks.NotInterior, match="8 px of the tile edge"):
        masks.check(mask, bucket="medium")


def test_a_mask_over_the_area_cap_is_refused():
    mask = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    mask[8:-8, 8:-8] = True  # the whole interior — 86% of the tile
    with pytest.raises(masks.OverAreaCap, match="40%"):
        masks.check(mask, bucket="large")


def test_a_mask_outside_the_bucket_it_is_filed_under_is_refused():
    mask = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    mask[20:40, 20:40] = True  # d = 22.6 px — a small-bucket void filed as medium
    with pytest.raises(masks.OutOfBucket, match="medium"):
        masks.check(mask, bucket="medium")


def test_an_empty_mask_is_refused_rather_than_scored_as_a_void():
    with pytest.raises(masks.MaskError, match="no void at all"):
        masks.check(np.zeros((TILE_PX, TILE_PX), dtype=bool), bucket="small")


def test_a_draw_that_would_violate_a_constraint_is_retried_and_never_trimmed(monkeypatch):
    """The acceptance criterion a silently-trimming generator would otherwise still pass.

    The first attempt returns a mask over the cap. A generator that trimmed would return that
    mask short of its area; this one throws the draw away and asks the family again.
    """
    over_cap = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    over_cap[8:-8, 8:-8] = True
    honest = masks.rectangle_mask
    calls = []

    def first_one_is_bad(d, rng, **kw):
        calls.append(d)
        return over_cap.copy() if len(calls) == 1 else honest(d, rng, **kw)

    monkeypatch.setitem(masks.FAMILIES, "rectangle", first_one_is_bad)
    drawn = draw(family="rectangle", bucket="medium")

    assert drawn.attempts == 2
    assert not np.array_equal(drawn.mask, over_cap)
    assert drawn.mask.sum() < over_cap.sum()
    # And the second attempt is a *different draw*, not the first one repaired.
    assert calls[0] != calls[1]
    masks.check(drawn.mask, bucket="medium")


def test_a_draw_that_cannot_satisfy_the_constraints_is_refused(monkeypatch):
    over_cap = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    over_cap[8:-8, 8:-8] = True
    monkeypatch.setitem(masks.FAMILIES, "rectangle", lambda d, rng, **kw: over_cap.copy())
    with pytest.raises(masks.MaskError, match=f"{masks.MAX_ATTEMPTS} draws"):
        draw(family="rectangle", bucket="medium")


def test_a_bucket_the_area_cap_could_not_satisfy_is_refused_at_the_config():
    # #121 from the other end: §3.2's constraints are jointly unsatisfiable above 159.86 px, and
    # the settled record is a 158 px top rather than a clamp. A config that moved the boundary
    # back is refused where the boundary lives, so no draw ever has to be trimmed.
    with pytest.raises(ValueError, match="area cap"):
        Masks(large_px=(128, 160))


# --- Determinism --------------------------------------------------------------


def test_the_same_cell_redraws_the_same_mask():
    assert np.array_equal(draw().mask, draw().mask)


def test_a_different_tile_family_or_bucket_is_a_different_draw():
    base = masks.encode_rle(draw().mask)
    assert masks.encode_rle(draw(tile_id="N45_E007_00001").mask) != base
    assert masks.encode_rle(draw(family="rectangle").mask) != base
    assert masks.encode_rle(draw(bucket="medium").mask) != base


def test_the_committed_seed_reproduces_its_own_row_on_its_own():
    """What the fifth column is *for*: a row is re-drawable from what the row carries.

    Not the generator's seed — the draw's. `mask_rle` and `seed` are two views of one draw, and
    a row whose seed no longer produced its mask is the one corruption an md5 over the whole
    file cannot localise.
    """
    drawn = draw(tile_id="N45_E007_00042", family="rectangle", bucket="large")
    again = masks.draw_from_seed(drawn.seed, family="rectangle", bucket="large")
    assert np.array_equal(again.mask, drawn.mask)


def test_moving_the_generator_seed_moves_every_draw():
    moved = draw(geometry=Masks(seed=Masks().seed + 1))
    assert not np.array_equal(moved.mask, draw().mask)


# --- The eval mask list -------------------------------------------------------


def test_the_eval_mask_list_is_one_row_per_tile_family_and_bucket():
    rows = masks.eval_rows(tile_ids(5))
    assert len(rows) == 5 * len(CELLS)
    assert {(r.tile_id, r.family, r.bucket) for r in rows} == {
        (t, f, b) for t in tile_ids(5) for f, b in CELLS
    }


def test_every_row_files_its_mask_under_the_bucket_its_diameter_falls_in():
    for row in masks.eval_rows(tile_ids(3)):
        area = int(masks.decode_rle(row.mask_rle).sum())
        assert masks.bucket_of(masks.equivalent_diameter(area)) == row.bucket


@pytest.fixture
def frozen_tiles(tmp_path):
    """An artifacts directory holding the one thing the mask stage reads: §3.1's test list."""
    write_tile_strata(
        [(tile, i % 20) for i, tile in enumerate(tile_ids(4))], tmp_path / TEST_TILES_NAME
    )
    return tmp_path


@pytest.fixture
def cfg(frozen_tiles):
    return BenchmarkConfig().replace(
        paths=BenchmarkConfig().paths.replace(artifacts_dir=str(frozen_tiles))
    )


def test_the_stage_writes_the_five_column_artifact(cfg, frozen_tiles):
    written = masks.freeze(cfg)
    table = pq.read_table(frozen_tiles / EVAL_MASKS_NAME)

    assert table.column_names == ["tile_id", "family", "bucket", "mask_rle", "seed"]
    assert table.num_rows == written.rows == 4 * len(CELLS)
    assert written.md5 == file_md5(frozen_tiles / EVAL_MASKS_NAME)


def test_the_artifact_records_the_generator_config_hash_the_bank_will_carry(cfg, frozen_tiles):
    """§3.4's determinism contract is *one* contract, so the hash is in the file, not beside it.

    `artifact_hash` is the one #151's bank sidecar quotes: it covers §3.1's tiles and §3.2's
    masks and nothing about this box, so two artifacts that agree on it were generated from the
    same recipe whatever paths they were written through.
    """
    masks.freeze(cfg)
    metadata = pq.read_schema(frozen_tiles / EVAL_MASKS_NAME).metadata
    assert metadata[b"artifact_hash"].decode() == cfg.artifact_hash
    assert metadata[b"generator"].decode() == masks.GENERATOR


def test_a_rewrite_of_the_same_draw_is_byte_identical(cfg, frozen_tiles):
    first = masks.freeze(cfg).md5
    (frozen_tiles / EVAL_MASKS_NAME).unlink()
    assert masks.freeze(cfg).md5 == first


def test_rewriting_the_unchanged_artifact_is_a_no_op_rather_than_a_refusal(cfg):
    assert masks.freeze(cfg).md5 == masks.freeze(cfg).md5


def test_an_eval_mask_list_whose_content_hash_has_moved_is_not_overwritten(cfg, frozen_tiles):
    masks.freeze(cfg)
    moved = cfg.replace(masks=cfg.masks.replace(seed=cfg.masks.seed + 1))
    with pytest.raises(FileExistsError, match="already frozen at md5"):
        masks.freeze(moved)


def test_the_guard_reads_the_committed_draw_and_not_the_bytes_it_arrived_in(cfg, frozen_tiles):
    """§3.4's refusal is about the recipe, so a re-encoding of the same draw is not a change.

    A parquet footer carries the writing library's own version string, so a guard keyed on byte
    equality would refuse a `freeze` that had re-derived the identical 30,000 masks, the day
    pyarrow was upgraded — a refusal about the environment wearing the words of one about the
    draw. The same rows written under a different codec stand in for that here.
    """
    import pyarrow.parquet as pq

    rows = masks.eval_rows(tile_ids(4), cfg.masks)
    committed = frozen_tiles / EVAL_MASKS_NAME
    masks.freeze(cfg)
    table = pq.read_table(committed)
    pq.write_table(table, committed, compression="snappy")
    reencoded = file_md5(committed)

    # Not a refusal, and not a rewrite: the committed bytes are what every run record quotes.
    assert masks.freeze(cfg).md5 == reencoded
    assert file_md5(committed) == reencoded
    assert len(rows) == pq.read_table(committed).num_rows


def test_the_stage_refuses_a_box_whose_tile_lists_are_not_frozen_yet(tmp_path):
    cfg = BenchmarkConfig().replace(
        paths=BenchmarkConfig().paths.replace(artifacts_dir=str(tmp_path))
    )
    with pytest.raises(FileNotFoundError, match="freeze"):
        masks.freeze(cfg)


# --- Through seam 1 -----------------------------------------------------------


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """One real `freeze` of the shipped corpus, and what it said — shared by the class below.

    The `NotImplementedError` is the expected outcome and not a failure to write: `freeze` now
    commits §3.1's lists, §3.2's eval mask list and #151's train mask bank, and then names the
    stage #154 builds.

    Its console output is captured here rather than re-frozen under `capsys` for one test: 5,000
    test tiles is 30,000 draws and half a minute, and this suite is the gate before every push.
    The bank stage below it is held to K = 1 for the same reason — §3.2's K = 50 over 20,000
    head-train tiles is 1,000,000 draws and twenty minutes, and what this file is about is the
    eval mask list. `tests/test_benchmark_bank.py` is where the bank's own shape is asserted.
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
                    "--set", "masks.bank_draws_per_tile=1",
                    "--set", f'paths.mask_bank_path="{bank}"',
                ]
            )
    return directory, said.getvalue()


@pytest.mark.skipif(
    not (CORPUS_500K / "manifest.parquet").exists()
    or not Path(dataset_config.SPLIT_BLOCKS_PATH).exists(),
    reason="the shipped corpus or the committed split (#77) is not present",
)
class TestAgainstTheShippedCorpus:
    """§3.4's 30,000 rows at their real budget — 5,000 test tiles x 2 families x 3 buckets."""

    def test_freeze_commits_the_thirty_thousand_synthetic_rows(self, frozen):
        directory, _ = frozen
        table = pq.read_table(directory / EVAL_MASKS_NAME)
        assert table.num_rows == 30_000
        assert set(table["family"].to_pylist()) == set(Masks().synthetic_families)
        assert set(table["bucket"].to_pylist()) == {b for b, _ in Masks().buckets}

    def test_every_committed_row_names_a_tile_from_the_committed_test_list(self, frozen):
        directory, _ = frozen
        committed = {
            t for t, _ in read_tile_strata(directory / TEST_TILES_NAME, what="test list")
        }
        table = pq.read_table(directory / EVAL_MASKS_NAME, columns=["tile_id"])
        assert set(table["tile_id"].to_pylist()) == committed
        assert len(committed) == 5_000

    def test_every_eval_tile_appears_once_per_bucket_in_both_synthetic_families(self, frozen):
        # §3.2: "every eval tile appears once per bucket for the two synthetic families", so
        # bucket scores are computed over identical tile sets. A family or bucket that had
        # dropped a tile would print a perfectly normal column.
        directory, _ = frozen
        table = pq.read_table(directory / EVAL_MASKS_NAME)
        seen: dict[tuple[str, str], set[str]] = {}
        for tile, family, bucket in zip(
            table["tile_id"].to_pylist(),
            table["family"].to_pylist(),
            table["bucket"].to_pylist(),
        ):
            seen.setdefault((family, bucket), set()).add(tile)
        assert len(seen) == len(CELLS)
        assert {len(tiles) for tiles in seen.values()} == {5_000}

    def test_the_committed_masks_satisfy_the_constraints_they_were_drawn_under(self, frozen):
        # The whole artifact, not a sample: this is the file every arm scores against, and the
        # check is 30,000 decodes of a committed column rather than 30,000 draws.
        directory, _ = frozen
        table = pq.read_table(directory / EVAL_MASKS_NAME)
        for bucket, rle in zip(table["bucket"].to_pylist(), table["mask_rle"].to_pylist()):
            masks.check(masks.decode_rle(rle), bucket=bucket)

    def test_the_freeze_says_what_it_committed_and_what_it_hashes_to(self, frozen):
        # The md5 is on screen for the same reason the tile lists' are: it is what the run
        # record quotes the artifact by, and a `run` scored against a stale eval mask list
        # prints exactly as wide a table as a good one.
        directory, said = frozen
        assert file_md5(directory / EVAL_MASKS_NAME) in said
        assert "30,000 eval mask rows" in said
        assert masks.GENERATOR in said


def test_the_generator_reads_the_bucket_boundaries_from_the_config():
    # One home for §3.2's numbers. `benchmark.masks` draws whatever geometry it is handed, and
    # the one it defaults to is the config's — which is the object #151's bank hashes with.
    assert draw().geometry == BenchmarkConfig().masks
    assert math.isclose(Masks().max_equivalent_diameter_px, 159.86, abs_tol=0.01)
