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

**The committed artifact** attaches at seam 1 against the shipped corpus — 30,000 rows is 5,000
test tiles x 2 families x 3 buckets, and nothing smaller can express it. The stage is also driven
directly off a hand-written tile list, so the round-trip through the parquet is asserted on a box
with no corpus at all.

**§3.3's real family** (#153) is asserted at two altitudes, because building it needs the network
and this suite is the gate before every push.

- *Offline, in the default gate*: the whole transplant pipeline over injected granule and surface
  readers — the crop-and-paste identity, the tripwired interpolation kernels, the host banding,
  the excluded counts and the determinism. This is the safety net, and it runs everywhere.
- *Live, opt-in*: :class:`TestTheRealFamilyAgainstTheLiveBucket`, behind `DEMAE_S3_TEST=1` — the
  flag `test_tiling.py` and `test_masks.py` already use for the Copernicus bucket. It builds the
  family the way `freeze` does, off 24 granules and one GLO-30 COG per cell, and asserts §3.3's
  measured arithmetic: 8,824 + 438 + 11 published instances, 705 excluded in the sub-row gaps,
  182 above the 160 px top. ~165 s, which is why it is not in the default gate.

The split follows `tests/test_benchmark_voids.py`'s rule for the same readers — nothing in this
package's routine tests touches the network — rather than inventing a second convention.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import math
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.ndimage import binary_dilation

from benchmark import launch, masks, voids
from benchmark import tiles as tiles_module
from benchmark.artifacts import EVAL_MASKS_NAME, TEST_TILES_NAME
from benchmark.config import BenchmarkConfig, Masks
from dataset import config as dataset_config
from dataset.materialize import file_md5
from pretrain.artifact import read_tile_strata, write_tile_strata
from tests.conftest import SPLIT_COGS

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


@pytest.fixture
def real():
    """A two-row stand-in for §3.3's family, so the artifact mechanics run without a granule.

    `freeze`'s `real` seam exists for exactly this: the transplant needs 24 granules and a COG
    read per cell, and none of the tests below are about the transplant — they are about the
    guard, the schema and the idempotency of the write. Two real rows rather than none, so the
    file these tests drive is the two-family file the benchmark actually commits.
    """
    return _real_family(count=2)


def _real_family(*, count: int = 2, **kw) -> masks.RealFamily:
    rows = []
    for label in range(1, count + 1):
        placed = transplant(real_shape(30 + label, 33), label=label)
        rows.append(
            masks.MaskRow(
                tile_id=placed.tile_id,
                family=Masks().real_family,
                bucket=placed.row.name,
                mask_rle=placed.mask_rle,
                seed=placed.seed,
            )
        )
    fields = dict(
        excluded_above_top=7,
        excluded_in_gaps=3,
        failed_placement=1,
        components_at_or_above_floor=count + 11,
        cells=1,
        hosts=len({r.tile_id for r in rows}),
        max_reuse=1,
        seconds=0.0,
    )
    return masks.RealFamily(rows=tuple(rows), **{**fields, **kw})


def test_the_stage_writes_the_five_column_artifact(cfg, frozen_tiles, real):
    written = masks.freeze(cfg, real=real)
    table = pq.read_table(frozen_tiles / EVAL_MASKS_NAME)

    assert table.column_names == ["tile_id", "family", "bucket", "mask_rle", "seed"]
    assert table.num_rows == written.rows == 4 * len(CELLS) + len(real.rows)
    assert written.md5 == file_md5(frozen_tiles / EVAL_MASKS_NAME)


def test_the_real_family_lands_in_the_same_five_columns_as_the_synthetic_ones(
    cfg, frozen_tiles, real
):
    """#153's first acceptance criterion. One file, one schema, one row shape."""
    masks.freeze(cfg, real=real)
    table = pq.read_table(frozen_tiles / EVAL_MASKS_NAME)
    families = table["family"].to_pylist()

    assert set(families) == {*Masks().synthetic_families, Masks().real_family}
    assert families.count(Masks().real_family) == len(real.rows)
    # The real rows carry §3.3's sub-row where a synthetic row carries §3.2's bucket, and every
    # one of them decodes to a mask the same way.
    for family, bucket, rle in zip(families, table["bucket"].to_pylist(), table["mask_rle"].to_pylist()):
        if family == Masks().real_family:
            assert bucket in {row.name for row in voids.size_rows()}
            assert masks.decode_rle(rle).shape == (TILE_PX, TILE_PX)


def test_both_families_carry_the_same_generator_and_config_hash(cfg, frozen_tiles, real):
    """#153's fifth acceptance criterion, and what makes C4's ordering claim confound-free.

    There is one metadata block because there is one artifact: a real row and a synthetic row
    differ in the shape of the void and in nothing else about how they were produced. Two files
    — or one file with two generator strings — is where a difference in *provenance* could hide
    inside a comparison that reads as a difference in shape.
    """
    masks.freeze(cfg, real=real)
    metadata = pq.read_schema(frozen_tiles / EVAL_MASKS_NAME).metadata

    assert metadata[b"generator"].decode() == masks.GENERATOR
    assert metadata[b"artifact_hash"].decode() == cfg.artifact_hash
    assert int(metadata[b"real_rows"]) == len(real.rows)
    assert int(metadata[b"synthetic_rows"]) == 4 * len(CELLS)


def test_the_artifact_records_what_the_real_family_excluded_rather_than_going_quiet(
    cfg, frozen_tiles, real
):
    """#153's fourth acceptance criterion — *counts, not silence*.

    The ceiling on large real voids is this benchmark's tile, not nature: giants over 400 px
    exist in abundance and cannot fit a 224 px tile under the 40% area cap. A histogram that
    simply stopped at 160 px would imply a scarcity that is not there, so the number that was
    cut off is committed beside the rows that survived.
    """
    masks.freeze(cfg, real=real)
    metadata = pq.read_schema(frozen_tiles / EVAL_MASKS_NAME).metadata

    assert int(metadata[b"real_excluded_above_top"]) == real.excluded_above_top == 7
    assert int(metadata[b"real_excluded_in_gaps"]) == real.excluded_in_gaps == 3
    assert int(metadata[b"real_failed_placement"]) == real.failed_placement == 1
    # §3.3's two counts are two questions: the published instances, and the wider figure that
    # also holds the gap components. Neither is the other.
    assert real.scored_components == len(real.rows) + real.excluded_in_gaps
    assert int(metadata[b"real_scored_components"]) == real.scored_components


def test_a_family_whose_attrition_moved_is_a_different_draw(cfg, frozen_tiles, real):
    """The excluded counts are part of the artifact's identity, not a note beside it."""
    masks.freeze(cfg, real=real)

    with pytest.raises(FileExistsError, match="already frozen at md5"):
        masks.freeze(cfg, real=_real_family(count=2, excluded_above_top=8))


def test_the_artifact_records_the_generator_config_hash_the_bank_will_carry(cfg, frozen_tiles, real):
    """§3.4's determinism contract is *one* contract, so the hash is in the file, not beside it.

    `artifact_hash` is the one #151's bank sidecar quotes: it covers §3.1's tiles and §3.2's
    masks and nothing about this box, so two artifacts that agree on it were generated from the
    same recipe whatever paths they were written through.
    """
    masks.freeze(cfg, real=real)
    metadata = pq.read_schema(frozen_tiles / EVAL_MASKS_NAME).metadata
    assert metadata[b"artifact_hash"].decode() == cfg.artifact_hash
    assert metadata[b"generator"].decode() == masks.GENERATOR


def test_a_rewrite_of_the_same_draw_is_byte_identical(cfg, frozen_tiles, real):
    first = masks.freeze(cfg, real=real).md5
    (frozen_tiles / EVAL_MASKS_NAME).unlink()
    assert masks.freeze(cfg, real=real).md5 == first


def test_rewriting_the_unchanged_artifact_is_a_no_op_rather_than_a_refusal(cfg, real):
    assert masks.freeze(cfg, real=real).md5 == masks.freeze(cfg, real=real).md5


def test_an_eval_mask_list_whose_content_hash_has_moved_is_not_overwritten(cfg, frozen_tiles, real):
    masks.freeze(cfg, real=real)
    moved = cfg.replace(masks=cfg.masks.replace(seed=cfg.masks.seed + 1))
    with pytest.raises(FileExistsError, match="already frozen at md5"):
        masks.freeze(moved, real=real)


def test_the_guard_reads_the_committed_draw_and_not_the_bytes_it_arrived_in(cfg, frozen_tiles, real):
    """§3.4's refusal is about the recipe, so a re-encoding of the same draw is not a change.

    A parquet footer carries the writing library's own version string, so a guard keyed on byte
    equality would refuse a `freeze` that had re-derived the identical 30,000 masks, the day
    pyarrow was upgraded — a refusal about the environment wearing the words of one about the
    draw. The same rows written under a different codec stand in for that here.
    """
    import pyarrow.parquet as pq

    rows = masks.eval_rows(tile_ids(4), cfg.masks)
    committed = frozen_tiles / EVAL_MASKS_NAME
    masks.freeze(cfg, real=real)
    table = pq.read_table(committed)
    pq.write_table(table, committed, compression="snappy")
    reencoded = file_md5(committed)

    # Not a refusal, and not a rewrite: the committed bytes are what every run record quotes.
    assert masks.freeze(cfg, real=real).md5 == reencoded
    assert file_md5(committed) == reencoded
    assert len(rows) + len(real.rows) == pq.read_table(committed).num_rows


def test_the_stage_refuses_a_box_whose_tile_lists_are_not_frozen_yet(tmp_path):
    cfg = BenchmarkConfig().replace(
        paths=BenchmarkConfig().paths.replace(artifacts_dir=str(tmp_path))
    )
    with pytest.raises(FileNotFoundError, match="freeze"):
        masks.freeze(cfg, real=real)


# --- Through seam 1 -----------------------------------------------------------


GRANULES = Path("data/srtm_voids/num")

#: The shipped corpus and #77's committed split — everything the synthetic budget needs, and
#: all of it on this box.
HAS_CORPUS = (
    (CORPUS_500K / "manifest.parquet").exists()
    and Path(dataset_config.SPLIT_BLOCKS_PATH).exists()
)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """§3.1's lists and §3.2's 30,000 rows at their real budget, off the shipped corpus.

    Drives `masks.freeze` directly, with §3.3's family **supplied** rather than built. That is
    the one thing this fixture cannot do offline: the real family reads one GLO-30 COG per cell
    from the live Copernicus bucket, and `tests/test_benchmark_voids.py` sets this package's
    rule — *nothing here touches the network*. The live build is asserted at its own real budget
    in :class:`TestTheRealFamilyAgainstTheLiveBucket`, behind the repo's `DEMAE_S3_TEST` flag.

    Its console output is captured here rather than re-frozen under `capsys` for one test: 5,000
    test tiles is 30,000 draws and half a minute, and this suite is the gate before every push.
    """
    directory = tmp_path_factory.mktemp("task1")
    cfg = BenchmarkConfig().replace(
        paths=BenchmarkConfig().paths.replace(artifacts_dir=str(directory))
    )
    tiles_module.freeze(cfg)
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        written = masks.freeze(cfg, real=_real_family(count=2))
        for line in masks.format_written(written, cfg.masks):
            print(line)
    return directory, said.getvalue()


@pytest.mark.skipif(not HAS_CORPUS, reason="the shipped corpus or the committed split (#77)")
class TestAgainstTheShippedCorpus:
    """§3.2's 30,000 rows at their real budget — 5,000 test tiles x 2 families x 3 buckets."""

    def test_freeze_commits_the_thirty_thousand_synthetic_rows(self, frozen):
        directory, _ = frozen
        table = pq.read_table(directory / EVAL_MASKS_NAME)
        synthetic = [f for f in table["family"].to_pylist() if f != Masks().real_family]
        assert len(synthetic) == 30_000
        assert set(synthetic) == set(Masks().synthetic_families)

    def test_every_committed_row_names_a_tile_from_the_committed_test_list(self, frozen):
        # The synthetic half, whose hosts *are* the list. The real half's hosts are a subset of
        # the same list by construction — `HostPool.from_test_list` reads it — and that is
        # asserted against a live build in `TestTheRealFamilyAgainstTheLiveBucket`, since this
        # fixture supplies a stand-in family whose host ids are fabricated.
        directory, _ = frozen
        committed = {
            t for t, _ in read_tile_strata(directory / TEST_TILES_NAME, what="test list")
        }
        table = pq.read_table(directory / EVAL_MASKS_NAME, columns=["tile_id", "family"])
        synthetic = {
            t
            for t, f in zip(table["tile_id"].to_pylist(), table["family"].to_pylist())
            if f != Masks().real_family
        }
        assert synthetic == committed
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
            if family != Masks().real_family:
                seen.setdefault((family, bucket), set()).add(tile)
        assert len(seen) == len(CELLS)
        assert {len(tiles) for tiles in seen.values()} == {5_000}
        # The real family is deliberately not in that shape: it carries no bucket axis, and its
        # rows repeat `tile_id` because ~9,000 voids share ~1,100 R4 hosts (§3.3, §5).

    def test_the_committed_masks_satisfy_the_constraints_they_were_drawn_under(self, frozen):
        # The whole artifact, not a sample: this is the file every arm scores against, and the
        # check is 30,000 decodes of a committed column rather than 30,000 draws. The real half
        # gets the same pass one test up, against §3.3's sub-rows rather than §3.2's buckets.
        directory, _ = frozen
        table = pq.read_table(directory / EVAL_MASKS_NAME)
        for family, bucket, rle in zip(
            table["family"].to_pylist(),
            table["bucket"].to_pylist(),
            table["mask_rle"].to_pylist(),
        ):
            if family != Masks().real_family:
                masks.check(masks.decode_rle(rle), bucket=bucket)

    def test_the_freeze_says_what_it_committed_and_what_it_hashes_to(self, frozen):
        # The md5 is on screen for the same reason the tile lists' are: it is what the run
        # record quotes the artifact by, and a `run` scored against a stale eval mask list
        # prints exactly as wide a table as a good one.
        directory, said = frozen
        assert file_md5(directory / EVAL_MASKS_NAME) in said
        assert "30,000 synthetic" in said
        assert masks.GENERATOR in said

    def test_the_freeze_prints_what_the_real_family_excluded(self, frozen):
        """The attrition is narrated, not just committed — #153's fourth criterion, on screen.

        The stand-in family this fixture supplies carries small made-up counts, so what is
        asserted here is that each one reaches the console at all. The real numbers — 705 in
        the gaps, 182 above the 160 px top — are asserted against a live build below.
        """
        _, said = frozen
        assert "scored components" in said
        assert "sub-row gaps and are excluded" in said
        assert f"above the {Masks().real_band_top_px} px top" in said
        assert "tile geometry rather than by scarcity" in said


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One real `python -m benchmark.launch freeze`, both families, off the live bucket.

    The `NotImplementedError` is the expected outcome and not a failure to write: `freeze`
    commits §3.1's lists, the extractor spec and the eval mask list, then names #151's stage.
    """
    directory = tmp_path_factory.mktemp("task1-real")
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        with pytest.raises(NotImplementedError, match="train mask bank"):
            launch.main(["freeze", "--artifacts-dir", str(directory)])
    return directory, said.getvalue()


@pytest.mark.skipif(
    not HAS_CORPUS
    or not any(GRANULES.glob("*.num.zip"))
    or os.environ.get("DEMAE_S3_TEST") != "1",
    reason="set DEMAE_S3_TEST=1, with the corpus and #127's granules, to build the real family",
)
class TestTheRealFamilyAgainstTheLiveBucket:
    """§3.3's family the way `freeze` builds it: 24 granules and one live GLO-30 COG per cell.

    Opt-in on `DEMAE_S3_TEST`, the flag `test_tiling.py` and `test_masks.py` already use for
    the live Copernicus bucket. It stays out of the default gate for the reason
    `tests/test_benchmark_voids.py` states for its own readers — nothing in this package's
    routine tests touches the network — and because the extraction is ~140 s of granule work
    that would more than double the suite every push. It is the test that would catch a family
    that had quietly stopped being produced, so it is written to be run, not to be admired.
    """

    def test_freeze_commits_the_real_family_beside_the_synthetic_one(self, built):
        """§3.3's published instances — the 8,824 + 438 + 11 the sub-rows carry.

        **Not 9,978.** That figure is §3.3's *scored components*, which also counts the 705 that
        placed cleanly and fell in the gaps between sub-rows — and those are excluded from the
        published rows by the same sentence of the plan that names them. Plan §3.4 and the build
        spec both called this half "9,978", conflating the two counts; the amendment is recorded
        in both, and this assertion is the arithmetic.
        """
        directory, _ = built
        table = pq.read_table(directory / EVAL_MASKS_NAME)
        metadata = pq.read_schema(directory / EVAL_MASKS_NAME).metadata

        real = [
            b
            for f, b in zip(table["family"].to_pylist(), table["bucket"].to_pylist())
            if f == Masks().real_family
        ]
        assert {name: real.count(name) for name in set(real)} == {
            "real-small": 8_824,
            "real-medium+": 438,
            "count-only-128-160": 11,
        }
        assert len(real) == 9_273 == int(metadata[b"real_rows"])
        assert int(metadata[b"real_scored_components"]) == 9_978
        assert int(metadata[b"real_excluded_in_gaps"]) == 705
        assert int(metadata[b"real_excluded_above_top"]) == 182
        assert table.num_rows == 30_000 + 9_273

    def test_every_real_mask_is_interior_and_files_under_its_own_sub_row(self, built):
        """The whole real half, decoded — the pass the synthetic half gets against its buckets."""
        directory, _ = built
        rows = {row.name: row for row in voids.size_rows()}
        table = pq.read_table(directory / EVAL_MASKS_NAME)
        for family, bucket, rle in zip(
            table["family"].to_pylist(),
            table["bucket"].to_pylist(),
            table["mask_rle"].to_pylist(),
        ):
            if family == Masks().real_family:
                masks.check_real(masks.decode_rle(rle), row=rows[bucket])

    def test_every_real_row_lands_on_a_clean_stratum_matched_test_tile(self, built):
        """#153's third criterion, against the committed artifact rather than against the code.

        §3.3's realised histogram is 99.94% R4, so essentially every host is an R4 tile — the
        measured pool is 1,111 clean R4 tiles in the 5,000-tile test draw, and 9,273 voids over
        them is ~8 shapes a host. That reuse is why §5 bootstraps over tiles, not instances.
        """
        directory, _ = built
        bands = voids.Bands.from_manifest(CORPUS_500K)
        pool = masks.HostPool.from_test_list(
            directory / TEST_TILES_NAME, CORPUS_500K, bands=bands
        )
        clean = {tile for hosts in pool.by_band.values() for tile in hosts}

        table = pq.read_table(directory / EVAL_MASKS_NAME)
        hosts = [
            t
            for t, f in zip(table["tile_id"].to_pylist(), table["family"].to_pylist())
            if f == Masks().real_family
        ]
        assert set(hosts) <= clean, "a host outside the clean pool is not a clean test tile"
        in_r4 = len(set(hosts) & set(pool.by_band[3]))
        assert in_r4 / len(set(hosts)) > 0.99

    def test_the_freeze_says_what_the_tile_geometry_cut_off(self, built):
        _, said = built
        assert "9,273 real" in said and "39,273 eval mask rows" in said
        assert "9,978 scored components" in said
        assert "705" in said and "182" in said


# --- §3.3's real family: the transplant ---------------------------------------


ROWS = {row.name: row for row in voids.size_rows()}

#: A band index with hosts in it. §3.3's realised histogram is 99.94% R4, so this is the band
#: essentially every real instance lands in — `band` 3 is `Bands.label` R4.
R4 = 3


def real_shape(height: int, width: int, *, hole: bool = True) -> np.ndarray:
    """A non-convex void shape of a given bounding box — real voids are strikingly non-compact.

    The notch matters: a rectangle is the one shape a resampling bug could leave looking right,
    because its bounding box, its area and its diameter all survive a nearest-neighbour scale.
    """
    shape = np.ones((height, width), dtype=bool)
    if hole:
        shape[: height // 2, width // 2 :] = False
    return shape


def shape_of_diameter(d: float) -> np.ndarray:
    """A notched shape whose *achieved* equivalent diameter is ``d``, for filing under a row."""
    for side in range(2, 400):
        candidate = real_shape(side, side)
        if masks.equivalent_diameter(int(candidate.sum())) >= d:
            return candidate
    raise AssertionError(f"no shape reached d={d}")


def host_pool(*bands_to_tiles) -> masks.HostPool:
    """A pool with tiles in the bands named, and nothing read off a corpus."""
    by_band = {band: tuple(tiles) for band, tiles in bands_to_tiles}
    return masks.HostPool(by_band=by_band, considered=sum(len(t) for _, t in bands_to_tiles))


POOL = host_pool((R4, ["N45_E007_00000", "N45_E007_00001", "N45_E007_00002"]))


def component(shape: np.ndarray, *, cell: str = "N27E088", label: int = 1) -> voids.Component:
    """A `Component` describing ``shape``, as `benchmark.voids` would have measured it."""
    area = int(shape.sum())
    return voids.Component(
        cell=cell,
        label=label,
        area_px=area,
        eqdiam_px=masks.equivalent_diameter(area),
        row=float(shape.shape[0]) / 2,
        col=float(shape.shape[1]) / 2,
        bbox_h=shape.shape[0],
        bbox_w=shape.shape[1],
        fits_interior=True,
        fits_area=True,
    )


def transplant(shape: np.ndarray, **kw):
    """Transplant ``shape``, filed under the sub-row its own achieved diameter falls in."""
    row = kw.pop("row", None)
    if row is None:
        row = voids.size_row_of(masks.equivalent_diameter(int(shape.sum())), voids.size_rows())
        assert row is not None, "the fixture shape has to land in one of §3.3's sub-rows"
    return masks.transplant(component(shape, **kw), shape, row=row, band=R4, pool=POOL)


def test_a_transplanted_void_is_the_source_shape_translated_and_nothing_else():
    """The ticket's sentence — *cropped and pasted, never resampled* — as an identity.

    Not "the same area" and not "the same diameter": the pasted pixels are asserted to be the
    source array, elementwise, at some offset. A resampled shape can hold its area and its
    bounding box and still be a different set of pixels, and that is the failure with nothing
    downstream to catch it — the family's whole claim is that these shapes are real.
    """
    shape = real_shape(37, 44)

    placed = transplant(shape)

    rows, cols = np.nonzero(placed.mask)
    box = placed.mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    assert box.shape == shape.shape
    assert np.array_equal(box, shape)
    assert int(placed.mask.sum()) == int(shape.sum())
    assert placed.mask.dtype == np.bool_


def test_a_transplant_preserves_the_components_own_pixel_count_exactly():
    # Any interpolation kernel worth the name moves this number; a nearest-neighbour one moves
    # it too, unless the scale is exactly 1 — in which case it is not a resampling at all.
    for height, width in [(17, 61), (33, 33), (64, 21), (100, 100)]:
        shape = real_shape(height, width)
        assert int(transplant(shape).mask.sum()) == int(shape.sum())


def test_a_transplanted_void_is_fully_interior_and_under_the_area_cap():
    geometry = Masks()
    margin = geometry.edge_margin_px
    for seed_label in range(40):
        placed = transplant(real_shape(60, 70), label=seed_label + 1)
        assert not placed.mask[:margin].any() and not placed.mask[-margin:].any()
        assert not placed.mask[:, :margin].any() and not placed.mask[:, -margin:].any()
        assert placed.mask.sum() <= geometry.max_area_frac * placed.mask.size


def test_the_same_component_transplants_the_same_way_twice():
    shape = real_shape(41, 39)
    first, second = transplant(shape), transplant(shape)

    assert first.seed == second.seed
    assert first.tile_id == second.tile_id
    assert np.array_equal(first.mask, second.mask)


def test_a_different_component_is_a_different_transplant():
    shape = real_shape(41, 39)
    a = transplant(shape, label=1)
    b = transplant(shape, label=2)
    c = transplant(shape, cell="N28E086", label=1)

    assert len({a.seed, b.seed, c.seed}) == 3
    assert not np.array_equal(a.mask, b.mask) or a.tile_id != b.tile_id


def test_moving_the_generator_seed_moves_every_transplant():
    shape = real_shape(41, 39)
    moved = masks.transplant(
        component(shape), shape, row=ROWS["real-small"], band=R4, pool=POOL,
        geometry=Masks().replace(seed=Masks().seed + 1),
    )
    assert moved.seed != transplant(shape).seed


def test_a_shape_too_big_for_the_interior_is_refused_and_never_trimmed():
    """§3.2's interiority is a constraint on a real void exactly as it is on a drawn one.

    There is no redraw available here — the shape is a measurement — so the only two answers
    are place it or refuse it. Trimming would file a clipped void under the sub-row its *whole*
    diameter fell in, which is the one repair `benchmark.masks` exists to not make.
    """
    interior = TILE_PX - 2 * Masks().edge_margin_px
    oversized = np.ones((interior + 1, 10), dtype=bool)

    with pytest.raises(masks.NotInterior):
        transplant(oversized)


def test_a_transplant_is_refused_when_no_host_carries_the_voids_band():
    shape = real_shape(30, 30)
    empty = host_pool((0, ["N45_E007_00000"]))

    with pytest.raises(masks.MaskError, match="R4"):
        masks.transplant(component(shape), shape, row=ROWS["real-small"], band=R4, pool=empty)


def test_a_transplanted_void_stays_in_the_size_row_it_is_filed_under():
    shape = shape_of_diameter(20.0)

    placed = transplant(shape, row=ROWS["real-small"])

    assert ROWS["real-small"].contains(masks.equivalent_diameter(int(placed.mask.sum())))
    with pytest.raises(masks.OutOfBucket):
        masks.check_real(placed.mask, row=ROWS["real-medium+"])


# --- The real family, end to end off injected granules ------------------------


#: The resampling entry points a transplant could plausibly reach for, by module and name. If
#: any of them fires while the family is being built, the shapes in the committed column are
#: not the shapes SRTM measured.
#:
#: `rasterio.warp` is in the list even though the array path never imports it, because the one
#: place a raster-level resample could enter this pipeline is
#: :func:`benchmark.voids.read_surface` — and a tripwire is for the edit that has not been made
#: yet. Absent modules are skipped rather than failing the fixture, so the list can name a
#: kernel a lean install does not carry.
INTERPOLATORS = [
    ("scipy.ndimage", name)
    for name in (
        "zoom",
        "map_coordinates",
        "affine_transform",
        "geometric_transform",
        "spline_filter",
        "spline_filter1d",
        "rotate",
        "shift",
    )
] + [
    ("numpy", "interp"),
    ("numpy", "resize"),
    ("rasterio.warp", "reproject"),
    ("rasterio.enums", "Resampling"),
]


@pytest.fixture
def no_interpolation(monkeypatch):
    """Make every resampling kernel raise, and hand back the list of the ones that were hit."""
    import importlib

    fired: list[str] = []

    for module_name, attribute in INTERPOLATORS:
        module = importlib.import_module(module_name)
        if not hasattr(module, attribute):
            continue

        def tripwire(*_args, _name=f"{module_name}.{attribute}", **_kw):
            fired.append(_name)
            raise AssertionError(f"{_name} was called on a mask")

        monkeypatch.setattr(module, attribute, tripwire)
    return fired


def _granule(shapes_at, *, size=voids.SRTM_PX // 6):
    """A `.num` raster with the given ``(row0, col0, shape)`` voids punched into observed ground."""
    num = np.full((size, size), 210, dtype=np.uint8)
    for row0, col0, shape in shapes_at:
        block = num[row0 : row0 + shape.shape[0], col0 : col0 + shape.shape[1]]
        block[shape] = 50
    return num


def _surface(size, slope_deg=30.0):
    dy = 111_320.0 / size
    rise = math.tan(math.radians(slope_deg)) * dy
    rows = np.arange(size, dtype=np.float32).reshape(-1, 1)
    return np.broadcast_to(1000.0 + rows * rise, (size, size)).astype(np.float32).copy()


@pytest.fixture
def one_granule():
    """A single-cell catalog: three real-small shapes on steep ground, read without a network."""
    sources = [real_shape(30, 33), real_shape(28, 40), real_shape(35, 31)]
    num = _granule([(80, 80, sources[0]), (300, 200, sources[1]), (400, 420, sources[2])])
    surface = _surface(num.shape[0] - 1)
    return sources, num, surface


def _build(num, surface, pool=POOL, **kw):
    return masks.real_rows(
        ["N27E088"],
        pool=pool,
        bands=TEST_BANDS,
        read_num_fn=lambda cell: num,
        read_surface_fn=lambda cell: surface,
        **kw,
    )


def test_the_real_family_transplants_every_published_instance(one_granule):
    sources, num, surface = one_granule

    family = _build(num, surface)

    assert len(family.rows) == len(sources)
    assert {row.family for row in family.rows} == {Masks().real_family}
    assert {row.bucket for row in family.rows} == {"real-small"}
    assert family.scored_components == len(sources) + family.excluded_in_gaps
    # Every committed mask carries a source shape's exact pixel count, and the multiset of
    # counts is the multiset the granule held — no shape was dropped, doubled or resized.
    assert sorted(int(masks.decode_rle(row.mask_rle).sum()) for row in family.rows) == sorted(
        int(shape.sum()) for shape in sources
    )


def test_no_interpolation_kernel_touches_a_mask(one_granule, no_interpolation):
    """#153's second acceptance criterion, taken literally.

    Below 50 degrees there is no resampling step to get wrong — SRTM 1 arc-second and GLO-30
    are the same lattice — so the honest assertion is not that the transplant interpolates
    *well* but that it does not interpolate at all. Every kernel it could have reached for is
    replaced with a tripwire, and the family is built through them.

    What this cannot see, and :func:`test_every_committed_real_mask_is_a_source_shape_translated`
    can: both readers are injected here, so no raster is opened and a resample *inside a reader*
    would go unnoticed. That is why the elementwise identity is asserted separately rather than
    this test being the whole argument — the tripwires prove the code path is clean, the
    identity proves the pixels are.
    """
    sources, num, surface = one_granule

    family = _build(num, surface)

    assert no_interpolation == []
    assert len(family.rows) == len(sources)


def test_every_committed_real_mask_is_a_source_shape_translated(one_granule):
    """The identity that survives without a tripwire: the pixels are the measured pixels."""
    sources, num, surface = one_granule
    wanted = {shape.tobytes(): shape for shape in sources}

    for row in _build(num, surface).rows:
        mask = masks.decode_rle(row.mask_rle)
        rows_, cols = np.nonzero(mask)
        box = mask[rows_.min() : rows_.max() + 1, cols.min() : cols.max() + 1]
        assert box.tobytes() in wanted, "a committed mask is not one of the granule's shapes"
        assert np.array_equal(box, wanted[box.tobytes()])


def test_every_real_row_names_a_host_from_the_pool_in_the_voids_own_band(one_granule):
    """#153's third acceptance criterion: clean, from the test list, and stratum-matched."""
    _, num, surface = one_granule
    pool = host_pool((0, ["flat-a", "flat-b"]), (R4, ["steep-a", "steep-b", "steep-c"]))

    family = _build(num, surface, pool=pool)

    # The fixture surface is 30 degrees, which is R4 under TEST_BANDS' interior edge of 20.
    assert {row.tile_id for row in family.rows} <= set(pool.by_band[R4])
    assert not {row.tile_id for row in family.rows} & set(pool.by_band[0])


def test_the_real_family_is_the_same_family_twice(one_granule):
    _, num, surface = one_granule

    first, second = _build(num, surface), _build(num, surface)

    assert [dataclasses.astuple(row) for row in first.rows] == [
        dataclasses.astuple(row) for row in second.rows
    ]


def test_the_real_family_counts_what_the_tile_geometry_and_the_gaps_cut_off():
    """The excluded components are counted where they fall, and never quietly dropped."""
    big = real_shape(*(2 * [_side(200.0)]))  # over the 160 px top
    gap = shape_of_diameter(55.0)  # between real-small and real-medium+
    small = real_shape(30, 33)
    num = _granule([(50, 50, big), (500, 60, gap), (500, 400, small)], size=900)

    family = _build(num, _surface(899))

    assert len(family.rows) == 1, "only the real-small shape is a published instance"
    assert family.excluded_above_top == 1
    assert family.excluded_in_gaps == 1
    assert family.scored_components == 2, "the gap component is scored and not published"
    assert family.components_at_or_above_floor == 3


def _side(d: float) -> int:
    """The side of a solid square whose equivalent diameter is ``d`` before the notch."""
    return max(1, int(round(math.sqrt(math.pi * (d / 2.0) ** 2) * math.sqrt(4 / 3))))


# --- The host pool ------------------------------------------------------------


def test_hosts_come_only_from_the_bands_they_were_filed_under():
    pool = host_pool((0, ["flat-a", "flat-b"]), (R4, ["steep-a", "steep-b"]))
    rng = np.random.default_rng(0)

    assert {pool.pick(R4, rng) for _ in range(50)} <= {"steep-a", "steep-b"}
    assert {pool.pick(0, rng) for _ in range(50)} <= {"flat-a", "flat-b"}


TEST_BANDS = voids.Bands(edges=(0.0, 1.0, 5.0, 20.0, 90.0))


def _corpus_with_slopes(factory, root, slopes: list[float]):
    """A corpus whose tiles carry the given ``slope_mean`` values, in manifest order."""
    per_cog = 2
    corpus = factory(
        root,
        SPLIT_COGS,
        tiles_per_cog=per_cog,
        metrics_fn=lambda c, j: {"slope_mean": slopes[c * per_cog + j]},
    )
    table = pq.read_table(corpus / "manifest.parquet", columns=["tile_id"])
    return corpus, table["tile_id"].to_pylist()


def _set_column(corpus, name: str, values):
    """Rewrite one manifest column — the `set_split` fixture's trick, for another column."""
    import pyarrow as pa

    from dataset.manifest import MANIFEST_SCHEMA

    path = corpus / "manifest.parquet"
    table = pq.read_table(path)
    field = table.schema.field(name)
    patched = table.set_column(
        table.schema.get_field_index(name), name, pa.array(values, type=field.type)
    )
    pq.write_table(patched.cast(MANIFEST_SCHEMA), path)


def _write_list(directory, tile_ids) -> Path:
    write_tile_strata([(t, 0) for t in tile_ids], directory / TEST_TILES_NAME)
    return directory / TEST_TILES_NAME


def test_the_host_pool_takes_its_tiles_from_the_test_list_and_bands_them(
    tmp_path, corpus_factory
):
    """Hosts are the committed test list's tiles, banded on the ground each one actually is."""
    slopes = [0.5, 3.0, 10.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    corpus, tile_ids = _corpus_with_slopes(corpus_factory, tmp_path / "corpus", slopes)
    listed = tile_ids[:4]

    pool = masks.HostPool.from_test_list(_write_list(tmp_path, listed), corpus, bands=TEST_BANDS)

    assert pool.considered == 4
    assert pool.tiles == 4
    assert {t for tiles in pool.by_band.values() for t in tiles} == set(listed)
    # 0.5 / 3 / 10 / 30 degrees against interior edges 1, 5, 20 — one tile per band.
    assert {band: len(tiles) for band, tiles in sorted(pool.by_band.items())} == {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
    }


def test_a_tile_the_band_cut_would_not_have_seen_is_not_a_host(tmp_path, corpus_factory):
    """§3.3's hosts are *clean* GLO-30 test tiles — the pool §5 cut its own band edges over.

    Cleanliness has one definition (:func:`benchmark.voids.eligible`) rather than two, because a
    host outside the pool the band edges were cut over carries a band index that means something
    slightly different from every other band index in the corpus.
    """
    corpus, tile_ids = _corpus_with_slopes(corpus_factory, tmp_path / "corpus", [30.0] * 10)
    _set_column(corpus, "water_affected", [i % 2 == 1 for i in range(len(tile_ids))])

    pool = masks.HostPool.from_test_list(
        _write_list(tmp_path, tile_ids), corpus, bands=TEST_BANDS
    )

    assert pool.considered == len(tile_ids)
    assert pool.tiles == len(tile_ids) // 2, "every water-affected tile has to be dropped"
    kept = {t for tiles in pool.by_band.values() for t in tiles}
    assert kept == {t for i, t in enumerate(tile_ids) if i % 2 == 0}


def test_a_test_list_with_no_clean_host_in_a_band_is_refused_rather_than_drawn_thin(
    tmp_path, corpus_factory
):
    corpus, tile_ids = _corpus_with_slopes(corpus_factory, tmp_path / "corpus", [30.0] * 10)
    _set_column(corpus, "water_affected", [True] * len(tile_ids))

    with pytest.raises(ValueError, match="no clean"):
        masks.HostPool.from_test_list(_write_list(tmp_path, tile_ids), corpus, bands=TEST_BANDS)


def test_the_generator_reads_the_bucket_boundaries_from_the_config():
    # One home for §3.2's numbers. `benchmark.masks` draws whatever geometry it is handed, and
    # the one it defaults to is the config's — which is the object #151's bank hashes with.
    assert draw().geometry == BenchmarkConfig().masks
    assert math.isclose(Masks().max_equivalent_diameter_px, 159.86, abs_tol=0.01)
