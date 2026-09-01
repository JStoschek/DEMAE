"""§3.4's dataset card — what `freeze` produced, in the document that says so (#154).

The card is the last thing `freeze` writes and the only one of its artifacts written for a
*reader* rather than for a later stage. So what this file asserts is not a schema: it is that
the two things plan §3.4 says a reader must be **told** rather than left to infer are on the
page, that every artifact is named by its md5, and that the generator config hash appears once
because the two mask artifacts carry one contract rather than two.

Two of those are inference traps and are tested as such. A reader who assumes the 104 MB bank
is in git draws the wrong conclusion about the determinism contract — which is precisely why
the spec resolved it explicitly instead of leaving it implicit — and a reader who assumes the
benchmark ships DEM pixels asks a redistribution question the module does not raise.

The third thing asserted here has no acceptance criterion of its own and is the reason the card
is not written through :func:`benchmark.artifacts.write_committed`: **the card is a function of
the draw and of nothing about this box**. Two boxes that freeze one config into different
directories, with the bank at different paths, produce the same bytes. Without that the card
would churn its md5 on every machine that touched it, and the file git carries would be a
record of whose laptop ran `freeze` last.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark import card, launch, masks
from benchmark.artifacts import (
    EVAL_MASKS_NAME,
    MASK_BANK_SIDECAR_NAME,
    TEST_TILES_NAME,
    TILE_LIST_NAMES,
    VAL_TILES_NAME,
)
from benchmark.config import BenchmarkConfig
from dataset.materialize import file_md5
from pretrain.artifact import write_tile_strata

#: A box's tile lists, at sizes a test can write. The card reads its row counts off these files
#: rather than off the config, on the contract family's rule that the committed file is what
#: later phases re-read — so the numbers here are what the card must print.
LIST_ROWS = {
    TILE_LIST_NAMES["20k"]: 40,
    TILE_LIST_NAMES["5k"]: 10,
    TILE_LIST_NAMES["1k"]: 2,
    VAL_TILES_NAME: 4,
    TEST_TILES_NAME: 8,
}


def write_lists(directory: Path) -> None:
    """§3.1's five committed files, small and real — two columns, sorted, LF-terminated."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in LIST_ROWS.items():
        write_tile_strata(
            [(f"N45_E007_{i:05d}", i % 20) for i in range(rows)], directory / name
        )


#: §3.3's measured attrition over #127's 24 granules, and it is an exact ladder:
#: 10,291 − 182 above the top − 131 failed placements = 9,978 scored components, less the 705
#: that fell in the sub-row gaps = 9,273 published rows. The real numbers rather than stand-ins,
#: because what the card owes here is the arithmetic a reader does down the page, and stand-ins
#: that did not subtract would let a card printing them in any order pass.
PUBLISHED_ROWS = 9_273


def real_family() -> masks.RealFamily:
    """§3.3's family as `freeze` measures it — its published rows, and what it excluded."""
    return masks.RealFamily(
        rows=tuple(
            masks.MaskRow(
                tile_id=f"N45_E007_{i % 8:05d}",
                family="real",
                bucket="real-small",
                mask_rle="0 1",
                seed=i,
            )
            for i in range(PUBLISHED_ROWS)
        ),
        excluded_above_top=182,
        excluded_in_gaps=705,
        failed_placement=131,
        components_at_or_above_floor=10_291,
        cells=24,
        hosts=1_250,
        max_reuse=7,
        seconds=1.5,
    )


def eval_masks(directory: Path, *, artifact_hash: str, real: masks.RealFamily) -> masks.EvalMaskList:
    """The eval mask list as `freeze` hands it on, over a stand-in file with a real md5."""
    path = directory / EVAL_MASKS_NAME
    path.write_bytes(b"thirty thousand synthetic rows, allegedly")
    return masks.EvalMaskList(
        path=path,
        rows=48 + len(real.rows),
        md5=file_md5(path),
        artifact_hash=artifact_hash,
        tiles=LIST_ROWS[TEST_TILES_NAME],
        seconds=2.0,
        real=real,
    )


def bank(directory: Path, bank_path: Path, *, artifact_hash: str) -> masks.TrainMaskBank:
    """The train mask bank as `freeze` hands it on — the sidecar, and the bytes it describes."""
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_bytes(b"one million rle rows, allegedly")
    draw = masks.BankDraw(
        generator=masks.GENERATOR,
        artifact_hash=artifact_hash,
        rows=2_000,
        tiles=LIST_ROWS[TILE_LIST_NAMES["20k"]],
        draws_per_tile=50,
        columns=",".join(masks.BANK_COLUMNS),
    )
    sidecar = masks.BankSidecar(
        draw=draw,
        draw_hash="d" * 32,
        md5=file_md5(bank_path),
        size_bytes=103_653_977,
    )
    (directory / MASK_BANK_SIDECAR_NAME).write_text(sidecar.render())
    return masks.TrainMaskBank(
        path=bank_path,
        sidecar_path=directory / MASK_BANK_SIDECAR_NAME,
        committed=sidecar,
        seconds=1_265.4,
        generated=True,
    )


@pytest.fixture
def box(tmp_path):
    """A frozen box: five tile lists, an eval mask list, a bank and its sidecar."""
    directory = tmp_path / "artifacts"
    write_lists(directory)
    cfg = BenchmarkConfig().replace(
        paths=BenchmarkConfig().paths.replace(
            artifacts_dir=str(directory), mask_bank_path=str(tmp_path / "bank.parquet")
        )
    )
    real = real_family()
    return cfg, eval_masks(directory, artifact_hash=cfg.artifact_hash, real=real), bank(
        directory, tmp_path / "bank.parquet", artifact_hash=cfg.artifact_hash
    )


@pytest.fixture
def text(box):
    """The rendered card for that box."""
    cfg, evaluated, banked = box
    return card.build(cfg, eval_masks=evaluated, bank=banked).render()


# --- Every artifact, by content -----------------------------------------------


def test_the_card_names_every_tile_list_by_its_md5(box, text):
    """The first acceptance criterion, over §3.1's four draws in their five files.

    By md5 rather than by name alone because that is what the run record quotes an artifact by:
    a benchmark scored against a stale list prints exactly as wide a table as a good one, and
    the card is where a reader goes to find out which list the number belongs to.
    """
    cfg, _, _ = box
    directory = Path(cfg.paths.artifacts_dir)
    for name in LIST_ROWS:
        assert name in text, name
        assert file_md5(directory / name) in text, name


def test_the_card_names_both_mask_artifacts_by_their_md5(box, text):
    _, evaluated, banked = box
    assert EVAL_MASKS_NAME in text and evaluated.md5 in text
    assert banked.path.name in text and banked.committed.md5 in text
    # The sidecar too: it is what git carries in the bank's place, so a reader looking for the
    # bank's md5 in the repository has to be told which file to look in.
    assert MASK_BANK_SIDECAR_NAME in text and banked.committed.draw_hash in text


def test_the_card_states_the_row_counts_it_read_off_the_committed_files(box, text):
    # Off the files rather than off the config: the committed file is what later phases re-read
    # (ADR-0005 §5), and a card that printed `tiles.head_train` would be describing the recipe
    # while claiming to describe the draw.
    for rows in LIST_ROWS.values():
        assert f"{rows:,}" in text


# --- One generator config hash, not two ---------------------------------------


def test_the_generator_config_hash_appears_once(box, text):
    """The second acceptance criterion, and it is a *count*.

    §3.4's determinism contract is one contract: the eval mask list and the train mask bank are
    generated from the same knobs and hash together. Printing the hash beside each artifact
    would render two identical strings a reader has to compare — which is exactly the reading
    the card exists to spare them.
    """
    cfg, _, _ = box
    assert text.count(cfg.artifact_hash) == 1


def test_a_card_over_two_artifacts_that_disagree_is_refused(box):
    """The same criterion from the other side: the card cannot *make* the contract one.

    A bank frozen under one generator config and an eval mask list frozen under another is the
    mid-change state §3.4's guard exists for. Rendering a card there would publish a single hash
    over two draws that do not share it — the one failure this section of the card can cause.
    """
    cfg, evaluated, banked = box
    moved = replace(
        banked,
        committed=replace(
            banked.committed, draw=replace(banked.committed.draw, artifact_hash="0" * 16)
        ),
    )
    with pytest.raises(ValueError, match="one generator config hash|artifact_hash"):
        card.build(cfg, eval_masks=evaluated, bank=moved)


# --- The two things a reader must be told rather than infer -------------------


def test_the_card_says_the_module_commits_tile_lists_and_masks_and_never_tiles(text):
    """Plan §3.4's sentence, on the page rather than in the reader's head.

    It is what keeps the artifact freely shippable under all three sets of terms below, so it is
    load-bearing for the terms section and not a nicety: nothing here is a redistribution
    question, because no DEM pixel is committed.
    """
    assert "never tiles" in text
    assert "index" in text and "redistribution" in text


def test_the_card_says_the_bank_is_not_in_the_repository_and_how_the_sidecar_stands_in(text):
    """The inference trap the spec resolved explicitly, closed on the page.

    A reader who assumes 104 MB of RLE is in git draws the wrong conclusion about the
    determinism contract — that the mask sequence is committed *as bytes*, when what is
    committed is its **hash**. Both halves are asserted: the absence, and the mechanism that
    makes the absence safe.
    """
    assert "not in this repository" in text
    assert MASK_BANK_SIDECAR_NAME in text
    assert "md5 has moved" in text
    assert "hash rather than as bytes" in text


# --- §3.3's excluded components -----------------------------------------------


def test_the_excluded_counts_are_printed_as_the_ladder_they_subtract_to(box, text):
    """The order is the argument: a reader should be able to do the subtraction down the page.

    The counts are exact by construction — `benchmark.voids.extract_cell` partitions the
    components it found rather than counting them twice — so a card that printed them in any
    order would be publishing an arithmetic identity a reader cannot see. This asserts both the
    identity and that the card's rows are in the order that shows it.
    """
    _, evaluated, _ = box
    real = evaluated.real
    assert (
        real.components_at_or_above_floor - real.excluded_above_top - real.failed_placement
        == real.scored_components
    )
    assert real.scored_components - real.excluded_in_gaps == len(real.rows)

    ladder = [
        f"{real.components_at_or_above_floor:,}",
        f"{real.excluded_above_top:,}",
        f"{real.failed_placement:,}",
        f"{real.scored_components:,}",
        f"{real.excluded_in_gaps:,}",
        f"{len(real.rows):,}",
    ]
    section = text.split("## §3.3's real-void family")[1].split("## Data terms")[0]
    assert list(ladder) == sorted(ladder, key=section.index), section


def test_the_card_records_the_real_familys_excluded_components_as_counts(box, text):
    """#153's fourth criterion, published beside the data rather than only in the code.

    The ceiling on large real voids is **this benchmark's tile, not nature**: giants over
    400 px exist in abundance and cannot fit a 224 px tile under §3.2's 40% area cap. A
    histogram that simply stopped would imply a scarcity that is not there, so the card states
    the cut-offs it was drawn under.
    """
    cfg, evaluated, _ = box
    real = evaluated.real
    assert f"{real.excluded_above_top:,}" in text
    assert f"{real.excluded_in_gaps:,}" in text
    assert f"{real.scored_components:,}" in text
    assert f"{cfg.masks.real_band_top_px} px" in text
    assert "not nature" in text or "not by scarcity" in text


def test_the_card_locates_the_histogram_cut_offs_and_not_only_the_counts(box, text):
    """The other half of #154's histogram line: the cut-offs, not just what they excluded.

    A count of components "in the gaps *between* the published size sub-rows" names the gaps
    without locating them, and a reader meeting the histogram beside this card cannot see where
    its two internal boundaries fall. So the three published sub-rows are on the page, and the
    two gaps they leave are stated as the ranges they are.

    The real family's top is 160 px where the synthetic large bucket stops at 158 (#121): the
    bands describe sizes real voids *have*, not draw targets the area cap constrains, so the
    card must print the real band rather than the bucket.
    """
    cfg, _, _ = box
    geometry = cfg.masks
    small, medium, large = geometry.small_px, geometry.medium_px, geometry.large_px

    for low, high in (small, medium, (large[0], geometry.real_band_top_px)):
        assert f"{low}\u2013{high}" in text, (low, high)
    # The two gaps a cleanly-placed component can still fall into.
    assert f"{small[1]}\u2013{medium[0]}" in text
    assert f"{medium[1]}\u2013{large[0]}" in text


# --- The three sets of data terms ---------------------------------------------


def test_the_card_states_all_three_sets_of_data_terms(text):
    """The third acceptance criterion — three sets, in the distributors' own words.

    In their words on purpose. The plan called SRTMGL1N "public domain" until #128; that is a
    legal conclusion the distributor does not make, and a card that repeated it would be the
    document a reader trusts making a claim the source does not.
    """
    assert "Copernicus GLO-30" in text
    assert "free basis for the general public" in text
    assert "not a standard open licence" in text.lower() or "not a standard open license" in text.lower()

    assert "SRTMGL1N" in text
    assert "openly shared, without restriction" in text
    assert "public domain" not in text.lower()

    assert "this repository's own" in text
    # Not legal advice, and said so — the precedent p101 set for this repo.
    assert "not legal advice" in text


# --- The card is a function of the draw, not of this box ----------------------


def test_two_boxes_freezing_one_config_write_the_same_card(tmp_path):
    """Why the card is not guarded by :func:`benchmark.artifacts.write_committed`.

    A card carrying an absolute path, a timestamp or a generation time would move on every box
    that touched it — and the file git carries would record whose laptop ran `freeze` last
    rather than what the freeze produced. So the box is deliberately absent from the render, and
    this is the assertion that keeps it absent.
    """
    rendered = []
    for n, seconds in ((1, 1_265.4), (2, 12.0)):
        directory = tmp_path / f"box{n}" / "artifacts"
        write_lists(directory)
        cfg = BenchmarkConfig().replace(
            paths=BenchmarkConfig().paths.replace(
                artifacts_dir=str(directory),
                mask_bank_path=str(tmp_path / f"box{n}" / "elsewhere" / "bank.parquet"),
            )
        )
        real = real_family()
        banked = replace(
            bank(directory, Path(cfg.paths.mask_bank_path), artifact_hash=cfg.artifact_hash),
            seconds=seconds,
            generated=n == 1,
        )
        rendered.append(
            card.build(
                cfg,
                eval_masks=eval_masks(directory, artifact_hash=cfg.artifact_hash, real=real),
                bank=banked,
            ).render()
        )
    assert rendered[0] == rendered[1]
    assert str(tmp_path) not in rendered[0]


# --- The stage --------------------------------------------------------------


def test_the_stage_writes_the_card_beside_the_artifacts_it_documents(box):
    cfg, evaluated, banked = box
    written = card.freeze(cfg, eval_masks=evaluated, bank=banked)
    assert written.path == Path(cfg.paths.artifacts_dir) / card.CARD_NAME
    assert written.path.read_text() == card.build(
        cfg, eval_masks=evaluated, bank=banked
    ).render()
    assert written.md5 == file_md5(written.path)
    assert written.rewritten


def test_re_freezing_an_unchanged_box_leaves_the_card_exactly_where_it_was(box):
    """Idempotence, and it is about the md5 in git rather than about the bytes on disk.

    Every artifact the card describes carries §3.4's own refusal, so by the time this stage runs
    `freeze` has already proven each of them is the one committed. What is left for the card is
    to not churn: a re-freeze that rewrote an identical document would move nothing and dirty a
    tree.
    """
    cfg, evaluated, banked = box
    first = card.freeze(cfg, eval_masks=evaluated, bank=banked)
    stamp = first.path.stat().st_mtime_ns
    again = card.freeze(cfg, eval_masks=evaluated, bank=banked)
    assert again.md5 == first.md5
    assert not again.rewritten
    assert again.path.stat().st_mtime_ns == stamp


def test_a_card_over_a_box_missing_a_tile_list_is_refused_by_name(box):
    # The card reads the committed files, so a partly-frozen box is a refusal rather than a
    # document with a row missing — which is the shape `benchmark.launch` refuses one level up.
    cfg, evaluated, banked = box
    (Path(cfg.paths.artifacts_dir) / VAL_TILES_NAME).unlink()
    with pytest.raises(FileNotFoundError, match=VAL_TILES_NAME):
        card.build(cfg, eval_masks=evaluated, bank=banked)


def test_the_stage_says_what_it_committed(box):
    cfg, evaluated, banked = box
    written = card.freeze(cfg, eval_masks=evaluated, bank=banked)
    said = "\n".join(card.format_written(written))
    assert card.CARD_NAME in said
    assert written.md5 in said


# --- Through the one command --------------------------------------------------


def test_freeze_is_the_last_verb_that_names_an_unbuilt_stage(monkeypatch, tmp_path, box):
    """`freeze` finishes what it starts now, which is the whole point of the last stage.

    Up to #154 every verb raised past the work it had, and `freeze` naming its missing stage was
    the defence against a command that wrote three of six artifacts and exited 0. The card was
    that missing stage, so what replaces the refusal is a zero — and this asserts it is a zero
    that arrives *after* the card is on disk rather than instead of it.
    """
    cfg, evaluated, banked = box
    directory = Path(cfg.paths.artifacts_dir)
    monkeypatch.setattr(launch.prereg, "write_bar", lambda path, plan: Path(path))
    monkeypatch.setattr(launch.tiles, "freeze", lambda config: {})
    monkeypatch.setattr(launch.extract, "write_spec", lambda config: directory / "spec.json")
    monkeypatch.setattr(launch.masks, "freeze", lambda config: evaluated)
    monkeypatch.setattr(launch.masks, "freeze_bank", lambda config: banked)
    (directory / "spec.json").write_text("{}")

    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        code = launch.main(
            [
                "freeze",
                "--artifacts-dir", str(directory),
                "--set", f'paths.mask_bank_path="{banked.path}"',
            ]
        )
    assert code == 0
    assert (directory / card.CARD_NAME).is_file()
    assert card.CARD_NAME in said.getvalue()
