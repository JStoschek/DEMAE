"""§2.3's ViT-S appendix cell — the one-seed rule, and what the record says about it (#164).

A4-S is the plan's one **single observation**: one seed at 20k, reported without a ±. §2.3's
rewrite demoted ViT-S explicitly — ViT-B is the primary capacity on both sides and R1's ViT-S
rides along as a cheap extra row — so the absence of the interval is the point of the cell
rather than a gap in it. What that needs from the code is two things a reader can check: a
second seed has to be **refused** where it would be produced, and the record has to **say**
that this row is one observation rather than leaving a reader to count.

The arm and its encoder were registered long before this ticket (§2.3, and #147 when A6-S was
dropped beside it). What #164 adds is the checkpoint the encoder was missing, the refusal, and
the block below.
"""

from __future__ import annotations

import pytest

from benchmark import arms, results
from benchmark.config import BenchmarkConfig

ARM = "A4-S"


def _resolution_for(arm_name: str):
    """A `Resolution` for one arm, against this repo's committed artifacts.

    Built through `artifacts.resolve` rather than by hand, on `test_benchmark_a9`'s rule: the
    block under test is then read off the same object `launch` writes its run record from.
    """
    from benchmark import artifacts

    cfg = BenchmarkConfig(arm=arm_name)
    return artifacts.resolve(cfg, arms.get(arm_name))


# --- The record's own sentence ------------------------------------------------


def test_the_run_record_says_this_row_is_a_single_observation():
    """The acceptance criterion, as the file a reviewer actually opens states it.

    Counting rows in the parquet would answer "how many seeds ran", which is a different
    question from "how many was this cell meant to be" — a cell that was *supposed* to carry
    three and lost two to a crash looks identical from the rows alone.
    """
    block = results.observations_block(_resolution_for(ARM))

    assert block["seeds"] == 1
    assert block["reported_with_interval"] is False
    assert "±" in block["note"]


def test_an_arm_that_is_not_a_single_observation_records_no_block():
    """`sampler_block`'s rule: a block on every row is one a reviewer stops reading."""
    for name in ("A4-R5", "A5-BICUBIC", "A9", "A1"):
        assert results.observations_block(_resolution_for(name)) is None


def test_the_block_reaches_the_record_under_its_own_key():
    """Absent rather than null for the arms that are not one, as `register` and `sampler` are."""
    single = results.run_record(
        _resolution_for(ARM), rows=0, seconds=0.0, parameters={}, metrics=[]
    )
    ordinary = results.run_record(
        _resolution_for("A4-R5"), rows=0, seconds=0.0, parameters={}, metrics=[]
    )
    assert single["observations"]["seeds"] == 1
    assert "observations" not in ordinary


# --- The cell is at 20k, and the sweep is not its ----------------------------


def test_the_appendix_cell_is_refused_at_the_label_efficiency_sizes():
    """§7: *"R4 and ViT-S are both out of the sweep (§2.3)"*, and §4.5 orders it the same way.

    The sweep's own budget is three cells at two seeds; a fourth arriving at 1k or 5k would be a
    row §7 did not price and §2.3 did not ask for. The appendix cell is one observation at 20k,
    which is two facts about the same cell — so the check that enforces the seed enforces this.
    """
    from benchmark import launch

    for size in ("1k", "5k"):
        with pytest.raises(ValueError, match="out of the label-efficiency sweep"):
            launch._checked_cell(BenchmarkConfig(arm=ARM, size=size), arms.get(ARM))


def test_the_sweep_is_ordinary_for_the_arms_it_prices():
    """A4-R5 at 5k is a cell the sweep has; the refusal is keyed on the arm, not on the size."""
    from benchmark import launch

    launch._checked_cell(BenchmarkConfig(arm="A4-R5", size="5k", seed=1), arms.get("A4-R5"))


# --- The refusal is a guardrail, and the escape hatch leaves a trace ----------


def test_widening_the_count_opens_the_second_seed_and_moves_the_config_hash():
    """The refusal stops the accidental second observation, not a determined one.

    `appendix_seeds` is deliberately outside `TUNED_FIELDS` — `Head.tuned` excludes the three
    seed counts so a smoke run at `head.seeds=1` is not read as a thumb on the scale — so
    `tune.assert_frozen` does not refuse a moved one and `--set head.appendix_seeds=3` really
    does open seed 1. What makes that deliberate act honest rather than silent is the second
    assertion: the count reaches `config_hash`, which is a column on **every** parquet row, so
    a row produced under a widened count cannot be read as one produced under §2.3's.
    """
    from benchmark import launch
    from benchmark.config import TUNED_FIELDS

    assert "appendix_seeds" not in TUNED_FIELDS

    base = BenchmarkConfig(arm=ARM, seed=1)
    widened = base.replace(head=base.head.replace(appendix_seeds=3))
    with pytest.raises(ValueError, match="single observation"):
        launch._checked_cell(base, arms.get(ARM))
    launch._checked_cell(widened, arms.get(ARM))  # allowed, and recorded:
    assert widened.config_hash != base.config_hash


# --- The head is the ViT-B counterpart's, at the one width that differs --------


def test_the_appendix_cell_trains_the_same_head_as_its_vit_b_counterpart():
    """§4.1's head is shared, and the encoder's width is its only delta — which is C1's sentence.

    So A4-S and A4-R5 must differ in `width` and in nothing else that reaches the head, or the
    capacity aside would be measuring the head as well as the backbone.
    """
    appendix, primary = arms.get(ARM), arms.get("A4-R5")
    recipe = {
        f
        for f in ("trains", "input_convention", "normalization")
        if getattr(appendix, f) != getattr(primary, f)
    }
    assert recipe == set(), recipe
    assert appendix.width == 384 and primary.width == 768


def test_the_appendix_cells_head_is_smaller_only_by_its_first_projection():
    """The head's parameter count moves with the width and with nothing else (§4.1)."""
    from benchmark import head

    counts = arms.parameter_counts(arms.get(ARM))
    assert counts["trainable"] == head.parameter_count(384)
    assert counts["total"] == arms.ENCODERS["demae-vits"].params + counts["trainable"]


# --- A6-S is still gone -------------------------------------------------------


def test_the_vit_s_imagenet_arm_is_still_refused():
    """#147 dropped A6-S on provenance, and #164 must not quietly reinstate it.

    The obvious way to reinstate it is exactly what this ticket does for A4-S — mount a
    checkpoint and pin it — so the refusal is asserted here rather than left to #147's own test.
    """
    assert "A6-S" not in arms.ARMS
    assert "imagenet-mae-vits" not in arms.ENCODERS
    with pytest.raises(ValueError, match="unknown arm 'A6-S'"):
        arms.get("A6-S")
