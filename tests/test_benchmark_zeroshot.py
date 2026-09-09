"""A2's mechanism and R12's disposition (#167).

Two halves, and they are the two halves of the ticket. The **mechanism** is §4.3's sentence
made executable — patches touching the void are dropped, the pretrained decoder reconstructs
them, the non-void pixels are composited back — and it is tested against a real
:class:`~pretrain.model.MaskedAutoencoder`, because what can go wrong is the wiring between a
void and a token grid, and a stubbed model cannot fail that way.

The **disposition** is R12 made recordable: A2 carries no C-claim, and the arm is a negative
result unless it beats A1 in a majority of §5's four ruggedness bands on RMSE. R12 is
discursive — it holds no constant a bar file can carry (ADR-0013) — so what is mechanised here
is not the *rule* but the **outcome**: the comparison is computed and its verdict is written
into the run record, where the memo and the reviewer read it.

Nothing here asserts the magnitude of an A2 score, on `tests/test_benchmark_a1.py`'s rule: the
pre-registration exists to leave the ordering open, and a test that pinned it would close the
very question R12 is about.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark import arms, zeroshot

PATCH_PX = 16
GRID_PX = 14


# --- The void, as a keep-set over the token grid -------------------------------


def _void(rows: slice, cols: slice) -> np.ndarray:
    void = np.zeros((224, 224), dtype=bool)
    void[rows, cols] = True
    return void


def test_a_patch_the_void_touches_at_all_is_dropped():
    """§4.3's rule is *touching*, not covering — one void pixel drops the whole patch."""
    void = np.zeros((224, 224), dtype=bool)
    void[17, 33] = True  # patch (1, 2), one pixel of 256

    keep = zeroshot.keep_set(void, patch_px=PATCH_PX)
    assert keep.shape == (GRID_PX * GRID_PX,)
    assert not keep[1 * GRID_PX + 2]
    assert keep.sum() == GRID_PX * GRID_PX - 1


def test_the_keep_set_is_row_major_over_the_patch_grid():
    """The order `patchify` cuts in. A transpose here reconstructs the wrong patches."""
    void = _void(slice(0, 16), slice(32, 48))  # patch row 0, patch column 2

    keep = zeroshot.keep_set(void, patch_px=PATCH_PX)
    dropped = np.flatnonzero(~keep)
    assert dropped.tolist() == [2], "row-major: (0, 2) is index 2, not 28"


def test_an_empty_void_drops_nothing():
    keep = zeroshot.keep_set(np.zeros((224, 224), dtype=bool), patch_px=PATCH_PX)
    assert keep.all()


def test_a_void_covering_every_patch_leaves_no_visible_token():
    keep = zeroshot.keep_set(np.ones((224, 224), dtype=bool), patch_px=PATCH_PX)
    assert not keep.any()


def test_a_void_that_is_not_a_whole_number_of_patches_is_refused():
    with pytest.raises(ValueError, match="224.*not a whole number of 15 px patches"):
        zeroshot.keep_set(np.zeros((224, 224), dtype=bool), patch_px=15)


# --- The inpainter -------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    """A small MAE with a decoder, `norm_pix` off — the regime §4.2 says void filling needs.

    ViT-S at 224 px would be a minute of init per test process; this is the same class with the
    same forward pass at a width that runs in milliseconds. What is being tested is the wiring
    between a void, a token grid and a composite, and that is width-independent.
    """
    return zeroshot.Inpainter(
        model=_tiny_model(), norm_pix=False, var_floor=1e-4, source="a fixture"
    )


def _tiny_model():
    from pretrain.config import Architecture, Loss, Masking, RunConfig
    from pretrain.model import MaskedAutoencoder

    cfg = RunConfig().replace(
        architecture=Architecture(
            img_px=224,
            patch_px=16,
            encoder_dim=32,
            encoder_depth=1,
            encoder_heads=2,
            decoder_dim=16,
            decoder_depth=1,
            decoder_heads=2,
        ),
        loss=Loss(norm_pix_loss=False),
        masking=Masking(ratio=0.75),
    )
    return MaskedAutoencoder.from_config(cfg)


def _terrain(seed: int = 0) -> np.ndarray:
    """A plausible int16 tile in metres: a tilted plane plus relief, nowhere exactly 0."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:224, 0:224]
    z = 800.0 + 0.6 * y + 0.35 * x + 40.0 * np.sin(x / 21.0) + rng.normal(0, 3.0, (224, 224))
    return np.asarray(np.rint(z), dtype=np.int16)


def test_the_fill_writes_inside_the_void_and_nowhere_else(tiny):
    """§2.4's composite rule, and what makes `score`'s units check a free assertion."""
    tile, void = _terrain(), _void(slice(64, 112), slice(80, 128))

    filled = tiny.infill(tile, void)

    assert filled.shape == tile.shape
    outside = ~void
    assert np.array_equal(filled[outside], tile[outside].astype(filled.dtype))


def test_the_fill_comes_back_in_metres(tiny):
    """Not per-tile standardized units. The whole tile is ~800-1000 m, so this is loud.

    Asserted as an order of magnitude rather than a value: what a random-weight decoder predicts
    is not a number this test has any business pinning, but *which units it is in* is.
    """
    tile, void = _terrain(), _void(slice(96, 128), slice(96, 128))

    inside = tiny.infill(tile, void)[void]
    assert np.isfinite(inside).all()
    assert np.abs(inside).mean() > 100.0, "standardized units would sit within a few sigma of 0"


def test_only_the_patches_the_void_touches_are_replaced(tiny):
    """The decoder predicts every patch; §4.3 composites back everything outside the void.

    A void of one pixel therefore moves exactly one pixel, however far the untrained decoder's
    prediction is from the truth elsewhere.
    """
    tile = _terrain()
    void = np.zeros((224, 224), dtype=bool)
    void[100, 100] = True

    filled = tiny.infill(tile, void)
    moved = np.flatnonzero(filled.ravel() != tile.astype(filled.dtype).ravel())
    assert moved.tolist() == [100 * 224 + 100]


def test_the_statistics_are_taken_over_the_non_void_pixels(tiny):
    """§2.1's inversion: a voided tile's true statistics are unknowable at inference.

    The check is that the void's *contents* cannot reach the output. Two tiles identical outside
    the void and wildly different inside it must fill identically — if the normalization read the
    voided pixels, they would not.
    """
    tile, void = _terrain(), _void(slice(64, 96), slice(64, 96))
    other = tile.copy()
    other[void] = np.int16(-30000)

    assert np.array_equal(tiny.infill(tile, void), tiny.infill(other, void))


def test_a_void_touching_every_patch_is_refused(tiny):
    """No visible token is no reconstruction — see `pretrain.model.keep_masking`.

    One void pixel per patch, not a full tile: 196 scattered pixels leave the tile 99.6% valid,
    so the per-tile statistics are fine and what fails is exactly the thing under test. A void
    covering every *pixel* would trip ADR-0012's empty-valid-mask refusal first, which is a
    different refusal and the subject of the test below.
    """
    void = np.zeros((224, 224), dtype=bool)
    void[8::16, 8::16] = True  # the centre pixel of all 196 patches

    assert not zeroshot.keep_set(void, patch_px=PATCH_PX).any()
    with pytest.raises(ValueError, match="every token is hidden"):
        tiny.infill(_terrain(), void)


def test_a_tile_that_is_entirely_ocean_is_refused(tiny):
    """`valid = land ∧ ¬void`, and an empty valid mask has no per-tile statistic (ADR-0012)."""
    with pytest.raises(ValueError, match="no valid pixels"):
        tiny.infill(np.zeros((224, 224), dtype=np.int16), _void(slice(0, 16), slice(0, 16)))


def test_the_inpainter_takes_raw_int16_metres_not_a_normalized_tile(tiny):
    with pytest.raises(TypeError, match="raw int16 metres"):
        tiny.infill(_terrain().astype(np.float32), _void(slice(0, 16), slice(0, 16)))


# --- norm_pix: the inversion that is not available -----------------------------


def test_a_norm_pix_checkpoint_is_refused_rather_than_inverted_with_the_truth():
    """The finding this ticket surfaced, made structural.

    With `norm_pix_loss` on, the decoder predicts each patch **standardized by that patch's own
    mean and variance** — and inside a void those two numbers are exactly what is unknown.
    Recovering them from the truth is not an inversion, it is the answer: the fill would carry
    the voided terrain's own level and spread, and A2's row would beat every arm in the table for
    a reason no column names.

    §4.2 states the underlying fact — "void filling needs absolute per-patch scale; norm-pix
    targets discard it" — and turns `norm_pix` off for A3's fine-tune. §4.3 is silent, and R5 and
    R4 were both pretrained with it **on**. So this refuses, naming what would have to be
    registered before an A2 number exists.
    """
    inpainter = zeroshot.Inpainter(
        model=_tiny_model(), norm_pix=True, var_floor=1e-4, source="a fixture"
    )
    with pytest.raises(zeroshot.ScaleUnavailable) as exc:
        inpainter.infill(_terrain(), _void(slice(64, 96), slice(64, 96)))

    message = str(exc.value)
    assert "norm_pix" in message
    assert "§4.3" in message, "the refusal names the section that would have to register it"
    assert "leak" in message.lower()


def test_the_norm_pix_refusal_is_raised_before_any_instance_is_scored():
    """A refusal that costs an eval pass is one nobody consults until the bill arrives."""
    inpainter = zeroshot.Inpainter(
        model=_tiny_model(), norm_pix=True, var_floor=1e-4, source="a fixture"
    )
    with pytest.raises(zeroshot.ScaleUnavailable):
        inpainter.check_ready()


def test_a_norm_pix_off_inpainter_is_ready(tiny):
    assert tiny.check_ready() is None


# --- R12: A2 carries no C-claim ------------------------------------------------


def test_a2_is_named_by_no_claim():
    """The register's own statement, asserted against the claims table rather than a constant."""
    for arm in ("A2-R5", "A2-R4"):
        assert zeroshot.c_claim(arms.get(arm)) is None


def test_the_arms_that_do_carry_a_claim_still_do():
    """The other half: a blanket `None` would satisfy the test above and say nothing."""
    assert zeroshot.c_claim(arms.get("A4-R5")) == "C1"


def test_the_disposition_marks_a2_as_carrying_no_c_claim():
    record = zeroshot.disposition(arms.get("A2-R5"), a1_rows=[], a2_rows=[])
    assert record["rule"] == "R12"
    assert record["c_claim"] is None
    assert record["carries_c_claim"] is False
    assert "insulated from C1 in both directions" in record["insulation"]


def test_the_disposition_never_calls_the_a2_to_a3_difference_task_adaptation():
    """R12's second part, asserted on the words the record actually carries.

    The clause the plan struck — "quantifies precisely what the task adaptation bought" — is not
    true of any outcome, because A3 changes two things at once. The record says so in the
    register's own words, and this test is what keeps the struck clause struck.
    """
    record = zeroshot.disposition(arms.get("A2-R5"), a1_rows=[], a2_rows=[])
    joint = record["a2_to_a3"]

    assert joint.startswith("a joint quantity")
    assert "never attributed to task adaptation alone" in joint
    # The struck clause, and the two ways it tends to come back. "task adaptation" itself is not
    # forbidden — the disclaimer has to be able to name what it disclaims — but attributing the
    # difference *to* it is, in any of its wordings.
    for forbidden in (
        "quantifies precisely what the task adaptation bought",
        "what task adaptation bought",
        "attributable to task adaptation",
        "due to task adaptation",
    ):
        assert forbidden not in joint


def test_a_non_a2_arm_has_no_r12_disposition():
    """R12 is A2's rule. A disposition on every row would be a register nobody reads."""
    assert zeroshot.disposition(arms.get("A1"), a1_rows=[], a2_rows=[]) is None


# --- R12: the majority-of-bands comparison -------------------------------------


def _rows(arm: str, per_band: dict[str, list[float]]) -> list[dict]:
    """Scored rows as the parquet holds them. Tile ids are unique *across* bands, because a
    tile's band is a function of its own `slope_mean` — a fixture that put one tile in four
    bands would make the instance key ambiguous in a way no real results file is."""
    return [
        {"arm": arm, "band": band, "tile_id": f"{band}-t{i}", "family": "blob",
         "bucket": "small", "instance": 0, "rmse": value}
        for band, values in per_band.items()
        for i, value in enumerate(values)
    ]


BANDS = ("R1", "R2", "R3", "R4")


def _both(a1: dict[str, list[float]], a2: dict[str, list[float]]):
    return _rows("A1", a1), _rows("A2-R5", a2)


def test_a2_beating_a1_in_three_of_four_bands_is_a_majority():
    a1, a2 = _both(
        {b: [10.0, 12.0] for b in BANDS},
        {"R1": [8.0, 9.0], "R2": [7.0, 8.0], "R3": [6.0, 7.0], "R4": [30.0, 31.0]},
    )
    outcome = zeroshot.band_comparison(a1_rows=a1, a2_rows=a2)

    assert outcome["bands_won"] == 3
    assert outcome["bands_compared"] == 4
    assert outcome["beats_a1_in_a_majority"] is True
    assert outcome["negative_result"] is False


def test_two_of_four_is_not_a_majority_and_is_a_negative_result():
    """The register's own repair (ADR-0013): the trigger is "does **not** beat", so a tie loses."""
    a1, a2 = _both(
        {b: [10.0] for b in BANDS},
        {"R1": [9.0], "R2": [9.0], "R3": [11.0], "R4": [11.0]},
    )
    outcome = zeroshot.band_comparison(a1_rows=a1, a2_rows=a2)

    assert outcome["bands_won"] == 2
    assert outcome["beats_a1_in_a_majority"] is False
    assert outcome["negative_result"] is True


def test_an_exactly_equal_band_is_not_won():
    a1, a2 = _both({b: [10.0] for b in BANDS}, {b: [10.0] for b in BANDS})
    outcome = zeroshot.band_comparison(a1_rows=a1, a2_rows=a2)
    assert outcome["bands_won"] == 0
    assert outcome["negative_result"] is True


def test_the_comparison_is_on_rmse_and_says_so():
    a1, a2 = _both({b: [10.0] for b in BANDS}, {b: [1.0] for b in BANDS})
    outcome = zeroshot.band_comparison(a1_rows=a1, a2_rows=a2)
    assert outcome["metric"] == "rmse"
    assert outcome["lower_is_better"] is True


def test_the_comparison_is_over_the_instances_both_arms_scored():
    """A2 scored on a subset of A1's rows compares on the intersection, and reports its size.

    Two arms scored over different mask lists would otherwise be compared band-mean to
    band-mean, which is a comparison of two different benchmarks that returns a number.
    """
    a1 = _rows("A1", {b: [10.0, 10.0, 1000.0] for b in BANDS})
    a2 = _rows("A2-R5", {b: [9.0, 9.0] for b in BANDS})
    outcome = zeroshot.band_comparison(a1_rows=a1, a2_rows=a2)

    assert outcome["instances_compared"] == 8, "two per band, not three"
    assert outcome["bands_won"] == 4


def test_a_band_neither_arm_scored_is_not_counted_as_won_or_lost():
    a1, a2 = _both({"R1": [10.0], "R2": [10.0]}, {"R1": [1.0], "R2": [1.0]})
    outcome = zeroshot.band_comparison(a1_rows=a1, a2_rows=a2)

    assert outcome["bands_compared"] == 2
    assert outcome["bands_won"] == 2
    # Two of the four bands are unscored, so "a majority of the four" is not yet answerable.
    assert outcome["beats_a1_in_a_majority"] is None
    assert outcome["negative_result"] is None
    assert "4" in outcome["why"] and "2" in outcome["why"]


def test_no_a1_rows_at_all_is_pending_rather_than_a_negative_result():
    """The distinction the memo turns on: *not computed yet* is not *did not beat A1*."""
    outcome = zeroshot.band_comparison(a1_rows=[], a2_rows=_rows("A2-R5", {"R1": [1.0]}))

    assert outcome["bands_compared"] == 0
    assert outcome["negative_result"] is None
    assert "A1" in outcome["why"]


def test_the_disposition_carries_the_comparison_and_the_reporting_verdict():
    a1, a2 = _both({b: [10.0] for b in BANDS}, {b: [1.0] for b in BANDS})
    record = zeroshot.disposition(arms.get("A2-R5"), a1_rows=a1, a2_rows=a2)

    assert record["comparison"]["bands_won"] == 4
    assert record["reported_as"] == zeroshot.REPORTED_ORDERING


def test_a_losing_a2_is_recorded_as_a_negative_result_in_the_registers_words():
    a1, a2 = _both({b: [1.0] for b in BANDS}, {b: [10.0] for b in BANDS})
    record = zeroshot.disposition(arms.get("A2-R5"), a1_rows=a1, a2_rows=a2)

    assert record["reported_as"] == zeroshot.REPORTED_NEGATIVE
    assert "does not transfer" in record["negative_result_statement"]
    assert "baseline half of a delta" in record["prominence"]


def test_the_comparison_reads_its_rows_more_than_once_and_says_the_same_thing():
    """Handed generators, not lists — the shape a caller streaming out of parquet would pass.

    The comparison walks each side three times (the intersection, then each arm's band means over
    it), so an implementation that took the arguments as it found them would answer over an
    exhausted iterator on the second pass: zero shared instances, zero bands compared, and a
    "pending" verdict on a cell that had every row it needed.
    """
    a1, a2 = _both({b: [10.0] for b in BANDS}, {b: [1.0] for b in BANDS})
    outcome = zeroshot.band_comparison(a1_rows=iter(a1), a2_rows=iter(a2))

    assert outcome["bands_compared"] == 4
    assert outcome["instances_compared"] == 4
    assert outcome["bands_won"] == 4


# --- R12's second clause, over the source itself -------------------------------


def test_neither_the_arm_registry_nor_this_module_attributes_the_difference_to_task_adaptation():
    """#167's fourth criterion names *the arm's code* as well as its record.

    Asserted over the source because that is what the criterion is about, and this repo already
    tests documents this way — `benchmark.prereg` checks the plan's own §6 text for the phrases
    its values were registered in. The clause §4.3 struck reads well and is easy to reintroduce
    in a docstring; what makes that a defect rather than a wording preference is that the memo is
    written from these strings.
    """
    from pathlib import Path

    sources = [
        Path(zeroshot.__file__).read_text(),
        arms.get("A2-R5").what,
        arms.get("A2-R4").what,
        arms.get("A3-R5").what,
        arms.get("A3-R4").what,
    ]
    for struck in (
        "quantifies precisely what the task adaptation bought",
        "what task adaptation bought",
        "what the task adaptation bought",
        "attributable to task adaptation",
        "due to task adaptation",
        "measures the task adaptation",
    ):
        for source in sources:
            assert struck not in source, f"{struck!r} is the clause R12 struck"
