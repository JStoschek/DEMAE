"""§4.4's from-scratch UNet — A7 (#162).

Two of these tests are the ticket's second acceptance criterion in executable form. "Loss,
schedule length and seeds match the frozen arms', **asserted rather than configured by hand**"
is not a claim about values that happen to be equal today: it is a claim that A7 *cannot* own
them. So the assertions are structural — :class:`benchmark.config.Unet` is refused if it grows
one of §4.1's recipe fields, and :func:`benchmark.train.train_unet` is refused a recipe that is
not the one the frozen arms read.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from benchmark import arms, config, unet


# --- The derived count, against the built one ---------------------------------


def test_parameter_count_matches_the_built_module():
    """§4.4's ~17M, derived arithmetically and then counted — head.build's rule (#159)."""
    built = unet.build(config.Unet())
    assert built.parameters == unet.parameter_count(config.Unet())


def test_the_recorded_count_is_the_plans_17m():
    counted = unet.parameter_count(config.Unet())
    assert counted == unet.PARAMETERS[config.Unet().base_width]
    assert 16_000_000 < counted < 18_500_000, counted


def test_a7_trains_an_order_of_magnitude_more_than_a_frozen_arm():
    """The sentence §4.4 replaced its old capacity anchor with, as a number."""
    from benchmark import head

    assert unet.parameter_count(config.Unet()) > 14 * head.parameter_count(768)


# --- §2.4's residual convention ------------------------------------------------


def test_the_residual_output_is_zeroed_so_a7_starts_as_a1():
    """A head that learns nothing scores exactly A1 (§2.4) — exact, not approximate."""
    torch = pytest.importorskip("torch")
    built = unet.build(config.Unet())
    image = torch.randn(2, 1, 224, 224)
    void = torch.zeros(2, 1, 224, 224)
    void[:, :, 40:80, 40:80] = 1.0
    assert torch.count_nonzero(built.module(image, void)) == 0


def test_the_module_takes_two_channels_and_returns_one():
    torch = pytest.importorskip("torch")
    built = unet.build(config.Unet())
    out = built.module(torch.randn(1, 1, 224, 224), torch.zeros(1, 1, 224, 224))
    assert out.shape == (1, 1, 224, 224)


# --- The input conditioning ----------------------------------------------------


def test_conditioning_standardizes_over_valid_pixels_only():
    """A7 declares no encoder diet; this is the UNet's own conditioning. See the module."""
    raw = np.zeros((1, 8, 8), dtype=np.float32)
    raw[0, 0:2] = 100.0
    raw[0, 2:4] = 200.0
    raw[0, 4:] = 9999.0  # inside the void, so it must not reach the statistic
    valid = np.ones((1, 8, 8), dtype=bool)
    valid[0, 4:] = False
    image, std = unet.condition(raw, np.rint(raw).astype(np.int16), valid)
    # mean 150, std 50 over the valid half — and nothing from the 9999s.
    assert std[0] == pytest.approx(50.0, rel=1e-5)
    assert image[0][valid[0]].mean() == pytest.approx(0.0, abs=1e-5)
    assert image[0][valid[0]].std() == pytest.approx(1.0, rel=1e-5)


def test_conditioning_is_finite_on_a_flat_tile():
    """A perfectly flat tile has zero std; the floor is what keeps the division finite."""
    raw = np.full((1, 8, 8), 42.0, dtype=np.float32)
    valid = np.ones((1, 8, 8), dtype=bool)
    image, std = unet.condition(raw, np.rint(raw).astype(np.int16), valid)
    assert np.isfinite(image).all()
    assert std[0] > 0


# --- The structural half of "asserted rather than configured by hand" ----------


def test_the_unet_recipe_cannot_own_a_frozen_arm_knob():
    """#162's second criterion, made impossible to violate rather than checked for.

    If a later ticket adds ``lr`` or ``epochs`` to :class:`~benchmark.config.Unet`, A7 stops
    sharing §4.1's schedule and nothing about the run would look wrong. So the config refuses
    to be *defined* that way, and this is the test that says so.
    """
    fields = {f.name for f in dataclasses.fields(config.Unet)}
    assert not (fields & set(config.SHARED_WITH_HEAD)), (
        f"§4.4 says A7 trains under the frozen arms' loss, schedule and seeds; "
        f"{sorted(fields & set(config.SHARED_WITH_HEAD))} would let it drift"
    )


def test_a_unet_recipe_that_grew_a_head_knob_is_refused():
    """The guard itself fires — the test above passes vacuously if it does not."""
    with pytest.raises(ValueError, match="§4.1"):
        config.assert_shares_head_recipe(("lr", "depth"))


def test_a7_is_refused_a_moved_head_knob_exactly_as_a_frozen_arm_is():
    """The enforcing half of #162's second criterion, and the one that has teeth.

    §4.1's recipe is tuned once on A6 and frozen for every arm; `benchmark.tune.assert_frozen`
    is what refuses a run that moved it. A7 trains under that recipe, so it has to be inside
    that gate — an arm exempted from it could be trained at a learning rate the frozen arms
    never saw, converge normally, and print a C3 bar that is not comparable to the thing it is
    a bar for.

    Asserted against A6 in the same test so the two are the *same* refusal rather than a second
    one written for A7: if the gate ever stops covering one of them, this fails.
    """
    from benchmark import launch, tune

    parser = launch.build_parser()
    for arm in ("A6", "A7"):
        with pytest.raises(tune.TuningError):
            launch.resolve(
                parser.parse_args(["run", "--arm", arm, "--set", "head.lr=0.0003"])
            )
    # And the unmoved recipe resolves for both — the gate refuses drift, not A7.
    for arm in ("A6", "A7"):
        launch.resolve(parser.parse_args(["run", "--arm", arm]))


# --- The registry's pair -------------------------------------------------------


def test_a7_records_both_parameter_counts():
    """#162's fourth criterion. They are equal for A7 — that is the point, see the docstring."""
    counts = arms.parameter_counts(arms.get("A7"))
    assert counts["total"] == counts["trainable"] == unet.parameter_count(config.Unet())


def test_a7s_pair_differs_from_a_frozen_arms_pair():
    """What "recorded and differ" is about: A7 against A4, not total against trainable."""
    a7 = arms.parameter_counts(arms.get("A7"))
    a4 = arms.parameter_counts(arms.get("A4-R5"))
    assert a7["total"] < a4["total"]
    assert a7["trainable"] > 14 * a4["trainable"]


# --- A7 through the shipping command ------------------------------------------
#
# The box these run on is #159's, imported rather than rebuilt: it is the *same* frozen corpus,
# tile lists, eval mask list and train mask bank the frozen arms are tested against, and that
# identity is the point — A7's whole job is to be a bar those arms are read beside, so a test
# box of its own would be the one place the comparison could quietly stop being like-for-like.
# (When #159 lands, these three belong in `conftest.py`; they are here because they are shared
# by two test modules now rather than one.)

from tests.test_benchmark_head import (  # noqa: E402
    box,  # noqa: F401
    flags,  # noqa: F401
    frozen,  # noqa: F401
    splits_of,
    table_of,
)

import pyarrow.parquet as pq  # noqa: E402

from benchmark import launch, train  # noqa: E402


@pytest.fixture
def a7(box, flags, frozen, capsys):
    """One real A7 cell through ``main(argv)``: train from scratch, then score.

    No ``stub`` fixture, and that absence is worth a sentence: every other training arm needs a
    stand-in encoder because a real ViT-B will not run in a unit test. A7 has no encoder, so
    what runs here is the shipping trunk itself at fixture scale — the only arm whose seam-1
    test exercises the real trainable part.
    """
    assert launch.main(["run", "--arm", "A7", *flags]) == 0
    return box, capsys.readouterr().out


def test_the_shipping_command_trains_a_unet_and_lands_a7_rows(a7):
    """#162's first criterion, at fixture scale — the shape the 20k x 3-seed run has."""
    box, _ = a7
    table = table_of(box)
    assert table.num_rows == len(splits_of(box)["test"]) * 2 * 3
    assert set(table.column("arm").to_pylist()) == {"A7"}


def test_the_run_record_carries_both_parameter_counts_and_they_are_the_unets(a7):
    """#162's fourth criterion, through the shipping path rather than off the registry."""
    box, _ = a7
    record = json.loads(
        (box / "results-runs.jsonl").read_text().strip().splitlines()[-1]
    )
    counts = record["parameters"]
    assert counts["total"] == counts["trainable"] == unet.parameter_count()
    # And what the pass itself recorded agrees with what the registry derived.
    assert record["training"]["kind"] == "unet"
    assert record["training"]["parameters"] == unet.parameter_count()
    assert record["training"]["width"] is None


def test_a7_trains_under_the_frozen_arms_schedule(box, flags, frozen):
    """#162's second criterion as a value check, on top of the structural one above.

    The schedule A7 builds and the schedule a frozen arm builds are the same object for the same
    tile count — asserted by constructing both from the recipe each arm actually reads, rather
    than by comparing two numbers a reader has to trust came from there.
    """
    cfg = launch.resolve(
        launch.build_parser().parse_args(["run", "--arm", "A7", *flags])
    ).config
    frozen_cfg = launch.resolve(
        launch.build_parser().parse_args(["run", "--arm", "A6", *flags])
    ).config
    steps = 7
    assert train.Cosine.of(
        cfg.head, steps_per_epoch=steps, epochs=cfg.head.epochs
    ) == train.Cosine.of(
        frozen_cfg.head, steps_per_epoch=steps, epochs=frozen_cfg.head.epochs
    )
    # The loss is literally the frozen arms' function, not a second implementation of it.
    from benchmark import head as head_module

    assert train.head.loss is head_module.loss


def test_a7_reads_the_committed_bank_rather_than_drawing(box, flags, frozen):
    """#162's third criterion. Same assertion #159 makes of the head, for the same reason.

    ``frozen`` is needed here for the same reason A6's tests need it: since #162 put A7 inside
    `tune.assert_frozen`'s gate, a test-scale `head.epochs` is a drift for A7 exactly as it is
    for A4 — which is the property `test_a7_is_refused_a_moved_head_knob...` asserts on purpose.
    """
    cfg = launch.resolve(
        launch.build_parser().parse_args(["run", "--arm", "A7", *flags])
    ).config
    train_ids = splits_of(box)["train"]
    banked = train.read_bank(cfg, train_ids)
    committed = pq.read_table(box / "train_mask_bank.parquet").to_pylist()
    for row in committed:
        if row["tile_id"] in set(train_ids):
            assert banked.rle(row["tile_id"], row["epoch"]) == row["mask_rle"]


def test_a7_requires_the_train_mask_bank_at_resolution():
    """The bank is a declared input of this arm, not merely something the loop happens to open."""
    from benchmark import artifacts

    assert artifacts._required_for("mask_bank", arms.get("A7"))
    assert not artifacts._required_for("mask_bank", arms.get("A1"))
    # And A7 needs no extractor spec, because it runs no encoder.
    assert not artifacts._required_for("extractor_spec", arms.get("A7"))


def test_tuning_is_refused_for_a7(box, flags, frozen):
    """§4.1's recipe is tuned on A6 and frozen for every arm — A7 is one of the arms.

    Refused in :func:`benchmark.tune.assert_tuning_arm`, upstream of everything: the rule is
    "tune on the neutral encoder", so it is A6-or-nothing rather than a list of arms each
    declining in its own words. A7 needs no branch of its own to be covered by it, and this test
    is here to say that it is covered rather than to test a second refusal.
    """
    from benchmark import tune

    with pytest.raises(tune.TuningError, match="tunes on A6"):
        launch.main(["run", "--arm", "A7", "--tune", *flags])


def test_a_head_arm_is_refused_by_the_from_scratch_pass(box, flags, frozen):
    """The two passes are siblings, and neither accepts the other's arm."""
    cfg = launch.resolve(
        launch.build_parser().parse_args(["run", "--arm", "A6", *flags])
    ).config
    with pytest.raises(train.TrainError, match="trains 'head'"):
        train.train_unet(cfg, arms.get("A6"))


# --- The micro-batch split is exact, not approximate ---------------------------


def test_splitting_a_batch_leaves_the_gradient_where_it_was():
    """`runtime.micro_batch` must not be a smaller batch wearing the recipe's name (#162).

    §4.4 trains A7 under §4.1's declared batch of 128, and a 32 GB card cannot forward that many
    224 px tiles through a trunk that is 48 channels wide from its first layer — a measured
    0.24 GiB/sample, so ~30.7 GiB against 31.36 available. The split is how the recipe survives
    the card, so what has to be true is that it changes *nothing about the number*.

    Asserted against the unsplit gradient itself rather than against convergence: two runs that
    both descend prove nothing, and §4.1's loss is a mean over **void pixels**, which is exactly
    the quantity a naive split gets wrong when the pieces carry unequal void.
    """
    torch = pytest.importorskip("torch")
    from benchmark import head as head_module
    from benchmark import train

    rng = np.random.default_rng(0)
    n = 6
    # Deliberately unequal void per sample, which is what a naive split mishandles.
    void = np.zeros((n, 32, 32), dtype=bool)
    for i in range(n):
        void[i, 2 : 4 + 3 * i, 2 : 4 + 3 * i] = True
    batch = train.Composed(
        raw_i16=rng.integers(-100, 100, (n, 32, 32)).astype(np.int16),
        a1_fill=rng.normal(0, 10, (n, 32, 32)).astype(np.float32),
        a1_i16=rng.integers(-100, 100, (n, 32, 32)).astype(np.int16),
        valid=~void,
        void=void,
    )

    def gradient(size: int) -> list:
        built = unet.build(config.Unet(depth=1, base_width=48), seed=0)
        module = built.module
        module.zero_grad(set_to_none=True)
        pixels = float(batch.void.sum())
        for piece in train.micro_batches(batch, size):
            v = torch.from_numpy(piece.void.astype(np.float32)).unsqueeze(1)
            image = torch.from_numpy(piece.a1_fill).unsqueeze(1)
            truth = torch.from_numpy(piece.raw_i16.astype(np.float32)).unsqueeze(1)
            a1 = torch.from_numpy(piece.a1_fill).unsqueeze(1)
            prediction = a1 + module(image, v) * v
            terms = head_module.loss(prediction, truth, v, slope_weight=0.1)
            (terms.total * (float(piece.void.sum()) / pixels)).backward()
        return [p.grad.clone() for p in module.parameters() if p.grad is not None]

    whole = gradient(0)
    split = gradient(2)
    assert len(whole) == len(split) and whole
    for a, b in zip(whole, split):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-7)


def test_micro_batches_partition_the_batch_in_order():
    from benchmark import train

    n = 7
    batch = train.Composed(
        raw_i16=np.arange(n * 4, dtype=np.int16).reshape(n, 2, 2),
        a1_fill=np.zeros((n, 2, 2), np.float32),
        a1_i16=np.zeros((n, 2, 2), np.int16),
        valid=np.ones((n, 2, 2), bool),
        void=np.ones((n, 2, 2), bool),
    )
    pieces = train.micro_batches(batch, 3)
    assert [p.raw_i16.shape[0] for p in pieces] == [3, 3, 1]
    assert np.array_equal(np.concatenate([p.raw_i16 for p in pieces]), batch.raw_i16)
    # 0 and "bigger than the batch" both mean the batch itself.
    assert train.micro_batches(batch, 0)[0] is batch
    assert train.micro_batches(batch, 99)[0] is batch


def test_the_micro_batch_is_environment_and_never_moves_the_hash():
    """It is a fact about the card, so two boxes must produce comparable rows (#162)."""
    base = config.BenchmarkConfig(arm="A7")
    split = config.BenchmarkConfig(
        arm="A7", runtime=base.runtime.replace(micro_batch=64)
    )
    assert split.config_hash == base.config_hash
