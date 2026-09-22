"""#182 — the full-resolution skip head: what it is, what it cannot see, and what it must not move.

Four groups of claims, in the order the ticket makes them:

1. **Nothing registered moved.** Every existing arm keeps its ``config_hash``, §4.1's head keeps
   its parameter count, its recipe keeps its fields, and R5's printed table keeps its eleven rows.
   The hashes are pinned as literals taken from ``main`` before this ticket touched anything, so a
   drift is caught as a changed sixteen-character string rather than as a re-derived agreement.
2. **The variant is the registered head plus a skip path**, differing across the three encoders
   only by the first projection's input width, and A1 at step 0 exactly.
3. **The skip input never sees the void's truth**, proved on a perturbed tile with a head whose
   output is non-zero, beside a negative control showing the path is live.
4. **The new arms are inside every gate their registered twins are inside**: the frozen-recipe
   refusal fires for a moved knob, and the committed inputs they read are the same set.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from benchmark import arms, artifacts, evaluate, head, launch, skiphead, train, tune, unet
from benchmark.config import TUNED_HEAD, BenchmarkConfig, Head
from tests.test_benchmark_head import (  # noqa: F401 - fixtures by name
    WIDTH,
    StubExtractor,
    box,
    flags,
    frozen,
    splits_of,
    stub,
    table_of,
)

SKIP_ARMS = ("A4-R5-SKIP", "A5-SKIP", "A6-SKIP")
TWIN = {"A4-R5-SKIP": "A4-R5", "A5-SKIP": "A5", "A6-SKIP": "A6"}

#: Every arm registered before #182, at `BenchmarkConfig(arm=name)`'s defaults, hashed on `main`
#: at 3a6f3c9. These are the strings 254,877 committed rows carry; none may move.
HASHES_BEFORE_P182 = {
    "A1": "a41e3e2bfef7916e",
    "A2-R5": "eeae1d52a761ea41",
    "A2-R4": "f9ecde0076aef5ba",
    "A3-R5": "db1c37c391cfc388",
    "A3-R4": "24956f0c06e34880",
    "A4-R5": "18489237db77449b",
    "A4-R4": "55cfba411c8f0738",
    "A5": "4fc3cf2879931fb8",
    "A6": "5a5488eb944005b9",
    "A7": "7263e69f88ba2c52",
    "A9": "7d7f8ae72a60623a",
    "A9-NOTHRESH": "d76138c3d2457c11",
    "A5-BICUBIC": "770f0e120bc13801",
    "A4-S": "716805ace1c42475",
}


# --- 1. Nothing registered moved ------------------------------------------------


def test_every_existing_arm_keeps_its_config_hash():
    """The variant is declared on the arm, not in the config, so the ``head`` group is unchanged.

    A field added to :class:`benchmark.config.Head` — even one defaulting to the registered head
    — would move every one of these, and with them the identity of every committed row.
    """
    for name, want in HASHES_BEFORE_P182.items():
        assert BenchmarkConfig(arm=name).config_hash == want, name


def test_the_new_arms_hash_differently_from_their_twins_and_from_each_other():
    """The arm name is in the hash, so a skip cell can never be filed under its twin's identity."""
    hashes = {name: BenchmarkConfig(arm=name).config_hash for name in SKIP_ARMS}
    assert len(set(hashes.values())) == len(SKIP_ARMS)
    for name, twin in TWIN.items():
        assert hashes[name] != HASHES_BEFORE_P182[twin]


def test_the_registered_head_is_untouched():
    """§4.1's head: same count, same recipe fields, same tuned values, same tensors."""
    assert head.PARAMETERS == {384: 1_082_017, 768: 1_181_089}
    for width, count in head.PARAMETERS.items():
        assert head.parameter_count(width) == count
        assert head.build(width).parameters == count
    assert [f.name for f in dataclasses.fields(Head)] == [
        "proj_dim", "stage_dims", "loss", "slope_weight", "optimizer", "lr", "weight_decay",
        "decay", "warmup_frac", "precision", "batch_size", "epochs", "early_stop_metric",
        "seeds", "label_sweep_seeds", "appendix_seeds",
    ]
    assert TUNED_HEAD.tuned == {
        "proj_dim": 256, "stage_dims": (256, 128, 64, 32), "loss": "l1", "slope_weight": 0.1,
        "optimizer": "adamw", "lr": 3e-3, "weight_decay": 0.05, "decay": "cosine",
        "warmup_frac": 0.05, "precision": "bf16", "batch_size": 128, "epochs": 50,
        "early_stop_metric": "val_rmse",
    }
    assert set(head.build(768).module.state_dict()) == {
        "norm.weight", "norm.bias", "project.weight", "project.bias",
        *(f"convs.{i}.{p}" for i in range(4) for p in ("weight", "bias")),
        *(f"norms.{i}.{p}" for i in range(4) for p in ("weight", "bias")),
        "residual.weight", "residual.bias",
    }


def test_the_printed_table_is_still_eleven_rows_and_the_skip_arms_are_appendix_cells():
    assert arms.PRINTED_ROWS == (
        "A1", "A2-R5", "A2-R4", "A3-R5", "A3-R4", "A4-R5", "A4-R4", "A5", "A6", "A7", "A9",
    )
    for name, twin in TWIN.items():
        arm, registered = arms.get(name), arms.get(twin)
        assert arm.printed is False
        assert arm.reads_full_res_tile and not registered.reads_full_res_tile
        assert arm.trains_head and arm.trains == registered.trains
        # Same encoder, same diet, same convention: the variant is the *only* difference.
        assert (arm.encoder, arm.normalization, arm.input_convention) == (
            registered.encoder, registered.normalization, registered.input_convention
        )
        assert name not in arms.LABEL_SWEEP_ARMS


def test_a_printed_or_misdeclared_skip_arm_is_refused():
    base = dict(
        encoder="demae-r5", trains="head", input_convention="a1_fill_plus_mask",
        normalization="per_tile", what="x",
    )
    with pytest.raises(ValueError, match="appendix cell"):
        arms.Arm(name="X", head_variant="skip", printed=True, **base)
    with pytest.raises(ValueError, match="unknown head variant"):
        arms.Arm(name="X", head_variant="dpt", printed=False, **base)
    with pytest.raises(ValueError, match="trains 'nothing'"):
        arms.Arm(
            name="X", head_variant="skip", printed=False,
            **{**base, "trains": "nothing", "input_convention": "token_drop"},
        )


# --- 2. The variant ----------------------------------------------------------------


@pytest.mark.parametrize("width,expected", [(384, 1_174_801), (768, 1_273_873)])
def test_the_skip_head_costs_the_recorded_figure_at_each_width(width, expected):
    """Recorded, derived, and counted off the built module — three ways that must agree."""
    assert skiphead.PARAMETERS[width] == expected
    assert skiphead.parameter_count(width) == expected
    assert skiphead.build(width).parameters == expected
    # The skip path's cost does not depend on the encoder: the projection is the only width term.
    assert expected - head.parameter_count(width) == 92_784


def test_the_registry_reports_the_skip_head_for_the_skip_arms():
    for name, twin in TWIN.items():
        width = arms.ENCODERS[arms.get(name).encoder].width
        assert arms.parameter_counts(arms.get(name))["trainable"] == skiphead.PARAMETERS[width]
        assert arms.parameter_counts(arms.get(twin))["trainable"] == head.PARAMETERS[width]


def test_the_three_skip_arms_instantiate_one_head_differing_only_in_the_input_width():
    """C1's precondition, carried to the variant: the decoder is not a variable across encoders."""
    shapes = {
        name: {
            k: tuple(v.shape)
            for k, v in skiphead.build(arms.ENCODERS[arms.get(name).encoder].width)
            .module.state_dict().items()
        }
        for name in SKIP_ARMS
    }
    reference = shapes["A6-SKIP"]
    for name, shape in shapes.items():
        assert set(shape) == set(reference), name
        differ = {k for k in shape if shape[k] != reference[k]}
        assert differ <= {"norm.weight", "norm.bias", "project.weight"}, (name, sorted(differ))


def test_the_skip_head_is_the_registered_head_plus_the_skip_tensors():
    """Same names in the same places; the stages read sixteen more channels and nothing else."""
    registered = {k: tuple(v.shape) for k, v in head.build(768).module.state_dict().items()}
    variant = {k: tuple(v.shape) for k, v in skiphead.build(768).module.state_dict().items()}
    added = {k for k in variant if k.startswith(("skip_", "refine"))}
    assert set(variant) - added == set(registered)
    for k in registered:
        if k.startswith("convs.") and k.endswith("weight"):
            out, into, kh, kw = registered[k]
            assert variant[k] == (out, into + skiphead.SKIP_CHANNELS, kh, kw), k
        else:
            assert variant[k] == registered[k], k


def _tile(rng: np.random.Generator, seed_scale: float = 1.0) -> np.ndarray:
    """A smooth synthetic tile in metres, so A1's interpolant has something to interpolate."""
    y, x = np.mgrid[0:224, 0:224].astype(np.float32)
    field = 900 + 3.0 * x + 1.5 * y + 60 * np.sin(x / 17) * np.cos(y / 23)
    field += rng.normal(0, 2.0, size=field.shape) * seed_scale
    return np.rint(field).astype(np.int16)


def _blob() -> np.ndarray:
    y, x = np.mgrid[0:224, 0:224]
    return ((y - 110) ** 2 + (x - 120) ** 2) < 40**2


def _skip_stack(module, device: str = "cpu") -> train.SkipStack:
    return train.SkipStack(
        extractor=StubExtractor(device=device), module=module, arm="A6-SKIP", device=device
    )


def test_a_zero_output_skip_head_returns_the_a1_fill_exactly():
    """A1 at step 0, through the stack the pass trains and scores through, to the bit."""
    cfg = BenchmarkConfig(arm="A6-SKIP")
    batch = train.compose([_tile(np.random.default_rng(0))], [_blob()], cfg)
    stack = _skip_stack(skiphead.build(WIDTH).module)
    (out,) = train.predict_composed(batch, stack, precision="fp32")
    assert out.dtype == np.float32
    assert np.array_equal(out, batch.a1_fill[0])


def test_void_interior_truth_never_reaches_the_skip_head():
    """#182's leakage criterion, proved rather than asserted.

    Two tiles identical outside the void and 500 m apart inside it, through the real
    composition and a skip head whose output is **not** zero. The residual must be identical to
    the bit — and a non-void pixel moved by one metre must change it, or the test proves only
    that the head ignores its input.
    """
    cfg = BenchmarkConfig(arm="A6-SKIP")
    void = _blob()
    rng = np.random.default_rng(1)
    raw = _tile(rng)
    leaked = raw.copy()
    leaked[void] += 500
    assert not np.array_equal(raw, leaked)
    # The control moves *shape* outside the void, not level: per-tile standardization removes a
    # constant offset exactly, so a uniform +1 m would be invisible to a correct head too.
    same_outside = raw.copy()
    same_outside[10:30, 10:30] += 30
    assert not void[10:30, 10:30].any()

    built = skiphead.build(WIDTH)
    with torch.no_grad():
        torch.nn.init.normal_(built.module.residual.weight, std=0.1)
        torch.nn.init.normal_(built.module.residual.bias, std=0.1)
    stack = _skip_stack(built.module)

    def residual_for(tile: np.ndarray) -> np.ndarray:
        batch = train.compose([tile], [void], cfg)
        with torch.no_grad():
            out, _ = train.residual_of(batch, stack, precision="fp32")
        return out.numpy()

    honest, cheating, control = residual_for(raw), residual_for(leaked), residual_for(same_outside)
    assert np.abs(honest).max() > 0, "a zero head would pass vacuously"
    assert np.array_equal(honest, cheating)
    assert not np.array_equal(honest, control)
    # And the reason, one level down: the fill, its rounding, the valid mask and the per-tile
    # statistic are all functions of the non-void pixels alone.
    a, b = train.compose([raw], [void], cfg), train.compose([leaked], [void], cfg)
    assert np.array_equal(a.a1_fill, b.a1_fill)
    assert np.array_equal(a.a1_i16, b.a1_i16)
    assert np.array_equal(a.valid, b.valid)
    assert np.array_equal(
        unet.condition(a.a1_fill, a.a1_i16, a.valid)[1],
        unet.condition(b.a1_fill, b.a1_i16, b.valid)[1],
    )


def test_the_skip_head_refuses_a_tile_that_is_not_full_resolution():
    module = skiphead.build(WIDTH).module
    tokens = torch.zeros(1, 196, WIDTH)
    with pytest.raises(head.HeadError, match="full resolution"):
        module(tokens, torch.zeros(1, 1, 224, 224), torch.zeros(1, 1, 112, 112))
    with pytest.raises(head.HeadError, match="built at"):
        module(torch.zeros(1, 196, WIDTH // 2), torch.zeros(1, 1, 224, 224), torch.zeros(1, 1, 224, 224))


# --- 4. Inside every gate -----------------------------------------------------------


def test_the_frozen_recipe_gate_refuses_a_moved_knob_for_the_skip_arms_exactly_as_for_a4():
    """`launch.resolve`'s gate is keyed on arm properties, and a new arm has escaped it before.

    Asserted over the registered twin in the same loop, so the two are the *same* refusal: if
    the gate ever stops covering one of them, this fails.
    """
    parser = launch.build_parser()
    for arm in ("A4-R5", *SKIP_ARMS):
        for knob in ("head.lr=0.0003", "head.epochs=1"):
            with pytest.raises(tune.TuningError, match="frozen for every arm"):
                launch.resolve(parser.parse_args(["run", "--arm", arm, "--set", knob]))
    # And the unmoved recipe resolves for all of them — the gate refuses drift, not the arms.
    for arm in ("A4-R5", *SKIP_ARMS):
        launch.resolve(parser.parse_args(["run", "--arm", arm]))


def test_the_skip_arms_are_refused_at_the_label_efficiency_rungs():
    """§7 prices the sweep at four arms; a fifth is a row nothing budgeted."""
    parser = launch.build_parser()
    for arm in SKIP_ARMS:
        with pytest.raises(ValueError, match="sweep"):
            launch.resolve(parser.parse_args(["run", "--arm", arm, "--size", "1k"]))


def test_the_skip_arms_read_the_same_committed_inputs_as_their_twins():
    cfg = BenchmarkConfig()
    for name, twin in TWIN.items():
        mine = artifacts.resolve(cfg.replace(arm=name), arms.get(name))
        theirs = artifacts.resolve(cfg.replace(arm=twin), arms.get(twin))
        assert {r.key: r.required for r in mine.inputs} == {
            r.key: r.required for r in theirs.inputs
        }


def test_the_resolution_line_names_the_skip_head_and_its_count(capsys):
    parser = launch.build_parser()
    resolution = launch.resolve(parser.parse_args(["run", "--arm", "A4-R5-SKIP", "--dry-run"]))
    text = launch.format_resolution(resolution, verb="run")
    assert "skip head" in text and "1,273,873" in text and "92,784" in text
    assert "frozen for every arm" in text


# --- Seam 1: the shipping command ----------------------------------------------------


def test_the_shipping_command_trains_a_skip_head_and_records_which_head_it_was(
    box, flags, stub, frozen, capsys
):
    """One A6-SKIP cell through `main(argv)`: train through the skip stack, then score.

    Beside an A6 cell on the same box, so the two heads are trained and scored by the one pass
    and the record says which was which.
    """
    from benchmark import results

    assert launch.main(["run", "--arm", "A6-SKIP", *flags]) == 0
    out = capsys.readouterr().out
    assert "skip head" in out and "1,273,873 (1.27M) at 768-d" in out
    table = table_of(box)
    assert set(table.column("arm").to_pylist()) == {"A6-SKIP"}
    assert table.num_rows == len(splits_of(box)["test"]) * 2 * 3

    (record,) = results.read_run_records(results.run_record_path(box / "results.parquet"))
    assert record["training"]["kind"] == "skiphead"
    assert record["training"]["parameters"] == 1_273_873
    assert record["parameters"]["trainable"] == 1_273_873
    assert record["training"]["epochs_run"] == 2

    assert launch.main(["run", "--arm", "A6", *flags]) == 0
    records = results.read_run_records(results.run_record_path(box / "results.parquet"))
    kinds = {r["cell"]["arm"]: r["training"]["kind"] for r in records}
    assert kinds == {"A6-SKIP": "skiphead", "A6": "head"}
