"""B7: Task 1's A7 UNet on one channel, step-matched to B2, tuning itself (#209).

**Seam 1** is :attr:`benchmark.landform_config.LandformConfig.steps`: B7's run length at each
rung, against plan §4.5's table, typed from the plan (1,280 / 1,280 / 12,520) rather than read
off B2's code, so the assertion holds whether or not #208's B2 is on this branch.

**Seam 2** is :func:`benchmark.unet.build`, Task 1's builder. A7's output is pinned by a
fingerprint of its seeded state dict taken *before* #209 touched the builder, so "A7 is
unchanged" is a comparison against the past and not against the code. B7's trunk is the same
spec with one input channel and ten class logits, its count derived by hand from A7's.

**Seam 3** is :func:`benchmark.launch.main`: a B7 cell end to end on #206's fixture box, and
``--tune --head unet`` writing B7's recipe lines and refused on any other arm.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark import landform_arms, landform_recipes, launch, unet
from benchmark.landform_config import RUNGS, LandformConfig, build_config
from tests.test_p206_linear_probe_cell import (  # noqa: F401  (fixtures)
    STEPS,
    TILES_PER_COG,
    _flags,
    box,
    corpus,
    records_of,
    table_of,
)

#: Plan §4.5's table, the "UperNet and B7 (batch 64, 40 epochs)" row.
PLAN_STEPS = {"1%": 1_280, "10%": 1_280, "100%": 12_520}


# --- Seam 1: the schedule -----------------------------------------------------


def test_b7_trains_b2_s_step_count_at_every_rung():
    for rung in RUNGS:
        b7 = LandformConfig(arm="B7", rung=rung)
        assert b7.head == "unet"
        assert b7.steps == PLAN_STEPS[rung], rung
        if "B2" in landform_arms.ARMS:  # once #208's row is merged, against it directly too
            assert LandformConfig(arm="B2", rung=rung).steps == b7.steps, rung


# --- Seam 2: Task 1's builder ---------------------------------------------------

#: A7's seeded state dict, fingerprinted on the builder as it stood before #209 (sha256 over
#: each key, shape and float32 bytes in state-dict order).
A7_FINGERPRINTS = {
    0: "86a9e2b49275122a0230fb344313691733f54ce3651edad1747de26f5f87727f",
    2: "940a22eb535fadad04054ab2d3f0a8364bb92175d96a4915975a5b7b7f9ff405",
}

#: A7's 17,465,377 with one input channel fewer at the first 3x3 conv (-1 x 48 x 9) and ten
#: output channels in place of one at the 1x1 (+9 x 48 weights, +9 biases).
B7_PARAMETERS = 17_465_377 - 48 * 9 + 9 * 48 + 9


def _fingerprint(module) -> str:
    import hashlib

    h = hashlib.sha256()
    for key, value in module.state_dict().items():
        h.update(key.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("seed", sorted(A7_FINGERPRINTS))
def test_task_one_s_a7_builder_output_is_unchanged(seed):
    pytest.importorskip("torch")
    from benchmark import unet

    built = unet.build(seed=seed)
    assert built.parameters == 17_465_377
    assert _fingerprint(built.module) == A7_FINGERPRINTS[seed]


def test_b7_s_trunk_is_a7_s_spec_on_one_channel_with_ten_class_logits():
    torch = pytest.importorskip("torch")
    from benchmark import unet
    from benchmark.landform_train import CLASSES

    built = unet.build(seed=0, in_channels=1, classes=CLASSES)
    assert built.parameters == B7_PARAMETERS == unet.parameter_count(in_channels=1,
                                                                     classes=CLASSES)
    out = built.module(torch.zeros(2, 1, 224, 224))
    assert out.shape == (2, CLASSES, 224, 224)
    # Not A7's zeroed residual: a classifier that starts at all-zero logits would be a
    # convention borrowed from §2.4 that nothing in §4.3 asks for.
    assert built.module.residual.weight.abs().sum() > 0


# --- Seam 3: the command --------------------------------------------------------

#: B7 at #206's test scale: batch 4, 5 epochs over the 8-tile floor -> ten steps at 1%.
UNET_TEST_RECIPE = ["--set", "unet.batch_size=4", "--set", "unet.epochs=5"]


@pytest.fixture
def flags(box, corpus):
    return [*_flags(box, corpus), *UNET_TEST_RECIPE]


def _cfg(flags, **cell) -> LandformConfig:
    return build_config(overrides=flags[2:][1::2], **cell)


@pytest.fixture
def b7_lines(box, flags):
    """B7's recipe lines at every rung, as though its own tuning pass had frozen them."""
    for rung in RUNGS:
        landform_recipes.write_line(_cfg(flags, arm="B7", rung=rung), lr=1e-3,
                                    scores={"0.001": 0.5})
    return box


def test_a_b7_cell_trains_from_scratch_and_appends_rows_and_a_run_record(
    b7_lines, flags, capsys
):
    assert launch.main(["run", "--arm", "B7", "--rung", "1%", "--seed", "0", *flags]) == 0
    out = capsys.readouterr().out
    assert "encoder          none" in out

    table = table_of(b7_lines)
    assert table.num_rows == 2 * TILES_PER_COG, "one row per test tile"
    rows = table.to_pylist()
    assert {(r["arm"], r["rung"], r["seed"], r["ladder"]) for r in rows} == {("B7", "1%", 0, 0)}
    assert all(np.asarray(r["confusion"]).shape == (100,) for r in rows)

    (record,) = records_of(b7_lines)
    assert record["cell"]["head"] == "unet"
    assert record["config_hash"] == _cfg(flags, arm="B7", rung="1%", seed=0).config_hash
    assert record["encoder"] is None
    assert record["parameters"] == {"encoder": 0, "trainable": B7_PARAMETERS,
                                    "total": B7_PARAMETERS}
    assert record["recipe_line"]["tuned_on"]["arm"] == "B7"
    assert record["training"]["kind"] == "unet"
    assert record["training"]["steps"] == STEPS
    assert len(record["training"]["val_reads"]) == 10
    # One naming rule for every Task 2 cell's checkpoint directory, the head included (#197 review).
    assert record["training"]["checkpoint"] == str(
        b7_lines / "heads" / "B7-unet-1%-seed0" / "best.pt")


@pytest.fixture
def bare(box):
    """#206's box before any tuning pass: no recipe lines."""
    (box / "task2" / landform_recipes.RECIPE_LINES_NAME).unlink()
    return box


def test_tune_head_unet_trains_b7_on_itself_and_freezes_its_line(bare, flags, capsys):
    import json

    assert launch.main(["run", "--tune", "--head", "unet", "--rung", "1%", *flags]) == 0
    lines = json.loads((bare / "task2" / landform_recipes.RECIPE_LINES_NAME).read_text())["lines"]
    assert set(lines) == {"unet"} and set(lines["unet"]) == {"1%"}
    line = lines["unet"]["1%"]
    assert line["tuned_on"] == {"arm": "B7", "seed": 0, "ladder": 0}
    assert tuple(line["grid"][:5]) == (1e-4, 3e-4, 1e-3, 3e-3, 1e-2) or \
        tuple(line["grid"][-5:]) == (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    assert line["lr"] in line["grid"]
    assert line["recipe"] == {"in_channels": 1, "optimizer": "adamw", "weight_decay": 0.05,
                              "batch_size": 4, "epochs": 5}
    record = json.loads(landform_recipes.tuning_record_path_for(bare / "task2", "unet", "1%")
                        .read_text())
    assert record["arm"] == "B7" and record["steps_per_point"] == STEPS // 5
    assert not (bare / "results.parquet").exists(), "a tuning pass scores nothing"


@pytest.mark.parametrize("arm", ["B1", "B6", "B1-R4", "B1-S"])
def test_tune_head_unet_on_any_other_arm_is_refused(arm, bare, flags, capsys):
    for dry in (["--dry-run"], []):
        with pytest.raises(ValueError, match=rf"tunes the unet head on B7 alone.*arm {arm}"):
            launch.main(["run", "--tune", "--head", "unet", "--arm", arm, "--rung", "1%",
                         *flags, *dry])
    assert not (bare / "task2" / landform_recipes.RECIPE_LINES_NAME).exists()


def test_b7_trains_under_the_optimizer_its_config_names_and_refuses_any_but_adamw(bare, flags):
    """#197 review: `unet.optimizer` is read by the trainer, not stated beside a hard-coded
    AdamW. A tuning pass would otherwise freeze an `sgd` line it never trained under."""
    with pytest.raises(ValueError, match=r"unet\.optimizer 'sgd'.*AdamW"):
        launch.main(["run", "--tune", "--head", "unet", "--rung", "1%", *flags,
                     "--set", 'unet.optimizer="sgd"'])
    assert not (bare / "task2" / landform_recipes.RECIPE_LINES_NAME).exists()


def test_every_head_a_cell_can_name_has_a_trainer():
    """The config holds a cell's head to `HEADS`, so this is what keeps `_run` from needing its
    own no-trainer refusal (#197 review)."""
    from benchmark import landform

    assert set(landform.TRAINERS) == set(landform_arms.HEADS)
