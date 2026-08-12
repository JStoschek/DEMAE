"""The frozen benchmark run object and its config hash (#147).

`pretrain.config.RunConfig`'s sibling, and the tests are its tests' sibling: the three
properties the rest of Task 1 leans on are that the object is frozen, that it round-trips
through a *partial* dict, and that it hashes to something two boxes agree on which covers the
recipe and not where the corpus is mounted.

The numeric assertions here are the plan's own cited defaults. They are pinned rather than
left to the docstrings because "every knob from §4 at its cited default" is the whole
specification of this module, and a default that drifts silently is a benchmark whose arms
stop being the arms the plan registered.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import pytest

import benchmark

from benchmark import arms
from benchmark.config import (
    LABEL_EFFICIENCY_SIZES,
    BenchmarkConfig,
    Paths,
    add_config_arguments,
    build_config,
    config_from_args,
    parse_override,
)
from dataset import config as dataset_config


# --- The package's one structural promise -------------------------------------


def _imported_modules(source: str) -> set[str]:
    """Every top-level package name this module imports, however it spells the import."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_the_package_is_torch_optional_the_way_dataset_is():
    """The freeze and scoring paths import no torch, so A1 builds on a box with no CUDA stack.

    Read off the source rather than off ``sys.modules``, which a suite that has already imported
    `pretrain` would make vacuous. An arm that runs an encoder will reach for torch — and when
    it does, it does so lazily, the way `dataset.loader` does (ADR-0003).
    """
    directory = Path(benchmark.__file__).parent
    offenders = {
        path.name for path in directory.glob("*.py") if "torch" in _imported_modules(path.read_text())
    }
    assert not offenders, f"{sorted(offenders)} import torch at module scope"


# --- §4's knobs, at the numbers the plan cites --------------------------------


def test_the_head_carries_section_four_ones_recipe():
    head = BenchmarkConfig().head
    assert (head.proj_dim, head.stage_dims) == (256, (256, 128, 64, 32))
    assert (head.loss, head.slope_weight) == ("l1", 0.1)
    assert (head.optimizer, head.lr, head.weight_decay) == ("adamw", 1e-3, 0.05)
    assert (head.decay, head.warmup_frac, head.precision) == ("cosine", 0.05, "bf16")
    assert (head.batch_size, head.epochs) == (128, 50)
    assert head.early_stop_metric == "val_rmse"
    # §4.1's seed counts: three per primary cell, two on the label sweep, and the one that is
    # an exception rather than a default — §2.3's ViT-S appendix cells.
    assert (head.seeds, head.label_sweep_seeds, head.appendix_seeds) == (3, 2, 1)


def test_the_fine_tune_arm_carries_section_four_twos_recipe():
    ft = BenchmarkConfig().finetune
    assert ft.epochs == 20
    assert (ft.encoder_lr, ft.layer_decay, ft.decoder_lr) == (1e-4, 0.75, 1.5e-3)
    assert ft.random_patch_frac == 0.25
    # §4.2: void filling needs absolute per-patch scale, so norm_pix is off — the one place
    # this benchmark deliberately inverts Phase 1's own default.
    assert ft.norm_pix_loss is False
    assert (ft.crossfade_px, ft.shift_offsets) == (16, (0, 112))


def test_the_tile_draws_are_section_three_ones():
    tiles = BenchmarkConfig().tiles
    assert (tiles.head_train, tiles.val, tiles.test) == (20_000, 2_000, 5_000)
    assert tiles.strata == 20
    assert (tiles.train_split, tiles.val_split, tiles.test_split) == ("train", "val", "test")


def test_the_mask_geometry_is_section_three_twos():
    masks = BenchmarkConfig().masks
    assert masks.synthetic_families == ("blob", "rectangle")
    assert (masks.small_px, masks.medium_px, masks.large_px) == ((16, 48), (64, 96), (128, 158))
    assert (masks.edge_margin_px, masks.max_area_frac) == (8, 0.40)
    assert masks.bank_draws_per_tile == 50


def test_the_large_buckets_top_is_inside_the_area_cap_by_construction():
    """§3.2's settled 158, checked against the cap rather than copied beside it.

    The published 160 overshot by 0.14 px: the 40% area cap permits an equivalent diameter of
    159.86 px, and #121 measured that away by lowering the top to 158 rather than clamping, so
    no draw is ever silently trimmed. Both numbers live here, so the config cannot carry a
    bucket the generator would have to trim.
    """
    masks = BenchmarkConfig().masks
    permitted = 2 * math.sqrt(masks.max_area_frac * dataset_config.TILE_PX**2 / math.pi)
    assert permitted == pytest.approx(159.86, abs=0.01)
    assert masks.max_equivalent_diameter_px == pytest.approx(permitted)
    assert masks.large_px[1] < permitted


def test_a_bucket_top_above_the_area_cap_is_refused_rather_than_trimmed():
    with pytest.raises(ValueError, match="40% area cap|area cap"):
        BenchmarkConfig().masks.replace(large_px=(128, 176))


def test_the_bank_is_fifty_draws_per_head_train_tile():
    cfg = BenchmarkConfig()
    assert cfg.bank_rows == 50 * 20_000 == 1_000_000


def test_the_extractor_contract_is_adr_twelves():
    extract = BenchmarkConfig().extract
    assert extract.reference_dtype == "fp32"  # ADR-0012 §6 — fp32 is the reference tensor
    assert extract.layers == 12 and extract.tokens == 196
    assert extract.emit_intermediates is True
    assert extract.valid_mask == "land_and_not_void"  # ADR-0012 §1


def test_the_intermediates_path_cannot_be_switched_off():
    # ADR-0012 §3: a last-layer-only extractor is precisely the option under which the map's
    # one cross-task constraint fails, and Task 2 quietly writes a second extractor.
    with pytest.raises(ValueError, match="intermediates"):
        BenchmarkConfig().extract.replace(emit_intermediates=False)


# --- The arm, the size and the seed -------------------------------------------


def test_an_arm_is_this_object_with_a_knob_moved():
    base = BenchmarkConfig()
    assert base.arm == "A4-R5"
    moved = base.replace(arm="A5")
    assert moved.arm == "A5" and moved.head == base.head


def test_an_unknown_arm_is_refused_and_the_registry_is_named():
    with pytest.raises(ValueError, match="unknown arm 'A8'"):
        BenchmarkConfig(arm="A8")


def test_the_label_efficiency_size_selects_the_nested_subset():
    assert LABEL_EFFICIENCY_SIZES == ("1k", "5k", "20k")
    assert [BenchmarkConfig(size=s).train_tiles for s in LABEL_EFFICIENCY_SIZES] == [
        1_000, 5_000, 20_000
    ]
    with pytest.raises(ValueError, match="unknown size"):
        BenchmarkConfig(size="10k")


def test_the_largest_subset_is_the_head_train_draw_itself():
    # The sweep is nested inside one committed draw (§3.1), so a config whose 20k subset was
    # bigger than the draw it is nested in would be asking for tiles that were never drawn.
    with pytest.raises(ValueError, match="head_train"):
        BenchmarkConfig(tiles=BenchmarkConfig().tiles.replace(head_train=5_000))


# --- Round-tripping -----------------------------------------------------------


def test_the_config_round_trips_through_a_plain_dict():
    cfg = BenchmarkConfig(arm="A5", seed=2)
    assert BenchmarkConfig.from_dict(cfg.to_dict()) == cfg
    json.dumps(cfg.to_dict())  # plain JSON types throughout — no dataclasses, no tuples


def test_a_partial_dict_fills_the_rest_from_the_plan():
    cfg = BenchmarkConfig.from_dict({"arm": "A6", "head": {"epochs": 10}})
    assert cfg.arm == "A6" and cfg.head.epochs == 10
    assert cfg.head.lr == BenchmarkConfig().head.lr  # everything else is still §4.1's
    assert cfg.masks == BenchmarkConfig().masks


def test_an_unknown_group_raises_rather_than_being_ignored():
    with pytest.raises(ValueError, match="unknown config group\\(s\\) \\['heads'\\]"):
        BenchmarkConfig.from_dict({"heads": {"epochs": 10}})


def test_an_unknown_knob_names_the_group_it_was_not_found_in():
    with pytest.raises(ValueError, match="unknown head knob\\(s\\) \\['learning_rate'\\]"):
        BenchmarkConfig.from_dict({"head": {"learning_rate": 1e-3}})


# --- The config hash ----------------------------------------------------------


def test_the_hash_is_stable_and_short_enough_to_read():
    cfg = BenchmarkConfig()
    assert cfg.config_hash == BenchmarkConfig().config_hash
    assert len(cfg.config_hash) == 16 and int(cfg.config_hash, 16) >= 0


def test_the_hash_moves_when_a_recipe_knob_moves():
    base = BenchmarkConfig()
    for moved in (
        base.replace(arm="A5"),
        base.replace(size="5k"),
        base.replace(seed=1),
        base.replace(head=base.head.replace(lr=5e-4)),
        base.replace(masks=base.masks.replace(bank_draws_per_tile=20)),
        base.replace(finetune=base.finetune.replace(epochs=10)),
    ):
        assert moved.config_hash != base.config_hash


def test_the_hash_does_not_move_when_an_environment_path_does():
    """§11's argument, carried into Phase 2: a rented box mounts the corpus somewhere else.

    Every field of :class:`Paths` is environment. Which encoder an arm reads is the *arm*, and
    the arm is hashed; where that checkpoint is mounted is a fact about the box, and its md5 is
    what the run record carries (see :mod:`benchmark.artifacts`).
    """
    base = BenchmarkConfig()
    moved = base.replace(
        paths=Paths(
            corpus_dir="/workspace/corpus_500k",
            artifacts_dir="/workspace/artifacts/task1",
            mask_bank_path="/scratch/bank.parquet",
            demae_r5_checkpoint="/workspace/r5/encoder.pt",
        )
    )
    assert moved.paths != base.paths
    assert moved.config_hash == base.config_hash


def test_the_generator_hash_does_not_move_with_the_cell():
    """§3.4 — the two mask artifacts carry one generator hash, so it is one contract.

    `freeze` writes inputs every cell shares. A hash that moved with ``--arm`` would be
    claiming the artifacts had, which is exactly the confusion a second determinism contract
    would produce.
    """
    base = BenchmarkConfig()
    for same in (base.replace(arm="A5"), base.replace(size="1k"), base.replace(seed=3)):
        assert same.artifact_hash == base.artifact_hash
        assert same.config_hash != base.config_hash
    assert base.replace(masks=base.masks.replace(seed=1)).artifact_hash != base.artifact_hash
    assert base.replace(tiles=base.tiles.replace(test=4_000)).artifact_hash != base.artifact_hash
    # And it is not the head's: a head knob moving does not re-draw a mask.
    assert base.replace(head=base.head.replace(lr=1e-4)).artifact_hash == base.artifact_hash


def test_the_recipe_view_carries_no_path_at_all():
    recipe = BenchmarkConfig().recipe_dict()
    assert "paths" not in recipe
    assert {"arm", "size", "seed", "tiles", "masks", "head"} <= set(recipe)


# --- The override surface -----------------------------------------------------


def test_an_override_is_read_as_json_and_falls_back_to_the_string():
    assert parse_override("head.epochs=10") == {"head": {"epochs": 10}}
    assert parse_override("head.slope_weight=0.2") == {"head": {"slope_weight": 0.2}}
    assert parse_override("finetune.norm_pix_loss=false") == {"finetune": {"norm_pix_loss": False}}
    assert parse_override("arm=A5") == {"arm": "A5"}  # not JSON, so a bare string
    assert parse_override("paths.corpus_dir=data/corpus_500k") == {
        "paths": {"corpus_dir": "data/corpus_500k"}
    }


def test_an_override_that_is_not_knob_equals_value_is_refused():
    with pytest.raises(ValueError, match="not KNOB=VALUE"):
        parse_override("head.epochs")
    with pytest.raises(ValueError, match="deeper than that"):
        parse_override("head.stages.0=256")


def test_the_overrides_are_applied_last_so_the_command_line_always_wins(tmp_path):
    path = tmp_path / "arm.json"
    path.write_text(json.dumps({"arm": "A6", "head": {"epochs": 5}}))
    cfg = build_config(config_path=path, overrides=["head.epochs=7"])
    assert cfg.arm == "A6" and cfg.head.epochs == 7


def test_the_one_consumer_honours_set():
    parser = add_config_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--set", "head.epochs=9", "--set", "arm=A5"])
    cfg = config_from_args(args)
    assert cfg.head.epochs == 9 and cfg.arm == "A5"


# --- The registry -------------------------------------------------------------


def test_the_registry_carries_the_eleven_printed_rows():
    assert arms.PRINTED_ROWS == (
        "A1", "A2-R5", "A2-R4", "A3-R5", "A3-R4", "A4-R5", "A4-R4", "A5", "A6", "A7", "A9"
    )
    assert set(arms.PRINTED_ROWS) <= set(arms.ARMS)


def test_the_vit_s_cells_are_registered_as_appendix_rather_than_printed():
    # §2.3 demoted ViT-S to an appendix aside no §0 claim rests on, and #136 fixed the printed
    # table's shape. An appendix cell that printed would widen a pre-registered table.
    for name in ("A4-S", "A6-S"):
        assert name in arms.ARMS and not arms.ARMS[name].printed


def test_the_ruled_out_cross_evals_are_not_in_the_registry_at_all():
    """A8 and A10 are out of scope (#128) and are *absent*, not disabled.

    A disabled entry would put a ruled-out arm back on a surface someone reads, so there is
    deliberately no special-cased message for either: they fail the same way any other typo
    does.
    """
    assert "A8" not in arms.ARMS and "A10" not in arms.ARMS
    with pytest.raises(ValueError, match="unknown arm 'A10'"):
        arms.get("A10")


def test_every_frozen_head_arm_declares_the_identical_head():
    # C1 isolates the representation and nothing else (§1): A4, A5 and A6 share one head and
    # differ only in the input channel width the first projection absorbs.
    frozen = [arms.ARMS[name] for name in ("A4-R5", "A4-R4", "A5", "A6")]
    assert {arm.trains for arm in frozen} == {"head"}
    assert {arm.input_convention for arm in frozen} == {"a1_fill_plus_mask"}
    assert {arms.ENCODERS[arm.encoder].width for arm in frozen} == {768}


def test_the_two_demae_inits_differ_by_the_axis_order_and_nothing_else():
    # ADR-0011 / §2.3's precondition: R4 learned mmselfsup's row-first sin-cos, R5 MAE's own.
    # Every name and every shape matches under either, so the registry is where it is declared.
    r5, r4 = arms.ENCODERS["demae-r5"], arms.ENCODERS["demae-r4"]
    assert (r5.pos_encoding, r4.pos_encoding) == ("sincos2d", "sincos2d_rowfirst")
    assert r5.width == r4.width == 768


def test_the_terramind_arm_declares_adr_elevens_re_derivation():
    # Not a convention living in extraction code: a declared field, so a bicubic resize cannot
    # be swapped in silently (ADR-0011, ADR-0012 §7.4).
    assert arms.ENCODERS["terramind"].pos_encoding == "rederived_14x14_crop"
