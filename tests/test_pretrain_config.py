"""The run config: the plan's cited defaults, the dict round-trip, and the config hash (#89).

Two kinds of test here, and the distinction matters. The *defaults* tests are transcription
checks — they pin the numbers plan §2-§8 cites, so a knob cannot drift without a failing
test naming the section it came from. The *behaviour* tests cover round-trip, validation and
the hash, which are properties of this module rather than of the plan.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from pretrain.config import RunConfig


# --- The plan's cited defaults (§2-§8) ----------------------------------------


def test_default_backbone_is_the_vit_s_iteration_config():
    a = RunConfig().architecture
    assert (a.encoder_dim, a.encoder_depth, a.encoder_heads) == (384, 12, 6)  # §2.1 ViT-S/16
    assert (a.img_px, a.patch_px, a.in_chans) == (224, 16, 1)
    assert a.pos_encoding == "sincos2d"  # §2.3; GSDPE is Phase 4


def test_default_decoder_is_shallow_and_narrow():
    a = RunConfig().architecture
    assert (a.decoder_depth, a.decoder_dim, a.decoder_heads) == (4, 256, 8)  # §2.2


def test_vit_base_preset_widens_the_decoder_and_holds_the_effective_batch():
    b = RunConfig.vit_base()
    assert (b.architecture.encoder_dim, b.architecture.encoder_depth) == (768, 12)
    assert (b.architecture.decoder_depth, b.architecture.decoder_dim) == (4, 512)  # §2.2
    # §6 drops the physical batch for ViT-B; accumulation absorbs it so the effective
    # batch — and therefore the LR — is the same recipe as ViT-S.
    assert b.optimization.batch_size == 512
    assert b.effective_batch == RunConfig().effective_batch == 2048
    assert b.lr == pytest.approx(1.2e-3)


def test_masking_defaults_to_the_mae_optimum():
    assert RunConfig().masking.ratio == 0.75  # §3
    assert RunConfig().masking.strategy == "random"  # random beats block/grid, MAE Table 1f


def test_loss_defaults_carry_both_dem_guards():
    loss = RunConfig().loss
    assert loss.norm_pix_loss is True  # §4
    assert loss.norm_pix_var_floor == 1e-4  # §4 guard 1: raised from the reference 1e-6
    assert loss.norm_pix_stats_dtype == "fp32"  # §4: fp32 stats regardless of autocast
    assert loss.masked_patches_only is True  # §4: all-patch loss costs ~0.5%
    assert loss.ocean_weighted is True  # §4: invalid pixels contribute zero
    assert loss.min_valid_patch_frac == 0.25  # §4: below this the patch leaves the loss


def test_optimizer_and_lr_follow_the_mae_recipe():
    o = RunConfig().optimization
    assert (o.optimizer, o.betas, o.weight_decay, o.fused) == ("adamw", (0.9, 0.95), 0.05, True)
    assert o.blr == 1.5e-4  # §6 MAE default
    assert (o.batch_size, o.accum_iter) == (1024, 2)
    assert o.grad_clip is None  # §6: MAE clips nothing; 1.0 only if bf16 misbehaves
    assert (o.dropout, o.drop_path, o.ema) == (0.0, 0.0, False)


def test_lr_is_derived_from_the_linear_scaling_rule_not_stored():
    cfg = RunConfig()
    assert cfg.effective_batch == 1024 * 2
    assert cfg.lr == pytest.approx(1.5e-4 * 2048 / 256)  # §6
    assert cfg.lr == pytest.approx(1.2e-3)
    # halve the accumulation and the LR follows, because it is computed, not typed
    halved = cfg.replace(optimization=cfg.optimization.replace(accum_iter=1))
    assert halved.lr == pytest.approx(6e-4)


def test_schedule_defaults_to_400_epochs_with_a_5_percent_warmup():
    s = RunConfig().schedule
    assert (s.epochs, s.warmup_epochs, s.decay) == (400, 20, "cosine")  # §6
    assert s.milestones == (50, 100, 200, 400)  # §6/§14, kept for the compute-vs-quality plot
    assert s.warmup_epochs / s.epochs == 0.05


def test_runtime_defaults_to_bf16_without_a_grad_scaler():
    r = RunConfig().runtime
    assert r.precision == "bf16"  # §7.1 — removes the official repo's fp16 NaN mode
    assert r.grad_scaler is False
    assert r.attention == "sdpa"  # §7.2
    assert (r.compile, r.compile_dynamic) == (True, False)  # §7.3 static shapes
    assert r.grad_checkpointing is False and r.channels_last is False  # §7.6 not used


def test_data_defaults_to_per_tile_normalization_and_geometric_augmentation():
    d = RunConfig().data
    assert d.normalization == "per_tile"  # §5 / ADR-0003 default
    assert d.dihedral is True  # §6: overhead terrain has no canonical orientation
    assert d.random_resized_crop is False  # §6: a recorded deviation from the MAE default
    assert d.corpus_dir  # the shipped 500k tier, by default
    assert d.train_split == "train" and d.val_split == "val"


def test_init_defaults_to_scratch_with_the_summed_patch_embed_for_the_continual_arm():
    i = RunConfig().init
    assert i.mode == "scratch"  # §8 — the clean elevation-only claim
    assert i.patch_embed_adapt == "sum"  # §8 — sum preserves scale, mean shrinks it 3x


# --- Validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "group, field, value",
    [
        ("masking", "ratio", 1.0),
        ("masking", "ratio", 0.0),
        ("architecture", "patch_px", 15),  # does not divide 224
        ("architecture", "decoder_heads", 7),  # does not divide 256
        ("schedule", "warmup_epochs", 500),  # longer than the run
        ("optimization", "accum_iter", 0),
        ("data", "normalization", "bogus"),
        ("runtime", "precision", "fp16"),  # §7.1 rules it out explicitly
        ("init", "mode", "bogus"),
    ],
)
def test_impossible_configs_are_refused_at_construction(group, field, value):
    cfg = RunConfig()
    sub = getattr(cfg, group)
    with pytest.raises(ValueError):
        cfg.replace(**{group: sub.replace(**{field: value})})


def test_milestones_past_the_last_epoch_are_dropped_so_the_200_epoch_arms_build():
    # §12's R2 and R3 read out at 200 epochs against the same milestone set. Refusing them
    # would make every ablation pass a second override to say "keep the ones that happen".
    cfg = RunConfig().replace(schedule=RunConfig().schedule.replace(epochs=200))
    assert cfg.schedule.milestones == (50, 100, 200)
    assert RunConfig.from_dict(cfg.to_dict()) == cfg  # and the round-trip is on the normalized set


def test_a_milestone_that_is_not_an_epoch_is_still_refused():
    cfg = RunConfig()
    with pytest.raises(ValueError):
        cfg.replace(schedule=cfg.schedule.replace(milestones=(0, 50)))


# --- Round-trip ---------------------------------------------------------------


def test_round_trips_dict_to_object_to_dict_unchanged():
    d = RunConfig().to_dict()
    assert RunConfig.from_dict(d).to_dict() == d


def test_round_trips_object_to_dict_to_object_unchanged():
    cfg = RunConfig.vit_base()
    assert RunConfig.from_dict(cfg.to_dict()) == cfg


def test_the_dict_is_plain_json_with_no_tuples_or_paths_left_in_it():
    d = RunConfig().to_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["optimization"]["betas"] == [0.9, 0.95]  # tuple -> list, and back on read
    assert RunConfig.from_dict(d).optimization.betas == (0.9, 0.95)


def test_from_dict_refuses_a_key_it_does_not_know():
    d = RunConfig().to_dict()
    d["masking"]["ratio_"] = 0.9  # a typo, not a knob
    with pytest.raises(ValueError, match="ratio_"):
        RunConfig.from_dict(d)


def test_from_dict_refuses_an_unknown_group():
    d = RunConfig().to_dict()
    d["mask"] = {}
    with pytest.raises(ValueError, match="mask"):
        RunConfig.from_dict(d)


def test_a_partial_dict_fills_the_rest_from_the_plan_defaults():
    # what an ablation's config file looks like: the one knob it moves, nothing else
    cfg = RunConfig.from_dict({"masking": {"ratio": 0.9}})
    assert cfg.masking.ratio == 0.9  # R2 (#83)
    assert cfg == RunConfig().replace(masking=RunConfig().masking.replace(ratio=0.9))


# --- The config hash ----------------------------------------------------------


def test_the_hash_is_insensitive_to_key_order():
    d = RunConfig().to_dict()
    shuffled = {
        k: dict(reversed(list(v.items()))) if isinstance(v, dict) else v
        for k, v in reversed(list(d.items()))
    }
    assert RunConfig.from_dict(shuffled).config_hash == RunConfig().config_hash


_PRINT_HASH = "from pretrain.config import RunConfig; print(RunConfig().config_hash)"


@pytest.mark.parametrize("hashseed", ["1", "2"])
def test_the_hash_is_stable_across_processes(hashseed):
    # the property that makes it usable as a W&B tag (§15.7): not `hash()`, which
    # PYTHONHASHSEED randomizes per process, and not `id()`. Two child interpreters at
    # different seeds must agree with this one.
    out = subprocess.run(
        [sys.executable, "-c", _PRINT_HASH],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONHASHSEED": hashseed},
    )
    assert out.stdout.strip() == RunConfig().config_hash


def test_moving_a_recipe_knob_moves_the_hash():
    cfg = RunConfig()
    r2 = cfg.replace(masking=cfg.masking.replace(ratio=0.9))
    r3 = cfg.replace(loss=cfg.loss.replace(norm_pix_loss=False))
    assert len({cfg.config_hash, r2.config_hash, r3.config_hash}) == 3


def test_moving_the_box_does_not_move_the_hash():
    # §11 resumes a run on a fresh vast.ai instance where the corpus sits somewhere else. The
    # hash answers "is this the same recipe?", so an environment field is excluded from it —
    # otherwise every resume on a new box would read as a config mismatch.
    cfg = RunConfig()
    moved = cfg.replace(
        data=cfg.data.replace(
            corpus_dir="/workspace/corpus_500k",
            split_assignment_path="/workspace/split.csv",
        ),
        run_id="R1b",
    )
    assert moved.config_hash == cfg.config_hash
    assert moved.to_dict() != cfg.to_dict()  # but the difference is still recorded


def test_only_the_environment_fields_are_left_out_of_the_hash():
    # the failure this guards is a *recipe* knob marked environment and so silently dropped
    # from the hash, which would let two different recipes tag one W&B run. Pinning the
    # exclusion list means adding one is a deliberate edit here.
    full, recipe = RunConfig().to_dict(), RunConfig().recipe_dict()
    left_out = set()
    for key, value in full.items():
        if isinstance(value, dict):
            left_out |= {f"{key}.{k}" for k in value if k not in recipe.get(key, {})}
        elif key not in recipe:
            left_out.add(key)
    assert left_out == {
        "run_id",
        "data.corpus_dir",
        "data.split_assignment_path",
        "data.split_blocks_path",
    }


def test_two_continual_arms_from_two_checkpoints_do_not_share_a_hash():
    # `init.checkpoint` is a path but not environment: which checkpoint §8's arm warm-starts
    # from *is* the arm, so R4-from-ImageNet and R4-from-anything-else must tag differently.
    cfg = RunConfig()
    arm = cfg.replace(init=cfg.init.replace(mode="imagenet_mae", checkpoint="mae_vit_base.pth"))
    other = arm.replace(init=arm.init.replace(checkpoint="something_else.pth"))
    assert arm.config_hash != other.config_hash != cfg.config_hash
