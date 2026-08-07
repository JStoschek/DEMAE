"""§14's encoder-only export (#88).

The property under test throughout is the one from P88 §4: the export is **not** the encoder
half of ``state_dict()``, because ``pos_embed`` is a non-persistent buffer and that file loads
into timm reporting one missing key and keeping timm's *random* positional embedding. Several
tests here therefore assert on the failure directly — including by mutation, dropping the
materialized tensor and checking that the strict load is what refuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pretrain import release
from pretrain.config import RunConfig
from pretrain.model import AXIS_ORDERS, MaskedAutoencoder, sincos2d_pos_embed

R5_ARCH = RunConfig.vit_base().architecture
R4_ARCH = RunConfig.vit_base_continual("/nonexistent/mae.pth").architecture
VIT_S_ARCH = RunConfig().architecture


def an_encoder(arch):
    """A real model's encoder-only export, at ``arch``."""
    model = MaskedAutoencoder(arch, mask_ratio=0.75)
    return release.encoder_state_dict(model.state_dict(), arch)


# --- The tensor that is not in the state dict ---------------------------------


def test_the_state_dict_does_not_carry_pos_embed_and_the_export_does():
    """The whole premise. If this ever inverts, the export's extra work is stale, not safe."""
    model = MaskedAutoencoder(R5_ARCH, mask_ratio=0.75)
    assert "pos_embed" not in model.state_dict()
    assert "pos_embed" in release.encoder_state_dict(model.state_dict(), R5_ARCH)


@pytest.mark.parametrize(
    "arch,expected", [(VIT_S_ARCH, "vit_small_patch16_224"), (R5_ARCH, "vit_base_patch16_224")]
)
def test_a_strict_timm_load_accepts_the_export(arch, expected):
    assert release.assert_timm_loadable(an_encoder(arch), arch) == expected


def test_without_pos_embed_a_strict_load_refuses_and_a_lax_one_does_not():
    """The bug this module exists for, both halves, by mutation.

    ``strict=False`` is what a consumer writes, and it accepts the broken file silently — so the
    export's guarantee has to come from the strict load, and this pins that it actually would
    have caught the omission rather than passing for some other reason.
    """
    import timm

    broken = an_encoder(R5_ARCH)
    del broken["pos_embed"]

    with pytest.raises(RuntimeError, match="pos_embed"):
        release.assert_timm_loadable(broken, R5_ARCH)

    model = timm.create_model(
        "vit_base_patch16_224", in_chans=R5_ARCH.in_chans, **release.TIMM_CREATE_KWARGS
    )
    before = model.pos_embed.detach().clone()
    missing, unexpected = model.load_state_dict(broken, strict=False)
    assert list(missing) == ["pos_embed"] and list(unexpected) == []
    # The silent half: the load "succeeded" and the model kept its own random embedding.
    assert torch.equal(model.pos_embed.detach(), before)


# --- The axis order, which is per-run ------------------------------------------


def test_the_two_axis_orders_ship_different_tensors():
    """P102 §4: mmselfsup's ordering is MAE's transpose, and they differ by 2.0 — not by name."""
    r5 = an_encoder(R5_ARCH)["pos_embed"]
    r4 = an_encoder(R4_ARCH)["pos_embed"]
    assert r5.shape == r4.shape
    assert not torch.equal(r5, r4)
    assert (r5 - r4).abs().max().item() == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize("arch", [R5_ARCH, R4_ARCH])
def test_the_export_ships_the_runs_own_ordering_rather_than_a_default(arch):
    """A default-regenerated buffer would ship R4 under R5's ordering, silently."""
    expected = sincos2d_pos_embed(
        arch.encoder_dim, arch.img_px // arch.patch_px, axis_order=AXIS_ORDERS[arch.pos_encoding]
    )
    assert torch.equal(an_encoder(arch)["pos_embed"], expected)


def test_a_pos_embed_already_in_the_state_dict_is_refused():
    """Two sources for one tensor is how they come to disagree; the export takes one."""
    model = MaskedAutoencoder(R5_ARCH, mask_ratio=0.75)
    state = dict(model.state_dict())
    state["pos_embed"] = torch.zeros(1, 197, 768)
    with pytest.raises(ValueError, match="already in the state dict"):
        release.encoder_state_dict(state, R5_ARCH)


# --- The release boundary ------------------------------------------------------


def test_the_decoder_does_not_reach_the_export():
    """§14 ships encoder-only, and ``mask_token`` is the entry a prefix test gets wrong."""
    keys = set(an_encoder(R5_ARCH))
    assert not [k for k in keys if k.startswith("decoder_")]
    assert "mask_token" not in keys


def test_an_unregistered_encoder_shape_has_no_name_to_ship_under():
    huge = R5_ARCH.replace(encoder_dim=1024, encoder_depth=24, encoder_heads=16)
    with pytest.raises(ValueError, match="no timm model registered"):
        release.timm_model_name(huge)


# --- Provenance: recognized, never assumed -------------------------------------


def test_a_scratch_run_is_verified_without_any_file():
    prov = release.init_provenance(RunConfig.vit_base())
    assert prov.mode == "scratch" and prov.verified is True
    assert prov.spec is None and prov.licence is None


def test_a_warm_start_with_no_file_presented_makes_no_licence_claim():
    cfg = RunConfig.vit_base_continual("/rented/box/mae_1600e.pth")
    prov = release.init_provenance(cfg)
    assert prov.mode == "imagenet_mae"
    assert prov.verified is False
    assert prov.licence is None
    # The path is still reported: it names the file someone would produce to verify the claim.
    assert prov.checkpoint == "/rented/box/mae_1600e.pth"


def test_a_warm_start_from_an_unrecognized_file_makes_no_licence_claim(tmp_path):
    """P101's rule at the release boundary: the wrong file must not inherit Apache-2.0."""
    impostor = tmp_path / "not_the_checkpoint.pth"
    torch.save({"state_dict": {"backbone.cls_token": torch.zeros(1, 1, 768)}}, impostor)
    cfg = RunConfig.vit_base_continual(str(impostor))
    prov = release.init_provenance(cfg, init_file=impostor)
    assert prov.verified is False
    assert prov.spec is None and prov.licence is None
    assert "no provenance claim" in prov.note


# --- The manifest and the shipping entry point ---------------------------------


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A real checkpoint at the release architecture, exported through the CLI.

    Written by hand rather than trained: this suite is about what the export *reads and writes*,
    and #94's own tests already cover the payload being a faithful round trip.

    **Module-scoped deliberately.** A ViT-B payload is ~400 MB on disk and the five tests below
    all read the same export, so a function-scoped fixture would build and write it five times —
    enough to fail on a full-suite run for want of temporary space rather than for any reason to
    do with the code.
    """
    from pretrain.checkpoint import RunState, save_checkpoint
    from pretrain.fingerprint import DataFingerprint

    tmp_path = tmp_path_factory.mktemp("release")

    cfg = RunConfig.vit_base()
    model = MaskedAutoencoder(cfg.architecture, mask_ratio=cfg.masking.ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fingerprint = DataFingerprint(
        corpus_dir="data/corpus_500k",
        dataset_version="bcbd6484",
        n_tiles=500_000,
        selection_list_md5="0" * 32,
        split_md5="1" * 32,
        split_blocks_md5="2" * 32,
        split_state="committed",
        split_note=None,
        git_sha="deadbee",
    )
    ckpt = tmp_path / "checkpoint_e0400_s000092400_milestone.pt"
    save_checkpoint(
        ckpt,
        config=cfg,
        fingerprint=fingerprint,
        state=RunState(epoch=400, step=92_400),
        model=model,
        optimizer=optimizer,
        rng={},
    )
    out = tmp_path / "release"
    assert release.main([str(ckpt), str(out)]) == 0
    return out


def test_the_shipping_command_writes_both_artifacts(exported):
    """#106's lesson: a seam nobody is on both sides of is a feature that does not exist."""
    assert (exported / "encoder.pt").is_file()
    assert (exported / "manifest.json").is_file()


def test_the_exported_file_loads_strictly_into_timm(exported):
    """End to end: what landed on disk is what a consumer loads, not what was in memory."""
    import timm

    state = torch.load(exported / "encoder.pt", map_location="cpu", weights_only=True)
    model = timm.create_model("vit_base_patch16_224", in_chans=1, **release.TIMM_CREATE_KWARGS)
    model.load_state_dict(state, strict=True)  # raises on any missing or unexpected key


def test_the_manifest_names_the_ordering_and_the_model_a_consumer_types(exported):
    m = json.loads((exported / "manifest.json").read_text())
    assert m["timm_model"] == "vit_base_patch16_224"
    assert m["timm_create_kwargs"]["in_chans"] == 1
    assert m["pos_encoding"] == "sincos2d"


def test_the_manifest_reports_epochs_completed_not_the_zero_based_index(exported):
    """P88 §1: W&B's ``_step`` is the 0-based epoch index and ``RunState.epoch`` is completed
    epochs. The released file is 400 completed, and the manifest must not quote the other one."""
    m = json.loads((exported / "manifest.json").read_text())
    assert m["epochs_completed"] == 400
    assert m["steps"] == 92_400


def test_the_manifest_carries_the_fourteen_fingerprints_and_provenance(exported):
    m = json.loads((exported / "manifest.json").read_text())
    assert m["fingerprint"]["dataset_version"] == "bcbd6484"
    assert m["fingerprint"]["split_md5"] == "1" * 32
    assert m["fingerprint"]["git_sha"] == "deadbee"
    assert m["init_provenance"]["mode"] == "scratch"
    assert m["init_provenance"]["verified"] is True
    assert m["config_hash"] and m["config"]


def test_the_manifest_hashes_what_it_shipped(exported):
    from pretrain.init import checkpoint_md5

    m = json.loads((exported / "manifest.json").read_text())
    assert m["encoder_md5"] == checkpoint_md5(exported / "encoder.pt")
    assert m["encoder_tensors"] == 150
