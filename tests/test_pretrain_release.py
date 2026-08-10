"""§14's encoder-only export (#88).

The property under test throughout is the one from P88 §4: the export is **not** the encoder
half of ``state_dict()``, because ``pos_embed`` is a non-persistent buffer and that file loads
into timm reporting one missing key and keeping timm's *random* positional embedding. Several
tests here therefore assert on the failure directly — including by mutation, dropping the
materialized tensor and checking that the strict load is what refuses.
"""

from __future__ import annotations

import csv
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


def a_history(epochs=400, probe_every=20):
    """One run's W&B history, shaped like R4's and R5's actually are.

    The cadence is the real one and matters to the tests below: ``_step`` runs 0-based, and the
    probe fires at ``_step`` 0, 20, ... 380 — completed 1, 21, ... 381 — which is precisely the
    set that never lands on a milestone (#131). The media and histogram columns are here because
    the real history carries them and the curve must not.
    """
    rows = []
    for step in range(epochs):
        row = {
            "_step": step,
            "_runtime": 61.0 * step,
            "_timestamp": 1.7e9 + step,
            "train/loss": 1.0 / (step + 1),
            "train/lr": 1e-3 * (1 - step / epochs),
            "val/loss": 1.1 / (step + 1),
            "recon/hillshade": {"_type": "image-file", "path": f"media/{step}.png"},
            "val/per_patch_loss": {"_type": "histogram", "bins": [1, 2], "values": [3]},
        }
        if step % probe_every == 0:
            row["probe/set_md5"] = "f" * 32
            row["probe/stratum_acc"] = 0.5 + step / (4 * epochs)
        rows.append(row)
    return rows


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


# --- The licence the weights ship under ----------------------------------------


def test_a_scratch_run_ships_under_the_release_licence():
    lic = release.release_licence({"mode": "scratch", "verified": True, "spec": None})
    assert lic["spdx"] == release.RELEASE_LICENCE
    assert lic["inherited_from"] is None
    assert "derive from no other licensed work" in lic["note"]


def test_a_verified_apache_warm_start_carries_its_chain_forward():
    """#103 bought exactly this: an Apache-2.0 init means one set of terms across every arm."""
    lic = release.release_licence({
        "mode": "imagenet_mae", "verified": True,
        "spec": {"name": "OpenMMLab MAE ViT-B/16", "url": "https://x", "licence": "Apache-2.0",
                 "licence_source": "https://github.com/open-mmlab/mmselfsup/blob/v1.0.0/LICENSE"},
    })
    assert lic["spdx"] == "Apache-2.0"
    assert lic["inherited_from"]["licence"] == "Apache-2.0"
    assert lic["inherited_from"]["licence_source"].endswith("LICENSE")


@pytest.mark.parametrize(
    "provenance",
    [
        pytest.param({"mode": "imagenet_mae", "verified": False, "spec": None}, id="unverified"),
        pytest.param(
            {"mode": "imagenet_mae", "verified": True,
             "spec": {"name": "the official checkpoint", "licence": "CC-BY-NC-4.0"}},
            id="incompatible",
        ),
    ],
)
def test_an_unsupported_chain_claims_no_licence_rather_than_defaulting(provenance):
    """P101's rule at the last boundary it can be broken at.

    The whole point of that ticket is that a licence nobody established is not Apache-2.0 by
    default. A release that silently claimed it over weights derived from a CC-BY-NC checkpoint
    would look identical to this one in every listing — same files, same manifest shape.
    """
    lic = release.release_licence(provenance)
    assert lic["spdx"] is None and lic["file"] is None and lic["text_md5"] is None
    assert "no licence is claimed" in lic["note"]


def test_the_shipped_licence_text_is_the_canonical_one():
    text = release.licence_text()
    assert "Apache License" in text and "Version 2.0, January 2004" in text
    from pretrain.init import checkpoint_md5
    path = Path(release.__file__).parent / "licences" / f"{release.RELEASE_LICENCE}.txt"
    assert checkpoint_md5(path) == release.LICENCE_TEXT_MD5


def test_an_altered_licence_text_is_refused(tmp_path, monkeypatch):
    """The one shipped file whose *content* nothing else checks: a truncated licence looks like
    a licence in every listing, and the artifact would be making a legal claim about text that
    nobody read."""
    fake_pkg = tmp_path / "pretrain"
    (fake_pkg / "licences").mkdir(parents=True)
    (fake_pkg / "licences" / f"{release.RELEASE_LICENCE}.txt").write_text("Apache License, sort of")
    monkeypatch.setattr(release, "__file__", str(fake_pkg / "release.py"))
    with pytest.raises(ValueError, match="not the canonical"):
        release.licence_text()


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

    from pretrain.init import checkpoint_md5

    cfg = RunConfig.vit_base()
    model = MaskedAutoencoder(cfg.architecture, mask_ratio=cfg.masking.ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # The blocks file is written first so the fingerprint can record its *real* md5: the export
    # refuses a blocks file that is not the one the run trained against, and a fixture that
    # hard-coded the hash would be testing the refusal rather than the happy path.
    blocks = a_blocks_file(tmp_path / "split_blocks.json")
    fingerprint = DataFingerprint(
        corpus_dir="data/corpus_500k",
        dataset_version="bcbd6484",
        n_tiles=500_000,
        selection_list_md5="0" * 32,
        split_md5="1" * 32,
        split_blocks_md5=checkpoint_md5(blocks),
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
    # The earlier milestones as name-only stubs. :func:`release.milestone_set` reads the
    # *directory listing* — `discover` parses epoch and step out of the filename and opens
    # nothing — so a real 400 MB payload here would test torch.save, not the milestone set.
    for epoch, step in ((50, 11_550), (100, 23_100), (200, 46_200)):
        (tmp_path / f"checkpoint_e{epoch:04d}_s{step:09d}_milestone.pt").write_bytes(b"stub")
    (tmp_path / "checkpoint_e0399_s000092169.pt").write_bytes(b"stub")  # not a milestone

    history = tmp_path / "history.json"
    history.write_text(json.dumps(a_history()))

    out = tmp_path / "release"
    argv = [str(ckpt), str(out), "--history", str(history), "--split-blocks", str(blocks)]
    assert release.main(argv) == 0
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


# --- The curves, and the epoch axis they are indexed by ------------------------


def test_the_curve_is_indexed_by_epochs_completed_not_wandb_step():
    """P88 §1's off-by-one, which is the entire reason this conversion is not a rename.

    ``_step`` 0 is the first epoch *finished*, so it is 1 completed. Getting this backwards
    shifts every point in every released curve by one epoch, in a way nothing downstream can see.
    """
    rows = release.curve_rows(a_history(epochs=3))
    assert [r[release.EPOCH_AXIS] for r in rows] == [1, 2, 3]
    assert "_step" not in rows[0]


def test_the_curve_drops_media_and_histograms_but_keeps_the_probe_md5():
    """A CSV cell holds a series, not an image reference — but ``set_md5`` is what makes two
    runs' probe points comparable, and #132 leaned on it, so it survives."""
    rows = release.curve_rows(a_history(epochs=1))
    assert "recon/hillshade" not in rows[0]
    assert "val/per_patch_loss" not in rows[0]
    assert rows[0]["probe/set_md5"] == "f" * 32
    assert rows[0]["train/loss"] == 1.0


def test_the_curve_refuses_a_history_that_already_carries_the_axis():
    with pytest.raises(ValueError, match="two sources for the epoch axis"):
        release.curve_rows([{"_step": 0, release.EPOCH_AXIS: 7}])


def test_a_history_row_with_no_step_has_no_epoch_to_place_it_at():
    with pytest.raises(ValueError, match="no `_step`"):
        release.curve_rows([{"train/loss": 0.5}])


def test_probe_columns_stay_sparse_rather_than_carried_forward(tmp_path):
    """The probe fires every 20 epochs. Forward-filling would turn 19 inferences into what looks
    like 19 measurements — the same confusion #131 is about, one axis over."""
    rows = release.curve_rows(a_history(epochs=40))
    assert "probe/stratum_acc" not in rows[1]
    path = release.write_curves(rows, tmp_path / "curves.csv")
    body = list(csv.DictReader(path.read_text().splitlines()))
    assert body[0]["probe/stratum_acc"] != ""
    assert body[1]["probe/stratum_acc"] == ""  # empty cell, not the previous value


def test_the_csv_leads_with_the_epoch_axis(tmp_path):
    rows = release.curve_rows(a_history(epochs=5))
    path = release.write_curves(rows, tmp_path / "curves.csv")
    header = path.read_text().splitlines()[0].split(",")
    assert header[0] == release.EPOCH_AXIS
    assert header[1:] == sorted(header[1:])


def test_the_probe_epochs_are_the_ones_with_a_committed_set():
    """§10.3 treats a probe against no committed set as no probe, so presence of the md5 — not
    of a metric column — is what counts as having run."""
    rows = release.curve_rows(a_history())
    assert release.probe_epochs(rows) == list(range(1, 400, 20))


# --- The milestone set, and the probe number it must not carry -----------------


def test_only_milestones_ride_along(exported):
    m = json.loads((exported / "manifest.json").read_text())
    epochs = [x[release.EPOCH_AXIS] for x in m["milestones"]]
    assert epochs == [50, 100, 200, 400]  # the plain e0399 checkpoint is not in the release


def test_no_milestone_carries_a_probe_number(exported):
    """#131, structurally. The probe never ran at a milestone epoch, so there is no measurement
    to put here — and the null key exists to stop one being added later."""
    m = json.loads((exported / "manifest.json").read_text())
    assert m["milestones"], "nothing to assert against"
    assert all(x["probe"] is None for x in m["milestones"])
    assert "has ever been probed" in m["probe_note"]


def test_the_milestone_and_probe_epochs_genuinely_do_not_intersect(exported):
    """The premise under #131's rule. If a future cadence change makes these overlap, this fails
    and the null probe column becomes the wrong answer rather than a silent one."""
    m = json.loads((exported / "manifest.json").read_text())
    milestones = {x[release.EPOCH_AXIS] for x in m["milestones"]}
    assert milestones.isdisjoint(m["curves"]["probed_epochs"])


def test_each_milestone_says_how_far_the_nearest_real_probe_was(exported):
    """So that carrying a number forward is a stated inference rather than an unlabelled one."""
    m = json.loads((exported / "manifest.json").read_text())
    nearest = {x[release.EPOCH_AXIS]: x["nearest_probed_epoch"] for x in m["milestones"]}
    assert nearest == {50: 41, 100: 101, 200: 201, 400: 381}


# --- The curves as a shipped artifact ------------------------------------------


def test_the_release_ships_a_curve_file_and_names_its_axis(exported):
    m = json.loads((exported / "manifest.json").read_text())
    assert (exported / "curves.csv").is_file()
    assert m["curves"]["file"] == "curves.csv"
    assert m["curves"]["epoch_axis"] == "epochs_completed"
    assert m["curves"]["rows"] == 400
    assert "_step + 1" in m["curves"]["epoch_axis_note"]


def test_the_curve_file_is_hashed_like_every_other_artifact(exported):
    from pretrain.init import checkpoint_md5

    m = json.loads((exported / "manifest.json").read_text())
    assert m["curves"]["md5"] == checkpoint_md5(exported / "curves.csv")


def test_a_release_with_no_history_says_so_rather_than_shipping_an_unlabelled_curve(tmp_path):
    """§14 requires curves. An export without them is incomplete, and the manifest is where that
    has to be visible — a missing key would read as an oversight, ``None`` reads as a fact."""
    rows = release.curve_rows(a_history(epochs=2))
    assert rows  # the helper works; the point below is the absence
    manifest = release.ReleaseManifest(
        source_checkpoint="c", source_checkpoint_md5="0" * 32, epochs_completed=1, steps=1,
        config_hash="h", timm_model="vit_base_patch16_224", timm_create_kwargs={},
        pos_encoding="sincos2d", encoder_tensors=1, encoder_parameters=1, encoder_file="e.pt",
        encoder_md5="1" * 32, fingerprint={}, init_provenance={}, curves=None, milestones=[],
        evaluation_regions=None, probe_note=release.PROBE_NOTE, config={},
        licence=release.release_licence({"mode": "scratch"}),
    )
    assert json.loads(json.dumps(manifest.to_dict()))["curves"] is None


# --- The write-up, and the sentence that must not come back --------------------


def a_manifest(**overrides):
    """A manifest with every field filled, for card tests that need no 400 MB checkpoint."""
    fields = dict(
        source_checkpoint="ckpt.pt", source_checkpoint_md5="0" * 32, epochs_completed=400,
        steps=92_400, config_hash="h", timm_model="vit_base_patch16_224",
        timm_create_kwargs={"in_chans": 1, "num_classes": 0, "global_pool": ""},
        pos_encoding="sincos2d", encoder_tensors=150, encoder_parameters=85_254_144,
        encoder_file="encoder.pt", encoder_md5="1" * 32,
        fingerprint={"dataset_version": "bcbd6484", "n_tiles": 500_000, "git_sha": "deadbee"},
        init_provenance={"mode": "scratch", "verified": True, "spec": None, "note": "n"},
        curves=None, milestones=[], evaluation_regions=None,
        probe_note=release.PROBE_NOTE, config={},
    )
    fields.update(overrides)
    # Derived rather than defaulted, so a card test that overrides the provenance gets the
    # licence that provenance actually supports — the two must never be settable apart.
    fields.setdefault("licence", release.release_licence(fields["init_provenance"]))
    return release.ReleaseManifest(**fields)


@pytest.mark.parametrize("claim", release.RETIRED_CLAIMS)
def test_the_card_never_makes_a_claim_p100_retired(claim, exported):
    """P100 §6.3(1), enforced rather than reviewed.

    "First open-weights encoder pretrained on elevation alone" is **false** — an Apache-2.0
    ViT-B/16 on ~9 M single-channel Copernicus DEM patches has been downloadable since
    2025-02-20. The sentence is the kind that gets re-typed from memory by someone who last read
    the plan before the re-check, so the test asserts on the strings rather than trusting review.
    """
    card = (exported / "MODEL_CARD.md").read_text()
    body = card.split("## Related work")[0]
    # Present only inside the struck-through "these are false" list, never as an assertion.
    assert f'~~"{claim}"~~' in body
    assert body.count(claim) == 1


def test_the_card_makes_the_narrowed_claim_with_its_qualifiers(exported):
    """Each qualifier does real work: MIM is not FSQ-VAE tokenization, IBM publishes no transfer
    number for its tokenizer, and TerraMesh samples the Sentinel grid rather than terrain."""
    card = (exported / "MODEL_CARD.md").read_text()
    for qualifier in ("masked-reconstruction", "raw single-channel", "transferable backbone",
                      "terrain-stratified"):
        assert qualifier in card


def test_the_strongest_competitor_is_named_first(exported):
    """P100 §6.3(2): a reviewer will find TerraMind whether or not the card does, so the only
    ordering that survives is the one that leads with it."""
    card = (exported / "MODEL_CARD.md").read_text()
    related = card.split("## Related work")[1].split("## The comparison")[0]
    assert related.index("TerraMind") < related.index("DINO-CV")
    assert related.index("TerraMind") < related.index("terrain-diffusion-30m")


def test_related_work_carries_at_least_four_entries_each_with_a_link_and_a_licence():
    """The minimum P100 §6.3(2) sets, and the three things each entry owes a reader."""
    assert len(release.RELATED_WORK) >= 4
    for work in release.RELATED_WORK:
        assert work.url.startswith("https://")
        assert work.licence and work.what_it_is and work.differs


def test_the_card_states_the_comparison_it_is_not_making(exported):
    """P100 §6.3(3): left silent the gap reads as an oversight; stated, it is a scope boundary."""
    card = (exported / "MODEL_CARD.md").read_text()
    assert "does not make" in card
    assert "No head-to-head has been run" in card
    assert "Phase-2 benchmark" in card


def test_the_card_reads_its_numbers_off_the_manifest_rather_than_repeating_them(exported):
    """A hand-maintained card beside a generated manifest eventually describes a different
    artifact than the one in the directory."""
    m = json.loads((exported / "manifest.json").read_text())
    card = (exported / "MODEL_CARD.md").read_text()
    assert f"{m['encoder_parameters']:,}" in card
    assert f"{m['epochs_completed']} epochs completed" in card
    assert m["config_hash"] in card
    assert m["pos_encoding"] in card


def test_the_card_tells_a_consumer_to_load_strictly(exported):
    """The strict load is the artifact's whole guarantee, so the card must not hand a reader the
    ``strict=False`` line that silently accepts a broken export."""
    card = (exported / "MODEL_CARD.md").read_text()
    assert "strict=True" in card
    assert "randomly initialized" in card


def test_a_warm_started_card_names_the_licence_and_the_over_reading_caveat():
    """#86's line: R4's init is an independent reproduction trailing the official weights by
    0.9 pt, so a narrow gap must not be over-read in either direction."""
    card = release.model_card(a_manifest(init_provenance={
        "mode": "imagenet_mae", "verified": True, "checkpoint": "/x/mae.pth",
        "spec": {"name": "OpenMMLab MAE ViT-B/16", "licence": "Apache-2.0"}, "note": "n",
    }))
    assert "Apache-2.0" in card
    assert "0.9 pt" in card and "over-read" in card


def test_an_unverified_warm_start_card_claims_no_licence():
    """P101 at the write-up boundary, not just in the manifest."""
    card = release.model_card(a_manifest(init_provenance={
        "mode": "imagenet_mae", "verified": False, "checkpoint": "/x/mae.pth",
        "spec": None, "note": "n",
    }))
    assert "not presented to the export" in card
    assert "No URL, md5 or licence is claimed" in card


def test_a_card_with_no_curves_says_the_release_is_incomplete():
    assert "§14-incomplete" in release.model_card(a_manifest(curves=None))


def test_the_card_ships_beside_the_weights(exported):
    """It is a model card: it has to travel with the artifact, not live in the repo."""
    assert (exported / "MODEL_CARD.md").is_file()
    assert (exported / "MODEL_CARD.md").read_text().startswith("# DEMAE")


def test_the_release_states_the_licence_of_the_weights_it_ships(exported):
    """The gap the first export shipped with: it named every *neighbour's* licence in related
    work and the *init's* licence under provenance, and said nothing about the files in the
    directory — the one fact a consumer needs before using them."""
    m = json.loads((exported / "manifest.json").read_text())
    assert m["licence"]["spdx"] == release.RELEASE_LICENCE
    card = (exported / "MODEL_CARD.md").read_text()
    assert "## Licence" in card
    assert f"released under [{release.RELEASE_LICENCE}]" in card


def test_the_licence_text_ships_beside_the_weights_rather_than_as_a_link(exported):
    """A release directory pulled out of a bucket carries no repo with it, and Apache-2.0 §4(a)
    asks a redistributor to hand over the License itself."""
    shipped = exported / release.LICENCE_FILE
    assert shipped.is_file()
    from pretrain.init import checkpoint_md5
    m = json.loads((exported / "manifest.json").read_text())
    assert checkpoint_md5(shipped) == m["licence"]["text_md5"] == release.LICENCE_TEXT_MD5


@pytest.mark.parametrize("proper", ["Copernicus GLO-30", "MIT", "Apache-2.0"])
def test_the_licence_section_keeps_proper_nouns_cased(proper):
    """The first published version ran ``str.capitalize()`` on these sentences to fix a leading
    letter, which lowercases *everything after* it — so consumers were handed "the corpus is
    copernicus glo-30" and "this repo's source code is mit". A licence section that mangles the
    names of the things it is making claims about is not a licence section anyone should rely on,
    and nothing here reads the rendered text except a human who happens to open the file.
    """
    card = release.model_card(a_manifest(init_provenance={
        "mode": "imagenet_mae", "verified": True, "checkpoint": "/x/mae.pth",
        "spec": {"name": "OpenMMLab MAE ViT-B/16", "url": "https://x", "licence": "Apache-2.0",
                 "licence_source": "https://x/LICENSE"}, "note": "n",
    }))
    section = card.split("## Licence")[1].split("## Loading it")[0]
    assert proper in section


def test_a_release_with_no_supportable_licence_ships_no_licence_file(tmp_path):
    """The refusal has to reach the *directory*, not just the manifest: a LICENSE file sitting
    beside weights is the claim, whatever the JSON says."""
    from pretrain.checkpoint import RunState, save_checkpoint
    from pretrain.fingerprint import DataFingerprint

    cfg = RunConfig.vit_base_continual("/rented/box/mae_1600e.pth")  # warm start, no file given
    model = MaskedAutoencoder(cfg.architecture, mask_ratio=cfg.masking.ratio)
    ckpt = tmp_path / "checkpoint_e0001_s000000001.pt"
    save_checkpoint(
        ckpt, config=cfg,
        fingerprint=DataFingerprint(
            corpus_dir="c", dataset_version="v", n_tiles=1, selection_list_md5="0" * 32,
            split_md5="1" * 32, split_blocks_md5="2" * 32, split_state="committed",
            split_note=None, git_sha="g"),
        state=RunState(epoch=1, step=1), model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3), rng={})

    out = tmp_path / "unlicensed"
    manifest = release.export_encoder(ckpt, out, verify_timm=False)
    assert manifest.licence["spdx"] is None
    assert not (out / release.LICENCE_FILE).exists()
    card = (out / "MODEL_CARD.md").read_text()
    assert "No licence is claimed for these weights" in card


# --- The Phase-2 evaluation-regions contract ----------------------------------


def a_blocks_file(path, cells=("+000_-010", "+020_+060"), test_tiles=14_906):
    """A committed split-blocks document, shaped like `artifacts/phase1/split_blocks_500k.json`."""
    path.write_text(json.dumps({
        "schema_version": 1,
        "assignment_md5": "d" * 32,
        "n_tiles": 500_000,
        "totals": {"train": 475_030, "val": 9_998, "test": test_tiles},
        "shares": {"train": 0.950185, "val": 0.019999, "test": 0.029816},
        "test": {"block": "geocell_id", "n_cells": len(cells), "cells": list(cells)},
        "val": {"block": "source_cog", "n_cogs": 2, "cogs": ["a", "b"]},
        "per_stratum": {"train": [1] * 20, "val": [1] * 20, "test": [2] * 20},
    }) + "\n")
    return path


def test_the_regions_are_verified_against_the_runs_own_fingerprint(tmp_path):
    from pretrain.init import checkpoint_md5

    blocks = a_blocks_file(tmp_path / "blocks.json")
    fp = {"split_blocks_md5": checkpoint_md5(blocks)}
    regions = release.evaluation_regions(blocks, fp)
    assert regions["cells"] == ["+000_-010", "+020_+060"]
    assert regions["n_tiles"] == 14_906
    assert regions["assignment_md5"] == "d" * 32


def test_another_splits_regions_are_refused(tmp_path):
    """The failure this check exists for. A cell list from a different split looks entirely
    reasonable, and Phase 2 would build evaluation sets inside geocells this encoder trained on —
    every downstream number quietly optimistic, with nothing in the artifact to show it."""
    blocks = a_blocks_file(tmp_path / "blocks.json")
    with pytest.raises(ValueError, match="refusing to ship one split's evaluation regions"):
        release.evaluation_regions(blocks, {"split_blocks_md5": "9" * 32})


def test_a_run_that_recorded_no_blocks_md5_cannot_have_its_regions_shipped(tmp_path):
    """Unverifiable is worse than absent here, because absent stops a task builder and
    plausible-looking does not."""
    blocks = a_blocks_file(tmp_path / "blocks.json")
    with pytest.raises(ValueError, match="recorded no split_blocks_md5"):
        release.evaluation_regions(blocks, {"split_blocks_md5": None})


def test_the_release_ships_the_regions_as_their_own_file(exported):
    m = json.loads((exported / "manifest.json").read_text())
    assert (exported / "evaluation_regions.json").is_file()
    on_disk = json.loads((exported / "evaluation_regions.json").read_text())
    assert on_disk == m["evaluation_regions"]
    assert on_disk["split_blocks_md5"] == m["fingerprint"]["split_blocks_md5"]


def test_the_regions_contract_states_the_measured_separation_and_not_the_corrected_claim():
    """#77 measured what plan §9 asserted, and the first export shipped §9's sentence anyway.

    §9 says blocking at 10° "puts hundreds of km between train and test, far beyond terrain
    autocorrelation ranges". P77 §3 measured a **97 km median** and a **3.0 km minimum**, with
    12.8% of test tiles inside 25 km of training data — a whole-cell hold-out has a perimeter.
    The correction landed in the research memo and not in §9, so the export copied the wrong
    sentence into the one artifact outside consumers read, in the paragraph justifying why a
    "generalizes to unseen regions" claim is honest. Asserted on the string, because that is the
    kind of sentence a future edit re-types from the plan.
    """
    contract = release.REGIONS_CONTRACT
    assert "hundreds of" not in contract
    assert "97 km" in contract and "3.0 km" in contract and "12.8%" in contract
    # And the remedy is priced rather than left as an exercise, per P77's own costing.
    assert "2,964" in contract and "12,931" in contract


def test_the_card_carries_the_measured_separation_to_the_reader(exported):
    card = (exported / "MODEL_CARD.md").read_text()
    assert "hundreds of" not in card
    assert "median of 97 km" in card


def test_the_card_names_every_held_out_geocell(exported):
    """A consumer holding only the release must be able to tell which regions were held out —
    the md5 alone lets them check a blocks file, not read one."""
    m = json.loads((exported / "manifest.json").read_text())
    card = (exported / "MODEL_CARD.md").read_text()
    for cell in m["evaluation_regions"]["cells"]:
        assert f"`{cell}`" in card
    assert "never from `train` geocells" in card


def test_a_release_with_no_blocks_tells_a_task_builder_what_to_do_instead():
    card = release.model_card(a_manifest(evaluation_regions=None))
    assert "does not name" in card
    assert "check it against that hash" in card


# --- Whose curves are these? ---------------------------------------------------


def test_a_history_document_carries_its_own_provenance(tmp_path):
    """Both accepted shapes: a bare list, and a document that says which run it came from."""
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(a_history(epochs=2)))
    rows, meta = release.read_history_document(bare)
    assert len(rows) == 2 and meta == {}

    doc = tmp_path / "doc.json"
    doc.write_text(json.dumps({"config_hash": "abc", "run_id": "1o7fon6u",
                               "history": a_history(epochs=2)}))
    rows, meta = release.read_history_document(doc)
    assert len(rows) == 2
    assert meta["config_hash"] == "abc" and meta["run_id"] == "1o7fon6u"


def test_another_runs_curves_are_refused(tmp_path, exported):
    """A curve from a different run of the same project plots perfectly — right length, right
    columns, a loss that descends. Nothing about the figure would look wrong and it would be
    describing other weights."""
    from pretrain.checkpoint import RunState, save_checkpoint
    from pretrain.fingerprint import DataFingerprint

    cfg = RunConfig()  # ViT-S, so this stays cheap
    model = MaskedAutoencoder(cfg.architecture, mask_ratio=cfg.masking.ratio)
    ckpt = tmp_path / "checkpoint_e0001_s000000001.pt"
    save_checkpoint(
        ckpt, config=cfg,
        fingerprint=DataFingerprint(
            corpus_dir="c", dataset_version="v", n_tiles=1, selection_list_md5="0" * 32,
            split_md5="1" * 32, split_blocks_md5="2" * 32, split_state="committed",
            split_note=None, git_sha="g"),
        state=RunState(epoch=1, step=1), model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3), rng={})

    history = tmp_path / "other_run.json"
    history.write_text(json.dumps({"config_hash": "not-this-run", "history": a_history(epochs=2)}))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="refusing to ship one run's curves"):
        release.export_encoder(ckpt, out, history=history, verify_timm=False)

    # And it must leave nothing behind. `encoder.pt` is the first file written, it loads into
    # timm perfectly, and alone in a directory it carries no provenance at all — a refused
    # export that leaves one has produced the exact artifact this module exists to prevent.
    assert not out.exists() or list(out.iterdir()) == []


def test_a_refused_region_check_also_writes_nothing(tmp_path):
    """Same rule for the other refusal, so ordering cannot regress for one and not the other."""
    from pretrain.checkpoint import RunState, save_checkpoint
    from pretrain.fingerprint import DataFingerprint

    cfg = RunConfig()
    model = MaskedAutoencoder(cfg.architecture, mask_ratio=cfg.masking.ratio)
    ckpt = tmp_path / "checkpoint_e0001_s000000001.pt"
    save_checkpoint(
        ckpt, config=cfg,
        fingerprint=DataFingerprint(
            corpus_dir="c", dataset_version="v", n_tiles=1, selection_list_md5="0" * 32,
            split_md5="1" * 32, split_blocks_md5="2" * 32, split_state="committed",
            split_note=None, git_sha="g"),
        state=RunState(epoch=1, step=1), model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3), rng={})

    out = tmp_path / "out2"
    with pytest.raises(ValueError, match="refusing to ship one split's"):
        release.export_encoder(
            ckpt, out, split_blocks=a_blocks_file(tmp_path / "b.json"), verify_timm=False)
    assert not out.exists() or list(out.iterdir()) == []


def test_curves_with_no_declared_run_are_shipped_but_marked_unverified(exported):
    """The fixture writes a bare list, which is the shape a plain W&B dump has. It is usable and
    it cannot show which run it came from, so the release says exactly that rather than implying
    a link it has not checked."""
    m = json.loads((exported / "manifest.json").read_text())
    assert m["curves"]["config_hash"] is None
    assert m["curves"]["provenance"].startswith("UNVERIFIED")
    card = (exported / "MODEL_CARD.md").read_text()
    assert "not shown to belong to these weights" in card


def test_a_declared_matching_run_is_recorded_as_verified(tmp_path):
    card = release.model_card(a_manifest(curves={
        "file": "curves.csv", "md5": "a" * 32, "epoch_axis": "epochs_completed",
        "epoch_axis_note": "n", "rows": 400, "columns": ["a"], "probed_epochs": [1],
        "run_id": "1o7fon6u", "config_hash": "c993d65cdbbde5ea",
        "provenance": "verified: the history file declares this checkpoint's config hash",
    }))
    assert "not shown to belong to these weights" not in card
