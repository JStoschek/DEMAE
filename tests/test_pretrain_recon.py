"""§10.2's reconstruction grids: metres, triptychs, and the hillshade (#98).

Four things are on trial, and they fail in four different ways.

**The numbers on screen are metres.** Not per-tile-standardized units, not per-patch
normalized ones — and with `norm_pix` on there are *two* normalizations between the decoder's
output and an elevation, either of which can be forgotten or applied twice. So the inversion is
asserted against a tile whose true elevation range is known, and against a model whose
prediction *is* the target, which must come back as the truth panel exactly.

**The three panels are comparable.** A grid is read as a comparison — masked against predicted
against truth, this epoch against the last — so the colour ramp and the vertical exaggeration
are derived from the **truth** alone. Anything derived from the prediction would rescale the
picture whenever the model moved, which is precisely when a reader is trying to see a change.

**A smeared prediction has to look smeared.** That is the failure §10.2 exists to catch (the
corpus card §3.1 prediction ADR-0008 accepted). A per-panel contrast stretch would boost a
flat prediction's shading back up to full range and hide it, so the hillshade is unstretched
and the exaggeration is shared.

**It renders on garbage.** R0's pre-flight (§15 item 5) requires the pipeline to run on a
randomly initialized model, so nothing here may assume a sensible prediction: a NaN, an inf or
a 10,000-sigma output has to produce a viewable grid rather than an exception — and rather
than a black tile that reads as a reconstruction failure.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dataset import gallery
from pretrain import recon
from pretrain.panel import PanelTile, ResolvedPanel, draw_panel, write_panel
from pretrain.recon import ReconGrids, TileReconstruction
from pretrain.sink import Image, MemoryLogger
from pretrain.train import Trainer, build_loader, steps_per_epoch_for
from tests.conftest import build_corpus
from tests.test_pretrain_train import TILES_PER_COG, smooth_tile, tiny_config

PX = 224

#: Two val COGs so every stratum in the fixture has the two tiles a panel needs.
RECON_COGS = [
    ("Copernicus_DSM_COG_10_N45_00_E006_00_DEM", "+000_+040", "train"),
    ("Copernicus_DSM_COG_10_N45_00_E007_00_DEM", "+000_+040", "val"),
    ("Copernicus_DSM_COG_10_N46_00_E008_00_DEM", "+000_+040", "val"),
    ("Copernicus_DSM_COG_10_N55_00_E010_00_DEM", "+010_+050", "test"),
    ("Copernicus_DSM_COG_10_S01_00_E036_00_DEM", "+030_+000", "train"),
]

#: A known elevation range, in metres, for the inversion assertion. Both ends are far from
#: zero so an ocean-fill bug cannot pass, and the span is wide enough that a per-tile-
#: standardized field (which would sit within a few units of 0) could never be mistaken for it.
LOW_M, HIGH_M = 220.0, 1_460.0


def ramp_tile(rng: np.random.Generator, base: float) -> np.ndarray:
    """A tile whose true elevation runs from :data:`LOW_M` to :data:`HIGH_M`, and nothing else.

    ``base`` is ignored on purpose: the point of this fixture is that the metres on screen are
    a number the test knows in advance, so the tile cannot depend on which COG it came from.
    """
    y, x = np.mgrid[0:PX, 0:PX] / (PX - 1)
    field = LOW_M + (HIGH_M - LOW_M) * (0.5 * x + 0.5 * y)
    return field.astype(np.float32)


@pytest.fixture(scope="module")
def recon_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("recon_corpus")
    return build_corpus(root, RECON_COGS, tiles_per_cog=TILES_PER_COG, tile_fn=smooth_tile)


@pytest.fixture(scope="module")
def ramp_corpus(tmp_path_factory):
    """A corpus of tiles whose elevation range the test knows exactly."""
    root = tmp_path_factory.mktemp("ramp_corpus")
    return build_corpus(root, RECON_COGS, tiles_per_cog=TILES_PER_COG, tile_fn=ramp_tile)


def panel_for(corpus, tmp_path):
    """Commit a panel over a fixture corpus, and return the path to it."""
    import pyarrow.parquet as pq

    table = pq.read_table(corpus / "manifest.parquet", columns=["tile_id", "stratum_id", "split"])
    cols = {c: table[c].to_numpy(zero_copy_only=False) for c in table.column_names}
    path = tmp_path / "val_panel.csv"
    write_panel(draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"]), path)
    return path


def config_for(corpus, tmp_path, **changes):
    cfg = tiny_config(corpus, **changes)
    return cfg.replace(
        data=cfg.data.replace(val_panel_path=str(panel_for(corpus, tmp_path))),
        monitoring=cfg.monitoring.replace(wandb_mode="disabled"),
    )


def trainer_for(cfg):
    loader = build_loader(cfg, cfg.data.train_split)
    return Trainer.from_config(
        cfg, steps_per_epoch=steps_per_epoch_for(loader, cfg), device="cpu"
    )


def a_reconstruction(**changes) -> TileReconstruction:
    """A hand-built reconstruction, for the render tests that need no model at all."""
    y, x = np.mgrid[0:PX, 0:PX] / (PX - 1)
    truth = LOW_M + (HIGH_M - LOW_M) * (0.5 * x + 0.5 * y)
    hidden = np.zeros((PX, PX), dtype=bool)
    hidden[:32, :32] = True
    fields = {
        "tile_id": "T_0_0",
        "stratum_id": 3,
        "lat": 45.0,
        "truth": truth,
        "predicted": truth.copy(),
        "hidden": hidden,
        "land": np.ones((PX, PX), dtype=bool),
    }
    return TileReconstruction(**(fields | changes))


# --- Inverting to metres ------------------------------------------------------


def test_the_truth_panel_is_the_tiles_true_elevation_in_metres(ramp_corpus, tmp_path):
    """The acceptance criterion: asserted against a tile whose elevation range is known."""
    cfg = config_for(ramp_corpus, tmp_path)
    grids = ReconGrids.from_config(cfg)
    tiles = grids.reconstruct(trainer_for(cfg))

    for tile in tiles:
        # int16 storage quantizes to whole metres, so 1 m is the whole tolerance available.
        assert tile.truth.min() == pytest.approx(LOW_M, abs=1.0)
        assert tile.truth.max() == pytest.approx(HIGH_M, abs=1.0)


def test_a_model_that_predicts_the_target_reconstructs_the_truth_exactly(recon_corpus, tmp_path):
    """Both normalizations, inverted: `norm_pix`'s per-patch one and the loader's per-tile one.

    A prediction equal to the (normalized) target is the one case where the answer is known
    in closed form, which makes it the sharp test of the inversion. Off by a per-patch mean
    and the picture is still a plausible landscape — just not this tile's.
    """
    cfg = config_for(recon_corpus, tmp_path)
    grids = ReconGrids.from_config(cfg)
    tiles = grids.reconstruct(trainer_for(cfg), predict=recon.predict_the_target)

    for tile in tiles:
        assert tile.predicted == pytest.approx(tile.truth, abs=1e-2)


def test_the_inversion_follows_the_norm_pix_flag_rather_than_assuming_it(
    recon_corpus, tmp_path
):
    """R3 (#84) ablates `norm_pix` off, and then the per-patch rescale must not happen."""
    cfg = config_for(recon_corpus, tmp_path)
    off = cfg.replace(loss=cfg.loss.replace(norm_pix_loss=False))
    tiles = ReconGrids.from_config(off).reconstruct(
        trainer_for(off), predict=recon.predict_the_target
    )
    for tile in tiles:
        assert tile.predicted == pytest.approx(tile.truth, abs=1e-2)


def test_the_predicted_panel_keeps_the_visible_patches_and_fills_only_the_hidden_ones(
    recon_corpus, tmp_path
):
    cfg = config_for(recon_corpus, tmp_path)
    tiles = ReconGrids.from_config(cfg).reconstruct(trainer_for(cfg))

    for tile in tiles:
        visible = ~tile.hidden
        assert visible.any() and tile.hidden.any()
        assert tile.predicted[visible] == pytest.approx(tile.truth[visible], abs=1e-2)
        # A random decoder does not reproduce the terrain it never saw.
        assert not np.allclose(tile.predicted[tile.hidden], tile.truth[tile.hidden])


def test_the_hidden_patches_are_whole_patches_of_the_model_s_own_mask(recon_corpus, tmp_path):
    cfg = config_for(recon_corpus, tmp_path)
    tile = ReconGrids.from_config(cfg).reconstruct(trainer_for(cfg))[0]
    patch = cfg.architecture.patch_px
    blocks = tile.hidden.reshape(
        PX // patch, patch, PX // patch, patch
    ).transpose(0, 2, 1, 3)
    per_patch = blocks.reshape(-1, patch * patch)
    assert set(per_patch.sum(axis=1).tolist()) <= {0, patch * patch}
    assert int((per_patch.sum(axis=1) > 0).sum()) == cfg.masked_tokens


# --- The ramp and the exaggeration are the truth's ----------------------------


def test_the_colour_ramp_comes_from_the_truth_so_the_picture_does_not_rescale_when_the_model_moves():
    truth = a_reconstruction()
    wild = a_reconstruction(predicted=a_reconstruction().truth * 40.0 - 5_000.0)
    assert recon.metre_ramp(wild) == recon.metre_ramp(truth)


def test_the_vertical_exaggeration_comes_from_the_truth_too():
    truth = a_reconstruction()
    flattened = a_reconstruction(predicted=np.full((PX, PX), 900.0))
    assert recon.vertical_exaggeration(flattened) == recon.vertical_exaggeration(truth)


def test_a_smeared_prediction_shades_flatter_rather_than_being_stretched_back_to_full_range():
    """The failure §10.2 is built to catch, as a number: less relief must mean less contrast.

    The field carries fine-scale corrugation on a broad slope — a stand-in for the drainage
    texture ADR-0008's shortfall is about — and the "prediction" is that field with the
    texture blurred out of it, which is what smearing *is*. A per-panel contrast stretch would
    make the two panels indistinguishable; unstretched, the blurred one shades visibly flatter.
    """
    y, x = np.mgrid[0:PX, 0:PX] / (PX - 1)
    truth = 300.0 + 200.0 * x + 12.0 * np.sin(24 * np.pi * y) * np.cos(18 * np.pi * x)
    tile = a_reconstruction(truth=truth, predicted=gallery.smooth(truth, 6.0))
    ve = recon.vertical_exaggeration(tile)

    sharp = recon.shade(tile.truth, land=tile.land, lat=tile.lat, vert_exag=ve)
    smeared = recon.shade(tile.predicted, land=tile.land, lat=tile.lat, vert_exag=ve)
    assert smeared.std() < 0.5 * sharp.std()


# --- What the panels paint ----------------------------------------------------


def test_the_masked_panel_hides_exactly_the_patches_the_model_hid():
    tile = a_reconstruction()
    heat = recon.heat(tile.truth, ramp=recon.metre_ramp(tile), land=tile.land, hidden=tile.hidden)
    assert np.all(heat[tile.hidden] == np.asarray(recon.MASK_RGB))
    assert not np.any(np.all(heat[~tile.hidden] == np.asarray(recon.MASK_RGB), axis=-1))


def test_ocean_is_painted_rather_than_shaded():
    """A flat sea shades to a uniform grey that reads as land — the gallery's own argument."""
    land = np.ones((PX, PX), dtype=bool)
    land[:, :64] = False
    tile = a_reconstruction(land=land)
    for panel in (
        recon.heat(tile.truth, ramp=recon.metre_ramp(tile), land=land),
        recon.shade(tile.truth, land=land, lat=tile.lat, vert_exag=1.0),
    ):
        assert np.all(panel[~land] == np.asarray(gallery.OCEAN_RGB))


def test_the_ramp_ignores_ocean_so_a_coastal_tile_is_not_scaled_by_its_sea():
    land = np.ones((PX, PX), dtype=bool)
    land[:, :112] = False
    truth = a_reconstruction().truth.copy()
    truth[~land] = 0.0
    lo, _ = recon.metre_ramp(a_reconstruction(truth=truth, land=land))
    assert lo > LOW_M / 2  # the sea's zeros never entered the percentiles


# --- Garbage in, garbage rendered (§15 item 5) --------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_prediction_is_painted_rather_than_crashing_the_render(bad):
    predicted = a_reconstruction().predicted.copy()
    predicted[10:20, 10:20] = bad
    tile = a_reconstruction(predicted=predicted)
    heat, shade = recon.render_tile(tile)

    for image in (heat, shade):
        assert np.isfinite(image.data).all()
    # Painted where the values were unusable, so a renderer bug does not read as a model one.
    band = heat.data[:, PX + recon.SEPARATOR_PX :][10:20, 10:20]
    assert np.all(band == recon.to_uint8(np.asarray(recon.BAD_RGB)))


def test_an_absurdly_scaled_prediction_saturates_rather_than_rescaling_the_grid():
    tile = a_reconstruction(predicted=a_reconstruction().predicted * 1e6)
    heat, _ = recon.render_tile(tile)
    assert np.isfinite(heat.data).all()
    assert heat.data.min() >= 0 and heat.data.max() <= 255


# --- The triptych -------------------------------------------------------------


def test_a_triptych_is_three_panels_wide_in_masked_predicted_truth_order():
    tile = a_reconstruction()
    heat, shade = recon.render_tile(tile)
    width = 3 * PX + 2 * recon.SEPARATOR_PX
    for image in (heat, shade):
        assert image.data.shape == (PX, width, 3)

    # The right-hand panel is the truth, and the middle one is this model's exact copy of it.
    predicted = heat.data[:, PX + recon.SEPARATOR_PX : 2 * PX + recon.SEPARATOR_PX]
    truth = heat.data[:, 2 * (PX + recon.SEPARATOR_PX) :]
    assert np.array_equal(predicted, truth)
    # ... while the left-hand panel is the truth with holes in it.
    masked = heat.data[:, :PX]
    assert not np.array_equal(masked, truth)
    assert np.array_equal(masked[~tile.hidden], truth[~tile.hidden])


def test_every_image_is_something_the_sink_will_accept():
    heat, shade = recon.render_tile(a_reconstruction())
    for image in (heat, shade):
        assert isinstance(image, Image)
        Image(image.data, caption=image.caption)  # the sink's own validation, run again
        assert image.data.dtype == np.uint8


def test_the_caption_says_which_tile_which_stratum_and_what_the_panels_are():
    heat, shade = recon.render_tile(a_reconstruction())
    for image in (heat, shade):
        assert "T_0_0" in image.caption
        assert "stratum 3" in image.caption
        assert "masked" in image.caption and "truth" in image.caption
    assert "m" in heat.caption  # the magnitudes the grey ramp deliberately does not carry


# --- The whole thing, on an untrained model -----------------------------------


def test_the_whole_pipeline_runs_on_a_randomly_initialized_model(recon_corpus, tmp_path):
    """§15's checklist item 5, and #98's own acceptance criterion."""
    cfg = config_for(recon_corpus, tmp_path)
    grids = ReconGrids.from_config(cfg)
    values = grids.render(trainer_for(cfg))

    assert len(values["recon/heatmap"]) == len(grids.panel)
    assert len(values["recon/hillshade"]) == len(grids.panel)
    assert values["recon/tiles"] == len(grids.panel)
    assert all(isinstance(i, Image) for i in values["recon/heatmap"])


def test_the_grids_reach_the_sink_as_images_at_the_epoch_they_were_rendered_for(
    recon_corpus, tmp_path
):
    cfg = config_for(recon_corpus, tmp_path)
    logger = MemoryLogger()
    logger.log(ReconGrids.from_config(cfg).render(trainer_for(cfg)), step=10)
    assert len(logger.at(10)["recon/heatmap"]) == 8


def test_the_same_model_renders_the_same_grid_twice(recon_corpus, tmp_path):
    """Fixed panel *and* fixed masks: two epochs differ because the model did, or not at all."""
    cfg = config_for(recon_corpus, tmp_path)
    grids = ReconGrids.from_config(cfg)
    trainer = trainer_for(cfg)
    first, second = grids.render(trainer), grids.render(trainer)
    for a, b in zip(first["recon/heatmap"], second["recon/heatmap"]):
        assert np.array_equal(a.data, b.data)
    for a, b in zip(first["recon/hillshade"], second["recon/hillshade"]):
        assert np.array_equal(a.data, b.data)


def test_the_panel_renders_in_stratum_order_every_time(recon_corpus, tmp_path):
    cfg = config_for(recon_corpus, tmp_path)
    grids = ReconGrids.from_config(cfg)
    captions = [i.caption for i in grids.render(trainer_for(cfg))["recon/heatmap"]]
    strata = [int(c.split("stratum ")[1].split()[0]) for c in captions]
    assert strata == sorted(strata)


def test_rendering_leaves_the_model_training_and_takes_no_gradient(recon_corpus, tmp_path):
    cfg = config_for(recon_corpus, tmp_path)
    trainer = trainer_for(cfg)
    trainer.model.train()
    ReconGrids.from_config(cfg).render(trainer)

    assert trainer.model.training
    assert all(p.grad is None for p in trainer.model.parameters())


def test_the_render_draws_nothing_from_the_training_stream(recon_corpus, tmp_path):
    """Same Heisenberg rule as §10.1's val pass: measuring a run must not perturb it."""
    cfg = config_for(recon_corpus, tmp_path)
    grids, trainer = ReconGrids.from_config(cfg), trainer_for(cfg)

    torch.manual_seed(0)
    before = torch.rand(3)
    torch.manual_seed(0)
    grids.render(trainer)
    assert torch.equal(torch.rand(3), before)


def test_no_panel_artifact_means_no_grids_rather_than_a_dead_run(recon_corpus, tmp_path):
    """A missing panel is a pre-flight failure (§15 item 3), not a reason to stop training."""
    cfg = config_for(recon_corpus, tmp_path)
    absent = cfg.replace(data=cfg.data.replace(val_panel_path=str(tmp_path / "nope.csv")))
    assert ReconGrids.from_config(absent) is None


def test_the_grids_carry_the_committed_panels_digest(recon_corpus, tmp_path):
    """`resolve_panel` proves the panel fits *this* corpus; only the md5 proves two runs
    compared the same 40 tiles — which is what §10.2 rests on and no single run can check."""
    from dataset.materialize import file_md5

    cfg = config_for(recon_corpus, tmp_path)
    grids = ReconGrids.from_config(cfg)
    assert grids.panel_md5 == file_md5(cfg.data.val_panel_path)


def test_a_cadence_of_zero_switches_the_grids_off(recon_corpus, tmp_path):
    cfg = config_for(recon_corpus, tmp_path)
    off = cfg.replace(monitoring=cfg.monitoring.replace(recon_every_epochs=0))
    assert ReconGrids.from_config(off) is None


def test_a_panel_that_no_longer_describes_the_corpus_is_refused_at_construction(
    recon_corpus, tmp_path
):
    """The panel's checks run when a run starts, not when the first grid is due 10 epochs in."""
    cfg = config_for(recon_corpus, tmp_path)
    stale = tmp_path / "stale.csv"
    write_panel([PanelTile("Copernicus_DSM_COG_10_N45_00_E006_00_DEM_0_0", 0)] * 2, stale)
    with pytest.raises(ValueError):
        ReconGrids.from_config(cfg.replace(data=cfg.data.replace(val_panel_path=str(stale))))


def test_the_panel_is_read_once_rather_than_re_read_every_ten_epochs(recon_corpus, tmp_path):
    """40 tiles held in memory is 8 MB; re-reading them is 40 random reads into a 47 GB corpus."""
    cfg = config_for(recon_corpus, tmp_path)
    grids = ReconGrids.from_config(cfg)
    first = [s["image"].clone() for s in grids.samples]
    grids.render(trainer_for(cfg))
    assert all(torch.equal(a, s["image"]) for a, s in zip(first, grids.samples))


def test_the_cadence_fires_on_the_tenth_epoch_and_on_the_untrained_zeroth(
    recon_corpus, tmp_path
):
    cfg = config_for(recon_corpus, tmp_path)
    grids = ReconGrids.from_config(cfg)
    assert [e for e in range(31) if grids.due(e)] == [0, 10, 20, 30]


def test_an_odd_tile_count_pads_the_contact_sheet_rather_than_cropping_half_of_it():
    """Cropping every row to the shortest one throws away a whole column, silently."""
    images = [recon.render_tile(a_reconstruction())[0] for _ in range(3)]
    sheet = recon.contact_sheet(images, columns=2)
    wide = images[0].data.shape[1]

    assert sheet.shape == (2 * PX, 2 * wide, 3)
    assert np.array_equal(sheet[:PX, wide:], images[1].data)  # the full row survives whole
    assert np.array_equal(sheet[PX:, :wide], images[2].data)  # and the short row is padded


# --- The entrypoint -----------------------------------------------------------


def test_the_entrypoint_renders_the_panel_on_an_untrained_model_and_writes_the_sheets(
    recon_corpus, tmp_path, capsys
):
    """§15's checklist item 5, as the command a person actually walks it in."""
    import json

    cfg = config_for(recon_corpus, tmp_path)
    config_file = tmp_path / "run.json"
    config_file.write_text(json.dumps(cfg.to_dict()))

    assert recon.main(["--config", str(config_file), "--out", str(tmp_path / "sheets")]) == 0
    out = capsys.readouterr().out
    assert "DEMAE Phase 1" in out
    assert "8 tiles over 4 strata" in out
    for name in ("heatmap", "hillshade"):
        assert (tmp_path / "sheets" / f"recon_panel_{name}.png").exists()


def test_the_entrypoint_says_so_rather_than_crashing_when_no_panel_is_committed(
    recon_corpus, tmp_path, capsys
):
    import json

    cfg = config_for(recon_corpus, tmp_path)
    absent = cfg.replace(data=cfg.data.replace(val_panel_path=str(tmp_path / "nope.csv")))
    config_file = tmp_path / "run.json"
    config_file.write_text(json.dumps(absent.to_dict()))

    assert recon.main(["--config", str(config_file)]) == 1
    assert "No panel to render" in capsys.readouterr().out


def test_a_resolved_panel_of_one_tile_still_renders(recon_corpus, tmp_path):
    """The panel is data; nothing here may assume 40 of anything."""
    cfg = config_for(recon_corpus, tmp_path)
    grids = ReconGrids.from_config(cfg)
    one = ResolvedPanel(
        tiles=grids.panel.tiles[:1], rows=grids.panel.rows[:1], lat=grids.panel.lat[:1]
    )
    trimmed = ReconGrids(
        samples=grids.samples[:1], panel=one, normalizer=grids.normalizer,
        mask_seed=grids.mask_seed, cadence=grids.cadence, chunk=grids.chunk,
    )
    assert len(trimmed.render(trainer_for(cfg))["recon/heatmap"]) == 1
