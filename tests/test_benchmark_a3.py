"""A3's fine-tune, over a real MAE (#166).

`tests/test_benchmark_composite.py` holds the half of §4.2 that is arithmetic. This is the half
that needs a model: the mask sampler that swaps §3.4's voids in for MAE's random 75%, the
inference composite driven through an actual encoder and decoder, and the pass that trains them.

The model is small but **shaped like the real one** — 224 px tiles, 16 px patches, a 196-token
grid and R5's own 0.75 mask ratio — because every property under test is a property of that
geometry. Its width and depth are not, so they are tiny.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from benchmark import finetune, zeroshot
from benchmark.artifacts import EVAL_MASKS_NAME, TEST_TILES_NAME
from benchmark.config import BenchmarkConfig, Finetune

TILE = 224
PATCH = 16
GRID = TILE // PATCH


@pytest.fixture
def model():
    """A real `MaskedAutoencoder` at the benchmark's geometry and a toy width."""
    from pretrain.config import Architecture, Loss, RunConfig
    from pretrain.model import MaskedAutoencoder

    cfg = RunConfig().replace(
        architecture=Architecture(
            img_px=TILE,
            patch_px=PATCH,
            encoder_dim=32,
            encoder_depth=2,
            encoder_heads=4,
            decoder_dim=16,
            decoder_depth=1,
            decoder_heads=2,
        ),
        loss=Loss(norm_pix_loss=True),
    )
    built = MaskedAutoencoder.from_config(cfg)
    built.eval()
    return built


def _void(top=64, left=64, size=64) -> np.ndarray:
    out = np.zeros((TILE, TILE), dtype=bool)
    out[top : top + size, left : left + size] = True
    return out


# --- §4.2's mask sampler ------------------------------------------------------


def test_the_void_is_always_hidden_and_the_budget_is_pretraining_s(model):
    """The recipe, as two assertions: every void patch is dropped, and 49 tokens survive.

    49 is not a number this module picked — it is `model.n_visible`, off the checkpoint's own
    mask ratio. That is what makes §7's costing ("an A3 epoch *is* a pretraining epoch") a fact
    about the arm rather than a hope about it.
    """
    void = _void()
    keep = finetune.finetune_keep_set(
        void, patch_px=PATCH, n_visible=model.n_visible, rng=np.random.default_rng(0)
    )

    assert keep.shape == (model.n_patches,)
    assert int(keep.sum()) == model.n_visible
    touched = zeroshot.touched_patches(void, patch_px=PATCH).reshape(-1)
    assert not keep[touched].any()


def test_the_keep_count_is_uniform_across_wildly_different_voids(model):
    """What makes the batch possible at all — `keep_masking` gathers at one width.

    §3.4's bank spans ~2 to ~108 touched patches. A sampler whose keep count tracked the void
    would make a 128-tile batch into eighty forward passes of one or two tiles.
    """
    rng = np.random.default_rng(1)
    counts = {
        int(
            finetune.finetune_keep_set(
                _void(size=size), patch_px=PATCH, n_visible=model.n_visible, rng=rng
            ).sum()
        )
        for size in (16, 48, 96, 144)
    }

    assert counts == {model.n_visible}


def test_the_survivors_are_scattered_rather_than_hugging_the_void(model):
    """§4.2's *"keeps seeing global context"*, as a property of where the kept tokens are.

    The top-up is drawn uniformly over the patches the void does not touch, so the visible set
    covers the tile. A sampler that kept the void's neighbours would give the encoder a local
    extrapolation problem rather than pretraining's global one.
    """
    keep = finetune.finetune_keep_set(
        _void(), patch_px=PATCH, n_visible=model.n_visible, rng=np.random.default_rng(2)
    ).reshape(GRID, GRID)

    rows = np.flatnonzero(keep.any(axis=1))
    cols = np.flatnonzero(keep.any(axis=0))
    assert rows.min() < 3 and rows.max() > GRID - 4
    assert cols.min() < 3 and cols.max() > GRID - 4


def test_a_void_larger_than_the_mask_budget_is_refused(model):
    """A void touching more patches than pretraining hides has no budget left to top up.

    §3.2's 40% area cap keeps every drawn void far below this — the bank's largest touches 108
    of 196 against a 147 budget — so this is a refusal about a draw the generator should never
    produce, said rather than absorbed into a silently smaller keep-set.
    """
    everything = np.ones((TILE, TILE), dtype=bool)

    with pytest.raises(ValueError, match="more patches than"):
        finetune.finetune_keep_set(
            everything, patch_px=PATCH, n_visible=model.n_visible, rng=np.random.default_rng(3)
        )


def test_the_sampler_is_a_function_of_its_seed(model):
    """Two draws from one seed agree; two seeds do not. §3.4's determinism, one level down."""
    args = dict(patch_px=PATCH, n_visible=model.n_visible)
    first = finetune.finetune_keep_set(_void(), rng=np.random.default_rng(7), **args)
    again = finetune.finetune_keep_set(_void(), rng=np.random.default_rng(7), **args)
    other = finetune.finetune_keep_set(_void(), rng=np.random.default_rng(8), **args)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


# --- The composite, driven through a real encoder and decoder -----------------


@pytest.fixture
def tile():
    """A smooth int16 tile in metres — terrain with no patch grid of its own."""
    y, x = np.mgrid[0:TILE, 0:TILE].astype(np.float64)
    return np.rint(600.0 + 90.0 * np.sin(y / 50.0) + 60.0 * np.cos(x / 37.0)).astype(np.int16)


def test_the_composite_returns_metres_and_leaves_the_terrain_alone(model, tile):
    """The two contracts every filler in this benchmark is held to (§8's first gate's shape)."""
    void = _void()
    out = finetune.Compositor(model=model, recipe=Finetune(), masks_cfg=BenchmarkConfig().masks)

    filled = out.infill(tile, void)

    assert filled.dtype == np.float64
    np.testing.assert_array_equal(filled[~void], tile.astype(np.float64)[~void])


def test_the_crossfade_runs_four_grids_and_the_shifts_only_for_large_voids(model, tile):
    """#166's third and fourth criteria, counted at the seam where the decodes happen.

    A small void pays four decodes — the cross-fade's own grids. A near-tile-scale one pays
    sixteen: §4.2's four shifted encodings, each cross-faded.
    """
    compositor = finetune.Compositor(
        model=model, recipe=Finetune(), masks_cfg=BenchmarkConfig().masks
    )

    assert compositor.decodes_for(_void(size=48)) == 4
    assert compositor.decodes_for(_void(top=40, left=40, size=150)) == 16


def test_the_composite_suppresses_the_patch_grid_a_single_decode_prints(model, tile):
    """#166's fifth criterion, through the model rather than over synthetic fields.

    An untrained decoder has no reason to make neighbouring patches agree, so its single-decode
    reconstruction prints the 16 px grid loudly — which is exactly the artifact §4.2's cross-fade
    exists for. The claim is made on `grid_energy`, a direct reading of the grid, **not** on §5's
    seam-gap column: #166 asks that the composite suppress the grid rather than that some other
    column absorb it, and a column cannot be its own evidence.
    """
    void = _void(top=48, left=48, size=112)
    compositor = finetune.Compositor(
        model=model, recipe=Finetune(), masks_cfg=BenchmarkConfig().masks
    )

    single = compositor.infill(tile, void, crossfade=False)
    faded = compositor.infill(tile, void)

    inner = np.zeros_like(void)
    inner[64:144, 64:144] = True  # well inside the void: no void boundary in this window
    before = finetune.grid_energy(single[64:144, 64:144], patch_px=PATCH)
    after = finetune.grid_energy(faded[64:144, 64:144], patch_px=PATCH)

    assert before > 0.0
    assert after < before / 2.0


def test_a_void_reaching_the_tile_edge_is_refused(model, tile):
    """The shifted grids are cut from a reflect-padded tile, so they lose an 8 px border.

    A 224 px window of a 224 px tile, shifted by half a patch, has to give up 8 px somewhere —
    here the far edge, since the shifted window starts 8 px earlier. §3.2's `edge_margin_px` is
    8, so every drawn void clears it with exactly zero slack, and a void that did not would be
    blended from a field that does not cover it. Refused rather than quietly filled from the
    native grid alone.
    """
    edged = np.zeros((TILE, TILE), dtype=bool)
    edged[TILE - 40 :, TILE - 40 :] = True
    compositor = finetune.Compositor(
        model=model, recipe=Finetune(), masks_cfg=BenchmarkConfig().masks
    )

    with pytest.raises(ValueError, match="edge"):
        compositor.infill(tile, edged)


# --- The recipe -----------------------------------------------------------------


def test_the_recipe_is_4_2_s_and_norm_pix_is_off():
    """§4.2's numbers, in the config rather than in the loop that reads them."""
    recipe = BenchmarkConfig().finetune

    assert (recipe.epochs, recipe.encoder_lr, recipe.decoder_lr) == (20, 1e-4, 1.5e-3)
    assert recipe.layer_decay == 0.75
    assert recipe.norm_pix_loss is False
    assert recipe.loss == "mse"


def test_layer_wise_decay_scales_the_encoder_and_leaves_the_decoder_alone(model):
    """§4.2's *lr 1e-4 (encoder) with 0.75 layer-wise decay, 1.5e-3 (decoder)*.

    Three claims: the decoder trains at its own flat rate, the encoder's blocks are ordered
    lowest-rate-first, and the top of the encoder reaches its declared 1e-4.
    """
    recipe = Finetune()
    groups = finetune.parameter_groups(model, recipe)

    decoder = [g for g in groups if g["part"] == "decoder"]
    encoder = sorted(
        (g for g in groups if g["part"] == "encoder"), key=lambda g: g["layer"]
    )
    assert decoder and {g["lr"] for g in decoder} == {recipe.decoder_lr}
    assert [g["lr"] for g in encoder] == sorted(g["lr"] for g in encoder)
    assert encoder[-1]["lr"] == pytest.approx(recipe.encoder_lr)
    assert encoder[0]["lr"] == pytest.approx(
        recipe.encoder_lr * recipe.layer_decay ** (len(model.blocks) + 1)
    )


# --- §4.2's recipe guard ------------------------------------------------------


def test_a_cell_with_a_moved_finetune_knob_is_refused():
    """The gate A3 fell through until #166: `tune.assert_frozen` never sees this arm.

    Asked of `--dry-run`, which is where it has to bite — a check inside the pass would let a
    pre-flight print the moved rate under §4.2's words and exit 0.
    """
    from benchmark import launch

    with pytest.raises(ValueError, match="drifted from the declared line"):
        launch.main(
            ["run", "--arm", "A3-R5", "--dry-run", "--set", "finetune.decoder_lr=1e-2"]
        )


def test_the_declared_recipe_is_one_object_both_inits_load():
    """`TUNED_HEAD`'s identity rule, one arm along: R5 and R4 load the same recipe, not a copy."""
    from benchmark.config import DECLARED_FINETUNE

    assert BenchmarkConfig().finetune is DECLARED_FINETUNE


# --- The arm reaches the run path ---------------------------------------------


def test_a3_names_its_pretraining_checkpoint_in_the_resolution_line():
    """#173's second blocker, as a pre-flight rather than a surprise.

    §14 exports encoder-only weights, so the `encoder.pt` A4 initializes from cannot start an arm
    that trains a decoder. A3 answers ``False`` to `runs_pretrained_decoder` — it initializes a
    decoder rather than running one — so every gate keyed on that property used to skip it, and
    the run would have reached `load_model` before anything said which file was missing.
    """
    from benchmark import arms, artifacts

    resolution = artifacts.resolve(BenchmarkConfig(), arms.get("A3-R5"))

    assert resolution.mae_checkpoint is not None
    assert "demae_r5_mae_checkpoint" in resolution.mae_checkpoint.absent_note
    assert arms.get("A3-R5").reads_mae_checkpoint
    assert arms.get("A2-R5").reads_mae_checkpoint


def test_a3_reads_the_train_mask_bank():
    """#166's second criterion, at the resolution line: the same file the frozen arms read."""
    from benchmark import arms, artifacts

    assert artifacts.resolve(BenchmarkConfig(), arms.get("A3-R5")).reads("mask_bank")


def test_scoring_a3_without_a_finetune_is_refused():
    """A pass handed no weights would score the *pretrained* decoder under A3's name — A2's row."""
    from benchmark import arms, evaluate

    with pytest.raises(NotImplementedError, match="handed no fine-tune"):
        evaluate.FILLERS["token_drop"](BenchmarkConfig(), arms.get("A3-R5"), trained=None)


# --- A3 end to end, through the shipping command ------------------------------
#
# The seam `tests/test_benchmark_a1.py` and `tests/test_benchmark_a2.py` hold, one arm along:
# `python -m benchmark.launch run` fine-tunes A3 against §3.4's committed bank and scores it
# through §4.2's composite, and the rows and the run record come out the other side.
#
# The model is the real ViT-B shape, because `benchmark.zeroshot.load_inpainter` checks the
# registry's declared width, depth and `pos_embed` axis order against the file — a stand-in of
# another shape would test the wiring around the refusals instead of the refusals. Its decoder is
# one narrow block and the cell runs one epoch over four tiles, which is what keeps that
# affordable.

import shutil

import pyarrow.parquet as pq

from benchmark import masks, metrics, results
from benchmark.artifacts import TILE_LIST_NAMES, VAL_TILES_NAME
from benchmark.config import Finetune as FinetuneRecipe
from pretrain.artifact import write_tile_strata
from tests.conftest import build_corpus
from tests.test_benchmark_a1 import _slope_mean
from tests.test_benchmark_a2 import _write_checkpoint

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMMITTED = REPO / "artifacts" / "task1"
PLAN = REPO / "docs" / "research" / "task1-void-fill-plan.md"
E2E_COGS = [
    ("Copernicus_DSM_COG_10_N45_00_E006_00_DEM", "+000_+040", "train"),
    ("Copernicus_DSM_COG_10_N46_00_E008_00_DEM", "+000_+040", "val"),
    ("Copernicus_DSM_COG_10_N47_00_E009_00_DEM", "+000_+040", "test"),
]
E2E_TILES_PER_COG = 2
ARM = "A3-R5"


@pytest.fixture
def declared(monkeypatch):
    """Stand in a test-scale §4.2 recipe — `assert_declared`'s own legitimate move, simulated.

    `tests/test_benchmark_head.py::frozen`'s counterpart, and it exists for the same reason: the
    guard refuses a moved knob, so the honest route to a seam-1 test is to move the declared
    line, which is what the refusal itself tells a build session to do. Twenty epochs of a ViT-B
    over a fixture corpus is not a slower suite; it is no seam-1 test of the arm this ticket
    ships.
    """
    from benchmark import config as config_module

    monkeypatch.setattr(
        config_module, "DECLARED_FINETUNE", FinetuneRecipe(epochs=1), raising=True
    )


@pytest.fixture
def e2e_box(tmp_path):
    """A frozen box: a corpus, §3.1's lists, the eval masks, the bank, and a **pretraining** file."""
    corpus = build_corpus(
        tmp_path / "corpus", E2E_COGS, tiles_per_cog=E2E_TILES_PER_COG, metrics_fn=_slope_mean
    )
    directory = tmp_path / "artifacts"
    directory.mkdir()

    table = pq.read_table(corpus / "manifest.parquet", columns=["tile_id", "source_cog"])
    split_of = {stem: split for stem, _, split in E2E_COGS}
    tiles: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for tile_id, stem in zip(
        table.column("tile_id").to_pylist(), table.column("source_cog").to_pylist()
    ):
        tiles[split_of[stem]].append(tile_id)
    for split in tiles:
        tiles[split].sort()

    for name in TILE_LIST_NAMES.values():
        write_tile_strata([(t, 0) for t in tiles["train"]], directory / name)
    write_tile_strata([(t, 0) for t in tiles["val"]], directory / VAL_TILES_NAME)
    write_tile_strata([(t, 0) for t in tiles["test"]], directory / TEST_TILES_NAME)

    base = BenchmarkConfig()
    cfg = BenchmarkConfig(
        paths=base.paths.replace(
            artifacts_dir=str(directory),
            corpus_dir=str(corpus),
            mask_bank_path=str(tmp_path / "train_mask_bank.parquet"),
        ),
        masks=base.masks.replace(bank_draws_per_tile=2),
        tiles=base.tiles.replace(test=len(tiles["test"])),
    )
    masks.write_eval_masks(
        masks.eval_rows(tiles["test"]),
        directory / EVAL_MASKS_NAME,
        artifact_hash=cfg.artifact_hash,
        real=masks.RealFamily.none(),
    )
    masks.freeze_bank(cfg)
    for name in ("task1-prereg-bar.json", metrics.GOLDEN_ARRAYS, metrics.GOLDEN_SIDECAR):
        shutil.copy(COMMITTED / name, directory / name)

    _write_checkpoint(tmp_path / "r5-pretrain.pt")
    tmp_path.joinpath("splits.json").write_text(json.dumps(tiles))
    return tmp_path


@pytest.fixture
def e2e_flags(e2e_box):
    splits = json.loads((e2e_box / "splits.json").read_text())
    return [
        "--set", f'paths.corpus_dir="{e2e_box / "corpus"}"',
        "--set", f'paths.artifacts_dir="{e2e_box / "artifacts"}"',
        "--set", f'paths.results_path="{e2e_box / "results.parquet"}"',
        "--set", f'paths.mask_bank_path="{e2e_box / "train_mask_bank.parquet"}"',
        "--set", f'paths.demae_r5_mae_checkpoint="{e2e_box / "r5-pretrain.pt"}"',
        "--set", f'paths.plan_path="{PLAN}"',
        "--set", f'runtime.checkpoint_dir="{e2e_box / "finetunes"}"',
        "--set", 'runtime.device="cpu"',
        "--set", "runtime.loader_workers=0",
        "--set", "masks.bank_draws_per_tile=2",
        "--set", f"tiles.test={len(splits['test'])}",
        "--set", "finetune.epochs=1",
        "--set", "head.batch_size=2",
    ]


@pytest.fixture
def a3_scored(declared, e2e_box, e2e_flags, capsys):
    from benchmark import launch

    assert launch.main(["run", "--arm", ARM, *e2e_flags]) == 0
    return e2e_box, capsys.readouterr().out


def test_a3_rows_land_in_the_parquet(a3_scored):
    """#166's first criterion, at fixture scale: one row per committed eval instance."""
    box, _ = a3_scored
    table = pq.read_table(box / "results.parquet")
    rows = table.take(
        [i for i, name in enumerate(table.column("arm").to_pylist()) if name == ARM]
    )

    assert rows.num_rows > 0
    assert rows.num_rows == pq.read_table(box / "artifacts" / EVAL_MASKS_NAME).num_rows
    assert set(rows.column("init").to_pylist()) == {"demae-r5"}


def test_the_run_record_says_a_fine_tune_produced_the_weights(a3_scored):
    """The training block, and the two numbers §6 item 4 prints for this arm.

    A3 trains every parameter it has, so its pair is equal — A7's shape, and for A7's reason.
    The registry cannot derive it (§2.2's decoder shape is undeclared), so the number here is the
    one the loaded model reported.
    """
    box, _ = a3_scored
    record = [
        r
        for r in results.read_run_records(results.run_record_path(box / "results.parquet"))
        if r["cell"]["arm"] == ARM
    ][-1]

    assert record["training"]["kind"] == "finetune"
    assert record["training"]["epochs_run"] == 1
    assert record["parameters"]["total"] == record["parameters"]["trainable"] > 0
    assert record["training"]["parameters"] == record["parameters"]["total"]


def test_a3_trains_against_the_committed_bank(a3_scored):
    """#166's second criterion, end to end: the bank is among the files the pass read."""
    box, out = a3_scored
    record = [
        r
        for r in results.read_run_records(results.run_record_path(box / "results.parquet"))
        if r["cell"]["arm"] == ARM
    ][-1]

    assert any("head_train" in source for source in record["training"]["read"])
    assert "mask bank" in out
