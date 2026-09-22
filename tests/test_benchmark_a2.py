"""A2 end to end through `python -m benchmark.launch run` (#167).

The same seam `tests/test_benchmark_a1.py` holds, one arm along: the shipping command scores
DEMAE zero-shot over the committed eval mask list, lands the rows, and writes a run record that
carries **R12's disposition** — no C-claim, the A2→A3 difference as a joint quantity, and the
majority-of-bands comparison against A1 with its outcome.

The fixture's checkpoint is a **real pretraining checkpoint of a real ViT-B-shaped MAE**, written
by `pretrain.checkpoint`'s own writer. That is not gold-plating: what #167 had to get right is
the join between a void, a token grid and a decoder, and the registry's declared width, depth and
`pos_embed` axis order are checked against the file on the way in — so a stand-in of another
shape would test the wiring around the refusals instead of the refusals. Its decoder is one
narrow block, because the decoder's shape is the only part of §2.2 the registry does not declare
and 85M encoder parameters already dominate the second this costs.

Its `norm_pix_loss` is **off** by default, which is §4.2's regime rather than R5's. That was the ticket's
finding and it is asserted here as a refusal rather than papered over: R5 and R4 were both
pretrained with it on, and a `norm_pix` decoder has no inversion to metres that does not read the
answer out of the void. See `benchmark.zeroshot` and `test_a_norm_pix_checkpoint_is_refused`.

No score's magnitude is asserted, on the A1 file's rule: R12 exists to leave A2's ordering
against A1 open, and a test that pinned it would close the question the register is about.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from benchmark import arms, launch, masks, metrics, results, zeroshot
from benchmark.artifacts import EVAL_MASKS_NAME, TEST_TILES_NAME
from benchmark.config import BenchmarkConfig
from pretrain.artifact import write_tile_strata
from tests.conftest import build_corpus
from tests.test_benchmark_a1 import TEST_COGS, TILES_PER_COG, _slope_mean

REPO = Path(__file__).resolve().parents[1]
COMMITTED = REPO / "artifacts" / "task1"
PLAN = REPO / "docs" / "research" / "task1-void-fill-plan.md"

ARM = "A2-R5"


def _write_checkpoint(
    path: Path,
    *,
    norm_pix: bool = False,
    width: int = 768,
    jitter: float = 0.0,
    pos_encoding: str = "sincos2d",
) -> Path:
    """A pretraining checkpoint of a ViT-B-shaped MAE — encoder, decoder and `mask_token`.

    Written through `pretrain.checkpoint.save_checkpoint`, the same writer a real run uses, so
    what A2 loads here is laid out exactly as what it would load on a rented box.
    """
    from pretrain.checkpoint import RunState, save_checkpoint
    from pretrain.config import Architecture, Loss, RunConfig
    from pretrain.fingerprint import DataFingerprint
    from pretrain.model import MaskedAutoencoder

    # `box` hard-links its checkpoint to one shared file, so writing *through* the name would
    # rewrite every other test's copy. `save_checkpoint` ends in `os.replace`, which moves the
    # name and leaves the shared inode alone; the unlink says so out loud and keeps that true
    # for a caller who reaches for a plainer writer later.
    path.unlink(missing_ok=True)
    cfg = RunConfig().replace(
        architecture=Architecture(
            img_px=224,
            patch_px=16,
            encoder_dim=width,
            encoder_depth=12,
            encoder_heads=12,
            decoder_dim=16,
            decoder_depth=1,
            decoder_heads=2,
            pos_encoding=pos_encoding,
        ),
        loss=Loss(norm_pix_loss=norm_pix),
    )
    model = MaskedAutoencoder.from_config(cfg)
    if jitter:
        # `from_config` seeds init inside a `fork_rng`, so one config builds one bit-identical
        # model. Moving a weight is what makes a second checkpoint a *different* one.
        with torch.no_grad():
            model.decoder_pred.bias.add_(jitter)
    save_checkpoint(
        path,
        config=cfg,
        fingerprint=DataFingerprint(
            corpus_dir="fixture",
            dataset_version="fixture",
            n_tiles=len(TEST_COGS) * TILES_PER_COG,
            selection_list_md5="0" * 32,
            split_md5=None,
            split_blocks_md5=None,
            split_state="committed",
            split_note=None,
            git_sha="0" * 40,
        ),
        state=RunState(epoch=400, step=92_400),
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        rng={},
    )
    return path


@pytest.fixture(scope="session")
def pristine_checkpoint(tmp_path_factory) -> Path:
    """The unmodified checkpoint, written **once per worker** instead of once per test.

    326 MB of ViT-B weights that most tests below only ever read. At function scope this one
    file put ~8 GB of scratch on disk per suite run, which is what filled the box. It is the
    same bytes either way: `from_config` seeds its init inside a `fork_rng`, so one config
    builds one bit-identical model.
    """
    return _write_checkpoint(tmp_path_factory.mktemp("pristine") / "r5-pretrain.pt")


def _link_checkpoint(src: Path, dst: Path) -> Path:
    """Hard-link `src` to `dst`, falling back to a copy on a filesystem that will not.

    A link rather than a copy because the tests read this file and do not write through it —
    see `_write_checkpoint` for the one rule that keeps that safe. `dst` is a real path inside
    the test's own `tmp_path`, so a retained failure directory still holds a whole checkpoint.
    """
    try:
        os.link(src, dst)
    except OSError:  # cross-device, or a filesystem with no hard links
        shutil.copy(src, dst)
    return dst


@pytest.fixture
def box(tmp_path, pristine_checkpoint):
    """The A1 box, plus the one file A2 needs that no other arm reads."""
    corpus = build_corpus(
        tmp_path / "corpus", TEST_COGS, tiles_per_cog=TILES_PER_COG, metrics_fn=_slope_mean
    )
    directory = tmp_path / "artifacts"
    directory.mkdir()

    table = pq.read_table(corpus / "manifest.parquet", columns=["tile_id"])
    tile_ids = sorted(table.column("tile_id").to_pylist())
    write_tile_strata([(tile_id, 0) for tile_id in tile_ids], directory / TEST_TILES_NAME)
    masks.write_eval_masks(
        masks.eval_rows(tile_ids),
        directory / EVAL_MASKS_NAME,
        artifact_hash=BenchmarkConfig().artifact_hash,
        real=masks.RealFamily.none(),
    )
    for name in ("task1-prereg-bar.json", metrics.GOLDEN_ARRAYS, metrics.GOLDEN_SIDECAR):
        shutil.copy(COMMITTED / name, directory / name)

    _link_checkpoint(pristine_checkpoint, tmp_path / "r5-pretrain.pt")
    return tmp_path


@pytest.fixture
def flags(box):
    return [
        "--set", f'paths.corpus_dir="{box / "corpus"}"',
        "--set", f'paths.artifacts_dir="{box / "artifacts"}"',
        "--set", f'paths.results_path="{box / "results.parquet"}"',
        "--set", f'paths.plan_path="{PLAN}"',
        "--set", f'paths.demae_r5_mae_checkpoint="{box / "r5-pretrain.pt"}"',
        "--set", f"tiles.test={len(TEST_COGS) * TILES_PER_COG}",
    ]


@pytest.fixture
def scored(box, flags, capsys):
    """A1 first, then A2 — the order §4.5 runs them in, and the order R12's rule needs.

    A1 is not scenery here. R12's contingency is an *ordering against A1*, so a box with no A1
    rows cannot answer it; running both is what makes the disposition below a verdict rather than
    a "pending".
    """
    assert launch.main(["run", "--arm", "A1", *flags]) == 0
    assert launch.main(["run", "--arm", ARM, *flags]) == 0
    return box, capsys.readouterr().out


def _without_checkpoint(flags: list[str]) -> list[str]:
    """`flags` without the mae-checkpoint override — dropped as a `--set value` **pair**."""
    pairs = list(zip(flags[::2], flags[1::2]))
    return [item for pair in pairs if "demae_r5_mae_checkpoint" not in pair[1] for item in pair]


def table_of(box):
    return pq.read_table(box / "results.parquet")


def a2_rows(box):
    table = table_of(box)
    arm_column = table.column("arm").to_pylist()
    return table.take([i for i, name in enumerate(arm_column) if name == ARM])


def records_of(box):
    return results.read_run_records(results.run_record_path(box / "results.parquet"))


def a2_record(box):
    return [r for r in records_of(box) if r["cell"]["arm"] == ARM][-1]


# --- The rows -----------------------------------------------------------------


def test_a2_rows_land_in_the_parquet(scored):
    """#167's first criterion. One row per committed eval instance, under this arm's name."""
    box, _ = scored
    rows = a2_rows(box)

    assert rows.num_rows > 0
    assert rows.num_rows == pq.read_table(box / "artifacts" / EVAL_MASKS_NAME).num_rows
    assert set(rows.column("init").to_pylist()) == {"demae-r5"}


def test_a2_scores_the_same_instances_a1_did(scored):
    """The two arms are one benchmark, which is the precondition R12's comparison rests on."""
    box, _ = scored
    table = table_of(box)
    arm_column = table.column("arm").to_pylist()

    def keys(name):
        rows = table.take([i for i, a in enumerate(arm_column) if a == name])
        return set(zip(rows.column("tile_id").to_pylist(), rows.column("mask_rle_md5").to_pylist()))

    assert keys(ARM) == keys("A1")


def test_no_head_is_trained_and_the_record_says_so(scored):
    """#167's first criterion, other half: zero-shot means zero trainable parameters."""
    box, _ = scored
    record = a2_record(box)

    assert record["parameters"]["trainable"] == 0
    assert record["parameters"]["total"] == arms.ENCODERS["demae-r5"].params
    assert record["resolved"]["arm"] == ARM


def test_the_rows_are_scored_in_metres(scored):
    """§2.1's inversion, at the CLI altitude — `score` refuses a fill that is not, so a row
    existing at all is the assertion. What is checked here is that it is not degenerate."""
    box, _ = scored
    rmse = np.array(a2_rows(box).column("rmse").to_pylist())

    assert np.isfinite(rmse).all()
    assert (rmse > 0).all(), "an untrained-on-this-task decoder does not reconstruct exactly"


# --- The run record's R12 disposition -----------------------------------------


def test_the_record_marks_a2_as_carrying_no_c_claim(scored):
    """#167's second criterion."""
    box, _ = scored
    register = a2_record(box)["register"]

    assert register["rule"] == "R12"
    assert register["c_claim"] is None
    assert register["carries_c_claim"] is False
    assert "insulated from C1 in both directions" in register["insulation"]


def test_the_record_computes_the_majority_of_bands_comparison_and_its_outcome(scored):
    """#167's third criterion, and the shape §4.3 argues for.

    Four bands, RMSE, lower better, against the A1 rows in the same cell. What the outcome *is*
    is left to the run — asserting it would pre-empt R12.
    """
    box, _ = scored
    comparison = a2_record(box)["register"]["comparison"]

    assert comparison["metric"] == "rmse"
    assert comparison["lower_is_better"] is True
    assert comparison["bands_registered"] == 4
    assert comparison["bands_compared"] == 4, "the fixture spreads slope_mean across all four"
    assert comparison["instances_compared"] == a2_rows(box).num_rows
    assert set(comparison["per_band"]) == {"R1", "R2", "R3", "R4"}
    assert isinstance(comparison["beats_a1_in_a_majority"], bool)
    assert comparison["negative_result"] is not comparison["beats_a1_in_a_majority"]


def test_the_reported_verdict_follows_the_comparison(scored):
    box, _ = scored
    register = a2_record(box)["register"]

    expected = (
        zeroshot.REPORTED_NEGATIVE
        if register["comparison"]["negative_result"]
        else zeroshot.REPORTED_ORDERING
    )
    assert register["reported_as"] == expected


def test_the_record_never_calls_the_a2_to_a3_difference_task_adaptation(scored):
    """#167's fourth criterion, asserted over the **whole** record rather than one field.

    The clause R12 struck is one a later ticket could reintroduce anywhere in this payload — in
    a gate's note, in the arm's `what` string — so the search is over the serialized record.
    """
    box, _ = scored
    document = json.dumps(a2_record(box))

    for struck in (
        "quantifies precisely what the task adaptation bought",
        "what task adaptation bought",
        "attributable to task adaptation",
        "due to task adaptation",
        "what the task adaptation bought",
    ):
        assert struck not in document
    assert "joint quantity" in document


def test_the_disposition_is_printed_even_when_the_contingency_did_not_fire(scored):
    """A rule that only speaks when it triggers is one nobody can tell was consulted."""
    _, out = scored
    assert "R12" in out and "no C-claim" in out and "A2→A3 joint" in out


def test_no_other_arm_carries_a_register_block(scored):
    """R12 is A2's rule; a block on every row is one a reviewer stops reading."""
    box, _ = scored
    a1 = [r for r in records_of(box) if r["cell"]["arm"] == "A1"][-1]
    assert "register" not in a1


def test_the_comparison_is_pending_rather_than_negative_when_a1_has_not_run(box, flags):
    """*Not computed yet* is not *did not beat A1* — the distinction the memo turns on."""
    assert launch.main(["run", "--arm", ARM, *flags]) == 0
    comparison = a2_record(box)["register"]["comparison"]

    assert comparison["bands_compared"] == 0
    assert comparison["negative_result"] is None
    assert a2_record(box)["register"]["reported_as"] == zeroshot.REPORTED_PENDING
    assert "A1 runs first" in comparison["why"]


# --- What `run` refuses -------------------------------------------------------


def test_a_box_with_no_pretraining_checkpoint_is_refused_before_the_pass(box, flags):
    """§14's release is encoder-only, and this arm *is* the pretrained decoder."""
    with pytest.raises(FileNotFoundError, match="pretrained MAE decoder"):
        launch.main(["run", "--arm", ARM, *_without_checkpoint(flags)])


def test_the_refusal_names_the_field_and_where_to_find_the_file(box, flags):
    with pytest.raises(FileNotFoundError) as exc:
        launch.main(["run", "--arm", ARM, *_without_checkpoint(flags)])

    message = str(exc.value)
    assert "paths.demae_r5_mae_checkpoint" in message
    assert "source_checkpoint" in message, "the release manifest names the file that can drive it"


def test_a_release_encoder_export_is_refused_for_having_no_decoder(box, flags, tmp_path):
    """The failure the strict load exists to make loud rather than silent."""
    from pretrain.checkpoint import load_checkpoint

    full = load_checkpoint(box / "r5-pretrain.pt")
    payload = torch.load(box / "r5-pretrain.pt", map_location="cpu", weights_only=True)
    from pretrain.model import is_decoder_parameter

    payload["model"] = {k: v for k, v in full.model.items() if not is_decoder_parameter(k)}
    # `torch.save` truncates in place, unlike `save_checkpoint` — unlink to drop this test's
    # link to the shared checkpoint before writing a private one over the name.
    (box / "r5-pretrain.pt").unlink()
    torch.save(payload, box / "r5-pretrain.pt")

    with pytest.raises(ValueError, match="carries no decoder"):
        launch.main(["run", "--arm", ARM, *flags])


def test_a_checkpoint_from_a_differently_shaped_run_is_refused(box, flags):
    """No md5 is published for a pretraining checkpoint, so the declared fields are the check."""
    _write_checkpoint(box / "r5-pretrain.pt", width=384)

    with pytest.raises(ValueError, match="records width 384 .*declares demae-r5 as 768"):
        launch.main(["run", "--arm", ARM, *flags])


def test_a_norm_pix_checkpoint_scores_through_the_registered_estimator(box, flags):
    """#173's resolution, at the CLI altitude.

    R5 and R4 were both pretrained with `norm_pix_loss` on, so this is the state the real box is
    in. #167 refused it for want of a registered inversion; §4.3 now registers one, and the run
    record says which two numbers stood in for the per-patch scale the void hides.
    """
    _write_checkpoint(box / "r5-pretrain.pt", norm_pix=True)

    assert launch.main(["run", "--arm", "A1", *flags]) == 0
    assert launch.main(["run", "--arm", ARM, *flags]) == 0

    record = a2_record(box)
    assert record["training"]["kind"] == "zeroshot"
    scale = record["training"]["scale"]
    assert (scale["level"], scale["spread"], scale["spread_power"]) == ("a1_fill", "visible_idw", 2.0)
    assert scale["reads_void"] is False and "#173" in scale["statement"]
    table = a2_rows(box)
    assert table.num_rows > 0 and np.isfinite(table.column("rmse").to_numpy()).all()


def test_a_moved_estimator_is_refused_at_resolve(box, flags):
    """§4.3's estimator sets A2's numbers, so a run may not move it — `finetune.assert_declared`'s
    rule one arm over, and refused before `--dry-run` returns."""
    with pytest.raises(ValueError, match="drifted from the declared line"):
        launch.main(["run", "--arm", ARM, "--dry-run", *flags, "--set", "zeroshot.spread_power=3"])
    with pytest.raises(ValueError, match="does not register"):
        zeroshot.load_inpainter(
            box / "r5-pretrain.pt", recipe=BenchmarkConfig().zeroshot.replace(level="truth")
        ).check_ready()


# --- §4.3's estimator, as arithmetic -------------------------------------------

_TILE = 224


def _plane_tile(*, slope_x: float = 2.0, slope_y: float = -1.0, level: float = 1200.0) -> np.ndarray:
    y, x = np.mgrid[0:_TILE, 0:_TILE]
    return np.round(level + slope_x * x + slope_y * y).astype(np.int16)


def test_patch_view_is_patchify_and_unpatch_inverts_it():
    """The numpy permutation is the model's, so a per-patch statistic lands on the patch the
    decoder predicted — a transposed view would still be 196 x 256 and still score."""
    import torch
    from pretrain.model import MaskedAutoencoder
    from pretrain.config import RunConfig

    model = MaskedAutoencoder.from_config(RunConfig())
    field = np.random.default_rng(0).standard_normal((_TILE, _TILE)).astype(np.float32)

    ours = zeroshot.patch_view(field, patch_px=model.patch_px)
    theirs = model.patchify(torch.from_numpy(field)[None, None])[0].numpy()
    np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_array_equal(zeroshot.unpatch(ours, patch_px=model.patch_px), field)


def test_idw_weights_sum_to_one_and_fall_with_distance():
    grid = 14
    weights = zeroshot.idw_weights(np.array([0]), np.array([1, 2, 15]), grid=grid, power=2.0)

    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    # (0,1) is one patch away, (0,2) two, (1,1) is sqrt(2): 1 : 1/4 : 1/2 before normalization.
    np.testing.assert_allclose(weights[0] / weights[0, 0], [1.0, 0.25, 0.5])
    with pytest.raises(ValueError, match="both a target and a source"):
        zeroshot.idw_weights(np.array([3]), np.array([3, 4]), grid=grid, power=2.0)


def test_the_registered_estimate_recovers_a_plane_without_reading_the_void():
    """On a plane the thin-plate fill is the truth (its affine term reproduces a plane exactly,
    where `fillnodata`'s inverse-distance average does not) and every patch has the same spread —
    so the estimate equals the truth to numerical tolerance, and a perfect decoder inverted
    through it returns the tile. The oracle `pretrain.recon` uses, with the void's own statistics
    replaced by §4.3's estimates."""
    from benchmark.config import BenchmarkConfig as Cfg
    from dataset.metrics import land_mask
    from benchmark import extract

    raw = _plane_tile()
    void = np.zeros((_TILE, _TILE), dtype=bool)
    void[60:150, 70:160] = True
    valid = land_mask(raw) & ~void
    normalized = extract.normalize(raw[None], valid[None], normalization="per_tile")[0]
    keep = zeroshot.keep_set(void, patch_px=16)
    cfg = Cfg()

    mean, std = zeroshot.estimate_scale(
        normalized, raw_i16=raw, void=void, valid=valid, keep=keep, patch_px=16,
        var_floor=1e-4, recipe=cfg.zeroshot, fill_recipe=cfg.fill.replace(variant="tps"),
    )
    truth = zeroshot.patch_view(normalized, patch_px=16).astype(np.float64)
    touched = ~keep
    assert touched.sum() > 0
    np.testing.assert_allclose(mean[touched], truth[touched].mean(axis=1), atol=2e-3)
    np.testing.assert_allclose(std[touched], truth[touched].std(axis=1, ddof=1), rtol=2e-3)

    # A perfect decoder emits each patch in its own standardized units; the registered
    # inversion returns the plane.
    standardized = (truth - truth.mean(axis=1, keepdims=True)) / truth.std(axis=1, ddof=1, keepdims=True)
    back = zeroshot.invert_patches(standardized, mean, std, patch_px=16)
    np.testing.assert_allclose(back, normalized, atol=5e-3)


def test_the_estimate_never_reads_the_void():
    """Change every void pixel and the estimate does not move: the two numbers come from A1's
    fill and the seen patches, which is the whole of what §4.3 registers."""
    from benchmark.config import BenchmarkConfig as Cfg
    from dataset.metrics import land_mask
    from benchmark import extract

    rng = np.random.default_rng(1)
    raw = (_plane_tile() + rng.integers(-40, 40, (_TILE, _TILE))).astype(np.int16)
    void = np.zeros((_TILE, _TILE), dtype=bool)
    void[40:120, 100:170] = True
    cfg = Cfg()

    def estimate(tile):
        valid = land_mask(tile) & ~void
        normalized = extract.normalize(tile[None], valid[None], normalization="per_tile")[0]
        return zeroshot.estimate_scale(
            normalized, raw_i16=tile, void=void, valid=valid,
            keep=zeroshot.keep_set(void, patch_px=16), patch_px=16,
            var_floor=1e-4, recipe=cfg.zeroshot, fill_recipe=cfg.fill,
        )

    altered = raw.copy()
    altered[void] = 5000
    for ours, theirs in zip(estimate(raw), estimate(altered)):
        np.testing.assert_array_equal(ours, theirs)


def test_a2_does_not_wait_on_the_extractor_spec_it_never_consults(scored):
    """ADR-0012's contract governs features out of a frozen encoder; A2 produces none.

    Asserted through a *passing* run — the fixture writes no `extractor_spec.json` — with the
    resolution line still naming it, because what is on this box is a fact about the box.
    """
    box, out = scored
    assert not (box / "artifacts" / "extractor_spec.json").exists()
    assert "extractor spec" in out and "not read by this arm" in out


def test_a2_does_not_wait_on_the_train_mask_bank_it_never_reads(scored):
    """Zero-shot trains nothing, so §3.4's bank is not an input — A1's rule, same reason."""
    _, out = scored
    assert "mask bank" in out


# --- The resolution line ------------------------------------------------------


def test_the_resolution_line_names_both_checkpoints(scored):
    """Two files, two states. Which one a box is missing is which thing to go and mount."""
    _, out = scored
    assert "mae checkpoint" in out
    assert "r5-pretrain.pt" in out


def test_the_record_carries_the_pretraining_checkpoint_by_md5(scored):
    """A table has to be answerable to the weights that produced it."""
    box, _ = scored
    mae = a2_record(box)["resolved"]["mae_checkpoint"]

    assert mae["path"].endswith("r5-pretrain.pt")
    assert len(mae["md5"]) == 32


def test_the_run_id_moves_when_the_pretraining_checkpoint_does(box, flags, capsys):
    """`run_id` moves when anything a results row is answerable to moves (#158's rule)."""
    assert launch.main(["run", "--arm", ARM, *flags]) == 0
    first = a2_record(box)["run_id"]

    _write_checkpoint(box / "r5-pretrain.pt", jitter=0.5)  # same shape, different weights
    assert launch.main(["run", "--arm", ARM, *flags]) == 0
    assert a2_record(box)["run_id"] != first


# --- The other registered cell ------------------------------------------------


@pytest.fixture
def r4_flags(box, flags):
    """R4's continual-init cell. Its checkpoint records **mmselfsup's row-first** axis order.

    Not a detail: R4 warm-started from ImageNet MAE, whose fixed sin-cos embedding is the
    transpose of MAE's (P102 §4), and the model rebuilds that buffer from its own config. A file
    recording the wrong order loads clean, runs clean, and puts every token at the transposed
    position — so `benchmark.arms.ENCODERS`' declared `pos_encoding` is checked against it.
    """
    _write_checkpoint(box / "r4-pretrain.pt", pos_encoding="sincos2d_rowfirst")
    return [*flags, "--set", f'paths.demae_r4_mae_checkpoint="{box / "r4-pretrain.pt"}"']


def test_a2_r4_rows_land_in_the_parquet_too(box, r4_flags):
    """#167's first criterion names both cells, and R4 is §2.3's pre-registered secondary."""
    assert launch.main(["run", "--arm", "A1", *r4_flags]) == 0
    assert launch.main(["run", "--arm", "A2-R4", *r4_flags]) == 0

    table = table_of(box)
    arm_column = table.column("arm").to_pylist()
    rows = table.take([i for i, name in enumerate(arm_column) if name == "A2-R4"])

    assert rows.num_rows == pq.read_table(box / "artifacts" / EVAL_MASKS_NAME).num_rows
    assert set(rows.column("init").to_pylist()) == {"demae-r4"}

    record = [r for r in records_of(box) if r["cell"]["arm"] == "A2-R4"][-1]
    assert record["register"]["rule"] == "R12"
    assert record["register"]["c_claim"] is None
    assert record["register"]["comparison"]["bands_compared"] == 4


def test_r4s_axis_order_is_checked_against_the_registry(box, flags):
    """The transpose that loads clean, runs clean, and reconstructs the wrong terrain."""
    _write_checkpoint(box / "r4-pretrain.pt", pos_encoding="sincos2d")  # MAE's, not R4's
    wrong = [*flags, "--set", f'paths.demae_r4_mae_checkpoint="{box / "r4-pretrain.pt"}"']

    with pytest.raises(ValueError, match="pos_encoding 'sincos2d'.*declares demae-r4"):
        launch.main(["run", "--arm", "A2-R4", *wrong])


def test_both_registered_a2_cells_are_governed_by_r12():
    """The rule is read off a declared field, so neither cell can drift out from under it."""
    for name in ("A2-R5", "A2-R4"):
        assert zeroshot.governs(arms.get(name))
    for name in ("A1", "A3-R5", "A4-R5", "A7", "A9"):
        assert not zeroshot.governs(arms.get(name))
