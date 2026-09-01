"""§4.1's shared head, its one tuning pass, and A6 end to end (#159).

**Seam 1** for everything that can attach there: `python -m benchmark.launch run --arm A6` trains
a head against the committed bank and lands rows in the results parquet, and `--tune` writes the
record of the one pass §4.1 allows. The standing defence in this repo is a test that drives
`main(argv)` rather than one that drives each piece.

Two claims cannot attach at seam 1 and are asserted at the function seam instead, each for a
stated reason:

* **the parameter counts**, because 1.08M and 1.18M are a claim about an architecture rather than
  about a run, and A5's 768-d instantiation needs terratorch and a 1.1 GB tokenizer to reach
  through the CLI while :func:`benchmark.head.build` needs neither;
* **the loss**, whose two terms are closed-form arithmetic — a masked mean and a Sobel
  difference — that a training curve can only ever be consistent with.

## The stub encoder

ADR-0012's extractor refuses a checkpoint whose md5 this plan does not publish, and A6's is
deliberately unpinned (`benchmark.arms.ENCODERS`) until a box mounts one. So there is no way to
drive a real ViT through `run` in a test, and the spec's own answer is the one taken here: *a
tiny fixture corpus and a stub encoder*. :class:`StubExtractor` implements the contract's
surface — ``extract(raw_i16, valid_mask, *, arm) -> Features`` in ADR-0012 §5's canonical layout,
plus a frozen ``encoder(arm)`` — and nothing else. What it does **not** stub is the composition:
the A1 fill, the valid mask, the void, the head, the loss, the bank and the split guard are all
the shipping code.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from benchmark import arms, evaluate, head, launch, masks, metrics, train, tune
from benchmark.artifacts import (
    MASK_BANK_SIDECAR_NAME,
    TILE_LIST_NAMES,
    VAL_TILES_NAME,
    TEST_TILES_NAME,
)
from benchmark.config import TUNED_HEAD, BenchmarkConfig, Head
from pretrain.artifact import write_tile_strata
from tests.conftest import build_corpus

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
COMMITTED = REPO / "artifacts" / "task1"
PLAN = REPO / "docs" / "research" / "task1-void-fill-plan.md"

#: Two COGs per split, so head-train, val and test are disjoint the way §3.1's draw makes them
#: and the split guard has something real to refuse.
COGS = [
    ("Copernicus_DSM_COG_10_N45_00_E006_00_DEM", "+000_+040", "train"),
    ("Copernicus_DSM_COG_10_N45_00_E007_00_DEM", "+000_+040", "train"),
    ("Copernicus_DSM_COG_10_N46_00_E008_00_DEM", "+000_+040", "val"),
    ("Copernicus_DSM_COG_10_N47_00_E009_00_DEM", "+000_+040", "test"),
    ("Copernicus_DSM_COG_10_N48_00_E010_00_DEM", "+000_+040", "test"),
]
TILES_PER_COG = 2

#: The stub encoder's width. A6 is registered at 768 and the head is built at whatever the
#: registry says, so the stub answers at the registered width rather than choosing one.
WIDTH = arms.ENCODERS["imagenet-mae"].width


def _slope_mean(cog: int, tile: int) -> dict:
    """A spread of `slope_mean`, so §5's quartile bands have something to cut."""
    return {"slope_mean": 0.5 + 2.0 * (cog * TILES_PER_COG + tile)}


# --- The stub encoder ---------------------------------------------------------


class _StubEncoder(torch.nn.Module):
    """A frozen module with one parameter, so "the encoder did not move" has something to check."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(WIDTH), requires_grad=False)


class StubExtractor:
    """ADR-0012's surface, and nothing behind it. See the module docstring.

    The tokens are a deterministic function of the tile it was handed — a 14x14 mean-pool of the
    normalized input, broadcast to the registered width — rather than random. That matters for
    exactly one test: a head that cannot see the tile cannot reduce a loss on it, and "the head
    trained" would then be an untestable claim.
    """

    def __init__(self, spec=None, cfg=None, *, device: str = "cpu") -> None:
        self.spec = spec
        self.config = cfg
        self.device = device
        self.precision: dict = {}
        self._model = _StubEncoder()

    def encoder(self, arm: str):
        self._model.eval()
        return self._model

    def extract(self, raw_i16, valid_mask, *, arm: str):
        from benchmark import extract as extract_module

        raw, mask = extract_module._as_batch(raw_i16, valid_mask)
        x = extract_module.normalize(raw, mask, normalization="per_tile")
        tile = torch.from_numpy(x).unsqueeze(1)
        pooled = torch.nn.functional.adaptive_avg_pool2d(tile, (14, 14))
        tokens = pooled.reshape(tile.shape[0], 1, 196).transpose(1, 2).repeat(1, 1, WIDTH)
        return extract_module.Features(
            output=tokens.float(),
            intermediates=tuple(tokens.float() for _ in range(12)),
            cls=None,
        )


@pytest.fixture
def stub(monkeypatch):
    """Every path that builds an extractor gets the stub, including `launch`'s."""
    from benchmark import extract as extract_module

    monkeypatch.setattr(extract_module, "Extractor", StubExtractor)
    return StubExtractor


# --- The box ------------------------------------------------------------------


@pytest.fixture
def frozen(monkeypatch):
    """Stand in a test-scale frozen head recipe — the guard's own legitimate move, simulated.

    §4.1 freezes the *schedule length* along with the learning rate, and rightly: A5 trained for
    twenty epochs beside A4's fifty is exactly the unfairness C1 must not have, and #162 makes it
    an acceptance criterion of its own. So `--set head.epochs=1` is refused by
    :func:`benchmark.tune.assert_frozen`, and it has to be — which leaves a seam-1 test of a
    training arm with one honest route: move the frozen line, which is what the refusal itself
    tells a build session to do.

    Fifty epochs of a 1.18M-parameter head is ~1 TFLOP of CPU per test here, so the alternative
    was not a slower suite; it was no seam-1 test of the arm this ticket ships.
    """
    from benchmark import config as config_module

    monkeypatch.setattr(
        config_module, "TUNED_HEAD", Head(epochs=2, batch_size=2), raising=True
    )


@pytest.fixture
def box(tmp_path):
    """A frozen box: a corpus, §3.1's three lists, the eval masks, the bank, and a checkpoint."""
    corpus = build_corpus(
        tmp_path / "corpus", COGS, tiles_per_cog=TILES_PER_COG, metrics_fn=_slope_mean
    )
    directory = tmp_path / "artifacts"
    directory.mkdir()

    table = pq.read_table(corpus / "manifest.parquet", columns=["tile_id", "source_cog"])
    split_of = {stem: split for stem, _, split in COGS}
    tiles: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for tile_id, stem in zip(
        table.column("tile_id").to_pylist(), table.column("source_cog").to_pylist()
    ):
        tiles[split_of[stem]].append(tile_id)
    for split in tiles:
        tiles[split].sort()

    # §3.1's draws, written exactly as `benchmark.tiles` writes them. The 1k and 5k rungs are
    # nested prefixes of the 20k list (§3.1), so the fixture writes the same rows to all three.
    for name in TILE_LIST_NAMES.values():
        write_tile_strata([(t, 0) for t in tiles["train"]], directory / name)
    write_tile_strata([(t, 0) for t in tiles["val"]], directory / VAL_TILES_NAME)
    write_tile_strata([(t, 0) for t in tiles["test"]], directory / TEST_TILES_NAME)

    # The same knobs `flags` sets, because `artifact_hash` covers the `tiles` and `masks` groups
    # and §3.4's guard refuses a bank drawn under a different one. The fixture and the command
    # have to agree about the draw or the box is not a frozen box.
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
        directory / "eval_masks.parquet",
        artifact_hash=cfg.artifact_hash,
        real=masks.RealFamily.none(),
    )
    masks.freeze_bank(cfg)
    for name in ("task1-prereg-bar.json", metrics.GOLDEN_ARRAYS, metrics.GOLDEN_SIDECAR):
        shutil.copy(COMMITTED / name, directory / name)
    # ADR-0012's contract, and a file for `artifacts.resolve` to hash as A6's checkpoint. The
    # stub never opens it; what is being satisfied here is the pre-flight, which is a fact about
    # the box and is checked whatever the encoder turns out to be.
    from benchmark import extract as extract_module

    extract_module.write_spec(cfg)
    (tmp_path / "imagenet_mae.pth").write_bytes(b"stub")

    tmp_path.joinpath("splits.json").write_text(json.dumps(tiles))
    return tmp_path


def splits_of(box) -> dict[str, list[str]]:
    return json.loads((box / "splits.json").read_text())


@pytest.fixture
def flags(box):
    """The overrides that point one cell at the fixture box, at test scale."""
    return [
        "--set", f'paths.corpus_dir="{box / "corpus"}"',
        "--set", f'paths.artifacts_dir="{box / "artifacts"}"',
        "--set", f'paths.results_path="{box / "results.parquet"}"',
        "--set", f'paths.mask_bank_path="{box / "train_mask_bank.parquet"}"',
        "--set", f'paths.imagenet_mae_checkpoint="{box / "imagenet_mae.pth"}"',
        "--set", f'paths.plan_path="{PLAN}"',
        "--set", f'runtime.checkpoint_dir="{box / "heads"}"',
        "--set", 'runtime.device="cpu"',
        "--set", "runtime.loader_workers=0",
        "--set", "masks.bank_draws_per_tile=2",
        "--set", f"tiles.test={len(splits_of(box)['test'])}",
        # The `frozen` fixture stands in a recipe carrying these; a cell that moved them without
        # it is refused, which is `test_a_cell_with_its_own_hyperparameter_is_refused`.
        "--set", "head.epochs=2",
        "--set", "head.batch_size=2",
    ]


def table_of(box):
    return pq.read_table(box / "results.parquet")


# --- The parameter counts, asserted rather than quoted -------------------------


@pytest.mark.parametrize("width,expected", [(384, 1_082_017), (768, 1_181_089)])
def test_the_head_costs_the_corrected_figure_at_each_width(width, expected):
    """#159's first criterion. §4.1 says "~3M" and §7 says that is wrong; this is the number.

    Asserted twice by different means, which is the whole point of the closed form: the module is
    counted, and the arithmetic over §4.1's four stage dimensions is derived, and a stage
    dimension that moved without the count moving with it makes them disagree.
    """
    assert head.parameter_count(width) == expected
    assert head.build(width).parameters == expected
    assert head.PARAMETERS[width] == expected
    assert f"{expected / 1e6:.2f}M" in head.format_parameters(width)


def test_the_registry_reports_the_head_it_actually_builds():
    """§6 item 4 prints these, so the registry's number and the module's are one number."""
    for name in ("A4-R5", "A4-R4", "A5", "A6"):
        arm = arms.get(name)
        counts = arms.parameter_counts(arm)
        assert counts["trainable"] == head.build(arms.ENCODERS[arm.encoder].width).parameters
    # A4-S is the 384-d appendix cell, and it is where the two figures differ.
    assert arms.parameter_counts(arms.get("A4-S"))["trainable"] == 1_082_017


def test_a4_a5_and_a6_instantiate_one_head_differing_only_in_the_input_width():
    """#159's second criterion, and C1's precondition: the decoder is not a variable.

    Compared structurally rather than by parameter count — two heads can agree on a total and
    disagree on where it sits. Every tensor is required to match in name and shape **except**
    the first projection's weight, which is the one place the encoder's width enters.
    """
    built = {
        name: head.build(arms.ENCODERS[arms.get(name).encoder].width)
        for name in ("A4-R5", "A5", "A6", "A4-S")
    }
    shapes = {
        name: {k: tuple(v.shape) for k, v in b.module.state_dict().items()}
        for name, b in built.items()
    }
    reference = shapes["A6"]
    for name, shape in shapes.items():
        assert set(shape) == set(reference), f"{name} has a different set of tensors"
        differ = {k for k in shape if shape[k] != reference[k]}
        assert differ <= {"norm.weight", "norm.bias", "project.weight"}, (
            f"{name} differs from A6 at {sorted(differ)}, which is more than the input width"
        )
    assert shapes["A4-S"]["project.weight"] == (256, 384, 1, 1)
    assert shapes["A6"]["project.weight"] == (256, 768, 1, 1)


# --- The residual --------------------------------------------------------------


def test_a_zero_output_head_returns_the_a1_fill_unchanged():
    """#159's third criterion, at the function seam and exactly rather than approximately.

    `build` zeroes the output convolution, so this is also the state every training run starts
    from: epoch 0's first forward *is* A1, which is what makes a loss curve readable.
    """
    built = head.build(WIDTH)
    tokens = torch.randn(2, 196, WIDTH)
    void = torch.zeros(2, 1, 224, 224)
    void[:, :, 60:120, 60:120] = 1
    residual = built.module(tokens, void).detach().numpy()
    assert not residual.any()

    a1 = np.linspace(0, 1000, 224 * 224, dtype=np.float32).reshape(224, 224)
    mask = np.zeros((224, 224), dtype=bool)
    mask[60:120, 60:120] = True
    assert np.array_equal(head.apply_residual(a1, residual[0, 0], mask), a1)


def test_the_residual_never_moves_ground_outside_the_void():
    """§2.4's convention: outside the void is measured ground no arm is asked to predict."""
    a1 = np.zeros((224, 224), dtype=np.float32)
    residual = np.full((224, 224), 7.0, dtype=np.float32)
    mask = np.zeros((224, 224), dtype=bool)
    mask[10:20, 10:20] = True

    out = head.apply_residual(a1, residual, mask)
    assert np.array_equal(out[mask], np.full(int(mask.sum()), 7.0, dtype=np.float32))
    assert not out[~mask].any()


# --- §4.1's loss ---------------------------------------------------------------


def test_the_loss_reads_void_pixels_and_nothing_else():
    """Masked L1 in metres: moving ground outside the void must not change the number."""
    truth = torch.zeros(1, 1, 224, 224)
    void = torch.zeros(1, 1, 224, 224)
    void[:, :, 100:110, 100:110] = 1

    inside = truth.clone()
    inside[:, :, 100:110, 100:110] = 2.0
    outside = truth.clone()
    outside[:, :, 0:10, 0:10] = 2.0

    assert head.loss(inside, truth, void, slope_weight=0.0).elevation.item() == pytest.approx(2.0)
    assert head.loss(outside, truth, void, slope_weight=0.0).elevation.item() == pytest.approx(0.0)


def test_the_slope_term_is_the_weighted_second_half_and_can_be_switched_off():
    """0.1x, and the 0.0 the tuning grid is allowed to select — both readable off the total."""
    truth = torch.zeros(1, 1, 224, 224)
    void = torch.ones(1, 1, 224, 224)
    ramp = torch.arange(224, dtype=torch.float32).reshape(1, 1, 1, 224).repeat(1, 1, 224, 1)

    terms = head.loss(ramp, truth, void, slope_weight=0.1)
    assert terms.total.item() == pytest.approx(
        terms.elevation.item() + 0.1 * terms.slope.item(), rel=1e-6
    )
    assert terms.slope.item() > 0
    off = head.loss(ramp, truth, void, slope_weight=0.0)
    assert off.total.item() == pytest.approx(off.elevation.item(), rel=1e-6)


def test_sobel_recovers_a_known_gradient():
    """A unit-per-pixel ramp has gradient 1 along its own axis and 0 across it."""
    ramp = torch.arange(224, dtype=torch.float32).reshape(1, 1, 1, 224).repeat(1, 1, 224, 1)
    gy, gx = head.sobel_gradients(ramp)
    interior = (slice(None), slice(None), slice(1, -1), slice(1, -1))
    assert gx[interior].abs().mean().item() == pytest.approx(1.0, rel=1e-5)
    assert gy[interior].abs().max().item() == pytest.approx(0.0, abs=1e-5)


# --- Tuned once, and frozen ----------------------------------------------------


def test_every_arm_loads_the_same_frozen_head_recipe_object():
    """#159's fifth criterion, in its strongest available form: not equal, *the same object*."""
    assert BenchmarkConfig(arm="A4-R5").head is TUNED_HEAD
    assert BenchmarkConfig(arm="A5").head is BenchmarkConfig(arm="A6").head
    assert BenchmarkConfig(arm="A4-S", size="1k", seed=2).head is TUNED_HEAD


def test_a_cell_with_its_own_hyperparameter_is_refused(box, flags, stub, frozen):
    """The rule with teeth: an arm cannot quietly carry a learning rate of its own."""
    with pytest.raises(tune.TuningError, match="frozen for every arm") as exc:
        launch.main(["run", "--arm", "A6", "--set", "head.lr=0.0003", *flags])
    assert "--tune" in str(exc.value), "the refusal must name the one legitimate way out"


def test_a_seed_count_is_not_a_tuned_hyperparameter(box, flags, stub):
    """`head.seeds` is how many observations §4.1 asks for, not a knob the pass may move."""
    tune.assert_frozen(BenchmarkConfig(head=Head(seeds=1)))
    with pytest.raises(tune.TuningError):
        tune.assert_frozen(BenchmarkConfig(head=Head(slope_weight=0.3)))


def test_a_tuning_pass_is_not_locked_out_by_the_record_it_is_there_to_replace(
    box, flags, stub, frozen
):
    """The guard's own deadlock, closed (#159).

    A tuning pass that selects something new leaves a record disagreeing with the frozen line —
    which is exactly the state `assert_records_the_frozen_recipe` refuses. Re-tuning is the
    remedy for that state, so a guard that refused it too would leave deleting the record as the
    only way out, and the record is the one file that says what the disagreement is.

    A *scoring* cell in the same state is still refused, which is the half that matters.
    """
    directory = box / "artifacts"
    (directory / tune.TUNING_NAME).write_text(
        json.dumps({"arm": "A6", "selected": {"lr": 0.0003}, "searched": [], "grid": {}})
    )
    with pytest.raises(tune.TuningError, match="never folded into"):
        launch.main(["run", "--arm", "A6", *flags])
    # ...and the same box, with `--tune`, runs.
    assert launch.main(["run", "--arm", "A6", "--tune", *flags]) == 0


def test_the_grid_searches_only_the_knob_the_plan_argues_for_least():
    """§4.1 states a value for every knob; the one this pass may move is the learning rate.

    The rest are refused a search on purpose. `slope_weight` and the loss's shape are §4.1's
    *design* of the objective; `epochs` and `batch_size` are §7 budget lines that #162 turns into
    a fairness criterion. A grid over any of them would let a val RMSE re-decide a plan decision,
    which is what this benchmark's own "Out of Scope" section forbids.
    """
    assert set(tune.GRID) == {"lr"}
    assert TUNED_HEAD.lr in tune.GRID["lr"]
    assert len(tune.points()) == len(tune.GRID["lr"])
    # §4.1's own value must not sit on an end of the grid: a winner on a boundary is an
    # unbracketed optimum, which is what sent the first real pass back for a wider search.
    assert tune.GRID["lr"][0] < TUNED_HEAD.lr < tune.GRID["lr"][-1]
    searchable = set(tune.GRID)
    for name in ("slope_weight", "loss", "epochs", "batch_size", "stage_dims"):
        assert name not in searchable, f"{name} is §4.1's to state, not this pass's to move"


def test_a_tuning_record_the_config_never_absorbed_is_refused(box, flags, stub, frozen):
    """A pass that selected something new, and the one-line edit it owed, never made.

    Without this the `OWED` console line is the only thing standing between a search result and
    a table trained at the recipe that search rejected — and a console line is read once, by
    whoever ran the pass.
    """
    directory = box / "artifacts"
    (directory / tune.TUNING_NAME).write_text(
        json.dumps({"arm": "A6", "selected": {"lr": 0.0003}, "searched": [], "grid": {}})
    )
    with pytest.raises(tune.TuningError, match="never folded into"):
        launch.main(["run", "--arm", "A6", *flags])


def test_tuning_refuses_an_arm_that_is_not_the_neutral_one(box, flags, stub, frozen):
    """§4.1 tunes on A6. Tuning on the arm C1 rests on is the thumb on the scale."""
    with pytest.raises(tune.TuningError, match="neutral encoder"):
        launch.main(["run", "--arm", "A4-R5", "--tune", *flags])


# --- Val only ------------------------------------------------------------------


def test_a_test_tile_is_not_reachable_from_the_training_path(box, flags, stub, frozen):
    """#159's fourth criterion. The guard is the object, so this asks the object.

    The `TrainingTiles` built here is the one the pass builds — same config, same committed
    lists — and every committed test tile is refused by it.
    """
    cfg = launch.resolve(launch.build_parser().parse_args(["run", "--arm", "A6", *flags])).config
    tiles = train.TrainingTiles.read(cfg)
    splits = splits_of(box)

    assert set(tiles.head_train) == set(splits["train"])
    assert set(tiles.val) == set(splits["val"])
    for tile_id in splits["test"]:
        with pytest.raises(train.TrainError, match="neither"):
            tiles.allow(tile_id)


def test_the_training_path_names_no_test_artifact():
    """The other half: the roster of committed files a training pass may open."""
    assert TEST_TILES_NAME not in train.TRAINING_ARTIFACTS
    assert "eval_masks.parquet" not in train.TRAINING_ARTIFACTS
    assert VAL_TILES_NAME in train.TRAINING_ARTIFACTS
    with pytest.raises(train.TrainError, match="tuning reads val only"):
        train.assert_reads_val_only([VAL_TILES_NAME, TEST_TILES_NAME])


def test_the_val_draw_is_one_void_per_val_tile_and_spans_every_geometry(box, flags, stub, frozen):
    """What early stopping is measured against, and why it is not six voids per tile."""
    cfg = launch.resolve(launch.build_parser().parse_args(["run", "--arm", "A6", *flags])).config
    val = train.val_instances(cfg, splits_of(box)["val"])

    assert len(val) == len(splits_of(box)["val"])
    assert len({item.tile_id for item in val}) == len(val)
    # Round-robin over §3.2's six cells: with two val tiles that is the first two of the deal.
    assert [(item.family, item.bucket) for item in val] == [
        ("blob", "small"), ("blob", "medium")
    ]


# --- Training reads the bank ---------------------------------------------------


def test_training_reads_the_committed_bank_rather_than_drawing(box, flags, stub, frozen):
    """#159's last criterion. The voids a head trains on are the ones §3.4 committed."""
    cfg = launch.resolve(launch.build_parser().parse_args(["run", "--arm", "A6", *flags])).config
    train_ids = splits_of(box)["train"]
    banked = train.read_bank(cfg, train_ids)

    committed = pq.read_table(box / "train_mask_bank.parquet").to_pylist()
    for row in committed:
        if row["tile_id"] in set(train_ids):
            assert banked.rle(row["tile_id"], row["epoch"]) == row["mask_rle"]
    # And the bank is K *different* draws per tile, not one draw repeated.
    for tile_id in train_ids:
        assert len(set(banked.by_tile[tile_id])) == cfg.masks.bank_draws_per_tile


def test_the_loader_composes_in_worker_processes_too(box, flags, stub, frozen):
    """`runtime.loader_workers` defaults to 4, so the worker path is the one a GPU run takes.

    §7 measures head training as 4.1x loader-bound — the A1 fill of each banked void is the cost
    — so the shipping default is workers, and everything else here runs at 0 to keep the suite
    sharded cheaply. That leaves exactly one thing untested by the rest of this file: whether
    :class:`benchmark.train.Examples` survives the process boundary. It holds a config and a
    :class:`benchmark.train.TrainingTiles` and opens its own store per worker precisely so that it
    does, under ``spawn`` as well as ``fork``, and a pickling failure here is one that would only
    ever show up on the rented box.
    """
    # `--set` is applied in order and the last one wins, so appending is how a test moves a
    # knob `flags` already set.
    assert (
        launch.main(
            ["run", "--arm", "A6", *flags, "--set", "runtime.loader_workers=2"]
        )
        == 0
    )
    assert table_of(box).num_rows == len(splits_of(box)["test"]) * 2 * 3


def test_a_bank_missing_this_cells_tiles_is_refused(box, flags, stub, frozen):
    """A bank frozen against a different head-train draw, caught structurally."""
    cfg = launch.resolve(launch.build_parser().parse_args(["run", "--arm", "A6", *flags])).config
    with pytest.raises(train.TrainError, match="carries no draws"):
        train.read_bank(cfg, [*splits_of(box)["train"], "not-a-tile"])


# --- A6 through the shipping command ------------------------------------------


@pytest.fixture
def a6(box, flags, stub, frozen, capsys):
    """One real A6 cell through `main(argv)`: train, then score. Returns the box and the console."""
    assert launch.main(["run", "--arm", "A6", *flags]) == 0
    return box, capsys.readouterr().out


def test_the_shipping_command_trains_a_head_and_lands_a6_rows(a6):
    """A6's primary cell, end to end, through the only command that produces a result."""
    box, out = a6
    table = table_of(box)

    # §3.2: every test tile once per bucket in both synthetic families.
    assert table.num_rows == len(splits_of(box)["test"]) * 2 * 3
    assert set(table.column("arm").to_pylist()) == {"A6"}
    assert set(table.column("init").to_pylist()) == {"imagenet-mae"}
    assert "1,181,089 (1.18M) at 768-d" in out
    assert "val_rmse" in out


def test_the_run_record_carries_the_head_it_scored_with(a6):
    """Which epoch's weights produced these rows, and which lists the training pass read."""
    box, _ = a6
    from benchmark import results

    (record,) = results.read_run_records(results.run_record_path(box / "results.parquet"))
    training = record["training"]
    assert training["parameters"] == 1_181_089
    assert training["width"] == 768
    assert training["best_epoch"] in (0, 1)
    assert training["epochs_run"] == 2
    assert Path(training["checkpoint"]).name == train.BEST_CHECKPOINT
    assert TEST_TILES_NAME not in training["read"]
    assert record["parameters"]["trainable"] == 1_181_089
    # §4.1 declares bf16 and a CPU box cannot honour it, so the record says both rather than
    # letting the declared one stand for what actually ran.
    assert training["precision_declared"] == "bf16"
    assert training["precision_used"] == "fp32"
    assert training["device"] == "cpu"


def test_the_head_checkpoints_every_epoch(box, flags, stub, frozen):
    """§7: heads checkpoint per epoch like everything else in this repo."""
    assert launch.main(["run", "--arm", "A6", *flags]) == 0
    written = sorted((box / "heads").glob("A6-20k-seed0/*.pt"))
    assert [p.name for p in written] == ["best.pt", "epoch000.pt", "epoch001.pt"]


def test_three_seeds_land_three_sets_of_rows(box, flags, stub, frozen):
    """§4.1's three seeds per primary cell — the shape A6 ships in with this ticket."""
    for seed in (0, 1, 2):
        assert launch.main(["run", "--arm", "A6", "--seed", str(seed), *flags]) == 0
    table = table_of(box)
    assert sorted(set(table.column("seed").to_pylist())) == [0, 1, 2]
    assert table.num_rows == 3 * len(splits_of(box)["test"]) * 2 * 3


def test_the_encoder_is_asserted_frozen_on_the_way_out(box, flags, stub, frozen, monkeypatch):
    """A frozen arm whose encoder trained would win the table and no column would say why."""
    original = train.assert_encoder_frozen
    seen = []

    def spy(model, *, name, before=None):
        seen.append((name, before is not None))
        return original(model, name=name, before=before)

    monkeypatch.setattr(train, "assert_encoder_frozen", spy)
    assert launch.main(["run", "--arm", "A6", *flags]) == 0
    assert seen == [("imagenet-mae", True)], "the check must compare weights, not just flags"


def test_an_untrained_head_is_not_something_the_pass_will_score(box, flags, stub, frozen):
    """A zero-output head scores exactly A1, which is the quietest way for an arm to be wrong."""
    cfg = launch.resolve(launch.build_parser().parse_args(["run", "--arm", "A6", *flags])).config
    with pytest.raises(NotImplementedError, match="no trained head"):
        evaluate.FILLERS["a1_fill_plus_mask"]([], [], cfg, None)


# --- The tuning pass through the shipping command ------------------------------


def test_the_tuning_pass_writes_its_record_and_scores_nothing(box, flags, stub, frozen, capsys):
    """§4.1's one pass: the grid on val, a record beside the artifacts, and no printed row."""
    assert (
        launch.main(
            ["run", "--arm", "A6", "--tune", *flags]
        )
        == 0
    )
    out = capsys.readouterr().out
    # Read back through the module's own reader, which is also what `assert_frozen` consults.
    written = tune.read_tuning(box / "artifacts" / tune.TUNING_NAME)

    assert written["arm"] == "A6"
    assert len(written["searched"]) == len(tune.points())
    assert set(written["selected"]) == set(tune.GRID)
    assert TEST_TILES_NAME not in written["reads"]
    assert not (box / "results.parquet").exists(), "a val-only pass must not print a row"
    assert "scored" in out and "nothing" in out


def test_the_tuning_record_says_whether_the_frozen_recipe_survived(box, flags, stub, frozen):
    """A `False` here is a one-line edit owed to `config.TUNED_HEAD`, and the record says so."""
    cfg = launch.resolve(
        launch.build_parser().parse_args(["run", "--arm", "A6", *flags])
    ).config
    tuning = tune.tune(cfg, grid={"lr": (1e-3,)}, epochs=1)

    assert tuning.selected == {"lr": TUNED_HEAD.lr}
    assert tuning.agrees is True
    assert "unchanged" in "\n".join(tuning.lines())
