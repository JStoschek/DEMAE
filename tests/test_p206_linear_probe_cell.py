"""`run --task landform`: one linear-probe cell end to end, its parquet and its run record (#206).

**Seam 1** is :func:`benchmark.launch.main`. A B1 cell on the fixture corpus trains §4.1's
linear probe over a stub encoder, scores the test tiles and appends per-tile confusion counts to
Task 2's own parquet, with a run record beside it. Every refusal the ticket names is asserted
there too, by the words it refuses in.

A few numbers cannot be seen from the command and are checked below it, as the spec allows: the
probe's parameter count, the cycling sampler, the ten val reads, the dihedral pairing and the
per-tile counts.

## The box

Built for real wherever a real writer exists: the corpus by :func:`tests.conftest.build_corpus`,
the ladders by ``freeze --task landform``'s own ladder stage, the label rasters and resume log by
:class:`benchmark.geomorphon_pass.LabelStore`, the fingerprints by its renderer, the recipe line
by :func:`benchmark.landform_recipes.write_line` (#207's writer), and the bar by copying the
committed one. Two things are faked: the labels (a function of the tile, standing in for WBT) and
the encoder (:class:`StubExtractor`, whose checkpoint is a few bytes pinned for the test).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from benchmark import arms, geomorphon_pass, landform, landform_recipes, launch, results
from benchmark.config import BenchmarkConfig
from benchmark.landform_config import GEOMORPHON_CLASSES, LandformConfig, build_config
from pretrain.artifact import write_tile_strata
from tests.conftest import build_corpus

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "docs" / "research" / "task2-landform-plan.md"
COMMITTED_BAR = REPO / "artifacts" / "task2" / "task2-prereg-bar.json"

TILES_PER_COG = 2
#: Three train geocells of one COG each, a val COG, and a test geocell of two COGs.
COGS = [
    ("Copernicus_DSM_COG_10_N10_00_E000_00_DEM", "+000_+010", "train"),
    ("Copernicus_DSM_COG_10_N10_00_E001_00_DEM", "+010_+010", "train"),
    ("Copernicus_DSM_COG_10_N10_00_E002_00_DEM", "+020_+010", "train"),
    ("Copernicus_DSM_COG_10_N10_00_E003_00_DEM", "+030_+010", "val"),
    ("Copernicus_DSM_COG_10_N20_00_E010_00_DEM", "+040_+020", "test"),
    ("Copernicus_DSM_COG_10_N20_00_E011_00_DEM", "+040_+020", "test"),
]
TRAIN_TILES = 3 * TILES_PER_COG
#: The fixture's rungs: 2 tiles at 1%, so a batch of 4 spans two passes of the sampler.
BUDGETS = (2, 4, TRAIN_TILES)

#: A test-scale linear recipe: batch 4, 5 epochs, floor 8 tiles -> 5 x ceil(8 / 4) = 10 steps
#: at the 1% rung, so the ten val reads land one per step.
TEST_RECIPE = [
    "--set", "linear.batch_size=4",
    "--set", "linear.epochs=5",
    "--set", "schedule.step_floor=8",
]
STEPS = 10
WIDTH = arms.ENCODERS["demae-r5"].width
CHECKPOINT_BYTES = b"a stub encoder for #206"


# --- The stub encoder ---------------------------------------------------------


class _StubEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)


class StubExtractor:
    """ADR-0012's surface by registry encoder name, and nothing behind it.

    The tokens are a 14x14 mean-pool of the per-tile-normalized input, broadcast to the
    registered width, so a probe can learn something from them. The masks it is handed are kept,
    so a test can see what the run called valid.
    """

    masks: list = []

    def __init__(self, spec=None, cfg=None, *, device: str = "cpu") -> None:
        self.spec = spec
        self.device = device
        self.precision: dict = {}
        self._model = _StubEncoder()

    def encoder(self, arm=None, *, encoder_name=None):
        self._model.eval()
        return self._model

    def precision_agreement(self, arm=None, *, encoder_name=None):
        return None

    def extract(self, raw_i16, valid_mask, *, arm=None, encoder_name=None):
        from benchmark import extract as extract_module

        raw, mask = extract_module._as_batch(raw_i16, valid_mask)
        StubExtractor.masks.append((raw.copy(), mask.copy()))
        x = extract_module.normalize(raw, mask, normalization="per_tile")
        tile = torch.from_numpy(x).unsqueeze(1)
        pooled = torch.nn.functional.adaptive_avg_pool2d(tile, (14, 14))
        width = arms.ENCODERS[encoder_name].width
        tokens = pooled.reshape(tile.shape[0], 1, 196).transpose(1, 2).repeat(1, 1, width)
        return extract_module.Features(
            output=tokens.float(), intermediates=tuple(tokens.float() for _ in range(12)), cls=None
        )


@pytest.fixture
def stub(monkeypatch):
    from benchmark import extract as extract_module

    StubExtractor.masks = []
    monkeypatch.setattr(extract_module, "Extractor", StubExtractor)
    return StubExtractor


# --- The box ------------------------------------------------------------------


def fake_label(raw: np.ndarray) -> np.ndarray:
    """A deterministic stand-in for WBT: the tile's elevation cut into ten bands, a corner
    ignored (as ocean would be)."""
    edges = np.quantile(raw, np.linspace(0, 1, 11)[1:-1])
    label = np.digitize(raw, edges).astype(np.uint8)
    label[:8, :8] = 255
    return label


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("p206_corpus")
    return build_corpus(
        root,
        COGS,
        tiles_per_cog=TILES_PER_COG,
        metrics_fn=lambda c, j: {"stratum_id": (2 * c + j) % 20, "slope_mean": 0.5 + c + 0.1 * j},
    )


def _split_lists(corpus) -> dict[str, list[tuple[str, int]]]:
    rows = pq.read_table(
        corpus / "manifest.parquet", columns=["tile_id", "stratum_id", "split"]
    ).to_pylist()
    out: dict[str, list] = {"train": [], "val": [], "test": []}
    for row in rows:
        out[row["split"]].append((row["tile_id"], row["stratum_id"]))
    return {k: sorted(v) for k, v in out.items()}


def _flags(box, corpus) -> list[str]:
    return [
        "--task", "landform",
        "--set", f'paths.artifacts_dir="{box / "task2"}"',
        "--set", f'paths.task1_artifacts_dir="{box / "task1"}"',
        "--set", f'paths.results_path="{box / "results.parquet"}"',
        "--set", f'paths.corpus_dir="{corpus}"',
        "--set", f'paths.labels_dir="{box / "labels"}"',
        "--set", f'paths.plan_path="{PLAN}"',
        "--set", f'paths.demae_r5_checkpoint="{box / "release" / "encoder.pt"}"',
        "--set", f'paths.imagenet_mae_checkpoint="{box / "release" / "encoder.pt"}"',
        "--set", f'runtime.checkpoint_dir="{box / "heads"}"',
        "--set", 'runtime.device="cpu"',
        "--set", "runtime.loader_workers=0",
        "--set", f"ladders.budgets={list(BUDGETS)}",
        *TEST_RECIPE,
    ]


def pin_stub_checkpoint(monkeypatch, *encoders: str) -> str:
    """Pin these registry encoders to the stub checkpoint's md5 — the reviewed line a real
    checkpoint has, made for a file the test can write."""
    md5 = hashlib.md5(CHECKPOINT_BYTES).hexdigest()
    for name in encoders:
        monkeypatch.setitem(
            arms.ENCODERS, name, dataclasses.replace(arms.ENCODERS[name], checkpoint_md5=md5)
        )
    return md5


@pytest.fixture
def box(tmp_path, corpus, monkeypatch, capsys):
    """A frozen Task 2 box at test scale, with a recipe line for linear at every rung."""
    from benchmark import extract as extract_module
    from benchmark.evaluate import TileStore

    pin_stub_checkpoint(monkeypatch, "demae-r5", "imagenet-mae")
    lists = _split_lists(corpus)
    task1, task2 = tmp_path / "task1", tmp_path / "task2"
    task2.mkdir(parents=True)
    write_tile_strata(lists["train"], task1 / "head_train_tiles.csv")
    write_tile_strata(lists["val"], task1 / "val_tiles.csv")
    write_tile_strata(lists["test"], task1 / "test_tiles.csv")
    base = BenchmarkConfig()
    extract_module.write_spec(base.replace(paths=base.paths.replace(artifacts_dir=str(task1))))
    (tmp_path / "release").mkdir()
    (tmp_path / "release" / "encoder.pt").write_bytes(CHECKPOINT_BYTES)
    shutil.copy(COMMITTED_BAR, task2 / COMMITTED_BAR.name)

    # The labels: rasters and resume log where the label pass leaves them, then the committed
    # fingerprints and config the way the geomorphon stage writes them.
    cfg = build_config(overrides=_flags(tmp_path, corpus)[2:][1::2])
    chash = geomorphon_pass.label_config_hash(cfg.geomorphon)
    store = geomorphon_pass.LabelStore(tmp_path / "labels" / chash, "fake-labeller")
    tiles = TileStore(corpus)
    records = [
        store.put("primary", t, fake_label(tiles.read(t).raw_i16), "fake-input")
        for split in ("train", "val", "test")
        for t, _ in lists[split]
    ]
    fingerprints = geomorphon_pass.fingerprint_path(task2, "primary")
    fingerprints.parent.mkdir()
    fingerprints.write_bytes(geomorphon_pass.render_fingerprints(records))
    (task2 / geomorphon_pass.CONFIG_NAME).write_text(json.dumps({
        "label_config_hash": chash,
        "variants": {"primary": {
            "tiles": len(records),
            "set_fingerprint": geomorphon_pass.set_fingerprint(records),
            "fingerprints_md5": hashlib.md5(fingerprints.read_bytes()).hexdigest(),
        }},
        "class_weights": {c: 1.0 + 0.1 * i for i, c in enumerate(GEOMORPHON_CLASSES)},
    }, indent=2))

    # The ladders, by freeze's own ladder stage.
    every = landform.FREEZE_STAGES
    monkeypatch.setattr(
        landform, "FREEZE_STAGES", tuple(s for s in every if s.label == "ladders")
    )
    assert launch.main(["freeze", *_flags(tmp_path, corpus)]) == 0
    monkeypatch.setattr(landform, "FREEZE_STAGES", every)

    # The recipe lines #207 will write, here at test scale, as though B6 had tuned them.
    for rung in ("1%", "10%", "100%"):
        landform_recipes.write_line(
            cfg.replace(arm="B6", rung=rung), lr=1e-2, scores={"0.01": 0.5}
        )
    capsys.readouterr()
    return tmp_path


@pytest.fixture
def flags(box, corpus):
    return _flags(box, corpus)


def cell(*argv: str) -> list[str]:
    return ["run", "--rung", "1%", "--seed", "0", *argv]


def table_of(box):
    return pq.read_table(box / "results.parquet")


def records_of(box) -> list[dict]:
    return results.read_run_records(results.run_record_path(box / "results.parquet"))


# --- The tracer bullet ----------------------------------------------------------


def test_a_b1_cell_appends_confusion_rows_and_a_run_record_carrying_every_hash(
    box, flags, stub, capsys
):
    assert launch.main([*cell(), *flags]) == 0
    table = table_of(box)
    assert table.num_rows == 2 * TILES_PER_COG, "one row per test tile"
    row = table.to_pylist()[0]
    assert (row["arm"], row["rung"], row["seed"], row["ladder"]) == ("B1", "1%", 0, 0)
    assert (row["label_set"], row["variant"]) == ("geomorphon", "primary")
    confusion = np.asarray(row["confusion"])
    assert confusion.shape == (100,)
    label = np.load(box / "labels" / geomorphon_pass.label_config_hash(
        LandformConfig().geomorphon) / "primary" / f"{row['tile_id']}.npy")
    assert confusion.sum() == int((label != 255).sum()), "counts over non-ignore pixels only"

    (record,) = records_of(box)
    cfg = build_config(arm="B1", rung="1%", seed=0, overrides=flags[2:][1::2])
    assert record["config_hash"] == cfg.config_hash
    spec = box / "task1" / "extractor_spec.json"
    assert record["extractor_spec_md5"] == hashlib.md5(spec.read_bytes()).hexdigest()
    committed = json.loads((box / "task2" / geomorphon_pass.CONFIG_NAME).read_text())
    assert record["labels"]["label_config_hash"] == committed["label_config_hash"]
    assert record["labels"]["set_fingerprint"] == committed["variants"]["primary"][
        "set_fingerprint"]
    ladder = box / "task2" / "ladder_0.csv"
    assert record["ladder_md5"] == hashlib.md5(ladder.read_bytes()).hexdigest()
    assert record["recipe_line"]["lr"] == 1e-2 and record["recipe_line"]["tuned_on"]["arm"] == "B6"
    assert record["git_sha"]
    assert record["parameters"] == {
        "encoder": 85_254_144, "trainable": 9_226, "total": 85_254_144 + 9_226,
    }
    assert record["class_collapse"]["count"] == len(record["class_collapse"]["classes"])
    assert record["timing"]["head"] == "linear" and record["timing"]["seconds_per_step"] > 0
    assert record["training"]["steps"] == STEPS


def _corpus_of(flags) -> Path:
    value = next(v for v in flags if v.startswith("paths.corpus_dir="))
    return Path(value.split("=", 1)[1].strip('"'))


def test_the_extractor_is_handed_the_land_mask_and_every_test_tile_is_scored(box, flags, stub):
    """Story 41: valid pixels are land, which the extractor recomputes its statistics from."""
    launch.main([*cell(), *flags])
    ids = sorted(row["tile_id"] for row in table_of(box).to_pylist())
    assert ids == [t for t, _ in _split_lists(_corpus_of(flags))["test"]]
    assert StubExtractor.masks, "the encoder was never called"
    for raw, mask in StubExtractor.masks:
        np.testing.assert_array_equal(mask, raw != 0)  # land: everything but exact-zero ocean


# --- L20: appended once --------------------------------------------------------


def test_a_second_write_of_a_finished_cell_is_refused_and_writes_nothing(box, flags, stub):
    assert launch.main([*cell(), *flags]) == 0
    before = (box / "results.parquet").read_bytes()
    with pytest.raises(ValueError, match="this cell is finished.*never replaced \\(L20\\)"):
        launch.main([*cell(), *flags])
    assert (box / "results.parquet").read_bytes() == before
    assert len(records_of(box)) == 1


def test_a_rerun_with_an_infrastructure_reason_is_appended_beside_the_first_and_recorded(
    box, flags, stub
):
    assert launch.main([*cell(), *flags]) == 0
    assert launch.main([*cell("--rerun-reason", "oom"), *flags]) == 0
    rows = table_of(box).to_pylist()
    assert sorted({row["attempt"] for row in rows}) == [0, 1]
    assert len(rows) == 2 * 2 * TILES_PER_COG, "the finished run's rows are kept"
    first, second = records_of(box)
    assert "rerun" not in first
    assert second["rerun"] == {"reason": "oom", "attempt": 1}
    assert first["run_id"] != second["run_id"]


def test_a_rerun_reason_that_is_not_an_infrastructure_failure_is_refused(flags, capsys):
    with pytest.raises(SystemExit):
        launch.main([*cell("--rerun-reason", "class-collapse"), *flags])
    assert "invalid choice" in capsys.readouterr().err


def test_another_seed_on_the_same_parquet_is_a_new_cell(box, flags, stub):
    assert launch.main([*cell(), *flags]) == 0
    assert launch.main(["run", "--rung", "1%", "--seed", "1", *flags]) == 0
    assert {(r["seed"], r["ladder"]) for r in table_of(box).to_pylist()} == {(0, 0), (1, 1)}


def test_rerun_reason_is_refused_under_task_one_by_name():
    with pytest.raises(ValueError, match="`--rerun-reason` is a landform cell flag"):
        launch.main(["run", "--dry-run", "--rerun-reason", "crash"])


# --- Refused by name -----------------------------------------------------------


def test_an_unpinned_encoder_is_refused(box, flags, stub, monkeypatch):
    monkeypatch.setitem(
        arms.ENCODERS, "demae-r5",
        dataclasses.replace(arms.ENCODERS["demae-r5"], checkpoint_md5=None),
    )
    with pytest.raises(ValueError, match="demae-r5 encoder is unpinned"):
        launch.main([*cell(), *flags])
    assert not (box / "results.parquet").exists()


def test_a_checkpoint_that_is_not_the_pinned_one_is_refused(box, flags, stub):
    (box / "release" / "encoder.pt").write_bytes(b"some other weights")
    with pytest.raises(ValueError, match="demae-r5 checkpoint .* its pin is"):
        launch.main([*cell(), *flags])
    assert not (box / "results.parquet").exists()


def test_a_recipe_that_drifted_from_its_frozen_line_is_refused_before_the_dry_run(
    flags, stub, capsys
):
    with pytest.raises(ValueError, match="recipe drifted from its frozen line.*linear.epochs 6"):
        launch.main([*cell("--dry-run"), *flags, "--set", "linear.epochs=6"])
    assert capsys.readouterr().out == ""


def test_a_cell_with_no_frozen_line_is_refused_naming_the_tuning_ticket(box, flags, stub):
    (box / "task2" / landform_recipes.RECIPE_LINES_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="no frozen recipe line for linear @ 1%.*#207"):
        launch.main([*cell(), *flags])


def test_a_seed_paired_with_another_ladder_is_refused(box, flags, stub):
    shutil.copy(box / "task2" / "ladder_1.csv", box / "task2" / "ladder_0.csv")
    with pytest.raises(ValueError, match="seed paired with another ladder"):
        launch.main([*cell(), *flags])
    with pytest.raises(ValueError, match="seed 3 has no ladder"):
        launch.main(["run", "--rung", "1%", "--seed", "3", *flags])


def test_a_label_raster_that_does_not_match_its_fingerprint_is_refused(box, flags, stub):
    chash = geomorphon_pass.label_config_hash(LandformConfig().geomorphon)
    test_id = _split_lists(_corpus_of(flags))["test"][0][0]
    path = box / "labels" / chash / "primary" / f"{test_id}.npy"
    label = np.load(path)
    label[100, 100] = (label[100, 100] + 1) % 10
    np.save(path, label)
    with pytest.raises(ValueError, match="label fingerprint mismatch: 1 of 8"):
        launch.main([*cell(), *flags])
    assert not (box / "results.parquet").exists()


# --- Task 1 untouched ----------------------------------------------------------


def test_task_two_rows_go_to_their_own_parquet_and_task_one_s_file_is_refused(box, flags, stub):
    """Task 2's rows are their own schema; pointed at a Task 1 parquet, the run refuses rather
    than merging into it."""
    assert launch.main([*cell(), *flags]) == 0
    assert "confusion" in table_of(box).schema.names
    assert "confusion" not in results.schema(["rmse"]).names
    task1 = box / "task1-results.parquet"
    results.append_rows([], task1, metrics=["rmse"])
    before = task1.read_bytes()
    with pytest.raises(ValueError, match="not Task 2's"):
        launch.main([*cell(), *flags, "--set", f'paths.results_path="{task1}"'])
    assert task1.read_bytes() == before


# --- L10: class collapse -------------------------------------------------------


def test_class_collapse_is_flagged_and_counted_never_excluded(box, flags, stub):
    """Ten steps over channel-constant stub tokens collapse classes: each is named in the
    record, and its pixels are still in the rows."""
    launch.main([*cell(), *flags])
    (record,) = records_of(box)
    collapse = record["class_collapse"]
    assert collapse["count"] == len(collapse["classes"]) > 0
    total = sum(np.asarray(r["confusion"]) for r in table_of(box).to_pylist()).reshape(10, 10)
    for name in collapse["classes"]:
        c = GEOMORPHON_CLASSES.index(name)
        assert total[c].sum() > 0 and total[c, c] == 0


def test_b6_b1_r4_and_b1_s_run_the_same_cell_through_the_same_command(
    box, flags, stub, monkeypatch
):
    """The other linear rows: each its own encoder, the same probe at that encoder's width."""
    from benchmark import extract as extract_module

    pin_stub_checkpoint(monkeypatch, "demae-r4", "demae-vits")
    spec = box / "task1" / "extractor_spec.json"
    spec.unlink()  # the pins moved, so the contract is re-frozen, as a real pin change would be
    base = BenchmarkConfig()
    extract_module.write_spec(base.replace(paths=base.paths.replace(artifacts_dir=str(box / "task1"))))
    checkpoint = box / "release" / "encoder.pt"
    mounts = [
        "--set", f'paths.demae_r4_checkpoint="{checkpoint}"',
        "--set", f'paths.demae_vits_checkpoint="{checkpoint}"',
    ]
    for arm in ("B6", "B1-R4", "B1-S"):
        assert launch.main([*cell("--arm", arm), *flags, *mounts]) == 0
    trained = {r["cell"]["arm"]: r["parameters"]["trainable"] for r in records_of(box)}
    assert trained == {"B6": 9_226, "B1-R4": 9_226, "B1-S": 4_618}
    assert {r["arm"] for r in table_of(box).to_pylist()} == {"B6", "B1-R4", "B1-S"}


# --- Below the seam: numbers the command cannot show -----------------------------

from benchmark import landform_train, probe  # noqa: E402


@pytest.mark.parametrize("width, count", [(768, 9_226), (384, 4_618)])
def test_the_linear_probe_is_a_layernorm_and_a_one_by_one_conv_to_ten_classes(width, count):
    """LayerNorm's 2 x d, then d x 10 weights and 10 biases: 1,536 + 7,690 at 768-d."""
    built = probe.build(width)
    assert built.parameters == count == probe.parameter_count(width)
    logits = built.module(torch.zeros(2, 196, width))
    assert tuple(logits.shape) == (2, 10, 224, 224)


def test_b1_and_b6_build_one_probe_differing_in_nothing_but_the_encoder():
    shapes = {
        arm: {k: tuple(v.shape) for k, v in probe.build(
            arms.ENCODERS[landform.landform_arms.get(arm).encoder].width
        ).module.state_dict().items()}
        for arm in ("B1", "B6", "B1-R4")
    }
    assert shapes["B1"] == shapes["B6"] == shapes["B1-R4"]


def test_the_sampler_cycles_its_tiles_reshuffled_each_pass_and_a_batch_spans_two_passes():
    """Story 48 at the 1% rung's shape: 3 tiles, batch 2."""
    ids = ["a", "b", "c"]
    batches = landform_train.cycling_batches(ids, 2, np.random.default_rng(0))
    stream = [t for _ in range(6) for t in next(batches)]
    assert all(len(b) == 2 for b in [stream[i:i + 2] for i in range(0, 12, 2)])
    passes = [stream[i:i + 3] for i in range(0, 12, 3)]
    assert all(sorted(p) == ids for p in passes), "each pass is a permutation of the rung"
    assert len({tuple(p) for p in passes}) > 1, "reshuffled between passes"


@pytest.mark.parametrize(
    "steps, want",
    [(240, (24, 48, 72, 96, 120, 144, 168, 192, 216, 240)),
     (2_370, (237, 474, 711, 948, 1185, 1422, 1659, 1896, 2133, 2370)),
     (25, (3, 5, 8, 10, 13, 15, 18, 20, 23, 25))],
)
def test_val_is_read_at_each_tenth_of_the_steps_the_last_step_included(steps, want):
    assert landform_train.val_read_steps(steps, 10) == want


def test_the_dihedral_turn_moves_the_tile_and_its_label_together():
    """The label is a function of the tile pixel by pixel, so after a joint turn it still is."""
    raw = np.arange(224 * 224, dtype=np.int16).reshape(224, 224) % 997
    label = (raw % 10).astype(np.uint8)
    seen = set()
    for k in range(landform_train.DIHEDRAL):
        r, lab = landform_train.augment(raw, label, k)
        np.testing.assert_array_equal(lab, (r % 10).astype(np.uint8))
        seen.add(r.tobytes())
    assert len(seen) == 8, "eight distinct symmetries of the square"


def test_the_confusion_counts_only_non_ignore_pixels_truth_major():
    truth = np.array([[0, 0, 1], [2, 255, 1]], dtype=np.uint8)
    pred = np.array([[0, 1, 1], [2, 3, 0]], dtype=np.uint8)
    m = landform_train.confusion(truth, pred, ignore=255).reshape(10, 10)
    assert m.sum() == 5
    assert (m[0, 0], m[0, 1], m[1, 1], m[1, 0], m[2, 2]) == (1, 1, 1, 1, 1)


def test_boundary_counts_match_at_two_pixels_and_miss_at_three():
    """A vertical class edge (boundary pixels in columns 9 and 10), predicted 2 then 3 columns
    off. A boundary pixel is matched when it lies within 2 px of the other map's boundary."""
    truth = np.zeros((20, 20), dtype=np.uint8)
    truth[:, 10:] = 1
    near, far = truth.copy(), truth.copy()
    near[:, 8:10] = 1  # predicted boundary in columns 7 and 8: 2 and 1 px from column 9
    far[:, 7:10] = 1   # predicted boundary in columns 6 and 7: 3 and 2 px from column 9
    exact = landform_train.boundary_counts(truth, truth, ignore=255)
    assert exact["bf1_true_boundary"] == exact["bf1_true_matched"] == 40
    within = landform_train.boundary_counts(truth, near, ignore=255)
    assert within["bf1_pred_matched"] == within["bf1_pred_boundary"] == 40
    beyond = landform_train.boundary_counts(truth, far, ignore=255)
    assert beyond["bf1_pred_boundary"] == 40
    assert beyond["bf1_pred_matched"] == 20  # column 7 only
    assert beyond["bf1_true_matched"] == 20  # column 9 only (2 px from column 7)


def _lines_cfg(tmp_path, **cell) -> LandformConfig:
    cfg = LandformConfig(arm="B6", **cell)
    return cfg.replace(paths=cfg.paths.replace(artifacts_dir=str(tmp_path)))


def test_a_recipe_line_is_written_once_idempotently_and_never_replaced(tmp_path):
    """#207's writer: one line per head per rung; the same line again is a no-op; a different
    lr for a frozen rung is refused."""
    cfg = _lines_cfg(tmp_path, rung="10%")
    path, line = landform_recipes.write_line(cfg, lr=3e-3, scores={"0.003": 0.4})
    first = path.read_bytes()
    landform_recipes.write_line(cfg, lr=3e-3, scores={"0.003": 0.4})
    assert path.read_bytes() == first
    with pytest.raises(ValueError, match="already freezes linear @ 10% at lr 0.003"):
        landform_recipes.write_line(cfg, lr=1e-2)
    body = json.loads(first)
    assert body["lines"]["linear"]["10%"]["recipe"] == {
        "optimizer": "adamw", "weight_decay": 0.0, "batch_size": 256, "epochs": 30,
    }
    assert body["lines"]["linear"]["10%"]["tuned_on"] == {"arm": "B6", "seed": 0, "ladder": 0}
    assert landform_recipes.line_for(cfg.replace(arm="B1")) == line


def test_only_the_tuning_arm_writes_a_line_and_a_line_from_another_arm_is_refused(tmp_path):
    with pytest.raises(ValueError, match="tunes it on B6"):
        landform_recipes.write_line(_lines_cfg(tmp_path).replace(arm="B1"), lr=1e-3)
    cfg = _lines_cfg(tmp_path)
    path, _ = landform_recipes.write_line(cfg, lr=1e-3)
    body = json.loads(path.read_text())
    body["lines"]["linear"]["100%"]["tuned_on"]["arm"] = "B1"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="tuned on B1; §4.5 tunes the linear head on B6 alone"):
        landform_recipes.line_for(cfg)


def test_a_widened_grid_is_provenance_and_not_drift(tmp_path):
    cfg = _lines_cfg(tmp_path)
    landform_recipes.write_line(cfg, lr=1e-1, grid=(3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1))
    assert landform_recipes.line_for(cfg).lr == 1e-1


def test_a_class_present_in_truth_at_iou_zero_is_collapse_and_an_absent_one_is_not():
    m = np.zeros((10, 10), dtype=np.int64)
    m[0, 0] = 5  # flat: right
    m[1, 0] = 3  # peak: present, never right
    assert landform_train.class_collapse(m.ravel()) == ["peak"]
