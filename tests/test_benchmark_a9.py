"""A9 end to end through `python -m benchmark.launch run` (#168).

The same seam `tests/test_benchmark_a1.py` and `tests/test_benchmark_a2.py` hold, one arm along:
the shipping command puts the A1 fill through TerraMind's tokenizer over the committed eval mask
list, lands the rows, and writes a run record carrying A9's own precision figure and the 286.3 M
parameters it actually ran.

**The tokenizer is stubbed here and the stub is the point of the file, not a shortcut.** A real
one is 1.15 GB, needs terratorch, and produces a number no test may assert — §8's rule is that no
claim about a *result's magnitude* is tested, because the pre-registration exists to leave those
open. What #168 has to get right is the join: that the decode is asked for the tile's own raster
and checked against it, that the noise is seeded, that the round trip is written **inside the
void and nowhere else**, and that the extractor the filler loaded reaches the run record. Every
one of those is a property of the wiring, and a stub whose decode deliberately differs from the
input everywhere tests them harder than the real model would — the real decoder's output is close
to its input, so a pipeline that forgot to composite would still nearly pass.

What needs the real weights is `load`'s refusals — the md5, the constants, ADR-0011's crop, the
strict 601-tensor load — and those run under `requires_terramind` below, skipped on a box without
the cache exactly as `tests/test_benchmark_extract.py` skips its own.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from benchmark import arms, evaluate, launch, masks, metrics, passthrough, results, train
from benchmark.artifacts import EVAL_MASKS_NAME, TEST_TILES_NAME
from benchmark.extract import SPEC_FILENAME
from benchmark.config import BenchmarkConfig, Tokenizer
from pretrain.artifact import write_tile_strata
from tests.conftest import build_corpus
from tests.test_benchmark_a1 import TEST_COGS, TILES_PER_COG, _slope_mean

REPO = Path(__file__).resolve().parents[1]
COMMITTED = REPO / "artifacts" / "task1"
PLAN = REPO / "docs" / "research" / "task1-void-fill-plan.md"

ARM = "A9"

#: TerraMind's DEM affine (ADR-0012 §7.2), quoted so the arithmetic tests below run on a box with
#: no terratorch. `benchmark.extract` is where the pair is *selected by name and checked by value*;
#: `test_benchmark_extract.py` is what pins these two numbers against the installed package.
MEAN, STD = 670.665, 951.272


# --- A tokenizer-shaped stub ---------------------------------------------------


class StubTokenizer:
    """Shaped like `DiVAE` where A9 touches it, and deliberately unlike it everywhere else.

    Its decode is a **constant offset from a per-batch noise draw** rather than a reconstruction:
    it ignores the latent's content, so the array it returns is nowhere near the tile it came
    from. That is what makes the compositing test a real one — every pixel of this decode is
    wrong, so a fill that wrote outside the void misses by hundreds of metres and
    `benchmark.metrics.score` refuses it.
    """

    image_size = 256

    def __init__(self, *, raster: int | None = None) -> None:
        #: What `decode_quant` returns instead of the raster it was asked for. `None` honours the
        #: request; an int is #163's finding 1, reproduced.
        self.raster = raster
        self.calls: list[dict] = []

    def encode(self, tensor):
        batch = tensor.shape[0]
        return torch.zeros((batch, 5, 14, 14)), None, None

    def decode_quant(self, quant, *, timesteps, image_size, verbose, generator):
        self.calls.append(
            {"timesteps": timesteps, "image_size": image_size, "seeded": generator is not None}
        )
        px = image_size if self.raster is None else self.raster
        noise = torch.randn((quant.shape[0], 1, px, px), generator=generator)
        # Far from any real tile, in TerraMind's normalized units: 8 sigma is ~7.6 km of offset
        # once `to_metres` has run, which no corpus tile is within.
        return noise + 8.0


def stub_pass_through(*, raster: int | None = None) -> passthrough.PassThrough:
    """A `PassThrough` over the stub. `load`'s own refusals are tested with the real weights."""
    return passthrough.PassThrough(
        model=StubTokenizer(raster=raster),
        device="cpu",
        timesteps=BenchmarkConfig().tokenizer.timesteps,
        seed=0,
        mean=MEAN,
        std=STD,
    )


# --- The box -------------------------------------------------------------------


@pytest.fixture
def box(tmp_path):
    """`tests/test_benchmark_a1.py`'s box, plus the two inputs A9 reads and A1 does not.

    The **extractor spec**, because A9 runs an encoder the committed contract governs — unlike
    A2, whose mechanism is the pretrained decoder and which `benchmark.artifacts._required_for`
    therefore exempts. And a file at `paths.terramind_checkpoint`, because `launch._run` refuses
    an arm whose encoder this box has not mounted *before* it reaches the filler; its bytes are
    never read here, since `passthrough.load` is stubbed.
    """
    corpus = build_corpus(
        tmp_path / "corpus", TEST_COGS, tiles_per_cog=TILES_PER_COG, metrics_fn=_slope_mean
    )
    directory = tmp_path / "artifacts"
    directory.mkdir()

    table = pq.read_table(corpus / "manifest.parquet", columns=["tile_id", "stratum_id"])
    tile_ids = sorted(table.column("tile_id").to_pylist())
    write_tile_strata([(tile_id, 0) for tile_id in tile_ids], directory / TEST_TILES_NAME)
    masks.write_eval_masks(
        masks.eval_rows(tile_ids),
        directory / EVAL_MASKS_NAME,
        artifact_hash=BenchmarkConfig().artifact_hash,
        real=masks.RealFamily.none(),
    )
    for name in (
        "task1-prereg-bar.json",
        SPEC_FILENAME,
        metrics.GOLDEN_ARRAYS,
        metrics.GOLDEN_SIDECAR,
    ):
        shutil.copy(COMMITTED / name, directory / name)
    (tmp_path / "tokenizer.pt").write_bytes(b"not read: `passthrough.load` is stubbed")
    return tmp_path


@pytest.fixture
def flags(box):
    return [
        "--set", f'paths.corpus_dir="{box / "corpus"}"',
        "--set", f'paths.artifacts_dir="{box / "artifacts"}"',
        "--set", f'paths.results_path="{box / "results.parquet"}"',
        "--set", f'paths.mask_bank_path="{box / "train_mask_bank.parquet"}"',
        "--set", f'paths.terramind_checkpoint="{box / "tokenizer.pt"}"',
        "--set", f'paths.plan_path="{PLAN}"',
        "--set", f"tiles.test={len(TEST_COGS) * TILES_PER_COG}",
    ]


@pytest.fixture
def a9_flags(flags):
    """`flags`, plus the one A9 cells carry: the contract's reference dtype.

    The default is bf16 and `passthrough.load` refuses it — see
    `test_a9_refuses_a_cell_that_declares_a_reduced_compute_dtype`, which is this fixture's
    absence. Every `run` below goes through here, so the flag is in the record of each.
    """
    return [*flags, "--set", 'extract.compute_dtype="fp32"']


@pytest.fixture
def stubbed(monkeypatch):
    """Every `run` in this file loads the stub. Returns the one that was built."""
    built: list[passthrough.PassThrough] = []

    def install(*, raster=None):
        def load(cfg, arm, *, device=None):
            model = stub_pass_through(raster=raster)
            model.timesteps = cfg.tokenizer.timesteps
            model.seed = cfg.seed
            built.append(model)
            return model

        monkeypatch.setattr(passthrough, "load", load)
        return built

    return install


@pytest.fixture
def scored(box, a9_flags, stubbed, capsys):
    """One A9 cell through the shipping command. Returns the box, the output and the stub."""
    built = stubbed()
    assert launch.main(["run", "--arm", ARM, *a9_flags]) == 0
    return box, capsys.readouterr().out, built[0]


def table_of(box):
    return pq.read_table(box / "results.parquet")


def records_of(box):
    return results.read_run_records(results.run_record_path(box / "results.parquet"))


# --- The shipping command scores A9 -------------------------------------------


def test_the_shipping_command_writes_a9_rows_to_the_results_parquet(scored):
    """#168's first acceptance criterion: A9's rows land, at §7's cell and seed counts.

    One cell and one seed, which is §7's count for this arm and not a simplification here: §7
    lists A1, A2, A3 and A9 as single-pass, and the three-seed line covers the arms that train
    something. A9 trains nothing, so a second seed would re-run one decode against different
    noise and print a ± on a quantity nothing varied.
    """
    box, _, _ = scored
    table = table_of(box)

    assert table.num_rows == len(TEST_COGS) * TILES_PER_COG * 2 * 3
    assert set(table.column("arm").to_pylist()) == {ARM}
    assert set(table.column("init").to_pylist()) == {"terramind-tokenizer"}
    assert set(table.column("seed").to_pylist()) == {0}
    assert set(table.column("size").to_pylist()) == {"20k"}


def test_the_decode_is_asked_for_the_tiles_raster_and_not_the_decoders(scored):
    """#163's finding 1, as the thing that is passed rather than as a note in a memo.

    `decode_quant` sizes its output from the decoder's own 256 px training resolution unless told
    otherwise, so a 224 px tile comes back 256 px and every §5 metric computes against a resampled
    tile without raising.
    """
    _, _, model = scored
    calls = model.model.calls
    assert calls, "the pass decoded nothing"
    assert {call["image_size"] for call in calls} == {224}
    assert 224 != StubTokenizer.image_size, "the default is what this is guarding against"


def test_every_decode_is_seeded(scored):
    """#163's finding 2. A9 scores in a single pass with no seeds to average over, so an unseeded
    decode is unreproducible rather than merely noisy."""
    _, _, model = scored
    assert all(call["seeded"] for call in model.model.calls)


def test_the_decode_runs_at_terraminds_own_timestep_count(scored):
    _, _, model = scored
    assert {call["timesteps"] for call in model.model.calls} == {50}
    assert Tokenizer().timesteps == 50


def test_the_pass_batches_at_the_recipes_batch_size(box, a9_flags, stubbed):
    """The convention declares the batch, and for A9 the declaration is also recipe.

    Asserted through the *number of decodes*: 48 instances at a batch of 16 is three calls, and a
    filler that had ignored `Fill.batch` would show one call per instance.
    """
    built = stubbed()
    assert launch.main(["run", "--arm", ARM, *a9_flags, "--set", "tokenizer.batch_size=16"]) == 0
    instances = len(TEST_COGS) * TILES_PER_COG * 2 * 3
    assert len(built[0].model.calls) == -(-instances // 16)


# --- The fill is a fill --------------------------------------------------------


def test_the_round_trip_is_written_inside_the_void_and_nowhere_else(scored):
    """§2.4's convention, and the thing the stub's deliberately-wrong decode makes testable.

    Nothing here asserts a score. What it asserts is that the pass *produced* scores at all: the
    stub's decode is 8 sigma from any real tile, so had the composite written outside the void,
    `benchmark.metrics.score` would have refused the fill as not being the tile it was handed and
    `run` would have raised rather than returning 0.
    """
    box, _, _ = scored
    table = table_of(box)
    for name in ("rmse", "nmad", "slope_rmse"):
        values = np.array(table.column(name).to_pylist(), dtype=float)
        assert np.isfinite(values).all()


def test_composite_takes_the_decode_only_where_the_void_is():
    a1 = np.arange(16, dtype=np.float64).reshape(4, 4)
    decoded = np.full((4, 4), -999.0)
    void = np.zeros((4, 4), dtype=bool)
    void[1, 1] = void[2, 3] = True

    out = passthrough.composite(a1, decoded, void)

    assert out[1, 1] == -999.0 and out[2, 3] == -999.0
    assert np.array_equal(out[~void], a1[~void])
    assert not np.shares_memory(out, a1), "the caller's A1 fill is not the scratch space"


def test_composite_refuses_a_decode_that_is_not_finite_inside_the_void():
    """A NaN here would reach the parquet as a column of nulls rather than as a failure."""
    a1 = np.zeros((4, 4))
    decoded = np.zeros((4, 4))
    decoded[0, 0] = np.nan
    void = np.zeros((4, 4), dtype=bool)
    void[0, 0] = True

    with pytest.raises(passthrough.PassThroughError, match="non-finite"):
        passthrough.composite(a1, decoded, void)

    # ...and a non-finite pixel the void does not cover is not this function's business: it is
    # discarded by the composite before anything scores it.
    outside = np.zeros((4, 4))
    outside[3, 3] = np.inf
    void[:] = False
    void[0, 0] = True
    assert np.isfinite(passthrough.composite(a1, outside, void)).all()


def test_to_metres_is_the_exact_inverse_of_the_declared_affine():
    """ADR-0012 Decision 1: the diet is one global affine with no per-tile term."""
    from benchmark import extract

    raw = np.array([[[-120, 0, 3141], [17, 2200, 8]]], dtype=np.int16)
    mask = np.ones(raw.shape, dtype=bool)
    standardized = ((raw.astype(np.float32) - np.float32(MEAN)) / np.float32(STD)).astype(
        np.float32
    )

    assert passthrough.to_metres(standardized, mean=MEAN, std=STD) == pytest.approx(
        raw.astype(np.float64), abs=1e-2
    )
    assert extract.normalize.__doc__  # the forward half this inverts lives there, not here


# --- Reproducibility -----------------------------------------------------------


def test_a_batchs_noise_is_a_function_of_its_voids_and_not_of_its_position():
    """Why `batch_key` hashes the masks instead of counting batches.

    A counter would make an instance's noise a fact about how far a pass had got, so a resumed or
    re-ordered pass would score the same instance against different noise and no column would say
    so.
    """
    first = [np.zeros((8, 8), dtype=bool), np.ones((8, 8), dtype=bool)]
    first[0][2, 3] = True

    assert passthrough.batch_key(first) == passthrough.batch_key(list(first))
    assert passthrough.batch_key(first) != passthrough.batch_key(first[::-1])
    assert passthrough.batch_key(first) != passthrough.batch_key(first[:1])

    moved = [first[0].copy(), first[1]]
    moved[0][2, 3], moved[0][4, 5] = False, True
    assert passthrough.batch_key(moved) != passthrough.batch_key(first)


def test_the_same_cell_and_the_same_batch_draw_the_same_noise():
    a = passthrough.generator_for(0, "abc")
    b = passthrough.generator_for(0, "abc")
    assert torch.equal(torch.randn(64, generator=a), torch.randn(64, generator=b))

    for other in (passthrough.generator_for(1, "abc"), passthrough.generator_for(0, "abd")):
        again = passthrough.generator_for(0, "abc")
        assert not torch.equal(
            torch.randn(64, generator=again), torch.randn(64, generator=other)
        )


def test_the_generator_is_a_cpu_one_whatever_the_device():
    """`PipelineCond` draws with `torch.randn(shape, generator=g)` and no `device`, then moves the
    result. A device generator raises there rather than being honoured."""
    assert passthrough.generator_for(0, "k").device.type == "cpu"


def test_two_runs_of_one_cell_produce_identical_rows(box, a9_flags, stubbed):
    """The whole of what seeding buys: A9 has no seeds to average over, so the row *is* the run.

    On CPU, where `torch.randn` and the stub's arithmetic are both exact. The claim on a card is
    narrower and is stated where it is made (`generator_for`): two runs differ by the card's
    kernel selection, not by the draw.
    """
    built = stubbed()
    assert launch.main(["run", "--arm", ARM, *a9_flags]) == 0
    first = table_of(box).column("rmse").to_pylist()
    assert launch.main(["run", "--arm", ARM, *a9_flags]) == 0
    assert table_of(box).column("rmse").to_pylist() == first
    assert len(built) == 2, "two passes, not one cached one"


def test_the_batch_size_is_recipe_because_the_noise_is_drawn_per_batch():
    """The batch moves the numbers, so it moves the config hash and the two cannot be conflated.

    `PipelineCond.__call__` draws one `torch.randn` over the whole batch, row-major, so an
    instance's noise depends on how many instances share its batch. A field outside the hash would
    let a box that had to halve the batch for memory write those rows into one parquet under one
    identity.
    """
    base = BenchmarkConfig()
    assert base.config_hash != base.replace(tokenizer=Tokenizer(batch_size=8)).config_hash
    assert base.config_hash != base.replace(tokenizer=Tokenizer(timesteps=25)).config_hash
    # ...where the device, which cannot move a number, does not.
    assert base.config_hash == base.replace(
        runtime=base.runtime.replace(device="cpu")
    ).config_hash


def test_a_tokenizer_recipe_that_cannot_run_is_refused():
    with pytest.raises(ValueError, match="timesteps"):
        Tokenizer(timesteps=0)
    with pytest.raises(ValueError, match="batch_size"):
        Tokenizer(batch_size=0)


# --- The refusals --------------------------------------------------------------


def test_a_decode_that_comes_back_at_the_wrong_raster_is_refused(box, a9_flags, stubbed):
    """#163's finding 1 again, from the other side: passed *and* checked.

    The argument is advisory as far as this module can tell — what would go wrong silently is the
    raster, so the raster is what is asserted. A terratorch that stopped honouring `image_size`
    would otherwise resample every tile in the table.
    """
    stubbed(raster=256)
    with pytest.raises(passthrough.PassThroughError, match="came back 256x256"):
        launch.main(["run", "--arm", ARM, *a9_flags])


def test_an_encoder_that_is_not_the_contracts_is_refused():
    """A5 and A9 read one encoder, so any difference at all means A9's row is about another model.

    Three ways the two builds can part, and all three refuse: a different class, a different set
    of tensor names, and a different value under one shared name. Driven with two-parameter
    stand-ins, because what is tested is that a difference *refuses* — that the real builds agree
    is `test_the_encoder_half_is_the_one_the_contract_builds`, on a box with the weights.
    """

    class Encoder(torch.nn.Module):
        def __init__(self, weight: float, *, name: str = "w") -> None:
            super().__init__()
            setattr(self, name, torch.nn.Parameter(torch.full((2, 2), weight)))

    class Other(Encoder):
        pass

    assert passthrough.assert_same_encoder(Encoder(1.0), Encoder(1.0)) == {
        "tensors": 1, "class": "Encoder", "identical": True
    }
    with pytest.raises(passthrough.PassThroughError, match="a Encoder where the contract"):
        passthrough.assert_same_encoder(Encoder(1.0), Other(1.0))
    with pytest.raises(passthrough.PassThroughError, match="differ at 'w'"):
        passthrough.assert_same_encoder(Encoder(1.0), Encoder(1.5))
    with pytest.raises(passthrough.PassThroughError, match="tensor name"):
        passthrough.assert_same_encoder(Encoder(1.0), Encoder(1.0, name="v"))


def test_the_convention_is_built_and_names_no_ticket():
    """`FILLERS` is complete so an unbuilt convention names its ticket. A9's is no longer one."""
    assert evaluate.FILLERS["a1_fill_through_tokenizer"] is evaluate._tokenizer_pass_through
    assert set(evaluate.FILLERS) == set(arms.INPUT_CONVENTIONS)


# --- The run record ------------------------------------------------------------


def test_the_run_record_says_a9_ran_at_the_reference_dtype(scored):
    """ADR-0012 §6, and how #168 settled the seam `launch._measured_agreement` had deferred.

    There is no second number to record, because there is no second dtype: A9 runs fp32 and a cell
    that declares otherwise is refused. That is `precision_block`'s third branch and it is the
    true reading — the alternative, a bf16-versus-fp32 figure measured on a ViT-B forward, would
    describe the small half of an arm whose cost and whose output are the diffusion decoder's.
    """
    box, _, _ = scored
    (record,) = records_of(box)

    assert record["precision"]["agreement"] is None
    assert record["precision"]["absent_because"] == "the run declares the reference dtype (fp32)"


def test_a9_refuses_a_cell_that_declares_a_reduced_compute_dtype(box, flags):
    """The refusal is the point: a run record that names a dtype the pass did not use is exactly
    the failure `benchmark.passthrough` is shaped around.

    Reached without stubbing `load`, because this is `load`'s own refusal — and it fires before
    the 1.15 GB read, which is why the fixture's checkpoint can stay eight words of placeholder.
    """
    assert BenchmarkConfig().extract.compute_dtype == "bf16", "the default this refuses"
    with pytest.raises(passthrough.PassThroughError, match="compute_dtype=fp32"):
        launch.main(["run", "--arm", ARM, *flags])


def test_the_run_record_counts_what_a9_actually_ran(scored):
    """§6 item 4's pair, and the one registry entry whose total is counted rather than quoted.

    286,316,821 is the whole tokenizer: p118 §8's 90,172,416 encoder, a 196,140,560 diffusion
    decoder and 3,845 across the quantizer and the post-quant projection. A9's trainable half is
    a true 0 — it is the arm that trains nothing over a counted encoder.
    """
    box, _, _ = scored
    (record,) = records_of(box)
    assert record["parameters"] == {"total": 286_316_821, "trainable": 0}
    assert arms.ENCODERS["terramind-tokenizer"].params == 90_172_416 + 196_140_560 + 3_845


def test_the_record_carries_the_tokenizer_recipe_the_run_was_launched_with(scored):
    box, _, _ = scored
    (record,) = records_of(box)
    assert record["resolved"]["config"]["tokenizer"] == {"timesteps": 50, "batch_size": 32}


def test_a9_reads_the_extractor_contract_and_not_the_train_mask_bank(scored):
    """Two membership questions the registry answers from A9's declared fields.

    It consults the committed contract — unlike A2, which runs the pretrained decoder — and it
    reads none of §3.1's training draws, because it trains nothing. Both fall out of
    `trains="nothing"` and `runs_pretrained_decoder`, which is why they are fields.
    """
    box, _, _ = scored
    (record,) = records_of(box)
    artifacts = record["resolved"]["artifacts"]
    assert artifacts["extractor_spec"]["required"]
    assert not artifacts["mask_bank"]["required"]
    assert arms.get(ARM).is_inference_only
    assert not arms.get(ARM).runs_pretrained_decoder


# --- With the real weights, where a box has them -------------------------------


def _cached_terramind() -> Path | None:
    """The tokenizer, resolved **offline** — importing this module downloads nothing."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    hit = try_to_load_from_cache(
        "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM", "TerraMind_Tokenizer_DEM.pt"
    )
    return Path(hit) if isinstance(hit, str) else None


TERRAMIND = _cached_terramind()
requires_terramind = pytest.mark.skipif(
    TERRAMIND is None, reason="TerraMind's DEM tokenizer is not in this box's Hugging Face cache"
)


def _real_cfg() -> BenchmarkConfig:
    """An A9 cell pointed at the cached tokenizer, at the dtype A9 runs."""
    base = BenchmarkConfig(arm=ARM)
    return base.replace(
        paths=base.paths.replace(terramind_checkpoint=str(TERRAMIND)),
        extract=base.extract.replace(compute_dtype="fp32"),
    )


@requires_terramind
def test_the_whole_tokenizer_loads_strictly_off_the_pinned_file():
    """601 tensors, no missing and no unexpected, and the counts §6 item 4 prints.

    The registry's own `pretrained=True` would fetch from the hub and its `ckpt_path` branch loads
    `strict=False` and logs what did not match. This is the check that the shape the registry
    builds and the file `extractor_spec.json` pins are one model.
    """
    model = passthrough.load(_real_cfg(), arms.get(ARM), device="cpu")
    counts = passthrough.parameter_count(model.model)

    assert counts["encoder"] == 90_172_416 - 46_080, "ADR-0011's crop, on the loaded model"
    assert counts["decoder"] == 196_140_560
    assert arms.ENCODERS["terramind-tokenizer"].params == counts["total"] + 46_080


@requires_terramind
def test_the_encoder_half_is_the_one_the_contract_builds():
    """0.0 was #163's form of this claim. `load` makes it before a tile is read, and exactly."""
    model = passthrough.load(_real_cfg(), arms.get(ARM), device="cpu")
    from benchmark import extract

    contract = extract.Extractor(
        extract.load_spec(extract.spec_path(_real_cfg())), _real_cfg(), device="cpu"
    ).encoder(ARM)
    assert passthrough.assert_same_encoder(model.model.encoder, contract)["identical"]
