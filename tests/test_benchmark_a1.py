"""A1 end to end through `python -m benchmark.launch run` (#158).

**Seam 1**, which is where this ticket's acceptance lives: the shipping command writes rows to
the results parquet for A1 over the eval mask list, the run record carries the recipe and the
md5s the run was launched with, and both of §8's norm-inversion gates fire on the way in. The
standing defence in this repo is a test that drives `main(argv)` rather than one that drives each
piece — *the composition* is the named failure — so nothing here calls the pass directly.

The fixture is a tiny corpus and a hand-written set of committed inputs rather than a real
`freeze`. That is deliberate and it is not a shortcut: §3.1's draw refuses a train pool short of
20,000 tiles, so a `freeze`-driven fixture would need a 2 GB corpus in `tmp_path` to test a verb
that never reads the train pool. What `run` actually consumes is the *committed files*, and those
are written here exactly as `freeze` writes them — same writer, same format — which is the
contract family's own rule that the committed file is what later phases re-read.

What is not asserted anywhere here is the magnitude of any score. A1's numbers are the first this
benchmark produces and every C-claim is a comparison against them; asserting one would pre-empt
the very claims the pre-registration exists to leave open.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmark import arms, evaluate, launch, masks, metrics, results
from benchmark.artifacts import EVAL_MASKS_NAME, TEST_TILES_NAME
from benchmark.config import BenchmarkConfig
from tests.conftest import build_corpus
from pretrain.artifact import write_tile_strata

REPO = Path(__file__).resolve().parents[1]
COMMITTED = REPO / "artifacts" / "task1"
PLAN = REPO / "docs" / "research" / "task1-void-fill-plan.md"

#: Four test COGs, so the eligible pool has enough tiles for §5's four quartile bands to be
#: distinct. Every one is `test`: `run` scores the test split and draws nothing.
TEST_COGS = [
    ("Copernicus_DSM_COG_10_N45_00_E006_00_DEM", "+000_+040", "test"),
    ("Copernicus_DSM_COG_10_N45_00_E007_00_DEM", "+000_+040", "test"),
    ("Copernicus_DSM_COG_10_N46_00_E008_00_DEM", "+000_+040", "test"),
    ("Copernicus_DSM_COG_10_N46_00_E009_00_DEM", "+000_+040", "test"),
]

TILES_PER_COG = 2


def _slope_mean(cog: int, tile: int) -> dict:
    """A spread of `slope_mean`, so the quartile cut has something to cut.

    The corpus's own `stratum_id` is left at its constant: §5's bands come off `slope_mean` and
    must not come off the k-means label, and a fixture where the two agreed could not tell the
    difference.
    """
    return {"slope_mean": 0.5 + 2.0 * (cog * TILES_PER_COG + tile)}


@pytest.fixture
def box(tmp_path):
    """A box holding a corpus, the committed inputs `run` reads, and nowhere else to write."""
    corpus = build_corpus(
        tmp_path / "corpus", TEST_COGS, tiles_per_cog=TILES_PER_COG, metrics_fn=_slope_mean
    )
    directory = tmp_path / "artifacts"
    directory.mkdir()

    table = pq.read_table(corpus / "manifest.parquet", columns=["tile_id", "stratum_id"])
    tile_ids = sorted(table.column("tile_id").to_pylist())
    write_tile_strata(
        [(tile_id, 0) for tile_id in tile_ids], directory / TEST_TILES_NAME
    )
    # §3.2's own writer over §3.2's own generator: the eval mask list this scores against is the
    # artifact `freeze` would have committed for this test list, not a hand-rolled stand-in.
    # §3.3's half is deliberately absent — A1 scores whatever rows the list holds, and building
    # the real family here would want 24 granules and a GLO-30 read per cell (#153).
    masks.write_eval_masks(
        masks.eval_rows(tile_ids),
        directory / EVAL_MASKS_NAME,
        artifact_hash=BenchmarkConfig().artifact_hash,
        real=masks.RealFamily.none(),
    )
    for name in (
        "task1-prereg-bar.json",
        metrics.GOLDEN_ARRAYS,
        metrics.GOLDEN_SIDECAR,
    ):
        shutil.copy(COMMITTED / name, directory / name)

    return tmp_path


@pytest.fixture
def flags(box):
    return [
        "--set", f'paths.corpus_dir="{box / "corpus"}"',
        "--set", f'paths.artifacts_dir="{box / "artifacts"}"',
        "--set", f'paths.results_path="{box / "results.parquet"}"',
        "--set", f'paths.plan_path="{PLAN}"',
        "--set", f"tiles.test={len(TEST_COGS) * TILES_PER_COG}",
    ]


@pytest.fixture
def scored(box, flags, capsys):
    """One real A1 cell through the shipping command. Returns the box and what it printed."""
    assert launch.main(["run", "--arm", "A1", *flags]) == 0
    return box, capsys.readouterr().out


def table_of(box):
    return pq.read_table(box / "results.parquet")


def records_of(box):
    return results.read_run_records(results.run_record_path(box / "results.parquet"))


# --- The shipping command writes rows -----------------------------------------


def test_the_shipping_command_writes_rows_to_the_results_parquet(scored):
    """#158's first acceptance criterion, and ADR-0009's template applied to `run`."""
    box, _ = scored
    table = table_of(box)

    tiles = len(TEST_COGS) * TILES_PER_COG
    # §3.2: every test tile once per bucket in both synthetic families.
    assert table.num_rows == tiles * 2 * 3
    assert set(table.column("arm").to_pylist()) == {"A1"}
    assert set(table.column("family").to_pylist()) == {"blob", "rectangle"}
    assert set(table.column("bucket").to_pylist()) == {"small", "medium", "large"}


def test_every_registered_metric_reaches_the_parquet_headline_and_appendix_alike(scored):
    """"Computed and printed are different questions" — #136, and #158's fifth criterion.

    The appendix columns are printed nowhere and are in the file, which is what makes `report` a
    pure function of it. The roster is read off the bar rather than listed here, so this test
    cannot drift from R1 the way a second copy would.
    """
    box, _ = scored
    table = table_of(box)
    roster = metrics.default_bar.__wrapped__()  # uncached: this box's bar, not the repo's

    registered = set(
        results.read_run_records(results.run_record_path(box / "results.parquet"))[0][
            "produced"
        ]["metrics"]
    )
    assert registered == set(roster.registered_metrics)
    assert {"mae", "p95", "profile_curvature", "slope_rmse_eroded"} <= registered, "appendix"
    assert {"rmse", "nmad", "bias_abs", "slope_rmse", "aspect_error", "seam_gap"} <= registered

    for name in registered:
        values = np.array(table.column(name).to_pylist(), dtype=float)
        assert np.isfinite(values).all(), name


def test_the_row_key_is_the_nine_the_ticket_states(scored):
    box, _ = scored
    assert results.KEY_COLUMNS == (
        "arm", "init", "size", "seed", "family", "bucket", "band", "tile_id", "instance",
    )
    assert set(results.KEY_COLUMNS) <= set(table_of(box).schema.names)


def test_the_mask_geometry_travels_with_the_scores(scored):
    """The void's area, its equivalent diameter and the committed mask's identity.

    C2 is a claim about void *size*, so a file that recorded only the bucket could not answer it
    at any finer grain than three — and the RLE hash is what proves a table was scored against
    the mask list that was frozen.
    """
    box, _ = scored
    table = table_of(box)
    area = np.array(table.column("void_area_px").to_pylist())
    eqdiam = np.array(table.column("void_eqdiam_px").to_pylist())

    assert (area > 0).all()
    assert eqdiam == pytest.approx(2.0 * np.sqrt(area / np.pi))
    assert all(len(h) == 32 for h in table.column("mask_rle_md5").to_pylist())


def test_the_scored_masks_are_the_committed_ones(scored):
    """Byte-identical inputs across arms is the whole point of committing the mask list."""
    box, _ = scored
    committed = pq.read_table(box / "artifacts" / EVAL_MASKS_NAME)
    expected = {
        (t, f, b): results.mask_rle_md5(r)
        for t, f, b, r in zip(
            committed.column("tile_id").to_pylist(),
            committed.column("family").to_pylist(),
            committed.column("bucket").to_pylist(),
            committed.column("mask_rle").to_pylist(),
        )
    }
    table = table_of(box)
    got = {
        (t, f, b): h
        for t, f, b, h in zip(
            table.column("tile_id").to_pylist(),
            table.column("family").to_pylist(),
            table.column("bucket").to_pylist(),
            table.column("mask_rle_md5").to_pylist(),
        )
    }
    assert got == expected


# --- The run record -----------------------------------------------------------


def test_the_run_record_carries_the_recipe_and_the_md5s_it_was_launched_with(scored):
    """#158's second criterion, and `test_..._carries_the_recipe_the_run_was_launched_with`'s heir.

    The extractor spec's md5 is in here and deliberately not in Phase 1's frozen four-field data
    fingerprint (ADR-0012 §7): a results table says which contract produced it without moving a
    hash four committed artifacts quote.
    """
    box, _ = scored
    (record,) = records_of(box)
    resolved = record["resolved"]

    assert resolved["config"]["arm"] == "A1"
    # The recipe round-trips and hashes to what the record claims — which is the property that
    # makes the record answerable, rather than a hash typed beside a config nobody re-read.
    rebuilt = BenchmarkConfig.from_dict(resolved["config"])
    assert rebuilt.config_hash == resolved["config_hash"]
    assert rebuilt.artifact_hash == resolved["artifact_hash"]
    assert rebuilt.arm == "A1" and rebuilt.fill.variant == "fillnodata"
    assert resolved["git_sha"]

    artifacts = resolved["artifacts"]
    for key in ("eval_masks", "test_tiles", "prereg_bar"):
        assert len(artifacts[key]["md5"]) == 32, key
    # Named even where absent: the two the ticket calls out by name are in the record whether or
    # not this box has them, so "which spec, which bank" is answerable off the record alone.
    assert "extractor_spec" in artifacts and "mask_bank" in artifacts
    # And a null md5 says *which* kind of null it is. Without `required` the two readings — this
    # arm does not read it, and `freeze` never wrote it — are indistinguishable in the record.
    assert artifacts["mask_bank"] == {
        "path": artifacts["mask_bank"]["path"],
        "md5": None,
        "required": False,
    }
    assert artifacts["eval_masks"]["required"] is True
    assert json.dumps(record), "a results table is answerable to plain JSON"


def test_the_run_record_carries_the_parameter_counts_total_and_trainable(scored):
    """#158's eighth criterion. A1 has no parameters at all, so both are a true 0."""
    box, _ = scored
    (record,) = records_of(box)
    assert record["parameters"] == {"total": 0, "trainable": 0}


def test_the_parameter_counts_are_a_pair_and_never_a_single_number():
    """§6 item 4's reason: A7's 17M reads as handicapped beside A4's 86.4M until you see both."""
    assert arms.parameter_counts(arms.get("A1")) == {"total": 0, "trainable": 0}
    # A2 is zero-shot over a counted encoder: total known, trainable exactly zero.
    assert arms.parameter_counts(arms.get("A2-R5")) == {"total": 85_254_144, "trainable": 0}
    # The head does not exist yet (#159), and a count this registry guessed would print as a
    # measurement — so it is `None` rather than 1,180,000.
    assert arms.parameter_counts(arms.get("A4-R5"))["trainable"] is None


def test_a_rerun_of_the_same_cell_replaces_its_rows_and_appends_its_record(box, flags):
    """Two passes over one cell would double-count every tile they agree on, in every aggregate.

    The record is append-only against exactly that: the *fact* that the cell was scored twice
    survives the row replacement, which is what makes the replacement safe rather than lossy.
    """
    assert launch.main(["run", "--arm", "A1", *flags]) == 0
    first = table_of(box).num_rows
    assert launch.main(["run", "--arm", "A1", *flags]) == 0

    assert table_of(box).num_rows == first
    records = records_of(box)
    assert len(records) == 2
    assert records[0]["run_id"] == records[1]["run_id"], "nothing the row is answerable to moved"


# --- The two gates ------------------------------------------------------------


def test_both_norm_inversion_gates_fire_and_say_so(scored):
    """#158's third and fourth criteria — §8's two gates, at the CLI altitude."""
    box, out = scored
    (record,) = records_of(box)

    assert "fill(identity-mask) == input" in out
    assert "golden tile" in out
    assert set(record["gates"]) == {"identity_mask", "golden_tile"}
    assert record["gates"]["identity_mask"]["pixels"] == 224 * 224


def test_the_golden_tile_is_asserted_through_the_shipping_path(scored):
    """#158's fourth criterion: the same numbers at the CLI altitude and at the function seam.

    The fixture's fill is hand-checked and is *not* an A1 fill — it is the truth plus a 4 m offset
    plus a ±1.5 m ripple, built to light up every registered column for a different reason. What
    the gate proves is that this run's scoring path and this run's bar reproduce the committed
    numbers before an arm scores 30,000 instances against them.
    """
    box, _ = scored
    (record,) = records_of(box)
    golden = metrics.read_golden(box / "artifacts")

    assert record["gates"]["golden_tile"]["metrics"] == sorted(golden.expected)
    # The function-seam numbers from ticket 10, re-asserted here off the same committed file.
    at_the_seam = metrics.score(
        golden.truth, golden.filled, golden.void, golden.px, bar=metrics.bar_for(
            BenchmarkConfig().replace(
                paths=BenchmarkConfig().paths.replace(
                    artifacts_dir=str(box / "artifacts"), plan_path=str(PLAN)
                )
            )
        )
    )
    for name, expected in golden.expected.items():
        assert at_the_seam[name] == pytest.approx(expected, rel=1e-12, abs=1e-12), name


def test_a_golden_tile_that_no_longer_scores_its_committed_numbers_stops_the_run(box, flags):
    """The gate is a refusal, not a warning: it fires before a single instance is scored."""
    sidecar = box / "artifacts" / metrics.GOLDEN_SIDECAR
    doc = json.loads(sidecar.read_text())
    doc["expected"]["rmse"] = 99.0
    sidecar.write_text(json.dumps(doc))

    with pytest.raises(ValueError, match="does not score what it is committed to score"):
        launch.main(["run", "--arm", "A1", *flags])
    assert not (box / "results.parquet").exists(), "it scored something before refusing"


def test_a_missing_golden_tile_stops_the_run_rather_than_skipping_the_gate(box, flags):
    (box / "artifacts" / metrics.GOLDEN_SIDECAR).unlink()
    with pytest.raises(FileNotFoundError, match="golden tile"):
        launch.main(["run", "--arm", "A1", *flags])


def test_the_identity_gate_catches_a_variant_that_moves_untouched_ground(box, flags, monkeypatch):
    """The gate's actual subject, and the reason it is aimed below `fill` rather than at it.

    `fill` restores every pixel outside the void from its input, so with an empty void it returns
    the input *by construction* — a gate pointed there would pass on a variant returning noise.
    Pointed at the variant, it fails: this one rescales by a hair, which is what a units or dtype
    bug looks like, and the alternative to failing is a full pass of plausible numbers.
    """
    monkeypatch.setitem(
        evaluate.fill.VARIANTS,
        "fillnodata",
        lambda tile, void, recipe: np.asarray(tile, dtype=np.float32) * 1.0001,
    )
    with pytest.raises(ValueError, match=r"fill\(identity-mask\) != input.*fillnodata variant"):
        launch.main(["run", "--arm", "A1", *flags])
    assert not (box / "results.parquet").exists(), "it scored something before refusing"


def test_an_inverted_mask_is_caught_where_a_committed_mask_is_decoded(box, flags):
    """The polarity guard the identity gate cannot be — there is no mask to invert when it runs.

    §3.2 caps a void at 40% of the tile and `freeze` checks every draw against that cap without
    ever trimming, so a decoded mask over it did not come from the generator. An inverted decode
    turns a 3% void into a 97% one, which fills almost the whole tile and scores a table nothing
    else would refuse.
    """
    committed = pq.read_table(box / "artifacts" / EVAL_MASKS_NAME)
    inverted = [
        masks.encode_rle(~masks.decode_rle(rle))
        for rle in committed.column("mask_rle").to_pylist()
    ]
    pq.write_table(
        committed.set_column(
            committed.schema.get_field_index("mask_rle"),
            "mask_rle",
            pa.array(inverted, type=pa.string()),
        ),
        box / "artifacts" / EVAL_MASKS_NAME,
    )
    with pytest.raises(ValueError, match="over §3.2's 40% cap"):
        launch.main(["run", "--arm", "A1", *flags])


# --- The ruggedness bands -----------------------------------------------------


def test_the_bands_are_quartiles_of_slope_mean_over_the_eligible_test_pool(scored):
    """#158's sixth criterion, and the distinction it turns on.

    The corpus's `stratum_id` is a k=20 k-means label over the frozen metric vector: its numbers
    are *names* with no ordering, so "band 1 is flatter than band 2" cannot be said about them.
    §5's bands are quartile cuts of one ordered quantity, which is what makes R12's "a majority
    of the four ruggedness bands" a statement with a direction.
    """
    box, _ = scored
    bands = evaluate.bands_for(
        BenchmarkConfig().replace(
            paths=BenchmarkConfig().paths.replace(corpus_dir=str(box / "corpus"))
        )
    )
    manifest = pq.read_table(box / "corpus" / "manifest.parquet", columns=["slope_mean"])
    pool = np.array(manifest.column("slope_mean").to_pylist(), dtype=float)

    assert bands.count == 4
    assert bands.edges == pytest.approx(tuple(np.quantile(pool, [0.0, 0.25, 0.5, 0.75, 1.0])))
    assert list(bands.edges) == sorted(bands.edges), "quartiles of one ordered quantity"

    labels = set(table_of(box).column("band").to_pylist())
    assert labels <= {"R1", "R2", "R3", "R4"}
    assert len(labels) > 1, "the fixture spreads slope_mean, so the bands must separate it"


def test_a_bands_edge_is_not_the_k_means_stratum_label(box):
    """The two must be able to disagree — the fixture holds `stratum_id` constant to prove it."""
    manifest = pq.read_table(
        box / "corpus" / "manifest.parquet", columns=["stratum_id", "slope_mean"]
    )
    assert len(set(manifest.column("stratum_id").to_pylist())) < len(
        set(manifest.column("slope_mean").to_pylist())
    )


# --- The registry -------------------------------------------------------------


def test_the_registry_holds_a1_to_a7_and_a9_and_neither_ruled_out_arm():
    """#158's seventh criterion. A8 and A10 are absent rather than disabled (#128).

    A disabled entry would put a ruled-out arm back on a surface someone reads, so there is no
    entry and no special-cased message — they fail the way any other unknown name does.
    """
    assert set(arms.ARMS) == {
        "A1", "A2-R5", "A2-R4", "A3-R5", "A3-R4", "A4-R5", "A4-R4",
        "A5", "A6", "A7", "A9", "A4-S",
    }
    assert not any(name.startswith(("A8", "A10")) for name in arms.ARMS)


@pytest.mark.parametrize("name", ["A8", "A10"])
def test_a_ruled_out_arm_is_refused_generically_rather_than_by_a_special_case(name, flags):
    with pytest.raises(ValueError, match=f"unknown arm '{name}'") as exc:
        launch.main(["run", "--arm", name, *flags])
    assert "ruled out" not in str(exc.value), "a special-cased message is a surface too"


#: The conventions that have an implementation rather than a ticket. ``classical_fill`` is
#: A1's (#158) and ``token_drop`` is A2's (#167); the other two name the tickets that build them.
BUILT_CONVENTIONS = ("classical_fill", "token_drop")


def test_every_registered_input_convention_has_an_entry_and_the_unbuilt_ones_name_a_ticket():
    """A1 runs first and its numbers gate the rest, so the others say which ticket builds them.

    Asserted on the table rather than through `main`, and the reason is itself the point: every
    non-A1 arm is refused *earlier* than its fill — for a mask bank it trains against, or a
    checkpoint this box has not mounted — so no CLI path reaches the unbuilt conventions today.
    What can go wrong here is the table falling behind the registry, which is a completeness
    claim about a dict.

    The entries are **factories** — ``(cfg, arm) -> filler`` — since #167: a convention that runs
    an encoder has to load one once per pass rather than once per instance, and ``token_drop`` is
    shared by two arms whose weights differ. So the refusal is raised by the filler a factory
    returns, which is exactly where a built convention would do its work.
    """
    assert set(evaluate.FILLERS) == set(arms.INPUT_CONVENTIONS)

    for convention, factory in evaluate.FILLERS.items():
        if convention in BUILT_CONVENTIONS:
            continue
        with pytest.raises(NotImplementedError, match=f"{convention}.*not built") as exc:
            factory(None, None)(None, None)
        assert "#" in str(exc.value), f"{convention} does not name the ticket that builds it"


# --- What `run` refuses -------------------------------------------------------


def test_a1_does_not_wait_on_the_train_mask_bank_it_never_reads(scored):
    """§3.4's bank is #151's, and A1 trains nothing — so the first number does not wait on it.

    Asserted through a *passing* run: the fixture never writes a bank, and the cell scores. The
    resolution line still names the bank, because what is on this box is a fact about the box.
    """
    box, out = scored
    assert "mask bank" in out and "not read by this arm" in out
    assert table_of(box).num_rows > 0


def test_a_training_arm_still_refuses_a_box_with_no_bank(box, flags):
    """The other half of the same rule: an arm that *does* read it is refused for want of it."""
    with pytest.raises(FileNotFoundError, match="mask bank"):
        launch.main(["run", "--arm", "A4-R5", *flags])


def test_a_run_against_an_unfrozen_box_still_says_freeze_first(tmp_path, box, flags):
    (box / "artifacts" / EVAL_MASKS_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="benchmark.launch freeze"):
        launch.main(["run", "--arm", "A1", *flags])


def test_a_mask_list_naming_a_tile_the_test_list_does_not_is_refused(box, flags):
    """A re-drawn test list beside a stale mask list — a benchmark scoring untested ground."""
    write_tile_strata([("N45_E007_99999", 0)], box / "artifacts" / TEST_TILES_NAME)
    with pytest.raises(ValueError, match="does not"):
        launch.main(["run", "--arm", "A1", *flags])


def test_a_dry_run_still_writes_nothing_now_that_run_writes_something(box, flags):
    before = sorted(p.relative_to(box).as_posix() for p in box.rglob("*"))
    assert launch.main(["run", "--arm", "A1", "--dry-run", *flags]) == 0
    assert sorted(p.relative_to(box).as_posix() for p in box.rglob("*")) == before
