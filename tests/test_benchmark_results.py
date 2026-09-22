"""The results parquet and the run record — the refusals seam 1 cannot reach (#158).

`tests/test_benchmark_a1.py` drives all of this through `python -m benchmark.launch run`, which
is where the behaviour belongs. What is here is the set of claims a *passing* run cannot
exercise: a row short of the roster, a file written under a different pre-registration, and the
two properties of `run_id` that make re-running a cell idempotent rather than duplicative.

No corpus and no pixels: this module is about the file.
"""

from __future__ import annotations

import pytest

from benchmark import artifacts, results
from benchmark.config import BenchmarkConfig, Paths, Tiles

ROSTER = ("rmse", "nmad", "seam_gap")


def config(tmp_path, arm="A1", **paths) -> BenchmarkConfig:
    return BenchmarkConfig(
        arm=arm,
        tiles=Tiles(test=8),
        paths=Paths(
            artifacts_dir=str(tmp_path / "artifacts"),
            results_path=str(tmp_path / "results.parquet"),
            mask_bank_path=str(tmp_path / "bank.parquet"),
            **paths,
        ),
    )


def row(tile_id="N45_E007_00000", *, arm="A1", size="20k", seed=0, metrics=None):
    return results.ResultRow(
        arm=arm,
        init=None,
        size=size,
        seed=seed,
        family="blob",
        bucket="small",
        band="R1",
        tile_id=tile_id,
        instance=0,
        void_area_px=1600,
        void_eqdiam_px=45.135,
        mask_rle_md5="0" * 32,
        run_id="deadbeefdeadbeef",
        config_hash="0123456789abcdef",
        metrics=dict.fromkeys(ROSTER, 1.0) if metrics is None else metrics,
    )


# --- The roster is the schema -------------------------------------------------


def test_a_row_short_of_the_roster_is_refused_rather_than_landing_a_null_column(tmp_path):
    """A column of nulls reads as a metric this arm scored badly, which is the worse failure."""
    short = row(metrics={"rmse": 1.0, "nmad": 2.0})
    with pytest.raises(ValueError, match="missing \\['seam_gap'\\]"):
        results.append_rows([short], tmp_path / "results.parquet", metrics=ROSTER)


def test_a_row_carrying_a_metric_outside_the_roster_is_refused(tmp_path):
    """R1's roster is what the parquet's metric columns are — a name outside it has no column."""
    extra = row(metrics={**dict.fromkeys(ROSTER, 1.0), "hillshade_ssim": 0.9})
    with pytest.raises(ValueError, match="unregistered \\['hillshade_ssim'\\]"):
        results.append_rows([extra], tmp_path / "results.parquet", metrics=ROSTER)


def test_the_schema_is_a_function_of_the_roster_rather_than_a_constant():
    """A results file whose columns were typed in this module could disagree with the bar."""
    names = results.schema(ROSTER).names
    assert names == list(results.columns(ROSTER))
    assert names[-len(ROSTER):] == list(ROSTER)
    assert set(results.KEY_COLUMNS) | set(results.GEOMETRY_COLUMNS) <= set(names)


def test_a_file_written_under_a_different_pre_registration_is_not_merged(tmp_path):
    """Two rosters is two pre-registrations, which is not something to reconcile silently."""
    path = tmp_path / "results.parquet"
    results.append_rows([row()], path, metrics=ROSTER)
    with pytest.raises(ValueError, match="two different"):
        results.append_rows(
            [row(metrics={"rmse": 1.0, "mae": 1.0})], path, metrics=("rmse", "mae")
        )


# --- Replace by cell ----------------------------------------------------------


def test_rewriting_a_cell_replaces_only_that_cell(tmp_path):
    path = tmp_path / "results.parquet"
    results.append_rows([row("a"), row("b")], path, metrics=ROSTER)
    results.append_rows([row("c", seed=1)], path, metrics=ROSTER)
    _, total = results.append_rows([row("d")], path, metrics=ROSTER)

    table = results.read_rows(path)
    assert total == 2
    # Seed 1 survived; the two seed-0 rows were replaced by the one this call wrote.
    assert sorted(zip(table.column("tile_id").to_pylist(), table.column("seed").to_pylist())) == [
        ("c", 1),
        ("d", 0),
    ]


def test_replacing_every_row_a_file_holds_is_not_a_typing_error(tmp_path):
    """The empty-keep case: a file holding one cell, re-run. `take` on an untyped empty list
    fails with a kernel error about `(string, null)` that says nothing about cells."""
    path = tmp_path / "results.parquet"
    results.append_rows([row("a")], path, metrics=ROSTER)
    _, total = results.append_rows([row("a")], path, metrics=ROSTER)
    assert total == 1


def test_a_file_that_passed_through_pandas_is_appended_to_and_comes_back_canonical(tmp_path):
    """`scripts/pull_box_results.py` wrote the canonical parquet through pandas, which types every
    string column `large_string` and stamps pandas metadata on the schema. Same columns, same
    values, other Arrow types — and a strict `schema !=` refused every later laptop `run` with a
    message about pre-registrations (#173 lost a full A2 pass to it). Cast back, not refused."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "results.parquet"
    results.append_rows([row("a"), row("b", seed=1)], path, metrics=ROSTER)
    table = pq.read_table(path)
    wide = pa.schema(
        [
            (f.name, pa.large_string() if pa.types.is_string(f.type) else f.type)
            for f in table.schema
        ],
        metadata={b"pandas": b"{}"},
    )
    pq.write_table(table.cast(wide), path)
    assert pq.read_schema(path) != results.schema(ROSTER)

    _, total = results.append_rows([row("c")], path, metrics=ROSTER)

    assert total == 2  # seed 0's row replaced, seed 1's kept
    assert pq.read_schema(path).remove_metadata() == results.schema(ROSTER)
    assert sorted(results.read_rows(path).column("tile_id").to_pylist()) == ["b", "c"]


def test_a_file_scored_under_another_roster_is_still_refused(tmp_path):
    path = tmp_path / "results.parquet"
    results.append_rows([row("a")], path, metrics=ROSTER)

    with pytest.raises(ValueError, match="two different pre-registrations"):
        results.append_rows(
            [row("b", metrics={**{m: 1.0 for m in ROSTER}, "extra_metric": 1.0})],
            path,
            metrics=[*ROSTER, "extra_metric"],
        )


# --- The run record -----------------------------------------------------------


def test_the_run_id_is_stable_for_a_run_nothing_moved_under(tmp_path):
    """What makes re-running an unchanged cell idempotent rather than a second identity."""
    resolution = artifacts.resolve(config(tmp_path))
    assert results.run_id(resolution) == results.run_id(artifacts.resolve(config(tmp_path)))


def test_the_run_id_moves_when_an_input_this_row_is_answerable_to_moves(tmp_path):
    """The recipe, any committed input's md5, the checkpoint, the git SHA — the whole list."""
    directory = tmp_path / "artifacts"
    directory.mkdir()
    bar = directory / "task1-prereg-bar.json"
    bar.write_text("{}")
    before = results.run_id(artifacts.resolve(config(tmp_path)))

    bar.write_text('{"moved": true}')
    assert results.run_id(artifacts.resolve(config(tmp_path))) != before

    moved_recipe = config(tmp_path).replace(seed=7)
    assert results.run_id(artifacts.resolve(moved_recipe)) != before


def test_the_run_id_does_not_move_for_an_artifact_this_arm_never_read(tmp_path):
    """The day #151 first writes a train mask bank, no A1 row's identity may move.

    A1 trains nothing, so the bank is not one of its inputs — and an id that moved for a reason
    the rows do not share would announce a change to a computation whose every input was
    identical. The record still carries the bank by md5; what it does not do is *hash* it.
    """
    before = results.run_id(artifacts.resolve(config(tmp_path)))

    (tmp_path / "bank.parquet").write_bytes(b"one million rle rows, at last")
    resolution = artifacts.resolve(config(tmp_path))

    assert results.run_id(resolution) == before
    # ...and it is still recorded, so provenance did not lose the fact that a bank now exists.
    assert resolution.to_dict()["artifacts"]["mask_bank"]["md5"] is not None


def test_the_run_id_does_move_for_the_same_artifact_when_the_arm_reads_it(tmp_path):
    """The other half: A4-R5 trains a head against the bank, so its identity is answerable to it."""
    training = config(tmp_path).replace(arm="A4-R5")
    before = results.run_id(artifacts.resolve(training))

    (tmp_path / "bank.parquet").write_bytes(b"one million rle rows, at last")
    assert results.run_id(artifacts.resolve(training)) != before


def test_the_record_lives_beside_the_parquet_and_is_appended_never_replaced(tmp_path):
    """A cell scored twice is a fact about how the table was produced — see `benchmark.results`."""
    path = tmp_path / "results.parquet"
    assert results.run_record_path(path) == tmp_path / "results-runs.jsonl"

    resolution = artifacts.resolve(config(tmp_path))
    for _ in range(3):
        results.write(
            [row()],
            resolution,
            metrics=ROSTER,
            seconds=1.0,
            parameters={"total": 0, "trainable": 0},
        )
    assert len(results.read_run_records(results.run_record_path(path))) == 3
    assert results.read_rows(path).num_rows == 1


def test_the_record_embeds_the_provenance_block_whole_rather_than_restating_it(tmp_path):
    """What a run *read* and what it *produced* are two questions — the `pretrain` split."""
    resolution = artifacts.resolve(config(tmp_path))
    record = results.run_record(
        resolution,
        rows=48,
        seconds=2.5,
        parameters={"total": 0, "trainable": 0},
        metrics=ROSTER,
    )
    assert record["resolved"] == resolution.to_dict()
    assert record["produced"] == {"rows": 48, "seconds": 2.5, "metrics": list(ROSTER)}
    assert record["cell"] == {"arm": "A1", "size": "20k", "seed": 0}


def test_an_unknown_parameter_count_prints_as_unknown_rather_than_as_a_number(tmp_path):
    """`None` is *not counted here yet* — a guessed count would print as a measurement."""
    resolution = artifacts.resolve(config(tmp_path))
    record = results.run_record(
        resolution,
        rows=1,
        seconds=0.1,
        parameters={"total": None, "trainable": None},
        metrics=ROSTER,
    )
    lines = results.format_written(
        tmp_path / "results.parquet", rows=1, total=1, record=record
    )
    assert any("not built yet total / not built yet trainable" in line for line in lines)


# --- ADR-0012 §6's measured agreement, in the record (#171) -------------------

#: p157's TerraMind figures, as `Extractor._precision` returns them. A literal rather than a real
#: measurement because this module is about the *record*: what has to be true here is that the
#: block lands whole and re-readable, not that any encoder produced any particular cosine.
#: `tests/test_benchmark_extract.py` owns the measuring.
AGREEMENT = {
    "encoder": "terramind",
    "reference_dtype": "fp32",
    "compute_dtype": "bf16",
    "device": "mps",
    "tiles": 60,
    "tokens": 11760,
    "mean_cosine": 0.99888,
    "min_cosine": 0.94710,
    "p01_cosine": 0.98530,
    "relative_rms": 0.00420,
    "threshold": 0.99,
    "immaterial": True,
}


def test_the_measured_agreement_is_recorded_beside_the_contract_that_produced_it(tmp_path):
    """AC 1 and 2 — the other half of "a results table says which contract produced it".

    The spec md5 says *which* contract; this says how far the dtype it actually ran at moved the
    features. Both have to be readable off one record, and the figure has to arrive with what it
    was measured over rather than as a bare cosine.
    """
    resolution = artifacts.resolve(config(tmp_path, arm="A5"))
    record = results.run_record(
        resolution,
        rows=48,
        seconds=2.5,
        parameters={"total": 0, "trainable": 0},
        metrics=ROSTER,
        precision=results.precision_block(resolution, AGREEMENT),
    )
    assert record["precision"]["agreement"] == AGREEMENT
    assert record["precision"]["absent_because"] is None
    # Re-readable rather than a bare cosine: the dtype, the box, and the sample it was taken over.
    agreement = record["precision"]["agreement"]
    assert agreement["compute_dtype"] == "bf16" and agreement["reference_dtype"] == "fp32"
    assert agreement["device"] == "mps"
    assert (agreement["tiles"], agreement["tokens"]) == (60, 11760)
    # And the spec md5 is still where it was, in the provenance block.
    assert "extractor_spec" in record["resolved"]["artifacts"]


def test_a_run_at_the_reference_dtype_records_no_figure_rather_than_a_one(tmp_path):
    """AC 3. fp32 against itself is not a measurement, and a recorded 1.0 would read as one."""
    base = config(tmp_path, arm="A5")
    cfg = base.replace(extract=base.extract.replace(compute_dtype="fp32"))
    resolution = artifacts.resolve(cfg)
    block = results.precision_block(resolution, None)
    assert block["agreement"] is None
    assert "reference dtype" in block["absent_because"]


@pytest.mark.parametrize("arm", ["A1", "A7"])
def test_an_arm_with_no_encoder_records_an_absence_rather_than_nothing(tmp_path, arm):
    """AC 4. The `Resolved.required` problem again: a null figure has more than one reading, so
    the block says which one it is. *Unmeasured* is the key being absent altogether."""
    resolution = artifacts.resolve(config(tmp_path, arm=arm))
    block = results.precision_block(resolution, None)
    assert block["agreement"] is None
    assert "no encoder" in block["absent_because"]

    recorded = results.run_record(
        resolution,
        rows=1,
        seconds=0.1,
        parameters={"total": 0, "trainable": 0},
        metrics=ROSTER,
        precision=block,
    )
    unmeasured = results.run_record(
        resolution,
        rows=1,
        seconds=0.1,
        parameters={"total": 0, "trainable": 0},
        metrics=ROSTER,
    )
    assert recorded["precision"] == block
    assert "precision" not in unmeasured


def test_an_arm_that_consults_no_extractor_contract_says_so(tmp_path):
    """A2 runs the whole pretrained autoencoder and reads pixels off the decoder (#167), so the
    committed spec governs nothing it does. That is a third reading of a null figure, and it is
    not "the run declared fp32"."""
    resolution = artifacts.resolve(config(tmp_path, arm="A2-R5"))
    block = results.precision_block(resolution, None)
    assert block["agreement"] is None
    assert "extractor contract" in block["absent_because"]


def test_the_agreement_is_not_part_of_the_provenance_the_run_id_hashes(tmp_path):
    """AC 5, and the reason it is a separate block rather than a field of `resolved`.

    ADR-0012 §7 keeps it out of Phase 1's frozen four-field data fingerprint, which four
    committed artifacts quote. `run_id` is the sharper version of the same claim: the figure is
    measured on a batch, so a figure inside the id would give an unchanged cell a new identity
    every time the sample moved.
    """
    resolution = artifacts.resolve(config(tmp_path, arm="A5"))
    before = results.run_id(resolution)
    record = results.run_record(
        resolution,
        rows=1,
        seconds=0.1,
        parameters={"total": 0, "trainable": 0},
        metrics=ROSTER,
        precision=results.precision_block(resolution, AGREEMENT),
    )
    assert record["run_id"] == before
    assert record["resolved"] == resolution.to_dict()
    assert "precision" not in record["resolved"]
    assert not any("cosine" in key for key in record["resolved"])
