"""`python -m benchmark.launch` — the one command that freezes, runs or reports (#147).

The tracer bullet through the whole command surface, before anything it drives exists. Prior
art is `tests/test_pretrain_launch.py` and the reason is ADR-0009's: *a seam that nobody is on
both sides of is a feature that does not exist*, and the way this repo keeps that from
happening again is a test that drives `main(argv)` rather than one that drives each piece.

What is asserted here is the surface, because the surface is what #147 ships: the three verbs
dispatch, a fourth is refused, a typo'd override stops before the corpus is touched, `--dry-run`
assembles everything and writes nothing, and the run says what it resolved — which encoder,
which checkpoint md5, which mask bank, which spec, how many instances — before it takes a step.

The far side of the seam is deliberately loud rather than absent: past `--dry-run`, every verb
raises and names the stage that has not been built. A verb that quietly returned 0 having done
nothing is the shape of failure this file exists to prevent.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from benchmark import launch
from benchmark.config import BenchmarkConfig

#: What `freeze` will write, and therefore what `run` resolves. Written as bytes here, since
#: #147 resolves artifacts by presence and content hash and reads none of their schemas.
FROZEN = {
    "head_train_tiles.csv": b"tile_id,stratum_id\n",
    "val_tiles.csv": b"tile_id,stratum_id\n",
    "test_tiles.csv": b"tile_id,stratum_id\n",
    "eval_masks.parquet": b"eval masks",
    "extractor_spec.json": b"{}",
    "task1-prereg-bar.json": b"{}",
}


@pytest.fixture
def box(tmp_path):
    """A box with the paths this command reads, and nothing frozen onto it yet."""
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "release").mkdir()
    (tmp_path / "release" / "encoder.pt").write_bytes(b"not really an encoder")
    return tmp_path


@pytest.fixture
def flags(box):
    return [
        "--set", f'paths.artifacts_dir="{box / "artifacts"}"',
        "--set", f'paths.mask_bank_path="{box / "bank.parquet"}"',
        "--set", f'paths.demae_r5_checkpoint="{box / "release" / "encoder.pt"}"',
        "--set", f'paths.corpus_dir="{box / "corpus"}"',
    ]


@pytest.fixture
def frozen(box):
    """The same box after a `freeze` that this ticket does not implement."""
    for name, body in FROZEN.items():
        (box / "artifacts" / name).write_bytes(body)
    (box / "bank.parquet").write_bytes(b"one million rle rows, allegedly")
    return box


# --- The three verbs ----------------------------------------------------------


@pytest.mark.parametrize("verb", ["freeze", "run", "report"])
def test_every_verb_parses_and_dispatches(verb, flags, capsys):
    assert launch.main([verb, "--dry-run", *flags]) == 0
    assert "DEMAE Task 1" in capsys.readouterr().out


def test_a_fourth_verb_is_refused(flags, capsys):
    with pytest.raises(SystemExit) as exc:
        launch.main(["score", "--dry-run", *flags])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_freeze_names_the_generator_hash_rather_than_a_cell(flags, capsys):
    """§3.4's contract, on screen: what `freeze` writes is not a function of the arm.

    A `freeze` header that named an arm — or printed a config hash that moved with ``--arm`` —
    would be claiming the committed inputs had a cell, which is the second determinism contract
    §3.4 exists to avoid having.
    """
    launch.main(["freeze", "--dry-run", *flags])
    out = capsys.readouterr().out
    assert BenchmarkConfig().artifact_hash in out
    assert BenchmarkConfig().config_hash not in out
    assert "arm A4-R5" not in out and "seed 0" not in out
    assert "encoder" not in out and "checkpoint" not in out  # freeze is CPU-only and torch-free


def test_only_run_and_report_name_the_parquet_the_table_comes_from(flags, capsys):
    # `report` is a pure function of that file (#136's "computed and printed are different
    # questions"), so re-rendering the table never re-runs an arm. `freeze` never touches it.
    parquet = BenchmarkConfig().paths.results_path
    for verb, expected in (("run", True), ("report", True), ("freeze", False)):
        launch.main([verb, "--dry-run", *flags])
        assert (parquet in capsys.readouterr().out) is expected


def test_the_command_is_the_only_one_that_freezes_runs_or_reports():
    # ADR-0009's rule carried into Phase 2. The verbs are the surface; a second entry point
    # with its own `main` would be the `pretrain.monitor` shape all over again.
    assert tuple(launch.VERBS) == ("freeze", "run", "report")


# --- A typo stops before anything is spent ------------------------------------


def test_a_typo_in_an_override_stops_before_the_corpus_is_touched(flags, capsys):
    with pytest.raises(ValueError, match="unknown head knob\\(s\\) \\['epoch'\\]"):
        launch.main(["run", "--dry-run", *flags, "--set", "head.epoch=10"])
    assert capsys.readouterr().out == "", "it resolved something before refusing the config"


def test_a_typo_in_the_group_names_the_groups_there_are(flags):
    with pytest.raises(ValueError, match="unknown config group\\(s\\) \\['mask'\\]"):
        launch.main(["run", "--dry-run", *flags, "--set", "mask.seed=1"])


def test_an_unknown_arm_is_refused_by_name(flags):
    with pytest.raises(ValueError, match="unknown arm 'A8'"):
        launch.main(["run", "--dry-run", "--arm", "A8", *flags])


# --- What --dry-run assembles, and what it writes -----------------------------


def test_a_dry_run_assembles_the_config_the_artifacts_the_extractor_and_the_arm(flags, capsys):
    assert launch.main(["run", "--dry-run", "--arm", "A5", "--size", "5k", "--seed", "2", *flags]) == 0
    out = capsys.readouterr().out
    assert "arm A5" in out and "5k" in out and "seed 2" in out  # the cell
    assert BenchmarkConfig(arm="A5", size="5k", seed=2).config_hash in out  # the config
    assert "terramind" in out  # the arm's encoder, off the registry
    assert "terramind_v1 normalization" in out  # §2.1's diet, which the encoder would not refuse
    assert "12 blocks + `output`" in out  # ADR-0012's contract, as this run will call it
    assert "extractor spec" in out  # and the committed file that re-checks itself at load
    assert "eval masks" in out and "mask bank" in out  # the artifacts


def test_a_dry_run_writes_nothing(box, flags):
    before = sorted(p.relative_to(box).as_posix() for p in box.rglob("*"))
    for verb in ("freeze", "run", "report"):
        assert launch.main([verb, "--dry-run", *flags]) == 0
    assert sorted(p.relative_to(box).as_posix() for p in box.rglob("*")) == before


def test_a_dry_run_says_so_rather_than_looking_like_a_run(flags, capsys):
    launch.main(["run", "--dry-run", *flags])
    assert "--dry-run: nothing was written" in capsys.readouterr().out


# --- The resolution line ------------------------------------------------------


def test_the_run_prints_what_it_resolved_before_it_takes_a_step(frozen, flags, capsys):
    """§15-shaped pre-flight: a run scoring the wrong arm should not look like a normal run.

    Five things, and every one of them is a fact about the box rather than about the recipe —
    which is exactly why the config hash does not cover them and why they are printed instead.
    """
    launch.main(["run", "--dry-run", *flags])
    out = capsys.readouterr().out
    checkpoint_md5 = hashlib.md5((frozen / "release" / "encoder.pt").read_bytes()).hexdigest()
    bank_md5 = hashlib.md5((frozen / "bank.parquet").read_bytes()).hexdigest()
    spec_md5 = hashlib.md5(FROZEN["extractor_spec.json"]).hexdigest()

    assert "demae-r5" in out  # which encoder
    assert checkpoint_md5 in out  # which checkpoint
    assert bank_md5 in out  # which mask bank
    assert spec_md5 in out  # which extractor spec
    assert "30,000 synthetic" in out  # how many instances


def test_the_instance_count_is_the_test_draw_times_the_families_times_the_buckets(frozen, flags, capsys):
    # 5,000 test tiles x 2 synthetic families x 3 buckets = 30,000, plus the real family's own
    # rows, which have no bucket axis and are counted off the artifact rather than derived.
    launch.main(["run", "--dry-run", *flags, "--set", "tiles.test=1000"])
    assert "6,000 synthetic" in capsys.readouterr().out


def test_a_missing_artifact_is_named_as_missing_rather_than_shown_a_stale_hash(flags, capsys):
    launch.main(["run", "--dry-run", *flags])
    out = capsys.readouterr().out
    assert "eval masks" in out and "not frozen" in out
    assert "mask bank" in out


def test_an_unmounted_checkpoint_is_named_rather_than_silently_absent(flags, capsys):
    # A6's ImageNet-MAE checkpoint has no path on this box, and a resolution line that simply
    # omitted it is one a build session reads as "resolved".
    launch.main(["run", "--dry-run", "--arm", "A6", *flags])
    out = capsys.readouterr().out
    assert "imagenet-mae" in out
    assert "paths.imagenet_mae_checkpoint" in out


def test_an_arm_with_no_encoder_says_so_rather_than_printing_an_empty_checkpoint(flags, capsys):
    # A1 is classical interpolation and A7 is a UNet from scratch: neither reads an encoder,
    # and neither should print a checkpoint line that reads as unresolved.
    launch.main(["run", "--dry-run", "--arm", "A1", *flags])
    out = capsys.readouterr().out
    assert "no encoder" in out
    assert "checkpoint " not in out


def test_the_resolution_records_itself_as_the_run_record_carries_it(frozen, flags):
    resolution = launch.resolve(launch.build_parser().parse_args(["run", "--dry-run", *flags]))
    record = resolution.to_dict()
    assert record["config_hash"] == resolution.config.config_hash
    assert record["arm"] == "A4-R5"
    assert record["artifacts"]["extractor_spec"]["md5"] == hashlib.md5(
        FROZEN["extractor_spec.json"]
    ).hexdigest()
    assert json.dumps(record)  # plain JSON — this is what a results table is answerable to


# --- What the flags reach -----------------------------------------------------


def test_the_cell_flags_are_config_knobs_rather_than_a_second_recipe(flags, capsys):
    launch.main(["run", "--dry-run", "--arm", "A7", "--size", "1k", "--seed", "1", *flags])
    assert BenchmarkConfig(arm="A7", size="1k", seed=1).config_hash in capsys.readouterr().out


def test_set_beats_the_cell_flag_it_duplicates(flags, capsys):
    # `--set` is the general escape hatch and is applied last, exactly as `--corpus` and
    # `--set data.corpus_dir=…` resolve in `pretrain.header.build_config`.
    launch.main(["run", "--dry-run", "--arm", "A7", *flags, "--set", "arm=A6"])
    assert "arm A6" in capsys.readouterr().out


def test_a_config_file_names_only_the_knobs_its_run_moves(box, flags, capsys):
    path = box / "arm.json"
    path.write_text(json.dumps({"arm": "A6", "head": {"epochs": 5}}))
    launch.main(["run", "--dry-run", "--config", str(path), *flags])
    out = capsys.readouterr().out
    assert BenchmarkConfig(arm="A6", head=BenchmarkConfig().head.replace(epochs=5)).config_hash in out


# --- The far side of the seam -------------------------------------------------


def test_a_run_past_the_dry_run_refuses_rather_than_pretending(frozen, flags, capsys):
    """Every verb's implementation is a later ticket, and it says so out loud.

    The alternative — returning 0 having done nothing — is the failure this repo has a name
    for. The resolution is still printed first, because what a refusal is *about* is the box
    it refused on.
    """
    with pytest.raises(NotImplementedError, match="`run` scores nothing yet"):
        launch.main(["run", *flags])
    assert "DEMAE Task 1" in capsys.readouterr().out


@pytest.mark.parametrize("verb", ["freeze", "report"])
def test_the_other_two_verbs_refuse_the_same_way(verb, frozen, flags):
    with pytest.raises(NotImplementedError, match="#147 shipped the command surface"):
        launch.main([verb, *flags])


def test_a_run_against_an_unfrozen_box_says_freeze_first(flags):
    # Before the stage tickets and after them, this is the refusal that comes first: there is
    # nothing to score against, and the message is the command that fixes it.
    with pytest.raises(FileNotFoundError, match="benchmark.launch freeze"):
        launch.main(["run", *flags])
