"""`python -m benchmark.launch --task landform` — Task 2 on the one shipping command (#198).

ADR-0009's rule, carried into Task 2: one command, three verbs, and a test that drives
``main(argv)`` rather than the pieces. ``--task`` picks which benchmark the verbs act on, and it
defaults to Task 1, so every existing command, test and config hash is unchanged.

What #198 ships is the selector and Task 2's sibling config. Nothing real is frozen, run or
reported yet, so the far side of ``--dry-run`` names the ticket that builds each stage rather
than returning 0 having done nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmark import landform, launch
from benchmark.config import BenchmarkConfig
from benchmark.landform_config import RUNG_TILES, RUNGS, LandformConfig, steps


@pytest.fixture
def box(tmp_path):
    """A box with the paths Task 2 reads and nothing frozen onto it."""
    (tmp_path / "task1").mkdir()
    (tmp_path / "task2").mkdir()
    (tmp_path / "release").mkdir()
    (tmp_path / "release" / "encoder.pt").write_bytes(b"not really an encoder")
    return tmp_path


@pytest.fixture
def flags(box):
    """Task 2's paths, pointed at the box. ``--task landform`` comes first, as a user types it."""
    return [
        "--task", "landform",
        "--set", f'paths.artifacts_dir="{box / "task2"}"',
        "--set", f'paths.task1_artifacts_dir="{box / "task1"}"',
        "--set", f'paths.results_path="{box / "results.parquet"}"',
        "--set", f'paths.demae_r5_checkpoint="{box / "release" / "encoder.pt"}"',
        "--set", f'paths.corpus_dir="{box / "corpus"}"',
    ]


# --- The selector -------------------------------------------------------------


@pytest.mark.parametrize("verb", ["freeze", "run", "report"])
def test_every_verb_resolves_task_two_under_a_dry_run(verb, flags, capsys):
    assert launch.main([verb, "--dry-run", *flags]) == 0
    out = capsys.readouterr().out
    assert "DEMAE Task 2" in out and "DEMAE Task 1" not in out
    assert "--dry-run: nothing was written" in out


def test_the_default_task_is_still_task_one(capsys):
    """Story 2: every command written before Task 2 still means what it meant."""
    assert launch.main(["run", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DEMAE Task 1" in out and BenchmarkConfig().config_hash in out


def test_an_unknown_task_is_refused_by_argparse(capsys):
    with pytest.raises(SystemExit) as exc:
        launch.main(["run", "--dry-run", "--task", "bare-earth"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


# --- Task 1 does not move ------------------------------------------------------

#: Every registered Task 1 arm at `BenchmarkConfig(arm=name)`'s defaults, hashed on the
#: integration branch at 102d64c before #198 touched anything. The first fourteen are #182's
#: pinned literals; the three skip arms are #182's own.
TASK1_HASHES = {
    "A1": "a41e3e2bfef7916e",
    "A2-R5": "eeae1d52a761ea41",
    "A2-R4": "f9ecde0076aef5ba",
    "A3-R5": "db1c37c391cfc388",
    "A3-R4": "24956f0c06e34880",
    "A4-R5": "18489237db77449b",
    "A4-R4": "55cfba411c8f0738",
    "A5": "4fc3cf2879931fb8",
    "A6": "5a5488eb944005b9",
    "A7": "7263e69f88ba2c52",
    "A9": "7d7f8ae72a60623a",
    "A9-NOTHRESH": "d76138c3d2457c11",
    "A5-BICUBIC": "770f0e120bc13801",
    "A4-S": "716805ace1c42475",
    "A4-R5-SKIP": "8463cfc28fc01c2a",
    "A5-SKIP": "85951227ab11d76e",
    "A6-SKIP": "11834da282564f03",
}

#: Cells off the defaults, hashed at the same commit, so the size and seed scalars are pinned
#: as well as the arm.
TASK1_CELL_HASHES = {
    ("A4-R5", "1k", 1): "8ca1b3a9d9b07d1e",
    ("A7", "5k", 0): "d341bbcd0aa948ac",
}


def test_every_committed_task_one_config_hash_is_unmoved():
    """Story 2, as a changed sixteen-character string rather than a re-derived agreement.

    #198 moved Task 1's serialization and hashing into functions both configs call, so this is
    the test that says the move was a move.
    """
    from benchmark import arms

    assert set(TASK1_HASHES) == set(arms.ARMS), "a Task 1 arm was added or dropped"
    for name, want in TASK1_HASHES.items():
        assert BenchmarkConfig(arm=name).config_hash == want, name
    for (arm, size, seed), want in TASK1_CELL_HASHES.items():
        assert BenchmarkConfig(arm=arm, size=size, seed=seed).config_hash == want, (arm, size)


def test_task_one_generator_hash_is_the_one_its_bank_sidecar_commits():
    committed = Path(__file__).resolve().parents[1] / "artifacts" / "task1" / "train_mask_bank.json"
    assert BenchmarkConfig().artifact_hash == json.loads(committed.read_text())["artifact_hash"]


# --- --dry-run resolves the whole cell and spends nothing ---------------------


def test_a_dry_run_prints_the_cell_it_resolved_before_any_step(box, flags, capsys):
    """Stories 4-5: arm, encoder pin, checkpoint md5, rung and tile count, head, label set and
    the step count — and the parts not built yet, named rather than dropped."""
    assert launch.main(["run", "--dry-run", "--rung", "1%", "--seed", "2", *flags]) == 0
    out = capsys.readouterr().out
    assert "arm B1, rung 1% (200 tiles), seed 2, head linear, label set geomorphon" in out
    assert LandformConfig(rung="1%", seed=2).config_hash in out
    assert "demae-r5, pinned md5 3ec5ebc7" in out
    checkpoint = hashlib.md5((box / "release" / "encoder.pt").read_bytes()).hexdigest()
    assert checkpoint in out
    lines = out.splitlines()
    assert any(line.startswith("recipe line") and "not built yet" in line for line in lines)
    # #201 built the geomorphon labels: their config resolves by path, not frozen on a bare box.
    assert any(line.startswith("geomorphon") and "not frozen yet" in line for line in lines)
    # #200 built the ladder: seed 2's, by path, and the rung as a prefix of it.
    assert any(line.startswith("ladder") and "ladder_2.csv" in line for line in lines)
    assert "the first 200 tiles of ladder 2" in out
    # #199 built the bar: it resolves by path, and says it is not frozen on a bare box.
    assert any(line.startswith("prereg bar") and "not frozen yet" in line for line in lines)


@pytest.mark.parametrize("rung, count", [("1%", "240"), ("10%", "240"), ("100%", "2,370")])
def test_the_linear_probe_trains_section_four_five_s_step_count(rung, count, flags, capsys):
    """§4.5's floor through the command: the 1% rung trains the 10% rung's steps."""
    launch.main(["run", "--dry-run", "--rung", rung, *flags])
    assert f"{count} steps" in capsys.readouterr().out


def test_a_dry_run_writes_nothing_and_never_reads_the_corpus(box, flags):
    before = sorted(p.relative_to(box).as_posix() for p in box.rglob("*"))
    for verb in ("freeze", "run", "report"):
        assert launch.main([verb, "--dry-run", *flags]) == 0
    assert sorted(p.relative_to(box).as_posix() for p in box.rglob("*")) == before
    assert not (box / "corpus").exists()


def test_freeze_names_the_generator_hash_and_its_stages_not_a_cell(flags, capsys):
    launch.main(["freeze", "--dry-run", *flags])
    out = capsys.readouterr().out
    assert LandformConfig().artifact_hash in out
    assert LandformConfig().config_hash not in out
    assert "arm B1" not in out and "encoder" not in out
    assert "prereg bar — built (#199)" in out
    assert "geomorphon labels — built (#201)" in out
    assert "iwahashi labels — built (#203)" in out
    assert "ladders — built (#200)" in out
    assert "dataset card — not built yet (#218)" in out


# --- A cell cannot be half one task and half the other ------------------------


def test_a_task_one_cell_flag_is_refused_under_task_two_by_name(flags, capsys):
    with pytest.raises(ValueError, match="`--size` is a void-fill cell flag"):
        launch.main(["run", "--dry-run", "--size", "20k", *flags])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "flag, value", [("--rung", "1%"), ("--head", "linear"), ("--label-set", "iwahashi")]
)
def test_a_task_two_cell_flag_is_refused_under_task_one_by_name(flag, value, capsys):
    with pytest.raises(ValueError, match=f"`{flag}` is a landform cell flag"):
        launch.main(["run", "--dry-run", flag, value])
    assert capsys.readouterr().out == ""


def test_a_task_one_knob_through_set_is_refused_under_task_two(flags):
    with pytest.raises(ValueError, match="unknown config group\\(s\\) \\['size'\\]"):
        launch.main(["run", "--dry-run", *flags, "--set", "size=20k"])
    with pytest.raises(ValueError, match="head is one of Task 2's cell scalars"):
        launch.main(["run", "--dry-run", *flags, "--set", "head.lr=3e-4"])


# --- A typo stops before anything is spent -----------------------------------


def test_a_typo_in_an_override_stops_before_the_corpus_is_touched(flags, capsys):
    with pytest.raises(ValueError, match="unknown linear knob\\(s\\) \\['epoch'\\]"):
        launch.main(["run", "--dry-run", *flags, "--set", "linear.epoch=10"])
    assert capsys.readouterr().out == "", "it resolved something before refusing the config"


@pytest.mark.parametrize(
    "argv, refusal",
    [
        (["--arm", "B10"], "unknown Task 2 arm 'B10'"),
        (["--rung", "5%"], "unknown rung '5%'"),
        (["--label-set", "geomorphons"], "unknown label set 'geomorphons'"),
        (["--head", "mlp"], "unknown head 'mlp'"),
        (["--head", "upernet"], "arm B1 trains \\['linear'\\], so head 'upernet'"),
    ],
)
def test_a_cell_that_does_not_exist_is_refused_by_name(argv, refusal, flags, capsys):
    with pytest.raises(ValueError, match=refusal):
        launch.main(["run", "--dry-run", *argv, *flags])
    assert capsys.readouterr().out == ""


def test_tuning_is_not_task_ones_pass_and_names_its_ticket(flags):
    with pytest.raises(ValueError, match="#207"):
        launch.main(["run", "--dry-run", "--tune", *flags])


# --- Task 2's own config hash ------------------------------------------------


def _printed_hash(argv, capsys) -> str:
    """The config hash a `run --dry-run` prints on its first line."""
    launch.main(["run", "--dry-run", *argv])
    return capsys.readouterr().out.splitlines()[0].rsplit("config ", 1)[1]


def test_task_two_has_its_own_config_hash_and_a_moved_knob_moves_only_it(flags, capsys):
    base = _printed_hash(flags, capsys)
    moved = _printed_hash([*flags, "--set", "linear.epochs=31"], capsys)
    assert moved != base
    assert {base, moved}.isdisjoint(TASK1_HASHES.values())
    assert _printed_hash([], capsys) == TASK1_HASHES["A4-R5"]


def test_where_it_runs_moves_no_hash_and_a_named_default_head_is_the_same_cell(flags, capsys):
    base = _printed_hash(flags, capsys)
    assert _printed_hash([*flags, "--set", "runtime.device=cpu"], capsys) == base
    assert _printed_hash([*flags, "--set", 'paths.plan_path="elsewhere.md"'], capsys) == base
    assert _printed_hash([*flags, "--head", "linear"], capsys) == base


def test_the_generator_hash_moves_with_the_labels_and_not_with_the_cell_or_a_head():
    base = LandformConfig()
    assert base.replace(linear=base.linear.replace(epochs=31)).artifact_hash == base.artifact_hash
    assert base.replace(rung="1%", seed=2).artifact_hash == base.artifact_hash
    moved = base.replace(geomorphon=base.geomorphon.replace(search=60))
    assert moved.artifact_hash != base.artifact_hash


def test_a_config_file_names_only_the_knobs_its_run_moves(box, flags, capsys):
    path = box / "cell.json"
    path.write_text(json.dumps({"rung": "10%", "upernet": {"epochs": 41}}))
    want = LandformConfig(rung="10%", upernet=LandformConfig().upernet.replace(epochs=41))
    assert _printed_hash([*flags, "--config", str(path)], capsys) == want.config_hash


def test_the_config_round_trips_through_plain_json():
    cfg = LandformConfig(rung="1%", seed=1, label_set="iwahashi")
    again = LandformConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg and again.config_hash == cfg.config_hash


# --- Past --dry-run, each verb names the ticket that builds it ----------------


@pytest.mark.parametrize("verb, ticket", [("run", "#206"), ("report", "#216")])
def test_past_the_dry_run_a_verb_names_its_ticket_rather_than_returning_zero(verb, ticket, flags):
    with pytest.raises(NotImplementedError, match=ticket):
        launch.main([verb, *flags])


def test_freeze_names_every_stage_it_has_no_writer_for_rather_than_returning_zero(
    flags, monkeypatch
):
    # The built stages need inputs this empty box does not hold (#200's ladders walk Task 1's
    # head-train list); each is tested with its own fixture. What is under test is the refusal.
    from benchmark import landform

    pending = tuple(stage for stage in landform.FREEZE_STAGES if stage.write is None)
    monkeypatch.setattr(landform, "FREEZE_STAGES", pending)
    with pytest.raises(NotImplementedError) as exc:
        launch.main(["freeze", *flags])
    assert [stage.ticket for stage in pending] == ["#218"]
    assert "#218" in str(exc.value)
    for ticket in ("#199", "#200", "#201", "#203"):
        assert ticket not in str(exc.value), f"{ticket}'s stage has a writer now"


# --- Below the seam: §4.5's step-count function ------------------------------


def test_the_step_count_function_reproduces_section_four_five_s_table():
    """Typed from the plan: linear 240 / 240 / 2,370; UperNet and B7 1,280 / 1,280 / 12,520."""
    cfg = LandformConfig()
    table = {
        "linear": (240, 240, 2_370),
        "upernet": (1_280, 1_280, 12_520),
        "unet": (1_280, 1_280, 12_520),
    }
    for head, want in table.items():
        recipe = getattr(cfg, head)
        got = tuple(
            steps(
                epochs=recipe.epochs,
                batch_size=recipe.batch_size,
                n_train=RUNG_TILES[rung],
                floor=cfg.schedule.step_floor,
            )
            for rung in RUNGS
        )
        assert got == want, head
