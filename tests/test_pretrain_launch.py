"""The launch entrypoint — the documented way to start a run (#95).

What a rented box invokes, and therefore the thing that has to be right *before* the hour is
paid for: the config it assembled, the levers it actually got (§7.2's attention path, §7.3's
compile), and the throughput it is achieving while it runs.

The tests here are about the command, not about training — the loop itself is #93's and is
tested in `test_pretrain_train.py`. What is asserted is that the flags reach the config, that
the counters reach the output, and that a run can be inspected without being started.
"""

from __future__ import annotations

import json

import pytest
import torch

from pretrain import launch
from pretrain.config import Architecture, Data, Optimization, RunConfig, Runtime, Schedule
from tests.conftest import SPLIT_COGS, build_corpus

#: 5 COGs x 4 tiles = 20 tiles, 8 of them train — 2 batches of 4, so 2 optimizer steps/epoch.
TILES_PER_COG = 4


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return build_corpus(
        tmp_path_factory.mktemp("launch_corpus"), SPLIT_COGS, tiles_per_cog=TILES_PER_COG
    )


def tiny_config(corpus) -> RunConfig:
    """R1's recipe at laptop scale, eager — compiling a 2-block model on CPU costs minutes."""
    return RunConfig(
        run_id="test",
        architecture=Architecture(
            encoder_dim=64, encoder_depth=2, encoder_heads=2,
            decoder_dim=64, decoder_depth=1, decoder_heads=2,
        ),
        optimization=Optimization(batch_size=4, accum_iter=1, blr=3e-3, lr_batch_base=4),
        schedule=Schedule(epochs=150, warmup_epochs=6, milestones=(150,)),
        data=Data(
            corpus_dir=str(corpus),
            split_assignment_path=str(corpus / "no-assignment.csv"),
            split_blocks_path=str(corpus / "no-blocks.json"),
            num_workers=0,
        ),
        runtime=Runtime(compile=False),
    )


@pytest.fixture
def config_file(corpus, tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(tiny_config(corpus).to_dict()))
    return path


# --- What a run is launched with ----------------------------------------------


def test_it_trains_from_a_config_file_and_reports_every_step(config_file, capsys):
    assert launch.main(["--config", str(config_file), "--steps", "4", "--log-every", "1"]) == 0
    out = capsys.readouterr().out
    assert "DEMAE Phase 1" in out  # #89's header, before anything is spent
    assert len([line for line in out.splitlines() if line.startswith("step ")]) == 4


def test_the_overrides_reach_the_run(config_file, capsys):
    launch.main(["--config", str(config_file), "--steps", "1", "--set", "masking.ratio=0.5"])
    out = capsys.readouterr().out
    assert "50% random — 98 hidden, 98 visible" in out  # off the header, so off the real config


def test_a_dry_run_says_what_would_happen_and_trains_nothing(config_file, capsys):
    assert launch.main(["--config", str(config_file), "--steps", "4", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DEMAE Phase 1" in out
    assert "4 steps" in out  # what it *would* run
    assert not [line for line in out.splitlines() if line.startswith("step ")]


def test_a_typo_in_an_override_stops_before_the_corpus_is_touched(config_file):
    with pytest.raises(ValueError, match="unknown masking knob"):
        launch.main(["--config", str(config_file), "--steps", "1", "--set", "masking.rate=0.5"])


# --- Steps, and the epochs they come from -------------------------------------


def test_epochs_are_converted_to_steps_by_what_the_split_actually_yields(corpus):
    cfg = tiny_config(corpus)
    from pretrain.train import build_loader, steps_per_epoch_for

    loader = build_loader(cfg, "train")
    assert steps_per_epoch_for(loader, cfg) == 2  # 8 train tiles / batch 4 / accum 1
    assert launch.resolve_steps(cfg, loader, steps=None, epochs=3) == 6
    assert launch.resolve_steps(cfg, loader, steps=5, epochs=None) == 5


def test_the_default_run_length_is_the_whole_schedule(corpus):
    # A launcher whose default was "one epoch" would make the recipe's 400 a documentation
    # detail. Asked for nothing, it runs the config's own schedule to its end.
    cfg = tiny_config(corpus)
    from pretrain.train import build_loader

    loader = build_loader(cfg, "train")
    assert launch.resolve_steps(cfg, loader, steps=None, epochs=None) == 150 * 2


def test_the_run_reports_the_epochs_it_will_take(config_file, capsys):
    launch.main(["--config", str(config_file), "--epochs", "2", "--dry-run"])
    assert "4 steps" in capsys.readouterr().out  # 2 epochs x 2 steps


# --- The levers, as measured rather than as configured ------------------------


def test_the_runtime_line_reports_the_attention_path_off_the_built_model(config_file, capsys):
    # Reported here, *refused* in `Trainer.from_config` — see
    # `test_a_model_whose_attention_is_not_what_the_recipe_asked_for_is_refused` in
    # test_pretrain_train.py, which is the path every way of building a trainer goes through.
    launch.main(["--config", str(config_file), "--steps", "1"])
    out = capsys.readouterr().out
    assert "attention sdpa" in out
    assert "compile off" in out


def test_the_runtime_line_names_the_device_and_the_parameter_split(corpus):
    from pretrain.model import MaskedAutoencoder

    cfg = tiny_config(corpus)
    line = launch.runtime_line(MaskedAutoencoder.from_config(cfg), cfg, device="cpu")
    assert "cpu" in line and "encoder" in line and "decoder" in line


# --- The counters, in the output ----------------------------------------------


def test_every_step_line_carries_the_three_throughput_numbers(config_file, capsys):
    launch.main(["--config", str(config_file), "--steps", "2", "--log-every", "1"])
    steps = [line for line in capsys.readouterr().out.splitlines() if line.startswith("step ")]
    assert len(steps) == 2
    for line in steps:
        assert "samples/s" in line and "loader" in line


def test_the_run_summarizes_per_epoch_and_over_the_whole_run(config_file, capsys):
    launch.main(["--config", str(config_file), "--steps", "4"])
    out = capsys.readouterr().out
    epochs = [line for line in out.splitlines() if line.startswith("epoch ")]
    assert len(epochs) == 2  # 2 steps per epoch, 4 steps
    assert all("samples/s" in line for line in epochs)
    assert "run " in out


def test_the_summary_separates_the_first_step_from_the_steady_state(config_file, capsys):
    # Step 1 pays for `torch.compile` and the loader's first fill. Averaged in, it is what
    # makes a 3,000 img/s box report 400 — and §7.5's gate is read off this line.
    launch.main(["--config", str(config_file), "--steps", "3"])
    out = capsys.readouterr().out
    assert "steady" in out


# --- The release backbone -----------------------------------------------------


def test_the_vit_base_release_config_builds_and_takes_a_step(corpus, capsys):
    """§2.1's ViT-B/16, launched the way R5 launches it — so #81 measures rather than debugs.

    The *shape* is the release recipe untouched: 768x12 encoder, 512x4 decoder, §2.1's 86M.
    Three knobs move, and all three are facts about this box rather than about the recipe — the
    physical batch (§12 already expects it to move on the real box, which is why the LR rule is
    written on the effective batch), the worker count, and compile, which on a CPU would spend
    minutes optimizing a graph nothing here is timing.

    The ViT-B throughput and peak-memory numbers themselves are **#81's**, measured on the
    rented 32 GB box. What this asserts is the thing that would otherwise be discovered there,
    on the clock: that the config builds, that the step runs, and that the parameter count is
    the one §2.1 budgets for.
    """
    try:
        code = launch.main([
            "--vit-base",
            "--corpus", str(corpus),
            "--assignment", str(corpus / "no-assignment.csv"),
            "--blocks", str(corpus / "no-blocks.json"),
            "--set", "optimization.batch_size=2",
            "--set", "optimization.accum_iter=1",
            "--set", "data.num_workers=0",
            "--set", "runtime.compile=false",
            "--steps", "1",
        ])
    except (MemoryError, torch.OutOfMemoryError) as exc:  # pragma: no cover - box-dependent
        pytest.skip(f"not enough memory for a ViT-B/16 step here: {exc}")
    assert code == 0
    out = capsys.readouterr().out
    assert "85.3M encoder" in out  # §2.1's 86M backbone, off the model it just built
    assert "768d x 12" in out
    assert len([line for line in out.splitlines() if line.startswith("step ")]) == 1


def test_a_single_step_run_reports_no_steady_state_rather_than_an_empty_one(config_file, capsys):
    # There is no steady state to report from one step, and a line reading 0.0 img/s would be
    # read as a measurement rather than as an absence.
    launch.main(["--config", str(config_file), "--steps", "1"])
    assert "steady" not in capsys.readouterr().out
