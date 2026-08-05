"""The launch entrypoint — the one command a run is started with (#95, #106).

What a rented box invokes, and therefore the thing that has to be right *before* the hour is
paid for: the config it assembled, the levers it actually got (§7.2's attention path, §7.3's
compile), the throughput it is achieving while it runs — and, since #106, the three things
whose absence nothing noticed.

**A 400-epoch run launched by the documented command produced no checkpoints, no W&B run and
no val pass.** Each piece existed: #94's `train_resumable`, #97's `EpochMonitor`, #98's grids,
#99's probe. Nothing composed them, so `sync_checkpoints.sh` would have reported `ok` over an
empty directory for forty hours and §11's survival argument was unbacked on the path that
ships. The `# --- What the shipping command actually does` group is that gap, as tests: a
checkpoint exists, a logger was opened, and an override survived.

The rest of the tests here are about the command rather than about training — the loop itself
is #93's and #94's, tested in `test_pretrain_train.py` and `test_pretrain_checkpoint.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pretrain import checkpoint as ckpt_mod
from pretrain import launch
from pretrain.config import Architecture, Data, Monitoring, Optimization, RunConfig, Runtime, Schedule
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
            # §10.2's panel and §10.3's probe set are committed against the real 500k corpus
            # and would be refused against this one; the tests that want them commit their own.
            val_panel_path=str(corpus / "no-panel.csv"),
            probe_set_path=str(corpus / "no-probe-set.csv"),
            num_workers=0,
        ),
        runtime=Runtime(compile=False),
        # The sink is off by default here, and the checkpoint directory is always passed: a
        # unit test must not open W&B runs or write `runs/test/` into whatever directory pytest
        # happened to be started from. `test_the_shipping_command_opens_a_run_and_logs_the_
        # epoch_panel` is where the real thing is exercised, offline, into a tmp path.
        monitoring=Monitoring(wandb_mode="disabled", probe_every_epochs=0),
    )


@pytest.fixture
def config_file(corpus, tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(tiny_config(corpus).to_dict()))
    return path


@pytest.fixture
def run(config_file, tmp_path):
    """`launch.main`, with the flags every test needs and none it is asserting on.

    The checkpoint directory is a tmp path rather than the `runs/<run_id>` default because the
    default is a *repository* path — and `test_the_checkpoint_directory_defaults_to_the_run_id`
    is where that default is checked, without writing anything.
    """

    def go(*argv, checkpoints=True):
        flags = ["--checkpoint-dir", str(tmp_path / "ckpt")] if checkpoints else []
        return launch.main(["--config", str(config_file), *flags, *argv])

    return go


# --- What a run is launched with ----------------------------------------------


def test_it_trains_from_a_config_file_and_reports_every_step(run, capsys):
    assert run("--steps", "4", "--log-every", "1") == 0
    out = capsys.readouterr().out
    assert "DEMAE Phase 1" in out  # #89's header, before anything is spent
    assert len([line for line in out.splitlines() if line.startswith("step ")]) == 4


def test_the_overrides_reach_the_run(run, capsys):
    run("--steps", "1", "--set", "masking.ratio=0.5")
    out = capsys.readouterr().out
    assert "50% random — 98 hidden, 98 visible" in out  # off the header, so off the real config


def test_a_dry_run_says_what_would_happen_and_trains_nothing(run, capsys):
    assert run("--steps", "4", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "DEMAE Phase 1" in out
    assert "4 steps" in out  # what it *would* run
    assert not [line for line in out.splitlines() if line.startswith("step ")]


def test_a_typo_in_an_override_stops_before_the_corpus_is_touched(run):
    with pytest.raises(ValueError, match="unknown masking knob"):
        run("--steps", "1", "--set", "masking.rate=0.5")


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


def test_the_run_reports_the_epochs_it_will_take(run, capsys):
    run("--epochs", "2", "--dry-run")
    assert "4 steps" in capsys.readouterr().out  # 2 epochs x 2 steps


# --- The levers, as measured rather than as configured ------------------------


def test_the_runtime_line_reports_the_attention_path_off_the_built_model(run, capsys):
    # Reported here, *refused* in `Trainer.from_config` — see
    # `test_a_model_whose_attention_is_not_what_the_recipe_asked_for_is_refused` in
    # test_pretrain_train.py, which is the path every way of building a trainer goes through.
    run("--steps", "1")
    out = capsys.readouterr().out
    assert "attention sdpa" in out
    assert "compile off" in out


def test_the_runtime_line_names_the_device_and_the_parameter_split(corpus):
    from pretrain.model import MaskedAutoencoder

    cfg = tiny_config(corpus)
    line = launch.runtime_line(MaskedAutoencoder.from_config(cfg), cfg, device="cpu")
    assert "cpu" in line and "encoder" in line and "decoder" in line


# --- The counters, in the output ----------------------------------------------


def test_every_step_line_carries_the_three_throughput_numbers(run, capsys):
    run("--steps", "2", "--log-every", "1")
    steps = [line for line in capsys.readouterr().out.splitlines() if line.startswith("step ")]
    assert len(steps) == 2
    for line in steps:
        assert "samples/s" in line and "loader" in line


def test_the_run_summarizes_per_epoch_and_over_the_whole_run(run, capsys):
    run("--steps", "4")
    out = capsys.readouterr().out
    epochs = [line for line in out.splitlines() if line.startswith("epoch ")]
    assert len(epochs) == 2  # 2 steps per epoch, 4 steps
    assert all("samples/s" in line for line in epochs)
    assert "run " in out


def test_the_summary_separates_the_first_step_from_the_steady_state(run, capsys):
    # Step 1 pays for `torch.compile` and the loader's first fill. Averaged in, it is what
    # makes a 3,000 img/s box report 400 — and §7.5's gate is read off this line.
    run("--steps", "3")
    out = capsys.readouterr().out
    assert "steady" in out


# --- §7.3 / #109: what a run says about its compiles ---------------------------


def test_a_healthy_run_signs_off_saying_seven_three_held():
    # The line #109 wanted and did not have. R1 finished 400 clean epochs and its last word on
    # the subject was a warning recommending a ~2.4x slowdown, printed with the step number of
    # the end of the run — the opposite end from where the compile actually happened.
    summary = launch.format_compile_summary(train=0, monitoring=18, first_step=None)
    assert "training step: 0 frames after warmup (§7.3 holds)" in summary
    assert "WARNING" not in summary
    assert "runtime.compile=false" not in summary


def test_an_eager_run_says_the_lever_is_off_rather_than_that_seven_three_held(run, capsys):
    # The fixture runs eager, which is every launch test here. "0 frames, §7.3 holds" would be
    # true and useless of a run that compiled nothing — and it is the reading that lets an
    # eager run pass for a fast one, which is the failure §7's whole verification exists for.
    run("--steps", "4")
    out = capsys.readouterr().out
    assert "compile          off (runtime.compile=false)" in out
    assert "§7.3 holds" not in out
    assert "WARNING" not in out


def test_the_footer_reports_where_the_recompile_was_first_seen_not_where_the_run_ended(run, capsys):
    # #109's third problem: the footer called the formatter with the *current* step where it
    # meant the *recorded first-seen* one, so the inline line said step 232 and the footer said
    # step 92,400 — for one event. Anyone reading only the footer, which is what a finished run
    # leaves you, was told it happened at the far end of the run.
    summary = launch.format_compile_summary(train=2, monitoring=18, first_step=231)
    assert "first at step 232" in summary
    assert "92400" not in summary and "step 4" not in summary
    assert "§7.3 does NOT hold" in summary


def test_the_footer_calls_the_monitoring_frames_expected_rather_than_a_fault():
    summary = launch.format_compile_summary(train=0, monitoring=18, first_step=None)
    assert "18 frame(s) at the first epoch boundary" in summary
    assert "expected" in summary
    assert "WARNING" not in summary


def test_the_first_training_step_recompile_does_not_recommend_turning_compile_off():
    # The remedy costs ~2.4x for all 400 epochs; the fault it is being offered for costs ~60 s
    # once. The flag belongs to the loop case, and only a second frame proves that case.
    first = launch.format_recompile_warning(1, 231)
    assert "step 232" in first and "2.4x" in first
    assert "runtime.compile=false" not in first

    second = launch.format_recompile_escalation(4, 1, 900)
    assert "runtime.compile=false" in second and "1 -> 4" in second
    assert "step 901" in second


# --- The release backbone -----------------------------------------------------


def test_the_vit_base_release_config_builds_and_takes_a_step(corpus, tmp_path, capsys):
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
            "--set", 'monitoring.wandb_mode="disabled"',
            "--set", f'data.val_panel_path="{corpus / "no-panel.csv"}"',
            "--set", f'data.probe_set_path="{corpus / "no-probe-set.csv"}"',
            "--checkpoint-dir", str(tmp_path / "ckpt"),
            "--steps", "1",
        ])
    except (MemoryError, torch.OutOfMemoryError) as exc:  # pragma: no cover - box-dependent
        pytest.skip(f"not enough memory for a ViT-B/16 step here: {exc}")
    assert code == 0
    out = capsys.readouterr().out
    assert "85.3M encoder" in out  # §2.1's 86M backbone, off the model it just built
    assert "768d x 12" in out
    assert len([line for line in out.splitlines() if line.startswith("step ")]) == 1


def test_a_single_step_run_reports_no_steady_state_rather_than_an_empty_one(run, capsys):
    # There is no steady state to report from one step, and a line reading 0.0 img/s would be
    # read as a measurement rather than as an absence.
    run("--steps", "1")
    assert "steady" not in capsys.readouterr().out


# --- What the shipping command actually does (#106) ---------------------------
#
# The three assertions the ticket names, plus the decisions around them. Each of these was
# green-by-absence before #106: the launcher called `train`, so there was no checkpoint to
# look for, no logger to have been opened, and an override that reached the header and then
# nothing that recorded it.


def test_the_shipping_command_writes_a_checkpoint(run, tmp_path):
    """§11's per-epoch trigger, on the path that ships.

    R0 (#81) found the documented command producing none at all, with `sync_checkpoints.sh`
    reporting `ok` over an empty directory every pass. A box that vanishes at hour 40 of a
    43-hour run then takes the whole run with it — which is the outcome §11 exists to prevent
    and the reason the runbook calls the sync loop "not optional".
    """
    assert run("--epochs", "1") == 0
    written = ckpt_mod.discover(tmp_path / "ckpt")
    assert written, "a whole epoch ended and nothing was checkpointed"
    assert written[-1].epoch == 1 and written[-1].step == 2  # 2 steps/epoch, one epoch done


def test_the_checkpoint_carries_the_recipe_the_run_was_launched_with(run, tmp_path):
    # The override survives all the way into what the run *recorded*, not merely into what it
    # printed. R2 (#83) and R3 (#84) are `--set` ablations, so a checkpoint that recorded R1's
    # ratio would be an ablation's released encoder mislabelled as the baseline's.
    run("--epochs", "1", "--set", "masking.ratio=0.5")
    loaded = ckpt_mod.load_checkpoint(ckpt_mod.latest(tmp_path / "ckpt").path)
    assert loaded.config.masking.ratio == 0.5
    assert loaded.config_hash == loaded.config.config_hash


def test_the_shipping_command_opens_a_run_and_logs_the_epoch_panel(config_file, tmp_path):
    """A real W&B run, offline, with §10's numbers in it.

    Offline rather than mocked for the reason `pretrain.sink` gives: it is the same code path
    as online with one field moved, it needs no account and no network, and the translation
    from #97's payload types is exactly the part a fake logger would not exercise. Not skipped
    when wandb is absent, either: it is a declared dependency of the `pretrain` extra
    (`pyproject.toml`), and one of #106's three named assertions quietly disappearing on the
    box where it is not installed is the shape of failure this whole ticket is about.
    """
    launch.main([
        "--config", str(config_file),
        "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--set", 'monitoring.wandb_mode="offline"',
        "--set", f'monitoring.wandb_dir="{tmp_path}"',
        "--epochs", "1",
    ])
    runs = list((tmp_path / "wandb").glob("offline-run-*"))
    assert len(runs) == 1, "the shipping command opened no W&B run"
    body = next(runs[0].glob("*.wandb")).read_bytes()
    for key in (b"train/loss", b"val/loss", b"val/per_patch_loss", b"throughput/samples_per_s"):
        assert key in body, key


def test_every_epoch_reports_the_val_pass_and_not_only_the_throughput(run, capsys):
    # §10.1's whole panel, on the console a human is watching. The launcher used to print an
    # epoch's samples/s and nothing else, because no val pass was being run to report.
    run("--epochs", "2")
    epochs = [line for line in capsys.readouterr().out.splitlines() if line.startswith("epoch ")]
    assert len(epochs) == 2
    for line in epochs:
        assert "train" in line and "val" in line and "lr" in line and "samples/s" in line


def test_an_alarm_reaches_the_terminal_the_run_is_being_watched_in(config_file, tmp_path, capsys):
    # An alarm that only exists in W&B is an alarm nobody sees during the run that fired it.
    cfg = json.loads(config_file.read_text())
    cfg["monitoring"]["loader_wait_alarm"] = 1e-6
    config_file.write_text(json.dumps(cfg))
    launch.main([
        "--config", str(config_file), "--checkpoint-dir", str(tmp_path / "ckpt"), "--epochs", "1"
    ])
    out = capsys.readouterr().out
    assert "[STOP]" in out and "loader_wait" in out


# --- Where the checkpoints go, and what a relaunch does -----------------------


def test_the_checkpoint_directory_defaults_to_the_run_id(corpus):
    # Two arms writing into one directory is how R2 resumes R1's weights, and with resume on by
    # default that would happen silently. `--set run_id=R2` is how an ablation is launched, so
    # the directory follows it without a second flag to remember.
    cfg = tiny_config(corpus)
    assert launch.checkpoint_directory(cfg, None) == Path("runs") / "test"
    assert launch.checkpoint_directory(cfg.replace(run_id="R2"), None) == Path("runs") / "R2"
    assert launch.checkpoint_directory(cfg, "/workspace/r1") == Path("/workspace/r1")


def test_relaunching_the_same_command_continues_rather_than_starting_over(run, capsys, tmp_path):
    """§11's actual failure mode: the instance is gone and the command is typed again.

    vast.ai instances vanish permanently, so a first launch and a relaunch are the same command
    on a fresh box. A launcher that needed `--resume` remembered at exactly that moment would
    be a launcher that silently restarted a 40-hour run from epoch 0 — and nothing about the
    loss curve would say so until someone compared it against the checkpoint's step count.
    """
    run("--epochs", "1")
    assert ckpt_mod.latest(tmp_path / "ckpt").step == 2
    capsys.readouterr()

    run("--epochs", "2")
    out = capsys.readouterr().out
    assert "resuming         " in out and "checkpoint_e0001" in out
    steps = [line for line in out.splitlines() if line.startswith("step ")]
    assert len(steps) == 2, "the relaunch retrained epoch 0 instead of continuing"
    assert ckpt_mod.latest(tmp_path / "ckpt").step == 4


def test_no_resume_starts_over_past_a_directory_that_already_holds_a_run(run, capsys, tmp_path):
    run("--epochs", "1")
    capsys.readouterr()
    run("--epochs", "1", "--no-resume")
    out = capsys.readouterr().out
    assert "resuming         (nothing" in out
    assert len([line for line in out.splitlines() if line.startswith("step ")]) == 2


def test_a_first_launch_says_there_is_nothing_to_resume_from(run, capsys):
    run("--epochs", "1")
    assert "resuming         (nothing to resume from — this is a first launch)" in \
        capsys.readouterr().out


def test_no_checkpoints_says_so_rather_than_quietly_writing_none(run, capsys, tmp_path):
    assert run("--epochs", "1", "--no-checkpoints", checkpoints=False) == 0
    out = capsys.readouterr().out
    assert "checkpoints      none" in out
    assert not (tmp_path / "ckpt").exists()


def test_a_dry_run_says_where_the_checkpoints_and_the_metrics_would_go(run, capsys, tmp_path):
    # §15's pre-flight checklist is walked in this form, and "where does this write" is item 4
    # and item 7's worth of it. Discovering the answer after the first epoch is discovering it
    # on the clock.
    run("--epochs", "1", "--dry-run")
    out = capsys.readouterr().out
    assert str(tmp_path / "ckpt") in out
    assert "every epoch and every 45 min" in out
    assert "wandb disabled" in out
    assert not (tmp_path / "ckpt").exists()


def test_the_checkpoint_interval_is_the_wall_clock_trigger_the_flag_names(run, capsys):
    run("--epochs", "1", "--checkpoint-minutes", "30", "--dry-run")
    assert "every epoch and every 30 min" in capsys.readouterr().out


# --- §10.2's grids and §10.3's probe, through the command that ships ----------


@pytest.fixture(scope="module")
def instrumented_corpus(tmp_path_factory):
    """A corpus with two val COGs, so every stratum has the two tiles a panel needs."""
    from tests.test_pretrain_recon import RECON_COGS
    from tests.test_pretrain_train import smooth_tile

    return build_corpus(
        tmp_path_factory.mktemp("launch_instruments"),
        RECON_COGS,
        tiles_per_cog=TILES_PER_COG,
        tile_fn=smooth_tile,
    )


def test_the_grids_and_the_probe_fire_on_the_shipping_path_too(instrumented_corpus, tmp_path):
    """§10.2 and §10.3 reach the sink from `launch.main`, not only from the loop under it.

    The ticket lists them among what a 400-epoch R1 would not have produced — "no recon grids,
    no probe" — and a test that drove `train_epochs` would have been green throughout the
    period when nothing called it. So this drives the command, with both cadences at 1, and
    reads the artifacts off a real offline run.
    """
    from pretrain.probe import draw_probe_set, write_probe_set
    from tests.test_pretrain_recon import panel_for

    manifest = instrumented_corpus / "manifest.parquet"
    probe_set = write_probe_set(draw_probe_set(manifest, size=8), tmp_path / "probe.csv")
    cfg = tiny_config(instrumented_corpus).replace(
        data=tiny_config(instrumented_corpus).data.replace(
            val_panel_path=str(panel_for(instrumented_corpus, tmp_path)),
            probe_set_path=str(probe_set),
        ),
    )
    config_file = tmp_path / "instrumented.json"
    config_file.write_text(json.dumps(cfg.to_dict()))

    assert launch.main([
        "--config", str(config_file),
        "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--set", "monitoring.recon_every_epochs=1",
        "--set", "monitoring.probe_every_epochs=1",
        # 8 probe tiles over 2 val COGs, and a query's own COG is excluded from its bank
        # (corpus card §4.6) — so the shipped k of 20 has nothing to vote over here.
        "--set", "monitoring.probe_k=2",
        "--set", 'monitoring.wandb_mode="offline"',
        "--set", f'monitoring.wandb_dir="{tmp_path}"',
        "--epochs", "1",
    ]) == 0

    run = next((tmp_path / "wandb").glob("offline-run-*"))
    body = next(run.glob("*.wandb")).read_bytes()
    assert b"recon/heatmap" in body and b"recon/hillshade" in body
    assert b"probe/stratum_acc" in body and b"probe/set_md5" in body
    # Real PNGs, two triptychs per panel tile — what a `wandb sync` would upload.
    assert list(run.glob("files/media/images/recon/*.png"))


def test_the_probe_point_is_printed_where_the_person_watching_will_see_it(
    instrumented_corpus, tmp_path, capsys
):
    # With §10.3's caveat attached: the person most likely to over-read a probe number is the
    # one watching the run it came out of.
    from pretrain.probe import draw_probe_set, write_probe_set

    base = tiny_config(instrumented_corpus)
    cfg = base.replace(
        data=base.data.replace(
            probe_set_path=str(
                write_probe_set(
                    draw_probe_set(instrumented_corpus / "manifest.parquet", size=8),
                    tmp_path / "probe.csv",
                )
            )
        )
    )
    config_file = tmp_path / "probed.json"
    config_file.write_text(json.dumps(cfg.to_dict()))
    launch.main([
        "--config", str(config_file),
        "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--set", "monitoring.probe_every_epochs=1",
        "--set", "monitoring.probe_k=2",
        "--epochs", "1",
    ])
    out = capsys.readouterr().out
    assert "probe" in out.lower() and "SSL4EO" in out
