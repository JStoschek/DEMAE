"""The numbers a run reports every epoch, and the alarms wired onto them (#97, plan §10).

Four things are on trial, and they fail in four different ways.

**The val loss is comparable epoch to epoch, or it is nothing.** §10.1's whole design rests on
one property — the same tiles under the same masks — and a val pass that redrew its masks would
produce a number that moves for reasons that have nothing to do with the model. So the tests
here are about *reproduction*: the same model scores the same loss twice, exactly, and a
different mask seed scores a different one.

**The val pass must not perturb the run it is measuring.** A pass that consumed from the ambient
RNG would change the masks the *next training step* draws, which makes a run's trajectory depend
on whether anyone was watching. That is a Heisenberg bug and no curve would show it.

**§4's distribution is not the mean.** The whole point of logging it is §4's rule — *if the
top tail is near-flat patches, raise the variance floor before concluding anything about the
model* — and answering that needs the cross-tab of loss against flatness, not a scalar.

**The throughput panel displays #95's counters rather than growing a second stopwatch.** The
issue splits it that way deliberately, and a monitor that measured its own wall clock would
disagree with the entrypoint's numbers by whatever it forgot to include.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from pretrain import monitor as monitor_mod
from pretrain.config import RunConfig
from pretrain.monitor import EpochMonitor, LossHistogram, evaluate, train_epochs
from pretrain.sink import Histogram, MemoryLogger
from pretrain.throughput import ThroughputSummary
from pretrain.train import Trainer, build_loader, seed_run, steps_per_epoch_for
from tests.conftest import SPLIT_COGS, build_corpus
from tests.test_pretrain_train import TILES_PER_COG, smooth_tile, tiny_config


@pytest.fixture(scope="module")
def smooth_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("monitor_corpus")
    return build_corpus(root, SPLIT_COGS, tiles_per_cog=TILES_PER_COG, tile_fn=smooth_tile)


@pytest.fixture
def cfg(smooth_corpus) -> RunConfig:
    # The sink is off by default here: every test that cares about what was logged passes its
    # own MemoryLogger, and the ones that do not should not be opening W&B runs.
    base = tiny_config(smooth_corpus)
    return base.replace(monitoring=base.monitoring.replace(wandb_mode="disabled"))


@pytest.fixture
def trained(cfg):
    """A trainer and the val loader beside it — what a val pass needs and nothing more."""
    seed_run(cfg)
    train_loader = build_loader(cfg, cfg.data.train_split)
    trainer = Trainer.from_config(
        cfg, steps_per_epoch=steps_per_epoch_for(train_loader, cfg), device="cpu"
    )
    val_loader = build_loader(cfg, cfg.data.val_split, augment=False, shuffle=False)
    return trainer, val_loader


SEED = RunConfig().monitoring.val_mask_seed


# --- §10.1: the fixed-mask val pass -------------------------------------------


def test_the_val_pass_reproduces_exactly_on_an_unchanged_model(trained):
    trainer, val_loader = trained
    first = evaluate(trainer, val_loader, mask_seed=SEED)
    second = evaluate(trainer, val_loader, mask_seed=SEED)
    assert first.loss == second.loss  # exactly: the same tiles, under the same masks
    assert (first.tiles, first.patches) == (second.tiles, second.patches)


def test_a_different_mask_seed_scores_a_different_number(trained):
    # The guard on the test above: it must be reproducing a *measurement*, not a constant.
    trainer, val_loader = trained
    assert evaluate(trainer, val_loader, mask_seed=SEED).loss != evaluate(
        trainer, val_loader, mask_seed=SEED + 1
    ).loss


def test_the_val_loss_moves_when_the_model_does(trained, cfg):
    # And the other guard: a fixed mask must not have frozen the number itself.
    trainer, val_loader = trained
    before = evaluate(trainer, val_loader, mask_seed=SEED).loss
    for group in [[b] for b in build_loader(cfg, "train")]:
        trainer.step(group, step=20)
    assert evaluate(trainer, val_loader, mask_seed=SEED).loss != before


def test_the_val_pass_draws_nothing_from_the_training_stream(trained):
    # A val pass that consumed ambient entropy would change the masks the next training step
    # draws — so a run's trajectory would depend on whether anyone was measuring it.
    trainer, val_loader = trained
    torch.manual_seed(1234)
    expected = torch.rand(4)
    torch.manual_seed(1234)
    evaluate(trainer, val_loader, mask_seed=SEED)
    assert torch.equal(torch.rand(4), expected)


def test_the_val_pass_scores_every_val_tile_exactly_once(trained, cfg):
    trainer, val_loader = trained
    assert evaluate(trainer, val_loader, mask_seed=SEED).tiles == len(val_loader.dataset)
    assert val_loader.dataset.split == cfg.data.val_split


def test_the_val_loss_is_a_mean_over_patches_and_not_a_mean_of_batch_means(cfg):
    # With `drop_last` off the last val batch is short, and averaging batch means would weight
    # its tiles more heavily than the rest — a val number that moves with the batch size.
    odd = cfg.replace(optimization=cfg.optimization.replace(batch_size=3))
    seed_run(odd)
    trainer = Trainer.from_config(odd, steps_per_epoch=1, device="cpu")
    loader = build_loader(odd, odd.data.val_split, augment=False, shuffle=False)
    assert [len(b["image"]) for b in loader] == [3, 1]  # the shape the two answers differ on

    result = evaluate(trainer, loader, mask_seed=SEED)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    weighted_sum = weight_sum = 0.0
    batch_means = []
    with torch.no_grad():
        for batch in loader:
            out = trainer.score(batch, generator=generator).out
            weighted_sum += float((out.per_patch * out.weight).sum())
            weight_sum += float(out.weight.sum())
            batch_means.append(float(out.loss))
    assert result.loss == pytest.approx(weighted_sum / weight_sum, rel=1e-6)
    assert result.loss != pytest.approx(sum(batch_means) / len(batch_means), rel=1e-6)


def test_no_gradient_and_no_training_mode_survives_the_val_pass(trained):
    trainer, val_loader = trained
    trainer.model.train()
    evaluate(trainer, val_loader, mask_seed=SEED)
    assert trainer.model.training  # restored, so the next epoch trains as it would have
    assert all(p.grad is None for p in trainer.model.parameters())


# --- §4: the per-patch loss distribution --------------------------------------


def histogram_over(losses, *, flat=None, weight=None) -> LossHistogram:
    """A one-batch histogram over ``losses``, shaped ``(1, n)`` as the loss module emits."""
    per_patch = torch.tensor([losses], dtype=torch.float32)
    hist = LossHistogram()
    hist.update(
        per_patch,
        torch.ones_like(per_patch) if weight is None else torch.tensor([weight], dtype=torch.float32),
        torch.tensor([flat or [False] * len(losses)], dtype=torch.bool),
    )
    return hist


def test_the_distribution_counts_every_scored_patch_and_no_unscored_one():
    # §4 excludes an all-ocean patch and every unmasked one from the objective; a distribution
    # that binned them anyway would describe a population the loss never saw.
    hist = histogram_over([0.1, 0.2, 0.3, 0.4], weight=[1.0, 1.0, 0.0, 0.0])
    assert hist.n == 2
    assert hist.mean == pytest.approx(0.15)


def test_quantiles_come_back_in_order_and_bracket_the_bulk():
    hist = histogram_over(list(np.linspace(0.1, 2.0, 200)))
    p50, p90, p99 = hist.quantile(0.5), hist.quantile(0.9), hist.quantile(0.99)
    assert p50 < p90 < p99
    assert 0.9 < p50 < 1.2  # the median of a uniform 0.1-2.0 spread, to within a log bin
    assert hist.largest == pytest.approx(2.0)


def test_a_near_flat_patch_is_cross_tabbed_against_its_loss():
    # §4's rule needs the *joint* distribution: a top tail made of near-flat patches means
    # raise the variance floor, and a top tail of real terrain means look at the model.
    losses = [0.1, 0.2, 0.3, 40.0, 50.0]
    flat_tail = histogram_over(losses, flat=[False, False, False, True, True])
    real_tail = histogram_over(losses, flat=[True, True, False, False, False])
    # 0.7 of five values lands between 0.3 and 40.0, and the tail is bin-granular: it is the
    # two decades holding 40 and 50, not an interpolated cut.
    assert flat_tail.tail_flat_fraction(0.7) == pytest.approx(1.0)
    assert real_tail.tail_flat_fraction(0.7) == pytest.approx(0.0)
    # Same marginal, opposite answers — which is the whole reason the cross-tab exists.
    assert flat_tail.flat_fraction == real_tail.flat_fraction == pytest.approx(0.4)


def test_a_collapsed_target_shows_as_a_spike_at_the_bottom_of_the_distribution():
    # The blind spot §10.1's collapse alarm cannot cover — a run already broken at epoch 0, so
    # there is no baseline to be a fraction of. A broken norm_pix target puts every patch here.
    assert histogram_over([1e-12] * 10).collapsed_fraction == pytest.approx(1.0)
    assert histogram_over([0.4] * 10).collapsed_fraction == pytest.approx(0.0)


def test_a_non_finite_loss_is_counted_apart_rather_than_filed_as_the_largest():
    # `np.digitize` sorts a NaN past the last boundary, so binning it would make it the biggest
    # loss in the pass — corrupting p99, max and the tail flat fraction, which are exactly the
    # numbers §4's top-tail rule is read off. A diverged run must not also rewrite the
    # diagnostic used to decide whether the *variance floor* is wrong.
    hist = histogram_over([0.1, 0.2, float("nan"), float("inf")])
    assert hist.n == 2 and hist.n_nonfinite == 2
    assert hist.largest == pytest.approx(0.2)
    assert hist.quantile(0.99) < 1.0
    assert hist.mean == pytest.approx(0.15)


def test_a_batch_of_nothing_but_non_finite_losses_still_reports_the_count():
    hist = histogram_over([float("nan")] * 4)
    assert hist.n == 0 and hist.n_nonfinite == 4
    assert np.isnan(hist.mean)


def test_an_empty_distribution_reports_nothing_rather_than_dividing_by_zero():
    hist = LossHistogram()
    assert hist.n == 0
    assert np.isnan(hist.quantile(0.5)) and np.isnan(hist.mean)
    assert np.isnan(hist.tail_flat_fraction(0.9))


def test_the_distribution_becomes_a_histogram_payload_the_sink_understands():
    payload = histogram_over([0.1, 0.5, 0.9]).payload()
    assert isinstance(payload, Histogram)
    assert sum(payload.counts) == 3
    assert len(payload.edges) == len(payload.counts) + 1


def test_the_real_val_pass_carries_a_distribution_and_not_only_a_mean(trained):
    trainer, val_loader = trained
    result = evaluate(trainer, val_loader, mask_seed=SEED)
    assert result.distribution.n == result.patches > 0
    assert result.distribution.quantile(0.9) >= result.distribution.quantile(0.5)


def test_a_perfectly_flat_tile_is_recognised_as_flat_by_the_variance_floor(cfg, trained):
    # The synthetic check §15.6 asks for: a tile with no relief at all has per-patch variance
    # under §4's floor, and the cross-tab has to say so or the rule cannot be applied.
    trainer, _ = trained
    flat = {
        "image": torch.zeros(2, 1, 224, 224),
        "land_mask": torch.ones(2, 224, 224, dtype=torch.bool),
    }
    hist = LossHistogram()
    monitor_mod.accumulate(hist, trainer, flat, var_floor=cfg.loss.norm_pix_var_floor)
    assert hist.flat_fraction == pytest.approx(1.0)


def test_the_flatness_statistic_uses_the_pixel_weights_the_loss_itself_used(trained):
    # Under §4's ocean weighting "near-flat" means flat over a patch's *valid* pixels. Derived
    # unweighted, a half-ocean patch measures its variance across the land/ocean step — the
    # largest gradient in the tile — and never looks flat, so the population §4's rule exists to
    # find would be invisible. Two derivations of one thing is how that drifts, so there is one.
    trainer, _ = trained
    batch = {
        "image": torch.zeros(1, 1, 224, 224),
        "land_mask": torch.ones(1, 224, 224, dtype=torch.bool),
    }
    batch["image"][0, 0, :, :112] = 5.0  # half a flat plateau, half flat ocean fill
    batch["land_mask"][0, :, 112:] = False
    scored = trainer.score(batch)
    assert scored.pixel_weights is not None  # the loss weighted, so the diagnostic must too

    hist = LossHistogram()
    monitor_mod.accumulate(hist, trainer, batch, var_floor=1e-4)
    # Every patch is constant over its valid pixels, so weighted variance is 0 everywhere.
    assert hist.flat_fraction == pytest.approx(1.0)


# --- §10.4: the throughput dashboard ------------------------------------------


def summary(*, samples=8192, seconds=4.0, loader_seconds=0.2, steps=4) -> ThroughputSummary:
    return ThroughputSummary(
        steps=steps, samples=samples, seconds=seconds, loader_seconds=loader_seconds, epoch=0
    )


def test_the_panel_reports_the_three_numbers_95_measures(cfg):
    logger = MemoryLogger()
    record = one_epoch(cfg, logger, throughput=summary())
    logged = logger.at(0)
    assert logged["throughput/samples_per_s"] == pytest.approx(2048.0)
    assert logged["throughput/step_time_s"] == pytest.approx(1.0)
    assert logged["throughput/loader_wait_frac"] == pytest.approx(0.05)
    assert record.throughput.steps == 4


def test_the_counters_come_from_95_rather_than_a_second_stopwatch_here(cfg):
    # The issue splits it this way on purpose. A monitor that timed the loop itself would
    # disagree with the entrypoint's numbers by whatever it forgot to include.
    handed_in = summary(seconds=10.0, loader_seconds=3.0)
    record = one_epoch(cfg, MemoryLogger(), throughput=handed_in)
    assert record.throughput is handed_in


def test_a_loader_over_the_ten_percent_threshold_alarms(cfg):
    logger = MemoryLogger()
    record = one_epoch(cfg, logger, throughput=summary(seconds=10.0, loader_seconds=3.0))
    assert [a.name for a in record.alarms] == ["loader_wait"]
    assert logger.at(0)["alarms/loader_wait"] == 1
    assert logger.at(0)["alarms/fired"] == 1


def test_a_loader_keeping_up_leaves_the_panel_quiet(cfg):
    record = one_epoch(cfg, MemoryLogger(), throughput=summary())
    assert record.alarms == ()


# --- The epoch record ---------------------------------------------------------


def one_epoch(cfg, logger, *, epoch: int = 0, throughput=None, losses=(1.0, 0.9), **kwargs):
    """Drive an :class:`EpochMonitor` through one epoch without training anything."""
    monitor = EpochMonitor(cfg, trainer=None, logger=logger, val_loader=None, **kwargs)
    for i, loss in enumerate(losses):
        monitor.on_step(FakeStep(step=i, epoch=float(epoch), lr=1e-3, loss=loss))
    return monitor.end_of_epoch(epoch, throughput=throughput)


class FakeStep:
    """A :class:`~pretrain.train.StepRecord` in the shape the monitor reads it."""

    def __init__(self, *, step, epoch, lr, loss):
        self.step, self.epoch, self.lr, self.loss = step, epoch, lr, loss
        self.micro_batches, self.grad_norm, self.throughput = 1, None, None


def test_an_epoch_reports_the_mean_of_its_steps_and_the_lr_it_ended_at(cfg):
    record = one_epoch(cfg, MemoryLogger(), losses=(1.0, 0.8, 0.6))
    assert record.train_loss == pytest.approx(0.8)
    assert record.steps == 3 and record.lr == pytest.approx(1e-3)


def test_the_epoch_scalars_reach_the_sink_under_stable_keys(cfg):
    logger = MemoryLogger()
    one_epoch(cfg, logger)
    logged = logger.at(0)
    assert logged["train/loss"] == pytest.approx(0.95)
    assert logged["train/lr"] == pytest.approx(1e-3)
    assert logged["train/steps"] == 2


def test_an_epoch_with_no_val_loader_says_so_rather_than_reporting_a_stale_number(cfg):
    # A record that carried the previous epoch's number forward would make the trend a lie.
    record = one_epoch(cfg, MemoryLogger())
    assert record.val is None and record.val_loss is None


def test_a_run_that_alarms_tells_the_callback_as_well_as_the_sink(cfg):
    seen = []
    logger = MemoryLogger()
    monitor = EpochMonitor(cfg, trainer=None, logger=logger, val_loader=None, on_alarm=seen.append)
    for epoch, loss in enumerate([1.0, 0.95, 0.9, 0.004]):
        monitor.on_step(FakeStep(step=epoch, epoch=float(epoch), lr=1e-3, loss=loss))
        record = monitor.end_of_epoch(epoch)
    assert [a.name for a in record.alarms] == ["collapse"]
    assert [a.name for a in seen] == ["collapse"]
    assert logger.at(3)["alarms/collapse"] == 1


def test_the_alarm_trajectory_is_the_epoch_series_and_not_the_step_series(cfg):
    # §10.1's conditions are written in epochs — "a plateau in the first 20 epochs", "val
    # rising while train falls". Fed the step series they would fire on within-epoch noise.
    monitor = EpochMonitor(cfg, trainer=None, logger=MemoryLogger(), val_loader=None)
    for epoch in range(3):
        for step in range(4):
            monitor.on_step(
                FakeStep(step=epoch * 4 + step, epoch=float(epoch), lr=1e-3, loss=1.0 - 0.01 * step)
            )
        monitor.end_of_epoch(epoch)
    assert [e for e, _ in monitor.train_trajectory] == [0, 1, 2]


def test_the_run_summary_records_the_best_val_and_what_fired(cfg):
    logger = MemoryLogger()
    monitor = EpochMonitor(cfg, trainer=None, logger=logger, val_loader=None)
    for epoch, loss in enumerate([1.0, 0.95, 0.9, 0.004]):
        monitor.on_step(FakeStep(step=epoch, epoch=float(epoch), lr=1e-3, loss=loss))
        monitor.end_of_epoch(epoch)
    monitor.finish()
    assert logger.summaries["train/best_loss"] == pytest.approx(0.004)
    assert logger.summaries["alarms/fired"] == 1
    assert "collapse" in logger.summaries["alarms/names"]


# --- The whole thing, against a run that actually trains -----------------------


def test_a_short_real_run_reports_a_falling_loss_and_a_val_number_every_epoch(cfg):
    logger = MemoryLogger()
    records = train_epochs(cfg, epochs=6, device="cpu", logger=logger)
    assert [r.epoch for r in records] == [0, 1, 2, 3, 4, 5]
    assert all(r.val is not None for r in records)
    assert records[-1].train_loss < records[0].train_loss
    # Every epoch's panel reached the sink, distribution included.
    for epoch in range(6):
        logged = logger.at(epoch)
        assert {"train/loss", "val/loss", "val/per_patch_loss", "throughput/samples_per_s"} <= set(
            logged
        )
        assert isinstance(logged["val/per_patch_loss"], Histogram)


def test_the_val_loss_of_a_real_run_reproduces_when_the_run_does(cfg):
    # §10.1's property, end to end: same seed, same config, same val numbers epoch for epoch.
    first = [r.val.loss for r in train_epochs(cfg, epochs=3, device="cpu")]
    second = [r.val.loss for r in train_epochs(cfg, epochs=3, device="cpu")]
    assert first == second


def test_every_epoch_of_a_real_run_gets_a_val_pass_with_no_cadence_to_skip_it(cfg):
    # §10.1 says "train loss + val loss every epoch" with no qualification, and there is
    # deliberately no cadence knob: skipping the pass would take §4's per-patch distribution
    # and the val-side collapse alarm off the skipped epochs with it.
    records = train_epochs(cfg, epochs=4, device="cpu")
    assert [r.val is not None for r in records] == [True] * 4
    assert not hasattr(cfg.monitoring, "val_every_epochs")


def test_a_real_run_reports_its_throughput_from_the_loop_that_ran_it(cfg):
    records = train_epochs(cfg, epochs=2, device="cpu")
    for record in records:
        assert record.throughput is not None
        # Two train COGs of four tiles each, every one seen once per epoch.
        assert record.throughput.samples == 2 * TILES_PER_COG
        assert record.throughput.seconds > 0


def test_the_whole_epoch_loop_runs_through_a_real_offline_wandb_run(cfg, tmp_path):
    # The acceptance criterion the mocks cannot cover: a real short training run, a real W&B
    # offline run, no account and no network, with the grids' and probes' payload kind included.
    offline = cfg.replace(
        monitoring=cfg.monitoring.replace(
            wandb_mode="offline", wandb_project="demae-test", wandb_dir=str(tmp_path / "wb")
        )
    )
    records = train_epochs(offline, epochs=2, device="cpu")
    assert len(records) == 2
    runs = list((tmp_path / "wb" / "wandb").glob("offline-run-*"))
    assert len(runs) == 1
    body = next(runs[0].glob("*.wandb")).read_bytes()
    for key in (b"train/loss", b"val/loss", b"val/per_patch_loss", b"throughput/loader_wait_frac"):
        assert key in body, key


# --- The entrypoint -----------------------------------------------------------


def test_the_entrypoint_runs_n_epochs_and_prints_the_panel(cfg, tmp_path, capsys):
    config_file = tmp_path / "run.json"
    config_file.write_text(json.dumps(cfg.to_dict()))
    assert monitor_mod.main(["--config", str(config_file), "--epochs", "2"]) == 0
    out = capsys.readouterr().out
    assert "DEMAE Phase 1" in out  # #89's header, before the first epoch
    lines = [line for line in out.splitlines() if line.startswith("epoch ")]
    assert len(lines) == 2
    assert "val" in lines[0] and "samples/s" in lines[0] and "loader" in lines[0]


def test_the_entrypoint_prints_an_alarm_where_a_reader_will_see_it(cfg, tmp_path, capsys):
    # An alarm that only exists in W&B is an alarm nobody sees during the run that fires it.
    loud = cfg.replace(monitoring=cfg.monitoring.replace(loader_wait_alarm=1e-6))
    config_file = tmp_path / "run.json"
    config_file.write_text(json.dumps(loud.to_dict()))
    assert monitor_mod.main(["--config", str(config_file), "--epochs", "1"]) == 0
    out = capsys.readouterr().out
    assert "[STOP]" in out and "loader_wait" in out
