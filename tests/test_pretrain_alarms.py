"""§10.1's alarms, on trajectories built to trigger them — and on one built not to (#97).

The ticket's framing is the whole test strategy: *"alarms that fire, rather than conditions
written in a doc"*. So every test here is a synthetic loss trajectory and an assertion about
what came back, and the most important one is :func:`healthy` — an alarm that fires on a good
run is worse than no alarm, because it is the one people learn to ignore.

The trajectories are shaped like the plan expects a real run to look (§10.1: *"fast warmup drop
then slow decay for hundreds of epochs"*), not like a straight line, so "quiet on a healthy
run" is a claim about the curve MAE actually produces.
"""

from __future__ import annotations

import math

import pytest

from pretrain.alarms import STOP, WATCH, Alarm, check_alarms
from pretrain.config import Monitoring

M = Monitoring()


def series(values, start: int = 0) -> list[tuple[int, float]]:
    """``[(epoch, loss), ...]`` from a bare list of losses."""
    return [(start + i, float(v)) for i, v in enumerate(values)]


def healthy(n: int = 60, floor: float = 0.35, tau: float = 8.0) -> list[float]:
    """A believable MAE curve: 1.0 at epoch 0, fast warmup drop, then a long slow decay."""
    return [floor + (1.0 - floor) * math.exp(-e / tau) for e in range(n)]


def names(alarms: list[Alarm]) -> set[str]:
    return {a.name for a in alarms}


# --- Quiet on a healthy run ---------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 20, 21, 60, 400])
def test_nothing_fires_on_a_healthy_run_at_any_length(n):
    train = healthy(n)
    val = [t * 1.03 for t in train]  # val sits a little above train and tracks it down
    assert check_alarms(M, train=series(train), val=series(val), loader_wait_fraction=0.04) == []


def test_nothing_fires_before_there_is_a_trajectory_to_judge():
    # Epoch 0 of every run: one point, no history. Firing here would alarm every run at start.
    assert check_alarms(M, train=series([1.0]), val=series([1.0])) == []
    assert check_alarms(M, train=[], val=[]) == []


def test_a_converged_run_whose_val_has_flattened_is_not_overfitting():
    # The false positive that would teach people to ignore the overfitting alarm: val at its
    # own minimum, train still creeping down. `overfit_min_rise` is what separates the two.
    train = [0.30 - 0.0005 * e for e in range(40)]
    val = [0.40 + 0.0001 * (e % 3) for e in range(40)]  # flat, jittering under the 2% rise
    assert "overfitting" not in names(check_alarms(M, train=series(train), val=series(val)))


# --- The floor underneath the three: a loss that is not a number --------------


def test_a_nan_loss_fires_rather_than_silencing_every_other_alarm():
    # The hole this closes: every threshold below compares *false* against a NaN, so without a
    # check for it the harness is quietest at the exact moment the run dies. §13 lists bf16
    # loss spikes as a real risk, so this is not a hypothetical shape.
    alarms = check_alarms(M, train=series([1.0, 0.9, float("nan")]), val=[])
    (alarm,) = [a for a in alarms if a.name == "diverged"]
    assert alarm.action == STOP and alarm.epoch == 2
    assert "bf16" in alarm.message


def test_an_infinite_loss_fires_the_same_alarm():
    assert "diverged" in names(check_alarms(M, train=series([1.0, float("inf")]), val=[]))


def test_a_diverged_run_reports_one_alarm_and_not_three_silent_passes():
    # Reporting "diverged" beside three conditions that returned None would read as three
    # checks made and cleared, when in fact none of them could have fired.
    alarms = check_alarms(M, train=series([1.0] * 25 + [float("nan")]), val=series([1.0] * 26))
    assert [a.name for a in alarms] == ["diverged"]


def test_a_diverged_run_on_a_starved_loader_still_reports_the_loader():
    # The loader wait is not a loss condition, so a NaN does not suppress it: that run has two
    # problems and hiding one of them behind the other helps nobody.
    alarms = check_alarms(
        M, train=series([1.0, float("nan")]), val=[], loader_wait_fraction=0.4
    )
    assert names(alarms) == {"diverged", "loader_wait"}


def test_a_healthy_run_never_looks_diverged():
    assert "diverged" not in names(check_alarms(M, train=series(healthy(60)), val=[]))


# --- §10.1 alarm 1: sudden collapse toward zero -------------------------------


def test_a_loss_that_falls_to_nothing_fires_the_collapse_alarm():
    # The broken-`norm_pix`-target symptom: the target measures as identically flat, so
    # predicting zero is exactly right and the loss goes to zero while nothing is learned.
    alarms = check_alarms(M, train=series([1.0, 0.92, 0.85, 0.004]), val=[])
    (alarm,) = [a for a in alarms if a.name == "collapse"]
    assert alarm.action == STOP and alarm.epoch == 3
    assert "norm_pix" in alarm.message


def test_a_single_epoch_drop_by_a_large_factor_fires_it_even_above_the_floor():
    # The *sudden* half. 0.08 is still 8% of where the run started — above `collapse_frac` —
    # but a 10x fall between two consecutive epochs is the target moving, not a model learning.
    alarms = check_alarms(M, train=series([1.0, 0.95, 0.90, 0.08]), val=[])
    assert "collapse" in names(alarms)


def test_a_val_collapse_fires_even_when_train_looks_ordinary():
    # Both series are scored against the same target, so either one collapsing is the symptom.
    alarms = check_alarms(M, train=series(healthy(6)), val=series([1.0, 0.9, 0.8, 0.7, 0.6, 0.002]))
    (alarm,) = [a for a in alarms if a.name == "collapse"]
    assert "val" in alarm.message


def test_the_warmup_drop_of_a_real_run_is_not_a_collapse():
    # The guard on the two above: MAE's own curve falls fast in warmup, and an alarm that
    # cannot tell that from a broken target fires on every run in its first ten epochs.
    assert "collapse" not in names(check_alarms(M, train=series(healthy(30)), val=[]))


# --- §10.1 alarm 2: val rising while train falls ------------------------------


def test_val_rising_while_train_falls_fires_the_overfitting_alarm():
    # Plausible at 500k non-independent tiles — corpus card §4.6 on effective dataset size.
    train = [1.0 - 0.02 * e for e in range(30)]
    val = [1.0 - 0.02 * e for e in range(10)] + [0.80 + 0.01 * e for e in range(20)]
    alarms = check_alarms(M, train=series(train), val=series(val))
    (alarm,) = [a for a in alarms if a.name == "overfitting"]
    assert alarm.action == WATCH  # §13's mitigation is "stop the extension", not "stop now"
    assert "epoch 10" in alarm.message  # where val was last at its best


def test_both_losses_rising_is_instability_and_not_overfitting():
    # Overfitting is a *divergence* between the two. If train is rising too, the diagnosis is
    # something else entirely (§13's bf16 loss spikes), and calling it overfitting misleads.
    rising = [0.5 + 0.01 * e for e in range(30)]
    assert "overfitting" not in names(check_alarms(M, train=series(rising), val=series(rising)))


def test_the_patience_is_counted_in_epochs_and_not_in_val_points():
    # The threshold is documented in epochs ("2.5% of the 400-epoch schedule"), so it has to
    # keep meaning that however often a run happens to score val. Same trajectory, val recorded
    # on every other epoch: the alarm must still wait ten *epochs*, not ten measurements.
    train = series([1.0 - 0.02 * e for e in range(40)])
    val = [(e, 1.0 - 0.02 * e) for e in range(0, 10, 2)] + [
        (e, 0.80 + 0.01 * e) for e in range(10, 40, 2)
    ]
    best_epoch = min(val, key=lambda p: p[1])[0]
    assert best_epoch == 8

    # Four val points past the best, but only eight epochs — under the ten-epoch patience.
    short = [p for p in val if p[0] <= best_epoch + 8]
    assert len([p for p in short if p[0] > best_epoch]) == 4
    assert "overfitting" not in names(check_alarms(M, train=train, val=short))

    # One more measurement, and now ten epochs have passed: it fires.
    long = [p for p in val if p[0] <= best_epoch + 10]
    assert "overfitting" in names(check_alarms(M, train=train, val=long))


def test_the_alarm_waits_out_its_patience_before_firing():
    # Ten epochs of a worse val is 2.5% of the 400-epoch schedule. Firing on the first bad val
    # pass would fire on noise from a ~10k-tile split.
    train = [1.0 - 0.02 * e for e in range(30)]
    val = [1.0 - 0.02 * e for e in range(10)] + [0.80 + 0.01 * e for e in range(20)]
    short = M.overfit_patience  # the rise starts at index 10
    assert "overfitting" not in names(
        check_alarms(M, train=series(train[: 10 + short - 1]), val=series(val[: 10 + short - 1]))
    )
    assert "overfitting" in names(
        check_alarms(M, train=series(train[: 11 + short]), val=series(val[: 11 + short]))
    )


# --- §10.1 alarm 3: a plateau at high loss in the first 20 epochs -------------


def test_a_run_still_at_its_starting_loss_at_epoch_20_fires_the_plateau_alarm():
    # The LR-or-normalization-bug signal: §6's warmup reaches its peak LR at epoch 20, so a
    # model that is going to learn has taken a visible bite out of the loss by then.
    stuck = [1.0 - 0.0005 * e for e in range(25)]
    alarms = check_alarms(M, train=series(stuck), val=[])
    (alarm,) = [a for a in alarms if a.name == "plateau"]
    assert alarm.action == STOP and alarm.epoch == 24
    assert "learning rate" in alarm.message or "normalization" in alarm.message


def test_the_plateau_alarm_says_nothing_before_the_warmup_has_finished():
    # At epoch 10 a high loss is just a run that has not started; §6's LR is still ramping.
    stuck = [1.0 - 0.0005 * e for e in range(M.plateau_by_epoch)]
    assert "plateau" not in names(check_alarms(M, train=series(stuck), val=[]))


def test_a_run_that_is_high_but_still_moving_is_not_a_plateau():
    # Both conditions have to hold. A run at 95% of its starting loss that is still improving
    # 2% every five epochs is slow, not stuck, and "slow" is not an alarm.
    moving = [1.0 * 0.996**e for e in range(25)]
    assert moving[20] > M.plateau_frac * moving[0]
    assert "plateau" not in names(check_alarms(M, train=series(moving), val=[]))


# --- §10.4 / §7.5: the loader wait threshold ----------------------------------


def test_a_loader_wait_over_ten_percent_fires_and_says_to_stop():
    (alarm,) = check_alarms(M, train=series(healthy(5)), val=[], loader_wait_fraction=0.14)
    assert alarm.name == "loader_wait" and alarm.action == STOP
    assert "14" in alarm.message and "10" in alarm.message


def test_a_loader_keeping_up_is_quiet():
    assert check_alarms(M, train=series(healthy(5)), val=[], loader_wait_fraction=0.09) == []
    assert check_alarms(M, train=series(healthy(5)), val=[], loader_wait_fraction=None) == []


def test_the_threshold_is_a_config_knob_and_moving_it_moves_the_alarm():
    strict = M.replace(loader_wait_alarm=0.05)
    assert check_alarms(strict, train=series(healthy(5)), val=[], loader_wait_fraction=0.06)
    assert not check_alarms(M, train=series(healthy(5)), val=[], loader_wait_fraction=0.06)


# --- What an alarm is ---------------------------------------------------------


def test_every_alarm_names_the_epoch_the_action_and_why():
    train = [1.0, 0.9, 0.001]
    alarms = check_alarms(M, train=series(train), val=[], loader_wait_fraction=0.5)
    assert len(alarms) == 2
    for alarm in alarms:
        assert alarm.action in (STOP, WATCH)
        assert alarm.epoch >= 0
        assert len(alarm.message) > 40  # a diagnosis, not a name repeated
        assert str(alarm)  # printable, because a run prints these to stdout as well as W&B


def test_alarms_come_back_in_a_stable_order_so_a_log_line_is_diffable():
    train = [1.0, 0.9, 0.001]
    twice = [
        check_alarms(M, train=series(train), val=[], loader_wait_fraction=0.5) for _ in range(2)
    ]
    assert [a.name for a in twice[0]] == [a.name for a in twice[1]]
