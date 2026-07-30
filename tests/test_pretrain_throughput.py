"""The throughput counters: samples/s, step time, and the loader wait fraction (#95, §7.5).

The loader is the plan's **#1 named risk** — DailyMAE got 5.8x end-to-end from IO alone — and
§7.5's gate is a number, not a feeling: measured samples/s at least 2x model demand, and a
wait fraction under ~10%. This module is where that number comes from; #97 builds the
dashboard and the alarm on top of it.

Everything here is timed against an **injected clock**, so the tests assert exact seconds
rather than "it went up". A counter tested against the wall is a counter that can be off by a
factor and still pass.

The one property worth stating out loud, because it is what makes the wait fraction mean
anything: **the wait is the time spent blocked inside the loader's ``next``**, not the time
the loader spent working. A prefetching DataLoader with workers does most of its work during
the previous step's backward pass; what is left over — the residual the prefetch did not
cover — is the only part that costs wall clock, and it is the only part counted.
"""

from __future__ import annotations

import pytest

from pretrain import throughput as tp


class FakeClock:
    """A clock the test drives: nothing moves except when something is told to take time."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class SlowLoader:
    """A loader whose every batch costs ``wait`` seconds to produce. Re-iterable, like the real one."""

    def __init__(self, clock: FakeClock, batches: list, *, wait: float) -> None:
        self.clock, self.batches, self.wait = clock, batches, wait
        self.passes = 0

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        self.passes += 1
        for batch in self.batches:
            self.clock.advance(self.wait)
            yield batch


def batches(n: int, *, size: int = 4) -> list:
    return [{"image": [0] * size} for _ in range(n)]


# --- One step -----------------------------------------------------------------


def test_a_step_reports_its_wall_time_its_samples_and_what_it_waited_for():
    clock = FakeClock()
    meter = tp.ThroughputMeter(SlowLoader(clock, batches(2), wait=0.1), clock=clock)
    pulled = [batch for batch in meter][:2]
    clock.advance(0.8)  # the forward/backward/step, after both micro-batches arrived
    record = meter.mark(step=0, epoch=0, samples=8)

    assert len(pulled) == 2
    assert record.seconds == pytest.approx(1.0)  # 0.1 + 0.1 waiting, 0.8 computing
    assert record.loader_seconds == pytest.approx(0.2)
    assert record.samples_per_second == pytest.approx(8.0)
    assert record.loader_wait_fraction == pytest.approx(0.2)
    assert (record.step, record.epoch, record.samples) == (0, 0, 8)


def test_each_step_is_timed_from_the_end_of_the_last_one():
    # Not from the start of the run: the intervals have to partition the wall clock, or the
    # per-epoch samples/s below would be an average of overlapping windows.
    clock = FakeClock()
    loader = SlowLoader(clock, batches(4), wait=0.25)
    meter = tp.ThroughputMeter(loader, clock=clock)
    it = iter(meter)
    for step in range(2):
        next(it), next(it)
        clock.advance(0.5)
        record = meter.mark(step=step, epoch=0, samples=8)
        assert record.seconds == pytest.approx(1.0), step
        assert record.loader_seconds == pytest.approx(0.5), step
    assert clock.t == pytest.approx(2.0)


def test_only_the_time_inside_the_loader_counts_as_waiting():
    # The distinction the §7.5 gate rests on. A step that spends 90% of its wall clock in the
    # backward pass is healthy; one that spends 90% waiting is IO-bound, and both have the
    # same step time.
    clock = FakeClock()
    meter = tp.ThroughputMeter(SlowLoader(clock, batches(1), wait=0.05), clock=clock)
    next(iter(meter))
    clock.advance(4.95)
    record = meter.mark(step=0, epoch=0, samples=4)
    assert record.loader_wait_fraction == pytest.approx(0.01)


def test_a_step_that_took_no_measurable_time_reports_no_wait_rather_than_dividing_by_zero():
    # A coarse clock is a plausible box, and a ZeroDivisionError out of a *counter* would take
    # down a run that was training perfectly well.
    clock = FakeClock()
    meter = tp.ThroughputMeter(SlowLoader(clock, batches(1), wait=0.0), clock=clock)
    next(iter(meter))
    record = meter.mark(step=0, epoch=0, samples=4)
    assert record.seconds == 0.0
    assert record.loader_wait_fraction == 0.0
    assert record.samples_per_second == 0.0


# --- The loader underneath ----------------------------------------------------


def test_the_meter_hands_the_loaders_batches_through_untouched():
    clock = FakeClock()
    original = batches(3)
    loader = SlowLoader(clock, original, wait=0.0)
    meter = tp.ThroughputMeter(loader, clock=clock)
    assert list(meter) == original
    assert len(meter) == 3  # `micro_batch_groups` and `steps_per_epoch_for` both ask


def test_re_iterating_the_meter_re_iterates_the_loader_so_epochs_still_reshuffle():
    # #93's loop re-iterates the loader once per epoch, and that is what re-draws the shuffle.
    # A meter that cached one pass would train 400 epochs of one ordering.
    clock = FakeClock()
    loader = SlowLoader(clock, batches(2), wait=0.0)
    meter = tp.ThroughputMeter(loader, clock=clock)
    list(meter), list(meter)
    assert loader.passes == 2


def test_the_wait_accumulates_across_an_epoch_boundary():
    clock = FakeClock()
    meter = tp.ThroughputMeter(SlowLoader(clock, batches(2), wait=0.5), clock=clock)
    list(meter)
    list(meter)
    assert meter.waited == pytest.approx(2.0)


# --- Per epoch ----------------------------------------------------------------


def steps_of(*specs) -> list:
    """``(epoch, samples, seconds, loader_seconds)`` tuples as :class:`StepThroughput`."""
    return [
        tp.StepThroughput(step=i, epoch=e, samples=n, seconds=s, loader_seconds=w)
        for i, (e, n, s, w) in enumerate(specs)
    ]


def test_per_epoch_sums_the_steps_of_each_epoch_and_keeps_them_in_order():
    summaries = tp.per_epoch(steps_of((0, 8, 1.0, 0.1), (0, 8, 3.0, 0.3), (1, 8, 2.0, 0.6)))
    assert [s.epoch for s in summaries] == [0, 1]
    first, second = summaries
    assert (first.steps, first.samples, first.seconds) == (2, 16, 4.0)
    assert first.loader_seconds == pytest.approx(0.4)
    assert first.samples_per_second == pytest.approx(4.0)
    assert first.loader_wait_fraction == pytest.approx(0.1)
    assert (second.steps, second.samples) == (1, 8)


def test_an_epochs_throughput_is_its_totals_and_not_the_mean_of_its_steps():
    # The difference is the whole point on a box where step 1 pays for the compile and the
    # loader's first fill: 2 samples/s and 100 samples/s average to 51, but two steps that
    # produced 8 samples in 4.08 s did 1.96 samples/s, which is what an hour of rent buys.
    slow, fast = (0, 4, 4.0, 3.9), (0, 4, 0.08, 0.0)
    only = tp.per_epoch(steps_of(slow, fast))[0]
    assert only.samples_per_second == pytest.approx(8 / 4.08)
    assert only.loader_wait_fraction == pytest.approx(3.9 / 4.08)


def test_a_whole_run_summarizes_the_same_way_an_epoch_does():
    run = tp.summarize(steps_of((0, 8, 1.0, 0.5), (1, 8, 1.0, 0.5)))
    assert (run.steps, run.samples, run.seconds) == (2, 16, 2.0)
    assert run.epoch is None  # a run is not an epoch, and its label should not claim to be
    assert run.samples_per_second == pytest.approx(8.0)


def test_a_summary_accumulates_one_step_at_a_time():
    # What a long run needs: R1 is ~195,000 optimizer steps, and a footer computed by
    # re-summing every step it has ever taken is quadratic in the run's own length.
    running = tp.summarize([])
    for step in steps_of((0, 8, 1.0, 0.5), (1, 8, 3.0, 0.5)):
        running = running.plus(step)
    assert running == tp.summarize(steps_of((0, 8, 1.0, 0.5), (1, 8, 3.0, 0.5)))


def test_accumulating_keeps_whatever_epoch_the_summary_was_labelled_with():
    epoch = tp.summarize([], epoch=3).plus(
        tp.StepThroughput(step=0, epoch=3, samples=8, seconds=1.0, loader_seconds=0.1)
    )
    assert epoch.epoch == 3 and epoch.steps == 1


def test_summarizing_nothing_is_an_empty_summary_rather_than_an_error():
    # A run killed before its first step still prints a footer.
    empty = tp.summarize([])
    assert (empty.steps, empty.samples, empty.seconds) == (0, 0, 0.0)
    assert empty.samples_per_second == 0.0
    assert tp.per_epoch([]) == []


# --- The line a human reads ---------------------------------------------------


def test_the_formatted_counters_carry_all_three_numbers():
    line = tp.format_throughput(
        tp.StepThroughput(step=0, epoch=0, samples=2048, seconds=0.5, loader_seconds=0.05)
    )
    assert "4096" in line  # samples/s
    assert "0.500" in line or "500" in line  # step time
    assert "10" in line  # the wait fraction, as a percentage
