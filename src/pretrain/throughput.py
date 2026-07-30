"""What the loop is actually achieving per second, and how much of it is waiting (#95, §7.5).

Three numbers, per step and per epoch: **samples/s**, **step time**, and the **loader wait
fraction**. This module produces them; [#97](https://github.com/JStoschek/DEMAE/issues/97)
builds the dashboard and the ~10% wait alarm on top of them.

They are not decoration. §7.5 names the dataloader as the plan's *#1 risk* — DailyMAE got
**5.8x end to end** by fixing IO alone — and the §15 pre-flight gate is a measurement:
*measured samples/s at least 2x model demand, in the first rented hour*. That gate needs a
loader-wait number, and a loader-wait number needs someone to have written down what "waiting"
means.

## What counts as waiting

**The time spent blocked inside the loader's ``next``** — and nothing else. Not the time the
loader spent working: with ``num_workers=8`` and prefetching, most of that happens *during*
the previous step's backward pass and costs zero wall clock. What is left is the residual the
prefetch failed to cover, which is exactly the part that makes a step slower, and exactly the
part §7.5's threshold is about.

This is why the meter wraps the loader rather than sitting beside it: the interval being timed
is a ``next()`` call, and only the thing holding the iterator can see it.

## Sums, never averages

:func:`per_epoch` adds the seconds and adds the samples, then divides once. A mean of
per-step rates would say something quite different on the shape every real run has — one step
that pays for the compile and the loader's first fill, then thousands of fast ones — and it
would be the flattering version. See
`test_an_epochs_throughput_is_its_totals_and_not_the_mean_of_its_steps`.

## The first step is not the steady state, and this module does not hide that

Step 0 carries `torch.compile`'s whole compilation (§7.3), the worker pool's startup and the
page cache's first fill. It is included, honestly, in everything here — a summary that quietly
dropped it would be a summary of a run nobody had. Callers that want the steady state say so
by summarizing a slice: #95's entrypoint prints both, labelled.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass


def _rate(quantity: float, seconds: float) -> float:
    """``quantity / seconds``, or 0.0 when no time was measured.

    A clock coarser than a step is a plausible box (and a fake clock in a test certainly is),
    and a ``ZeroDivisionError`` raised by a *counter* would kill a run that was training
    perfectly well. Zero is the honest reading of "too fast for this clock to see".
    """
    return quantity / seconds if seconds > 0 else 0.0


class _Rates:
    """The two derived numbers a step and a run share, defined once.

    Both of the types below hold the same four quantities — samples, seconds, loader seconds,
    and a label — so both answer the same two questions, and §7.5's gate is written in the
    answers. Two copies of ``samples / seconds`` is two places for the zero-second guard to be
    got wrong, and one of them would be the one a rented box reads.
    """

    @property
    def samples_per_second(self) -> float:
        """The headline number §7.5's gate is written in."""
        return _rate(self.samples, self.seconds)

    @property
    def loader_wait_fraction(self) -> float:
        """0.0-1.0. §7.5 stops and fixes the loader at ~0.10."""
        return _rate(self.loader_seconds, self.seconds)


@dataclass(frozen=True)
class StepThroughput(_Rates):
    """What one optimizer step cost. Carried on :class:`~pretrain.train.StepRecord`."""

    #: The optimizer step this timed — :attr:`StepRecord.step <pretrain.train.StepRecord>`.
    step: int
    #: The whole epoch this step sat in, floored. The key :func:`per_epoch` groups on, held
    #: here so the counters aggregate without needing the record they came from.
    epoch: int
    #: Samples in the effective batch — every micro-batch's rows, summed.
    samples: int
    #: **Step time**: wall clock from the end of the previous step to the end of this one, so
    #: consecutive steps partition the run's wall clock rather than overlapping.
    seconds: float
    #: Of which, blocked in the loader. See the module docstring for what that does and does
    #: not include.
    loader_seconds: float


@dataclass(frozen=True)
class ThroughputSummary(_Rates):
    """Steps added up: one epoch's, or a whole run's (:attr:`epoch` ``None``).

    One type for both because they are the same arithmetic, and because a second type would be
    a second place for the sums-not-averages rule to be got wrong.
    """

    steps: int
    samples: int
    seconds: float
    loader_seconds: float
    #: The epoch summarized, or ``None`` for a summary that spans epochs. A run is not an
    #: epoch and its label should not claim to be one.
    epoch: int | None = None

    def plus(self, step: StepThroughput) -> ThroughputSummary:
        """This summary with one more step folded in, keeping its :attr:`epoch` label.

        How a long run keeps a footer without keeping every step. R1 is ~195,000 optimizer
        steps; a total recomputed from the whole history on each of them is quadratic in the
        run's own length, and the memory to hold that history is the same mistake spelled
        differently. A caller that already has the list can still use :func:`summarize`.
        """
        return ThroughputSummary(
            steps=self.steps + 1,
            samples=self.samples + step.samples,
            seconds=self.seconds + step.seconds,
            loader_seconds=self.loader_seconds + step.loader_seconds,
            epoch=self.epoch,
        )


def summarize(steps: Iterable[StepThroughput], *, epoch: int | None = None) -> ThroughputSummary:
    """Add up whatever steps are handed over — a whole run, or a slice past the warmup.

    Folded through :meth:`ThroughputSummary.plus` rather than summed four times here, so the
    two ways a caller can aggregate — all at once, or one step at a time as #95's entrypoint
    does — are the same arithmetic and cannot drift apart.
    """
    total = ThroughputSummary(steps=0, samples=0, seconds=0.0, loader_seconds=0.0, epoch=epoch)
    for step in steps:
        total = total.plus(step)
    return total


def per_epoch(steps: Iterable[StepThroughput]) -> list[ThroughputSummary]:
    """One summary per epoch, in the order the epochs happened.

    Grouped on :attr:`StepThroughput.epoch`, which the loop fills from the schedule's
    fractional epoch — so an epoch here is the schedule's epoch, the same unit §10's per-epoch
    losses are reported in, and the two panels line up row for row.
    """
    grouped: dict[int, list[StepThroughput]] = {}
    for step in steps:
        grouped.setdefault(step.epoch, []).append(step)
    return [summarize(group, epoch=epoch) for epoch, group in sorted(grouped.items())]


class ThroughputMeter:
    """Wraps the loader, times the loop, and hands back one :class:`StepThroughput` per step.

    Used in place of the loader itself::

        meter = ThroughputMeter(loader)
        for group in micro_batch_groups(meter, accum_iter, steps=steps):
            record = trainer.step(group)
            counters = meter.mark(step=record.step, epoch=int(record.epoch), samples=...)

    Two jobs in one object, and they are inseparable: the wait can only be measured by
    whatever calls ``next``, and it is meaningless except as a fraction of the interval the
    same object is timing. Everything it wraps stays visible — ``len`` and the batches pass
    through untouched, and re-iterating re-iterates the loader, which is what makes #93's
    per-epoch reshuffle still happen.

    The clock starts at construction, so the first step includes whatever the caller did
    between building the meter and asking for a batch. Build it immediately before the loop.

    It **keeps no history**. Each :class:`StepThroughput` is handed to the caller and forgotten,
    because R1 is ~195,000 steps and a meter that accrued them would be a slow memory leak
    inside the thing measuring the run's health. :func:`per_epoch` and
    :meth:`ThroughputSummary.plus` are how a caller aggregates whatever it chose to keep.
    """

    def __init__(self, loader, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self.loader = loader
        self.clock = clock
        #: Seconds spent blocked in ``next`` since construction, across every epoch.
        self.waited = 0.0
        self._interval_started = clock()
        self._interval_waited = 0.0

    def __len__(self) -> int:
        """The loader's length: ``micro_batch_groups`` and ``steps_per_epoch_for`` both ask."""
        return len(self.loader)

    def __iter__(self) -> Iterator:
        """One pass over the loader, charging the time each batch took to arrive.

        The ``StopIteration`` path is timed too. At ``num_workers > 0`` the end of an epoch is
        where the workers are drained and restarted, and a loader that stalls there stalls the
        run — leaving it out would put that cost in "compute" and hide it.
        """
        iterator = iter(self.loader)
        while True:
            started = self.clock()
            try:
                batch = next(iterator)
            except StopIteration:
                self.waited += self.clock() - started
                return
            self.waited += self.clock() - started
            yield batch

    def mark(self, *, step: int, epoch: int, samples: int) -> StepThroughput:
        """Close the interval a step occupied and open the next one.

        Called *after* the optimizer step, so the interval covers the batches that step pulled
        plus the step itself — everything between the end of the previous step and now.
        """
        now = self.clock()
        record = StepThroughput(
            step=step,
            epoch=epoch,
            samples=samples,
            seconds=now - self._interval_started,
            loader_seconds=self.waited - self._interval_waited,
        )
        self._interval_started, self._interval_waited = now, self.waited
        return record


def format_throughput(counters: StepThroughput | ThroughputSummary) -> str:
    """The three numbers as one line fragment — what a step line and an epoch line share.

    Deliberately identical for a step and a summary: an epoch's throughput and a step's are
    the same three quantities, and printing them in two shapes is how a reader ends up
    comparing a rate against a mean of rates.
    """
    return (
        f"{counters.samples_per_second:8.1f} samples/s  "
        f"{counters.seconds:6.3f}s  "
        f"loader {counters.loader_wait_fraction:5.1%}"
    )
