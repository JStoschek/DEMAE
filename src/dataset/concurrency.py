"""The bounded, order-preserving fan-out primitive shared by the parallel reads.

One scheduling helper, written against :class:`~concurrent.futures.Executor` rather than
a specific pool, so the same bounded/ordered scheduling serves both a *thread* pool (the
I/O-bound COG reads) and a *process* pool (:func:`dataset.catalog.scan`'s CPU-bound
scoring) without a second implementation (#38). It lived in the retired ADR-0001
pipeline; it outlives that pipeline because it is pure scheduling, owing nothing to how a
corpus is stratified.
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import Executor, Future
from typing import Callable, Final, Iterable, Iterator, Literal, TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


def ordered_futures(
    ex: Executor,
    fn: Callable[[_T], _R],
    items: Iterable[_T],
    in_flight: int,
) -> Iterator[tuple[_T, "Future[_R]"]]:
    """Yield ``(item, future)`` in input order, keeping ``in_flight`` submitted.

    Two properties it has to hold at once:

    - **Input order.** A caller consuming these in order gets a result sequence
      independent of which task finished first, so a parallel run is identical to the
      ``workers=1`` path — only the wall-clock changes.
    - **Bounded memory.** ``Executor.map`` submits every item up front and buffers each
      finished result until the one ahead of it is consumed; when each result is a whole
      COG's worth of rows or arrays, one slow task could pin the rest in memory behind it.
      A sliding window keeps the pool fed while bounding the queue and everything held in
      it.

    Handing back the future rather than its value is what lets a caller decide per item
    whether a failed task is fatal (re-raise) or something to record and step over
    (:func:`dataset.catalog.scan`).
    """
    it = iter(items)
    pending: deque[tuple[_T, "Future[_R]"]] = deque()
    for item in it:
        pending.append((item, ex.submit(fn, item)))
        if len(pending) >= in_flight:
            break
    while pending:
        yield pending.popleft()
        nxt = next(it, None)
        if nxt is not None:
            pending.append((nxt, ex.submit(fn, nxt)))


# --- Which pool, and how wide (#38, and re-measured for the read pass by #49) ------

#: How a pass fans its per-COG work out. Both live-read stages take it as an argument:
#: ``"process"`` is what each one actually runs on, ``"thread"`` is the in-process path
#: a caller with an injected, unpicklable reader needs — see :func:`make_executor`.
ExecutorKind = Literal["thread", "process"]

#: Mild oversubscription: a worker blocked on an S3 read leaves its core idle, and
#: another worker's compute fills it. Measured, not assumed — see :func:`default_workers`.
WORKERS_PER_CORE: Final = 2
#: The top of the measured curve on the dev box, and where its memory stops being
#: free: a worker peaks at 337 MB (measured RSS), so 16 is ~5.4 GB resident. A
#: *default*, not a limit — an explicit ``workers=`` is taken at its word, and a
#: bigger or same-region box should re-measure before raising it.
MAX_DEFAULT_WORKERS: Final = 16
#: Where thread scaling flattens: measured 2.56 → 2.27 → 2.37 s/COG at 4, 8 and 16
#: threads, so 8 is the top and 16 is already the far side of it (issue #38).
THREAD_GIL_KNEE: Final = 8


def _check_executor(kind: str) -> ExecutorKind:
    """Reject an unknown executor at the first gate rather than the last.

    Both :func:`default_workers` and :func:`make_executor` branch on the kind, and
    :func:`scan` calls them in that order — so without this the bad value would get a
    plausible worker count before anything complained.
    """
    if kind not in ("thread", "process"):
        raise ValueError(f"unknown executor {kind!r}; expected 'thread' or 'process'")
    return kind  # type: ignore[return-value]


def default_workers(kind: ExecutorKind = "process") -> int:
    """How many COGs to score in parallel — measured on this box, not assumed (#38).

    C1's projection divided a *serial* per-COG cost by 16 workers, which assumed
    linear thread scaling. Scoring is dominated by numpy neighbourhood compute under
    the GIL, so the two executors want very different counts, and neither wants 16.
    One sweep, 24 spread COGs per config on the dev box (8 logical cores), against a
    measured serial baseline of 5.51 s/COG (``docs/research/c3-throughput.json``):

    - **thread** — 2.56 @4, 2.27 @8, 2.37 @16. Best case **2.4x**, reached at 8 and
      given back by 16. Extra threads past :data:`THREAD_GIL_KNEE` buy contention.
    - **process** — 1.89 @4, 1.22 @6, 1.12 @8, 1.11 @12, 0.87 @16, 0.94 @20. Best
      case **6.3x**, and it keeps improving *past* the core count because a worker
      waiting on a COG read is not using one. It turns over at ~2.5x cores, hence
      :data:`WORKERS_PER_CORE` capped at :data:`MAX_DEFAULT_WORKERS`.

    The oversubscription is transport-dependent: this box is in the US and the bucket
    is in ``eu-central-1``, so there is a lot of read latency to hide. A same-region
    run has less, and its knee will sit nearer the core count — re-measure with
    ``notebooks/c3_throughput.py`` rather than porting this number.

    ``os.process_cpu_count()`` (3.13+) respects an affinity mask or a container CPU
    limit; ``os.cpu_count()`` is the 3.12 fallback. Both count *logical* cores, so on
    an asymmetric box (this dev machine: 6P + 2E) the E-cores are included — they
    contribute, just less, and mild oversubscription absorbs the difference anyway.
    """
    cores = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    if _check_executor(kind) == "thread":
        return min(THREAD_GIL_KNEE, cores)
    return max(1, min(WORKERS_PER_CORE * cores, MAX_DEFAULT_WORKERS))


def make_executor(kind: ExecutorKind, workers: int) -> Executor:
    """Build the pool a caller fans across — the one place the two paths differ.

    Processes are forced onto the **spawn** start method rather than taking the
    platform default. macOS spawns anyway, but the production run is a rented Linux
    box where the default is fork, and forking a parent that has already opened GDAL
    handles and a boto3 client inherits both into a child that must not share them.
    Spawning everywhere means the box that runs the pass behaves like the box the
    pass was measured on.
    """
    if _check_executor(kind) == "thread":
        from concurrent.futures import ThreadPoolExecutor

        return ThreadPoolExecutor(max_workers=workers)

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    )
