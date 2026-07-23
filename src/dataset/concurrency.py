"""The bounded, order-preserving fan-out primitive shared by the parallel reads.

One scheduling helper, written against :class:`~concurrent.futures.Executor` rather than
a specific pool, so the same bounded/ordered scheduling serves both a *thread* pool (the
I/O-bound COG reads) and a *process* pool (:func:`dataset.catalog.scan`'s CPU-bound
scoring) without a second implementation (#38). It lived in the retired ADR-0001
pipeline; it outlives that pipeline because it is pure scheduling, owing nothing to how a
corpus is stratified.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Executor, Future
from typing import Callable, Iterable, Iterator, TypeVar

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
