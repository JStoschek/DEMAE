"""Sharded int16 memmap storage (ADR-0002 §1-2).

Tiles are appended into fixed-size ``.npy`` shards of shape
``(rows_per_shard, 224, 224)`` int16, written through :func:`numpy.lib.format`'s
memmap so the corpus is O(1)-random-access by ``(shard_id, row_index)`` at read
time — the answer to the project's #1 risk (I/O starving the GPU). The writer
sits behind the tiny :class:`TileSink` interface so a WebDataset backend can slot
in for Phase-3 multi-node streaming without touching the pipeline.

The writer is also **resumable** (issue #51 item 5). A multi-hour materialization
pass that dies mid-shard must not throw that shard away: :meth:`ShardWriter.checkpoint`
flushes the open memmap and returns the :class:`ShardState` watermark that says how
much of it is durable, and passing that state back into a new writer reopens the file
and appends from exactly there. That is what makes a half-written shard a resume point
rather than a thing to discard.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.lib.format import open_memmap

from . import config

SHARD_PREFIX = "shard"


@dataclass(frozen=True)
class TileRef:
    """Where a written tile lives, plus its integrity hash."""

    shard_id: int
    row_index: int
    content_hash: int  # uint64 blake2b-8 of the raw int16 LE bytes


def content_hash(raw_i16: np.ndarray) -> int:
    """64-bit blake2b hash of a tile's raw int16 **little-endian** bytes.

    Endianness is pinned so the hash identifies terrain content, not the byte
    order of whatever array produced it — the same integrity token on every
    platform (ADR-0002 §3: integrity + near-duplicate detection).
    """
    b = np.ascontiguousarray(raw_i16, dtype="<i2").tobytes()
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


def shard_path(out_dir: Path | str, shard_id: int, prefix: str = SHARD_PREFIX) -> Path:
    return Path(out_dir) / f"{prefix}_{shard_id:05d}.npy"


@dataclass(frozen=True)
class ShardState:
    """How much of a shard set is durable — the resume watermark (issue #51 item 5).

    ``shard_id``/``shard_row`` are where the *next* tile goes; ``shard_row_counts``
    holds the finished shards behind it. A default-constructed state means "nothing
    written yet", so a caller with no checkpoint on disk can pass one unconditionally.

    It is a plain value object because it has to survive a process death: the driver
    round-trips it through JSON alongside the rest of its progress record.
    """

    shard_id: int = 0
    shard_row: int = 0
    shard_row_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # Normalized to a tuple so a state read back from JSON (a list) compares equal
        # to the one the writer handed out, and so the value stays hashable/frozen.
        object.__setattr__(self, "shard_row_counts", tuple(self.shard_row_counts))

    @property
    def total_rows(self) -> int:
        """Tiles durably written across every shard — the manifest's expected length."""
        return sum(self.shard_row_counts) + self.shard_row

    def as_dict(self) -> dict:
        """JSON-safe form — the driver stores this in its progress record."""
        return {
            "shard_id": self.shard_id,
            "shard_row": self.shard_row,
            "shard_row_counts": list(self.shard_row_counts),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShardState":
        return cls(
            shard_id=int(d["shard_id"]),
            shard_row=int(d["shard_row"]),
            shard_row_counts=tuple(d["shard_row_counts"]),
        )


class TileSink(Protocol):
    """Minimal write interface the pipeline depends on (ADR-0002 §1)."""

    def add(self, raw_i16: np.ndarray) -> TileRef: ...

    def close(self) -> None: ...


def _normalize_resume(state: ShardState, rows_per_shard: int) -> ShardState:
    """Fold a watermark that sits exactly on a shard boundary into "shard finished".

    A checkpoint taken after the last row of a shard but before the next ``add``
    rolled reports ``shard_row == rows_per_shard``. That shard is complete and at its
    declared capacity, so the state is rewritten as "counts + that shard, next shard,
    row 0" — which the writer already knows how to start from, and which keeps
    ``shard_row_counts`` honest without reopening a file that has no room left.
    """
    if state.shard_row > rows_per_shard:
        raise ValueError(
            f"resume watermark row {state.shard_row} exceeds rows_per_shard "
            f"{rows_per_shard}"
        )
    if state.shard_row < rows_per_shard:
        return state
    return ShardState(
        shard_id=state.shard_id + 1,
        shard_row=0,
        shard_row_counts=(*state.shard_row_counts, rows_per_shard),
    )


class ShardWriter:
    """Append tiles into rolling fixed-size int16 ``.npy`` shards.

    Pass ``resume=`` a :class:`ShardState` from a previous run's
    :meth:`checkpoint` to continue appending into a half-written shard instead of
    starting over (issue #51 item 5).
    """

    def __init__(
        self,
        out_dir: Path | str,
        *,
        rows_per_shard: int = config.ROWS_PER_SHARD,
        tile_px: int = config.TILE_PX,
        dtype: type = config.STORAGE_DTYPE,
        prefix: str = SHARD_PREFIX,
        resume: ShardState | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows_per_shard = rows_per_shard
        self.tile_px = tile_px
        self.dtype = dtype
        self.prefix = prefix
        resume = _normalize_resume(resume or ShardState(), rows_per_shard)
        self.shard_row_counts: list[int] = list(resume.shard_row_counts)
        # Nothing is opened here — `add` opens lazily, exactly as the fresh path does.
        # A resume that landed on a shard boundary was normalized to "start the next
        # shard", so only a genuinely half-written shard takes the reopen path.
        self._resume_row = resume.shard_row
        self._shard_id = resume.shard_id if self._resume_row else resume.shard_id - 1
        self._mm: np.memmap | None = None
        self._row = resume.shard_row

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _roll(self) -> None:
        self._finalize_current()
        self._shard_id += 1
        self._row = 0
        self._mm = open_memmap(
            shard_path(self.out_dir, self._shard_id, self.prefix),
            mode="w+",
            dtype=self.dtype,
            shape=(self.rows_per_shard, self.tile_px, self.tile_px),
        )

    def _reopen(self) -> None:
        """Map the half-written shard the resume watermark points at, for appending.

        The file is at full declared capacity because :meth:`close` — the only thing
        that truncates — never ran on it. If it is *not*, the watermark is describing
        a shard that has already been finalized (or a file that was replaced), and
        appending would silently drop rows; that raises rather than guesses.
        """
        path = shard_path(self.out_dir, self._shard_id, self.prefix)
        mm = open_memmap(path, mode="r+")
        rows = mm.shape[0]
        if rows != self.rows_per_shard:
            del mm
            raise ValueError(
                f"cannot resume into {path.name}: it holds {rows} rows, not the "
                f"{self.rows_per_shard} a writable shard has — it was already finalized"
            )
        self._mm = mm
        self._resume_row = 0

    def _finalize_current(self) -> None:
        if self._mm is None:
            return
        n = self._row
        self._mm.flush()
        path = shard_path(self.out_dir, self._shard_id, self.prefix)
        del self._mm
        self._mm = None
        if n < self.rows_per_shard:
            # Rewrite the trailing shard so its declared shape is its real length
            # (no zero-padding rows masquerading as ocean tiles).
            full = np.load(path, mmap_mode="r")
            data = np.array(full[:n])
            del full
            np.save(path, data)
        self.shard_row_counts.append(n)

    def add(self, raw_i16: np.ndarray) -> TileRef:
        if raw_i16.shape != (self.tile_px, self.tile_px):
            raise ValueError(
                f"tile shape {raw_i16.shape} != ({self.tile_px}, {self.tile_px})"
            )
        if self._mm is None:
            self._reopen() if self._resume_row else self._roll()
        elif self._row >= self.rows_per_shard:
            self._roll()
        assert self._mm is not None
        idx = self._row
        self._mm[idx] = raw_i16.astype(self.dtype, copy=False)
        self._row += 1
        return TileRef(
            shard_id=self._shard_id, row_index=idx, content_hash=content_hash(raw_i16)
        )

    def checkpoint(self) -> ShardState:
        """Flush what has been written and return the watermark a resume starts from.

        Deliberately **not** :meth:`close`: closing truncates the trailing shard to its
        real length, which would make it unappendable and so end the build. This
        flushes the open memmap to disk and reports where the next tile goes, leaving
        the shard at full capacity and writable. Everything at or before the returned
        watermark survives a kill; anything after it is re-done by the resumed run.
        """
        if self._mm is not None:
            self._mm.flush()
        return ShardState(
            shard_id=max(self._shard_id, 0),
            shard_row=self._row,
            shard_row_counts=tuple(self.shard_row_counts),
        )

    def close(self) -> None:
        self._finalize_current()

    @property
    def num_shards(self) -> int:
        return len(self.shard_row_counts)
