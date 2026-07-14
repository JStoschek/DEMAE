"""Sharded int16 memmap storage (ADR-0002 §1-2).

Tiles are appended into fixed-size ``.npy`` shards of shape
``(rows_per_shard, 224, 224)`` int16, written through :func:`numpy.lib.format`'s
memmap so the corpus is O(1)-random-access by ``(shard_id, row_index)`` at read
time — the answer to the project's #1 risk (I/O starving the GPU). The writer
sits behind the tiny :class:`TileSink` interface so a WebDataset backend can slot
in for Phase-3 multi-node streaming without touching the pipeline.
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


class TileSink(Protocol):
    """Minimal write interface the pipeline depends on (ADR-0002 §1)."""

    def add(self, raw_i16: np.ndarray) -> TileRef: ...

    def close(self) -> None: ...


class ShardWriter:
    """Append tiles into rolling fixed-size int16 ``.npy`` shards."""

    def __init__(
        self,
        out_dir: Path | str,
        *,
        rows_per_shard: int = config.ROWS_PER_SHARD,
        tile_px: int = config.TILE_PX,
        dtype: type = config.STORAGE_DTYPE,
        prefix: str = SHARD_PREFIX,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows_per_shard = rows_per_shard
        self.tile_px = tile_px
        self.dtype = dtype
        self.prefix = prefix
        self.shard_row_counts: list[int] = []
        self._shard_id = -1
        self._mm: np.memmap | None = None
        self._row = 0

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
        if self._mm is None or self._row >= self.rows_per_shard:
            self._roll()
        assert self._mm is not None
        idx = self._row
        self._mm[idx] = raw_i16.astype(self.dtype, copy=False)
        self._row += 1
        return TileRef(
            shard_id=self._shard_id, row_index=idx, content_hash=content_hash(raw_i16)
        )

    def close(self) -> None:
        self._finalize_current()

    @property
    def num_shards(self) -> int:
        return len(self.shard_row_counts)
