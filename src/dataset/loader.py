"""Stage 6 — the Phase-1 read path (ADR-0003).

A map-style ``torch.utils.data.Dataset`` over the Parquet manifest with lazy
per-worker memmaps and a pluggable :class:`Normalizer` (the per-tile-vs-global
ablation hook). It resolves ``manifest[i] -> (shard_id, row_index) ->
memmap[shard_id][row_index]``, casts int16 -> fp32, derives the land mask
*before* normalization, normalizes, adds the channel axis (C=1, channel-ready),
and returns the ADR-0003 dict sample.

``torch`` is imported lazily (only when a :class:`TileDataset` is constructed), so
the ETL core and the numpy-only :class:`Normalizer` strategies import without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from . import config
from .shards import shard_path

# --- Normalization strategies (pure numpy — the ablation hook, ADR-0003 §3) ---

_STD_FLOOR = 1e-6  # guard; ETL already drops std < 3 m tiles


class Normalizer(Protocol):
    def __call__(self, x: np.ndarray, *, mean: float, std: float) -> np.ndarray: ...


class PerTileNormalizer:
    """``(x - mean) / std`` using the sample's own manifest stats (default)."""

    name = "per_tile"

    def __call__(self, x: np.ndarray, *, mean: float, std: float) -> np.ndarray:
        return (x - mean) / max(std, _STD_FLOOR)


class GlobalNormalizer:
    """``(x - μ) / σ`` using corpus-wide land stats from ``corpus_meta.json``."""

    name = "global"

    def __init__(self, global_mean: float, global_std: float) -> None:
        self.global_mean = global_mean
        self.global_std = max(global_std, _STD_FLOOR)

    def __call__(self, x: np.ndarray, *, mean: float, std: float) -> np.ndarray:
        return (x - self.global_mean) / self.global_std


def make_normalizer(name: str, meta: dict | None = None) -> Normalizer:
    """Build the normalizer selected by loader config (ADR-0003 §3)."""
    if name == "per_tile":
        return PerTileNormalizer()
    if name == "global":
        if meta is None:
            raise ValueError("global normalization needs corpus_meta for μ/σ")
        return GlobalNormalizer(meta["global_mean"], meta["global_std"])
    raise ValueError(f"unknown normalization {name!r} (expected per_tile | global)")


# --- Dataset (lazy torch) -----------------------------------------------------

_DATASET_CLASS = None


def _dataset_class():
    """Build (once) the torch-subclassed Dataset. Imports torch lazily."""
    global _DATASET_CLASS
    if _DATASET_CLASS is not None:
        return _DATASET_CLASS

    import pyarrow.parquet as pq
    import torch
    from torch.utils.data import Dataset

    from .normalize import land_mask

    class _TileDataset(Dataset):
        def __init__(
            self,
            corpus_dir: Path | str,
            *,
            normalization: str = "per_tile",
            channels: tuple[str, ...] = config.INPUT_CHANNELS,
        ) -> None:
            self.corpus_dir = Path(corpus_dir)
            self.shard_dir = self.corpus_dir / "shards"
            self.channels = channels

            table = pq.read_table(self.corpus_dir / "manifest.parquet")
            # Keep the index in memory as plain arrays/lists (loaded once).
            self._shard_id = table.column("shard_id").to_numpy()
            self._row_index = table.column("row_index").to_numpy()
            self._mean = table.column("mean").to_numpy()
            self._std = table.column("std").to_numpy()
            self._stratum = table.column("stratum_id").to_numpy()  # v2 rename (#48)
            self._tile_id = table.column("tile_id").to_pylist()

            import json

            meta_path = self.corpus_dir / "corpus_meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
            self.normalizer = make_normalizer(normalization, meta)

            # Lazily-opened, per-worker memmap handles (NOT opened here — that
            # would fork/pickle them into every worker). Empty dict is fork-safe.
            self._memmaps: dict[int, np.memmap] = {}

        def __len__(self) -> int:
            return len(self._tile_id)

        def _memmap(self, shard_id: int) -> np.memmap:
            mm = self._memmaps.get(shard_id)
            if mm is None:
                mm = np.load(shard_path(self.shard_dir, int(shard_id)), mmap_mode="r")
                self._memmaps[shard_id] = mm
            return mm

        def __getitem__(self, index: int) -> dict:
            i = index
            shard_id = int(self._shard_id[i])
            row = int(self._row_index[i])
            raw = self._memmap(shard_id)[row]  # int16 (H, W)

            mask = land_mask(raw)  # raw != 0, derived BEFORE normalization (ADR-0003 §4)
            mean = float(self._mean[i])
            std = float(self._std[i])
            x = raw.astype(np.float32)
            x = self.normalizer(x, mean=mean, std=std)

            # Channel-ready: (C, H, W) with C = len(channels) (1 in Phase 0).
            image = np.repeat(x[None, ...], len(self.channels), axis=0)

            return {
                "image": torch.from_numpy(np.ascontiguousarray(image)),
                "land_mask": torch.from_numpy(np.ascontiguousarray(mask)),
                "mean": torch.tensor(mean, dtype=torch.float32),
                "std": torch.tensor(std, dtype=torch.float32),
                "tile_id": self._tile_id[i],
                "stratum_id": int(self._stratum[i]),
            }

    _DATASET_CLASS = _TileDataset
    return _DATASET_CLASS


def TileDataset(
    corpus_dir: Path | str,
    *,
    normalization: str = "per_tile",
    channels: tuple[str, ...] = config.INPUT_CHANNELS,
):
    """Construct the corpus Dataset (see :func:`_dataset_class`).

    A factory rather than a bare class so importing this module never requires
    torch; torch is imported the moment a dataset is actually built.
    """
    return _dataset_class()(
        corpus_dir, normalization=normalization, channels=channels
    )


def build_dataloader(
    dataset,
    *,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
    **overrides,
):
    """Wrap a :func:`TileDataset` in a DataLoader with ADR-0003 §1's read plumbing.

    Encodes the recommended perf flags — global ``shuffle`` (a perfect per-epoch
    shuffle is free over memmap random access), ``pin_memory``,
    ``persistent_workers``, and default ``prefetch_factor`` — so callers get the
    tuned read path by default. Any flag is overridable via ``**overrides``.
    """
    from torch.utils.data import DataLoader

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    kwargs.update(overrides)
    return DataLoader(dataset, **kwargs)
