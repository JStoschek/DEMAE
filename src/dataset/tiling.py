"""Stage 2 — COG reads behind a swappable :class:`CogReader` interface.

Two reads matter (recon §5): one cheap **x8-overview** read per COG (feeds both
the ocean screen and TRI scoring for every candidate window in that COG), then
**full-resolution 224x224 window** reads for the tiles that survive selection.
Opening a COG dominates cost, so window reads are batched per COG to amortize the
open across many windows.

The interface is a Protocol so the pipeline can run against the real
:class:`RasterioCogReader` (``/vsis3`` anonymous access) or an in-memory
:class:`SyntheticCogReader` — the latter makes the whole recipe testable and
powers the throughput harness's synthetic mode without touching S3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import config
from .geometry import CogRef


@dataclass(frozen=True)
class Overview:
    """A decimated view of a whole COG plus its full-resolution dimensions."""

    array: np.ndarray  # (ov_h, ov_w) float32
    full_h: int
    full_w: int

    @property
    def ov_h(self) -> int:
        return self.array.shape[0]

    @property
    def ov_w(self) -> int:
        return self.array.shape[1]


class CogReader(Protocol):
    """The COG-access surface the pipeline depends on."""

    def list_cogs(self) -> list[CogRef]: ...

    def read_overview(self, cog: CogRef, factor: int) -> Overview: ...

    def read_windows(
        self, cog: CogRef, origins: list[tuple[int, int]], tile: int
    ) -> list[np.ndarray]: ...


class RasterioCogReader:
    """Read Copernicus COGs from ``s3://copernicus-dem-30m`` via GDAL /vsis3."""

    def __init__(self, bucket: str = config.COPERNICUS_GLO30_BUCKET) -> None:
        self.bucket = bucket

    def _path(self, cog: CogRef) -> str:
        return f"/vsis3/{self.bucket}/{cog.stem}/{cog.stem}.tif"

    def list_cogs(self) -> list[CogRef]:
        from . import discovery

        return discovery.list_cog_refs(self.bucket)

    def read_overview(self, cog: CogRef, factor: int) -> Overview:
        import rasterio

        with rasterio.Env(**config.AWS_NO_SIGN_ENV):
            with rasterio.open(self._path(cog)) as ds:
                full_h, full_w = ds.height, ds.width
                oh = math.ceil(full_h / factor)
                ow = math.ceil(full_w / factor)
                # out_shape at the overview resolution makes GDAL serve the stored
                # x{factor} overview rather than resampling full-res (recon §5).
                arr = ds.read(1, out_shape=(oh, ow)).astype(np.float32)
        return Overview(array=arr, full_h=full_h, full_w=full_w)

    def read_windows(
        self, cog: CogRef, origins: list[tuple[int, int]], tile: int
    ) -> list[np.ndarray]:
        import rasterio
        from rasterio.windows import Window

        out: list[np.ndarray] = []
        with rasterio.Env(**config.AWS_NO_SIGN_ENV):
            with rasterio.open(self._path(cog)) as ds:  # open once, read many
                for row0, col0 in origins:
                    win = Window(col_off=col0, row_off=row0, width=tile, height=tile)
                    out.append(ds.read(1, window=win).astype(np.float32))
        return out


class SyntheticCogReader:
    """In-memory reader over fabricated full-res arrays (tests + throughput).

    Overviews are simple strided decimations of the source array, which is enough
    to exercise the scoring/selection/materialization flow deterministically.
    """

    def __init__(self, tiles: dict[str, np.ndarray]) -> None:
        # keyed by COG stem -> full-resolution float32 array
        self._tiles = {k: np.asarray(v, dtype=np.float32) for k, v in tiles.items()}
        self._refs = [self._ref(stem) for stem in self._tiles]

    @staticmethod
    def _ref(stem: str) -> CogRef:
        from .geometry import parse_cog_key

        return parse_cog_key(stem)

    def list_cogs(self) -> list[CogRef]:
        return list(self._refs)

    def read_overview(self, cog: CogRef, factor: int) -> Overview:
        a = self._tiles[cog.stem]
        ov = a[::factor, ::factor]
        return Overview(array=ov.astype(np.float32), full_h=a.shape[0], full_w=a.shape[1])

    def read_windows(
        self, cog: CogRef, origins: list[tuple[int, int]], tile: int
    ) -> list[np.ndarray]:
        a = self._tiles[cog.stem]
        return [a[r : r + tile, c : c + tile].astype(np.float32) for r, c in origins]
