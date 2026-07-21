"""End-to-end corpus builder — wires ADR-0001 + ADR-0002 into one run.

Data flow (CONTEXT.md stage map):

    discovery -> score (x8 overview: ocean screen + TRI) -> stratified select
              -> materialize (full-res 224 reads) -> clip/stats/degenerate
              -> shards + manifest + corpus_meta.json

The COG reader is injected (:class:`~dataset.tiling.CogReader`), so the same
orchestration runs against the live bucket or an in-memory synthetic corpus.
Wall-clock is measured per stage (see :class:`Timings`) to feed the cloud-burst
decision in :mod:`dataset.throughput`.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol, TypeVar

import numpy as np

log = logging.getLogger(__name__)

_T = TypeVar("_T")
_R = TypeVar("_R")


def ordered_futures(
    ex: ThreadPoolExecutor,
    fn: Callable[[_T], _R],
    items: Iterable[_T],
    in_flight: int,
) -> Iterator[tuple[_T, "Future[_R]"]]:
    """Yield ``(item, future)`` in input order, keeping ``in_flight`` submitted.

    The scheduling primitive under every parallel COG read here. Two properties it
    has to hold at once:

    - **Input order.** A caller consuming these in order gets a result sequence
      independent of which read finished first, so the corpus (and the catalog) is
      identical to the ``workers=1`` path — only the wall-clock changes.
    - **Bounded memory.** ``Executor.map`` submits every item up front and buffers
      each finished result until the one ahead of it is consumed. Each result here is
      a whole COG's worth of rows or arrays, so one slow read could pin the rest of
      the corpus in memory behind it. A sliding window keeps the pool fed while
      bounding the queue and everything held in it.

    Handing back the future rather than its value is what lets a caller decide per
    item whether a failed read is fatal (:func:`_ordered_map`, which re-raises) or
    something to record and step over (:func:`dataset.catalog.scan`).
    """
    from collections import deque

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


def _ordered_map(
    fn: Callable[[_T], _R], items: Iterable[_T], workers: int
) -> Iterator[_R]:
    """``map(fn, items)`` yielding in input order, run concurrently when
    ``workers > 1``.

    The COG reads that dominate the build (overview scoring, full-res windows) are
    independent and I/O-bound — GDAL/rasterio release the GIL during the S3 fetch —
    so a thread pool overlaps the transatlantic latency the throughput probe
    measured. Ordering and the memory bound both come from :func:`ordered_futures`;
    a read that raises propagates here, aborting the build.
    """
    if workers <= 1:
        yield from map(fn, items)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _, fut in ordered_futures(fn=fn, ex=ex, items=items, in_flight=workers * 2):
            yield fut.result()


class _Bar(Protocol):
    """The tiny slice of a progress bar the pipeline drives (tqdm satisfies it)."""

    def update(self, n: int = ...) -> object: ...

    def __enter__(self) -> "_Bar": ...

    def __exit__(self, *exc: object) -> object: ...


# A progress factory turns ``(total, desc)`` into a context-managed bar. Injected
# so the ETL core stays free of any progress dependency (the CLI wires tqdm in);
# ``None`` -> the no-op bar below, which is what every test and library call gets.
ProgressFactory = Callable[[int, str], _Bar]


class _NullBar:
    def update(self, n: int = 1) -> None:
        pass

    def __enter__(self) -> "_NullBar":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _null_factory(total: int, desc: str) -> _NullBar:
    return _NullBar()

from . import config, manifest, normalize, stratify
from .geometry import (
    CogRef,
    geocell_id,
    grid_windows,
    lat_band,
    window_center_lonlat,
)
from .manifest import TileRecord, make_tile_id
from .shards import ShardWriter
from .tiling import CogReader


@dataclass
class _Candidate:
    """A scored candidate window, carrying everything a manifest row needs."""

    cog: CogRef
    win_row0: int
    win_col0: int
    tri: float
    lon: float
    lat: float
    lat_band: str
    geocell: str
    tile_id: str
    stratum: int = -1  # filled after global stratification


@dataclass
class Timings:
    """Wall-clock breakdown, the raw material for the cloud-burst decision."""

    n_cogs_scored: int = 0
    overview_read_s: float = 0.0  # sum of x8-overview read times
    score_s: float = 0.0  # TRI + ocean scoring compute
    n_windows_read: int = 0  # full-res window reads at materialization
    window_read_s: float = 0.0  # sum of full-res read times
    process_s: float = 0.0  # prepare_tile + hashing + shard write

    @property
    def per_cog_overview_s(self) -> float:
        return self.overview_read_s / self.n_cogs_scored if self.n_cogs_scored else 0.0

    @property
    def per_window_read_s(self) -> float:
        return self.window_read_s / self.n_windows_read if self.n_windows_read else 0.0

    @property
    def per_window_process_s(self) -> float:
        return self.process_s / self.n_windows_read if self.n_windows_read else 0.0


@dataclass
class PipelineResult:
    """Outcome of a corpus build."""

    n_candidates: int
    n_selected: int
    n_written: int
    drop_reasons: dict[str, int]
    dataset_version: str
    manifest_path: Path
    corpus_meta_path: Path
    shard_dir: Path
    global_mean: float
    global_std: float
    timings: Timings = field(default_factory=Timings)


def run_pipeline(
    reader: CogReader,
    out_dir: Path | str,
    *,
    per_stratum_quota: int = config.PER_STRATUM_QUOTA,
    per_tile_cap: int = config.PER_TILE_WINDOW_CAP,
    tri_factor: int = config.TRI_OVERVIEW_FACTOR,
    seed: int = config.RANDOM_SEED,
    limit_cogs: int | None = None,
    snapshot_date: str = "unknown",
    code_version: str = "unknown",
    build_timestamp: str | None = None,
    progress: ProgressFactory | None = None,
    workers: int = 1,
) -> PipelineResult:
    """Build a corpus into ``out_dir`` and return a :class:`PipelineResult`.

    ``limit_cogs`` truncates discovery for smoke runs; ``per_stratum_quota`` sets
    the per-stratum target (equal quotas -> ``5 * quota`` total, ADR-0001 §5).
    ``progress`` is an optional bar factory for the two long stages (scoring every
    COG, materializing every selected tile); ``None`` runs silently. ``workers``
    fans the I/O-bound COG reads across a thread pool (``1`` = sequential); the
    written corpus is identical for any ``workers`` (reads are consumed in COG
    order), so the throughput probe still measures single-thread per-unit costs.
    """
    out_dir = Path(out_dir)
    shard_dir = out_dir / "shards"
    timings = Timings()
    make_bar = progress or _null_factory

    cogs = reader.list_cogs()
    if limit_cogs is not None:
        cogs = cogs[:limit_cogs]

    candidates = _score(reader, cogs, tri_factor, timings, make_bar, workers)

    # Global stratification of the candidate TRI distribution, then quota fill.
    if candidates:
        tri_values = np.array([c.tri for c in candidates], dtype=np.float64)
        edges = stratify.quantile_edges(tri_values, config.N_STRATA)
        strata = stratify.assign_strata(tri_values, edges)
        for c, s in zip(candidates, strata):
            c.stratum = int(s)
        select_input = [
            stratify.Candidate(
                index=i,
                stratum=c.stratum,
                geocell=c.geocell,
                cog_stem=c.cog.stem,
                tri=c.tri,
                tile_id=c.tile_id,
            )
            for i, c in enumerate(candidates)
        ]
        chosen = stratify.select_quota(
            select_input,
            per_stratum_quota=per_stratum_quota,
            per_tile_cap=per_tile_cap,
            seed=seed,
        )
        selected = [candidates[c.index] for c in chosen]
    else:
        selected = []

    rows, drops, gstats = _materialize(
        reader, selected, shard_dir, timings, make_bar, workers
    )

    # The quota is filled from overview-scored candidates; the authoritative
    # full-res screen in prepare_tile can still drop some, so the written corpus
    # may undershoot the target. ADR-0001 §5 tolerates under-fill but wants it
    # visible — surface it rather than silently shipping a short corpus.
    target = per_stratum_quota * config.N_STRATA
    if len(candidates) and len(selected) < target:
        log.info(
            "quota under-filled at selection: %d/%d (candidate scarcity)",
            len(selected),
            target,
        )
    if len(rows) < len(selected):
        log.warning(
            "%d selected windows dropped at full-res materialization %s; "
            "wrote %d/%d tiles",
            len(selected) - len(rows),
            dict(drops),
            len(rows),
            target,
        )

    global_mean, global_std = gstats.finalize()

    manifest_path = out_dir / "manifest.parquet"
    manifest.write_manifest(rows, manifest_path)
    corpus_meta_path = out_dir / "corpus_meta.json"
    meta = manifest.write_corpus_meta(
        corpus_meta_path,
        rows=rows,
        global_mean=global_mean,
        global_std=global_std,
        snapshot_date=snapshot_date,
        build_timestamp=build_timestamp or _utc_now(),
        code_version=code_version,
    )

    return PipelineResult(
        n_candidates=len(candidates),
        n_selected=len(selected),
        n_written=len(rows),
        drop_reasons=dict(drops),
        dataset_version=meta["dataset_version"],
        manifest_path=manifest_path,
        corpus_meta_path=corpus_meta_path,
        shard_dir=shard_dir,
        global_mean=global_mean,
        global_std=global_std,
        timings=timings,
    )


# --- Stage: score every candidate window from one x8-overview read per COG -----


def _score_one(
    reader: CogReader, cog: CogRef, tri_factor: int
) -> tuple[list[_Candidate], float, float]:
    """Score every land window of one COG from a single x8-overview read.

    Pure w.r.t. the COG (no shared state), so it is safe to run across the thread
    pool; returns its candidates plus the read/compute times to accumulate.
    """
    t0 = time.perf_counter()
    ov = reader.read_overview(cog, tri_factor)
    t1 = time.perf_counter()
    tri_map = stratify.tri_riley(ov.array)
    cands: list[_Candidate] = []
    for win_row0, win_col0 in grid_windows(ov.full_w, ov.full_h):
        o_r0, o_r1, o_c0, o_c1 = stratify.window_overview_slice(
            win_row0=win_row0,
            win_col0=win_col0,
            tile=config.TILE_PX,
            full_h=ov.full_h,
            full_w=ov.full_w,
            ov_h=ov.ov_h,
            ov_w=ov.ov_w,
        )
        region = ov.array[o_r0:o_r1, o_c0:o_c1]
        if stratify.is_zero_fill(region):  # coarse pre-screen on the overview
            continue
        tri = stratify.window_tri(tri_map, o_r0=o_r0, o_c0=o_c0, o_r1=o_r1, o_c1=o_c1)
        lon, lat = window_center_lonlat(
            origin_lon=cog.origin_lon,
            origin_lat=cog.origin_lat,
            width=ov.full_w,
            height=ov.full_h,
            win_row0=win_row0,
            win_col0=win_col0,
        )
        cands.append(
            _Candidate(
                cog=cog,
                win_row0=win_row0,
                win_col0=win_col0,
                tri=tri,
                lon=lon,
                lat=lat,
                lat_band=lat_band(lat),
                geocell=geocell_id(lon, lat),
                tile_id=make_tile_id(cog.stem, win_row0, win_col0),
            )
        )
    t2 = time.perf_counter()
    return cands, t1 - t0, t2 - t1


def _score(
    reader: CogReader,
    cogs: list[CogRef],
    tri_factor: int,
    timings: Timings,
    make_bar: ProgressFactory,
    workers: int,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    scored = _ordered_map(lambda c: _score_one(reader, c, tri_factor), cogs, workers)
    with make_bar(len(cogs), "scoring COGs") as bar:
        for cands, read_s, score_s in scored:
            candidates.extend(cands)  # extended in COG order -> deterministic
            timings.overview_read_s += read_s
            timings.score_s += score_s
            timings.n_cogs_scored += 1
            bar.update(1)
    return candidates


# --- Stage: materialize selected windows at full resolution -------------------


class _GlobalStats:
    """Running corpus-wide land-pixel mean/std (ADR-0003 global-norm handoff)."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.sumsq = 0.0
        self.count = 0

    def update(self, raw_i16: np.ndarray) -> None:
        vals = raw_i16[normalize.land_mask(raw_i16)].astype(np.float64)
        self.sum += float(vals.sum())
        self.sumsq += float(np.square(vals).sum())
        self.count += vals.size

    def finalize(self) -> tuple[float, float]:
        if self.count == 0:
            return 0.0, 0.0
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean * mean, 0.0)
        return mean, math.sqrt(var)


@dataclass
class _CogWindows:
    """One COG's selected windows, read at full-res and awaiting the writer."""

    stem: str
    cands: list[_Candidate]
    windows: list[np.ndarray]
    read_s: float


def _read_cog_windows(
    reader: CogReader, stem: str, cands: list[_Candidate], cog: CogRef
) -> _CogWindows:
    """Full-res read of every selected window in one COG (thread-pool safe: it
    only touches its own COG and returns arrays, never the shared writer)."""
    origins = [(c.win_row0, c.win_col0) for c in cands]
    t0 = time.perf_counter()
    windows = reader.read_windows(cog, origins, config.TILE_PX)
    return _CogWindows(stem, cands, windows, time.perf_counter() - t0)


def _materialize(
    reader: CogReader,
    selected: list[_Candidate],
    shard_dir: Path,
    timings: Timings,
    make_bar: ProgressFactory,
    workers: int,
) -> tuple[list[TileRecord], Counter, _GlobalStats]:
    # Group by COG so each source is opened once for all its selected windows.
    by_cog: dict[str, list[_Candidate]] = defaultdict(list)
    cog_ref: dict[str, CogRef] = {}
    for c in selected:
        by_cog[c.cog.stem].append(c)
        cog_ref[c.cog.stem] = c.cog

    rows: list[TileRecord] = []
    drops: Counter = Counter()
    gstats = _GlobalStats()

    # Reads fan out across the pool; the shard writer, global stats, and row order
    # stay on this single thread (the writer is not thread-safe, and sequential
    # writes in COG order are what make shard assignment reproducible). Results
    # arrive in COG order, so the corpus is identical to the workers=1 path.
    read = _ordered_map(
        lambda item: _read_cog_windows(reader, item[0], item[1], cog_ref[item[0]]),
        by_cog.items(),
        workers,
    )
    with ShardWriter(shard_dir) as writer, make_bar(len(selected), "materializing tiles") as bar:
        for batch in read:
            timings.window_read_s += batch.read_s
            timings.n_windows_read += len(batch.windows)

            p0 = time.perf_counter()
            for cand, win in zip(batch.cands, batch.windows):
                tile, reason = normalize.prepare_tile(win)
                if tile is None:
                    drops[reason] += 1
                else:
                    ref = writer.add(tile.raw_i16)
                    gstats.update(tile.raw_i16)
                    rows.append(
                        TileRecord(
                            tile_id=cand.tile_id,
                            shard_id=ref.shard_id,
                            row_index=ref.row_index,
                            source_cog=f"{batch.stem}/{batch.stem}.tif",
                            win_row0=cand.win_row0,
                            win_col0=cand.win_col0,
                            lon=cand.lon,
                            lat=cand.lat,
                            lat_band=cand.lat_band,
                            geocell_id=cand.geocell,
                            tri=cand.tri,
                            stratum=cand.stratum,
                            mean=tile.mean,
                            std=tile.std,
                            land_frac=tile.land_frac,
                            content_hash=ref.content_hash,
                        )
                    )
                bar.update(1)
            timings.process_s += time.perf_counter() - p0
    return rows, drops, gstats


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
