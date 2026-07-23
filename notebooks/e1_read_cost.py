"""E1 (#49) — prices the full-resolution materialization pass, and its fallback.

Question this answers
---------------------
E1 reads **50,000 full-resolution 224x224 windows across 15,834 COGs** off
``s3://copernicus-dem-30m`` and writes ~5 GB of int16 shards. Nobody has measured
that workload. The one throughput number this repo owns — C3's 0.64 s/COG at 16
processes (:doc:`c3-scoring-throughput`) — was measured on the *scoring* pass, which
is ~98% GIL-holding numpy over a whole x8 overview. E1 is the opposite shape: a
windowed range-GET per tile, trivial clip-and-stats compute, a memmap append. So
C3's "processes beat threads 4.9x vs 2.43x" may well **invert** here, and its worker
knee (16 on 8 cores) is not transferable either.

C3's own lesson is the one this script exists to obey: assuming a scaling factor
rather than measuring it turned a 4.19 h projection into a real 13.7 h.

What it measures
----------------
Four things, in the order they decide the answer:

- **probe** — an instrumented serial pass that separates ``open`` from ``read`` from
  ``prepare`` per COG, bucketed by how many windows the COG owns (1 / 2 / p99 / max).
  This is what says how much batching the open actually buys at a mean of 3.16
  windows/COG, and whether the max-44 COG behaves like the median-2 one;
- **sweep** — thread vs process vs serial, over a worker sweep, on one band-spread
  draw. Speedups are quoted against a **measured** serial run, never an assumed one;
- **write** — ``prepare_tile`` + ``ShardWriter.add`` + ``content_hash`` in isolation,
  with no network in the loop, to answer whether the write path is ever the
  bottleneck or is free next to the reads;
- **project** — the 50k debug tier and the Phase-F 200-500k scale tier, with the
  per-COG spread each tier actually implies rather than the 50k spread carried over.

Experimental hygiene
--------------------
- **Each sweep config runs in a fresh subprocess.** GDAL keeps a per-process block
  cache and ``/vsis3`` keeps a per-process file-info cache, so a thread config run
  in the parent would inherit whatever the previous config warmed and read faster
  for a reason that has nothing to do with threads. Re-invoking this file per config
  makes every config equally cold.
- **No warm-up discount.** Every config re-streams every window from ``eu-central-1``,
  so all of them carry the same US-latency penalty.
- **Wall clock end to end, per config.** No dividing a serial number by a worker count.
- **The sample is the real selection list**, band-spread over its own 15,834 COGs by
  :func:`dataset.sampling.sample_cogs`, taking *all* of a drawn COG's selected
  windows. Windows/COG falls from 3.79 in the 0-50 band to 1.83 above 85, so a draw
  that ignored bands would misprice the open-amortization it is here to measure.

Needs the live bucket for ``probe``/``sweep`` (they are transport measurements), so
neither is part of the test suite. ``write`` and ``project`` are offline.

Run::

    uv run python notebooks/e1_read_cost.py probe
    uv run python notebooks/e1_read_cost.py sweep
    uv run python notebooks/e1_read_cost.py write
    uv run python notebooks/e1_read_cost.py project

    N_COGS=48 uv run python notebooks/e1_read_cost.py sweep
    CONFIGS=serial:1,thread:16 uv run python notebooks/e1_read_cost.py sweep

Writes: ``docs/research/e1-read-cost.json`` — one section per mode, the artifact
behind the table in ``docs/research/e1-read-cost.md``.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from dataset import config, normalize
from dataset.concurrency import ordered_futures
from dataset.geometry import parse_cog_key
from dataset.sampling import band_of, sample_cogs
from dataset.shards import ShardWriter, content_hash
from dataset.tiling import RasterioCogReader

REPO = Path(__file__).resolve().parents[1]
SELECTION = REPO / "artifacts/phase_d/selection_debug_50k.csv"
RESEARCH = REPO / "docs/research"
OUT_JSON = RESEARCH / "e1-read-cost.json"
SCRATCH = REPO / "data/_e1_scratch"  # under data/, which is gitignored

#: Phase-D's realized 50k draw (map #46's hand-over block), the projection base.
DEBUG_WINDOWS = 50_000
DEBUG_COGS = 15_834

#: (executor, workers). ``serial`` is the baseline every speedup is quoted against,
#: and it is *measured*, not assumed. Threads run well past the core count because a
#: worker blocked on a range-GET is not holding a core; processes are swept over the
#: same range so the comparison is like-for-like.
DEFAULT_CONFIGS = (
    ("serial", 1),
    ("thread", 4),
    ("thread", 8),
    ("thread", 16),
    ("thread", 24),
    ("thread", 32),
    ("thread", 48),
    ("process", 4),
    ("process", 8),
    ("process", 16),
)


# --- the sample ---------------------------------------------------------------


def load_selection(path: Path = SELECTION) -> dict[str, list[tuple[int, int]]]:
    """Group the committed selection list by COG stem.

    ``window_id`` is ``<stem>:<row0>:<col0>``. Grouping is not a convenience — it is
    the reason ``tiling.read_windows`` takes a *list* of origins: one open serves
    every window the COG owns, and the driver cannot get that amortization unless it
    sees a COG's windows together. Origins are sorted so a run is order-stable.
    """
    groups: dict[str, list[tuple[int, int]]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            stem, r0, c0 = row["window_id"].rsplit(":", 2)
            groups.setdefault(stem, []).append((int(r0), int(c0)))
    return {k: sorted(v) for k, v in sorted(groups.items())}


def draw(
    groups: dict[str, list[tuple[int, int]]],
    n: int,
    seed: int = config.RANDOM_SEED,
    win_min: int = 1,
    win_max: int = 1 << 30,
):
    """A band-spread draw of ``n`` COGs from the selection list's own frame.

    ``sample_cogs`` apportions by latitude band and round-robins geo-cells inside
    each, and any prefix of its result is itself a spread sample. Drawing from the
    *selection list's* COGs rather than the 26,450-COG global frame keeps the
    windows/COG distribution — the thing being priced — the real one.

    ``win_min``/``win_max`` restrict the draw to COGs of a given window-count class.
    That is how :func:`mode_mix` gets two samples with *different* blocks-per-COG out
    of one frame: with two mixes the per-COG and per-block costs separate, and a
    projection can be re-mixed for a tier whose spread is not the 50k tier's.
    """
    pool = {k: v for k, v in groups.items() if win_min <= len(v) <= win_max}
    refs = [parse_cog_key(s) for s in pool]
    return [(r.stem, pool[r.stem]) for r in sample_cogs(refs, n, seed=seed)]


# --- the unit of work ---------------------------------------------------------


def work_cog(job: tuple[str, list[tuple[int, int]]]) -> dict:
    """Read and prepare every selected window of one COG. **This is the driver's
    inner loop** — deliberately, so the sweep prices the real thing.

    Returns prepared int16 tiles rather than raw float32 windows: the preparation is
    cheap and local, and doing it in the worker halves what a process pool has to
    ship back over its pipe (100 KB int16 vs 200 KB float32 per tile). The shard
    write stays in the parent — a memmap append is stateful and must stay serial to
    keep row order deterministic.
    """
    stem, origins = job
    reader = RasterioCogReader()
    windows = reader.read_windows(parse_cog_key(stem), origins, config.TILE_PX)

    tiles: list[np.ndarray] = []
    stats: list[tuple[float, float, float]] = []
    reasons: Counter[str] = Counter()
    for w in windows:
        prepared, reason = normalize.prepare_tile(w)
        reasons[reason] += 1
        if prepared is not None:
            tiles.append(prepared.raw_i16)
            stats.append((prepared.mean, prepared.std, prepared.land_frac))
    return {"stem": stem, "tiles": tiles, "stats": stats, "reasons": dict(reasons)}


# --- mode: probe --------------------------------------------------------------


def probe_cog(stem: str, origins: list[tuple[int, int]]) -> dict:
    """One COG, with ``open`` / ``read`` / ``prepare`` timed apart.

    Instrumenting the phases beats timing a batched run against an unbatched one:
    re-opening the same object in the same process hits GDAL's ``/vsis3`` file-info
    cache, so an empirical "unbatched" number would flatter batching's alternative
    and understate the win. The decomposition below prices a *cold* open once and
    lets the arithmetic do the rest.
    """
    import rasterio
    from rasterio.windows import Window

    reader = RasterioCogReader()
    path = reader._path(parse_cog_key(stem))

    reads: list[float] = []
    preps: list[float] = []
    reasons: Counter[str] = Counter()
    t0 = time.perf_counter()
    with rasterio.Env(**config.AWS_NO_SIGN_ENV):
        with rasterio.open(path) as ds:
            t_open = time.perf_counter() - t0
            for r0, c0 in origins:
                t1 = time.perf_counter()
                arr = ds.read(
                    1, window=Window(col_off=c0, row_off=r0, width=config.TILE_PX, height=config.TILE_PX)
                ).astype(np.float32)
                reads.append(time.perf_counter() - t1)
                t2 = time.perf_counter()
                _, reason = normalize.prepare_tile(arr)
                preps.append(time.perf_counter() - t2)
                reasons[reason] += 1
    return {
        "stem": stem,
        "n_windows": len(origins),
        "open_s": t_open,
        "read_s": reads,
        "prepare_s": preps,
        "reasons": dict(reasons),
    }


def mode_probe() -> dict:
    """Serial, instrumented, over four COG size classes."""
    groups = load_selection()
    sizes = {k: len(v) for k, v in groups.items()}
    by_size: dict[int, list[str]] = {}
    for stem, n in sizes.items():
        by_size.setdefault(n, []).append(stem)

    n_max = max(by_size)
    p99 = int(np.percentile(list(sizes.values()), 99))
    per_class = int(os.environ.get("PROBE_PER_CLASS", "6"))

    classes: list[tuple[str, int]] = [("min:1", 1), ("median:2", 2), (f"p99:{p99}", p99), (f"max:{n_max}", n_max)]
    records = []
    for label, size in classes:
        stems = by_size.get(size, [])
        if not stems:
            continue
        # Band-spread inside the class, so a class is not accidentally all-equatorial.
        picked = [r.stem for r in sample_cogs([parse_cog_key(s) for s in stems], min(per_class, len(stems)))]
        for stem in picked:
            rec = probe_cog(stem, groups[stem])
            rec["class"] = label
            rec["band"] = band_of(parse_cog_key(stem))
            records.append(rec)
            print(
                f"  {label:>9} {stem[-16:]:>16}  n={rec['n_windows']:<3} "
                f"open {rec['open_s']:5.2f}s  read {np.sum(rec['read_s']):6.2f}s "
                f"({np.mean(rec['read_s']):5.3f}/win)  prep {np.sum(rec['prepare_s']) * 1e3:5.1f}ms"
            )
    return {"per_class": per_class, "records": records}


# --- mode: sweep --------------------------------------------------------------


def _executor(kind: str, workers: int) -> Executor:
    if kind == "thread":
        return ThreadPoolExecutor(max_workers=workers)
    if kind == "process":
        import multiprocessing as mp

        # spawn explicitly: fork would hand a child the parent's already-open GDAL
        # handles and boto3 client, which must not be shared (C3's note).
        return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    raise ValueError(kind)


def _maxrss_mb() -> float:
    """Peak RSS of this process **and** its worst child, in MB.

    ``ru_maxrss`` is bytes on Darwin and kilobytes on Linux; ``RUSAGE_CHILDREN``
    reports the max of any single reaped child, not their sum, so a process-pool
    figure has to be scaled by the worker count to get a resident total.
    """
    div = 1 << 20 if platform.system() == "Darwin" else 1 << 10
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / div
    kid = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / div
    return round(max(me, kid), 1)


def run_config(
    kind: str, workers: int, n_cogs: int, seed: int, write: bool, win_min: int = 1, win_max: int = 1 << 30
) -> dict:
    """Time one ``(executor, workers)`` config end to end over the drawn sample.

    ``write`` threads the real :class:`ShardWriter` into the loop so the sweep can be
    run both ways: the difference between them is the write path's true cost *in
    situ*, which is the only place it matters.
    """
    jobs = draw(load_selection(), n_cogs, seed, win_min, win_max)
    n_windows = sum(len(o) for _, o in jobs)
    n_blocks = sum(blocks_touched(o) for _, o in jobs)

    out_dir = SCRATCH / f"{kind}-{workers}"
    shutil.rmtree(out_dir, ignore_errors=True)
    sink = ShardWriter(out_dir) if write else None

    n_tiles = 0
    reasons: Counter[str] = Counter()
    hashes: list[int] = []

    t0 = time.perf_counter()
    if kind == "serial":
        for job in jobs:
            res = work_cog(job)
            reasons.update(res["reasons"])
            for t in res["tiles"]:
                n_tiles += 1
                if sink is not None:
                    hashes.append(sink.add(t).content_hash)
    else:
        with _executor(kind, workers) as ex:
            # ordered_futures keeps input order (so the shard row order is identical
            # to the serial path's) while bounding how many COGs' arrays are resident.
            for _job, fut in ordered_futures(ex, work_cog, jobs, in_flight=workers * 2):
                res = fut.result()
                reasons.update(res["reasons"])
                for t in res["tiles"]:
                    n_tiles += 1
                    if sink is not None:
                        hashes.append(sink.add(t).content_hash)
    elapsed = time.perf_counter() - t0
    if sink is not None:
        sink.close()
    shutil.rmtree(out_dir, ignore_errors=True)

    return {
        "executor": kind,
        "workers": workers,
        "write": write,
        "n_cogs": len(jobs),
        "n_windows": n_windows,
        "n_blocks": n_blocks,
        "n_tiles": n_tiles,
        "reasons": dict(reasons),
        "wall_s": round(elapsed, 2),
        "per_cog_s": round(elapsed / max(1, len(jobs)), 3),
        "per_window_s": round(elapsed / max(1, n_windows), 4),
        "maxrss_mb": _maxrss_mb(),
        # GDAL's block cache is per *process* and defaults to 5% of RAM. One COG wants
        # 16 blocks x 4 MB = 64 MB of it, so a thread pool shares one cache N ways while
        # a process pool gets N of them — the asymmetry this column exists to expose.
        "gdal_cachemax": os.environ.get("GDAL_CACHEMAX", "default"),
        "hash_sample": hashes[:3],
    }


def mode_sweep() -> dict:
    n_cogs = int(os.environ.get("N_COGS", "40"))
    seed = int(os.environ.get("SEED", str(config.RANDOM_SEED)))
    write = os.environ.get("WRITE", "0") == "1"
    raw = os.environ.get("CONFIGS")
    configs = (
        [(k.strip(), int(w)) for k, _, w in (c.partition(":") for c in raw.split(","))]
        if raw
        else list(DEFAULT_CONFIGS)
    )

    results = []
    for kind, workers in configs:
        # Fresh subprocess per config: no config inherits another's GDAL block cache
        # or /vsis3 file-info cache. This is the difference between measuring an
        # executor and measuring cache warmth.
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "_one", kind, str(workers), str(n_cogs), str(seed), str(int(write))],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"config {kind}:{workers} failed")
        rec = json.loads(proc.stdout.strip().splitlines()[-1])
        results.append(rec)
        print(
            f"  {rec['executor']:>7} x{rec['workers']:<3} {rec['wall_s']:7.1f}s  "
            f"{rec['per_cog_s']:5.2f} s/COG  {rec['per_window_s']:6.3f} s/win  "
            f"rss {rec['maxrss_mb']:6.1f} MB  ({rec['n_tiles']}/{rec['n_windows']} tiles)"
        )

    if os.environ.get("EXTEND") == "1" and OUT_JSON.exists():
        # Fold extra configs into the sweep already on disk rather than re-paying for
        # the 7-minute serial baseline. Keyed on cache size too, so a rerun at a
        # different GDAL_CACHEMAX is a new row, not an overwrite.
        prev = json.loads(OUT_JSON.read_text()).get("sweep", {}).get("results", [])
        key = lambda r: (r["executor"], r["workers"], r.get("gdal_cachemax", "default"))
        merged = {key(r): r for r in prev}
        merged.update({key(r): r for r in results})
        results = sorted(merged.values(), key=lambda r: (r["executor"], r["workers"]))

    base = next((r for r in results if r["executor"] == "serial"), results[0])
    for r in results:
        r["speedup_vs_serial"] = round(base["per_cog_s"] / r["per_cog_s"], 2)
    return {
        "n_cogs_per_config": base["n_cogs"],
        "n_windows_per_config": base["n_windows"],
        "seed": seed,
        "write_in_loop": write,
        "cpu_count": os.cpu_count(),
        "baseline": f"{base['executor']}x{base['workers']}",
        "results": results,
    }


# --- mode: write --------------------------------------------------------------


def mode_write() -> dict:
    """Price the local half — prepare, hash, memmap append — with no network.

    Synthetic terrain rather than real windows on purpose: this measures the *code
    path*, and its cost is a function of array size, not of what the array contains.
    Reading 50k real windows to time an append would just re-measure the network.
    """
    n = int(os.environ.get("WRITE_N", "20000"))
    rng = np.random.default_rng(config.RANDOM_SEED)
    # Plausible terrain: smooth-ish, in-range metres, land everywhere (so no tile is
    # rejected and the append path runs for all n).
    base = rng.normal(1200, 400, size=(64, config.TILE_PX, config.TILE_PX)).astype(np.float32)
    pool = np.clip(base, 1.0, 8000.0)

    t0 = time.perf_counter()
    prepared = [normalize.prepare_tile(pool[i % 64])[0] for i in range(n)]
    t_prepare = time.perf_counter() - t0

    tiles = [p.raw_i16 for p in prepared if p is not None]
    t0 = time.perf_counter()
    for t in tiles:
        content_hash(t)
    t_hash = time.perf_counter() - t0

    out_dir = SCRATCH / "write"
    shutil.rmtree(out_dir, ignore_errors=True)
    t0 = time.perf_counter()
    sink = ShardWriter(out_dir)
    for t in tiles:
        sink.add(t)
    sink.close()  # finalizes (and truncates) the trailing shard — read the count after
    t_write = time.perf_counter() - t0
    n_shards = sink.num_shards
    bytes_on_disk = sum(p.stat().st_size for p in out_dir.glob("*.npy"))
    shutil.rmtree(out_dir, ignore_errors=True)

    # add() does the hash too, so isolate the pure memmap cost.
    t_memmap = t_write - t_hash
    return {
        "n_tiles": len(tiles),
        "prepare_s": round(t_prepare, 3),
        "prepare_us_per_tile": round(t_prepare / len(tiles) * 1e6, 1),
        "hash_s": round(t_hash, 3),
        "hash_us_per_tile": round(t_hash / len(tiles) * 1e6, 1),
        "write_s": round(t_write, 3),
        "write_us_per_tile": round(t_write / len(tiles) * 1e6, 1),
        "memmap_us_per_tile": round(t_memmap / len(tiles) * 1e6, 1),
        "shards": n_shards,
        "bytes_on_disk": bytes_on_disk,
        "mb_per_s": round(bytes_on_disk / (1 << 20) / t_write, 1),
    }


# --- mode: mix ----------------------------------------------------------------


def mode_mix() -> dict:
    """Separate the per-COG and per-block costs **at the production parallelism**.

    The 50k projection can scale the 300-COG sweep wall clock directly, because that
    draw sits within 3.2% of the corpus on blocks/COG. Phase F cannot: as the budget
    grows the COG count saturates near the keep-pool's ~21k cells while blocks keep
    accumulating, so blocks/COG rises from 3.16 to ~8.3 and the *mix* moves. A single
    blended s/COG carried into that regime would misprice it in the expensive
    direction — exactly C1's error in a new costume.

    Two draws of the same size at the same config, one deliberately block-light (COGs
    holding a single window) and one block-heavy (9+ windows), give two equations::

        wall = n_cogs * c_fixed + n_blocks * c_block

    which solve for both terms. ``c_fixed`` is the open, ``c_block`` is the fetch and
    inflate of one 1024x1024 block, both already divided by whatever the pool wins.

    ⚠️ **The solve is confounded, and the recorded fit is kept only as evidence of
    that.** Window count is not independent of latitude: rugged low-latitude cells win
    many windows and polar cells win few, so the ``heavy`` draw comes out 79.7%
    equatorial against ``light``'s 40.3%. Copernicus drops columns poleward and rough
    terrain deflates worse, so the two draws do not share one bytes-per-block — the
    two-point solve blends two different ``c_block``s and then under-predicts the
    mixed draw by ~25% (54.0 s modelled vs 69-76 s measured, three runs). What the
    mode still establishes soundly is the *ranking*: 300 COGs cost 25.8 s at 1.30
    blocks/COG and 133.2 s at 8.76, so blocks — not COGs, not windows — carry the
    cost. Projections therefore use a draw whose blocks/COG **matches the tier**
    rather than this fit; ``heavy`` at 8.76 is the Phase-F-shaped one.
    """
    n_cogs = int(os.environ.get("N_COGS", "300"))
    seed = int(os.environ.get("SEED", str(config.RANDOM_SEED)))
    kind, workers = os.environ.get("MIX_CONFIG", "process:16").split(":")

    runs = []
    for label, lo, hi in (("light", 1, 1), ("heavy", 9, 1 << 30)):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "_one", kind, workers, str(n_cogs), str(seed), "0", str(lo), str(hi)],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        if proc.returncode != 0:
            print(proc.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"mix draw {label} failed")
        rec = json.loads(proc.stdout.strip().splitlines()[-1])
        rec["mix"] = label
        # Band composition is what exposes the confound, so it is recorded, not left
        # to be rediscovered: it is a pure function of the draw, no bytes involved.
        bands = Counter(band_of(parse_cog_key(st)) for st, _ in draw(load_selection(), n_cogs, seed, lo, hi))
        rec["bands"] = dict(sorted(bands.items()))
        rec["equatorial_frac"] = round(bands["0-50"] / max(1, sum(bands.values())), 3)
        runs.append(rec)
        print(
            f"  {label:>5}  {rec['n_cogs']} COGs  {rec['n_windows']:5d} win  {rec['n_blocks']:5d} blk  "
            f"({rec['n_blocks'] / rec['n_cogs']:5.2f} blk/COG)  {rec['wall_s']:7.1f}s"
        )

    # [N_a B_a][c_fixed]   [wall_a]
    # [N_b B_b][c_block] = [wall_b]
    a, b = runs
    det = a["n_cogs"] * b["n_blocks"] - b["n_cogs"] * a["n_blocks"]
    c_fixed = (b["n_blocks"] * a["wall_s"] - a["n_blocks"] * b["wall_s"]) / det
    c_block = (a["n_cogs"] * b["wall_s"] - b["n_cogs"] * a["wall_s"]) / det
    print(f"\n  solved: wall = {c_fixed:.4f} s/COG + {c_block:.4f} s/block   (at {kind}:{workers})")
    print("  [confounded — the two draws differ in band mix; see the docstring]")
    return {
        "config": f"{kind}:{workers}",
        "n_cogs": n_cogs,
        "seed": seed,
        "runs": runs,
        "c_fixed_s_per_cog": round(c_fixed, 4),
        "c_block_s_per_block": round(c_block, 4),
        "confounded": True,
        "confound": (
            "window count correlates with latitude, so the light/heavy draws are 40.3% "
            "vs 79.7% equatorial and do not share one bytes-per-block; the fit "
            "under-predicts the mixed draw by ~25%. Use mix-matched scaling instead."
        ),
    }


# --- mode: gil ----------------------------------------------------------------


def open_only(stem: str) -> float:
    """Open a COG and read nothing. The header fetch, in isolation."""
    import rasterio

    t0 = time.perf_counter()
    with rasterio.Env(**config.AWS_NO_SIGN_ENV):
        with rasterio.open(RasterioCogReader()._path(parse_cog_key(stem))):
            pass
    return time.perf_counter() - t0


def mode_gil() -> dict:
    """Why threads cap at ~2.8x when the workload looks I/O-bound.

    The first hypothesis — GDAL block-cache thrashing under a shared 819 MB cache —
    is **refuted** by the sweep: raising ``GDAL_CACHEMAX`` to 4 GB made ``thread:16``
    *slower*. The remaining candidate is a GIL-held serial fraction, and Amdahl puts
    it near 30% for a 2.8x ceiling. ``rasterio.open`` is the suspect: ``_io.pyx``
    releases the GIL around ``GDALRasterIO`` (the window reads) but the dataset-open
    path does not, and at 3.16 windows/COG the open is ~30% of serial cost.

    So: time *opens alone* serially, threaded and process-parallel. If threading
    opens gains nothing while processes scale, the serial fraction is located.
    """
    n = int(os.environ.get("GIL_N", "200"))
    stems = [s for s, _ in draw(load_selection(), n)]

    out = {}
    t0 = time.perf_counter()
    for s in stems:
        open_only(s)
    out["serial"] = round(time.perf_counter() - t0, 2)

    for kind, workers in (("thread", 8), ("thread", 16), ("process", 8)):
        t0 = time.perf_counter()
        with _executor(kind, workers) as ex:
            list(ex.map(open_only, stems))
        out[f"{kind}{workers}"] = round(time.perf_counter() - t0, 2)

    base = out["serial"]
    speedups = {k: round(base / v, 2) for k, v in out.items()}
    for k, v in out.items():
        print(f"  open-only {k:>9} {v:7.2f}s  {speedups[k]:5.2f}x")
    return {"n_cogs": n, "wall_s": out, "speedup_vs_serial": speedups}


# --- mode: determinism --------------------------------------------------------


def mode_determinism() -> dict:
    """Does a parallel run write the *same shard rows in the same order* as serial?

    #51 item 8 flags this as an open risk — "a parallel executor will not [impose
    ordering] for free". It does not have to be free, but it does have to be paid
    for once, and :func:`dataset.concurrency.ordered_futures` is where this repo
    already pays: it yields ``(item, future)`` in **input** order while keeping the
    pool fed, so the parent appends tiles in selection-list order regardless of which
    COG finished first. This mode is the demonstration rather than the assertion —
    the same bar #44 set for the selection list.
    """
    n_cogs = int(os.environ.get("N_COGS", "60"))
    seed = int(os.environ.get("SEED", str(config.RANDOM_SEED)))
    configs = [("serial", 1), ("thread", 16), ("process", 8)]

    runs = []
    for kind, workers in configs:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "_hashes", kind, str(workers), str(n_cogs), str(seed)],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        if proc.returncode != 0:
            print(proc.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"determinism config {kind}:{workers} failed")
        rec = json.loads(proc.stdout.strip().splitlines()[-1])
        runs.append({"executor": kind, "workers": workers, **rec})
        print(f"  {kind:>7} x{workers:<3} {rec['n_tiles']} tiles  digest {rec['digest'][:16]}")

    base = runs[0]
    for r in runs:
        r["matches_serial"] = r["digest"] == base["digest"]
    return {"n_cogs": n_cogs, "seed": seed, "identical": all(r["matches_serial"] for r in runs), "runs": runs}


def hashes_for(kind: str, workers: int, n_cogs: int, seed: int) -> dict:
    """The ordered content-hash sequence a run would write into the shards."""
    import hashlib

    jobs = draw(load_selection(), n_cogs, seed)
    hs: list[int] = []
    if kind == "serial":
        for job in jobs:
            hs.extend(content_hash(t) for t in work_cog(job)["tiles"])
    else:
        with _executor(kind, workers) as ex:
            for _job, fut in ordered_futures(ex, work_cog, jobs, in_flight=workers * 2):
                hs.extend(content_hash(t) for t in fut.result()["tiles"])
    digest = hashlib.blake2b(np.asarray(hs, dtype="<u8").tobytes(), digest_size=16).hexdigest()
    return {"n_tiles": len(hs), "digest": digest}


# --- mode: project ------------------------------------------------------------

B_PX = 1024  # COG internal block edge, measured (3600x3600 image => 4x4 = 16 blocks)


def blocks_touched(origins: list[tuple[int, int]], tile: int = config.TILE_PX) -> int:
    """How many 1024x1024 COG blocks a set of window origins pulls.

    This is the real unit of network cost. A 224x224 window cannot be fetched on its
    own: GDAL decompresses whole blocks, so a window costs 1, 2 or 4 blocks depending
    on how it straddles the grid — and two windows in the same block cost nothing
    extra. Windows, not COGs, is the wrong denominator for a projection.
    """
    s = set()
    for r, c in origins:
        for rr in range(r // B_PX, (r + tile - 1) // B_PX + 1):
            for cc in range(c // B_PX, (c + tile - 1) // B_PX + 1):
                s.add((rr, cc))
    return len(s)


def _keep_pool() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(stem_code, win_row0, win_col0)`` for every ``keep`` row of the gated catalog."""
    from dataset.catalog import read_catalog

    t = read_catalog(REPO / "data/catalog_gated")
    t = t.filter(t["keep"].to_numpy(zero_copy_only=False))
    stems = t["stem"].to_numpy(zero_copy_only=False)
    codes, _ = _codes(stems)
    return codes, t["win_row0"].to_numpy(), t["win_col0"].to_numpy()


def _codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, inv = np.unique(values, return_inverse=True)
    return inv.astype(np.int64), uniq


def _spread(codes: np.ndarray, r0: np.ndarray, c0: np.ndarray, idx: np.ndarray) -> tuple[int, int]:
    """Distinct COGs and distinct blocks a draw of ``idx`` rows touches."""
    sc, rr, cc = codes[idx], r0[idx], c0[idx]
    br0, br1 = rr // B_PX, (rr + config.TILE_PX - 1) // B_PX
    bc0, bc1 = cc // B_PX, (cc + config.TILE_PX - 1) // B_PX
    key = sc * 256
    ids = np.concatenate([key + br0 * 16 + bc0, key + br0 * 16 + bc1, key + br1 * 16 + bc0, key + br1 * 16 + bc1])
    return int(np.unique(sc).size), int(np.unique(ids).size)


def mode_project() -> dict:
    """Assemble the 50k and Phase-F projections from the measured pieces.

    Phase F's spread is **estimated, not assumed**: a uniform draw of the same budget
    from the gated keep-pool, *calibrated against the real 50k list* — if the uniform
    proxy reproduces Phase D's realized 15,834 COGs / 49,984 blocks, its 200k and
    500k numbers are trustworthy to the same degree; if it does not, the ratio is
    carried through as an explicit correction rather than quietly ignored.
    """
    doc = json.loads(OUT_JSON.read_text())
    groups = load_selection()
    real_cogs = len(groups)
    real_blocks = sum(blocks_touched(o) for o in groups.values())

    codes, r0, c0 = _keep_pool()
    rng = np.random.default_rng(config.RANDOM_SEED)
    tiers = {}
    for budget in (DEBUG_WINDOWS, 200_000, 500_000):
        idx = rng.choice(codes.size, size=budget, replace=False)
        n_cog, n_blk = _spread(codes, r0, c0, idx)
        tiers[str(budget)] = {"uniform_cogs": n_cog, "uniform_blocks": n_blk}

    cal = tiers[str(DEBUG_WINDOWS)]
    cog_corr = real_cogs / cal["uniform_cogs"]
    blk_corr = real_blocks / cal["uniform_blocks"]
    for v in tiers.values():
        v["cogs"] = round(v["uniform_cogs"] * cog_corr)
        v["blocks"] = round(v["uniform_blocks"] * blk_corr)

    return {
        "pool_rows": int(codes.size),
        "realized_50k": {"cogs": real_cogs, "blocks": real_blocks},
        "calibration": {"cog_correction": round(cog_corr, 4), "block_correction": round(blk_corr, 4)},
        "tiers": tiers,
        "bytes_per_tile": config.TILE_PX * config.TILE_PX * 2,
        "rows_per_shard": config.ROWS_PER_SHARD,
    }


# --- io -----------------------------------------------------------------------


def save(section: str, payload: dict) -> None:
    RESEARCH.mkdir(parents=True, exist_ok=True)
    doc = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    doc[section] = payload
    doc["box"] = {
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "bucket": config.COPERNICUS_GLO30_BUCKET,
    }
    OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\n[wrote {OUT_JSON} :: {section}]")


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "probe"

    if mode == "_one":  # internal: one sweep config, in its own process
        kind, workers, n_cogs, seed, write = argv[2], int(argv[3]), int(argv[4]), int(argv[5]), bool(int(argv[6]))
        lo = int(argv[7]) if len(argv) > 7 else 1
        hi = int(argv[8]) if len(argv) > 8 else 1 << 30
        print(json.dumps(run_config(kind, workers, n_cogs, seed, write, lo, hi)))
        return 0
    if mode == "_hashes":  # internal: one determinism run, in its own process
        print(json.dumps(hashes_for(argv[2], int(argv[3]), int(argv[4]), int(argv[5]))))
        return 0

    if mode == "probe":
        save("probe", mode_probe())
    elif mode == "sweep":
        save("sweep", mode_sweep())
    elif mode == "write":
        save("write", mode_write())
    elif mode == "mix":
        save("mix", mode_mix())
    elif mode == "gil":
        save("gil", mode_gil())
    elif mode == "determinism":
        save("determinism", mode_determinism())
    elif mode == "project":
        save("project", mode_project())
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    return 0


if __name__ == "__main__":  # spawn-safe: process workers re-import this module
    raise SystemExit(main(sys.argv))
