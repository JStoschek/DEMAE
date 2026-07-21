# Phase-C C3 — scoring throughput, and the corrected C1 cost model

<!-- Cover sheet for the C3 throughput measurement. Regenerate the sweep with
     `uv run python notebooks/c3_throughput.py`; it needs the live bucket. -->

Resolves issue [#38](https://github.com/JStoschek/DEMAE/issues/38) (part of the Phase-C
map, [#32](https://github.com/JStoschek/DEMAE/issues/32)). Corrects the cost model
[C1 (#33)](https://github.com/JStoschek/DEMAE/issues/33) recorded, and unblocks the
[#35](https://github.com/JStoschek/DEMAE/issues/35) global pass, whose code was ready
and whose *runtime* was the only thing in the way.

- Sweep: [`notebooks/c3_throughput.py`](../../notebooks/c3_throughput.py)
- Measurements: [`c3-throughput.json`](c3-throughput.json)
- Dev box: 8 logical cores (6P + 2E), US, against `copernicus-dem-30m` in
  `eu-central-1`. Provenance: snapshot `2026-07-21`, frame of 26,450 COGs.

## The problem

C1 projected the global scoring pass at **4.19 h @16 workers**. The 300-COG dry run
(2026-07-20) measured **1.87 s/COG → 13.7 h**. C1 was not wrong about per-COG *cost*;
it was wrong about *parallelism* — it measured a serial per-COG cost and divided by 16
workers, which assumes linear thread scaling. Scoring is ~98% numpy neighbourhood
compute, and that compute holds the GIL.

## The measured scaling curve

One sweep, 24 band-spread COGs per config (a prefix of the production `--sample-cogs`
draw at the same seed, so it is itself proportional to the corpus), every config on the
same machine state. Speedups are against a **measured** serial run, not an assumed one:

| executor | workers | wall | s/COG | corpus (26,450) | vs 1 process |
|---|---|---|---|---|---|
| thread | 4 | 61.4s | 2.56 | 18.79 h | 2.15x |
| thread | 8 | 54.4s | 2.27 | 16.66 h | **2.43x** |
| thread | 16 | 56.9s | 2.37 | 17.40 h | 2.33x |
| process | 1 | 132.2s | 5.51 | 40.47 h | 1.00x |
| process | 4 | 45.4s | 1.89 | 13.90 h | 2.91x |
| process | 6 | 29.2s | 1.22 | 8.94 h | 4.53x |
| process | 8 | 27.0s | 1.12 | 8.25 h | 4.90x |
| process | 12 | 26.7s | 1.11 | 8.18 h | 4.94x |
| process | 16 | 20.9s | 0.87 | 6.41 h | **6.32x** |
| process | 20 | 22.6s | 0.94 | 6.90 h | 5.87x |

Three things this settles:

1. **Threads cap at ~2.4x, and no number of them does better.** 16 threads are *worse*
   than 8. That ceiling is the whole of C1's error — its projection needed 16x.
   (Threads beat one process at all because the S3 reads release the GIL; it is the
   compute that serializes.)
2. **Processes scale, and the knee sits past the core count** — 16 workers on 8 cores,
   turning over by 20. A worker blocked on a COG read is not using a core, so mild
   oversubscription pays. This is **transport-dependent**: a same-region burst has less
   read latency to hide and its knee will sit nearer the core count. Re-run the sweep
   there rather than porting the 16.
3. **Memory is the other limit.** Measured peak RSS of a single worker is **337 MB**
   (a COG's full-res array plus its overview and the window slices), so 16 workers is
   ~5.4 GB resident.

`dataset.catalog.default_workers` encodes exactly this: 2× cores capped at 16 for
processes, 8 for threads.

## The two code changes

1. **`RasterioMaskReader` is now picklable** — the one contained blocker. `MaskCache`
   eagerly built the `s3_fetch` closure, so a worker could not receive one
   (`Can't pickle local object 's3_fetch.<locals>.fetch'`). It is built on first use —
   which its docstring already claimed — and dropped on pickle, so each spawned worker
   builds its own boto3 client by construction rather than by luck.
2. **`texture`'s median filter takes a NaN-free fast path.** `nanmedian` pays for a
   masked sort per cell; over an odd, all-finite 3×3 neighbourhood the median *is* the
   middle element by partition, which is what `median` returns. Bit-exact, **3.3x** on
   the filter (12.0 ms → 3.7 ms), and most windows take it because the ocean fill only
   appears at the coast. Same bar as the `arctan` hoist — ADR-0004 froze this vector —
   so `tests/test_metrics.py` holds both implementations against each other.

The pool is built on the **spawn** start method explicitly. macOS spawns anyway, but
the production run is a rented Linux box where the default is fork, and forking a
parent that has already opened GDAL handles and a boto3 client inherits both into a
child that must not share them.

## The re-timed pass (the deliverable)

`python -m dataset.catalog --sample-cogs 300` — the same proportional band-spread draw
as the 2026-07-20 baseline, so it extrapolates honestly:

| | executor | wall | s/COG | corpus (26,450) |
|---|---|---|---|---|
| 2026-07-20 | 16 threads | 9m19s | 1.87 | **13.7 h** |
| 2026-07-21 | 16 processes | **3m11s** | **0.64** | **4.68 h** |

49,712 rows and 0 anomalies on both runs. **2.9x**, and back inside C1's original
ballpark — by a different route: not 16 threads, and not by dividing a serial number
by a worker count.

## The frame did not move

A pass that scores faster but differently is not a faster pass, it is a different
catalog — and every downstream tier reads these values. The two 300-COG catalogs
compare **bit-identical**: 49,712 rows × 22 columns, none differing.

```python
import numpy as np
from dataset.catalog import read_catalog, COLUMNS
old = read_catalog("data/catalog_sample").sort_by("window_id")   # 2026-07-20
new = read_catalog("data/catalog_retime").sort_by("window_id")   # 2026-07-21
for c in COLUMNS:
    a, b = (t[c].to_numpy(zero_copy_only=False) for t in (old, new))
    assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
```

## What is still open

4.68 h is *this* box, US → `eu-central-1`. It is a corrected model, not a same-region
promise: on a rented same-region box the reads get cheaper, the compute does not, and
both the total and the worker knee move. Re-run `notebooks/c3_throughput.py` there
before trusting either number.
