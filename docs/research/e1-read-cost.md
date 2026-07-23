# Phase-E E1 — the full-resolution materialization pass, priced

<!-- Cover sheet for the E1 read-pass measurement. Regenerate with
     `uv run python notebooks/e1_read_cost.py {probe,sweep,gil,mix,write,determinism,project}`;
     probe/sweep/gil/mix/determinism need the live bucket. -->

Resolves issue [#49](https://github.com/JStoschek/DEMAE/issues/49) (part of the Phase-E
map, [#46](https://github.com/JStoschek/DEMAE/issues/46)). Unblocks
[#51](https://github.com/JStoschek/DEMAE/issues/51), which has to choose an executor.

- Harness: [`notebooks/e1_read_cost.py`](../../notebooks/e1_read_cost.py)
- Measurements: [`e1-read-cost.json`](e1-read-cost.json)
- Dev box: 8 logical cores (6P + 2E), 17.2 GB, macOS arm64, US, against
  `copernicus-dem-30m` in `eu-central-1` — the same box C1/C3 ran on.
- Sample: the committed [`selection_debug_50k.csv`](../../artifacts/phase_d/selection_debug_50k.csv),
  band-spread over its own 15,834 COGs by `dataset.sampling.sample_cogs`, taking every
  selected window of each drawn COG.

## The answer

**The 50k debug tier costs ~1 hour.** 16 processes, ~2.9 GB resident, 5.02 GB on disk
in 4 shards. Phase F is affordable too and by a wider margin than its budget suggests:
**10x the windows costs ~2.5x the wall clock.** No fallback is required at any tier;
one is named below anyway, with the number that would trigger it.

| tier | COGs | blocks | windows/COG | wall @ process:16 | disk | shards |
|---|---|---|---|---|---|---|
| **50k (debug)** | 15,834 | 49,984 | 3.16 | **1.03–1.06 h** | 5.02 GB | 4 |
| 200k (Phase F) | 19,930 | 128,962 | ~10 | 1.81–2.46 h | 20.07 GB | 14 |
| 500k (Phase F) | 21,154 | 176,051 | ~24 | 2.48–2.61 h | 50.18 GB | 34 |

## What the cost unit actually is

The ticket asks what batching window reads per COG buys. The premise needed correcting
before it could be answered: **a 224² window is not a thing that can be fetched.**

Copernicus GLO-30 COGs are 3600×3600 float32, deflate-compressed, internally tiled at
**1024×1024** — sixteen blocks per COG, ~4 MB each uncompressed, objects of 20–49 MB on
the wire. GDAL cannot serve a 224² window without fetching and inflating every block it
touches. So the unit of network cost is the **block**, and two windows in one block cost
what one does.

An instrumented serial pass (`probe`, 13 COGs across four size classes) decomposes to:

```
read_seconds ≈ 0.197 + 0.241 × blocks_touched          (R² = 0.932)
open_seconds ≈ 0.427                                    (flat; independent of COG size)
```

That model, applied to the whole selection list, predicts 6.09 h serial; the 300-COG
sweep measured 6.03–6.22 h. Two independent routes, one number.

### So what does batching buy?

Within a COG, a lot — and at corpus scale, almost nothing:

| windows in COG | COGs | mean blocks | **blocks/window** | measured s/window |
|---|---|---|---|---|
| 1 | 5,437 | 1.34 | 1.34 | ~0.27 |
| 2 | 3,607 | 2.43 | 1.21 | ~0.30 |
| 3–4 | 3,675 | 3.68 | 1.09 | — |
| 5–8 | 2,203 | 5.61 | 0.93 | — |
| 9–16 | 770 | 8.29 | 0.74 | ~0.17 |
| 17+ | 142 | 11.60 | 0.54 | 0.066 (the max-44 COG) |
| **whole 50k** | **15,834** | **3.16** | **1.000** | — |

The max-44 COG really is a different animal from the median-2 one — 4.5x cheaper per
window. It is also 142 COGs out of 15,834. Because Phase D's draw is *median 2*, the
corpus lands at **49,984 blocks for 50,000 windows**: batching is still worth doing (it
saves the repeated `open`, which is 30% of serial cost) but it does **not** deliver a
per-window discount at this spread. It will at Phase F, where blocks/window falls to
0.35 — which is exactly why Phase F is sub-linear.

## The executor: processes, and it is not close

One sweep, 300 band-spread COGs (1,124 windows, 977 blocks) per config, **each config in
a fresh subprocess** so none inherits another's GDAL block cache or `/vsis3` file-info
cache. Speedups are against a **measured** serial run.

| executor | workers | GDAL cache | wall | s/COG | s/window | peak RSS | vs serial |
|---|---|---|---|---|---|---|---|
| serial | 1 | default | 424.6s | 1.42 | 0.378 | 287 MB | 1.00x |
| thread | 4 | default | 174.7s | 0.58 | 0.155 | 414 MB | 2.43x |
| thread | 8 | default | 153.9s | 0.51 | 0.137 | 461 MB | **2.76x** |
| thread | 16 | default | 167.0s | 0.56 | 0.149 | 488 MB | 2.54x |
| thread | 16 | 4 GB | 184.6s | 0.61 | 0.164 | 630 MB | 2.30x |
| thread | 24 | default | 230.9s | 0.77 | 0.205 | 652 MB | 1.84x |
| thread | 32 | default | 320.5s | 1.07 | 0.285 | 658 MB | 1.32x |
| thread | 32 | 4 GB | 277.1s | 0.92 | 0.247 | 608 MB | 1.53x |
| thread | 48 | default | 412.3s | 1.37 | 0.367 | 835 MB | 1.03x |
| process | 4 | default | 130.1s | 0.43 | 0.116 | 260 MB | 3.26x |
| process | 8 | default | 87.8s | 0.29 | 0.078 | 221 MB | 4.83x |
| process | 16 | default | 71.8s | 0.24 | 0.064 | 180 MB | **5.92x** |
| process | 24 | default | 71.5s | 0.24 | 0.064 | 154 MB | 5.95x |
| process | 32 | default | 64.3s | 0.21 | 0.057 | 154 MB | **6.61x** |

**C3's conclusion did not invert.** The ticket expected it might — E1 is a windowed read
with trivial compute, the mirror image of the scoring pass's 98%-numpy profile. It
behaves the same anyway: threads cap at **2.76x** (C3: 2.43x), processes reach **6.61x**
(C3: 6.32x), and threads past 16 are actively destructive — 48 threads is serial speed.

### Why threads fail, in two experiments

**Hypothesis 1 — GDAL block-cache thrashing. Refuted.** GDAL's cache is per *process* and
defaults to 5% of RAM = **819 MB** here. One COG wants 16 × 4 MB = 64 MB of it, so ~12.8
concurrent COGs fit — suspiciously close to where the thread curve turns over. But
raising `GDAL_CACHEMAX` to 4 GB made `thread:16` *slower* (167.0s → 184.6s) and left
`thread:32` far short. Not the mechanism.

**Hypothesis 2 — a GIL-held serial fraction. Confirmed, and located.** Amdahl puts the
serial fraction near 33% for a 2.76x ceiling, and `open` is 0.427s of a 1.42 s/COG serial
cost — 30%. Timing *opens alone*, reading nothing (`gil` mode, 200 COGs):

| | wall | vs serial |
|---|---|---|
| serial | 93.90s | 1.00x |
| 8 threads | 67.54s | 1.39x |
| 16 threads | 108.66s | **0.86x** — slower than one thread |
| 8 processes | 13.61s | **6.90x** |

`rasterio.open()` is effectively GIL-serialized. rasterio releases the GIL around
`GDALRasterIO` — the window reads — but not around the dataset-open path, and at a mean
of 3.16 windows/COG the open is a third of the work. This is **structural, not tunable**:
no thread count and no cache size fixes it, and its share *grows* as windows/COG falls.

Two consequences worth carrying forward. Phase F, at ~10–24 windows/COG, dilutes the open
and would let threads do relatively better — but processes still win, so it changes
nothing. And a same-region box shrinks the *reads* while leaving the GIL-bound open
untouched, so the serial fraction rises there and processes matter **more**, not less.

### The recommendation

**`process`, 16 workers** — which is exactly what `dataset.catalog.default_workers("process")`
already returns on this box (2× cores, capped at 16). The driver reuses it; nothing new
to encode.

`process:32` buys a further ~10% (64.3s vs 71.8s) for ~2 GB more resident. Not worth
complicating the driver for a 6-minute saving on a 1-hour pass, but it is the first
escape hatch if the pass is slower than projected.

**Memory: ~2.9 GB at 16 workers** (16 × 179.8 MB measured peak per worker — the figure
falls with worker count because each holds one COG's blocks, not a share of the corpus).
Comfortable on 17.2 GB, and below C3's 5.4 GB for its 16 process workers.

## The write path is free

`prepare_tile` + `content_hash` + `ShardWriter.add`, measured with no network in the loop
(20,000 tiles, synthetic terrain — the cost is a function of array size, not content):

| step | per tile | 50k total |
|---|---|---|
| `prepare_tile` (clip → int16 → land stats) | 296.6 µs | 14.8 s |
| `content_hash` (blake2b over 100 KB) | 233.1 µs | 11.7 s |
| memmap append + trailing-shard finalize | 251.4 µs | 12.6 s |
| **write path total** | **484.5 µs** | **24.2 s** |

**39.1 s against a ~3,700 s read pass — 1.05%.** At 197 MB/s the memmap is never the
constraint. Keep the writer serial in the parent, as the harness does: it costs nothing
and it is what makes shard row order deterministic.

## Determinism under parallelism — settled here, not deferred

[#51](https://github.com/JStoschek/DEMAE/issues/51) item 8 flags that "shard *ordering* is
only fixed if the driver imposes it, and a parallel executor will not do that for free."
It does not have to be free, only paid for once, and this repo already paid:
`dataset.concurrency.ordered_futures` yields `(item, future)` in **input** order while
keeping the pool fed. Over 60 COGs / 199 tiles, hashing the full ordered sequence of
`content_hash` values:

| | tiles | digest |
|---|---|---|
| serial | 199 | `c3a65e607a89a37b` |
| 16 threads | 199 | `c3a65e607a89a37b` |
| 8 processes | 199 | `c3a65e607a89a37b` |

Byte-identical. Demonstrated, not asserted — the bar
[#44](https://github.com/JStoschek/DEMAE/issues/44) set. The driver gets deterministic
shard ordering at any executor and worker count for free, provided it consumes through
`ordered_futures` and writes in the parent.

## Phase F, and why it is sub-linear

The number the map cares about: whether the scale tier is affordable on this box at all.
It is, comfortably.

Phase F's spread is **estimated, not assumed**. A uniform draw of each budget from the
gated keep-pool (3,052,735 rows), **calibrated against the real 50k list** — the proxy
predicts 17,150 COGs / 58,958 blocks where Phase D actually realized 15,834 / 49,984, so
corrections of 0.923 and 0.848 are carried through to the larger tiers rather than
ignored.

| budget | COGs | blocks | blocks/window | wall | disk | shards |
|---|---|---|---|---|---|---|
| 50,000 | 15,834 | 49,984 | 1.000 | 1.03–1.06 h | 5.02 GB | 4 |
| 200,000 | 19,930 | 128,962 | 0.645 | 1.81–2.46 h | 20.07 GB | 14 |
| 500,000 | 21,154 | 176,051 | 0.352 | 2.48–2.61 h | 50.18 GB | 34 |

**10x the windows, ~2.5x the time.** COG count saturates near the keep-pool's ~21k cells
while blocks/window falls from 1.000 to 0.352 — the batching discount that the 50k
tier's median-2 spread cannot collect, Phase F collects in full.

Each tier is projected by scaling a *measured* draw whose blocks/COG matches it (the
mixed 300-COG draw at 3.26 for 50k; a deliberately block-heavy draw at 8.76 for Phase F),
bracketed by scaling on COGs and on blocks. The 50k bracket is tight (1.03–1.06 h)
because its reference draw sits within 3.2% of the corpus on blocks/COG; 200k's is wider
because its mix (6.47) falls between the two reference draws.

### A model that did not work, and why

`mix` mode tries to solve `wall = n_cogs·c_fixed + n_blocks·c_block` from two same-sized
draws of different density. It returns `0.0240 s/COG + 0.0479 s/block` at `process:16` —
and **that fit is confounded; it is recorded as evidence, not used.** It under-predicts
the mixed draw by ~25% (54.0 s modelled against 69.2 / 71.8 / 76.4 s measured).

The confound: window count is not independent of latitude. Rugged low-latitude cells win
many windows, polar cells win few, so the block-heavy draw comes out **79.7% equatorial**
against the block-light draw's **40.3%**. Copernicus drops columns poleward and rough
terrain deflates worse, so the two draws do not share one bytes-per-block, and a
two-point solve blends two different `c_block`s.

What the mode does establish soundly is the ranking, and it is stark: the same 300 COGs
cost **25.8 s** at 1.30 blocks/COG and **133.2 s** at 8.76. Blocks carry the cost — not
COGs, not windows. Hence mix-matched scaling above, rather than a fitted constant.

## The fallback, named

**It is not needed.** 1.03–1.06 h for the debug tier and 2.48–2.61 h for a 500k scale
tier are both ordinary attended runs on this box. Named anyway, in the order to reach for
them, with the number that triggers each:

1. **`process:32`** — measured, +10% (64.3s vs 71.8s on the sweep sample), costs ~2 GB
   more resident. *Trigger: the realized pass exceeds ~1.5 h.*
2. **Re-run the sweep before trusting anything else.** *Trigger: the realized pass exceeds
   **2.1 h** (2× projection).* Something about the transport has changed and the knee has
   moved with it; C3's lesson is that the next number should be measured, not reasoned to.
3. **A same-region rented box** (`eu-central-1`) — would graduate a provisioning ticket.
   Block fetches are 56% of serial cost and are pure transatlantic latency plus bandwidth,
   so this is the large lever. *Trigger: only Phase F, and only if (1) and (2) leave it
   above ~6 h.* Re-measure there: the reads get cheaper, the GIL-bound open does not, so
   the serial fraction **rises** and the process knee moves — porting the 16 would repeat
   C1's error.
4. **An accepted long unattended run.** At these numbers, moot.

Note what is *not* on this list: reducing the tier, loosening a gate, or dropping windows.
Nothing about the cost justifies touching the sampling design.

## Two findings for neighbouring tickets

**Attrition ([#50](https://github.com/JStoschek/DEMAE/issues/50)).** Across 1,124 real
full-res windows, `prepare_tile` rejected exactly **one**, as `degenerate` (land-std < 3 m).
Zero `zero_fill`, zero `no_land` — Phase C's gates are doing their job upstream. That is
**~0.09%**, extrapolating to roughly 45 windows of the 50k. The contract still has to be
decided, but it is being decided for a handful of rows, not for a hole in a stratum.

**Resumability ([#51](https://github.com/JStoschek/DEMAE/issues/51) item 5).** The ticket
budgets for "a multi-hour S3 pass that dies at 80%". At ~1 hour, a total restart costs an
hour. Per-COG checkpointing remains the natural granularity — the COG *is* the unit of
work — but the case for making a half-written shard a resume point is now weak, and #51
can scope accordingly.

## What is still open

These are this box, US → `eu-central-1`, on one afternoon. Repeat runs of the same config
varied 69.2–76.4 s (±5%), so treat the projections as ±10%, not as promises. The
executor conclusion is robust to that — processes beat threads by 2.4x, far outside the
noise — but the hours are not precise to two digits. Re-run
`notebooks/e1_read_cost.py sweep` on any other box before trusting its numbers there.
