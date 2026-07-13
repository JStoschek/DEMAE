# ADR 0003 — Dataloader design (Phase 0)

- **Status:** Accepted
- **Date:** 2026-07-13
- **Context ticket:** "Decide the dataloader design" (GitHub issue #6), Phase-0 wayfinder map (#1)
- **Builds on:** [ADR 0001 — Corpus-construction recipe](0001-corpus-construction-recipe.md), [ADR 0002 — Storage format & manifest](0002-storage-format-and-manifest.md)

## Context

The read path for Phase-1 MAE pretraining against the corpus of ADR 0001 + 0002 (sharded
memmap **int16** NPY shards + a Parquet manifest, raw-on-disk with normalization / mask /
channel-dim deferred to load time). Pretraining is **single-GPU** — target hardware is an
**RTX 5090 with a local NVMe cache** — with **random-access shuffle each epoch**. The #1
project risk is I/O starving the GPU. The stored form is deliberately decision-neutral so the
plan's **per-tile-vs-global normalization ablation** is a loader-config flip. MAE random token
masking lives in the Phase-1 model, **not** the loader.

## Decision

### 1. Read plumbing — map-style Dataset, lazy per-worker memmaps, global shuffle

- **Map-style `torch.utils.data.Dataset`** over the manifest rows (Parquet loaded once into memory
  as the index). `__len__` = row count; `__getitem__(i)` resolves `manifest[i] → (shard_id,
  row_index) → memmap[shard_id][row_index]`.
- **Lazy per-worker `np.memmap` handles.** Memmaps are **not** opened in `__init__` (that would
  fork/pickle them into every worker). Each worker opens a shard's memmap on first access and caches
  the handle in a per-worker dict keyed by `shard_id`. Fork-safe; O(1) after warmup.
- **Global `DataLoader(shuffle=True)`.** memmap gives true O(1) random access, so we get a *perfect*
  per-epoch shuffle for free — no WebDataset-style shuffle buffers (a stated reason ADR 0002 chose
  memmap).
- **Perf flags:** a few `num_workers`, `pin_memory=True`, `persistent_workers=True`, default
  `prefetch_factor`. The Dataset returns **fp32 CPU tensors**; bf16 autocast is a Phase-1 model concern.

Rejected: an `IterableDataset` with sharded workers — unnecessary given random-access memmap and it
would reintroduce approximate shuffling.

### 2. Sample schema — a dict

`__getitem__` returns:

```
{
  "image":     (C, H, W) float32,   # normalized; C = 1 in Phase 0
  "land_mask": (H, W) bool,         # True = land, False = exact-zero ocean/void
  "mean":      float32,             # per-tile land stats (for inversion / logging)
  "std":       float32,
  "tile_id":   str,
  "stratum":   int,                 # ruggedness quintile (for stratified metrics)
}
```

A dict (over a bare tensor / `(image, mask)` tuple) so Phase 1 can optionally mask ocean from the
loss, invert reconstructions back to metres for the viz grids, and log per-tile / per-stratum. Default
collate batches these cleanly (strings → list). `stratum` is included for stratified metrics /
optional curriculum.

### 3. Normalization — a pluggable strategy (the ablation hook)

A `Normalizer` selected by loader config, applied after the int16→fp32 cast:

- **`per_tile`** (default): `(x − mean) / std` using the sample's **manifest** stats (ADR-0001
  per-tile standardization).
- **`global`**: `(x − μ) / σ` using **corpus-wide** land-pixel stats read from `corpus_meta.json`.

Flipping the config *is* the plan's per-tile-vs-global normalization ablation — no re-materialization.
Per-patch `norm_pix_loss` (original MAE) is a **Phase-1 model/loss** concern and is explicitly out of
the loader.

### 4. Mask & channel layout

- **Mask** derived **before** normalization as `land_mask = (raw != 0)` (the zeros move once
  standardized). Normalization is applied to *all* pixels (ADR 0001); the mask rides alongside as
  metadata for optional loss-masking.
- **Channel-ready:** image is `(C, H, W)` with **C = 1** (`["elevation"]`) in Phase 0. A channel list in
  config + a compute-and-stack transform yields `(3, H, W)` when slope/curvature are added (Phase 3),
  with the model's patch-embed `C` configurable. The loader owns adding the channel axis.

### 5. Throughput — reasoned, not benchmarked (per this ticket's decision)

On the target 5090 + NVMe, the loader has comfortable headroom:

- **Demand:** a ViT-S/16 MAE step on a 5090 (~1.5–2× a 4090; encoder sees only ~25% of tokens) is
  ~30–60 ms per batch-256 → the loader must sustain ~**4–8k samples/s**.
- **Per-sample cost:** memmap slice (~100 KB) + int16→fp32 cast + affine normalize + `!=0` mask over
  ~50k elements ≈ a few hundred µs single-thread → thousands/s per worker, tens of thousands/s across a
  few workers — 1–2 orders of magnitude over demand.
- **Disk is not the bottleneck:** the 50k debug corpus is ~5 GB int16 and **fits in RAM**, so after
  epoch 1 the OS page cache serves reads at RAM speed. Cold, 8k×100 KB = 800 MB/s is trivial for NVMe.
  At Phase-3 scale (500k ≈ 50 GB > RAM) reads hit NVMe but stay well inside its random-read envelope.

**Conclusion: no GPU starvation.** The synthetic memmap micro-benchmark (fabricate int16 shards +
manifest, measure single-thread and multi-worker samples/s) remains available for the build ticket to
run against real shards if empirical confirmation is wanted.

## Consequences

- **Handoff to build (#7):** the pipeline must compute and store corpus-wide **land-pixel μ/σ** in
  `corpus_meta.json` so the `global` normalization mode works. The Dataset/loader code is small enough
  to land in the same `dataset/` module (`loader.py`).
- **Design spine complete:** ADRs 0001–0003 fully specify the corpus, its storage, and its read path —
  the build ticket (#7) now has a settled spec, and nothing else in Phase 0 requires a decision.
- **Reproducibility:** global shuffle uses the DataLoader generator seed; the corpus itself is pinned by
  ADR 0002's `dataset_version`.

## Alternatives considered

- **`IterableDataset` / sharded-worker streaming** — rejected; random-access memmap makes it
  unnecessary and it weakens shuffle quality.
- **Bare-tensor or `(image, mask)` return** — rejected in favour of the dict (needs stats/id/stratum).
- **Baking normalization into storage** — already rejected in ADR 0002; would kill the ablation.
- **Benchmarked throughput** — deferred to an optional build-time check; reasoned headroom suffices here.
