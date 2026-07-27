# ADR 0003 — Dataloader design (Phase 0)

- **Status:** Accepted — §3 **amended** (nothing superseded); see [Amendments](#amendments)
- **Date:** 2026-07-13
- **Context ticket:** "Decide the dataloader design" (GitHub issue #6), Phase-0 wayfinder map (#1)
- **Builds on:** [ADR 0001 — Corpus-construction recipe](0001-corpus-construction-recipe.md), [ADR 0002 — Storage format & manifest](0002-storage-format-and-manifest.md)
- **Amended by:** Phase F, 2026-07-27 ([issue #70](https://github.com/JStoschek/DEMAE/issues/70)) — §3's
  `global` statistic is **tier-specific**, so the ablation is within-tier only; and the mode's measured
  effect on the signal is recorded. See [Amendments](#amendments)

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
  *(Qualified 2026-07-27 — Amendment 1: "corpus-wide" means **this** corpus. Each tier accumulates
  its own μ/σ, so the pair is not portable across corpora.)*

Flipping the config *is* the plan's per-tile-vs-global normalization ablation — no re-materialization.
*(Qualified 2026-07-27 — Amendment 1: the flip is a **within-tier** comparison; two tiers under
`global` normalize against different constants.)*
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

## Amendments

### Amendment 1 (2026-07-27) — Phase F pins the `global` statistic as tier-specific

**Source:** Phase-F wayfinder map ([#67](https://github.com/JStoschek/DEMAE/issues/67)), ticket
[#70](https://github.com/JStoschek/DEMAE/issues/70). **Supersedes:** nothing — §3's two modes stand
exactly as decided. This adds a *constraint* §3 left implicit and a *characterization* it never made.

#### The decision: `tier_specific`

Every corpus tier accumulates its **own** corpus-wide land-pixel μ/σ from its own tiles, as
`dataset.materialize` already does in one streaming pass. The scale tier does **not** inherit the
50k debug tier's `951.1396…` / `1012.9799…`.

The consequence, which §3 did not say out loud: **the per-tile-vs-global ablation is a within-tier
comparison only.** Two tiers under `global` mode normalize against different constants, so a
50k-`global` run and a 500k-`global` run differ in their denominator as well as their data and are
not a controlled comparison of anything.

That costs nothing, because no cross-tier comparison was ever wanted.
[ADR-0005 §6](0005-sampling-design.md) already settled that the tiers are independent draws — "the
debug tier is a fast validation corpus, not a scaling ablation" — and
[the project plan](../../DEMAE_Project_Plan.md) schedules the normalization ablation in Phase 3 on
the scale corpus. The rejected branch, `inherit_debug` (freeze the debug tier's pair project-wide),
would have bought exactly the cross-tier comparison ADR-0005 disavowed, and charged for it by
normalizing every future corpus against a statistic measured on a different one.

Both branches were **pre-registered before the scale corpus existed**, in
[`f-gate-bar.json`](../research/f-gate-bar.json); this ticket *selected* one. Gate F's `F8` gates the
recorded values against the shipped tiles.

#### The characterization: what `global` mode does to the signal

`global` mode is well-posed — the μ/σ pair is perfectly well-defined — but its *effect* is worth
stating, because it is surprising enough to be misread as a bug. Measured over all 49,998 tiles of
the debug tier, normalizing by its own `μ = 951.14`, `σ = 1012.98`:

| quantity | value |
|---|---|
| median tile's within-tile relief (per-tile `std`) | 23.8 m → **0.024** normalized |
| p75 / p95 | 70.7 m → 0.070 / 214.2 m → 0.211 |
| tiles whose normalized `std` < 0.10 | **82%** |
| between-tile spread ÷ within-tile spread — `std(mean) / median(std)` | **42×** |

So under `global`, a tile's *absolute altitude* carries roughly **42×** the variance of its
*terrain shape*. Under a pixel-MSE MAE objective the loss is dominated by predicting the tile's mean
elevation, and landform structure — the thing DEMAE exists to learn — is a small residual on top.

This is a legitimate ablation arm, not a defect: `per_tile` deliberately discards absolute altitude
and `global` deliberately keeps it, and the plan's expectation is already that per-tile wins. It is
recorded here so that **Phase 1 does not read a normalized tile with `std ≈ 0.02` as a loader
bug**, and so the ablation's likely outcome has a stated mechanism rather than a post-hoc one.

**Per-stratum μ/σ was considered and rejected here.** It would compress the between-tile term
towards the within-tile one and is the obvious third option — but the k=20 strata are a *label, not
a rank* (ADR-0005), a per-stratum normalizer makes reconstructions invertible only against a
stratum assignment, and the ablation this ADR exists to enable is the plan's two-arm one. It is
recorded as a considered alternative, **not opened as a knob**; wanting it is a new decision.

#### Handoff

`corpus_meta.json` gains a top-level `normalization_stat_source` (`"tier_specific"`), written by
`manifest.write_corpus_meta` from `config.NORMALIZATION_STAT_SOURCE`. It sits **outside
`build_params`**, so it does not enter `dataset_version` — the same slot and the same reasoning as
`attrition_contract`. Under `tier_specific` the pair is a pure function of the tiles the fingerprint
already hashes, so nothing is lost; and a `build_params` key would have changed *every* corpus's
`dataset_version`, including the debug tier's `6fde6a58…`, which the committed E2 report and gallery
quote and ADR-0006/0007 rest on. A corpus card written before this amendment has no such key and is
`tier_specific` by construction.

## Alternatives considered

- **`IterableDataset` / sharded-worker streaming** — rejected; random-access memmap makes it
  unnecessary and it weakens shuffle quality.
- **Bare-tensor or `(image, mask)` return** — rejected in favour of the dict (needs stats/id/stratum).
- **Baking normalization into storage** — already rejected in ADR 0002; would kill the ablation.
- **Benchmarked throughput** — deferred to an optional build-time check; reasoned headroom suffices here.
