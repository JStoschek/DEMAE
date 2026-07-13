# ADR 0002 — Storage format & tile-manifest schema (Phase 0)

- **Status:** Accepted
- **Date:** 2026-07-13
- **Context ticket:** "Decide storage format + tile-manifest schema" (GitHub issue #5), Phase-0 wayfinder map (#1)
- **Builds on:** [ADR 0001 — Corpus-construction recipe](0001-corpus-construction-recipe.md)

## Context

The corpus produced by ADR 0001 must be persisted for Phase-1 pretraining, which is
**single-GPU**, reads a **pre-materialized local cache**, and needs **random-access shuffle
each epoch**. The #1 project risk is **I/O starving the GPU** (MAE's encoder only sees 25%
of tokens, so the loader must be fast). The format must scale to 200–500k tiles in Phase 3.
Tiles are small (224×224) and, per ADR 0001, elevation-only but channel-ready. The plan also
wants an **early per-tile-vs-global normalization ablation**, which the stored form must not
foreclose.

## Decision

### 1. Storage format — sharded memmap NPY

Tiles are written into fixed-size **`.npy` shards** (~1–2 GB each) and read via `numpy.memmap`,
giving **O(1) random access** by `(shard_id, row_index)` with near-zero overhead — the fastest
answer to the I/O-starvation risk, and no approximate-shuffle compromise. Dependencies are just
numpy + pyarrow. Sharding keeps transfer, parallel materialization, and checkpointing manageable
and scales cleanly to 500k.

The **writer sits behind a thin interface** so a **WebDataset** backend can be added later *if*
Phase 3 goes multi-node or pure-object-store streaming — not built now.

Rejected: **WebDataset** (optimized for sequential streaming / multi-node we don't have;
approximate buffer shuffle); **Zarr** (extra dependency, little gain over memmap for a local
single-GPU cache); **GeoTIFF crops** (many small files, slow; geo-semantics already normalized away).

### 2. Stored payload — raw int16-metre elevation, single channel

Each shard is an `(rows_per_shard, 224, 224)` array of **int16 metres**: the ADR-0001
plausibility-clipped raw elevation, rounded to the nearest metre.

- **int16 metres, not fp32.** Storage dtype is independent of the training compute dtype (the
  loader casts int16 → bf16/fp32 for free). Given Copernicus's **±4 m** vertical accuracy at 30 m
  GSD, sub-metre decimals are mostly noise, not signal; int16 metres is effectively lossless w.r.t.
  real terrain information and is the DEM-world convention (SRTM, ASTER GDEM ship as int16 metres).
  It **halves storage** (≈5 GB vs 10 GB at 50k; 50 GB vs 100 GB at 500k, where it matters).
- **Not fp16:** fp16 quantises relative to magnitude (~8 m step near 8000 m — wrecks high mountains);
  int16 quantises uniformly at 1 m, the correct failure profile for elevation.
- **Quantisation-banding guard:** uniform 1 m quantisation can faintly band the flattest tiles after
  standardisation. This is neutralised by tightening ADR-0001's degenerate-drop threshold to
  **ε ≥ 3 m** (`std < ε` → drop), so no *kept* tile is coarse enough to band. *(This concretises the
  ε left open in ADR 0001.)*
- **Raw, not normalized.** Storing raw + per-tile stats in the manifest makes the plan's
  per-tile-vs-global normalization ablation a **loader-config change, not a re-materialization**.
  Normalization is a loader transform (§ Handoff).
- **Mask not stored — recomputed on load** as `raw == 0` (ADR-0001's exact-zero ocean fill; kept
  windows are <75% ocean). The raw array is the single source of truth. An explicit stored mask is a
  forward hook, added only if a future invalidity source appears that isn't exact-zero (e.g. flagged
  artifact voids).
- **Single channel on disk;** the channel axis (C=1 → 3 later) is added loader-side.

Phase-4 note: the 3DEP 1 m lidar corpus has real sub-metre signal and would choose fp32 (or
int16-decimetre) via the same per-corpus storage interface. Out of scope here.

### 3. Manifest — single Parquet + `corpus_meta.json`

A **single Parquet** file indexes the whole corpus (columnar, typed, compact, loads in one shot;
50k–500k rows is a few MB). A **`corpus_meta.json`** sidecar holds corpus-level constants and the
reproducibility fingerprint. SQLite (no indexed queries needed — the loader loads all rows) and CSV
(untyped) rejected.

**Per-row schema:**

| field | type | purpose |
|---|---|---|
| `tile_id` | str | stable id — deterministic `{cog_stem}_{row0}_{col0}` |
| `shard_id` | int | which `.npy` shard |
| `row_index` | int | position in the shard memmap — **the storage join** |
| `source_cog` | str | provenance (COG key) |
| `win_row0`, `win_col0` | int | window pixel origin in the source COG |
| `lon`, `lat` | float | window-centre coordinates |
| `lat_band` | str | resolution band (`0-50`, `50-60`, …) — Phase-4 GSDPE |
| `geocell_id` | str | 10° lon/lat cell — geographic spread + Phase-2 hold-out |
| `tri` | float | ruggedness score (ADR 0001 §4) |
| `stratum` | int | ruggedness quintile 0–4 |
| `mean`, `std` | float | per-tile land-only stats — loader normalization + inversion |
| `land_frac` | float | fraction of land pixels — QC / auditing |
| `content_hash` | uint64 | 64-bit hash (xxh3 / blake2b-8) of the tile's raw int16 LE bytes — integrity + near-duplicate detection |
| `split` | str? | null; Phase-2 fills the geographic hold-out |

**`corpus_meta.json` (corpus card):** normalization method, tile size, storage dtype, clip bounds
`[−500, 9000]`, ε, per-stratum quotas, random seed, Copernicus snapshot date, build timestamp, code
version, and **`dataset_version`** — a blake2b fingerprint over `(schema version + sorted
(tile_id, content_hash) + build params)` identifying this exact corpus build for reproducibility and
the model card.

## Consequences

- **Loader path (handoff to #6):** load Parquet → for sample *i*, read `(shard_id, row_index)` from the
  memmap → cast int16 → bf16/fp32 → apply `(x − mean)/std` (or an alternate norm for the ablation) →
  derive mask `raw == 0` → add channel axis.
- **Integrity:** `content_hash` per row + `dataset_version` fingerprint let any consumer verify the
  shards match the manifest and cite an exact build.
- **Ablation-ready:** raw-on-disk + stats-in-manifest keep per-tile-vs-global normalization a config flip.
- **Scale:** ~5 GB (50k) / ~50 GB (500k) of shard data; manifest a few MB. Fine on NVMe.
- **Couples back to ADR 0001:** fixes the degenerate-drop threshold at **ε ≥ 3 m**.

## Alternatives considered

- **WebDataset / Zarr / GeoTIFF** storage — rejected (see §1).
- **fp32 or fp16 payload** — rejected in favour of int16 metres (see §2).
- **Storing normalized values** — rejected; would foreclose the normalization ablation.
- **Storing an explicit mask** — deferred; exactly derivable as `raw == 0` today.
- **SQLite / CSV manifest** — rejected in favour of Parquet.
