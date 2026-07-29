# ADR 0002 — Storage format & tile-manifest schema (Phase 0)

- **Status:** Accepted — §3a's `split` row **amended** (nothing superseded); see [Amendments](#amendments)
- **Date:** 2026-07-13
- **Context ticket:** "Decide storage format + tile-manifest schema" (GitHub issue #5), Phase-0 wayfinder map (#1)
- **Builds on:** [ADR 0001 — Corpus-construction recipe](0001-corpus-construction-recipe.md)
- **Amended by:** Phase 1, 2026-07-28 ([issue #77](https://github.com/JStoschek/DEMAE/issues/77)) — the
  reserved `split` column is filled for the **scale tier** in Phase **1**, not Phase 2, and the
  assignment ships as a committed artifact. See [Amendments](#amendments)

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

---

## Addendum §3a — Metric-vector manifest schema (Phase E)

- **Status:** Accepted
- **Date:** 2026-07-23
- **Context ticket:** "Redraw the manifest schema for the metric-vector corpus" (GitHub issue #48), Phase-E wayfinder map (#46)
- **Amends:** §3 (per-row manifest schema) only. §1–2 (sharded int16 memmap storage) are untouched. `MANIFEST_SCHEMA_VERSION` bumps **1 → 2**.

### Why §3 had to move

§3 fixed the per-row schema in the **TRI era**. Phase D retired that path wholesale (#44): selection now runs on the **frozen 6-metric vector** (ADR-0004) clustered into **k = 20 k-means strata** (#41), and the selection is a committed list (#44), not a seeded quintile scheme. So two v1 columns had become lies — `tri` (a float nothing computes any more) and `stratum` (declared as a *ruggedness quintile 0–4*, but the live label is a k=20 id). This addendum records the honest v2 schema. The original §3 stays above as the historical TRI-era record.

### The redrawn schema (v2)

Per-row columns, in on-disk order. "Source" is where the driver gets each value: **materialization** computes it from the full-res read; **catalog** copies it from `data/catalog_gated/` by window id; **selection** copies it from the Phase-D selection list (`artifacts/phase_d/selection_debug_50k.csv`).

| column | type | source | note |
|---|---|---|---|
| `tile_id` | str | materialization | `{cog_stem}_{row0}_{col0}` |
| `shard_id`, `row_index` | int32 | materialization | storage join |
| `source_cog` | str | catalog | provenance |
| `win_row0`, `win_col0` | int32 | window id | window origin |
| `lon`, `lat` | float64 | catalog | window-centre |
| `lat_band` | str | catalog | resolution band |
| `geocell_id` | str | catalog | 10° cell |
| `mean`, `std`, `land_frac` | float64 | materialization | per-tile land stats (full-res) |
| `stratum_id` | int32 | **selection** | k=20 label — **renamed** from `stratum` |
| `provenance` | str | **selection** | `{base, floor:coastline}` (#42), copied unchanged |
| `slope_mean` | float64 | catalog | frozen metric (raw, untransformed) |
| `local_relief` | float64 | catalog | frozen metric |
| `texture` | float64 | catalog | frozen metric |
| `elevation_relief_ratio` | float64 | catalog | frozen metric |
| `mean_elevation` | float64 | catalog | frozen metric |
| `geomorphon_entropy` | float64 | catalog | frozen metric |
| `edm_edited_frac` | float64 | catalog | edited-pixel fraction |
| `flm_fill_frac` | float64 | catalog | fill-mask fraction |
| `wbm_water_frac` | float64 | catalog | aggregate water fraction |
| `water_affected` | bool | catalog | any water in window |
| `coastline` | bool | catalog | coastline window |
| `content_hash` | uint64 | materialization | integrity / near-dup |
| `split` | str? | reserved | **stays declared-and-null** this tier; Phase-2 fills the geographic hold-out |

**Dropped:** `tri`. It is not withheld data (which the map keeps as a column); it is a dead concept that will never be repopulated, so a null column would be a fossil. **Renamed:** `stratum` → `stratum_id` — reusing the name in place with a changed meaning is how a schema quietly lies; the new name is the one #44 already ships upstream. **Copied-column names are the catalog's verbatim** (`edm_edited_frac`, `flm_fill_frac`, `wbm_water_frac`), not the plan's `edm_frac`/`flm_frac`/`wbm_frac` shorthand, so the manifest lines up column-for-column with its source.

### Denormalize, not join

Every added column already exists per-window in the gated catalog, joinable by window id. The manifest **denormalizes** them — copies the values in — rather than carrying a join key. Rationale: the catalog is a 4.4M-row **sampling frame that outlives this tier** (Phase F reuses it), while the manifest describes **this 50k build**; a corpus consumer or the model card should read one self-contained file, not be forced to possess and keep-in-sync a giant candidate frame to interpret a 50k corpus. This also matches how §3 already treated `mean`/`std`/`land_frac`/`lat_band`/`geocell_id` — per-window facts denormalized in. Size is a non-issue (a few MB either way at 50k). The 6 metric values are the **raw catalog values** (ADR-0004's `ln(x+1)` transforms are k-means-only), so E2.1a compares corpus-vs-pool like-for-like.

### The fingerprint pins the selection, not the seed

`build_params()` (which feeds `dataset_version` and is embedded verbatim in `corpus_meta.json`) **drops the bare `seed`** — #43/#44 established the seed is *not* the contract, the persisted selection list is — and gains four Phase-D fingerprints:

- `selection_list_md5` — the core pin: one hash fixes *which* windows, their `stratum_id`, and their `provenance`, since all three columns are copied from that file.
- `catalog_version` — pins the catalog the denormalized metric/mask columns were copied from; nothing else in the fingerprint covers those (the tile `content_hash` only covers raw elevation bytes).
- `centroids_md5`, `scaler_md5` — redundant on the hash (the selection list already pins the labels), recorded for a self-documenting Phase-D provenance chain in the model card.

### Consequences

- **Loader:** the `stratum` → `stratum_id` rename touches the loader — it reads that column and re-exposes it in the training sample. That is a two-line rename (`src/dataset/loader.py`), not a read-path redesign; the sample key is likewise `stratum_id`.
- **Self-contained corpus:** the manifest + `corpus_meta.json` now fully describe the build with no dependency on the catalog at consumption time.
- **`split` reserved:** kept declared, typed `string`, nullable, written null for the whole 50k tier — an honest forward hook (Phase 2 will fill it), unlike the dropped `tri`. *(Phase F, 2026-07-28: the hook now carries a **measured hazard**. At 500 k the draw takes 23% of a source COG's inventory on average, with 2,788 COGs at half or more and 423 drawn dry, so a random train/val split puts neighbouring windows on both sides. [`f-honesty-notes.md`](../research/f-honesty-notes.md) §4.6 tells downstream users to split on `geocell_id` or `source_cog` until Phase 2 fills this column.)*

## Amendments

### Amendment 1 (2026-07-28) — the `split` column is filled in Phase 1, for the scale tier

**Source:** Phase-1 wayfinder map ([#76](https://github.com/JStoschek/DEMAE/issues/76)), ticket
[#77](https://github.com/JStoschek/DEMAE/issues/77). **Supersedes:** nothing about the *schema* —
`split` is still a nullable `string`, still the last column, and the type never moved. What changes
is the sentence attached to it: §3 and §3a both say "Phase-2 fills the geographic hold-out", and
that is now wrong about **who** and **when**.

#### What changed, and why the phase moved

The column is filled by `dataset.splits` in **Phase 1**, before pretraining rather than after it,
because a split that arrives in Phase 2 arrives too late to be honest. The Phase-1 plan needs a
held-out `val` pool for masked-recon loss and the kNN probe *during* pretraining (§10), and it needs
`test` regions that pretraining provably never saw (§14) — both of which require the assignment to
exist before the first run, not after. Phase 2 remains the *consumer* of the `test` cells; it is no
longer their author.

**Only the scale tier is filled.** `data/corpus_500k/manifest.parquet` carries `train` / `val` /
`test`; the 50k debug tier's column stays null, because nothing has asked it for a split and a
second assignment would be a second contract to keep in step.

#### What does not change

- **`dataset_version` does not move.** It hashes schema version + sorted `(tile_id, content_hash)` +
  `build_params` (§3), none of which the column touches. The shipped `bcbd6484…` is byte-identical
  across the fill, checked before and after.
- **The hazard §3a recorded stands** — it is the *reason* for the design, not a thing the fill
  retires. The split is geographically blocked precisely because the corpus card's §4.6 numbers say
  a random one leaks. The card's advice to downstream users (split on `geocell_id` or `source_cog`)
  is now discharged by the shipped column rather than left to them.
- **The artifact, not the code, is the contract** — as for the selection list and the ledger.
  `artifacts/phase1/split_scale_500k.csv` (`tile_id, split`, md5 in `split_blocks_500k.json`) is what
  later phases re-read; `dataset.splits` re-running is a check, not the source of truth.

The design, the judgement calls it required, and the two plan claims it corrected are argued in
[`p77-split.md`](../research/p77-split.md).
