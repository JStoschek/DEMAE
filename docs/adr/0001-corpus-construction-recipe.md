# ADR 0001 — Corpus-construction recipe (Phase 0)

- **Status:** Accepted
- **Date:** 2026-07-13
- **Context ticket:** "Decide the corpus-construction recipe" (GitHub issue #4), Phase-0 wayfinder map (#1)
- **Inputs:** reconnaissance findings in [`docs/research/copernicus-glo30-recon.md`](../research/copernicus-glo30-recon.md)

## Context

Phase 0 must turn the global Copernicus DEM GLO-30 collection into a terrain-diverse,
per-tile-normalized corpus of 224×224 elevation tiles for ViT-MAE pretraining. The
reconnaissance surfaced the constraints that shape every choice below: 26,454 released
1° tile prefixes; **columns shrink poleward** (3600→2400→1800→1200→720→360, rows always
3600); **ocean is filled with exactly `0.0` and there is no no-data sentinel**, but
genuine land can be zero-or-negative (Netherlands −12 m); values are float32 metres over
the **EGM2008** geoid (a DSM); COGs carry ×2/×4/×8 overviews and support cheap windowed
reads. Standing constraint: **elevation-only output, architected channel-ready** (adding
slope/curvature later is a config flag, not a rewrite).

## Decision

### 1. Tiling geometry — pixel-square windows, no reprojection, no latitude cap

224×224 **pixel** windows are the unit. We accept the mild ground anisotropy rather than
reprojecting or capping latitude. Copernicus's poleward longitude coarsening keeps GSD
roughly constant (~24–31 m E–W across bands; ~30.9 m N–S), so the corpus is effectively
single-resolution ~30 m — matching the plan's assumption. Worst pixel anisotropy is
~1.55:1 near 50° (not at the pole); per-tile standardization and shape-learning absorb it.
Reprojection is unjustified cost/artifacts for a debug corpus in a phase *before*
GSD-aware encoding; a hard latitude cap would discard glaciated/tundra terrain diversity
to solve a problem masking + std-guard + stratification already handle. The **manifest
records each window's latitude band** so Phase-4 GSDPE is a config change, not a re-tile.

### 2. Window layout — non-overlapping grid, over-generate then stratify-subsample

Each COG is gridded into `floor(W/224)×floor(H/224)` non-overlapping windows with the last
row/column **flushed to the tile edge** (no coverage gap, no partial windows). Geometry
only *enumerates and scores* candidates across all 26,454 tiles; the stratifier (§4–5)
performs the actual selection to 50k. Deterministic and reproducible (a Phase-0 virtue),
with no near-duplicate train tiles. Overlap/jitter is rejected — it buys diversity we don't
need at 50k debug scale and costs determinism.

### 3. Land masking — exact-zero fraction ≥ 0.75 drops the window

Ocean is discriminated by **exact `0.0`** pixels (Copernicus's ocean fill; genuine land is
almost never exactly `0.0`, and sub-sea-level land is negative). A window is dropped iff its
**exact-zero fraction ≥ 0.75**, which discards mostly-ocean windows while **keeping
coastlines** (the plan wants shoreline structure oversampled). Fully-ocean and
withheld-country cells are already absent prefixes (free). No external coastline polygon —
stays value-based and channel-free; revisit only if the exact-zero trick proves leaky
(e.g. large salt-flats digitized to exactly 0). Exact-zero pixels are **excluded from the
normalization statistics** (§8). Inland lakes (flat but non-zero) are legitimately kept.

### 4. Ruggedness stratification — TRI on the ×8 overview, quantile quintiles

The ruggedness metric is **TRI (Riley: mean absolute elevation difference to the 8
neighbours)**, computed on the **×8 overview** and aggregated per window. TRI is preferred
over plain elevation-std because it flags genuine *roughness* (dunes, drainage texture,
canyons) rather than merely-tilted smooth planes. Computing it on the ×8 overview (~240 m
scale) is cheap and monotonic enough for *ranking*. The global candidate-window TRI
distribution is cut at **quintiles** into 5 strata (data-driven, not fixed absolute
thresholds). **One ×8-overview read per tile yields both the ocean signal (§3) and the
ruggedness score for all of that tile's windows.**

### 5. Quotas & geographic spread — equal per stratum, 10° geo-cell round-robin

Target distribution is **equal quota per stratum** (~10k each → 50k). Because the natural
distribution is flat-dominated, uniform-across-strata automatically oversamples rugged
terrain while capping flat and preserving a representative flat tail. Knob: skew quotas
toward upper strata if a sampled preview looks too flat. Geographic spread is enforced by
tagging each window with a **10°×10° lon/lat cell**, applying a **per-1°-tile window cap**,
and **filling each stratum's quota round-robin across geo-cells** so rugged samples span
many ranges rather than clustering (e.g. all Himalaya). Explicit climate/tectonic regime
labels are deferred to a possible Phase-3 refinement (an external dependency, marginal gain
at debug scale). The 10° cell id is recorded in the manifest for the Phase-2 geographic
hold-out (the hold-out *unit* itself remains a Phase-2 decision).

### 6. Normalization — per-tile standardization, land-only stats, no percentile clip

Per-tile standardization (subtract mean, divide std). **Statistics are computed over land
pixels only** (exact-zero excluded), then applied to *all* pixels so the window stays a
coherent image (ocean pixels land at a meaningful large-negative z). **No percentile
clipping** — a considered **departure from plan §2.4.4**: Copernicus is radar-sourced
(TanDEM-X), so its extremes are almost all *real terrain* (on the Everest tile `max` is
~2,500 m above p99 — the actual summit), and percentile-clipping would flatten exactly the
high-relief signal the corpus exists to capture. Only a conservative **absolute plausibility
clip to [−500 m, 9000 m]** is applied, to catch fill/artifact garbage without touching real
terrain. Windows with **land-std < ε** are dropped as degenerate. (Percentile clipping
becomes relevant for 3DEP lidar in Phase 4.)

### 7. Per-tile outputs

Each materialized tile emits: the **normalized elevation** channel (channel-ready layout),
a **land/validity mask** (1 = land, 0 = exact-zero ocean/void — future-proofs Phase-1
loss-masking and the Phase-2 void-filling eval), and the stored **mean, std, and clip
bounds** for inverting reconstructions.

## Consequences

- **Pipeline cost** ≈ **26,454 ×8-overview reads (scoring)** + **~50,000 full-resolution
  224² reads (materialization)** — the concrete workload the build ticket (#7) must
  measure, and the basis for the local-vs-cloud-burst decision (still fog on the map).
- **Manifest fields required** (handed to #5 "storage format + manifest"): source COG,
  window origin, lon/lat, **latitude band**, **10° geo-cell id**, **TRI value + stratum**,
  **mean/std + clip bounds**, and a reference to the **land mask**.
- **Smoke-tier size** (a subset of cells/tiles run end-to-end before the full 50k) remains
  fog, sharpened once the pipeline module exists.
- **Reproducibility:** the grid + quantile strata + fixed round-robin order are
  deterministic; any tie-breaking uses a fixed seed.
- **Honesty note for the model card:** the corpus has coverage gaps over a few withheld
  countries (absent prefixes) and a deliberate rugged-oversampled, terrain-diverse (not
  geographically uniform) distribution.

## Alternatives considered

- **Reproject windows to a metric/equal-area grid** — rejected: resampling cost + artifacts
  + a target-GSD choice, unjustified before Phase-4 GSD-aware encoding.
- **Hard latitude cap (≤60°)** — rejected: discards terrain diversity to solve a problem
  masking + std-guard + stratification already address.
- **Random / jittered windows** — rejected: near-duplicates and non-determinism at debug scale.
- **Elevation-std as the ruggedness metric** — rejected in favour of TRI (std flags tilted
  smooth planes as rugged).
- **Rugged-skewed quotas** — held as a knob, not the default.
- **External regime labels / coastline polygon** — deferred; unnecessary dependencies at
  debug scale.
- **Per-tile percentile clipping (plan §2.4.4)** — rejected for the radar-sourced GLO-30
  corpus; would flatten real peaks. Absolute plausibility clip only.
