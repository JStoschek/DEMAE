# ADR 0001 — Corpus-construction recipe (Phase 0)

- **Status:** Accepted — **partly superseded** (§4's one-read property amended by Gate A3,
  see [Amendments](#amendments); §4–5's metric basis superseded by [ADR-0004](0004-frozen-metric-vector.md))
- **Date:** 2026-07-13
- **Context ticket:** "Decide the corpus-construction recipe" (GitHub issue #4), Phase-0 wayfinder map (#1)
- **Inputs:** reconnaissance findings in [`docs/research/copernicus-glo30-recon.md`](../research/copernicus-glo30-recon.md)
- **Amended by:** Gate A3, 2026-07-19 ([issue #21](https://github.com/JStoschek/DEMAE/issues/21)) — the one-read
  property of §4 is spent; see [Amendments](#amendments)
- **Amended by:** Phase C2, 2026-07-20 ([issue #34](https://github.com/JStoschek/DEMAE/issues/34)) — §3's
  exact-zero test is **confirmed** against a per-pixel cross-check and its leak trigger closed;
  the coastline stratum it keeps is **narrowed** to sea-touching windows; see [Amendments](#amendments)
- **Superseded by:** [ADR-0004](0004-frozen-metric-vector.md), 2026-07-20 (Gate B,
  [issue #28](https://github.com/JStoschek/DEMAE/issues/28)) — §4–5's single-axis TRI
  stratification *metric basis* is replaced by the frozen 6-metric vector. The
  allocation design §5 built on the strata (quotas, geo round-robin) is redrawn by
  Phase D, not by ADR-0004.

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
(e.g. large salt-flats digitized to exactly 0).
*(Trigger fired and closed 2026-07-20 — Amendment 2: the trick is **not** leaky (11 px in
144.5 M; Salar de Uyuni scores exact-zero 0.0000), and the "keeping coastlines" clause is
narrowed — a kept coastline window must also be sea-touching.)*
Exact-zero pixels are **excluded from the
normalization statistics** (§8). Inland lakes (flat but non-zero) are legitimately kept.

### 4. Ruggedness stratification — TRI on the ×8 overview, quantile quintiles

*(Superseded 2026-07-20 by [ADR-0004](0004-frozen-metric-vector.md): the single TRI
ruggedness axis is replaced wholesale by the frozen 6-metric vector — measured against
a Spearman correlation matrix, not chosen a priori.)*

The ruggedness metric is **TRI (Riley: mean absolute elevation difference to the 8
neighbours)**, computed on the **×8 overview** and aggregated per window. TRI is preferred
over plain elevation-std because it flags genuine *roughness* (dunes, drainage texture,
canyons) rather than merely-tilted smooth planes. Computing it on the ×8 overview (~240 m
scale) is cheap and monotonic enough for *ranking*. The global candidate-window TRI
distribution is cut at **quintiles** into 5 strata (data-driven, not fixed absolute
thresholds). **One ×8-overview read per tile yields both the ocean signal (§3) and the
ruggedness score for all of that tile's windows.**
*(Superseded 2026-07-19 — Amendment 1: five metrics now require a full-resolution read at
scoring time, so scoring is no longer one read per tile.)*

### 5. Quotas & geographic spread — equal per stratum, 10° geo-cell round-robin

*(Partly superseded 2026-07-20 by [ADR-0004](0004-frozen-metric-vector.md): the
single-axis TRI quintile strata this section allocates over are gone — replaced by the
frozen 6-metric k-means vector. The **allocation design** here (quota-per-stratum, the
10° geo-cell round-robin, the per-tile window cap) is not decided by ADR-0004 either;
it is redrawn by Phase D, which is charted as its own effort.)*

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
  *(Stale as of 2026-07-19 — Amendment 1: scoring now also incurs full-resolution reads.)*
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

## Amendments

### Amendment 1 (2026-07-19) — Gate A3 spends the one-read property

- **Source:** [issue #21](https://github.com/JStoschek/DEMAE/issues/21), decided HITL against the
  committed calibration table ([#20](https://github.com/JStoschek/DEMAE/issues/20)); closed the
  Phase-A map ([#13](https://github.com/JStoschek/DEMAE/issues/13)).
- **What it supersedes:** §4's one-read property ("one ×8-overview read per tile yields both the
  ocean signal and the ruggedness score") and the Consequences cost model that follows from it.
  The original decision text above is left as written — this note records what overrode it.

**Verdict.** Five of the nine Phase-A metrics are scored at full resolution (30 m), in-region:
`slope_mean`, `slope_std`, `normals_dispersion`, `texture`, `geomorphon_entropy`. Four stay on
the ×8 overview: `local_relief`, `mean_elevation`, `elevation_relief_ratio`, and `tpi_proxy`
(whose ρ ≈ 1.0 is *by construction* — [#22](https://github.com/JStoschek/DEMAE/issues/22) scores it
on a box sized in window units, so both reads cover the same ground; it is not a survival test,
but it needs no full-res read either).

**Why the fork does not decompose — full-res is a *step* cost, not a per-metric cost.** Pulling
the 30 m pixels for a tile is the expense; computing one metric from that read versus five is
near-identical work. So the question was binary: does *any* kept metric force a full-res read at
scoring time? The gradient trio (`slope_mean`, `slope_std`, `normals_dispersion`) does — they pass
pooled at ρ ≈ 0.90 but collapse ≥50°, and per [#15](https://github.com/JStoschek/DEMAE/issues/15)
that band is 55.2% of cells, i.e. the corpus *majority*. They can't be dropped (slope is the most
fundamental terrain descriptor, and Phase A does not pre-prune — B2 owns redundancy), and they
can't stay on the overview. Once the trio buys the read, `texture` and `geomorphon_entropy` ride
along at ~zero marginal cost, which also hands B2 coherently-scored axes to prune against. This
is **one architectural decision plus four free riders**, not five independent metric judgements.

**Not settled here.** Implementing and benchmarking the full-res in-region scoring path is Phase
C1's job; a hybrid worth pricing there is restricting full-res reads to tiles ≥50° (slope passes
below 50° at ρ 0.95–0.96), at the price of a scoring-path seam at the boundary. The revised cost
model belongs with that work — this ADR only records that scoring is no longer one read per tile.

### Amendment 2 (2026-07-20) — C2 confirms the exact-zero test and narrows the coastline stratum

- **Source:** [issue #34](https://github.com/JStoschek/DEMAE/issues/34), decided HITL against a
  per-pixel cross-check ([`docs/research/ocean-gate-crosscheck.md`](../research/ocean-gate-crosscheck.md));
  Phase-C map ([#32](https://github.com/JStoschek/DEMAE/issues/32)). ADR-0004 parked this question
  for Phase C rather than deciding it on Phase-B evidence.
- **What it supersedes:** nothing in §3's *rule* — the gate is unchanged. It closes §3's own
  revisit trigger, narrows §3's "keeping coastlines" clause, and renames the window-level
  predicate. The original decision text above is left as written.

**Verdict — `WBM == 1` does not supersede the exact-zero test; it cannot.** Measured per pixel
over 2,880 windows / 144,506,880 px on 14 adversarially-chosen tiles (polders, deltas, salt flat,
fjords, atoll, inland seas at +176/−28/−430 m, ice-sheet control): **`WBM == 1` is a strict subset
of `elevation == 0.0`, with 0 counter-examples.** Copernicus paints the ocean fill and the ocean
mask from the same act, so WBM catches nothing the zero-test misses and can only ever drop *less*
(583 windows vs 590 at the 0.75 threshold; only-WBM = 0). The gate stays `exact_zero_frac ≥ 0.75`.

**§3's leak trigger is fired and closed.** Exactly-zero pixels WBM calls not-water: **11 of
144,506,880 (7.6 × 10⁻⁸)** — and none are salt flat. **Salar de Uyuni scores `exact_zero_frac`
0.0000 across all 256 of its windows.** Sub-sea-level land is negative, not zero, as §3 assumed.
This ADR should not be re-litigated on the leak question again.

**What *is* narrowed — the coastline stratum.** The 2.13 M pixels where the tests disagree are all
WBM 2/3: river and lake surfaces Copernicus flattened to exactly `0.0` because they sit at sea
level. So a high-zero-fraction window is not necessarily coastal. In the candidate band
(`exact_zero_frac ∈ [0.05, 0.75)`), 141 of 245 windows contain **no ocean pixel at all** — flat
delta water, median `land_std` **2.23 m** against **69.99 m** for the sea-touching 104. §3's
"keeping coastlines" therefore now requires **`wbm1_frac > 0`**; elevation alone provably cannot
make this call, and `wbm1_frac` is stored raw by C3 regardless, so the discriminator is free.
Band *values* remain [#36](https://github.com/JStoschek/DEMAE/issues/36)'s to lock.

**Vocabulary.** `stratify.is_ocean` → `is_zero_fill`; drop reason `"ocean"` → `"zero_fill"`.
`config.OCEAN_FILL_VALUE` keeps its name — this measurement is precisely what confirms `0.0` *is*
the ocean fill. The distinction: a **pixel** of `0.0` is ocean fill; a **window** that is 75%
zeros is not necessarily ocean.

**Coverage consequence: none.** 590 windows dropped before and after — the rule did not move.

**Not settled here.** Threshold *values* (including the coastline band edges) are C4
([#36](https://github.com/JStoschek/DEMAE/issues/36)). One incidental finding is handed to it:
WBM classifies the **Salar de Uyuni salt flat as water** despite it being genuine terrain at
3,656 m, so the default `wbm_water > 0.30` cap would drop 141 of its 256 windows — a terrain
family C4's attrition analysis must check it is not hollowing out.

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
