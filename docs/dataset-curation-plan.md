# DEMAE dataset curation plan (v2 corpus)

A phased, gated plan for building the DEMAE pretraining corpus, synthesizing the three
verified research reports:

- [`dem-corpus-construction-deep-research.md`](research/dem-corpus-construction-deep-research.md) — strategy: curation beats size; masks; feasibility
- [`dem-corpus-curation-recipe.md`](research/dem-corpus-curation-recipe.md) — mechanics: FSCS, Neyman, quotas, dedup, validation
- [`tile-metric-selection.md`](research/tile-metric-selection.md) — metrics: candidate set, refuted redundancies, smoke-tier pruning pass

**Relation to existing ADRs.** This plan *keeps* ADR-0001 §1–3 and §6–8 (tiling geometry,
non-overlapping grid, exact-zero land mask, per-tile standardization, outputs) and
ADR-0002 wholesale (int16 shards, Parquet manifest). It *supersedes* ADR-0001 §4–5
(TRI-quintile strata, equal quotas) with a metric-vector + k-means + Neyman design, and
adds EDM/FLM/WBM quality gating. Phase B ends by writing **ADR-0004** to record exactly
that supersession — nothing is silently overridden. The read budget is relaxed per the
2026-07-16 decision: full-res scoring on an in-region box is an accepted fallback if
Gate A3 fails; ADR-0004 notes this.

**Design stance.** Every contested choice is a **knob** with a default, an evidence tag
(`[verified]` = survived 3-vote verification; `[project]` = our call, no precedent), and
an explicit amendment trigger (§9). Strong defaults, cheap to change.

---

## Phase A — Foundations: masks, metric library, calibration

*Goal: everything the scoring pass needs, unit-tested, before touching all 26k tiles.*

**A1. Acquire the EDM/FLM/WBM quality masks.**
They are **not** in `s3://copernicus-dem-30m` — fetch from the ESA/Copernicus
distribution (PRISM/panda or the public GLO-30 mask packages) `[verified]`.
- Deliverable: `src/dataset/masks.py` — resolve + read the three masks for a given 1°
  cell; cache locally (masks are byte rasters, small).
- Also deliverable: **prevalence measurement** — on a ~500-tile global sample, the
  fraction of pixels per EDM code and FLM source. This number decides whether strict
  filtering is affordable (open question in report 1) and calibrates the Phase-C
  thresholds before they're set.
- Fallback if mask access is painful: WBM-only gating (water is the dominant artifact)
  and defer EDM/FLM columns to the scale tier — recorded as a knob, not a silent skip.

**A2. Metric library — all eight candidates, pure numpy/scipy, unit-tested.**
`src/dataset/metrics.py` (extends `prototypes/terrain_stratify_proto.py`), one function
per metric, each testable on synthetic surfaces from `src/dataset/surfaces.py`:

> **Correction (issue #16).** This section originally cited `src/dataset/synthetic.py`
> for the test surfaces. That was a slip: `synthetic.py` is the fabricated
> Copernicus-shaped S3 bucket for zero-network pipeline tests. The oracle surfaces
> live in a new `src/dataset/surfaces.py` and `synthetic.py` is untouched.

| metric | formula sketch | unit test oracle |
|---|---|---|
| local relief | `max − min` | ramp of known range |
| slope mean, slope std | gradient magnitude with **anisotropic dx/dy per latitude band** `[verified caveat]` | inclined plane → exact slope, zero std |
| elevation-relief ratio (ERR ≡ hypsometric integral) | `(mean−min)/(max−min)`, guard `max==min` | plane → 0.5; plateau-with-pit → high |
| Iwahashi–Pike texture | `median_filter(z,k) − z` → sign threshold → pit/peak density, ~10-cell circular kernel `[verified]` | sinusoid frequency sweep |
| normals dispersion (spherical std of unit normals) | 1 − ‖mean normal‖ style | any plane → 0 regardless of tilt |
| geomorphon class entropy *(candidate only)* | 10-class histogram → entropy | synthetic ridge/valley |
| mean elevation | `mean` (land-only, already computed) | trivial |
| TPI-proxy *(held in reserve)* | window mean − large-neighborhood mean on overview | synthetic basin |

All metrics take an array + pixel-size pair, so they run identically on full-res windows
and ×8 overviews. Exact-zero pixels are excluded from all statistics (ADR-0001 §3).

**A3. Overview-vs-full-res calibration.** On ~100 tiles spanning latitude bands and
terrain types: compute every metric per window at full res and from the ×8 overview;
report Spearman rank correlation per metric.
- **Gate A3:** ρ ≥ 0.9 → overview scoring stands (relief expected to pass: <5% drift
  30→90 m `[verified]`). Any metric below 0.9 → either drop it or invoke the accepted
  full-res scoring fallback for the global pass — an explicit decision, not a default.

*Definition of done: metrics library green in CI; masks readable; calibration table +
prevalence table committed to `docs/research/`.*

## Phase B — Smoke-tier empirical study (the pruning pass)

*Goal: convert every unresolved redundancy question into a measured answer. This phase
exists because the near-collinearity of slope/relief/roughness was **refuted, not
confirmed** `[verified]` — nobody has measured it at 6.7 km; we will.*

**B1. Score a pilot frame.** ~10–20k candidate windows, spread round-robin across 10°
geo-cells (reuse ADR-0001 §5 machinery), all eight candidate metrics + mask fractions.

**B2. Correlation-prune (the Argudo workflow `[verified]`).** Spearman matrix + PCA over
the pilot frame; a notebook committed as the auditable artifact.
- Prune rule: within any block at |ρ| ≥ 0.9 keep one representative `[project]`.
- Expected outcome (to be confirmed, not assumed): keep **one of {relief, slope mean}**,
  **slope std** if it separates, **ERR**, **texture**, **mean elevation**; admit
  **normals dispersion** and **geomorphon entropy** only if independent *and*
  non-degenerate (VRM's documented failure mode is a ~97%-"level" distribution
  `[verified]`).
- Also check each metric's distribution for degeneracy and decide log-transforms here
  (ln slope and ln(relief+1) per the Iwahashi precedent `[verified]`).

**B3. Flat-wet sanity check.** The TPI-proxy/flat-wet axis is *demoted to open*, not
resolved `[verified: no surviving evidence]`. Test: do the surviving metrics separate
hand-picked flat-wet exemplars (Mississippi floodplain, Pantanal, playa, delta) from
flat-dry plateaus in descriptor space? If not → add the large-window TPI-proxy back to
the vector. This is the explicit re-entry trigger for TPI. The labelled contrast set —
4 flat-wet + 4 flat-dry, scored through the B1 frame (#25) — is committed at
[`research/exemplars.md`](research/exemplars.md)
([CSV](research/exemplars-per-window.csv)).

**B4. Freeze + record.**
- Freeze the metric vector (target 4–6 dims — the global-classification lineage never
  exceeds ~4 clustering inputs `[verified]`).
- Write **ADR-0004: corpus metrics & sampling v2** — supersedes ADR-0001 §4–5, records
  the measured correlation matrix as evidence, the frozen vector, the sampling design of
  Phase D, and the relaxed read budget.

- **Gate B:** ADR-0004 accepted. No global scoring until the vector is frozen.

## Phase C — Global scoring pass (the candidate catalog)

*Goal: one score per candidate window across all 26,454 COGs; the reusable sampling
frame for every corpus tier.*

**C1. Enumerate + score.** Existing `discovery.py`/`geometry.py` grid (ADR-0001 §2,
unchanged: non-overlapping, edge-flushed). Per window, from the ×8 overview (or full-res
if Gate A3 forced it): exact-zero fraction, frozen metrics, EDM/FLM/WBM fractions,
lat band, 10° geo-cell. Parallel COG reads already exist (`build.py`).

**C2. Quality gates** (all fractions stored regardless, so gates are re-runnable
filters, not destructive):
- ocean: exact-zero ≥ 0.75 drop (ADR-0001 §3, unchanged);
- degenerate: land-std < 3 m drop (ADR-0002, unchanged);
- **water: WBM water fraction > 0.30 drop** `[project — no published precedent exists]`;
  flattened water reads as spurious zero-texture, so windows *kept* with water > 0.05
  get a `water_affected` flag excluded from texture-based stratification distances;
- **edited/filled: EDM-edited or FLM-non-TanDEM fill > 0.25 drop** `[project]`,
  tuned against the A1 prevalence table so we know what coverage the threshold costs
  (steep forested regions concentrate voids — verify no stratum is hollowed out);
- coastline windows (kept, high-zero-fraction) tagged as their own reportable stratum
  rather than silently mixed in `[project]`.

**C3. Persist the catalog.** One Parquet (~a few million rows × ~20 cols — small). This
is the **sampling frame**: debug and scale tiers both draw from it without re-scoring.

*Definition of done: catalog on disk; summary stats notebook (global metric
distributions, gate attrition counts, coverage map).*

## Phase D — Stratify, allocate, select (the sampler)

*Goal: pick 50k window ids (debug tier) from the catalog, deterministically.*

**D1. Strata.** z-score the frozen (log-transformed) metric vector over the surviving
catalog `[verified: standardize, don't whiten]`; **k-means, k = 40** strata
(15/40 published precedent `[verified]`; 40 = knob). Fixed seed, persisted centroids.

**D2. Allocation of the 50k budget across the 40 strata.**
- Base: **Neyman** — n_h ∝ N_h·σ_h (σ_h = mean within-stratum distance to centroid)
  `[verified: variance-minimizing rule]`. Automatically caps huge flat strata without
  deleting them (don't over-prune flat terrain `[verified]`).
- **Floor:** max(500, 1% of budget) per stratum `[project]`, so no terrain family is
  absent (Prithvi's minimum-presence precedent `[verified]`).
- **Boosts** (Prithvi quota template `[verified]`): top-up pools for (a) the rarest
  strata by N_h, (b) high-descriptor-entropy windows (heterogeneous terrain), (c) the
  coastline stratum — sizes are knobs, default ~5% of budget each `[project]`.
- Retained from ADR-0001 §5: per-1°-tile window cap and **round-robin fill across 10°
  geo-cells within each stratum**, so rugged quota spans many mountain ranges.

**D3. Within-stratum selection: FSCS.** Sub-k-means with k = n_h inside each stratum;
take the window nearest each sub-centroid `[verified: Brus mechanics]`; geo-cell
round-robin as tie-breaker.
- **Dedup:** the non-overlapping grid already guarantees zero spatial overlap — strictly
  stronger than SSL4EO's ¾-width rule `[verified]`. No hashing/embedding dedup: no
  precedent for DEMs, and FSCS already suppresses look-alikes within strata. Revisit
  only if jittered windows are ever introduced.

**D4. Emit the selection list** (window ids + stratum + allocation provenance), seeded
and reproducible; `stratify.py` is the module this replaces/extends.

## Phase E — Materialize & validate (debug tier, 50k)

**E1. Materialize.** Full-res 224² reads for selected windows → plausibility clip
[−500, 9000] m → int16 shards + manifest (ADR-0001 §6–7, ADR-0002 — all unchanged;
`build.py`/`shards.py`/`manifest.py` mostly as-is). **New manifest columns:** the frozen
metric values, `stratum_id`, `edm_frac`, `flm_frac`, `wbm_frac`, `water_affected`,
`coastline`, selection provenance (floor/boost/base). Schema addendum to ADR-0002.

**E2. Validation report** (the Brungard & Boettinger method `[verified]` — a committed
notebook, re-runnable per tier):
1. per-metric ECDF/histograms: corpus vs full candidate pool (graphical
   distribution-match);
2. chi-square: per-stratum counts vs intended allocation; hard check every floor met;
3. geographic coverage map + per-geo-cell counts;
4. a **visual gallery: 12 random tiles per stratum** — the human backstop that k-means
   strata are terrain-meaningful;
5. attrition table: what each quality gate discarded, by stratum (detects gate bias —
   e.g. edited-pixel gate hollowing out steep forest).
- **Gate E:** report reviewed; knobs adjusted and D–E re-run if a stratum is empty,
  a gallery looks wrong, or gate attrition is lopsided. D–E re-runs are cheap (catalog
  and shards decouple scoring from selection).

**E3. Record honesty notes** for the model card (extends ADR-0001's list): DSM/canopy
bias (land cover drives error more than slope `[verified]`), edited-pixel policy +
thresholds, withheld-country gaps, deliberate terrain-diverse (not geo-uniform)
distribution, high-latitude pixel anisotropy policy.

## Phase F — Scale tier (200–500k) and deferred items

- Re-run **D with a bigger budget** against the same catalog (same strata/centroids for
  comparability; re-clustering is a knob). Published plateau evidence says 200–500k
  crops is comfortably adequate `[verified]` — spend margin on curation, not more tiles.
- **Vegetation-cover column** (e.g. a global canopy product sampled per window) —
  deferred to scale tier; at debug scale the `water/edm` columns carry the quality
  story. Trigger to promote: Phase-2 probes show canopy-biased reconstructions.
- **Geographic hold-out**: region (10° cell) hold-out reserved in the manifest since
  ADR-0001; the split itself remains a Phase-2 decision (no verified precedent for EO
  holdout design — our call).
- **Anisotropy metric**: deferred with an explicit trigger — revisit if downstream tasks
  involve oriented fabrics (dunes, drumlins) `[no verified evidence at 6.7 km]`.

---

## 9. Knob table (what's amendable, and what would change it)

| knob | default | evidence | amendment trigger |
|---|---|---|---|
| metric vector | B2 survivors of {relief, slope μ, slope σ, ERR, texture, elev μ, (+normals disp, +geomorphon entropy)} | verified axes; composition measured in B2 | correlation matrix says otherwise |
| TPI-proxy | out | no surviving evidence | B3 flat-wet check fails |
| k (strata) | 40 | 15/40 precedent | silhouette/gallery shows over/under-split |
| allocation | Neyman + floor + boosts | Neyman verified; floor/boost sizes project | validation report imbalance |
| WBM water cap | 0.30 drop / 0.05 flag | none published — project | A1 prevalence + attrition table |
| EDM/FLM cap | 0.25 drop | none published — project | same |
| scoring source | ×8 overview | calibration A3 | any metric ρ < 0.9 → full-res in-region pass |
| dedup | grid-guaranteed zero overlap | SSL4EO precedent exceeded | jittered/overlapping windows introduced |
| debug budget | 50k | plateau evidence: ample | — |
| vegetation column | deferred | verified concern, no debug-scale need | canopy-biased Phase-2 probes |

## 10. Order of work (dependency chain)

A1 ∥ A2 → A3 → B1 → B2/B3 → **B4 (ADR-0004)** → C1 → C2/C3 → D → E1 → **E2 (gate)** → E3 → F.
The long-pole external dependency is A1 (ESA mask access) — start it first; everything
in A2–A3 proceeds in parallel without it.

## 11. Principal risks

1. **Mask access friction (A1)** — mitigated by the WBM-only fallback knob and by
   storing fractions as columns (gates re-runnable later).
2. **k-means strata not terrain-meaningful** — mitigated by the gallery review (E2.4)
   and the k knob; worst case, fall back to the Iwahashi 40-cluster global product as an
   external stratum label (published, downloadable).
3. **Quality gates hollowing out steep-forest strata** — attrition-by-stratum table
   (E2.5) catches it; loosen EDM cap per-stratum if needed.
4. **Overview scoring bias at high latitude** (anisotropic pixels + band discontinuities
   above 50°) — anisotropic dx/dy in A2 + lat-band column + A3 calibration sample
   includes high-lat tiles.
5. **DSM/canopy bias is unfixable at source** `[verified]` — not filtered away, only
   documented (E3); revisit vegetation column at scale tier.
