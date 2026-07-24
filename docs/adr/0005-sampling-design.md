# ADR 0005 — Sampling design: stratification, allocation, and within-stratum selection (Phase D / Gate E)

- **Status:** Accepted
- **Date:** 2026-07-24
- **Context ticket:** "Write and accept ADR-0005 — the sampling design" ([issue #68](https://github.com/JStoschek/DEMAE/issues/68)) — the **Phase-F wayfinder map** ([#67](https://github.com/JStoschek/DEMAE/issues/67)). First chartered as [#56](https://github.com/JStoschek/DEMAE/issues/56) on the Phase-E map and closed unfinished by Gate E's FAIL; see [Why this ADR waited](#why-this-adr-waited-twice).
- **Supersedes:** what remains of ADR-0001 §4–5 after [ADR-0004](0004-frozen-metric-vector.md) took the *metric basis* — the **strata construction** (TRI quintile cuts → k=20 k-means) and the **allocation design** (equal quota per stratum, the skew-quotas knob, the per-1°-tile window cap). The 10° geo-cell round-robin **survives, modified** — see [§8](#8-what-this-supersedes-in-adr-0001-precisely).
- **Depends on:** [ADR-0004](0004-frozen-metric-vector.md) (the frozen 6-metric vector — the descriptor space this stratifies); [ADR-0002 §3a](0002-storage-format-and-manifest.md) (the v2 manifest schema that consumes `stratum_id` and `provenance`)
- **Decision inputs (all committed):** [#40](https://github.com/JStoschek/DEMAE/issues/40) the frame · [#41](https://github.com/JStoschek/DEMAE/issues/41) k · [#42](https://github.com/JStoschek/DEMAE/issues/42) allocation ([`d2-allocation.md`](../research/d2-allocation.md), [`d2-allocation-nh.json`](../research/d2-allocation-nh.json)) · [#43](https://github.com/JStoschek/DEMAE/issues/43) FSCS mechanics + cost · [#45](https://github.com/JStoschek/DEMAE/issues/45) no nesting · [#44](https://github.com/JStoschek/DEMAE/issues/44) the sampler ([`d4-sampler.md`](../research/d4-sampler.md), [`d4-selection.json`](../research/d4-selection.json))
- **Validating evidence:** [`e2-validation-report.md`](../research/e2-validation-report.md) ([#52](https://github.com/JStoschek/DEMAE/issues/52)) — 7 build-integrity criteria pass, no Defect; [`e2-gallery.md`](../research/e2-gallery.md) ([#53](https://github.com/JStoschek/DEMAE/issues/53)) — **20/20 strata coherent**; the Gate E verdict ([#54](https://github.com/JStoschek/DEMAE/issues/54)), **including its FAIL** — see [§7](#7-the-evidence-this-design-is-accepted-on--including-the-part-that-failed)
- **Artifacts this design ships:** [`artifacts/phase_d/scaler.json`](../../artifacts/phase_d/scaler.json), [`artifacts/phase_d/centroids.json`](../../artifacts/phase_d/centroids.json), [`artifacts/phase_d/selection_debug_50k.csv`](../../artifacts/phase_d/selection_debug_50k.csv); code [`src/dataset/select.py`](../../src/dataset/select.py), [`src/dataset/centroids.py`](../../src/dataset/centroids.py)

> **Scope in one line.** This ADR records the design that turns a gated candidate catalog into a **selection list**: which rows k-means fits, how many strata, how the budget is split across them, which windows within a stratum get picked, and what makes the result reproducible. It **consolidates decisions already made and justifies them with evidence** — it does not re-open any of them. It says nothing about the metrics themselves ([ADR-0004](0004-frozen-metric-vector.md)), the gate thresholds ([#36](https://github.com/JStoschek/DEMAE/issues/36)), or what happens to a selected window at full resolution ([#50](https://github.com/JStoschek/DEMAE/issues/50)).

## Context

By the end of Phase C the project had a **candidate catalog**: one row per 224×224 window
of every Copernicus GLO-30 COG, each carrying the frozen 6-metric vector, the mask
fractions, and the geography keys — **3,052,735** rows surviving the quality gates. What
it did not have was a rule for turning three million candidates into fifty thousand tiles.

That rule is what ADR-0001 §4–5 originally supplied: cut a single TRI ruggedness axis at
quintiles, give each of the five strata an equal ~10k quota, and fill each quota
round-robin across 10° geo-cells with a per-1°-tile window cap. [ADR-0004](0004-frozen-metric-vector.md)
removed the axis that design stood on — the metric basis is now a 6-dimensional vector —
but deliberately left the *sampling* design undecided, marking it as Phase D's own effort.
Phase D ran that effort ([map #39](https://github.com/JStoschek/DEMAE/issues/39)) and
settled every part of it by measurement. This ADR is the record.

### Why this ADR waited (twice)

Phase D **deliberately declined** to write it. Its map says so in the destination: you
cannot honestly accept a sampling-design ADR before a validation report and a stratum
gallery have judged whether the k-means strata are actually terrain-meaningful. A design
that is internally consistent and a design that produces coherent terrain families are
different claims, and only the second is worth freezing. [`d4-sampler.md`](../research/d4-sampler.md)
was written as "the record ADR-0005 will be built from at Gate E", explicitly *not itself
an ADR*.

Gate E produced that evidence — and then returned **FAIL**
([#54](https://github.com/JStoschek/DEMAE/issues/54)), on an attrition Finding rather than
any build defect. The verdict closed [#56](https://github.com/JStoschek/DEMAE/issues/56)
unfinished on the stated grounds that it "documents a corpus the fresh map may redraw."

Both fresh maps have now returned **ACCEPT** — [ADR-0006](0006-accept-edm-flm-hollowing-stratum19.md)
and [ADR-0007](0007-accept-degenerate-flatness-gate-flat-strata.md). **Nothing was
redrawn.** Both gates stand at their Phase-C thresholds, the selection list on disk is the
one Phase D emitted, and the deferring condition has evaporated. The project has been
running with a hole at 0005 while re-running the very design it was meant to record, so
this ADR lands **early in Phase F, before the 500k draw** — the scale run should conform to
a written contract rather than be documented after the fact across two tiers.

## Decision

### 1. The stratification frame — all kept rows, both tags inside the fit

k-means fits **all 3,052,735 kept rows** of the gated catalog, in the 6-dimensional space
[ADR-0004](0004-frozen-metric-vector.md) froze, **z-scored** (not whitened) over the full
frame in one pass. The scaler ships fingerprinted at [`scaler.json`](../../artifacts/phase_d/scaler.json).

Two clauses of plan §C2 are **amended on measurement** ([#40](https://github.com/JStoschek/DEMAE/issues/40)):

- **`water_affected` rows are *not* held out of texture distances.** The plan assumed
  water-affected windows would distort the texture axis. Measured, they differ by
  **0.030 sd** in `texture` — the *smallest* gap of the six metrics.
- **`coastline` is a reporting axis, not a stratum.** No coastal region exists in
  descriptor space: max stratum purity is **4.9%**. Forcing coastlines into their own
  stratum would manufacture a terrain family the data does not contain.

Both exclusions were tested by re-fitting and comparing partitions: **ARI 0.698** and
**0.656** against a measured **noise floor of 0.607** — the agreement two fits get from
reseeding alone. Neither exclusion moves the strata as much as reseeding does, which is
why neither is worth its complexity. That noise floor is real multi-modality, not
sloppiness: at k=40 two fits 0.06% apart in inertia agreed on only ~61% of pairs. It
became a hard constraint on the choice of k.

### 2. k = 20 strata — Lloyd, chosen at the one measured boundary

**k = 20**, retiring plan §D1's precedent value of 40, which is dominated on every
measured axis ([#41](https://github.com/JStoschek/DEMAE/issues/41)).

The sweep found **no interior optimum**. Silhouette, seed stability and floor cost all
reward smaller k monotonically down to k=8, because the descriptor space is a
**continuum, not clumps** — there is no natural number of terrain families to discover.
With no optimum to find, k must be chosen against a constraint, and the sweep produced
exactly one: a **seed-stability cliff between k=20 and k=30** (ARI **0.846 → 0.639**).
k=20 is the largest k whose strata still reproduce across seeds. It also **halves the
floor cost** — a per-stratum floor consumes 20% of the budget at k=20 against 40% at k=40
— which is what restores Neyman's dynamic range in §3.

**Lloyd, not MiniBatch**: MiniBatch's seed-ARI collapses to 0.51, well below the noise
floor.

**Determinism comes from persisting the centroids, not from seeding.** The fit is not
bitwise-reproducible across thread counts, but that does not matter, because nothing
re-fits: [`centroids.json`](../../artifacts/phase_d/centroids.json) is committed and
`centroids.assign` — a pure `argmin`, ties to the lowest stratum index — labels the full
frame **identically under 1 vs 8 threads (0 disagreeing rows)**.

Realized occupancy: 20 strata, `N_h` from **32,229** to **292,363**, summing to
3,052,735. **No degenerate stratum.**

### 3. Allocation — pure Neyman, plus one within-stratum coastal floor

The budget is split by **pure Neyman allocation**, `n_h ∝ N_h·σ_h` (σ_h = mean distance to
the stratum centroid in standardized space), apportioned to *exactly* the budget by
**largest remainder**. No RNG enters any committed number.
([#42](https://github.com/JStoschek/DEMAE/issues/42), [`d2-allocation.md`](../research/d2-allocation.md).)

At the locked k=20 the other four mechanisms plan §D2 proposed **fall away on measurement**:

- **The `max(500, 1%)` floor is structurally inert.** The smallest Neyman share is
  **1.468% > 1%** at *every* budget, so the clamp fires on **0** strata. It is **kept as a
  declared, non-binding clamp** ([#36](https://github.com/JStoschek/DEMAE/issues/36)'s
  pattern: retire on evidence, keep declared) — which dissolves the ticket's central
  "floors-first vs raise-to-floor ordering" question, since nothing is ever raised.
- **Boost (a), rarest-strata, dropped.** Neyman already grants rare terrain a **1.8×
  per-row premium**; boosting on top double-counts the same rarity.
- **Boost (b), entropy, dropped.** `geomorphon_entropy` is already *inside* the metric
  vector, so boosting on it would re-weight the frozen vector through the back door and
  reopen [ADR-0004](0004-frozen-metric-vector.md).
- **Boost (c), coastline, became a within-stratum frame-rate floor.** Coast is a *location*
  tag, not a descriptor shape ([§1](#1-the-stratification-frame--all-kept-rows-both-tags-inside-the-fit)),
  so the correct guarantee is a **frame rate, not an over-boost**: reserve
  `F = round(0.0134 · budget)` picks, apportioned across strata by largest remainder on
  `coast_h` and capped at `min(n_h, coast_h)`. The coastal frame rate is **1.337%**
  (40,815 of 3,052,735 kept rows); at budget 50,000 the floor is **670**, i.e. 1.34%.
  It is enforced **by construction** — each stratum runs FSCS separately over its coastal
  and base rows — not as a post-hoc top-up, which is what gives every pick an exact
  provenance with no dependence on selection history.
- **The per-COG cap is retired.** `PER_TILE_WINDOW_CAP = 64` was a TRI-era knob; the
  geo-cell round-robin carries the spatial-spread job. The constant is **deleted**, but
  the sampler is bound to **report** the realized windows-per-COG distribution at every
  tier, so a later tier decides on evidence rather than on an untested 64.

Provenance vocabulary is exactly two labels: **`{base, floor:coastline}`** — the shape the
v2 manifest consumes ([ADR-0002 §3a](0002-storage-format-and-manifest.md)).

The realized 50k table ships at [`d2-allocation-nh.json`](../research/d2-allocation-nh.json):
`n_h` from **734 to 3,701** (a **5.04×** dynamic range), coastal 670, base 49,330.

### 4. Within-stratum selection — FSCS with a geo-cell round-robin that falls through

Within each stratum, windows are chosen by **FSCS** (feature-space coverage sampling,
Brus's mechanics): run sub-k-means at `k = quota` over the stratum's rows and take the
window nearest each sub-centroid — coverage of descriptor space, not a random draw.
([#43](https://github.com/JStoschek/DEMAE/issues/43), [`d4-sampler.md`](../research/d4-sampler.md) §4.)

Two mechanics are load-bearing:

- **The round-robin must fall through.** "Nearest window to each sub-centroid", taken
  literally, **under-fills the quota** — two sub-centroids in one dense pocket resolve to
  the same row. So each sub-centroid walks its nearest rows and takes the first *unused*
  one in the **least-used 10° geo-cell**, falling through to the next-nearest. A row whose
  cell is already over-used yields to a less-used cell, which is what stops a stratum
  re-concentrating geographically while still looking well spread in descriptor space.
  A rare residual shortfall is closed exactly by a fill pass over the unused rows nearest
  any sub-centroid — **fall-through, never a short corpus**.
- **Cost was priced before it was committed.** Full-Lloyd sub-k-means over the real strata
  costs **~8 min at 50k / ~55 min at 500k**, RSS flat at ~1.9 GB. The feared `N_h·n_h`
  corner never occurs, because large strata draw floor-sized quotas. **MiniBatch is the
  named, unused fallback** if a budget ever past 500k forces one — at a measured **1.4%
  coverage cost**. The non-clustering spread rule is ruled out: its coverage is *worse
  than random*.

### 5. Determinism — the persisted artifact is the contract, never the seed

**The committed selection list is the determinism contract.** Phase E consumes the file;
it does not regenerate it. ([#43](https://github.com/JStoschek/DEMAE/issues/43),
[#44](https://github.com/JStoschek/DEMAE/issues/44).)

This is not stylistic. **Multithreaded Lloyd is not reproducible even at a fixed seed** —
scikit-learn sums centroids in a thread-scheduling-dependent float order, and
"nearest window" amplifies that ULP drift into a **58%-different selection** (same-seed
Jaccard 0.42). Two mechanisms hold the contract, and neither is "trust the seed":

- **Every sub-k-means fit is pinned single-threaded** via `threadpool_limits(1)`, the
  in-process lever that restores same-seed Jaccard 1.0 regardless of ambient
  `OMP_NUM_THREADS`.
- **Every tie-break is a keyed BLAKE2b hash**, never a NumPy `Generator` — NumPy guarantees
  bit-generator *streams* across releases but not the distributions layered on them. So the
  draw is stable across releases and independent of the order rows arrive from the catalog.

Determinism is **demonstrated, not asserted**: two full runs from the committed artifacts
produced **byte-identical** selection lists (md5 `3936840619…`, `diff` clean over all
50,000 rows). Strata are independent, so the sampler fans them across *processes* — each
fit still single-threaded, the merged list re-sorted — giving a result byte-identical to
the serial path at ~7.4 min for 50k, which is what keeps the scale tier affordable.

**Budget is an argument, the artifacts are inputs.** `select_windows(budget=…)` re-loads
the scaler and centroids and never re-fits them, so a new tier draws from the *same* strata
by construction.

### 6. No nesting between tiers — the draws are independent

The debug tier is **not** a subset of the scale tier. FSCS re-runs fresh at each budget and
the two lists overlap only by chance, so **no hierarchical draw exists and none should be
built** ([#45](https://github.com/JStoschek/DEMAE/issues/45)).

Nesting's sole benefit is a clean "more data, not *different* data" comparison, and no one
here needs it: the debug tier is a fast validation corpus, not a scaling ablation. The free
method of getting nesting (incremental spread) was already ruled out on coverage — worse
than random — and *forcing* nesting would strip the scale tier of its own best
representatives. **Comparability lives at the strata**, via the persisted centroids; and
because Neyman's shares are budget-invariant, the quota *counts* nest proportionally
regardless. Only window identity does not, which nothing depends on.

Recorded clean: the overlap is incidental and no string is attached to any later tier.

## 7. The evidence this design is accepted on — including the part that failed

Phase D deferred this ADR for evidence. Here is exactly what arrived, and what it does and
does not license.

**The strata are terrain-meaningful.** The 12-tiles-per-stratum gallery
([#53](https://github.com/JStoschek/DEMAE/issues/53)) was judged at Gate E as **20/20
coherent, 0 mixed, 0 incoherent**, with 20 terrain-family labels recorded in the verdict
([#54](https://github.com/JStoschek/DEMAE/issues/54)). This is the specific evidence Phase D
deferred to obtain, and it arrived clean. Note what the gallery deliberately does **not**
claim: *distinguishability between* strata is **disavowed**, because §2 established the
space is a continuum — adjacent strata *should* look similar at the boundary.

**The corpus faithfully implements this design.** Seven gating criteria pass with no Defect
([#52](https://github.com/JStoschek/DEMAE/issues/52)): 0 of 49,998 rows differ on metric
fidelity; the min realized stratum is **734** against a floor of 500 with neither
thin-stratum limb tripped; the coastal floor holds at **1.33805%** against the registered
1.337%; per-COG max **44** against the (inert) 64 tripwire.

**And the design did not sail through.** Gate E returned **FAIL** — on E2.5, the
per-(gate × stratum) attrition criterion, which fired on five pairs: `edited` × stratum 19,
and `degenerate` × strata {2, 5, 7, 17}. That was classed a **Finding, not a Defect**: the
corpus is provably not mis-built, so what was on trial was the *design*. Both firings have
since been examined against pre-registered bars and **consciously accepted** —
[ADR-0006](0006-accept-edm-flm-hollowing-stratum19.md) (the edit/fill cap hollowing
high-alpine stratum 19 by 18.1%, 12.1× the corpus rate, of which 94% is severe external
infill an MAE would learn as false terrain) and
[ADR-0007](0007-accept-degenerate-flatness-gate-flat-strata.md) (the flatness screen's
3.04–5.32× concentration on the flat strata, found structural rather than excess bias).

**This ADR accepts a design whose attrition behaviour was challenged and upheld — not one
that passed unquestioned.** Two honest riders follow from that:

1. **ADR-0007 carries a live obligation into Phase F.** Its accept was
   "accept-for-now, re-test at scale" on the **strata 2/17 drainage loss** — in those two
   strata the terrain axis pointed the *wrong way*: the tiles the gate removed were the
   more-dissected, drainage-bearing ones. That re-test is a distinct pre-registered
   criterion of Gate F, and its outcome is a named amendment trigger in [§9](#9-open-knobs--whats-amendable-and-what-would-change-it).
2. **All of this evidence is from the 50k debug tier.** Attrition *rates* are a property of
   the catalog and the gates, not of the budget, so they reproduce at 500k by construction
   — but **headroom is not**. The four flat strata sat at ≥53× their allocation at 50k and
   sit at ≥5.3× at 500k, with FSCS reaching ten times deeper into each pool. That is a real
   regime change, and it is why this ADR pre-registers its triggers rather than declaring
   itself validated at scale.

## 8. What this supersedes in ADR-0001, precisely

[ADR-0004](0004-frozen-metric-vector.md) took §4–5's *metric basis*. This ADR takes what
remained, but not all of it — one mechanism survives, and saying so precisely matters more
than a blanket supersession.

| ADR-0001 §4–5 clause | Disposition |
|---|---|
| TRI on the ×8 overview as the ruggedness score | Superseded by [ADR-0004](0004-frozen-metric-vector.md) (metric basis) |
| **Quintile cuts → 5 strata** | **Superseded here** — k=20 k-means over the frozen vector ([§2](#2-k--20-strata--lloyd-chosen-at-the-one-measured-boundary)) |
| **Equal quota per stratum (~10k each)** | **Superseded here** — pure Neyman ([§3](#3-allocation--pure-neyman-plus-one-within-stratum-coastal-floor)) |
| **Knob: "skew quotas toward upper strata if a preview looks too flat"** | **Superseded here** — Neyman sets the skew from measured σ_h; there is no manual dial |
| **Per-1°-tile window cap** | **Retired here** — inert at the debug tier (max 44 < 64); constant deleted, the metric reported instead ([§3](#3-allocation--pure-neyman-plus-one-within-stratum-coastal-floor)) |
| **10°×10° geo-cell round-robin** | **Survives, amended** — still the spatial-spread mechanism, now inside FSCS and required to **fall through** to the next-nearest window ([§4](#4-within-stratum-selection--fscs-with-a-geo-cell-round-robin-that-falls-through)) |
| 10° cell id recorded in the manifest for a Phase-2 hold-out | Unchanged; `split` stays declared-and-null ([ADR-0002 §3a](0002-storage-format-and-manifest.md)) |
| Deferral of explicit climate/tectonic regime labels | Unchanged |

The code implementing the superseded clauses was **deleted in the same change that shipped
the sampler** ([#44](https://github.com/JStoschek/DEMAE/issues/44)) — `dataset.stratify`,
`dataset.pipeline`, `dataset.build`, `dataset.throughput`, `dataset.synthetic`, and the
three stale constants — so the project never carried two live stratifiers.
`TRI_OVERVIEW_FACTOR` was deliberately **kept**: despite its name it is the live ×8-overview
downsample factor the scoring pass still reads.

**The manifest-schema question this ADR was once expected to answer is already answered
elsewhere.** [#48](https://github.com/JStoschek/DEMAE/issues/48) landed the v2 schema as
**ADR-0002 §3a**, where the storage decision belongs. It is recorded here, not re-decided.

## 9. Open knobs — what's amendable, and what would change it

In the plan §9 idiom. Each trigger is written **before** the 500k evidence exists, on the
same pre-registration principle Gate E ([#47](https://github.com/JStoschek/DEMAE/issues/47))
and the flatness null model ([#63](https://github.com/JStoschek/DEMAE/issues/63)) held: a
criterion chosen after seeing the numbers can be argued either way.

| knob | current setting | evidence | amendment trigger |
|---|---|---|---|
| **k (number of strata)** | 20, Lloyd, persisted centroids | no interior optimum; seed-stability cliff k=20→30 (ARI 0.846→0.639); gallery 20/20 coherent | a scale-tier gallery showing any incoherent or ≥3 mixed strata. **Phase F does not re-cluster** — the scale tier draws from the *same* persisted centroids by construction, which is what makes the tiers comparable at all |
| **allocation rule** | pure Neyman, largest-remainder to exact budget | 5.04× realized dynamic range; min realized 734, neither thin-stratum limb tripped | a Gate F thin-stratum firing at 500k |
| **`max(500, 1%)` floor clamp** | declared, structurally inert (smallest share 1.468%) | fires on 0 strata at every budget | a future k or budget whose smallest Neyman share falls under 1% — the clamp then binds with no code change, which is the point of keeping it declared |
| **coastal frame-rate floor** | `round(0.0134 · budget)`; 670 at 50k | frame rate 1.337%; E2.2d passed **by one tile** (1.33805%) | a Gate F coastal-rate miss. The one-tile margin at 50k is thin by construction, since attrition can remove a coastal pick |
| **per-COG cap** | **retired**; distribution reported every tier | mean 3.16, median 2, p99 16, **max 44** over 15,834 COGs, under the retired 64 | the realized max at 500k. At ~10× depth the densest COGs hold up to 256 available windows, so this is the tier the metric was retained for — [#42](https://github.com/JStoschek/DEMAE/issues/42) bound the scale tier to decide on it |
| **within-stratum selector** | FSCS, full Lloyd, geo-cell round-robin with fall-through | ~55 min at 500k, RSS flat ~1.9 GB | a budget past 500k → **MiniBatch**, the named fallback, at a measured 1.4% coverage cost |
| **tier nesting** | none — independent draws | [#45](https://github.com/JStoschek/DEMAE/issues/45) | someone actually needing a "more data, not different data" ablation. Reopening is an explicit forward rebuild — force the reproducible debug picks into a later draw *then* — never a retroactive change to a shipped list |
| **flatness screen × strata 2/17** | `land_std < 3.0 m`, global, no per-stratum override | [ADR-0007](0007-accept-degenerate-flatness-gate-flat-strata.md): structural, not excess bias — but the terrain axis pointed the wrong way in 2/17 | **ADR-0007's own Phase-F revisit trigger**: the 2/17 drainage re-test at 500k, a pre-registered Gate F criterion. This ADR does not pre-judge it |
| **vegetation / canopy column** | deferred, not built | verified concern; no debug-scale need | canopy-biased Phase-2 probes. The trigger needs evidence that cannot exist until pretraining runs |

## Consequences

- **The scale tier has a written contract to conform to.** Phase F re-runs *this* design at
  500k against the *same* persisted scaler and centroids, with budget as an argument. The
  design is not re-derived per tier.
- **Reproducing the corpus means re-reading committed files, not re-running k-means.** Every
  consumer loads `scaler.json`, `centroids.json` and the selection list. Anyone who
  re-fits gets different strata and should expect to — that is a property of the data
  (a continuum, ARI noise floor 0.607), not a bug.
- **A geographic reading of this corpus will mislead.** Neyman deliberately over-samples
  rare rugged terrain: the corpus is *terrain-diverse by design, not geo-uniform*. This is
  the single most misreadable property of the design and belongs in the model card
  ([#75](https://github.com/JStoschek/DEMAE/issues/75)).
- **Two coverage facts follow from the design, not from a defect.** 400 of 425 kept-pool
  geo-cells are represented at 50k (the 25 misses hold 77 kept windows in total), and 4 of
  20 strata contain no coastal rows at all, so the coastal floor spans 16.
- **The gallery's sample is reconstructible after the shards are gone**, because the
  selection list plus the committed seed regenerate the tile ids — the shards are
  gitignored, the metadata is the artifact.
- **`stratum_id` is a k=20 k-means label and nothing else.** It is not ordinal, not a
  ruggedness rank, and not comparable to the TRI-era `stratum` column, which is why
  [#48](https://github.com/JStoschek/DEMAE/issues/48) renamed rather than reused it.

## Out of scope

- **The metric vector and its transforms** — [ADR-0004](0004-frozen-metric-vector.md).
  This ADR stratifies the space; it does not choose it.
- **Gate thresholds** — the `zero_fill` / `degenerate` / `water` / `edited` predicates and
  their values ([#36](https://github.com/JStoschek/DEMAE/issues/36),
  [ADR-0006](0006-accept-edm-flm-hollowing-stratum19.md),
  [ADR-0007](0007-accept-degenerate-flatness-gate-flat-strata.md)). Sampling consumes the
  gated frame; it does not set the gates.
- **What happens to a *selected* window at full resolution** — the attrition contract is
  `accept_shortfall` ([#50](https://github.com/JStoschek/DEMAE/issues/50)): recorded, never
  backfilled. Backfill would write window ids absent from the committed list, which is
  precisely the contract [§5](#5-determinism--the-persisted-artifact-is-the-contract-never-the-seed) rests on.
- **Storage, manifest schema and the read path** — [ADR-0002](0002-storage-format-and-manifest.md)
  (+§3a) and [ADR-0003](0003-dataloader-design.md).
- **The Phase-2 hold-out split.** `split` is declared-and-null. The 10° cell id is recorded
  so a geographic split *can* be designed later; designing it is not a Phase-0 decision.
- **Re-litigating Gate E's pre-registered bar or its FAIL** — dispositioned in full by
  ADR-0006 + ADR-0007.

## Alternatives considered

- **k = 40 (plan §D1's precedent).** Rejected: dominated on every measured axis — worse
  seed stability (ARI 0.639 vs 0.846), worse silhouette, and double the floor cost (40% vs
  20% of budget), which compresses Neyman's dynamic range for no gain. The precedent came
  from a different domain and did not survive measurement here.
- **MiniBatch k-means for the main fit.** Rejected: seed-ARI collapses to 0.51, below the
  0.607 noise floor — the strata would not reproduce. Retained only as the named FSCS
  fallback past 500k.
- **Seeding for determinism.** Rejected as unachievable, not merely inelegant: multithreaded
  Lloyd is not reproducible at a fixed seed (same-seed Jaccard 0.42). Persisting the
  artifact is the only honest contract.
- **Holding `water_affected` out of texture distances; a dedicated `coastline` stratum.**
  Both rejected on measurement — a 0.030 sd texture gap and 4.9% max stratum purity
  respectively, with re-fit ARIs (0.698 / 0.656) above the reseeding noise floor.
- **Rarest-strata and entropy boosts.** Rejected: Neyman already grants a 1.8× per-row rare
  premium, and entropy is inside the frozen vector — boosting it would re-weight
  ADR-0004 through the back door.
- **Coastline as an over-boost rather than a frame-rate floor.** Rejected: coast is a
  location tag, not a descriptor shape, so the defensible guarantee is "represented at its
  frame rate", not "over-represented by a chosen factor".
- **Post-hoc coastal top-up** (run FSCS blind, then add coastal picks). Rejected: it makes
  a pick's provenance depend on selection history. Running FSCS separately over the coastal
  and base rows fixes provenance by construction.
- **Nesting the debug tier inside the scale tier.** Rejected — see
  [§6](#6-no-nesting-between-tiers--the-draws-are-independent).
- **A non-clustering spread rule instead of FSCS.** Rejected: measured coverage worse than
  random.
- **Chi-square / KS conformance tests on the realized draw.** Dropped at
  [#47](https://github.com/JStoschek/DEMAE/issues/47): the sampler is deterministic and
  consumes a committed quota table, so the expected discrepancy is exactly zero and a
  p-value is decoration.
