# ADR 0004 — Frozen metric vector for corpus stratification (Phase B / Gate B)

- **Status:** Accepted
- **Date:** 2026-07-20
- **Context ticket:** "Freeze the metric vector and write ADR-0004" ([issue #28](https://github.com/JStoschek/DEMAE/issues/28)) — **Gate B** of the Phase-B wayfinder map ([#23](https://github.com/JStoschek/DEMAE/issues/23))
- **Supersedes:** ADR-0001 §4–5 (single-axis TRI ruggedness stratification) — the metric *basis* only; the sampling/allocation design built on it is redrawn by Phase D, not here (see [Out of scope](#out-of-scope))
- **Inputs:** the measured pilot frame ([`docs/research/pilot-frame.md`](../research/pilot-frame.md), 15,000 windows, [#24](https://github.com/JStoschek/DEMAE/issues/24)); the correlation-prune notebook [`notebooks/b2_correlation_prune.ipynb`](../../notebooks/b2_correlation_prune.ipynb) ([#26](https://github.com/JStoschek/DEMAE/issues/26)); the flat-wet separation test [`docs/research/b3-flatwet-separation.md`](../research/b3-flatwet-separation.md) ([#27](https://github.com/JStoschek/DEMAE/issues/27))
- **Depends on:** Gate A3's read-budget decision ([#21](https://github.com/JStoschek/DEMAE/issues/21), ADR-0001 Amendment 1)

## Context

Phase B converts the corpus's redundancy question from hypothesis into measurement.
The deep-research pass ([`tile-metric-selection.md`](../research/tile-metric-selection.md))
came back **unresolved on redundancy**: the strongest collinearity claim ("slope,
relief, and roughness are near-collinear, one PC = 82%") was *refuted 0-3*, not
confirmed, and nothing survived verification on how the amplitude family behaves at
this window size (6.7 km). The literature's own prescription was therefore to *compute
everything cheap, correlate on the actual corpus, then prune* (Argudo et al. 2025).

Phase B did exactly that. The pilot frame ([#24](https://github.com/JStoschek/DEMAE/issues/24))
scored all nine `dataset.metrics` functions on 15,000 land windows **at the resolutions
production will use** (Gate A3, [#21](https://github.com/JStoschek/DEMAE/issues/21)) — so
the correlation matrix measures the metrics as they will actually be clustered, not a
proxy. B2 ([#26](https://github.com/JStoschek/DEMAE/issues/26)) ran the Spearman +
PCA + degeneracy pass and *proposed* a 6-dim survivor set; B3
([#27](https://github.com/JStoschek/DEMAE/issues/27)) settled the one open question the
prune left — whether `tpi_proxy` re-enters to carry a flat-wet/flat-dry axis.

This ADR **freezes** the result. Once accepted, the vector is fixed and Phase C
(global scoring) may begin. Its scope is deliberately narrow (see
[Out of scope](#out-of-scope)): it records the frozen vector, the measured evidence
behind the prune, and the read budget — nothing about how the vector is then sampled.

## Decision

### 1. The frozen metric vector — 6 dimensions

The k-means stratification vector is frozen at these **six** metrics, with the
log-transform set and scoring resolution shown:

| # | Metric | Axis | Scoring resolution | Transform (k-means only) |
|---|---|---|---|---|
| 1 | `slope_mean` | steepness | **full-res (30 m)** | `ln(slope_mean + 1)` |
| 2 | `local_relief` | vertical scale | ×8 overview (~240 m) | `ln(local_relief + 1)` |
| 3 | `texture` | surface roughness / dissection | **full-res (30 m)** | — |
| 4 | `elevation_relief_ratio` | profile shape (≡ hypsometric integral) | ×8 overview | — |
| 5 | `mean_elevation` | absolute altitude | ×8 overview | — |
| 6 | `geomorphon_entropy` | landform composition | **full-res (30 m)** | — |

The two log-transforms collapse long right tails toward symmetry so no single raw
scale dominates the k-means distance: `local_relief` skew +3.58 → −0.29, `slope_mean`
+2.48 → +0.68 under `ln(x+1)` (the `+1` also absorbs the ~0.8% exact-zero windows).
The other four survivors are left untransformed — wrong skew direction, a bounded
fraction, or genuine negatives (`mean_elevation` reaches −28 m) that break `ln`.
Because the prune rule and the PCA are **rank-based** (Spearman), a monotone transform
re-scales values without changing their order, so the transform choice does **not**
reopen any survival decision below — it is a downstream k-means concern only.

**Dropped from the nine candidates:** `slope_std` and `normals_dispersion` (redundant
— §2), and `tpi_proxy` (tested and pruned — §4).

### 2. The measured correlation matrix — evidence for the prune

The prune rule is mechanical: within any block at Spearman |ρ| ≥ 0.90, keep one
representative. The full pairwise-complete matrix over the 15,000-window frame (each
pair re-ranked on the windows where both metrics are defined, so NaN columns never
poison a pair they aren't in):

|                          | local_relief | slope_mean | slope_std | ERR | mean_elev | texture | normals_disp | geom_entropy | tpi_proxy |
|--------------------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| **local_relief**         | 1.00 | 0.81 | 0.80 | −0.11 | 0.26 | 0.54 | 0.81 | −0.03 | −0.01 |
| **slope_mean**           | 0.81 | 1.00 | **0.95** | −0.11 | 0.12 | 0.85 | **0.99** | 0.25 | −0.01 |
| **slope_std**            | 0.80 | **0.95** | 1.00 | −0.21 | 0.05 | 0.79 | **0.99** | 0.21 | −0.05 |
| **elevation_relief_ratio** | −0.11 | −0.11 | −0.21 | 1.00 | 0.18 | −0.10 | −0.16 | 0.01 | 0.41 |
| **mean_elevation**       | 0.26 | 0.12 | 0.05 | 0.18 | 1.00 | 0.11 | 0.09 | −0.02 | 0.11 |
| **texture**              | 0.54 | 0.85 | 0.79 | −0.10 | 0.11 | 1.00 | 0.84 | 0.44 | −0.00 |
| **normals_dispersion**   | 0.81 | **0.99** | **0.99** | −0.16 | 0.09 | 0.84 | 1.00 | 0.23 | −0.02 |
| **geomorphon_entropy**   | −0.03 | 0.25 | 0.21 | 0.01 | −0.02 | 0.44 | 0.23 | 1.00 | −0.03 |
| **tpi_proxy**            | −0.01 | −0.01 | −0.05 | 0.41 | 0.11 | −0.00 | −0.02 | −0.03 | 1.00 |

**One block crosses the 0.90 line:** `{slope_mean, slope_std, normals_dispersion}`
(ρ 0.95–0.99) — three names for essentially one ordering of windows. Keep one →
**`slope_mean`** (slope is the most fundamental terrain descriptor). Everything else
sits below the line: the next-strongest pairs are `slope_mean`↔`texture` (0.85) and
`local_relief`↔`slope_mean` (0.81) — strong but survivable, related not redundant.

The **PCA** (on standardized ranks of the eight survival candidates; `tpi_proxy`
excluded as it is B3's question) confirms four independent directions cover 94.5% of
variance, one per non-slope-block survivor:

| PC | variance | driven by | meaning |
|----|---------:|-----------|---------|
| PC1 | 55.6% | slope / relief / texture | the ruggedness bundle |
| PC2 | 15.4% | `mean_elevation` | altitude |
| PC3 | 14.1% | `geomorphon_entropy` | landform composition (its own axis) |
| PC4 |  9.4% | `elevation_relief_ratio` | profile shape |

**Two Phase-A verdicts were overturned by this measurement** — recorded here because
converting carried-in hypotheses into measured answers is the map's stated purpose:

- **`normals_dispersion` — carried in as "not redundant (tracks curvature, not
  steepness)"; measured 0.986 collinear with `slope_mean`.** Spearman is rank-based,
  so no log/rank transform can undo a redundancy that lives in the *ordering* of
  windows — the "rescue by transform" path cannot apply. Dropped on **redundancy**,
  not degeneracy.
- **`geomorphon_entropy` — carried in as "prune on degeneracy (zero-inflated flat
  tail)"; measured alive.** On the cell-balanced frame (which includes the world's
  mountains) it has only 0.8% zeros, real spread (p10 1.26 → p90 2.98), and owns PC3
  outright. The zero-inflation Phase A predicted from flat band-local samples does not
  hold here. **Kept.**

### 3. The relaxed read budget — the one-read property is spent

ADR-0001 §4 asserted a **one-read property**: a single ×8-overview read per tile
yielded both the ocean signal and the (single, TRI) ruggedness score for all of that
tile's windows. That property is now **consciously spent**. Three of the six frozen
metrics — `slope_mean`, `texture`, `geomorphon_entropy` — are scored at full
resolution; the other three ride the ×8 overview. Scoring a candidate tile therefore
costs **one full-resolution read** on top of the overview read, not one read total.

Per Gate A3's step-cost reasoning ([#21](https://github.com/JStoschek/DEMAE/issues/21)),
this is **one architectural decision, not a per-metric tally.** Pulling the 30 m pixels
for a tile is the expense; computing one metric from that read versus several is
near-identical work. The read is **forced** by the gradient family — `slope_mean` passes
pooled at ρ ≈ 0.90 but collapses ≥50° (55.2% of cells / 29.8% of land area), so it
cannot stay on the overview and cannot be dropped. Once `slope_mean` buys the read,
`texture` and `geomorphon_entropy` are computed from the same pixels at ~zero marginal
cost — **free riders on the read, not independent budget line-items.**

So the budget item is the **read, not the count** — do not read this as three (or, at
A3, five) independent full-res metric verdicts. A3 placed five metrics on the full-res
path; B2's prune (§2) removed two of them (`slope_std`, `normals_dispersion` folded
into `slope_mean`), leaving three full-res survivors. **Dropping free riders does not
shrink the step cost** — the tile is still pulled once — and does not reopen the
decision to spend the one-read property. Pricing the full-res scoring path (and a
possible ≥50°-only hybrid, since slope passes below 50° at ρ 0.95–0.96) is Phase C1's
job, exactly as ADR-0001 Amendment 1 left it.

### 4. `tpi_proxy` was tested and pruned — the re-entry trigger fired empty

`tpi_proxy` is not merely "not selected." The map carried one explicit re-entry
trigger for it: does it re-enter to carry a **flat-wet vs flat-dry** axis the six
survivors miss? B3 ([#27](https://github.com/JStoschek/DEMAE/issues/27)) tested this
against the eight labelled flat exemplars, z-scored into the pilot frame. The six
survivors already separate flat-wet from flat-dry **6/8** (Fisher margin **1.30**);
**appending `tpi_proxy` leaves the Fisher margin and leave-one-out score identical to
three decimals and rescues no exemplar.** The trigger fired empty → `tpi_proxy` **stays
pruned.** It is not redundant with the survivors (its largest ρ with any of them is
+0.41, with `elevation_relief_ratio`) — it is a near-independent axis, but the axis it
measures is plateau-edge / basin position, orthogonal to wetness. Independence without
relevance. Evidence: [`docs/research/b3-flatwet-separation.md`](../research/b3-flatwet-separation.md).

### 5. Known scope boundary — featureless saturated pans

The frozen vector carries **channeled** wetness well: the three water-sculpted wet
exemplars (Ganges–Brahmaputra delta, Mississippi floodplain, Pantanal wetland) all land
on the wet side with strong margins (+1.2 … +1.7). It **cannot** distinguish a
*featureless saturated pan* (Etosha) from a dry plateau — such wetness has **no
terrain-shape signature** (Etosha reads slope 0.44, texture 0.0001: dry-plateau
numbers), so no terrain metric, `tpi_proxy` included, can supply it. That signal is
real but lives in the **EDM/WBM masks** (Etosha reads `edm_edited_frac` 0.895), which
the frame scores as separate columns from the terrain vector. Featureless-pan wet/dry
is therefore **out of the terrain vector's scope by design, delegated to the masks** —
recorded here so a future reader sees the miss as a deliberate scope choice, not an
omission.

## Consequences

- **The vector is frozen; Phase C may begin.** Gate B is met. Global scoring reads
  each candidate tile's six frozen metrics at the resolutions in §1.
- **Scoring cost is one ×8-overview read + one full-resolution read per candidate
  tile** (§3), superseding ADR-0001's one-read cost model. The concrete benchmark and
  the ≥50°-only hybrid are Phase C1's to price.
- **`local_relief` is the first metric to cut** if the vector later proves heavy — it
  is the survivor most spanned by the others: it loads on PC1's ruggedness bundle
  alongside `slope_mean` and `texture` (ρ 0.81 with slope) without owning an
  independent axis of its own (§2). `texture` sits marginally higher with slope
  (0.85) but carries the distinct roughness/dissection signal (ρ 0.44 with
  `geomorphon_entropy`), so it is not the one to drop.
- **Wet/dry discrimination is split across two channels** — the terrain vector for
  water-*sculpted* wetness, the EDM/WBM masks for featureless saturated ground (§5).
  Any downstream stratifier that wants full wet/dry coverage must read both.
- **Reproducibility:** the frozen vector, its transforms, and the correlation evidence
  regenerate from the committed pilot frame via `notebooks/b2_correlation_prune.ipynb`
  and `notebooks/b3_flatwet_separation.py`.

## Out of scope

This ADR is deliberately bounded to the *vector*, not its *use*:

- **Phase D's sampling design is not decided here.** The plan's B4 text folds k=40,
  Neyman allocation, per-stratum floors, rugged boosts, and FSCS into ADR-0004; this
  ADR **declines that** and records the deferral so the next reader sees it as a choice,
  not an omission. How the frozen vector is clustered and allocated into a corpus is
  charted as its own effort, so its frontier is mapped before it is locked.
- **The global scoring pass (Phase C) and its quality-gate thresholds** — Phase C's
  work, tuned against the A1 prevalence table.
- **Whether `WBM == 1` supersedes ADR-0001 §3's exact-zero water test** — parked for
  the Phase-C gates; it needs a per-pixel cross-check, not a Phase-B pruning question.

## Relation to ADR-0001

This ADR **supersedes ADR-0001 §4–5's metric basis.** §4 chose a *single* ruggedness
axis (TRI on the ×8 overview, quantile quintiles); the frozen 6-metric vector replaces
that single axis wholesale. §5's stratification was defined *over those TRI quintiles*;
with a 6-dim k-means vector in its place, the single-axis quintile strata are gone. What
§5 built *on top* of the strata — quota-per-stratum, the 10° geo-cell round-robin, the
per-tile window cap — is **sampling/allocation design**, redrawn by Phase D (above), not
by this ADR. ADR-0001 §4's one-read cost model was already marked stale by its own
Amendment 1 (Gate A3); §3 too, this ADR does not touch — coastal-keep exact-zero masking,
normalization, per-tile outputs all stand.

## Alternatives considered

- **Keep `slope_std` and/or `normals_dispersion` as separate roughness axes** —
  rejected on the measurement: both sit at ρ ≥ 0.95 with `slope_mean` (§2), one
  ordering of windows under three names. Rank-based, so no transform separates them.
- **Re-admit `tpi_proxy` for a flat-wet axis** — rejected: its re-entry trigger fired
  empty (§4). Independent of the survivors but orthogonal to wetness.
- **Drop `geomorphon_entropy` on the carried-in degeneracy verdict** — rejected: the
  degeneracy did not reproduce on the cell-balanced frame; it owns PC3 (§2).
- **Fold Phase D's sampling design into this ADR** (the plan's literal B4 text) —
  declined to keep the freeze auditable and let Phase D map its own frontier
  (Out of scope).
