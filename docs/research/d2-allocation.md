# Compose the allocation rule: Neyman, floors, boosts, caps — and their order

Resolves issue [#42](https://github.com/JStoschek/DEMAE/issues/42) (Phase-D map,
[#39](https://github.com/JStoschek/DEMAE/issues/39)). Plan §D2 stacks five mechanisms —
Neyman allocation, a per-stratum floor, three boost pools, a per-COG cap, and a geo-cell
round-robin — and never states how they compose. This is that composition, written as an
**ordered, total procedure** with every collision case answered, measured against the
locked k=20 centroids ([#41](https://github.com/JStoschek/DEMAE/issues/41)). It is the
per-stratum quota table the sampler ([#44](https://github.com/JStoschek/DEMAE/issues/44))
consumes; the selection itself — FSCS sub-k-means, provenance tagging, the round-robin —
is #44's.

Artifacts: [`docs/research/d2-allocation-nh.json`](d2-allocation-nh.json) (the committed
n_h table). Reproduction (throwaway harness):
[`prototypes/d2_allocation_proto.py`](../../prototypes/d2_allocation_proto.py), which
reads the committed [`centroids.json`](../../artifacts/phase_d/centroids.json) and
[`scaler.json`](../../artifacts/phase_d/scaler.json) — no COG is touched.

Copyright / attribution (mask spec §5, Art. 6(b)): *produced using Copernicus WorldDEM-30
© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under
COPERNICUS by the European Union and ESA; all rights reserved.*

## 1. The headline: k=20 dissolved most of §D2

Plan §D2 was written against the incumbent k=40, where the `max(500, 1% of budget)` floor
consumes **40% of the budget** before Neyman allocates anything, and three ~5% boost
pools take another 15% — leaving Neyman to shape ~45%. At that geometry the *order* of the
mechanisms genuinely is the design: floors-first and raise-to-floor produce different
corpora.

[#41](https://github.com/JStoschek/DEMAE/issues/41) locked **k=20**, which halves the
floor's *nominal* cost to 20%. Measuring the actual allocation at k=20 halves it again, to
**zero**: the floor never fires. Two of the three boosts turn out redundant with machinery
this map already has, and the third — coastline — was already re-scoped by
[#40](https://github.com/JStoschek/DEMAE/issues/40) from a stratum into a scattered tag.
What remains is **pure Neyman with a coastal floor**, and the "order" question the ticket
framed as central mostly dissolves.

## 2. The spine: pure Neyman, floor as an inert clamp

Neyman allocation is `n_h ∝ N_h · σ_h`, where `σ_h` is the **mean within-stratum distance
to the centroid** in the scaler's standardized space — the variance proxy ADR-0004's
"standardize, don't whiten" stance makes readable as sd-units. Big strata and spread-out
strata draw more; the two terms partly cancel because at k=20 the small strata are also
the high-σ ones (the sparse, unusual terrain families).

**Measured: the §D2 floor is structurally inert.** The smallest pure-Neyman allocation is
**734 windows** (stratum 6, the rarest at N_h=32,229) — a 1.47% share against a 1% floor.
No stratum falls below `max(500, 1%·budget)=500`. And because Neyman's *shares* do not
depend on the budget, the smallest share stays 1.47% at every budget: at the Phase-F
500k tier the floor is 5,000 but the smallest Neyman draw is 7,340. **The floor never
fires at any budget.** This is the same shape [#36](https://github.com/JStoschek/DEMAE/issues/36)
found for `zero_fill` — a constant carried from an earlier era that measurement shows does
not bind.

So the "floors-first vs raise-to-floor" ordering question has no teeth: since nothing is
below the floor, raise-to-floor *is* pure Neyman, and floors-first is the only ordering
that would change anything — and all it does is carve out 20% for equal splits, re-
compressing the exact 5.0× Neyman dynamic range (734–3,701) that #41 fought to win back.
The floor is retained in code as a **non-binding raise-to-floor clamp** and documented as
inert — free insurance against a Phase-F re-fit changing stratum proportions, shaping
nothing here.

## 3. The boosts: two dropped, one becomes the coastal floor

**Boost (a), rarest strata by N_h — dropped.** Neyman is *already* a rare-terrain boost:
the σ term makes the rarest stratum's rows **1.8× more likely** to be selected than the
largest stratum's (22,784 vs 12,660 picks per million rows). A boost pool would pour
another ~5% of budget onto strata that are already over-represented per-row. The original
justification — minimum presence, the Prithvi precedent — is met twice over: the floor
clamp guarantees ≥500, and at k=20 the rarest stratum holds 32,229 rows (64× the floor),
so no terrain family is anywhere near vanishing. If more rare-terrain emphasis were ever
wanted, the honest lever is a σ exponent (`N_h · σ_h^p`, p>1), tuned and recorded — not a
flat pool bolted on the side.

**Boost (b), high-descriptor-entropy windows — dropped.** It was never defined, and every
concrete reading is forbidden, redundant, or unbuilt. Boosting by the `geomorphon_entropy`
metric re-weights one of ADR-0004's six frozen axes at selection time — un-freezing the
metric vector through the back door. Boosting by distance-from-own-centroid fights FSCS,
whose sub-k-means already spreads a stratum's picks across its full internal structure
(tails included), *and* over-weights the least-representative rows. A windowed spatial
entropy is an undefined new measurement with no formula in the plan. The map's binding
preference is "do not ship it ambiguous"; there is no interpretation worth building.

**Boost (c), the coastline stratum — became a within-stratum floor.** #40 dissolved the
coastal stratum (max stratum purity 4.9% — no coast region exists in descriptor space) and
handed this ticket the debt: guarantee coastal presence across whatever strata the coastal
rows land in. See §4.

## 4. The coastal floor: frame rate, reserved within-stratum

Coastline is **40,815 windows = 1.34% of the frame**, a thin minority everywhere (the
most-coastal stratum is only 3.5% coastal), and **69.7% of it also carries
`water_affected`** — so a large coastal boost is mostly a water boost. Under pure Neyman
with proportional-within-stratum picking the 50k would *naturally* contain ~669 coastal
windows, but FSCS picks nearest-to-sub-centroid, not proportionally, and a scattered 3.5%
minority can be under-picked with nothing noticing until E2. That gap is the #40 debt.

The floor is set at **frame rate, 670 (1.34% of budget)** — not an over-boost. The reason
is the one that killed the coastal stratum: coast is a **location/gate property, not a
descriptor shape**, and the corpus is stratified by the six frozen descriptors. Over-
weighting coast reaches outside those descriptors to bias a descriptor-balanced sample;
frame rate is the honest minimum that guarantees coast is *never under-represented relative
to reality* without front-running E2's validation (where the evidence to justify more
coastal weight, and ADR-0005, belong).

It is enforced by **up-front within-stratum reservation** (Mechanism A): stratum h reserves
`c_h = round(670 · coast_h / 40,815)` of its quota for coastal windows, capped at
`min(n_h, coast_h)`, apportioned by largest-remainder so `Σ c_h = 670` exactly. FSCS then
runs twice inside the stratum — over its coastal rows for the `c_h` picks (provenance
`floor:coastline`) and over the rest for `base_h = n_h − c_h` (provenance `base`). This
guarantees the floor by construction, gives exact per-row provenance (a shipped
requirement of the map), stays fully deterministic, and distributes across strata exactly
where coastal rows fall. The rejected alternative (run FSCS blind, top up only on
shortfall) is more faithful to the inert-clamp philosophy but needs an eviction rule and
makes a pick's provenance depend on selection history — which the map's per-row provenance
requirement rules out.

## 5. The per-COG cap: retired

`PER_TILE_WINDOW_CAP = 64` caps how many windows one 1°×1° COG contributes. It is an
ADR-0001 / TRI-era constant, one of the three the map names for deliberate retirement.
**At the debug tier it cannot bind:** 50,000 windows over 24,118 covered COGs is ~2 per
COG, and the geo-cell round-robin scatters a stratum's picks across its 10° cells before
any COG can accumulate. It is retired; the round-robin carries the "don't let one place
dominate" job at 10° granularity.

Honesty note on the scale tier: the rule is budget-parameterized and Phase F re-runs it at
200–500k, where the average rises to ~21 windows/COG and the densest COGs (256 available
windows) could plausibly push a single COG past 64. The cap is inert *here* but not provably
inert *there*, and the definitive fire/no-fire number needs the selection to actually run.
So retirement is paired with a requirement on #44: **report the realized windows-per-COG
distribution at each tier.** If the scale tier ever shows a COG dominating, a cap is
reintroduced with an evidence-based threshold, not the untested 64.

## 6. The ordered, total procedure

Inputs: kept frame (3,052,735 rows), committed k=20 centroids + scaler, budget B, coastal
floor F = round(0.0134·B). For the debug tier B=50,000, F=670.

0. **Assign** every kept row to its stratum via `centroids.assign` (deterministic argmin,
   ties to lowest stratum index). Yields `N_h`; in standardized space, `σ_h` = mean
   distance to the stratum centroid.
1. **Neyman base.** `w_h = N_h · σ_h`; apportion B by **largest-remainder** (floor the
   exact shares, hand leftover slots to the largest remainders, ties by lowest index, no
   RNG) → integer `n_h` summing to exactly B. Check the `max(500, 1%·B)` clamp; it is
   inert (0 strata below).
2. **Coastal reservation.** Apportion F by largest-remainder on `coast_h`, capped at
   `min(n_h, coast_h)` → `c_h`, summing to exactly F. Strata with no coastal rows get 0.
3. **Base split.** `base_h = n_h − c_h`.
4. **Selection (handoff to #44).** Within each stratum, FSCS sub-k-means picks `c_h`
   coastal windows (over the coastal rows, tag `floor:coastline`) and `base_h` from the
   rest (tag `base`); per-stratum geo-cell round-robin over the stratum's *occupied* cells,
   with fall-through to the next-nearest cell when one is exhausted (#43), ensures spatial
   spread and exact fill against the ragged 425/472 grid.

**Collision cases, all answered:**

- **`n_h > N_h`** (Neyman asks for more than a stratum holds): impossible here (max n_h
  3,701 ≪ min N_h 32,229); the assertion is retained as a guard for a Phase-F re-fit.
- **A stratum with < 500 rows** (would the floor be a floor, or `min(500, N_h)`?): moot at
  k=20 (rarest 32,229). Only returns on a re-fit; the clamp would resolve to `min(500,
  N_h)` if it ever bound.
- **`c_h > coast_h` or `c_h > n_h`**: capped by construction.
- **Empty geo-cells** (47 hollow of 472): the round-robin iterates only over cells a
  stratum actually occupies and never assumes the grid is full; fall-through fills the
  quota exactly.
- **Rounding**: largest-remainder, deterministic, ties by lowest index — no RNG enters any
  committed number, per the map's determinism contract.

## 7. The committed n_h table (k=20, budget=50,000)

| stratum | N_h | σ_h | coast_h | **n_h** | coastal | base |
|--------:|----:|----:|--------:|--------:|--------:|-----:|
| 0 | 186,950 | 1.08 | 1,662 | 3,220 | 27 | 3,193 |
| 1 | 168,835 | 1.04 | 3,093 | 2,807 | 51 | 2,756 |
| 2 | 150,898 | 0.96 | 1,036 | 2,316 | 17 | 2,299 |
| 3 | 130,184 | 1.05 | 841 | 2,185 | 14 | 2,171 |
| 4 | 232,598 | 0.99 | 1,426 | 3,662 | 23 | 3,639 |
| 5 | 118,638 | 1.05 | 449 | 1,991 | 7 | 1,984 |
| 6 | 32,229 | 1.43 | 0 | 734 | 0 | 734 |
| 7 | 139,429 | 1.17 | 0 | 2,604 | 0 | 2,604 |
| 8 | 292,363 | 0.79 | 2,344 | 3,701 | 38 | 3,663 |
| 9 | 166,251 | 1.03 | 2,306 | 2,724 | 38 | 2,686 |
| 10 | 148,603 | 1.19 | 1,340 | 2,817 | 22 | 2,795 |
| 11 | 50,767 | 1.25 | 0 | 1,015 | 0 | 1,015 |
| 12 | 185,487 | 1.08 | 5,935 | 3,203 | 97 | 3,106 |
| 13 | 194,931 | 1.07 | 3,573 | 3,324 | 59 | 3,265 |
| 14 | 143,914 | 1.07 | 4,915 | 2,467 | 81 | 2,386 |
| 15 | 115,886 | 1.13 | 4,064 | 2,098 | 67 | 2,031 |
| 16 | 220,981 | 0.88 | 3,816 | 3,087 | 63 | 3,024 |
| 17 | 105,302 | 1.10 | 1,095 | 1,847 | 18 | 1,829 |
| 18 | 215,052 | 0.90 | 2,920 | 3,092 | 48 | 3,044 |
| 19 | 53,437 | 1.30 | 0 | 1,106 | 0 | 1,106 |
| **Σ** | **3,052,735** | | **40,815** | **50,000** | **670** | **49,330** |

Budget exact (50,000) · coastal floor met (670, across 16 strata; 4 strata hold no coastal
rows) · every `n_h < N_h` · Neyman's full 5.0× dynamic range preserved (734–3,701).

## 8. Departures from plan §D2, with reasoning

| §D2 as written | This rule | Why |
|---|---|---|
| `max(500, 1%)` floor, applied | Retained as a **non-binding clamp** | Measured inert at every budget (min Neyman share 1.47% > 1%); #36 pattern |
| Boost (a): rarest strata, ~5% | **Dropped** | Neyman already grants rare strata a 1.8× per-row premium; no near-empty stratum at k=20 |
| Boost (b): high-entropy windows, ~5% | **Dropped** | Undefined; every reading reopens ADR-0004 or fights FSCS's own diversity |
| Boost (c): coastline stratum, ~5% | **Within-stratum frame-rate floor (670)** | No coast region in descriptor space (#40); coast is a location tag, not a shape |
| Per-COG cap = 64 | **Retired**; #44 reports realized per-COG | Inert at debug tier (~2/COG); TRI-era; round-robin covers spread |

Each departure is a measurement redrawing the plan, in the stance the map inherits from
#36: measure the constant, state plainly whether it binds, keep or retire on that basis.
