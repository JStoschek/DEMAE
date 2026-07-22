# Pricing FSCS at catalog scale — and the fallback

Resolves issue [#43](https://github.com/JStoschek/DEMAE/issues/43) (Phase-D map,
[#39](https://github.com/JStoschek/DEMAE/issues/39)). Plan §D3 fixes within-stratum
selection as **sub-k-means with k = n_h**: cluster a stratum into as many sub-clusters
as it has quota, take the window nearest each sub-centroid, geo-cell round-robin as the
tie-breaker `[verified: Brus mechanics]`. The mechanics were verified at B-time; the
**cost at catalog scale was never priced**, and the ticket is that price plus a named
fallback if it is unaffordable.

The concern the ticket states, in its own terms: a large stratum could hold 10⁵–10⁶ of
the 3,052,735 kept rows with n_h in the thousands, and k-means with k in the thousands
over ~10⁶ points is a different animal from the k≈40 global fit — this inner pass could
dominate the entire sampler. The Phase-C precedent is why it had to be measured first:
[#33](https://github.com/JStoschek/DEMAE/issues/33) projected 4.19 h for the global
pass and the real figure was 5.7 h, because a verified-mechanics assertion is not a
runtime.

- Harness: [`notebooks/d3_fscs_cost.py`](../../notebooks/d3_fscs_cost.py)
- Cost grid: [`d3-fscs-cost.json`](d3-fscs-cost.json) · determinism:
  [`d3-determinism.json`](d3-determinism.json) (multithreaded, as-specified) and
  [`d3-determinism-threads1.json`](d3-determinism-threads1.json) (the single-thread
  lever) · spread: [`d3-spread.json`](d3-spread.json) · projection:
  [`d3-projection.json`](d3-projection.json)
- Dev box: Apple M1 Pro (8 cores, 16 GB), scikit-learn 1.9, float32, against the
  committed scaler [`artifacts/phase_d/scaler.json`](../../artifacts/phase_d/scaler.json)

Copyright / attribution (mask spec §5, Art. 6(b)): *produced using Copernicus
WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided
under COPERNICUS by the European Union and ESA; all rights reserved.*

## TL;DR

**Affordable as specified.** Full-Lloyd FSCS over the *real* k=40 strata — every one of
which is between 26,984 and 157,508 rows ([#41](https://github.com/JStoschek/DEMAE/issues/41)'s
occupancy) — costs **~8 minutes** for the whole 50k debug tier and **~55 minutes** for
the 500k scale tier, both summed across all 40 strata. That is minutes, not the hours
the Phase-C precedent warned of, because the term the ticket feared — N_h·n_h — never
gets large: the strata that are large in N_h draw *small* n_h (a floor), and the strata
that draw large n_h are mid-size in N_h. Peak memory is flat at ~1.9 GB regardless of
(N_h, n_h) — the frame, not the sub-k-means. **The memory concern resolves to nothing.**

**The real substance is a determinism caveat, not a cost problem.** §D3 as literally run
— multithreaded Lloyd, then nearest-window — is **not reproducible even at a fixed
seed**: two identical-seed runs share only **42%** of their selected windows, because
scikit-learn's parallel Lloyd sums centroids in a non-associative float order and
"nearest window" amplifies the ULP-level drift into different picks. So the determinism
contract must be the **persisted selection list**, exactly as
[#40](https://github.com/JStoschek/DEMAE/issues/40) concluded for the centroids — the
seed is not enough. Pinning the fit to one thread restores bit-reproducibility, at a
wall-clock cost.

**Fallback, if a future budget ever forces one: MiniBatch sub-k-means.** It is faster in
both regimes (1.7 min / 23 min), costs **1.4%** in coverage against Lloyd, and is
same-seed reproducible where Lloyd is not. Sub-sample-then-Lloyd helps only in the
large-N_h/floor-n_h corner and collapses back onto Lloyd at scale-tier quotas. The
"replace FSCS with a cheap non-clustering spread rule" option is **ruled out** — its
coverage is *worse than uniform-random*.

## 1. The cost, as a function of (N_h, n_h)

Wall-clock seconds, mean peak RSS ~1.87 GB across *every* cell — the 73 MB standardized
frame plus the one Parquet read, never the sub-k-means. **The ticket's memory concern
resolves to nothing**; what follows is entirely about wall clock.

| N_h | n_h | lloyd | minibatch | subsampled |
|---:|---:|---:|---:|---:|
| 30,000 | 500 | 2.3 | 0.5 | 0.4 |
| 30,000 | 1,000 | 2.7 | 1.5 | 2.5 |
| 30,000 | 2,500 | 6.8 | 3.0 | 6.2 |
| 30,000 | 5,000 | 12.1 | 5.8 | 11.4 |
| 100,000 | 500 | 5.7 | 0.8 | 0.5 |
| 100,000 | 1,000 | 9.5 | 1.6 | 1.5 |
| 100,000 | 2,500 | 19.6 | 6.5 | 10.0 |
| 100,000 | 5,000 | 40.1 | 7.3 | 40.8 |
| 100,000 | 25,000 | 191.8 | 180.2 | 197.7 |
| 300,000 | 500 | 31.0 | 3.9 | 1.5 |
| 300,000 | 1,000 | 47.1 | 5.7 | 3.2 |
| 300,000 | 2,500 | 112.3 | 6.2 | 15.4 |
| 300,000 | 5,000 | 179.4 | 20.6 | 43.4 |
| 300,000 | 25,000 | 722.6 | 164.8 | 771.8 |
| 1,000,000 | 500 | 99.4 | 10.1 | 2.2 |

Cells above the map's realistic range (n_h = 25,000, and the whole N_h = 1,000,000 row)
are anchors, not operating points — see readings 2 and 3. `subsampled` at n_h = 5,000
and 25,000 equals `lloyd` because its 20× cap (100k, 500k) meets or exceeds the stratum,
so nothing is actually subsampled.

Three readings decide the verdict:

1. **Cost is roughly linear in N_h at fixed n_h, and *sub*-linear in n_h at fixed N_h.**
   At n_h = 500, tripling the rows (300k → 1M) costs 3.2× the time (31 → 99 s). At
   N_h = 300k, a 10× larger quota (500 → 5,000) costs only 5.8× the time (31 → 179 s):
   more centroids converge in fewer Lloyd iterations, so doubling the quota never
   doubles the cost. Both are Lloyd's expected behaviour — per-iteration cost is
   O(N_h·n_h) but the iteration count falls as n_h rises — so nothing here is a surprise
   regime.
2. **The product N_h·n_h is what costs, and the allocation rule keeps it small.** A
   stratum is either large in N_h and draws a floor (small n_h), or draws a large n_h
   because Neyman gave it budget — which happens to *high-variance* strata, and the
   high-variance strata are not the biggest ones. The worst realistic cell is a
   mid-size stratum at a scale-tier quota, which §2 prices directly.
3. **The corner the ticket feared does not occur.** k in the thousands over 10⁶ points
   needs a stratum of 10⁶ rows *and* a quota in the thousands. No k=40 stratum reaches
   10⁶ rows — the largest is 157,508 ([#41](https://github.com/JStoschek/DEMAE/issues/41))
   — so that cell is off the map. The 1,000,000-row row above was measured only to
   anchor the extrapolation, and even there memory stayed flat and cost stayed linear;
   the sampler will never be handed it.

## 2. The projection over the real occupancy

§2 replaces the synthetic grid maxima with #41's measured k=40 occupancy (26,984 –
157,508 rows, median 77,680) and the crude §D2 allocation shape — floor first, remainder
∝ N_h — because #42 has not settled the real rule and the cost model needs only the n_h
distribution's *size*, not its exact composition.

| tier | budget | floor | n_h range | **Lloyd** | MiniBatch | subsampled |
|---|---:|---:|---:|---:|---:|---:|
| debug | 50,000 | 500 | 726 – 2,212 | **8.0 min** | 1.7 min | 2.0 min |
| scale | 500,000 | 5,000 | 7,262 – 22,129 | **54.8 min** | 23.1 min | 71.8 min |

Costs are the measured grid fitted as a power law in (N_h, n_h) and summed over the 40
real strata. The 50k figure is **interpolation** — every real (N_h, n_h) pair falls
inside the measured grid — so it is trustworthy; the 500k figure is mild extrapolation
in n_h, bracketed by the measured n_h = 25,000 cells.

Two things to read from it. First, **the debug tier this map ships costs 8 minutes of
Lloyd** — the sub-k-means is a rounding error next to Phase E's full-res materialize, so
there is nothing to optimise and no reason to reach for a fallback. Second, **the
subsampled column *inverts* at the scale tier** (72 min, slower than Lloyd's 55): at
n_h in the thousands the 20× cap exceeds most strata, so subsampling does nothing but
add overhead. That is the measured reason the fallback of choice is MiniBatch, not
sub-sampling — MiniBatch is the only one of the three that is faster than Lloyd at
*both* tiers.

## 3. Determinism — the map's contract binds here identically

The map's standing bar: the selection must reproduce for the life of the project, and
NumPy's `Generator` distributions do not qualify. The measurement here is stronger and
more uncomfortable than a `Generator` worry — it is about the k-means itself.

Two runs at the **same seed**, same stratum (N_h = 300,000, n_h = 1,000), Jaccard
overlap of the selected window ids:

| method | same seed, two runs | different seeds |
|---|---:|---:|
| **lloyd (as specified)** | **0.42** | 0.03 |
| minibatch | 1.00 | 0.01 |
| subsampled | 1.00 | 0.01 |

**Full Lloyd, as §D3 specifies it, does not reproduce at a fixed seed.** Two identical
runs agree on 42% of their picks. The cause is not the seed and not NumPy: scikit-learn's
`algorithm="lloyd"` parallelises the assignment step across OpenMP threads, and the
centroid sums are reduced in a thread-scheduling-dependent order. Float addition is not
associative, so the centroids differ by ULPs between runs — and because selection then
takes the *window nearest each centroid*, in a dense stratum a ULP-level centroid shift
flips which window is nearest. The tiny cause is amplified into a 58%-different
selection.

Pinning the fit to a single thread removes the parallel reduction and restores
bit-reproducibility: the same run at `OMP_NUM_THREADS=1` scores **same-seed Jaccard 1.0**
for all three methods
([`d3_determinism_threads1.json`](../../prototypes/out/d3_determinism_threads1.json)).
That is the lever — the cause is confirmed to be the threading, not the seed — at a
wall-clock cost, since single-threaded Lloyd is several times slower.

The different-seed column is the expected echo of
[#40](https://github.com/JStoschek/DEMAE/issues/40)'s multi-modality finding: at this
scale a sub-k-means admits many near-equally-good partitions, so a different seed gives
an almost entirely different — but equally valid — selection (Jaccard 0.01–0.03). The
sub-k-means seed is *not* a meaningful contract even for the reproducible methods.

**So the determinism contract is the persisted selection list, full stop** — identical
to #40's conclusion that "determinism comes from the committed centroids, not from the
fit", one level down. [#44](https://github.com/JStoschek/DEMAE/issues/44) must ship the
selected window ids as *the* artifact and must not claim the selection can be regenerated
by re-running FSCS. If regenerability is wanted as defence-in-depth, the lever is
`OMP_NUM_THREADS=1` (Lloyd), or MiniBatch, which is reproducible by construction and
faster — see §4.

## 4. The fallbacks, and what they cost in accuracy

The ticket asks for a named fallback with its cost of accuracy even if FSCS is
affordable, so that a later budget change does not reopen a solved question. Three were
built and measured against uniform-random selection as the do-nothing baseline; "spread"
is two numbers that trade off — **coverage** (mean distance from a stratum row to its
nearest selected row, the quantity FSCS minimises) and **min separation** (the closest
pair among the selected, the near-duplicate guard).

One stratum (N_h = 300,000, n_h = 1,000), standardized descriptor space:

| method | coverage (↓ better) | min separation (↑ better) | reproducible | cost vs Lloyd |
|---|---:|---:|:---:|---|
| uniform-random | 0.250 | 0.078 | — | baseline |
| **lloyd (§D3)** | **0.226** | 0.170 | no | 1× |
| minibatch | 0.229 | 0.135 | yes | 0.05–0.9× |
| subsampled | 0.238 | 0.122 | yes | 0.02–1.0× |
| cheap (round-robin + spacing) | 0.280 | 0.416 | yes | — |

**Coverage is the number that matters** — it is exactly what FSCS exists to minimise
(every stratum row close to a chosen representative). Read down that column:

- **Lloyd wins**, as it should — it is the method that directly minimises it.
- **MiniBatch costs 1.4%** (0.229 vs 0.226) and **sub-sampling costs 5%** (0.238). Both
  are cheap insurance: if a future budget makes Lloyd's cost bite, either buys most of
  Lloyd's spread back for a fraction of the runtime, and both are seed-reproducible into
  the bargain.
- **The cheap non-clustering rule is *worse than random*** (0.280 vs 0.250). It has by
  far the best min-separation (0.416) because it spreads by *rejection* — forcing chosen
  windows apart — but that is the wrong objective: pushing points apart opens gaps, and
  coverage measures the gaps. It optimises the half of "spread" that FSCS does not need
  at the expense of the half it does. **Replacing FSCS with a spread rule is ruled out.**

So the named fallback, with its accuracy cost, is **MiniBatch sub-k-means at 1.4%
coverage** — the only alternative that is faster than Lloyd at both tiers (§2), within a
rounding error of its spread, and reproducible where Lloyd is not. Sub-sample-then-Lloyd
is a situational second (large N_h, floor n_h) that collapses onto Lloyd at scale-tier
quotas. Neither is *needed* — Lloyd is affordable as specified — but if the scale tier
grows past 500k, MiniBatch is the lever, and this measurement means that decision does
not have to be re-litigated.

### The dedup question is settled, not deferred

§D3 already answers it and this ticket does not reopen it: the non-overlapping grid
guarantees **zero spatial overlap**, strictly stronger than SSL4EO's ¾-width rule
`[verified]`, and FSCS suppresses look-alikes within strata. No hashing or embedding
dedup. The only trigger to revisit is jittered or overlapping windows, which the corpus
does not have.

## 5. What this hands forward

- **[#42](https://github.com/JStoschek/DEMAE/issues/42) (allocation)** — the cost model
  confirms allocation and selection are the coupled pair the map says they are: the
  selection cost is a function of the n_h distribution allocation produces, and the
  cheap regime depends on large strata drawing small quotas. An allocation rule that
  gave the *largest* strata the *largest* quotas would move FSCS toward the expensive
  corner; the §D2 shape does the opposite and should be kept.
- **[#44](https://github.com/JStoschek/DEMAE/issues/44) (sampler)** — builds §D3 as
  **full-Lloyd sub-k-means**, no fallback. Two implementation facts fall out of this
  measurement and must carry into the sampler: (a) "nearest window to each sub-centroid"
  **under-fills the quota** if taken literally — two sub-centroids in one dense pocket
  resolve to the same row, so the geo-cell round-robin tie-break must be a *fall-through
  to the next-nearest row*, not a de-duplication that drops the collision and returns
  fewer than n_h windows; (b) the sub-k-means seed is **not** the determinism contract —
  the selected window ids are, so the sampler persists the selection, exactly as the
  scaler and centroids are persisted.
- **The fallback, if the scale tier ever grows past 500k** — MiniBatch sub-k-means, at
  a measured 1.4% coverage cost and same rule (nearest window to each sub-centroid, with
  the round-robin fall-through). It is faster than Lloyd at both tiers and reproducible
  by construction. Do **not** reach for the non-clustering spread rule — its coverage is
  worse than random — and treat sub-sample-then-Lloyd as situational only.
