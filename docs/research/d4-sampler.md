# Build the sampler and emit the debug-tier selection list

Resolves issue [#44](https://github.com/JStoschek/DEMAE/issues/44) (Phase-D map,
[#39](https://github.com/JStoschek/DEMAE/issues/39)) — the map's **destination**. Nothing
is decided here: the frame ([#40](https://github.com/JStoschek/DEMAE/issues/40)), k
([#41](https://github.com/JStoschek/DEMAE/issues/41)), the allocation rule
([#42](https://github.com/JStoschek/DEMAE/issues/42)), the FSCS mechanics
([#43](https://github.com/JStoschek/DEMAE/issues/43)) and the no-nesting decision
([#45](https://github.com/JStoschek/DEMAE/issues/45)) all arrive settled. This ticket
makes them real, reproducibly. It is the record **ADR-0005 will be built from at Gate E** —
and, deliberately, *not itself an ADR*: you cannot honestly accept a sampling-design ADR
before Phase E2's validation report and the 12-tiles-per-stratum gallery have judged
whether the k-means strata are terrain-meaningful (map §Destination).

Artifacts:

- **[`artifacts/phase_d/selection_debug_50k.csv`](../../artifacts/phase_d/selection_debug_50k.csv)** —
  the selection list: 50,000 `window_id, stratum_id, provenance` rows, the columns Phase
  E1's manifest addendum consumes. Committed — it *is* the determinism contract.
- **[`docs/research/d4-selection.json`](d4-selection.json)** — the run summary: provenance
  counts, the realized windows-per-COG distribution, the per-stratum allocation.
- Code: [`src/dataset/select.py`](../../src/dataset/select.py); tests
  [`tests/test_select.py`](../../tests/test_select.py).
- Inputs, re-loaded not re-fit: the committed
  [`scaler.json`](../../artifacts/phase_d/scaler.json) (#40),
  [`centroids.json`](../../artifacts/phase_d/centroids.json) (#41), and the
  [`d2-allocation-nh.json`](d2-allocation-nh.json) quota table (#42) this run reproduces.

Copyright / attribution (mask spec §5, Art. 6(b)): *produced using Copernicus WorldDEM-30
© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under
COPERNICUS by the European Union and ESA; all rights reserved.*

## 1. What the sampler does, in order

`dataset.select.select_windows` is a pure function of loaded artifacts — no COG is
touched. Every step is deterministic:

0. **Standardize + assign.** The kept frame is z-scored by the committed scaler and each
   row labelled with its nearest centroid by `centroids.assign` — the pure `argmin` #41
   ships, ties to the lowest stratum index. This yields `N_h` (occupancy) and, in
   standardized space, `σ_h` (mean distance to the stratum centroid).
1. **Allocate** (`allocate`, budget-parameterized). `w_h = N_h·σ_h`, apportioned to
   *exactly* the budget by largest-remainder; then the coastal floor
   `F = round(0.0134·budget)` apportioned by largest-remainder on `coast_h`, capped at
   `min(n_h, coast_h)`; then `base_h = n_h − c_h`. No RNG enters any committed number.
2. **Select** (`fscs_select`, twice per stratum). Within each stratum, FSCS sub-k-means at
   `k = quota` runs once over the **coastal** rows for the `c_h` picks (provenance
   `floor:coastline`) and once over the **base** rows for the `base_h` picks (provenance
   `base`). Each sub-centroid contributes the window nearest it under a geo-cell
   round-robin, with fall-through to the next-nearest so the quota fills exactly.
3. **Emit.** The `window_id, stratum_id, provenance` triples, sorted by
   `(stratum_id, window_id)` so the file is byte-stable across runs.

**Budget is an argument, the artifacts are inputs.** `budget` is a parameter of
`select_windows`; the scaler and centroids are re-loaded and never re-fit. Phase F re-runs
this same code at 200–500k against the identical strata (plan §F), and #45 settled that
the debug tier is *not* nested inside the scale tier — FSCS re-runs fresh at each budget,
so there is no hierarchical draw to build.

## 2. The allocation reproduces #42's committed table exactly

`allocate`, handed the real frame's full-precision `N_h`, `σ_h`, `coast_h` at
budget 50,000, reproduces [`d2-allocation-nh.json`](d2-allocation-nh.json) **row for row** —
`n_h`, `coastal`, `base` and `N_h` all identical. This is asserted against the committed
table in `tests/test_select.py::test_allocation_reproduces_committed_table_on_real_catalog`
(skipped only where the gated catalog is absent). The σ_h in the JSON is rounded to four
decimals for legibility; feeding those rounded values back shifts a handful of `n_h` by ±1
(largest-remainder is sensitive to the last decimal), which is exactly why the sampler
recomputes σ_h at full float precision from the frame rather than reading the table.

Budget exact (50,000) · coastal floor exact (670) · every `n_h < N_h` · Neyman's full 5.0×
dynamic range preserved (734–3,701). The `max(500, 1%)` clamp fires on **0** strata (the
smallest Neyman share is 1.47% > 1%; #42 §2) — reported, not applied.

## 3. Determinism — the persisted list is the contract

The map's hard bar: two runs from the committed artifacts must produce byte-identical
selections, for the life of the project. Two mechanisms hold it, and neither is "trust the
seed":

- **The sub-k-means fit is pinned single-threaded.** #43 measured that multithreaded Lloyd
  is *not* reproducible even at a fixed seed (same-seed Jaccard 0.42): scikit-learn sums
  centroids in a thread-scheduling-dependent float order, and "nearest window" amplifies
  the ULP drift into a 58%-different selection. `_fit_subkmeans` runs the fit under
  `threadpoolctl.threadpool_limits(1)` — the in-process lever #43 found restores same-seed
  Jaccard 1.0, and it holds regardless of the ambient `OMP_NUM_THREADS` (the committed run
  also exported `OMP_NUM_THREADS=1` as belt-and-suspenders, but the pin does not depend on
  it).
- **Every tie-break is a keyed BLAKE2b hash, never a NumPy `Generator`.** NumPy guarantees
  its bit-generator *streams* across releases but not the distributions layered on them
  (`sampling.py`'s house rule, #40's scaler, #43's subsample). The round-robin's
  within-cell order and the fill pass both order by `sampling._shuffle_key`, keyed on the
  window id, so the draw is stable across releases and independent of the order rows
  arrive from the catalog.

But the guarantee the corpus rests on is not "re-run FSCS and trust it" — it is that **the
emitted window ids are shipped and re-loaded**, exactly as #40 concluded for the centroids
and #43 for the selection. The committed CSV is the contract; a re-run reproduces it
(single-thread), but Phase E consumes the file, it does not regenerate it.

Cost: each stratum's fit is single-threaded (for determinism), but the strata are
independent, so the sampler fans them across *processes* (`select_windows(..., workers=)`,
one per core by default) — each worker single-threaded, the merged list re-sorted, so the
result is byte-identical to the serial path. This matters at the **Phase-F scale tier**,
where a serial single-threaded pass runs into the hour; the debug tier is minutes and
dwarfed by Phase E's full-res materialize. There is no reason to reach for #43's MiniBatch
fallback (reserved for a Phase-F budget past 500k, at a measured 1.4% coverage cost).

## 4. The FSCS mechanics, and why the round-robin has to fall through

Plan §D3 selects within a stratum by sub-k-means at `k = n_h`: cluster the tag group into
as many sub-clusters as it has quota, take the window nearest each sub-centroid, geo-cell
round-robin as the tie-break. #43 flagged the one trap: "nearest window to each
sub-centroid", taken literally, **under-fills** the quota — two sub-centroids in one dense
pocket resolve to the same row, and a rule that just drops the duplicate returns fewer than
`n_h` windows. So the round-robin (`_nearest_representatives`) walks each sub-centroid's
`_NEIGHBOURS` nearest rows and takes the first *unused* one in the least-used geo-cell,
falling through to the next-nearest — a row whose cell is already over-used yields to a
less-used cell, which is what stops a stratum re-concentrating geographically while looking
well spread in descriptor space. A rare residual shortfall (a pocket that exhausts all of a
sub-centroid's neighbours) is closed *exactly* by `_fill_shortfall`, which adds the unused
rows nearest any sub-centroid — fall-through, per #43, never a short corpus.

The coastal floor is enforced **by construction**, not as a post-hoc top-up (#42 §4):
because each stratum runs FSCS separately over its coastal and base rows, a pick's
provenance is fixed by *which* draw selected it, giving the exact per-row provenance the
map requires without any dependence on selection history. The vocabulary is exactly two
labels, `{base, floor:coastline}` — the shape Phase E1's manifest addendum consumes.

## 5. The realized selection

The committed run (`python -m dataset.select`, ~7.4 min wall-clock across 6 processes on an
8-core M1 Pro) drew **50,000 windows**, budget exact, from the 15,834 COGs it touches.

**Provenance.** `base` 49,330 · `floor:coastline` **670** — the coastal floor is met to the
window, across the 16 strata that hold coastal rows (four hold none). Exactly the two-value
vocabulary Phase E1 consumes.

**Windows per COG — the metric #42 binds the sampler to report.** #42 retired the per-COG
cap (`PER_TILE_WINDOW_CAP = 64`) as inert at the debug tier but required the sampler to
*measure* the realized distribution at every tier, so Phase F can decide on evidence — not
the untested 64 — whether the denser scale tier needs a cap reintroduced. Over the 15,834
COGs drawn from:

| mean | median | p99 | max | densest COG |
|-----:|-------:|----:|----:|:------------|
| 3.16 | 2 | 16 | **44** | `…N29_00_E100…` (eastern Tibetan plateau edge) |

The distribution is a long thin tail: 5,437 COGs contribute a single window, the median COG
two, and only a handful exceed ~30. **The densest COG holds 44 windows — under the retired
cap of 64** — so even at its old value the cap would not have bound a single COG at the
debug tier. This is direct evidence for #42's reading: the geo-cell round-robin carries the
spatial-spread job at 10° granularity, and no COG dominates. The scale tier is where this
metric earns its keep (mean ~21/COG at 500k, and the densest COGs hold up to 256 available
windows); the sampler emits the same distribution there by the same code path, so the
fire/no-fire decision is made on the scale-tier numbers when Phase F has them.

**Determinism.** Two full runs from the committed artifacts produced **byte-identical**
selection lists (`diff` clean over all 50,000 rows) — the acceptance criterion, met.

## 6. Retiring the ADR-0001 TRI stratifier in the same change

The map's standing preference — *do not leave two live stratifiers* — is honoured here by
retiring the whole ADR-0001 TRI selection path that `dataset.select` supersedes:

- **`dataset.stratify`** (single-axis TRI quintile strata + quota fill) — deleted. Its two
  *live* members, the exact-zero land screen `zero_fraction` / `is_zero_fill`, moved intact
  to the new **`dataset.screen`** (referenced by the catalog scoring pass and the
  materialize normalizer — they are a pixel screen, not a stratifier).
- **`dataset.pipeline`** (the TRI score → quintile → quota → materialize orchestrator),
  **`dataset.build`** (its smoke-gated CLI), **`dataset.throughput`** (the cloud-burst
  probe built on it), and **`dataset.synthetic`** (its fabricated bucket) — deleted. The
  two genuinely reusable primitives they held moved to honest homes:
  `pipeline.ordered_futures` → **`dataset.concurrency`** (the process/thread fan-out the
  catalog scan still uses, #38), and `build.git_sha` / `build._today` →
  **`dataset.provenance`** (the artifact stamps the catalog, pilot, prevalence, exemplars
  and calibration reports all use).
- The three stale `config.py` constants named by #42/#44 — `N_STRATA`,
  `PER_STRATUM_QUOTA`, `PER_TILE_WINDOW_CAP` (and the now-orphaned `DEBUG_CORPUS_TILES`) —
  removed. `TRI_OVERVIEW_FACTOR` is **kept**: despite its name it is the live x8-overview
  downsample factor the metric scoring pass, the pilot and the A3 calibration all still
  read (ADR-0001 §4), and it is not one of the three the ticket named.

The materialize primitives Phase E rebuilds on — `dataset.normalize`, `dataset.shards`,
`dataset.manifest` — stay; `manifest.build_params` drops the four retired TRI-recipe fields
it recorded. The Phase-1 loader test, which had leaned on `run_pipeline` to fabricate a
corpus, now builds its tiny fixture straight from the storage primitives.
