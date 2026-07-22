# The stratification frame — what k-means actually sees

Resolves issue [#40](https://github.com/JStoschek/DEMAE/issues/40) (Phase-D map,
[#39](https://github.com/JStoschek/DEMAE/issues/39)). ADR-0004 froze *which* six metrics
stratify the corpus and left *which rows, in what units* to Phase D. This is that
measurement, and the three decisions it settles: the scaler's fit frame, and what
becomes of Phase C's two tags — `water_affected` and `coastline`.

It also **amends plan §C2 twice**. Both amendments replace an intention written before
the metric vector existed with what the catalog turns out to support, under the Phase-D
map's standing preference that measurement may redraw §D.

Artifact: [`artifacts/phase_d/scaler.json`](../../artifacts/phase_d/scaler.json),
produced by `python -m dataset.frame`. Code: [`src/dataset/frame.py`](../../src/dataset/frame.py).

Copyright / attribution (mask spec §5, Art. 6(b)): *produced using Copernicus WorldDEM-30
© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under
COPERNICUS by the European Union and ESA; all rights reserved.*

## 1. Method

The whole kept frame — **3,052,735 windows**, `data/catalog_gated/`, source
`code_version` `ad46e96`, gates as locked in
[`gate-attrition.md`](gate-attrition.md) §3. No sampling: every distributional number
below is the full frame. ADR-0004 §1's transforms are applied first
(`ln(x+1)` on `slope_mean` and `local_relief`, the other four raw), then a z-score —
**standardize, don't whiten**, so no PCA enters the distance.

Where a *partition* is needed, it comes from scikit-learn's k-means at **k = 40**,
which is the map's working figure, **not a chosen k** — that is
[#41](https://github.com/JStoschek/DEMAE/issues/41). Nothing below depends on 40 being
right; the tag questions are about whether a group is a *region of descriptor space*,
and that answer does not turn on the exact cluster count.

Partitions are compared by **adjusted Rand index** (ARI) over the rows both fits
contain: 1.0 is the same partition, 0.0 is chance. §4 establishes the noise floor that
makes those numbers readable.

## 2. The scaler — fitted over all 3,052,735 kept rows

| metric | transform | mean | sd |
|---|---|---:|---:|
| `slope_mean` | `ln(x+1)` | 1.6203 | 0.8066 |
| `local_relief` | `ln(x+1)` | 4.7555 | 1.1894 |
| `texture` | — | 0.1458 | 0.1372 |
| `elevation_relief_ratio` | — | 0.4292 | 0.1301 |
| `mean_elevation` | — | 815.1409 | 937.0323 |
| `geomorphon_entropy` | — | 2.3630 | 0.5631 |

**No subsample, and not for accuracy.** A fixed 100k subsample reproduces every mean to
within 0.35% relative and every sd to within 0.47% — subsampling would have been safe.
It is rejected because the full pass over 3.05 M rows takes **1.5 seconds**, so the
subsample buys nothing, while making a sampling seed part of a contract that has to
hold for the life of the project.

**Tagged rows are in the fit frame.** Fitting the scaler on the 2,893,724 non-`water_affected`
rows instead moves the six means by at most **0.029 sd** (`mean_elevation`; four of six
under 0.03). So the water question is not really a scaler question — it is settled in §3
on other grounds, and the scaler simply follows the frame k-means fits, which is the
only way the two can be guaranteed to agree.

### 2.1 The artifact carries a frame fingerprint

Phase F re-runs Phase D at a 200-500k budget against these strata, so the scaler is
**loaded, not re-derived** — a scaler silently re-fitted over a moved frame shifts every
centroid without changing a line of code, and produces a complete, plausible, wrong
corpus rather than an error. The artifact therefore records the frame it was fitted
over, and `frame.read_scaler(..., fingerprint=...)` refuses a mismatch:

| field | catches |
|---|---|
| `n_kept` | rows added or dropped |
| `gate_thresholds` | the same rows re-gated differently |
| `catalog_code_version` | the same gates over re-scored metrics |

## 3. The two tags

### 3.1 Where they sit in standardized space

Mean shift against all other kept rows, in sd:

| dimension | `water_affected` (159,011 · 5.21%) | `coastline` (40,815 · 1.34%) |
|---|---:|---:|
| `slope_mean` | +0.019 | +0.491 |
| `local_relief` | −0.187 | +0.459 |
| `texture` | **+0.030** | +0.178 |
| `elevation_relief_ratio` | −0.533 | −0.311 |
| `mean_elevation` | −0.551 | −0.726 |
| `geomorphon_entropy` | +0.506 | −0.085 |
| mean ‖z‖ from the global centroid | 2.124 (others 2.296) | 2.384 (others 2.286) |

The two tags overlap heavily: **28,452** of the 40,815 coastal windows (69.7%) are also
`water_affected`.

### 3.2 Amendment 1 — `water_affected` is fitted like any other row

> **Plan §C2**, as written: windows kept with water > 0.05 are "excluded from
> texture-based stratification distances."
>
> **Amended to**: `water_affected` windows are fitted like any other row. The tag is
> carried into the Phase-E manifest as a filter column.

Three reasons, in order of weight:

1. **The premise does not hold.** The exclusion exists to keep wet windows from
   polluting *texture*, and `texture` is the dimension where they differ least —
   **0.030 sd**, 17× smaller than the largest gap of the six. What actually
   distinguishes them is elevation and landform mix (`mean_elevation` −0.55,
   `elevation_relief_ratio` −0.53, `geomorphon_entropy` +0.51): they are lower, flatter
   and more mixed. That is what "wet" should look like, and it is signal the strata
   ought to see, not contamination.
2. **They are not an outlying blob.** Their mean distance from the global centroid is
   *smaller* than everyone else's (2.124 vs 2.296) — they sit slightly toward the middle
   of the cloud.
3. **k-means cannot express the exclusion anyway.** Dropping one of six dimensions for
   one row in twenty needs a per-row masked distance, which breaks the centroid update
   and rules out scikit-learn — a hand-written and separately validated k-means, for an
   effect measured at 0.030 sd.

Measured directly: fitting k=40 **without** the 159,011 rows and comparing where the
remaining rows land scores **ARI 0.698**, against a noise floor of 0.607 (§4). Excluding
them changes the partition *less* than re-running the same fit with a different seed.

Under a k=40 fit they scatter across **21 strata** to reach 90% of their mass, and the
most water-affected stratum is **15.8%** water-affected. There is no water stratum to
protect or to build.

### 3.3 Amendment 2 — `coastline` is a reporting axis, not a stratum

> **Plan §C2**, as written: coastal windows are "tagged as their own reportable stratum."
>
> **Amended to**: `coastline` is fitted like any other row, and the tag becomes a
> reporting axis *across* strata plus a boost pool in allocation. The allocation rule
> ([#42](https://github.com/JStoschek/DEMAE/issues/42)) owes a **minimum coastline count**
> in the selection — the presence guarantee a carve-out would have given.

`coastline` is defined by [#34](https://github.com/JStoschek/DEMAE/issues/34) as
`exact_zero_frac < 0.75 AND wbm1_frac > 0` — a fact about **where a window is**, not
about what its terrain looks like. The six frozen metrics cannot see the ocean; they see
slope and relief. The measurement bears that out: coastal windows scatter across
**21 strata** to reach 90% of their mass, and the most coastal stratum is **4.9%**
coastline — thinner even than water's 15.8%. No cluster wants to be the coast cluster.

Their descriptor signature is real but not localising: steep, high-relief, near sea level
(`slope_mean` +0.49, `local_relief` +0.46, `mean_elevation` −0.73) reads correctly as
cliff-and-headland, and plenty of inland terrain shares it.

So a carved-out 41st stratum would be a stratum that is **internally heterogeneous in
exactly the descriptors the whole design rests on**, and it would make `k` mean two
things at once — *k* descriptor clusters plus one tag class, with 40,815 rows pulled out
of the geometry that #41 is about to sweep. Fitting without them scores **ARI 0.656**
against the 0.607 floor: their presence in the fit is, again, not measurable.

## 4. The noise floor — and a warning for #41

The tag comparisons above only mean something against a control, and the control is
uncomfortable. Two k=40 fits of the **same frame**, differing only in seed:

| fit | ARI between two runs | mean matched-centroid displacement |
|---|---:|---:|
| full 3.05 M frame, `n_init=1` | 0.571 | 0.351 sd |
| 500k subsample, `n_init=3` | 0.627 | 0.321 sd |
| **full 3.05 M frame, `n_init=25`** | **0.607** | **0.274 sd** |
| 500k subsample, `n_init=25` | 0.889 | 0.076 sd |

Against that floor, both tag effects are **above** it:

| comparison | ARI |
|---|---:|
| noise floor — all rows vs all rows, two seeds | 0.607 |
| `water_affected` excluded vs included | **0.698** |
| `coastline` excluded vs included | **0.656** |

Excluding either group agrees with the all-rows fit *more* than two all-rows fits agree
with each other. That is the whole case for §3.

**The floor itself is the finding to carry forward.** The two full-frame `n_init=25`
fits land **0.06% apart in inertia** (2,546,257 vs 2,544,784) while agreeing on only
~61% of pairs. That is not a convergence failure — it is genuine multi-modality: at
k=40 this catalog admits many near-equally-good partitions. Three consequences for
[#41](https://github.com/JStoschek/DEMAE/issues/41):

- **An inertia difference between neighbouring k must clear this floor before it means
  anything.** An elbow drawn on differences of this size would be drawing noise.
- **Restarts are the lever, and they are affordable.** A full-frame fit costs 7-9 s in
  float32 at `n_init=1`; `n_init=25` costs ~4 minutes. Whatever k is chosen, the shipped
  fit should be generous with restarts.
- **Determinism comes from the committed centroids, not from the fit.** Re-running the
  fit is *not* guaranteed to reproduce the partition, so the artifact is the contract —
  exactly as it is for the scaler (§2.1). This is consistent with the map's standing
  bar; it just means the bar is met by persisting, not by seeding.

## 5. What this hands forward

- **[#41](https://github.com/JStoschek/DEMAE/issues/41) (choose k)** — sweeps the frame
  defined here: all 3,052,735 kept rows, all six standardized dimensions, both tags
  included. Must contend with §4's noise floor, and should ship centroids as a committed
  artifact beside the scaler.
- **[#42](https://github.com/JStoschek/DEMAE/issues/42) (allocation)** — inherits a hard
  requirement from §3.3: a **minimum coastline count** in the 50k, spread across
  whatever strata the coastal rows fall in. It also inherits an occupancy warning: at
  k=40 the strata came out between **25,641 and 160,255** rows, a 6.3× spread, which is
  what Neyman allocation and the §D2 floor will have to be composed against.
- **[#44](https://github.com/JStoschek/DEMAE/issues/44) (sampler)** — builds on
  `dataset.frame`; must add **`scikit-learn` to `pyproject.toml`'s runtime dependencies**
  (it is currently only present in the dev environment, undeclared), and retire the
  ADR-0001 TRI-era `stratify.py` rather than leaving two live stratifiers.
- **Phase E** — `water_affected` and `coastline` are both manifest filter columns, not
  frame decisions. E2's validation report can cut its stratum tables by either.
