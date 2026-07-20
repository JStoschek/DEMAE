# Phase-B B3 — flat-wet separation test + `tpi_proxy` re-entry decision

<!-- Cover sheet for the B3 separation analysis. Regenerate numbers with
     `python notebooks/b3_flatwet_separation.py`; the figure needs a matplotlib env. -->

Resolves issue #27 (part of the Phase-B map, #23). Settles the map's one explicit
**TPI re-entry trigger**: the B2 prune ([#26](https://github.com/JStoschek/DEMAE/issues/26))
proposed a 6-dim surviving vector but *demoted the flat-wet/flat-dry axis to open,
unresolved*. This is where it is settled by measurement, against the labelled
exemplar set ([#25](https://github.com/JStoschek/DEMAE/issues/25)).

- Analysis: [`notebooks/b3_flatwet_separation.py`](../../notebooks/b3_flatwet_separation.py)
- Figure: [`b3-flatwet-separation.png`](b3-flatwet-separation.png)
- Inputs: [`exemplars-per-window.csv`](exemplars-per-window.csv) (8 labelled windows),
  [`pilot-frame-per-window.csv`](pilot-frame-per-window.csv) (15k-window standardization basis)
- Provenance: snapshot `2026-07-20`.

## The test

In the descriptor space of the **6 surviving metrics**
(`slope_mean`, `local_relief`, `texture`, `elevation_relief_ratio`,
`mean_elevation`, `geomorphon_entropy` — with `ln(x+1)` on the first two, per #26),
do the 4 flat-wet exemplars separate from the 4 flat-dry? Then: does appending
`tpi_proxy` improve that separation?

Method (chosen so no single knob rigs the answer): exemplars are **z-scored to the
pilot frame** (the space production k-means runs in, not the 8 points); separation
is read three ways — leave-one-out (LOO) nearest-centroid classification, a
Fisher-style margin (wet/dry centroid gap ÷ pooled within-class spread), and a
per-exemplar signed margin `d(dry) − d(wet)` (>0 ⇒ wet side). Every quantity is
recomputed with `tpi_proxy` appended, to isolate its marginal value.

## Result — `tpi_proxy` adds nothing; the trigger fires empty

| | 6 survivors | 6 survivors + `tpi_proxy` |
|---|---:|---:|
| Fisher margin | **1.304** | **1.304** |
| LOO nearest-centroid | **6 / 8** | **6 / 8** |
| Etosha signed margin | −2.763 | −2.762 |

Appending `tpi_proxy` moves every number in the 3rd–4th decimal. It does not fix a
single exemplar, widen the margin, or rescue the one structural miss. **`tpi_proxy`
stays pruned.**

Why it is inert: on the pilot frame `tpi_proxy`'s largest Spearman ρ with any
survivor is only **+0.41** (`elevation_relief_ratio`) — it *is* a near-independent
axis, but the axis it measures is plateau-edge / basin position (its two most
negative exemplars are the dry Llano mesa and Tanezrouft basin), which is
orthogonal to wetness. Independence without relevance.

### Per-exemplar (6 survivors)

| Exemplar | class | pred | signed margin |
|---|---|---|---:|
| Pantanal | wet | wet | +1.70 |
| Mississippi Alluvial Plain | wet | wet | +1.31 |
| Ganges–Brahmaputra delta | wet | wet | +1.16 |
| Tanezrouft Basin | dry | **wet** | +0.52 ⟵ miss |
| Llano Estacado | dry | dry | −0.80 |
| Nullarbor Plain | dry | dry | −1.09 |
| Kalahari sand plain | dry | dry | −1.57 |
| Etosha Pan | wet | **dry** | −2.76 ⟵ miss |

## Decision (B3): `tpi_proxy` **out**, scope-limited

The surviving vector carries **channeled** wetness well — the three water-sculpted
wet cases (delta, floodplain, wetland) land on the wet side with strong margins
(+1.2 … +1.7). The one genuine miss, **Etosha**, is a featureless saturated salt
pan: it has *no terrain-shape signature of wetness* (slope 0.44, texture 0.0001 —
dry-plateau numbers). Its wet signal is real but lives in the **masks**
(`edm_edited_frac` 0.895, plus `wbm_water_frac`), which the frame scores
separately from the terrain vector.

So featureless-pan wet/dry is **out of the terrain vector's scope** — delegated to
the WBM/EDM masks — and `tpi_proxy` provably cannot supply it either. The vector
freezes at the **6 dims from #26, unchanged**. This limitation is recorded in
ADR-0004 (B4, [#28](https://github.com/JStoschek/DEMAE/issues/28)) so the next
reader knows a featureless-pan miss is a known scope boundary, not an oversight.

Copyright / attribution (mask spec §5, Art. 6(b)): *produced using Copernicus
WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018
provided under COPERNICUS by the European Union and ESA; all rights reserved.*
