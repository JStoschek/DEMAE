# E58 — Stratum-19 hollowing: good alpine terrain, or corrupted tiles?

**Ticket:** [Characterize the stratum-19 hollowing (#58)](https://github.com/JStoschek/DEMAE/issues/58) ·
**Map:** [edit/fill gate (#57)](https://github.com/JStoschek/DEMAE/issues/57) ·
**Seeded by:** [Gate E FAIL (#54)](https://github.com/JStoschek/DEMAE/issues/54) ·
`23d29ce`

One of two investigations feeding the loosen-vs-accept decision ([#60](https://github.com/JStoschek/DEMAE/issues/60)).
Method mirrors E2.5 exactly: dropped windows survive as columns ([#37](https://github.com/JStoschek/DEMAE/issues/37)),
so every stratum-19 candidate — kept or dropped — is scored against the committed Phase-D
centroids (`artifacts/phase_d/centroids.json`) and the `edited`-gate drops are pulled out.

## Verdict in one line

The `edited` gate removes **11,787** of stratum 19's **65,225** would-be members
(**18.1%**, **12.1×** the corpus `edited` rate) — and by
the metrics and — across the bulk of the distribution — by eye these are **the same high-alpine terrain
family as the kept tiles** (§3 carries the one honest caveat, at the extreme high-EDM tail). The loss is
real terrain, but it does **not** starve selection: stratum 19 still keeps
**53,437** candidates against an allocation of **1,106** tiles
(**48.3×** the pool it needs). The open risk is *representativeness* —
whether the 18% that is removed skews which alpine terrain the 1,106 sample from —
not availability. That risk is what [#60](https://github.com/JStoschek/DEMAE/issues/60) must weigh.

## 1. Edit and fill are co-located — the knob is the combined screen, not one mask

All **11,787** drops are **sole-cause** `edited` — none of the other four gates
(`zero_fill`/`degenerate`/`water`/`masks_missing`) co-fires, so the whole loss is attributable to this one
screen. But *within* the screen, edit and fill fire almost together:

| over the 0.25 cap | windows | share of drops |
|---|---|---|
| EDM (`edm_edited_frac` > 0.25) | **11,787** | 100% |
| FLM (`flm_fill_frac` > 0.25)  | 11,036 | 94% |
| **both** limbs over cap | **11,036** | 94% |
| EDM-only (edit fires, fill does not) | 751 | 6% |
| FLM-only (fill fires, edit does not) | 0 | 0% |

The gate is `(edm > 0.25) OR (flm > 0.25)`, so to *recover* a window both limbs must fall under their caps.
Two consequences for the loosening knob:

- **Touching fill alone recovers nothing** — FLM-only is **0**; fill is never the sole
  cause, exactly as C4 noted, so a fill-lenient rule is the wrong knob.
- **Touching edit alone recovers only 751** (6%) — because
  **11,036** of the drops (94%) are over *both* caps and the FLM limb would
  still catch them. In this terrain, editing and fill land on the **same pixels** (glacier / permanent-snow /
  radar-shadow voids get interpolated *and* flagged edited), so they are not independent knobs.

The real knob is therefore the **combined edit-or-fill screen** — a single loosened cap applied to both limbs
(global or per-stratum), not a per-mask split. A per-mask rule, in either direction, is the wrong instrument here.

## 2. A continuum sliced by the cap, not a wall of synthetic tiles

EDM editing-fraction of the dropped windows (quantiles): p10 **0.2803**,
p50 **0.4625**, p90 **0.8171**, max **1.0**.

| EDM band | dropped windows |
|---|---|
| 0.25–0.35 (just over the cap) | **3,279** |
| 0.35–0.50 | 3,332 |
| 0.50–0.75 | 3,399 |
| 0.75–1.00 | 1,777 |

**3,279** of the 11,787 drops (28%) sit in the first band just above 0.25.
The kept tiles press right up against the cap from below (kept EDM p90 **0.1465**,
p99 **0.2366**), so `edited_max = 0.25` is cutting through a *continuous*
editing gradient in this terrain family — not separating a clean population from a synthetic one. See
[`e58-stratum19-hollowing-edm.png`](e58-stratum19-hollowing-edm.png). This is the shape that makes a *modest*
per-stratum loosening (e.g. lifting the cap toward ~0.35–0.50 for high-alpine strata) plausible without
scooping up near-fully-synthetic tiles — though whether even a heavily-edited pixel carries trustworthy
elevation is [#59](https://github.com/JStoschek/DEMAE/issues/59)'s question, and this ticket does not pre-judge it.

## 3. The dropped terrain is the same family as the kept terrain

Median (p10–p90) over the six clustering metrics, dropped vs kept:

| metric | dropped | kept |
|---|---|---|
| mean_elevation | 3971.937 (3128.775–5046.919) | 4146.81 (3160.097–5168.343) |
| local_relief | 1897.789 (1372.722–2546.519) | 996.663 (578.258–1636.931) |
| slope_mean | 32.812 (27.934–36.943) | 22.382 (13.38–30.033) |
| texture | 0.259 (0.207–0.317) | 0.226 (0.119–0.329) |
| geomorphon_entropy | 1.519 (1.354–1.731) | 1.705 (1.471–2.026) |

The dropped windows are, if anything, slightly *higher and higher-relief* — the most glaciated end of an
already high-alpine stratum, exactly where editing is heaviest. They are not a different, degenerate
population that happens to land in stratum 19.

**By eye** ([gallery](e58-hollowing/)): across the bulk of the distribution the dropped surfaces are
*indistinguishable from the kept ones* — sharp ridgelines, cirques and valley glaciers, sound
glaciated-alpine terrain. Two honest caveats, both visible in [`dropped-by-edm.png`](e58-hollowing/dropped-by-edm.png):
(a) a minority at the extreme high-EDM tail (≥~70%) show visible smoothing consistent with heavy
interpolation; (b) **EDM fraction is a weak predictor of that** — several fully-edited (100%) tiles are still
crisp terrain while some ~70%-edited ones look smeared. So the hillshade cannot separate "trustworthy
elevation" from "edited but plausible" — which is exactly why *the terrain looks fine* is necessary but not
sufficient to justify loosening, and why [#59](https://github.com/JStoschek/DEMAE/issues/59) is the pivot.

- [`dropped-uniform.png`](e58-hollowing/dropped-uniform.png) — a representative random sample of the loss.
- [`dropped-by-edm.png`](e58-hollowing/dropped-by-edm.png) — spanning the cap→max EDM range; judge whether
  the *most-edited* end is where the terrain finally degrades.
- [`kept-uniform.png`](e58-hollowing/kept-uniform.png) — the kept stratum-19 tiles, for comparison.

Tile ids/coords are in [`e58-hollowing/manifest.json`](e58-hollowing/manifest.json) (seed `20260723`).

## 4. Where it is

The drops span **431** source COGs across **54** geocells,
latitude -86.78–68.84, longitude -174.16–175.84 —
the world's glaciated ranges, not one artifact-ridden tile. Heaviest geocells:
`+070_+030` (5242), `+090_+020` (1706), `+080_+020` (975), `+090_+030` (591), `+080_+040` (545), `+100_+030` (543).

## What this hands to the decision

- **Loosening is a coherent option, and its knob is the combined edit-or-fill cap** (global or per-stratum) —
  edit and fill are co-located here (94% over both caps), so a per-mask split
  is the wrong instrument. Recovers up to 11,787 genuine-looking alpine windows.
- **Accepting is also coherent**: the kept pool is 48.3× the allocation, so
  nothing is starved; the case for change rests on representativeness, not scarcity.
- **The pivotal unknown is [#59](https://github.com/JStoschek/DEMAE/issues/59)** — whether an edited/filled
  pixel carries trustworthy elevation. If it does, these tiles are safe to admit and loosening is well-motivated;
  if it does not, the hollowing is correct data-quality behaviour however good the terrain looks. This ticket
  establishes that the terrain is *real and wanted*; it deliberately leaves *is the elevation trustworthy* to #59.
