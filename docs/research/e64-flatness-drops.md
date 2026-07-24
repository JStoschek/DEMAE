# E64 — What the flatness gate drops in strata 2/5/7/17: featureless junk, or informative terrain?

**Ticket:** [Characterize what the flatness gate drops (#64)](https://github.com/JStoschek/DEMAE/issues/64) ·
**Map:** [flatness gate (#62)](https://github.com/JStoschek/DEMAE/issues/62) ·
**Seeded by:** [Gate E FAIL (#54)](https://github.com/JStoschek/DEMAE/issues/54) ·
`c8fb5e5`

The E58-analog — the sibling investigation feeding the loosen-vs-accept decision
([#66](https://github.com/JStoschek/DEMAE/issues/66)), run in parallel with the structural-null model
([#65](https://github.com/JStoschek/DEMAE/issues/65)). Method mirrors E2.5 / E58: dropped windows survive as
columns ([#37](https://github.com/JStoschek/DEMAE/issues/37)), so every would-be member of these strata —
kept *and* dropped — is scored against the committed Phase-D centroids (`artifacts/phase_d/centroids.json`) and the
`degenerate` (`land_std < 3 m`) drops pulled out. Centred on **stratum 5**, the decider (5.32×, the one pair
that survives both baselines — [#54 §5](https://github.com/JStoschek/DEMAE/issues/54)); 2/7/17 carried as
lower-confidence corroboration.

## Verdict in one line

The flatness cap does **not** drop a single population — what it removes is **stratum-dependent**, and no
single metric captures it, so this is a by-eye call as much as a numeric one. Centred on **stratum 5** (the
decider), the drops are the *flatter end of the same smooth low-relief family as the kept tiles* — dropped
texture (**0.001**) is **identical** to kept (**0.001**); the gate cuts purely on
overall relief, and the loss reads as genuinely low-information plains, **correctly** targeted. But in the
*marginal* strata **2 and 17** the dropped tiles are **more** dissected than the kept ones (§2) and carry
visible river/drainage structure by eye (§3) — the more plausible "informative terrain removed". Either way
selection is **not starved**: all four strata keep **≥53×** their allocation
(stratum 5: **118,638** kept vs **1,991** allocated). As with stratum 19 the live
question for [#66](https://github.com/JStoschek/DEMAE/issues/66) is *representativeness*, not scarcity — and
the terrain-quality signal here **splits from** the rate-excess signal ([#65](https://github.com/JStoschek/DEMAE/issues/65)):
the stratum with the largest excess (5) shows the *weakest* case for informative loss, the strata with the
smallest excess (2/17) the strongest.

## The four flat strata at a glance

| h | family | would-be members | degenerate drops | within-stratum | vs corpus | sole-cause | kept pool | kept / alloc |
|---|---|---|---|---|---|---|---|---|
| 5 | near-level plains / basins | 247,681 | 128,511 | 51.9% | 5.32× | 118,845 | 118,638 | 59.6× |
| 2 | low-relief plains with faint drainage | 217,614 | 65,447 | 30.1% | 3.08× | 62,287 | 150,898 | 65.2× |
| 7 | high-altitude flats / plateaus | 212,872 | 67,579 | 31.8% | 3.25× | 64,783 | 139,429 | 53.5× |
| 17 | low-relief lowlands with drainage / water | 168,601 | 50,046 | 29.7% | 3.04× | 41,300 | 105,302 | 57.0× |

Membership and drop counts reproduce E2.5's §5 table (stratum 5: 247,681 members, 128,511 fires, 51.9%,
5.32×) — the pipeline is the same, only the question is new.

## 1. How much is even recoverable — the flatness cap is not the only gate on these tiles

Loosening the `land_std` cap only recovers a window if **no other gate also caught it**. For stratum 5, of
the **128,511** degenerate drops, **118,845**
(92%) are **sole-cause** `degenerate`; **9,596**
also fire `water` and **3,836** also fire `zero_fill`. The water/zero-fill co-fires are
flat *because they are wet or void* — exactly what the flatness screen backstops — and a flatness loosening
would not readmit them. So the recoverable ceiling is the sole-cause count, not the headline drop. Stratum
**17** (the drainage/water family) carries the heaviest co-fire — 8,722
of its 50,046 drops also fire `water` — so it has the least to gain from a flatness
loosening; stratum **7** (ice, no liquid water) has **0** water co-fire.

## 2. Texture separates the strata, not the land_std bands — and it cuts both ways

The gate keys on `land_std`, which is blind to *fine-scale* structure. The revealing comparison is **dropped
vs kept texture within each stratum**, plus what the eye confirms about each:

| h | family | dropped texture | kept texture | dropped geomorphon-entropy | reading |
|---|---|---|---|---|---|
| 5 | near-level plains / basins | 0.001 | 0.001 | 0.584 | dropped texture **≡** kept — a pure `land_std` cut in a texture-homogeneous family |
| 2 | low-relief plains with faint drainage | 0.015 | 0.006 | 1.887 | dropped **more** textured than kept — the cut removes the drainage-bearing tiles |
| 7 | high-altitude flats / plateaus | 0.091 | 0.098 | 2.956 | polar ice plateau — high texture/entropy is amplified surface noise, not landforms |
| 17 | low-relief lowlands with drainage / water | 0.021 | 0.009 | 1.661 | dropped **more** textured than kept — lowland drainage + a heavy `water` co-fire |

Two honest cautions, in **opposite** directions, are why this can't be settled on `texture` alone:

- **`texture` understates drainage** (strata 2, 17). River networks and incised valleys are *meso*-scale —
  they raise `local_relief` and show plainly by eye but barely move a fine-scale texture metric. That the
  dropped tiles here still out-score the kept ones on texture is therefore a *floor* on their structure, not
  a ceiling.
- **`texture` and `geomorphon_entropy` overstate structure on ice** (stratum 7). Its drops score texture
  ~0.091 and entropy ~2.956 — near the corpus top — yet by eye are a uniform
  East-Antarctic/Greenland plateau surface: the metrics are reading amplified sub-metre sastrugi/noise and a
  geomorphon classifier firing on it, not landform variety.

Within **stratum 5** the picture is cleaner: `texture` is flat across the whole `land_std` range and equals
the kept level — there is no textured sub-population hiding among the drops:

| land_std band | dropped windows | texture median | texture p90 | geomorphon-entropy median |
|---|---|---|---|---|
| 0.0-1.0 m | 43,913 | 0.0012 | 0.0104 | 0.5057 |
| 1.0-2.0 m | 49,217 | 0.0013 | 0.0116 | 0.6299 |
| 2.0-2.5 m | 18,974 | 0.0008 | 0.0093 | 0.6062 |
| 2.5-3.0 m | 16,407 | 0.0007 | 0.0083 | 0.6174 |

`land_std` of the stratum-5 drops (p10 **0.4801**, p50 **1.377**, p90 **2.6045** m)
shows the cap slicing a *continuum* down toward zero — **43,913** (34%) sit in the
deepest-flat `<1 m` band. See [`e64-flatness-drops-landstd.png`](e64-flatness-drops-landstd.png).

## 3. By eye — the same family, flatter; with real drainage in 2/17 and blank ice in 7

Median (p10–p90) over `land_std` + the six clustering metrics, stratum-5 dropped vs kept:

| metric | dropped | kept |
|---|---|---|
| land_std | 1.377 (0.48–2.605) | 5.518 (3.409–10.703) |
| slope_mean | 0.475 (0.309–0.638) | 0.568 (0.424–0.691) |
| local_relief | 6.999 (2.214–12.934) | 25.38 (15.69–47.698) |
| texture | 0.001 (0.0–0.01) | 0.001 (0.0–0.006) |
| elevation_relief_ratio | 0.468 (0.299–0.607) | 0.469 (0.355–0.59) |
| mean_elevation | 150.598 (3.809–722.046) | 392.894 (96.125–1131.862) |
| geomorphon_entropy | 0.584 (0.101–1.203) | 0.764 (0.261–1.197) |

The stratum-5 drops are simply the *flatter, lower-relief* end of an already flat stratum — not a
compositionally different, degenerate population. **These tiles are flat, so the shader auto-exaggerates
toward its 60 m target (VE up to ~370×), and that amplification manufactures apparent roughness from
sub-metre noise** — so the eye must be read against the printed `land_std`/`texture`/`VE`, not on its own.

- **Stratum 5** ([`s05-dropped-uniform.png`](e64-flatness/s05-dropped-uniform.png),
  [`s05-dropped-by-landstd.png`](e64-flatness/s05-dropped-by-landstd.png),
  [`s05-kept-uniform.png`](e64-flatness/s05-kept-uniform.png)): smooth low-relief plains and basins. The
  deepest-flat draws (`land_std < 1 m`, 34% of the loss) render as amplified noise —
  featureless blanks correctly dropped; the rest are gentle surfaces with the occasional low-relief drainage
  line. The kept tiles are the *same family* at lower exaggeration — the cut is on relief, not on kind.
- **Stratum 2** ([`s02-dropped-uniform.png`](e64-flatness/s02-dropped-uniform.png)): genuine fluvial
  structure — meandering rivers (some with mapped water), dendritic drainage, incised valleys — interleaved
  with amplified-noise plains. The clearest "informative low-relief terrain" in the four.
- **Stratum 7** ([`s07-dropped-uniform.png`](e64-flatness/s07-dropped-uniform.png)): a uniform polar
  ice-plateau surface; the high texture/entropy metrics are the surface-noise artefact described in §2, not
  learnable landform. Correctly targeted despite the metrics.
- **Stratum 17** ([`s17-dropped-uniform.png`](e64-flatness/s17-dropped-uniform.png)): lowland drainage
  networks and incised channels **plus** coastal/lake tiles (the `water` co-fire) — a genuine-drainage
  population mixed with correctly-flagged water.

Tile ids/coords are in [`e64-flatness/manifest.json`](e64-flatness/manifest.json) (seed `20260723`).

## 4. Where it is

The stratum-5 drops span **6,357** source COGs across **362**
geocells, latitude -89.4–83.6 — the world's arid plains and basins
(heaviest: `-070_-040`, `+010_+010`, `+050_+040`, `+020_-030`, `+030_+010`), not one artifact-ridden tile.
Stratum 7's drops, by contrast, concentrate over the Antarctic/Greenland ice sheets
(`+110_-080`, `+090_-080`, `+120_-080`) — consistent with the
ice-surface reading.

## What this hands to the decision ([#66](https://github.com/JStoschek/DEMAE/issues/66))

- **The drop is a stratum-dependent mix, not uniform junk.** Stratum 5 (the biggest rate excess) removes
  genuinely low-information smooth plains — texture-identical to the kept tiles, the same family flatter;
  strata 2/17 (marginal excess) remove the *more*-dissected, drainage-bearing low-relief tiles — the real
  "informative terrain removed"; stratum 7 removes ice-plateau surface whose high metrics are noise.
- **The terrain axis and the rate axis point opposite ways across strata.** The strongest ACCEPT case on
  terrain (stratum 5) is the strongest LOOSEN case on rate excess, and vice-versa for 2/17 — the decision
  must reconcile #64 and #65, not read either alone.
- **The recoverable ceiling is the sole-cause count** (5: 118,845; 17 much less,
  gated by its water co-fire), not the headline drop.
- **Nothing is starved.** All four kept pools are ≥53× their allocations, so — as with stratum
  19 — the case for change rests on representativeness, not scarcity.
- **If loosened, the knob is the `land_std` cap** (global lower value vs per-stratum exemption — the shape
  [#63](https://github.com/JStoschek/DEMAE/issues/63) pinned). A per-stratum exemption for 2/17 (where the
  informative loss concentrates) fits this evidence better than a global re-tune that would also readmit
  stratum 5's low-information plains. Whether a near-constant surface carries MAE-learnable signal is
  ADR-0002 §2's premise, which this ticket does not reopen.
