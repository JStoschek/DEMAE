# Gate attrition — locking the Phase-C quality-gate thresholds

Resolves issue [#36](https://github.com/JStoschek/DEMAE/issues/36) (Phase-C map,
[#32](https://github.com/JStoschek/DEMAE/issues/32)). ADR-0004 parked the gate
thresholds for Phase C, to be set "against the A1 prevalence table and C3's measured
global distributions." C3 ([#35](https://github.com/JStoschek/DEMAE/issues/35)) put the
raw catalog on disk; this is the attrition analysis over it, and the thresholds it locks.

Companion data: [`gate-attrition-per-geocell.csv`](gate-attrition-per-geocell.csv)
(472 rows, one per 10° geo-cell) and [`gate-attrition-per-band.csv`](gate-attrition-per-band.csv)
(6 rows, one per latitude band).

Copyright / attribution (mask spec §5, Art. 6(b)): *produced using Copernicus WorldDEM-30
© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under
COPERNICUS by the European Union and ESA; all rights reserved.*

## 1. Method

The full raw catalog — **4,386,720 windows over 26,450 COGs**, `data/catalog/`,
`code_version` `ad46e96`, ungated — evaluated against the four drop gates and the two
non-destructive tags. No sampling: every number below is the whole frame.

```
zero_fill  := exact_zero_frac >= 0.75                       # ADR-0001 §3 + Amendment 2
degenerate := land_std < 3.0                                # ADR-0002 §104
water      := wbm_water_frac > 0.30                         # plan §9 [project]
edited     := edm_edited_frac > 0.25 | flm_fill_frac > 0.25  # plan §9 [project]
```

Two quantities are reported per gate, and the difference between them is the point of
this document:

- **fires** — windows the gate's predicate is true of, in isolation.
- **sole cause** — windows *only* that gate drops. This is the gate's real cost: remove
  the gate and exactly these windows return.

`land_std` is **NaN when a window has no land pixel** (`catalog.land_std`), and
`NaN < 3.0` is false. So `degenerate` cannot fire on an all-ocean window — 727,252
windows (16.58%) are in that state. This asymmetry drives §2.

## 2. `zero_fill` is inert — the water cap is the operative ocean gate

| gate | fires | sole cause | keep-rate cost |
|---|---|---|---|
| `zero_fill` ≥ 0.75 | 790,152 (18.01%) | **0** | **0.00%** |
| `degenerate` < 3 m | 428,050 (9.76%) | 367,112 | 8.37% |
| `water` > 0.30 | 901,958 (20.56%) | 72,666 | 1.66% |
| `edited` > 0.25 | 65,529 (1.49%) | 60,452 | 1.38% |

**Every one of `zero_fill`'s 790,152 windows is also caught by the water cap — 100.00%,
zero exceptions in 4.39 M.** Removing `zero_fill` from the gate set entirely leaves the
kept count byte-identical at 3,052,735. Sweeping its threshold across the whole plausible
range moves nothing:

| `zero_fill` ≥ | 0.50 | 0.60 | 0.75 | 0.90 | 0.999 |
|---|---|---|---|---|---|
| keep | 69.59% | 69.59% | 69.59% | 69.59% | 69.59% |

The mechanism: a window at ≥75% exact-zero is Copernicus water painted to `0.0`, which
WBM classifies as 1/2/3 — so `wbm_water_frac` is at or near 1.0 and the water cap takes
it first. `degenerate` cannot be the cover, because those same windows have no land pixel
and so a NaN `land_std`.

**This is a measurement, not a theorem, and it runs opposite to C2's.**
[C2](ocean-gate-crosscheck.md) proved `WBM == 1` ⊂ `elevation == 0.0` at the **pixel**
level. At the **window** level with these thresholds the containment inverts —
`water` ⊃ `zero_fill` — because the water cap aggregates all three water classes while
the elevation test needs a 75% supermajority. Nothing guarantees it holds under other
thresholds; it holds under *these*, over the whole frame.

The consequence worth stating plainly: **with the plan defaults, ocean removal rests
entirely on WBM** — the same mask C2 caught classifying the Salar de Uyuni, genuine
terrain at 3,656 m, as water. That is why `zero_fill` is kept rather than deleted (§3.1).

With the water cap switched off, `zero_fill` becomes load-bearing again and its threshold
starts to matter — 0.30 → 0.999 spans 70.27% → 72.24% keep. The backstop works; it is
simply never reached at the locked settings.

## 3. The locked thresholds

| gate | locked value | sole cause | status |
|---|---|---|---|
| `zero_fill` | `exact_zero_frac >= 0.75` | 0 | §9 default confirmed |
| `degenerate` | `land_std < 3.0` m | 367,112 (8.37%) | §9 default confirmed (ADR-0002-bound) |
| `water` | `wbm_water_frac > 0.30` drop; `> 0.05` → `water_affected` | 72,666 (1.66%) | §9 default confirmed |
| `edited` | `edm_edited_frac > 0.25` **or** `flm_fill_frac > 0.25` | 60,452 (1.38%) | §9 default confirmed |
| `coastline` | `exact_zero_frac < 0.75 & wbm1_frac > 0` | — | C2 composition; **lower edge dropped** |

**Kept: 3,052,735 of 4,386,720 = 69.59%.** Four of the five knob-table §9 defaults are
confirmed unamended. The single amendment is the coastline stratum's lower edge (§3.5).

Derived tags over the kept frame: `water_affected` flags **159,011** windows (5.21% of
kept); the coastline stratum holds **40,815** windows across 5,337 COGs and 319 geo-cells
(1.34% of kept).

### 3.1 `zero_fill` — kept at 0.75 as a WBM-independent backstop

Locked at the ADR-0001 §3 value. **Recorded explicitly as redundant-at-defaults
insurance, not a tuned threshold** — no value between 0.50 and 0.999 changes a single
window, so 0.75 is unfalsifiable from this catalog and is locked on principle.

The principle: it is the one ocean test that does not trust WBM. Days after C2 measured
WBM mislabelling a 3,656 m salt flat, deleting the mask-independent check to save nothing
would be a poor trade. It costs zero coverage and zero compute over columns already
stored.

A future session must not read the 0.75 as evidence-backed. It is inherited.

### 3.2 `degenerate` — kept at 3 m, and it is not a free knob

[ADR-0002](../adr/0002-storage-format-and-manifest.md) §48–49 and §104 fix ε ≥ 3 m: the
threshold is what "neutralise[s]" int16 quantisation banding, "so no *kept* tile is coarse
enough to band." **Lowering it below 3 m reintroduces the banding ADR-0002 exists to
prevent** — it would need that ADR amended first, not a knob-table edit.

The attrition evidence supports it independently. This is the largest marginal gate
(8.37%), so the risk that matters is whether it hollows the flat-terrain family. It does
not:

- **26.99%** of kept windows sit at `land_std` 3–10 m, and 8.83% in the 3–5 m sliver
  immediately above the cut. Flat terrain is abundantly represented.
- The geo-cells losing most to `degenerate` alone are **Antarctic ice plateau**
  (`-060_-080` 75.4%, `-180_-080` 71.7%, `+170_-080` 67.0%, all lat −80/−90). That
  surface is genuinely featureless, not under-sampled. Only 2 of them keep under 5%.
- The hollow-cell count is **flat at 47 across the entire 1–5 m sweep** — those 47 are the
  atoll cells of §4, which this gate never touches.

| `land_std` < | 1 m | 2 m | 3 m | 4 m | 5 m |
|---|---|---|---|---|---|
| keep | 76.55% | 73.06% | **69.59%** | 66.40% | 63.45% |

### 3.3 `water` — kept at 0.30 drop / 0.05 flag

A window more than 30% water fill has more than a third of its pixels carrying no
elevation signal. For MAE pretraining that is a third of patches being constant, which is
what the gate exists to prevent.

**The salt-flat hazard C2 raised is real but small, and it is not a hollowing.** Uyuni is
classified **WBM 2 (inland lake)** — 92.3% mean over its water-dropped windows, and
`exact_zero_frac` never exceeds 0.0000, so it is never mistaken for sea. Of the salar
COG's 141 water-dropped windows, **122 are also genuinely `degenerate`** (flat salt pan,
median `land_std` 1.08 m) — they are flat by measurement, not by misclassification.
**Only 19 windows are lost to the water cap alone.**

The cap's global sole-cause loss decomposes as: ocean 41,276, lake 27,311, river 4,079.
Of the 72,666 total, 39.6% have no sea pixel at all and 4.0% sit above 1,000 m — and the
high-elevation cases are dominated by genuine large lakes (Tibetan Plateau cells
`+080_+030` and `+090_+030` at ~4,700 m; East African rift lakes at `+030_-010`), not by
playas.

Alternatives were priced and declined: raising to 0.50 recovers 35,623 windows
(→ 70.40%); exempting `exact_zero_frac == 0` recovers 28,762 (+0.66 pp) but readmits the
Caspian, the Great Lakes and the Tibetan lakes, which are real water.

The `> 0.05` `water_affected` flag keeps the milder cases visible and **non-destructive** —
Phase D can revisit any of this from stored columns without re-scoring.

### 3.4 `edited` — kept at 0.25; FLM is inert, EDM carries it

**FLM is a second inert gate.** `flm_fill_frac > 0.25` fires on 38,788 windows and has
**0 sole cause** — every non-TanDEM-fill window is already caught by EDM or another gate.
It is locked at 0.25 on the same reasoning as §3.1: a free provenance backstop.

So the live threshold is EDM, 65,529 fires and 22,404 sole cause on its own. It
concentrates at the pole, and that concentration was checked rather than assumed:

| band | windows | keep | `zero_fill` | `degenerate` | `water` | `edited` |
|---|---|---|---|---|---|---|
| 0–50 | 3,036,416 | 69.27% | 583,993 | 275,783 | 656,014 | 45,810 |
| 50–60 | 393,440 | 76.25% | 57,503 | 19,136 | 73,625 | 5,292 |
| 60–70 | 458,496 | 75.98% | 76,418 | 21,426 | 92,496 | 2,335 |
| 70–80 | 364,160 | 60.58% | 63,804 | 74,606 | 70,275 | 1,125 |
| 80–85 | 105,408 | 64.51% | 8,434 | 27,672 | 9,548 | 315 |
| 85–90 | 28,800 | **43.58%** | 0 | 9,427 | 0 | **10,652** |

Band 85–90 is the worst in the frame and EDM is its dominant lever — switching the cap
off takes it to 67.3%. That is accepted: EDM-edited pixels are Copernicus's manual
corrections, heaviest over ice-sheet voids, and they are not measured terrain. **The band
is not starved** — it still keeps 12,552 windows across 1,800 COGs (~7 per COG) and is
0.66% of the frame. Raw fractions stay stored, so Phase D can loosen per-stratum if a
polar stratum turns out thin.

### 3.5 `coastline` — the lower edge is dropped

C2 settled the composition; only the band's lower edge `lo` was open, provisionally 0.05.
**It is removed.** The stratum is:

```
coastline := exact_zero_frac < 0.75 & wbm1_frac > 0
```

Two measurements retire the knob:

1. **`lo` buys no purity, only a smaller stratum.** Median `land_std` sits at ~50 m at
   *every* `lo` from 0.0 to 0.20. C2's purity problem — flat delta river at median 2.23 m
   against 69.99 m for real shoreline — was solved entirely by the sea-touching
   `wbm1_frac > 0` term. `lo` adds nothing to it.
2. **`lo = 0.0` is a no-op.** **Zero** kept windows have `exact_zero_frac == 0` alongside
   `wbm1_frac > 0`. That is C2's pixel invariant showing through at window level: a WBM-1
   pixel *is* an exact-zero pixel, so `wbm1_frac > 0` already implies
   `exact_zero_frac > 0`. The lower edge has no independent meaning.

| `lo` | band | kept | % of kept | median `land_std` |
|---|---|---|---|---|
| 0.0 | 93,502 | 40,815 | 1.34% | 50.87 |
| 0.01 | 87,352 | 35,249 | 1.15% | 50.49 |
| 0.05 | 77,847 | 26,811 | 0.88% | 49.82 |
| 0.20 | 57,558 | 8,908 | 0.29% | 50.01 |

## 4. Risk #3 — no region or terrain family is hollowed out

The plan's risk #3 asks that the gates not silently remove a region or a terrain family.
Checked four ways; **nothing fires.**

**Geo-cells that keep nothing: 47 of 472** (220 COGs, 1.18% of frame rows). Every one is
**≥95% `zero_fill`**, with a maximum degenerate share of 10%. The largest by COG count:

| geo-cell | COGs | windows | `zero_fill` share | region |
|---|---|---|---|---|
| `+160_+000` | 17 | 4,352 | 99.9% | Marshall Islands |
| `+140_+000` | 16 | 4,096 | 100.0% | Micronesia |
| `+170_+000`, `+170_-010` | 14 each | 3,584 each | 100.0% | Tuvalu / Fiji |
| `-140_-030` | 13 | 3,328 | 100.0% | French Polynesia |
| `+050_-010` | 11 | 2,816 | 99.7% | Seychelles / Chagos |

**C3's benign read is confirmed.** A 1° COG exists because a speck of land sits in it; at
224 px (6.7 km) every window is ≥75% sea. This is the ocean gate removing ocean, **not a
cap hollowing a terrain family** — atoll terrain has no 6.7 km window to give, and no
threshold conjures one. No per-region cap-loosening is warranted. (Per-*stratum*
loosening needs strata, which is Phase D.)

**Terrain families.** Flat terrain: 26.99% of kept windows at `land_std` 3–10 m (§3.2).
Salt flats: Uyuni loses 19 windows uniquely (§3.3). High-elevation lakes are genuinely
water (§3.3). Ice plateau is genuinely featureless (§3.2).

**Latitude bands.** All six retain a workable frame; the worst, 85–90°, keeps 12,552
windows over 1,800 COGs (§3.4).

**COGs.** 24,118 of 26,450 keep at least one window, median 122 each. The 2,332 that keep
none are the atoll and open-water COGs above.

## 5. What this hands forward

- **C5 ([#37](https://github.com/JStoschek/DEMAE/issues/37))** implements these five
  predicates as filter columns over the stored catalog and ships the summary notebook.
  All fractions remain stored; **no window is destructively removed.**
- **Phase D** inherits a 3,052,735-window sampling frame with `water_affected` on 159,011
  windows and a 40,815-window coastline stratum, and can revisit any threshold here from
  stored columns without re-scoring.
- **Two gates are inert at these settings** — `zero_fill` (§3.1) and `flm_fill_frac`
  (§3.4). Both are deliberately retained as backstops. Neither should be read as a tuned
  threshold, and the redundancy of the first should be re-checked if the water cap ever
  moves.
