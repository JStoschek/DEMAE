# Ocean gate cross-check — `WBM == 1` vs the exact-zero elevation test

Resolves issue [#34](https://github.com/JStoschek/DEMAE/issues/34) (Phase-C map,
[#32](https://github.com/JStoschek/DEMAE/issues/32)). ADR-0004 parked whether a WBM
"sea/ocean" classification should **supersede** [ADR-0001 §3](../adr/0001-corpus-construction-recipe.md)'s
exact-zero ocean drop, noting it "needs a per-pixel cross-check, not a Phase-B pruning
question." This is that cross-check.

Companion data: [`ocean-gate-crosscheck.csv`](ocean-gate-crosscheck.csv) — one row per
224² window (2,880 rows).

Copyright / attribution (mask spec §5, Art. 6(b)): *produced using Copernicus WorldDEM-30
© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under
COPERNICUS by the European Union and ESA; all rights reserved.*

## 1. Method

Three predicates, evaluated **per pixel** on the real `geometry.grid_windows` grid
(224², non-overlapping, edge-flushed — ADR-0001 §2):

```
Z  := elevation == 0.0      # ADR-0001 §3's test (config.OCEAN_FILL_VALUE)
W1 := WBM == 1              # Copernicus's "Ocean" class
WW := WBM in {1, 2, 3}      # Ocean / Lake / River — "water"
```

The DEM is read at **full resolution**. ×8 overviews are averaged, which destroys
exact-zero, so overview reads are invalid for this question (masks.py §2 notes the
mirror-image point for masks: their ×8 overviews *are* mode-resampled and
categorical-safe). WBM is co-registered bit-exactly, so the same window origins align
by construction (mask spec §6) — no second indexing scheme.

**The sample is adversarial, not representative.** 14 tiles, each chosen as a specific
way the two tests could disagree. A representative sample would be dominated by open
ocean and deep interior, where the two trivially agree; that measures nothing.

| Tile | Chosen because |
|---|---|
| `N52_00_E004`, `N51_00_E004` | Netherlands / Scheldt — sub-sea-level land (−12 m). ADR-0001 §3's *named* risk |
| `N23_00_E090` | Bangladesh delta — vast near-zero floodplain, tidal channels |
| `S21_00_W068` | Salar de Uyuni — the salt flat §3 said to revisit for ("digitized to exactly 0") |
| `N31_00_E031`, `N26_00_W081` | Nile delta, Everglades — near-zero land against open sea |
| `N43_00_W087`, `N41_00_E051`, `N31_00_E035` | Lake Michigan (+176 m), Caspian (−28 m), Dead Sea (−430 m) — water *not* at zero |
| `N03_00_W060` | Amazon — wide rivers (WBM 3) |
| `N62_00_E006`, `N70_00_E020`, `N02_00_W158` | Norway fjords, Arctic coast, Kiribati atoll — coastline shapes |
| `S85_00_E000` | Antarctic interior — control, no ocean at all |

**2,880 windows / 144,506,880 pixels.**

## 2. The per-pixel confusion

| Class | Pixels | Fraction |
|---|---:|---:|
| `Z & W1` — both say ocean | 31,320,452 | 0.216740 |
| `Z & !W1` — exact-zero **only** | 2,129,646 | 0.014737 |
| `W1 & !Z` — WBM **only** | **0** | **0.000000** |
| neither — land | 111,056,782 | 0.768522 |
| **Jaccard(Z, W1)** | | **0.936334** |

> **`WBM == 1` is a strict subset of `elevation == 0.0`.** Across 144.5 M pixels spanning
> polders, deltas, fjords, atolls and ice sheet there is **not one pixel** Copernicus
> labels Ocean that is not already exactly `0.0`. The ocean fill and the ocean mask are
> painted from the same act.

This answers the ticket's first two bullets directly: `WBM == 1` catches **nothing** the
zero-test misses. It can only ever drop *less*.

### What the disagreement actually is

All 2.13 M "zero-only" pixels are inland water at sea level:

| WBM class of the zero-only pixels | Pixels | Share of disagreement |
|---|---:|---:|
| 3 — River | 1,687,580 | 0.7924 |
| 2 — Lake | 442,055 | 0.2076 |
| 0 — No water | **11** | 0.0000 |

Copernicus flattens river and lake surfaces to a constant; near the coast that constant
*is* `0.0`. Concentrated in Bangladesh (916,256 px) and the Scheldt (713,327 px).

### ADR-0001 §3's salt-flat fear is empirically dead

§3 wrote its own revisit trigger: *"revisit only if the exact-zero trick proves leaky
(e.g. large salt-flats digitized to exactly 0)."*

**It is not leaky.** Exactly-zero pixels that WBM calls not-water: **11 of 144,506,880
(7.6 × 10⁻⁸)** — and they are not salt flats but 5 px in the Netherlands, 3 in the
Scheldt, 3 in Bangladesh. **Salar de Uyuni's `exact_zero_frac` is 0.0000 across all 256
windows.** Sub-sea-level land is negative, not zero, exactly as §3 assumed.

## 3. Per-tile

| Tile | Windows | zero frac | `W1` frac | water frac | disagree | zero-only px |
|---|---:|---:|---:|---:|---:|---:|
| N02_00_W158 | 256 | 0.99731 | 0.99731 | 0.99731 | 0.00000 | 0 |
| N03_00_W060 | 256 | 0.00000 | 0.00000 | 0.00553 | 0.00000 | 0 |
| N23_00_E090 | 256 | 0.07133 | 0.00000 | 0.11108 | 0.07133 | 916,256 |
| N26_00_W081 | 256 | 0.06980 | 0.06380 | 0.20895 | 0.00600 | 77,096 |
| N31_00_E031 | 256 | 0.52029 | 0.48754 | 0.53318 | 0.03274 | 420,569 |
| N31_00_E035 | 256 | 0.00000 | 0.00000 | 0.08308 | 0.00000 | 0 |
| N41_00_E051 | 256 | 0.00000 | 0.00000 | 1.00000 | 0.00000 | 0 |
| N43_00_W087 | 256 | 0.00000 | 0.00000 | 0.59821 | 0.00000 | 0 |
| N51_00_E004 | 160 | 0.09378 | 0.00493 | 0.11301 | 0.08885 | 713,327 |
| N52_00_E004 | 160 | 0.58899 | 0.58870 | 0.60630 | 0.00029 | 2,357 |
| N62_00_E006 | 128 | 0.46589 | 0.46588 | 0.47484 | 0.00001 | 41 |
| N70_00_E020 | 80 | 0.91430 | 0.91430 | 0.91438 | 0.00000 | 0 |
| S21_00_W068 | 256 | 0.00000 | 0.00000 | 0.51657 | 0.00000 | 0 |
| S85_00_E000 | 48 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 |

`W1 & !Z` is 0 in every tile — the subset relation holds without exception, not on average.

## 4. Window-level gates at 0.75

| Comparison | drops | only-A | only-B | agree |
|---|---:|---:|---:|---:|
| `exact_zero ≥ 0.75` | 590 (0.2049) | — | — | — |
| vs `wbm1 ≥ 0.75` | 583 (0.2024) | **7** | **0** | 2873 / 2880 |
| vs `wbm_water ≥ 0.75` | 1146 (0.3979) | 0 | 556 | 2324 / 2880 |

The **7** windows the zero-test drops and `W1` would keep are flat deltaic water —
Nile ×4, Everglades ×2, Scheldt ×1 — e.g. `ez=0.9790, wbm1=0.0000, land_std=1.15`. They
are correctly dropped; two would fall to the degenerate gate anyway.

The **556** extra drops under `wbm_water` are a *different concept* (inland water) and
belong to the separate water gate, not the ocean gate — see §6.

## 5. The coastline stratum

Plan §132 wants coastline windows tagged as their own reportable stratum so shoreline
structure is oversampled rather than silently mixed in. Composition of the candidate band
(`exact_zero_frac ∈ [0.05, 0.75)` — band *values* are [#36](https://github.com/JStoschek/DEMAE/issues/36)'s to lock):

| Subset | Windows | Where | median `land_std` | frac < 3 m |
|---|---:|---|---:|---:|
| **Sea-touching** (`wbm1_frac > 0`) | **104** | Norway 55, Netherlands 14, Everglades 11, Nile 10, Arctic 8, Scheldt 4, Kiribati 2 | **69.99 m** | 0.231 |
| **No ocean pixel** (`wbm1_frac == 0`) | **141** | Bangladesh 66, Scheldt 54, Nile 20, Everglades 1 | **2.23 m** | **0.787** |

All 141 have `wbm_water > 0`; **zero** windows are zero-elevation with no water class.

These are not the same terrain — 70 m of relief versus 2.2 m. Defining the coastline
stratum on `exact_zero_frac` alone makes **58% of it (141/245) flat delta river surface**,
which inverts the stratum's purpose and drags its metric distribution toward flat before
Phase D ever allocates over it.

Elevation alone provably cannot separate the two cases (both are exactly `0.0`).
`wbm1_frac` is the only signal that can — and it is stored raw regardless, so the
discriminator costs no extra read.

**Fate of the 141 de-coastlined windows** under the default gate stack: 111 dropped by
`land_std < 3 m`, 55 by `wbm_water > 0.30`, **118 by at least one**. Only **23** survive
into the general pool (Scheldt 15, Bangladesh 7, Nile 1), `land_std` 3.03–7.17 m, median
`wbm_water` 0.106 — so they carry the `water_affected` flag. No orphaned stratum.

## 6. Incidental finding — WBM calls Salar de Uyuni water (feeds #36)

Not this ticket's question, but measured here and material to the water gate:

| Tile | max `exact_zero_frac` | windows `wbm_water > 0.30` |
|---|---:|---:|
| S21_00_W068 (Salar de Uyuni, ~3,656 m) | 0.0000 | **141 / 256** |
| N41_00_E051 (Caspian) | 0.0000 | 256 / 256 |
| N43_00_W087 (Lake Michigan) | 0.0000 | 154 / 256 |
| N31_00_E035 (Dead Sea) | 0.0000 | 28 / 256 |

WBM classifies the **Salar de Uyuni salt flat as water** although it is genuine terrain at
3,656 m with zero exact-zero pixels. The default `wbm_water > 0.30` cap would drop 141 of
its 256 windows. Salt flats / playas are a real terrain family; #36 must check its
attrition analysis does not hollow them out. The ocean gate is unaffected — `exact_zero_frac`
is 0.0000 there, so §3 never touches it.

## 7. Resolution

**Keep ADR-0001 §3 unchanged as a pure elevation-value test; do not switch to WBM.**
Recorded as [ADR-0001 Amendment 2](../adr/0001-corpus-construction-recipe.md#amendment-2-2026-07-20--c2-confirms-the-exact-zero-test-and-narrows-the-coastline-stratum).

As re-runnable filters over columns C3 stores raw:

```python
drop_zero_fill = exact_zero_frac >= 0.75          # the gate, unchanged
coastline      = (0.05 <= exact_zero_frac < 0.75) & (wbm1_frac > 0)   # band values are #36's
```

**Coverage consequence: none.** 590 windows dropped before, 590 after — the rule did not
move. The only change is coastline stratum composition, 245 → 104 windows (−57.6%), which
is the intent rather than a cost.

The predicate is renamed `stratify.is_ocean` → `is_zero_fill` and the drop reason
`"ocean"` → `"zero_fill"`: a *pixel* value of `0.0` genuinely is the ocean fill (§2 proves
it), but a *window* that is 75% zeros is not necessarily ocean.
