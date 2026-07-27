# The 500,000-window scale-tier draw

**Ticket:** [Draw the 500k scale-tier selection list (#71)](https://github.com/JStoschek/DEMAE/issues/71) · **Map:** [Phase F (#67)](https://github.com/JStoschek/DEMAE/issues/67) · **List:** [`selection_scale_500k.csv`](../../artifacts/phase_f/selection_scale_500k.csv) · **Quota table:** [`f-allocation-nh-500k.json`](f-allocation-nh-500k.json) · **Numbers:** [`f-scale-draw.json`](f-scale-draw.json)

The same sampler, the same committed artifacts, a bigger budget. [#44](https://github.com/JStoschek/DEMAE/issues/44) built the draw so this ticket would be a config change and demonstrated it; this is that design being spent:

```bash
python -m dataset.select --budget 500000 --out artifacts/phase_f/selection_scale_500k.csv
```

## Determinism — demonstrated, not asserted

Two full runs from the committed artifacts, 500,000 rows each:

| run | md5 |
|---|---|
| 1 (committed) | `90d74c896d1fa04f7079b8491781c21e` |
| 2 (replicate) | `90d74c896d1fa04f7079b8491781c21e` |

The two files are **byte-identical**.

Wall time **~32 min** (run 2, measured end to end) across 6 worker processes, each sub-k-means fit pinned single-threaded ([#43](https://github.com/JStoschek/DEMAE/issues/43)'s lever) — inside [ADR-0005 §9](../adr/0005-sampling-design.md)'s projected ~55 min, and the reason the per-stratum fan-out exists.

Nothing about the mechanism changed at 10× the budget, because nothing about it *could*: every tie-break is a keyed BLAKE2b of the window id rather than a NumPy `Generator`, the fit is pinned single-threaded, and the persisted list — not the seed — is the contract.

## The list conforms to its own quota table

- **500,000 rows**, all distinct window ids (500,000).
- Per-stratum counts equal `n_h` for all 20 strata; coastal picks equal `c_h`.
- Provenance is exactly the two-value vocabulary [#42](https://github.com/JStoschek/DEMAE/issues/42) fixed: `base` 493,300, `floor:coastline` 6,700.

That is arithmetic, not a gate. Gate F's pass/fail bar is pre-registered in [`f-gate-bar.json`](f-gate-bar.json) ([#69](https://github.com/JStoschek/DEMAE/issues/69)) and executed by [#73](https://github.com/JStoschek/DEMAE/issues/73); this report supplies numbers to it and decides nothing.

### …but not to the `n_h` table the Gate F bar pre-registered

They differ on **12 of 20 strata**, by at most **±2** windows. Both tables sum to exactly 500,000.

| h | pre-registered `n_h` | drawn `n_h` | Δ |
|---|---|---|---|
| 1 | 28,072 | 28,071 | -1 |
| 3 | 21,851 | 21,852 | +1 |
| 4 | 36,612 | 36,614 | +2 |
| 8 | 37,015 | 37,013 | -2 |
| 9 | 27,239 | 27,238 | -1 |
| 10 | 28,175 | 28,173 | -2 |
| 11 | 10,147 | 10,148 | +1 |
| 12 | 32,031 | 32,032 | +1 |
| 14 | 24,673 | 24,672 | -1 |
| 15 | 20,974 | 20,975 | +1 |
| 16 | 30,874 | 30,873 | -1 |
| 18 | 30,921 | 30,923 | +2 |

**The cause is reproduced, not inferred.** The pre-registered table is reproduced bit-for-bit by feeding the 4-dp sigma_h that d2-allocation-nh.json PRINTS back into select.allocate, while the sampler recomputes sigma_h at full precision from the frame. #44 recorded this exact trap ('the JSON's 4-dp sigma_h is for legibility; feeding it back shifts a few n_h by +-1, which is why the sampler recomputes sigma_h from the frame') and asserts the full-precision path against the committed 50k table in tests/test_select.py. Both tables sum to the exact budget; the drawn one is the allocation rule ADR-0005 specifies, the pre-registered one is that rule evaluated on a display-rounded input.

**F2a compares the shipped list's per-stratum counts against this table, so as committed it would fire a DEFECT on a correct draw. The bar is #69's artifact and #71 does not edit a pre-registered file after drawing against it: the discrepancy is published here, with its cause reproduced, for a separate dated commit to correct before #73 executes the gate.**

## Realized windows-per-COG — the metric #42 bound this tier to decide on

| | @50k (#44) | @500k |
|---|---|---|
| mean | 3.16 | **21.83** |
| median | 2 | **14** |
| p99 | 16 | **118** |
| max | 44 | **233** |
| distinct COGs drawn | 15,834 | **22,909** |

[#42](https://github.com/JStoschek/DEMAE/issues/42) retired `PER_TILE_WINDOW_CAP = 64` as inert at the debug tier but required the sampler to *measure* the distribution at every tier, so the scale tier could decide on evidence whether a cap needs reintroducing. **The count alone cannot answer that**, which is what [#69](https://github.com/JStoschek/DEMAE/issues/69) established when it retired the cap as unfixable rather than merely stale: the ceiling on a COG is its own kept inventory (at most 256 windows), so the readable statistic is the fraction of that inventory drawn.

| fraction of a COG's own kept inventory | value |
|---|---|
| mean | 0.2291 |
| median | 0.1523 |
| p99 | 1.0 |
| max | 1.0 |
| max among COGs holding ≥32 kept windows | 1.0 |
| COGs drawn at ≥50% of inventory | 2,789 |
| COGs drawn in full | 423 (2 of them holding ≥32; largest 48 windows) |
| COGs in the kept pool, never drawn | 1,209 of 24,118 |

**The regime the retired cap was pointed at is now measured rather than predicted, and it confirms #69's reasoning.** The old cap of 64 would bind on **1,320** of the 22,909 COGs drawn and strip **40,211 windows** — 8.0% of the budget — which the sampler would then have to buy from worse representatives elsewhere. The densest COG yields **233**, which is 91% of everything that COG holds. No larger threshold rescues the guard, because the ceiling is a COG's own inventory rather than a number anyone picks: **2,789** COGs are drawn at ≥50% of what they hold and **423** in full — though only **2** of those hold ≥32 windows (largest 48), so exhaustion is overwhelmingly a small-COG effect, not the dense tail running dry.

The checks that survive the cap's retirement are mechanical — [`F6b`](f-gate-bar.json) (no COG yields more than it holds) and `F6c` (this distribution, published). The real residue is **spatial redundancy**: at a 23% mean draw rate, windows from one COG are neighbours, so the corpus's effective sample size sits below its row count. #69 handed that to [#75](https://github.com/JStoschek/DEMAE/issues/75)'s honesty notes as a disclosed property rather than gating it against an invented threshold — these are the numbers it should disclose.

## Per-stratum allocation and headroom

`headroom` is `N_h/n_h` — kept windows held per window the stratum must yield. **This is the number the budget actually moves.** Neyman shares are budget-invariant, so a 10× budget is a 10× shallower pool. [ADR-0007](../adr/0007-accept-degenerate-flatness-gate-flat-strata.md) leg 3 leaned on the four flat strata (2/5/7/17) keeping **≥53×** their allocation at 50k; measured here, those four sit at **≥5.35×** — the ≈5.3× the ADR predicted, confirmed rather than assumed.

| h | N_h | σ_h | coast_h | n_h | coastal | base | headroom |
|---|---|---|---|---|---|---|---|
| 0 | 186,950 | 1.0787 | 1,662 | 32,200 | 273 | 31,927 | 5.81× |
| 1 | 168,835 | 1.0413 | 3,093 | 28,071 | 508 | 27,563 | 6.01× |
| 2 | 150,898 | 0.9612 | 1,036 | 23,159 | 170 | 22,989 | 6.52× |
| 3 | 130,184 | 1.0512 | 841 | 21,852 | 138 | 21,714 | 5.96× |
| 4 | 232,598 | 0.9858 | 1,426 | 36,614 | 234 | 36,380 | 6.35× |
| 5 | 118,638 | 1.0510 | 449 | 19,909 | 74 | 19,835 | 5.96× |
| 6 | 32,229 | 1.4269 | 0 | 7,343 | 0 | 7,343 | 4.39× |
| 7 | 139,429 | 1.1698 | 0 | 26,043 | 0 | 26,043 | 5.35× |
| 8 | 292,363 | 0.7929 | 2,344 | 37,013 | 385 | 36,628 | 7.90× |
| 9 | 166,251 | 1.0261 | 2,306 | 27,238 | 379 | 26,859 | 6.10× |
| 10 | 148,603 | 1.1874 | 1,340 | 28,173 | 220 | 27,953 | 5.27× |
| 11 | 50,767 | 1.2518 | 0 | 10,148 | 0 | 10,148 | 5.00× |
| 12 | 185,487 | 1.0815 | 5,935 | 32,032 | 974 | 31,058 | 5.79× |
| 13 | 194,931 | 1.0678 | 3,573 | 33,236 | 586 | 32,650 | 5.87× |
| 14 | 143,914 | 1.0737 | 4,915 | 24,672 | 807 | 23,865 | 5.83× |
| 15 | 115,886 | 1.1335 | 4,064 | 20,975 | 667 | 20,308 | 5.52× |
| 16 | 220,981 | 0.8750 | 3,816 | 30,873 | 626 | 30,247 | 7.16× |
| 17 | 105,302 | 1.0986 | 1,095 | 18,472 | 180 | 18,292 | 5.70× |
| 18 | 215,052 | 0.9005 | 2,920 | 30,923 | 479 | 30,444 | 6.95× |
| 19 | 53,437 | 1.2955 | 0 | 11,054 | 0 | 11,054 | 4.83× |

Range **4.39×–7.90×**, thinnest at stratum **6**.

ADR-0007's Phase-F revisit trigger names strata **2/17** (with **5/7** as corroboration). Their headroom at this budget:

| h | N_h | n_h | headroom | role |
|---|---|---|---|---|
| 2 | 150,898 | 23,159 | 6.52× | gating |
| 17 | 105,302 | 18,472 | 5.70× | gating |
| 5 | 118,638 | 19,909 | 5.96× | corroborating |
| 7 | 139,429 | 26,043 | 5.35× | corroborating |

*headroom only — the 2/17 drainage re-test itself is Gate F's F3, pre-registered in docs/research/f-gate-bar.json (#69) and executed by #73.*

## What binds at this budget

| guard | setting | fires |
|---|---|---|
| `max(500, 1%·budget)` clamp | 5,000 | **0 of 20 strata** (smallest `n_h` 7,343) |
| coastal `min(n_h, coast_h)` cap | per stratum | **0 of 20 strata** |

The coastal reservation is `round(0.0134 · budget)` = **6,700** windows, a realized share of **1.3400%** against the kept frame's own coastal rate of **1.3370%**. It is a *frame-rate* floor, not a boost ([#42](https://github.com/JStoschek/DEMAE/issues/42) §4): coast is a location tag rather than a descriptor shape, so the honest minimum is that coast is never under-represented relative to reality.

Both guards were non-binding at 50k, and Neyman's budget-invariance made it likely they still were — but *likely* is not *measured*, which is why #71 asked. **Measured: neither binds.**

One precision about the coastal floor, so the row above is not read as more than it says. The reservation is a *within-stratum split*, not a post-hoc top-up ([#42](https://github.com/JStoschek/DEMAE/issues/42) §4): each stratum runs FSCS over its coastal rows for `c_h` and over the rest for `base_h`, so a `base` pick is never coastal by construction and the reservation **sets** the corpus's coastal share exactly rather than raising a floor under a share the draw would otherwise produce. What is measured here is that the split is feasible everywhere (the `min(n_h, coast_h)` cap fires on no stratum) and that the share it sets sits at the frame's own rate. What is *not* measured — and is named rather than estimated — is what an unreserved single-pass FSCS would have drawn: that is a second full 30-minute draw, and #71 did not ask for the counterfactual.

## The re-clustering knob is structurally inert

> §F: same strata/centroids for comparability; re-clustering is a knob

**Status: structurally inert — kept declared, never exercised.**

The committed centroids were fit on **3,052,735** rows; the kept frame a Phase-F refit would see today is **3,052,735** — identical.

The centroids were fit on the ENTIRE gated catalog (fit_rows 3,052,735 = the whole kept frame), not on the debug tier's 50,000 windows. A Phase-F refit would therefore run on identical input: same rows, same k=20, same Lloyd, same n_init. It cannot discover strata the committed fit did not already see, so its expected change is nil — while #41 established the FIT is not bitwise-reproducible across threads (assign() is). Refitting strictly loses: no expected gain, and a guaranteed loss of the cross-tier strata comparability the persisted artifacts exist to provide.

The knob stays declared in ADR-0005 §9 (k row) and in the plan's §F, and is never exercised — the #36/#42 pattern: retire on evidence, keep declared, so a future frame that genuinely moves the fit population makes it live again with no code change.

## Overlap with the debug tier — incidental, reported, unbinding

[#45](https://github.com/JStoschek/DEMAE/issues/45) ruled the scale draw **independent**: the 50k does not nest inside it and no hierarchical rule was built. The two lists share **30,998 windows** — **62.0%** of the debug tier and **6.2%** of the scale tier.

Two *uniform* draws of these sizes from the 3,052,735-row kept pool would average **8,189** shared windows, so the realized overlap is **3.8×** that. FSCS is deterministic and coverage-seeking: both tiers keep resolving to the same nearest-the-sub-centroid representatives, which is the sampler agreeing with itself rather than a nesting rule leaking in. It carries no obligation either way.

## Provenance

Reads committed artifacts only (`data/catalog_gated`, `artifacts/phase_d/{centroids,scaler}.json`, the selection list); draws nothing and touches no pipeline state. Regenerate with `python notebooks/f71_scale_draw.py` — every number above comes back except the replicate md5 and wall time, which are properties of a *run* rather than of the artifact and are passed in via `--replicate` / `--run-seconds` (the second run's CSV is not committed; the md5 recorded here is the evidence it existed and matched).
