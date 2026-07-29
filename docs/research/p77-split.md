# P77 — the geographic split, drawn and committed

Ticket [#77](https://github.com/JStoschek/DEMAE/issues/77) · map [#76](https://github.com/JStoschek/DEMAE/issues/76) · plan §9 · corpus `bcbd6484d8116199c5c1160e21ea0284` (499,934 tiles)

Assignment `a27ae38da558c33073646585609eff83` · [`artifacts/phase1/split_scale_500k.csv`](../../artifacts/phase1/split_scale_500k.csv) · [`artifacts/phase1/split_blocks_500k.json`](../../artifacts/phase1/split_blocks_500k.json)

## 1. What was drawn

| split | blocking unit | blocks | tiles | share | shortfall vs corpus | TV |
|---|---|---|---|---|---|---|
| `train` | — (the remainder) | — | 475,030 | 95.018% | 0.0296 | 0.0012 |
| `val` | whole source COG | 518 | 9,998 | 2.000% | 1.0029 | 0.0427 |
| `test` | whole 10° geocell | 13 | 14,906 | 2.982% | 0.5739 | 0.0240 |

The plan asks for ~95 / ~2 / ~3 and the draw lands on 95.018% / 2.000% / 2.982%. Both hold-outs carry all 20 strata. *Shortfall* is `dataset.splits.relative_shortfall` — the total relative stratum deficit against the corpus composition, the objective both draws minimize; *TV* is total-variation distance, reported because it is the familiar number and **not** used as an objective (it prices a missing 1.5% stratum at 0.015 and cannot see coverage).

The test search found its 13 cells among 328 eligible, improving the greedy seed from 1.1040 to 0.5739 in 4 swaps. Val drew 518 COGs of 22,203 eligible at mean depth 19.3, against 21.9 for the eligible pool. Val's shortfall is the higher of the two because its draw does not minimize it — §6 is where that trade is made and priced.

## 2. The 13 test regions

| geocell | region | tiles | COGs | lon | lat | strata |
|---|---|---|---|---|---|---|
| `+090_+040` | Mongolian Altai, Dzungaria and the Gobi | 2,974 | 100 | 95.2 | 44.8 | 20/20 |
| `+030_+000` | Ethiopian highlands and the East African Rift | 2,836 | 90 | 35.7 | 5.3 | 20/20 |
| `-100_+010` | southern Mexico — Sierra Madre del Sur and Tehuantepec | 2,133 | 47 | -95.2 | 17.8 | 20/20 |
| `+020_+060` | Fennoscandia — Finnish lakeland and Karelia | 1,523 | 95 | 24.9 | 65.9 | 14/20 |
| `-100_+060` | Canadian Shield — Kivalliq and western Hudson Bay | 1,052 | 83 | -94.8 | 64.8 | 14/20 |
| `-090_+050` | Canadian Shield — northern Ontario and the Hudson Bay lowlands | 873 | 54 | -86.9 | 52.0 | 13/20 |
| `+150_-070` | Antarctica — George V Land and the Wilkes coast | 743 | 16 | 154.3 | -69.4 | 15/20 |
| `-170_-090` | Antarctica — Ross Ice Shelf and the Transantarctic margin | 607 | 63 | -164.8 | -85.6 | 18/20 |
| `+020_+070` | Svalbard and the Barents shelf islands | 516 | 26 | 24.6 | 75.5 | 15/20 |
| `-170_+050` | western Alaska — Yukon-Kuskokwim delta and the Bering coast | 451 | 20 | -162.2 | 57.2 | 15/20 |
| `+150_-010` | Bismarck Archipelago and the Solomon Sea islands | 419 | 27 | 153.0 | -5.9 | 9/20 |
| `+050_-090` | East Antarctic interior | 395 | 77 | 55.1 | -83.8 | 2/20 |
| `+000_-010` | Gabon and the western Congo basin | 384 | 5 | 9.6 | -1.1 | 8/20 |

Pairwise separation runs 1,069–18,878 km (median 10,665 km) against the 1,000 km floor. These are Phase 2's downstream evaluation regions and no pretraining run may read them.

**What the pool the search saw is not.** §9 says "select 12-15 of the 425 occupied geocells"; the search ranged over **328 of 425**. A cell must hold between 374 and 3,749 tiles — a fortieth to a quarter of the test budget — which drops 4 cells as too large and 93 as too small. That narrowing is not neutral, and this is the part of the design most worth arguing with: the largest cells are the high mountains, so the band takes the rare families with them. Inside the band lie only 56.8% of stratum 6, 77.8% of stratum 19, 93.0% of stratum 11. Stratum 6 (the 4,300 m plateaus of Tibet and the Altiplano) and stratum 19 (high alpine) are held almost entirely by eight geocells of 3,000-4,100 tiles each; admitting one would have spent a fifth to a quarter of the test budget on a single region. The consequence is visible in §4 — those are the two strata furthest from their corpus share in `test` — and it is a cost of whole-geocell blocking at a 3% budget, not a defect in the search.

| latitude belt | corpus | `test` | `val` |
|---|---|---|---|
| 60S and south | 14.3% | 11.7% | 16.5% |
| 60S-0 | 21.6% | 5.4% | 22.2% |
| 0-30N | 19.9% | 33.3% | 20.4% |
| 30-60N | 29.4% | 28.8% | 24.7% |
| 60N and north | 14.9% | 20.7% | 16.2% |

Three of the thirteen cells are Antarctic, which looks polar-heavy until it is measured against the corpus: Antarctica is 14.3% of the tiles and 11.7% of the test pool, so the hold-out is *under*-weighted there, not over. The belt table is the check.

Two honest qualifications on "spanning distinct geomorphic provinces". First, the *pool* is what carries all 20 strata, not the cells: an individual cell holds as few as 2 of 20 (the East Antarctic interior is ice sheet and little else), and that is the expected shape of a real region rather than a defect in the draw. Second, the closest pair — `-100_+060` and `-090_+050` at 1,069 km — is glaciated Canadian Shield on both sides. The 1,000 km floor kept them apart in kilometres and not in geomorphology, so the set spans twelve distinct provinces and samples one of them twice.

## 3. The boundary effect — the plan's "hundreds of km" is the median's story, not the edge's

| hold-out | min | p1 | median | p95 | max | <10 km | <25 km | <50 km |
|---|---|---|---|---|---|---|---|---|
| `test` | 3.0 | 6.9 | 97.1 | 347.9 | 583.0 | 3.0% | 12.8% | 29.5% |
| `val` | 3.0 | 4.1 | 20.7 | 48.9 | 376.5 | 21.4% | 63.0% | 95.9% |

Distances in km from each held-out tile to the nearest `train` tile.

Plan §9 says geocell blocking "puts hundreds of km between train and test, far beyond terrain autocorrelation ranges". The median is **97 km**, not hundreds, and the minimum is **3.0 km** — a rim of test tiles sits one tile-width across the cell boundary from a train tile. Blocking at 10° bounds the *typical* separation, not the smallest one; a whole-cell hold-out has a perimeter, and 12.8% of test tiles lie within 25 km of training data. The claim is corrected here rather than quietly restated: the design is unchanged, because §9 specifies `train` as everything else and this ticket does not get to amend it.

Val is weaker by construction and the plan says so — COG blocking is the *minimum* honest unit. 21.4% of val tiles have a train tile within 10 km and the median is 20.7 km. Val loss therefore measures generalization across a ~110 km source-tile boundary, not across regions; that is what the `test` cells are for.

**The price of the obvious fix, unspent.** Excluding train tiles near a hold-out would cost 2,964 (0.62%) at 10 km, 12,931 (2.72%) at 25 km, 35,259 (7.42%) at 50 km of the training pool. Cheap — but a buffer is not in §9, and it would perturb the Neyman allocation a second time. Recorded for the map to decide, not applied.

## 4. Per-stratum train counts, and the Neyman perturbation (plan §9b)

| stratum | drawn `n_h` | train | val | test | drawn share | train share | shift |
|---|---|---|---|---|---|---|---|
| 0 | 32,200 | 30,557 | 699 | 940 | 0.0644 | 0.0643 | -0.11% |
| 1 | 28,071 | 26,738 | 475 | 858 | 0.0561 | 0.0563 | +0.26% |
| 2 | 23,159 | 21,958 | 475 | 720 | 0.0463 | 0.0462 | -0.20% |
| 3 | 21,852 | 20,720 | 475 | 652 | 0.0437 | 0.0436 | -0.20% |
| 4 | 36,614 | 34,833 | 696 | 1,085 | 0.0732 | 0.0733 | +0.14% |
| 5 | 19,909 | 18,950 | 362 | 593 | 0.0398 | 0.0399 | +0.19% |
| 6 | 7,343 | 7,038 | 138 | 167 | 0.0147 | 0.0148 | +0.89% |
| 7 | 26,043 | 24,726 | 448 | 869 | 0.0521 | 0.0521 | -0.07% |
| 8 | 37,013 | 35,172 | 729 | 1,096 | 0.0740 | 0.0740 | +0.02% |
| 9 | 27,238 | 26,024 | 504 | 710 | 0.0545 | 0.0548 | +0.56% |
| 10 | 28,173 | 26,808 | 538 | 827 | 0.0563 | 0.0564 | +0.16% |
| 11 | 10,148 | 9,597 | 240 | 311 | 0.0203 | 0.0202 | -0.46% |
| 12 | 32,032 | 30,489 | 629 | 914 | 0.0641 | 0.0642 | +0.19% |
| 13 | 33,236 | 31,670 | 662 | 904 | 0.0665 | 0.0667 | +0.30% |
| 14 | 24,672 | 23,375 | 542 | 755 | 0.0493 | 0.0492 | -0.28% |
| 15 | 20,975 | 19,849 | 460 | 666 | 0.0420 | 0.0418 | -0.39% |
| 16 | 30,873 | 29,223 | 744 | 892 | 0.0617 | 0.0615 | -0.37% |
| 17 | 18,472 | 17,463 | 438 | 558 | 0.0369 | 0.0368 | -0.49% |
| 18 | 30,923 | 29,384 | 591 | 944 | 0.0618 | 0.0619 | +0.02% |
| 19 | 11,054 | 10,456 | 153 | 445 | 0.0221 | 0.0220 | -0.44% |

Removing the hold-outs moves no stratum's share of the training pool by more than **0.89%** of itself (total variation 0.00119). The Neyman allocation the sampler drew is not quite the allocation the model trains on, which is the caveat §9b asks to be recorded; the size of it is the reason no correction is proposed.

## 5. The global μ/σ contamination (plan §9c)

The shipped constants — **955.67 / 1012.16 m** — were accumulated over every tile, including the 4.98% of land pixels now held out. Because `normalize.land_moments` sums quantities that are additive, and the manifest stores each tile's land `mean`/`std`/`land_frac`, the pair can be re-derived for any subset without reading a shard. The whole-corpus re-derivation returns 955.6729 / 1012.1607 m — the shipped numbers to four decimals, which is what licenses the rest of this table.

| pool | land μ (m) | land σ (m) | land pixels |
|---|---|---|---|
| `train` | 952.75 | 1013.43 | 23,780,547,356 |
| `val` | 859.43 | 933.51 | 500,228,536 |
| `test` | 1113.33 | 1006.85 | 745,970,731 |
| all (shipped) | 955.67 | 1012.16 | — |

Recomputing on `train` only would move the mean by **-2.92 m (-0.31%)** and the sd by **+1.27 m (+0.13%)**. §9c offered two options — recompute, or record the contamination as accepted. The train-only pair now exists at zero cost and is recorded here for Phase 3's global-norm ablation to use; `corpus_meta.json` is **not** rewritten, because the shipped pair is a Phase-0 record and a 0.31% shift does not justify moving a number four committed artifacts quote. The held-out pools are not interchangeable with train — `test` runs +161 m in the mean (mountain and plateau regions) and `val` -93 m.

## 6. Why val draws in hash order

| draw | COGs | tiles | mean depth | median depth | coastline | mean `land_frac` | TV | shortfall |
|---|---|---|---|---|---|---|---|---|
| shipped — per-stratum quota, hash order | 518 | 9,998 | 19.3 | 13 | 1.56% | 0.9971 | 0.0427 | 1.0029 |
| rejected — shortfall greedy | 3,111 | 9,998 | 3.21 | 2 | 12.40% | 0.9825 | 0.0007 | 0.0129 |
| corpus | — | — | 21.85 | — | 1.34% | 0.9977 | — | — |

The greedy variant matches the stratum composition better — TV 0.0007 against 0.0427 — and buys it by taking 3,111 COGs of mean depth 3.21. A COG holding two kept tiles is a COG the gates nearly emptied, so the pool it assembles carries coastline at **9.3×** the corpus rate (12.40% against 1.34%). Composition is not the only thing a val set has to look like. The shipped draw is size-blind — quota per stratum, COGs in keyed-hash order — and lands on the corpus's coastline rate to two decimals.

**What that trade costs, stated plainly.** Val's shortfall of 1.0029 is not a missing stratum — all 20 are present, the smallest holding 138 tiles — it is a sum of small relative deficits, and the larger cause is the quota's accounting rather than the hash order. A COG is charged whole to its **dominant** stratum, but it carries tiles from others, so a stratum's realized count is its own quota plus whatever its neighbours' COGs happened to bring: stratum 19 lands at 153 against a quota of 221 and stratum 16 at 744 against 617. The fix would be to charge each COG's tiles to their own strata and accept overshoot, or to swap COGs after the pass — both re-introduce a size-aware step, which is the thing the greedy row above shows is expensive. For val loss and a kNN probe on 20 classes, +-30% on a class count is not worth buying back at that price; it is recorded so a probe result that leans on the rare classes can be read with it in hand.

## 7. Determinism, and what §9d still owes

Re-drawing from the shipped manifest reproduces the committed assignment byte for byte (md5 `a27ae38da558c33073646585609eff83`, 4.2 s). As with the selection list, the contract is the persisted file and not the seed: every tie-break is a keyed BLAKE2b hash of the block id, and nothing here re-fits, re-clusters, or reads a COG.

§9d asks for the md5 "in the run config". There is no Phase-1 run config yet — R0 has not been built — so the digest is carried by [`artifacts/phase1/split_blocks_500k.json`](../../artifacts/phase1/split_blocks_500k.json), by this report, and by the commit that ships them. The obligation is **not discharged**: plan §15.7 requires the training run to be tagged with the split md5 alongside the git SHA, the config hash and the selection-list md5, and `config.SPLIT_ASSIGNMENT_PATH` names the file it must hash. Recorded here so the pre-flight checklist has something to check against.

