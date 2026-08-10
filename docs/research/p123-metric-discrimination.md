# P123 — which §5 metrics actually discriminate, and on which mask

Ticket [#123](https://github.com/JStoschek/DEMAE/issues/123) · map [#117](https://github.com/JStoschek/DEMAE/issues/117) · plan [`task1-void-fill-plan.md`](task1-void-fill-plan.md) §0, §5, §6, §9

> **What this document is.** A measurement, taken to settle a pre-registration.
> §5 listed nine metrics per cell and graded none of them. p119 had measured four
> (MAE / RMSE / p95 / slope); the ticket's own rule — *a metric that cannot separate
> A1 from truth cannot separate the arms either* — had never been applied to NMAD,
> aspect, profile curvature, hillshade SSIM or LPIPS. This scores all of them on
> **24,000 instances** (800 tiles × 2 families × 3 buckets × 5 A1 variants), CPU-only,
> 11.5 min, reusing p119's tile draw, seeded masks and fills so every instance here is
> the same instance there. Artifacts: `prototypes/out/p123_instances.parquet`,
> harness `prototypes/p123_metric_discrimination_proto.py` (gitignored, as p119's was).

## 1. Bottom line

**Three of the seven decisions this ticket came in with were wrong, and the measurement
overturned them.** The metric set that survives is five headline columns and four
appendix ones, down from nine ungraded numbers.

1. **The erosion rule is backwards.** Scoring derivatives on a void eroded by 1 px —
   proposed so that no kernel reads ground truth — **halves** slope's ability to
   separate arms (separation 1.89 → 0.91). The boundary ring is not where a lazy fill
   hides; it is where a *discontinuous* fill is caught.
2. **Horn buys nothing and costs the strata alignment.** Zevenbergen–Thorne's first
   derivatives are central differences, making ZT slope numerically identical to the
   corpus's own `slope_deg` (max divergence **0.0006°** over 12,000 instances). Horn
   differs ~7% in level and 0.98 in ranking.
3. **RMSE + p95 was the one redundant pair.** They correlate **0.989** within-cell, and
   p95 is the weakest of the three elevation metrics. NMAD is the strongest separator of
   all ten candidates and is *less* redundant with RMSE (0.921).

Two further results fell out that no one had asked for: **aspect is the only orthogonal
signal in the set** and earns a headline slot, and **hillshade SSIM is simultaneously the
weakest and among the most redundant** candidates, so it is cut.

## 2. How a metric was graded

Two numbers per candidate, both computed **inside a single (bucket, band) cell** and then
summarised across cells. Pooling is never used: terrain drives every metric at once, so a
pooled correlation makes everything look redundant and a pooled spread makes everything
look discriminating.

- **Separation** = (worst A1 variant − best A1 variant) / median within-variant IQR. The
  five A1 variants are a known-ordered ladder of fill quality, so a metric that cannot
  spread them apart relative to instance noise cannot rank arms either. Below ~1.0 the
  variant spread is inside the noise.
- **Redundancy** = within-cell Spearman against the rest of the set.

The A1 ladder is a *proxy* for the arms, and a generous one — the arms will be closer
together than `flat` and `tps` are. A metric failing here fails a weaker test than the one
it faces in the benchmark, which is what makes a failure decisive and a pass merely
permissive.

## 3. Separation and redundancy, all ten candidates

| metric | separation (median) | worst cell | ~RMSE | ~NMAD | ~slope | verdict |
|---|---|---|---|---|---|---|
| **NMAD** | **2.51** | 1.89 | +0.92 | — | +0.60 | **headline** |
| MAE | 2.32 | 1.59 | — | — | — | appendix |
| **slope (ZT, full mask)** | **1.89** | 1.44 | +0.61 | +0.60 | — | **headline** |
| **RMSE** | **1.88** | 1.22 | — | +0.92 | +0.61 | **headline** |
| **aspect (slope-weighted)** | **1.46** | 1.17 | **−0.07** | **−0.02** | **+0.26** | **headline** |
| p95 | 1.45 | **0.86** | **+0.99** | +0.90 | +0.60 | appendix |
| profile curvature | 1.07 | **0.28** | +0.21 | +0.23 | +0.72 | appendix |
| **mean per-instance \|bias\|** | 0.95 | 0.46 | +0.66 | +0.53 | +0.29 | **headline (guard)** |
| slope (ZT, eroded mask) | 0.91 | 0.49 | +0.61 | +0.60 | — | appendix (control) |
| hillshade SSIM | **0.56** | 0.42 | −0.42 | −0.48 | **−0.84** | **cut** |

LPIPS was cut without measurement, on §5's own standing hedge that its ImageNet features
are off-domain for hillshades.

**Aspect is the row to read twice.** It is near-zero correlated with every other surviving
metric while still separating variants well. Nothing else in the set measures what it
measures.

**SSIM is the other.** Weakest separator *and* −0.84 with slope: it is mostly a noisy
restatement of a metric already in the table. §9's hillshade panel figure carries the
perceptual argument instead, and is marked load-bearing so it does not get trimmed.

## 4. The boundary ring — the result that reversed the plan

Every derivative metric was computed twice: on the full void, and on the void eroded by 1 px
so that every scored pixel's 3×3 neighbourhood contains only predicted values.

The concern was a free ride: a kernel one pixel inside the void reads six true pixels, so an
arm could score well by doing nothing. Erosion removes that. It also removes the benchmark's
sensitivity:

| slope metric | separation (median) | worst cell |
|---|---|---|
| full void mask | **1.892** | 1.438 |
| eroded void mask | 0.913 | 0.491 |

**Why**, visible in the `flat` control (fill with the void-border mean — the closest thing
here to an arm that does nothing useful):

| variant | slope err, full | slope err, eroded | gap |
|---|---|---|---|
| tps | 4.213 | 4.285 | −0.072 |
| gdal_fill_si3 | 4.473 | 4.548 | −0.075 |
| linear | 4.615 | 4.619 | −0.005 |
| gdal_fill | 4.811 | 4.918 | −0.107 |
| **flat** | **8.385** | 7.057 | **+1.327** |

A bad fill leaves a cliff where it meets real terrain. The boundary ring straddles that
cliff and registers it. Eroding the ring deletes the evidence — it does not remove a free
ride, it hides a seam.

The free ride is real, and small: on the smooth interpolators the full mask flatters the
score by **0.005–0.107°**, against a cost of **17% of scored pixels** on small voids
(4% on large). The plan now includes the ring knowingly, publishes the eroded-mask number
as a contamination control, and says so in the fairness checklist rather than arguing the
effect away.

**The gap became a metric.** Its sign separates a discontinuous fill from a smooth one at
no extra cost, both masks being computed already. That is what now carries the operational
half of C2 in place of the dropped drainage clause, and it makes §4.2's cross-fade
auditable instead of assumed.

Contamination scales with the boundary-to-area ratio, as expected:

| bucket | px surviving erosion | slope inflation (tps) |
|---|---|---|
| small | 0.83 | 1.1–3.7% |
| medium | 0.92 | 0.5–1.8% |
| large | 0.96 | 0.1–1.1% |

## 5. The kernel — ZT, and why not Horn

| bucket | central difference (`slope_deg`) | Horn | Zevenbergen–Thorne |
|---|---|---|---|
| small | 3.476 | 3.140 | 3.476 |
| medium | 4.384 | 4.053 | 4.384 |
| large | 4.995 | 4.667 | 4.995 |

ZT and the corpus's central-difference `slope_deg` agree to **0.0006°** at worst over
12,000 instances — they are the same estimator, since ZT's *G* and *H* coefficients are
central differences by construction. Horn sits ~7% lower and ranks instances 0.980 the same
as central difference: it would change every printed number without changing a single
conclusion.

That matters because `slope_mean` — the same central-difference quantity — is what defines
the ruggedness bands the whole table stratifies on. Adopting Horn would have made
"benchmark slope" and "the axis the benchmark stratifies by" two different quantities, for
no gain. ZT also supplies profile curvature, so one kernel serves all three derivatives.

## 6. Bias — the metric that had to be defined correctly to exist at all

[#124](https://github.com/JStoschek/DEMAE/issues/124) flagged that §5 scored only inside the
void and carried no mean or median error, leaving a systematically-offset fill invisible.
The obvious fix fails silently. Pooled over the corpus, mean signed error is
**0.003–0.9 m** against MAE of **0.87–38.6 m** — it cancels, and would print a null column
for every arm.

Per instance it does not cancel:

| bucket | band | MAE | pooled signed bias | mean per-instance \|bias\| | share with \|bias\| > ½·MAE |
|---|---|---|---|---|---|
| small | R1 | 0.87 | +0.003 | 0.41 | 43% |
| small | R4 | 12.66 | +0.304 | 5.80 | 47% |
| large | R1 | 1.83 | +0.024 | 0.96 | 51% |
| large | R4 | 38.63 | +0.496 | 18.85 | 46% |

Roughly **half** of a typical fill's error is a DC offset, and the sign is arbitrary per
tile. So the pre-registered column is **mean per-instance |bias|**, and §5 now records why
the pooled statistic is refused — otherwise a reviewer asks for it and someone adds it back.

Its separation (0.95) is modest, and that is not a defect: it is a **guard**, not a ranker.
§5 says so explicitly so it is not deleted later for scoring poorly on a criterion it was
never meant to satisfy.

## 7. Amendments applied

- **§0, C2** — "drainage metrics" → "seam continuity", with the reason the operational
  claim moved rather than shrank.
- **§5** — rewritten: five headline columns (RMSE, NMAD, slope-ZT-full, aspect,
  mean |bias|) plus the seam gap; four appendix (MAE, p95, curvature, eroded-mask slope);
  hydrological bullet deleted, taking `pysheds`/`WhiteboxTools` with it; SSIM and LPIPS cut;
  the full-mask rule and the ZT choice stated with their measured justifications.
- **§5, stratification** — "the corpus's ruggedness strata (collapsed to 4 groups)" was not
  a defined operation: those strata are k-means clusters over a six-metric descriptor vector
  (ADR-0004) with no ruggedness ordering. Replaced with the **`slope_mean` quartiles over
  the eligible test pool** that p119 and p123 both measured on.
- **§6, item 2** — the full-mask scoring rule joins the fairness checklist, with the
  measured size of the free ride it grants.
- **§9** — the hillshade panel figure marked load-bearing, since it is now the only
  perceptual evidence in the deliverable.

## 8. Reproducing

```
PYTHONPATH=. uv run --group dev --with pandas python prototypes/p123_metric_discrimination_proto.py selftest
PYTHONPATH=. uv run --group dev --with pandas python prototypes/p123_metric_discrimination_proto.py run --per-band 200
PYTHONPATH=. uv run --group dev --with pandas python prototypes/p123_metric_discrimination_proto.py report
```

`selftest` gates eight seams before any number is trusted: Horn and ZT slope exact on a
plane; ZT curvature zero on a plane and analytically correct on a paraboloid (−0.02 for a
0.01 coefficient); a perfect fill scoring zero on every metric and SSIM exactly 1; a
constant offset registering as pure bias with no slope, aspect or NMAD error; erosion
removing exactly the boundary ring; injected noise scoring worse on the eroded mask than the
full one; flat truth leaving aspect undefined rather than zero; and p119's masks reproducing
bit-identically, which is what makes the two parquets joinable.
