# E65 — Structural-null model run: observed-vs-expected `degenerate` drop rate

**Ticket:** [#65](https://github.com/JStoschek/DEMAE/issues/65) · **Map:** [#62](https://github.com/JStoschek/DEMAE/issues/62) · **Bar (imported blind):** [`e63-flatness-null-bar.json`](e63-flatness-null-bar.json) · **Code:** [`notebooks/e65_flatness_null_model.py`](../../notebooks/e65_flatness_null_model.py) @ `6b446a0`

## Rate-signal: **ACCEPT**

The pre-registered structural-null model — the #47-analog committed *blind* in [#63](https://github.com/JStoschek/DEMAE/issues/63) — was executed against the committed gated catalog. It defines no model and no threshold of its own; every estimator and every fire-rule number is read from the bar. The two-tier rule fires a **LOOSEN** signal only if stratum 5 clears all three conditions (significant · residual ≥ 10 pp · ratio ≥ 1.5×) at **every** ladder rung. It **does not** — the signal is **ACCEPT**. Binding (smallest-residual) rung for stratum 5: **L0**.

This is a *rate-signal*, not the verdict. The accept/loosen decision ([#66](https://github.com/JStoschek/DEMAE/issues/66)) weighs it against #64's junk-vs-informative and starvation findings.

## Fit population

- **3,634,257** assignable rows (all six frozen metrics finite), kept and dropped — the would-be (pre-gate) membership, exactly as E2.5 built its table.
- **752,463** unassignable (NaN-metric) rows excluded — the stated blind spot (~6% of `degenerate`), disclosed in the bar.
- Target = the `degenerate` flag; it equals `land_std < 3.0` on **3,634,257 / 3,634,257** assignable rows, so the modelled target is exactly the gate.
- Anchors: **A_uniform** = 0.0976 (the original E2.5 null, credits zero structure); **A_saturated** = each stratum's own rate (residual 0, credits all structure).

## The ladder — stratum 5 (primary, the only stratum that can trigger LOOSEN)

Would-be members **247,681**, observed drops **128,511** (rate **0.519**). Anchor residuals: uniform **+0.421 pp / 5.32×**, saturated **0**.

| rung | expected | residual pp | ratio | excess count | ≥10 pp | ≥1.5× | sig | fires |
|---|---|---|---|---|---|---|---|---|
| L0 | 0.526 | -0.007 | 0.99× | -1,660 | · | · | · | — |
| L1 | 0.519 | -0.000 | 1.00× | -84 | · | · | · | — |
| L2 | 0.523 | -0.004 | 0.99× | -946 | · | · | · | — |
| L3 | 0.519 | -0.001 | 1.00× | -128 | · | · | · | — |

**Stratum 5 LOOSEN-signal (all rungs fire): NO.** L3 is the structure-generous, cross-fitted rung and is expected to bind.

## Corroborating strata 2 / 7 / 17 (reported only, never trigger)

| h | family | members | obs rate | exp L3 | residual L3 | ratio L3 | all-rungs fire |
|---|---|---|---|---|---|---|---|
| 2 | low-relief plains with faint drainage | 217,614 | 0.301 | 0.299 | +0.001 pp | 1.00× | NO |
| 7 | high-altitude flats / plateaus | 212,872 | 0.317 | 0.319 | -0.001 pp | 1.00× | NO |
| 17 | low-relief lowlands with drainage / water | 168,601 | 0.297 | 0.296 | +0.001 pp | 1.00× | NO |

## Significance leg

The Poisson-binomial null is evaluated by its large-N normal (CLT) approximation — exact evaluation is O(N²) at N≈250k and infeasible, and the bar itself flags this leg as *trivially met by any real excess, essentially non-binding at the debug tier*. It is pinned for form and for Phase-F re-use at smaller N. The `null_z` in the JSON shows how far past the 99th-pct each observed count sits.

## Provenance

Reads committed data only (`data/catalog_gated`, `artifacts/phase_d/{centroids,scaler}.json`); touches no pipeline state. Full per-rung numbers, anchors and null z-scores in [`e65-flatness-null-model.json`](e65-flatness-null-model.json).
