# Flatness-gate structural-null pre-registration (E63)

**Ticket:** [#63](https://github.com/JStoschek/DEMAE/issues/63) · **Map:** [#62](https://github.com/JStoschek/DEMAE/issues/62) · **Analog:** [#47](https://github.com/JStoschek/DEMAE/issues/47) (Gate E's own pre-registered bar)

**Committed artifact:** [`e63-flatness-null-bar.json`](e63-flatness-null-bar.json) — the machine-readable spec [#65](https://github.com/JStoschek/DEMAE/issues/65) imports and runs. This `.md` is the human companion; the JSON is the source of truth.

This file is pinned **before any number is computed**, the way [`e2-gate-bar.json`](e2-gate-bar.json) was pinned before the E2 report existed. The run-model ticket ([#65](https://github.com/JStoschek/DEMAE/issues/65)) is `blocked_by` this one so no one peeks first.

## The question, and the trap

Gate E's E2.5 table fired the `degenerate` (flatness, `land_std < 3 m`) gate on the four flat strata — within-stratum drop rates 30.1 / 51.9 / 31.7 / 29.7 % against a 9.8 % corpus rate (3.04–5.32×). But **a flatness gate biting flat strata is partly structural**: `land_std` correlates with `slope_mean` and `local_relief`, two of the six frozen metrics the strata are cut in. This file builds the expected-ratio model that separates *excess bias* from *expected structure* — the model the [#54](https://github.com/JStoschek/DEMAE/issues/54) verdict barred from being built post-hoc.

**The capacity trap is the whole difficulty.** Any single expected-ratio model pre-loads a verdict:

- A **saturated** model of `P(fire | six metrics)` predicts each stratum's own average rate (within a k-means cell the six metrics are ~constant) → expected ≈ observed → excess ≈ 0 → **always ACCEPT** (vacuous).
- A **too-rigid** (linear-in-logit) model under-predicts the joint-low slope&relief corner where the flat strata live → **manufactures fake excess** → biases toward the expensive LOOSEN.

So the fire rule does **not** pick one model. It requires the excess to **survive a whole capacity ladder**, and reports two fixed anchors either side so a reader sees exactly where the analysis stood.

## The model

- **Fit on** the 3,634,257 assignable catalog rows (six non-NaN metrics → z-score → stratum via `centroids.json`), kept *and* dropped, over would-be membership — as E2.5 built its table. The ~6 % of `degenerate` drops that are unassignable are out, and that limit is stated (as E2.5 stated its 44 %).
- **Features:** the six frozen metrics only, transformed exactly as k-means saw them (log on slope/relief, then z-score). **`land_std` is excluded** (the gate's own key — circular) and **`stratum_id` is excluded** (stratum-blindness *is* the test).
- **Capacity ladder** (all `sklearn`-native, every hyperparameter pinned):

  | rung | model | credits |
  |---|---|---|
  | L0 | main-effects logistic | first-order trend only |
  | L1 | + degree-2 interactions | the slope×relief corner |
  | L2 | per-metric cubic splines | curvature |
  | L3 | hist gradient boosting (`max_depth=3`, 5-fold cross-fitted) | structure-generous end |

  **Anchors** (reported, never gated): `A_uniform` = 9.76 % for all strata (the original E2.5 null, fires all four); `A_saturated` = each stratum's own rate (residual 0). Expected rate = mean predicted `P(fire)` over the stratum's would-be members.

## The fire rule (two-tier — stratum 5 decides)

**Stratum 5** fires a `LOOSEN` **rate-signal** iff, at **every** rung L0–L3, all three hold:

1. observed drop count > 99th-pct Poisson-binomial null *(trivially met at N≈250k — non-binding here, pinned for form and for Phase F at smaller tiers)*;
2. **observed − expected ≥ 10 pp** (absolute residual floor — E2.5's hollowing floor, now on the residual);
3. **observed / expected ≥ 1.5×** (relative residual floor — the E2.5 "3×" analog, softened because the denominator already credits structure).

The binding rung is expected to be **L3**; requiring all rungs yields the trajectory — residual decaying to ~0 by L2 reads `ACCEPT`, still-large at L3 reads a robust `LOOSEN`. **Strata 2/7/17** are computed identically and reported as **lower-confidence corroboration only** — never independently triggering. Both floors plus "survives every rung" make `LOOSEN` deliberately hard, matching the conservative posture (ACCEPT is the legitimate default; the sibling was accepted in [ADR-0006](../adr/0006-accept-edm-flm-hollowing-stratum19.md)).

## The knob (shape only; value is fog)

The §215–228 table has no row for the `land_std` screen. Shape: a global `degenerate_land_std_m` (default 3.0) **plus** an optional `degenerate_land_std_m_by_stratum` override map — a superset so the execution map is pre-committed to neither a global re-tune nor a pure exemption. The value, and the choice within the shape, are [#66](https://github.com/JStoschek/DEMAE/issues/66)'s call.

## Scope

This file produces a **rate-excess signal, not the verdict.** The accept/loosen decision ([#66](https://github.com/JStoschek/DEMAE/issues/66)) weighs it against [#64](https://github.com/JStoschek/DEMAE/issues/64)'s junk-vs-informative and starvation findings.
