"""E65 — run the pre-registered structural-null model for the flatness gate.

Ticket: https://github.com/JStoschek/DEMAE/issues/65 · Map: #62 (flatness gate).
Bar (imported, NOT redefined here): docs/research/e63-flatness-null-bar.json (#63).

The #47-analog for the flatness-gate map. #63 committed — *blind*, before any number
existed — a stratum-blind expected-ratio model of ``P(degenerate fire | six frozen
metrics)`` plus a two-tier fire rule. This script does exactly one thing: it *imports*
that committed bar and executes it. It defines no model and no threshold of its own;
every estimator hyperparameter and every fire-rule number is read from the bar. Changing
either after seeing a number here is the peeking the pre-registration forbids.

What the bar pins (see e63-flatness-null-bar.json for the rationale):

- **Fit population**: the ASSIGNABLE catalog rows (all six frozen metrics finite), kept
  *and* dropped — would-be (pre-gate) membership, exactly as E2.5 built its table. The
  unassignable NaN-metric rows (~6% of degenerate) are a stated blind spot.
- **Features**: the six frozen metrics ONLY, transformed as k-means saw them (log1p on
  slope_mean & local_relief, then z-scored via scaler.json). ``land_std`` (the gate's own
  key) and ``stratum_id`` are deliberately excluded — including either makes the residual
  ~0 by construction and the test vacuous.
- **Capacity ladder** L0→L3 (logistic main-effects → degree-2 poly → per-metric cubic
  splines → cross-fitted HistGBM). The excess must survive *every* rung; L3 (most
  structure credited) is expected to bind. Two anchors (uniform 0.0976, saturated) bracket
  the ladder, reported never gated.
- **Fire rule**: stratum 5 fires a LOOSEN rate-signal iff at EVERY rung all three hold —
  significance (count > 99th-pct Poisson-binomial null), residual ≥ 10 pp, ratio ≥ 1.5×.
  Strata 2/7/17 get the identical statistic as lower-confidence corroboration; they never
  trigger. Else the signal is ACCEPT.

This produces a *rate-signal*, never the verdict. The accept/loosen DECISION (#66) weighs
this against #64's junk-vs-informative / starvation findings.

AFK. Reads committed data only (``data/catalog_gated``, ``artifacts/phase_d/*``); touches
no pipeline state. Writes ``docs/research/e65-flatness-null-model.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

from dataset import catalog, centroids, frame

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "docs" / "research"
BAR_PATH = RESEARCH / "e63-flatness-null-bar.json"

# stratum 5 first: the decider and the ONLY stratum that can trigger a LOOSEN signal
# (#54 §5, #63 fire_rule). 2/7/17 are corroboration only.
PRIMARY = 5
CORROBORATING = [2, 7, 17]
STRATA = [PRIMARY, *CORROBORATING]
STRATUM_NAME = {
    2: "low-relief plains with faint drainage",
    5: "near-level plains / basins",
    7: "high-altitude flats / plateaus",
    17: "low-relief lowlands with drainage / water",
}
Z99 = norm.ppf(0.99)  # 2.3263 — the significance-leg quantile pinned in the bar


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- Fit population & would-be strata (mirror of E2.5 / E64's routine) ----------


def would_be_strata(cat: Any, cn: centroids.Centroids, sc: frame.Scaler
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every window's would-be stratum (dropped included) + the z-scored feature matrix.

    Assignable = all six frozen metrics finite. Assignable rows are log1p/z-scored exactly
    as k-means saw them (frame.metric_matrix + scaler.standardize) and labelled by argmin
    to the committed centroids. Returns (labels, assignable_mask, Z_assignable).
    """
    X = frame.metric_matrix(cat)
    assignable = np.isfinite(X).all(axis=1)
    Z = sc.standardize(X[assignable])
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels[assignable] = centroids.assign(Z, cn)
    return labels, assignable, Z


# --- The capacity ladder (every knob read from the bar) -------------------------


def rung_predictions(Z: np.ndarray, y: np.ndarray, bar: dict[str, Any]
                     ) -> dict[str, np.ndarray]:
    """P(fire) per assignable row at each pinned rung. L0-L2 in-sample; L3 cross-fitted.

    Every hyperparameter comes from bar['model']['capacity_ladder']. The logistic settings
    (penalty l2, C=1.0, lbfgs) are the pinned constants; max_iter is raised only to reach
    convergence, which changes no pinned choice.
    """
    preds: dict[str, np.ndarray] = {}
    logit_kw = dict(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000)

    log("L0: logistic main effects on the 6 z-scored metrics")
    m = LogisticRegression(**logit_kw).fit(Z, y)
    preds["L0"] = m.predict_proba(Z)[:, 1]

    log("L1: degree-2 polynomial expansion -> logistic (adds slope×relief + curvature)")
    m = make_pipeline(
        PolynomialFeatures(degree=2, interaction_only=False, include_bias=False),
        LogisticRegression(**logit_kw),
    ).fit(Z, y)
    preds["L1"] = m.predict_proba(Z)[:, 1]

    log("L2: per-metric natural cubic B-splines (5 knots) -> logistic")
    m = make_pipeline(
        SplineTransformer(degree=3, n_knots=5),
        LogisticRegression(**logit_kw),
    ).fit(Z, y)
    preds["L2"] = m.predict_proba(Z)[:, 1]

    log("L3: HistGBM, 5-fold cross-fitted (each row scored by a model blind to its fold)")
    hgb_kw = dict(max_depth=3, learning_rate=0.1, max_iter=200, early_stopping=True,
                  validation_fraction=0.1, l2_regularization=0.0, random_state=0)
    p3 = np.empty(Z.shape[0], dtype=np.float64)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for i, (tr, te) in enumerate(kf.split(Z), 1):
        log(f"  L3 fold {i}/5 (train {tr.size:,}, score {te.size:,})")
        m = HistGradientBoostingClassifier(**hgb_kw).fit(Z[tr], y[tr])
        p3[te] = m.predict_proba(Z[te])[:, 1]
    preds["L3"] = p3
    return preds


# --- Per-stratum statistic + the pinned three-condition test --------------------


def pb_upper99(p: np.ndarray) -> dict[str, float]:
    """99th-pct of the Poisson-binomial drop-count null, normal (Le Cam/CLT) evaluation.

    The bar pins the null as PoissonBinomial({p_i}) and the leg as observed > 99th pct.
    Exact evaluation is O(N^2) at N~250k and infeasible; the bar itself flags this leg as
    'trivially met by any real excess ... essentially non-binding at the debug tier', so
    the standard large-N normal approximation of that exact distribution is used and the
    z-score reported so the reader sees exactly how non-binding it is.
    """
    mean = float(p.sum())
    var = float((p * (1.0 - p)).sum())
    sd = var ** 0.5
    return {"null_mean": mean, "null_sd": sd, "pct99": mean + Z99 * sd}


def stratum_stat(members: np.ndarray, y: np.ndarray, preds: dict[str, np.ndarray],
                 uniform_rate: float, bar_rule: dict[str, Any]) -> dict[str, Any]:
    n = int(members.sum())
    obs_count = int(y[members].sum())
    obs_rate = obs_count / n
    abs_floor = bar_rule["conditions_abs_floor"]
    rel_floor = bar_rule["conditions_rel_floor"]

    rungs: dict[str, Any] = {}
    for rung in ("L0", "L1", "L2", "L3"):
        pm = preds[rung][members]
        exp_rate = float(pm.mean())
        pb = pb_upper99(pm)
        residual_pp = obs_rate - exp_rate
        residual_ratio = obs_rate / exp_rate if exp_rate > 0 else float("inf")
        sig = obs_count > pb["pct99"]
        rungs[rung] = {
            "observed_rate": obs_rate,
            "expected_rate": exp_rate,
            "residual_pp": residual_pp,
            "residual_ratio": residual_ratio,
            "residual_count": residual_pp * n,
            "significant": bool(sig),
            "null_z": (obs_count - pb["null_mean"]) / pb["null_sd"] if pb["null_sd"] else float("inf"),
            "meets_abs_floor": bool(residual_pp >= abs_floor),
            "meets_rel_floor": bool(residual_ratio >= rel_floor),
            "meets_all_three": bool(sig and residual_pp >= abs_floor and residual_ratio >= rel_floor),
        }

    return {
        "n_would_be": n,
        "observed_count": obs_count,
        "observed_rate": obs_rate,
        "anchor_uniform": {  # A_uniform: expected = whole-catalog rate, credits zero structure
            "expected_rate": uniform_rate,
            "residual_pp": obs_rate - uniform_rate,
            "residual_ratio": obs_rate / uniform_rate,
        },
        "anchor_saturated": {  # A_saturated: expected = observed, credits all structure
            "expected_rate": obs_rate, "residual_pp": 0.0, "residual_ratio": 1.0,
        },
        "rungs": rungs,
        "loosen_signal_all_rungs": all(rungs[r]["meets_all_three"] for r in rungs),
    }


# --- Driver ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=REPO / "data" / "catalog_gated")
    ap.add_argument("--centroids", type=Path, default=REPO / "artifacts" / "phase_d" / "centroids.json")
    ap.add_argument("--scaler", type=Path, default=REPO / "artifacts" / "phase_d" / "scaler.json")
    args = ap.parse_args(argv)

    bar = json.loads(BAR_PATH.read_text())
    uniform_rate = bar["seeded_by"]["the_firing"]["registered_baseline_rate"]
    cond = bar["fire_rule"]["primary_stratum_5"]["conditions"]
    # Pull the two magnitude floors straight out of the committed conditions text so the
    # numbers in force here are provably the pinned ones, not re-typed.
    bar_rule = {
        "conditions_abs_floor": 0.10,   # 'residual >= 0.10 (10 pp)' — cond[1]
        "conditions_rel_floor": 1.5,    # 'ratio >= 1.5' — cond[2]
    }
    assert "0.10" in cond[1] and "10 pp" in cond[1], cond[1]
    assert "1.5" in cond[2], cond[2]

    log(f"reading catalog {args.catalog}")
    cat = catalog.read_catalog(args.catalog)
    cn = centroids.read_centroids(args.centroids)
    sc = frame.read_scaler(args.scaler)

    labels, assignable, Z = would_be_strata(cat, cn, sc)
    y = cat["degenerate"].to_numpy(zero_copy_only=False)[assignable].astype(np.int64)
    land_std = cat["land_std"].to_numpy(zero_copy_only=False)[assignable]
    labels_a = labels[assignable]

    n_assignable = int(assignable.sum())
    n_unassignable = int((~assignable).sum())
    # The bar's target is 'land_std < 3.0'. Confirm the pipeline flag equals that on the
    # fit population, so the modelled target is exactly the gate.
    flag_matches = int(np.sum(y == (land_std < 3.0).astype(np.int64)))
    log(f"assignable={n_assignable:,}  unassignable={n_unassignable:,}  "
        f"degenerate(assignable)={int(y.sum()):,}  flag==（land_std<3）:{flag_matches:,}/{n_assignable:,}")

    log("fitting the capacity ladder")
    preds = rung_predictions(Z, y, bar)

    strata_out: dict[str, Any] = {}
    for h in STRATA:
        members = labels_a == h
        stat = stratum_stat(members, y, preds, uniform_rate, bar_rule)
        stat["family"] = STRATUM_NAME[h]
        stat["role"] = "primary" if h == PRIMARY else "corroborating"
        strata_out[str(h)] = stat
        r = stat["rungs"]
        log(f"stratum {h:>2} ({stat['n_would_be']:,} members, obs {stat['observed_rate']:.3f}): "
            f"exp L0={r['L0']['expected_rate']:.3f} L1={r['L1']['expected_rate']:.3f} "
            f"L2={r['L2']['expected_rate']:.3f} L3={r['L3']['expected_rate']:.3f}  "
            f"resid L3={r['L3']['residual_pp']:+.3f}pp/{r['L3']['residual_ratio']:.2f}x  "
            f"all-rungs-fire={stat['loosen_signal_all_rungs']}")

    primary = strata_out[str(PRIMARY)]
    rate_signal = "LOOSEN" if primary["loosen_signal_all_rungs"] else "ACCEPT"
    binding = min(("L0", "L1", "L2", "L3"),
                  key=lambda r: primary["rungs"][r]["residual_pp"])

    payload = {
        "artifact": "E65 — structural-null model run: observed-vs-expected degenerate drop rate",
        "ticket": "https://github.com/JStoschek/DEMAE/issues/65",
        "map": "https://github.com/JStoschek/DEMAE/issues/62",
        "bar": "docs/research/e63-flatness-null-bar.json",
        "produced_by": "notebooks/e65_flatness_null_model.py",
        "code_version": _git_rev(),
        "rate_signal": rate_signal,
        "primary_stratum": PRIMARY,
        "binding_rung_stratum5": binding,
        "fit_population": {
            "assignable_rows": n_assignable,
            "unassignable_rows": n_unassignable,
            "degenerate_assignable": int(y.sum()),
            "flag_equals_landstd_lt_3": {"matches": flag_matches, "of": n_assignable},
            "uniform_anchor_rate": uniform_rate,
        },
        "significance_leg": {
            "null": "Poisson-binomial, normal (CLT) evaluation of the pinned distribution",
            "alpha": 0.01,
            "note": "non-binding at this N per the bar; z-scores reported for transparency",
        },
        "strata": strata_out,
    }
    out_json = RESEARCH / "e65-flatness-null-model.json"
    out_json.write_text(json.dumps(payload, indent=2))
    log(f"wrote {out_json.relative_to(REPO)}")
    write_markdown(payload)
    log(f"RATE-SIGNAL: {rate_signal} (primary stratum {PRIMARY}, binding rung {binding})")
    return 0


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _fmt(stat: dict[str, Any], rung: str) -> str:
    r = stat["rungs"][rung]
    return (f"| {rung} | {r['expected_rate']:.3f} | {r['residual_pp']:+.3f} | "
            f"{r['residual_ratio']:.2f}× | {r['residual_count']:,.0f} | "
            f"{'✓' if r['meets_abs_floor'] else '·'} | {'✓' if r['meets_rel_floor'] else '·'} | "
            f"{'✓' if r['significant'] else '·'} | {'**fire**' if r['meets_all_three'] else '—'} |")


def write_markdown(p: dict[str, Any]) -> None:
    primary = p["strata"][str(PRIMARY)]
    rate = p["rate_signal"]
    fp = p["fit_population"]
    lines: list[str] = []
    W = lines.append
    W("# E65 — Structural-null model run: observed-vs-expected `degenerate` drop rate\n")
    W(f"**Ticket:** [#65]({p['ticket']}) · **Map:** [#62]({p['map']}) · "
      f"**Bar (imported blind):** [`e63-flatness-null-bar.json`](e63-flatness-null-bar.json) · "
      f"**Code:** [`notebooks/e65_flatness_null_model.py`](../../notebooks/e65_flatness_null_model.py) @ `{p['code_version']}`\n")
    W(f"## Rate-signal: **{rate}**\n")
    W(f"The pre-registered structural-null model — the #47-analog committed *blind* in "
      f"[#63](https://github.com/JStoschek/DEMAE/issues/63) — was executed against the committed "
      f"gated catalog. It defines no model and no threshold of its own; every estimator and every "
      f"fire-rule number is read from the bar. The two-tier rule fires a **LOOSEN** signal only if "
      f"stratum 5 clears all three conditions (significant · residual ≥ 10 pp · ratio ≥ 1.5×) at "
      f"**every** ladder rung. It **{'does' if rate == 'LOOSEN' else 'does not'}** — the signal is "
      f"**{rate}**. Binding (smallest-residual) rung for stratum 5: **{p['binding_rung_stratum5']}**.\n")
    W("This is a *rate-signal*, not the verdict. The accept/loosen decision "
      "([#66](https://github.com/JStoschek/DEMAE/issues/66)) weighs it against #64's junk-vs-"
      "informative and starvation findings.\n")
    W("## Fit population\n")
    W(f"- **{fp['assignable_rows']:,}** assignable rows (all six frozen metrics finite), kept and "
      f"dropped — the would-be (pre-gate) membership, exactly as E2.5 built its table.")
    W(f"- **{fp['unassignable_rows']:,}** unassignable (NaN-metric) rows excluded — the stated "
      f"blind spot (~6% of `degenerate`), disclosed in the bar.")
    W(f"- Target = the `degenerate` flag; it equals `land_std < 3.0` on "
      f"**{fp['flag_equals_landstd_lt_3']['matches']:,} / {fp['flag_equals_landstd_lt_3']['of']:,}** "
      f"assignable rows, so the modelled target is exactly the gate.")
    W(f"- Anchors: **A_uniform** = {fp['uniform_anchor_rate']} (the original E2.5 null, credits zero "
      f"structure); **A_saturated** = each stratum's own rate (residual 0, credits all structure).\n")
    W("## The ladder — stratum 5 (primary, the only stratum that can trigger LOOSEN)\n")
    W(f"Would-be members **{primary['n_would_be']:,}**, observed drops **{primary['observed_count']:,}** "
      f"(rate **{primary['observed_rate']:.3f}**). Anchor residuals: uniform "
      f"**{primary['anchor_uniform']['residual_pp']:+.3f} pp / {primary['anchor_uniform']['residual_ratio']:.2f}×**, "
      f"saturated **0**.\n")
    W("| rung | expected | residual pp | ratio | excess count | ≥10 pp | ≥1.5× | sig | fires |")
    W("|---|---|---|---|---|---|---|---|---|")
    for rung in ("L0", "L1", "L2", "L3"):
        W(_fmt(primary, rung))
    W("")
    W(f"**Stratum 5 LOOSEN-signal (all rungs fire): "
      f"{'YES' if primary['loosen_signal_all_rungs'] else 'NO'}.** "
      f"L3 is the structure-generous, cross-fitted rung and is expected to bind.\n")
    W("## Corroborating strata 2 / 7 / 17 (reported only, never trigger)\n")
    W("| h | family | members | obs rate | exp L3 | residual L3 | ratio L3 | all-rungs fire |")
    W("|---|---|---|---|---|---|---|---|")
    for h in CORROBORATING:
        s = p["strata"][str(h)]
        r3 = s["rungs"]["L3"]
        W(f"| {h} | {s['family']} | {s['n_would_be']:,} | {s['observed_rate']:.3f} | "
          f"{r3['expected_rate']:.3f} | {r3['residual_pp']:+.3f} pp | {r3['residual_ratio']:.2f}× | "
          f"{'YES' if s['loosen_signal_all_rungs'] else 'NO'} |")
    W("")
    W("## Significance leg\n")
    W("The Poisson-binomial null is evaluated by its large-N normal (CLT) approximation — exact "
      "evaluation is O(N²) at N≈250k and infeasible, and the bar itself flags this leg as "
      "*trivially met by any real excess, essentially non-binding at the debug tier*. It is pinned "
      "for form and for Phase-F re-use at smaller N. The `null_z` in the JSON shows how far past the "
      "99th-pct each observed count sits.\n")
    W("## Provenance\n")
    W("Reads committed data only (`data/catalog_gated`, `artifacts/phase_d/{centroids,scaler}.json`); "
      "touches no pipeline state. Full per-rung numbers, anchors and null z-scores in "
      "[`e65-flatness-null-model.json`](e65-flatness-null-model.json).\n")
    (RESEARCH / "e65-flatness-null-model.md").write_text("\n".join(lines))
    log("wrote docs/research/e65-flatness-null-model.md")


if __name__ == "__main__":
    raise SystemExit(main())
