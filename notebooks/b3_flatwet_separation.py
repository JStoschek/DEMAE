"""B3 (#27) — flat-wet separation test + tpi_proxy re-entry decision.

Resolves the Phase-B map's (#23) one explicit TPI re-entry trigger. The B2 prune
(#26) proposed a 6-dim surviving vector but demoted the flat-wet/flat-dry axis to
"open, unresolved". This settles it by measurement against the labelled exemplar
set (#25).

Question tested
---------------
In the descriptor space of the 6 SURVIVING metrics, do the flat-wet exemplars
separate from the flat-dry? And does appending ``tpi_proxy`` improve that
separation (its re-entry trigger)?

Method (chosen so no single knob can rig the answer)
----------------------------------------------------
* Features standardized against the PILOT FRAME (15k windows), not the 8
  exemplars: that is the space production k-means will run in, so distances must
  reflect production scale.
* Transforms mirror #26 exactly: ``ln(x+1)`` on ``slope_mean`` and
  ``local_relief``; the other four survivors and ``tpi_proxy`` left raw.
* Separation operationalized three ways:
    1. leave-one-out (LOO) nearest-centroid classification of the 8 exemplars;
    2. a Fisher-style margin = wet/dry centroid gap in pooled within-class-std
       units, over the FULL survivor space;
    3. per-exemplar signed margin ``d(dry-centroid) - d(wet-centroid)`` (>0 =>
       lands on the wet side), with Etosha (the featureless wet pan) tracked.
* Every quantity is recomputed WITH ``tpi_proxy`` appended, to isolate its
  marginal value.

Result (2026-07-20): survivors separate 6/8 (Fisher 1.30); appending tpi_proxy
leaves margin and LOO identical to 3 decimals and does not rescue the one
structural miss (Etosha). => tpi_proxy stays PRUNED. The Etosha miss is a
featureless saturated pan with no terrain-shape signature of wetness; its wet
signal lives in the EDM/WBM masks, outside the terrain vector's scope.

Runs on any env with numpy/pandas/scipy; renders the figure only if matplotlib
is importable (the repo's analysis env and plotting env are split).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "docs" / "research"
SURVIVORS = ["slope_mean", "local_relief", "texture",
             "elevation_relief_ratio", "mean_elevation", "geomorphon_entropy"]
LOG_COLS = {"slope_mean", "local_relief"}          # ln(x+1) per #26
CANDIDATE = "tpi_proxy"


def featurize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = {c: (np.log1p(df[c].astype(float).values) if c in LOG_COLS
               else df[c].astype(float).values) for c in cols}
    return pd.DataFrame(out, index=df.index)


def load():
    pilot = pd.read_csv(RESEARCH / "pilot-frame-per-window.csv")
    ex = pd.read_csv(RESEARCH / "exemplars-per-window.csv").rename(
        columns={"klass": "class"})
    return pilot, ex


def z_exemplars(pilot, ex, cols):
    """Exemplars z-scored to pilot-frame (transformed) mean/std."""
    P = featurize(pilot, cols)
    mu, sd = P.mean(axis=0), P.std(axis=0, ddof=0).replace(0, 1.0)
    return (featurize(ex, cols) - mu) / sd


def separation_report(pilot, ex, cols, label):
    Z = z_exemplars(pilot, ex, cols)
    wet = (ex["class"] == "flat-wet").values
    dry = (ex["class"] == "flat-dry").values
    names = ex["name"].values
    cw, cd = Z[wet].mean(axis=0).values, Z[dry].mean(axis=0).values

    gap = float(np.linalg.norm(cw - cd))
    within = float(np.sqrt(np.mean(
        [np.sum((r.values - cw) ** 2) for _, r in Z[wet].iterrows()] +
        [np.sum((r.values - cd) ** 2) for _, r in Z[dry].iterrows()])))

    correct, per = 0, []
    for i in range(len(Z)):
        m = np.ones(len(Z), bool); m[i] = False
        cw_i = Z[wet & m].mean(axis=0).values
        cd_i = Z[dry & m].mean(axis=0).values
        xi = Z.iloc[i].values
        dw, dd = np.linalg.norm(xi - cw_i), np.linalg.norm(xi - cd_i)
        pred_wet = dw < dd
        ok = pred_wet == wet[i]
        correct += ok
        per.append(dict(name=str(names[i]), true="wet" if wet[i] else "dry",
                        pred="wet" if pred_wet else "dry", ok=bool(ok),
                        signed_margin=float(dd - dw)))
    return dict(label=label, n_features=len(cols), features=list(cols),
                centroid_gap=gap, within_rms=within,
                fisher_margin=(gap / within if within else float("inf")),
                loo_correct=int(correct), per=per)


def make_figure(res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("[matplotlib unavailable — skipping figure]")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, rep in zip(axes, (res["survivors"], res["with_tpi"])):
        per = sorted(rep["per"], key=lambda p: p["signed_margin"])
        y = np.arange(len(per))
        colors = ["#1f77b4" if p["true"] == "wet" else "#d62728" for p in per]
        ax.barh(y, [p["signed_margin"] for p in per], color=colors,
                edgecolor=["k" if not p["ok"] else "none" for p in per],
                linewidth=[2 if not p["ok"] else 0 for p in per])
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(y); ax.set_yticklabels([p["name"] for p in per], fontsize=9)
        ax.set_title(f"{rep['label']}  (LOO {rep['loo_correct']}/8, "
                     f"Fisher {rep['fisher_margin']:.2f})", fontsize=10)
        ax.set_xlabel("signed margin  d(dry) - d(wet)   >0 = wet side")
    axes[0].legend(handles=[Patch(color="#1f77b4", label="true flat-wet"),
                            Patch(color="#d62728", label="true flat-dry"),
                            Patch(facecolor="w", edgecolor="k", lw=2,
                                  label="LOO misclassified")],
                   loc="lower right", fontsize=8)
    fig.suptitle("B3 (#27): wet/dry separation of the 8 exemplars — "
                 "appending tpi_proxy changes nothing", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    print(f"[wrote {path}]")


def main():
    pilot, ex = load()
    surv = separation_report(pilot, ex, SURVIVORS, "survivors (6)")
    tpi = separation_report(pilot, ex, SURVIVORS + [CANDIDATE],
                            "survivors + tpi_proxy (7)")

    P = featurize(pilot, SURVIVORS + [CANDIDATE]).dropna()
    rho = {c: float(spearmanr(P[c], P[CANDIDATE]).statistic) for c in SURVIVORS}

    for rep in (surv, tpi):
        print(f"\n=== {rep['label']} ===")
        print(f"  Fisher margin: {rep['fisher_margin']:.3f}   "
              f"LOO: {rep['loo_correct']}/8")
        for p in rep["per"]:
            print(f"    {p['name']:<28} true={p['true']:<3} pred={p['pred']:<3} "
                  f"margin={p['signed_margin']:+.3f}"
                  f"{'   <-- MISCLASSIFIED' if not p['ok'] else ''}")
    print("\n=== tpi_proxy Spearman rho vs each survivor (pilot frame) ===")
    for c, r in sorted(rho.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {c:<24} rho={r:+.3f}")

    make_figure({"survivors": surv, "with_tpi": tpi},
                RESEARCH / "b3-flatwet-separation.png")
    (RESEARCH / "b3-flatwet-separation.json").write_text(
        json.dumps({"survivors": surv, "with_tpi": tpi, "tpi_rho": rho}, indent=2))
    print(f"[wrote {RESEARCH / 'b3-flatwet-separation.json'}]")


if __name__ == "__main__":
    main()
