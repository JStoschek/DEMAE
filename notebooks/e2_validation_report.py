"""E2 (#52) — the quantitative validation report: does the corpus match its design?

This is Phase E's **Definition of Done** on the numbers side. E1 turned a committed
selection list into shards, a manifest and a ledger; this reads all three back and asks,
against a bar written *before* any of them existed, whether the corpus faithfully
implements the design and whether the design survives contact with the corpus.

The Brungard & Boettinger method (plan §180-189) is distribution-matching: you show the
sample's distributions beside the frame's rather than asserting a sample is good. Four
of E2's five analyses live here — the fifth, the 12-tiles-per-stratum gallery (#53), is a
human protocol and ships separately:

1. **E2.1 — metric distributions.** Per-metric ECDFs, corpus against the kept pool.
   Deliberately *not* a gate: the corpus is terrain-diverse, not pool-representative
   (Neyman grants the rarest stratum a 1.8x per-row premium), so "matches the pool" is
   the wrong success criterion and the bar says so in as many words. What *is* gated is
   fidelity (did E1 write the metric values it was told to, E2.1a) and drop selectivity
   (were the full-res failures terrain-correlated, E2.1b).
2. **E2.2 — allocation fidelity.** Realized per-stratum counts against
   [`d2-allocation-nh.json`](../docs/research/d2-allocation-nh.json), the thin-stratum
   condition, and the coastal floor honoured as the **rate** it always was.
3. **E2.3 — geographic coverage.** Per-geo-cell counts and a coverage map, published;
   with one tripwire, the retired per-COG cap of 64, watching for concentration
   selection never intended.
4. **E2.5 — attrition by stratum.** What each quality gate discarded per stratum, from
   the ungated catalog down to the materialized corpus. This is the table plan risk #3
   has been waiting on since Phase C.

## What this script does not do

**It does not render the verdict.** The bar's Defect-first / any-Finding-fails roll-up
belongs to the Gate E review (#54); this emits per-criterion pass/fail and, for anything
that fired, the bar's four-field payload. **It does not set a threshold either** — every
number it gates on comes off [`e2-gate-bar.json`](../docs/research/e2-gate-bar.json) via
:mod:`dataset.validate`, which refuses to run against a bar whose prose has drifted from
the constants it carries.

## Re-runnable per tier

Phase F re-runs this at 200-500k, so nothing here is a 50 k constant: the budget comes
from the selection list's length, the allocation from the committed n_h table, the live
attrition contract from ``corpus_meta.json``, and every path is an argument. The only
tier-specific values are the committed_inputs the bar itself names (md5, row count,
provenance split) — which is what E2.2a exists to check, and which Phase F re-registers
in its own bar.

Zero-network; numpy + pyarrow + matplotlib only (this repo's env has no pandas).

Run:   uv run python notebooks/e2_validation_report.py
Writes: docs/research/e2-validation-report.json  — every number the doc quotes
        docs/research/e2-metric-ecdfs.png
        docs/research/e2-allocation-fidelity.png
        docs/research/e2-coverage-map.png
        docs/research/e2-gate-attrition.png
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dataset import catalog, centroids, frame, gates, validate

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "docs" / "research"

#: ADR-0004 §1's transforms — the space the strata were cut in, so the space the ECDFs
#: are read in. A raw-space ECDF of ``local_relief`` is a spike at the origin.
LOG_METRICS = frame.LOG_METRICS
#: Quantiles printed per metric beside the W1 distances. Same set C5 used, so the corpus
#: column can be read straight against ``c5-catalog-summary.json``'s pool column.
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
#: ECDF sample points. 512 is well past what the eye resolves on a 4-inch panel and keeps
#: the committed JSON to a few hundred KB rather than a few hundred MB.
ECDF_POINTS = 512


# --- Loading ------------------------------------------------------------------


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_selection(path: Path) -> dict[str, Any]:
    """The Phase-D selection list as committed, plus the md5 of the bytes just parsed.

    The md5 is taken over the file this run actually read, not carried from the ticket:
    E2.2a's whole point is that "the selection list AS THE DRIVER LOADED IT" matches
    Phase D's commit, and a hash copied from elsewhere would check the wrong thing.
    """
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {
        "path": path,
        "md5": _md5(path),
        "window_id": [r["window_id"] for r in rows],
        "stratum_id": np.array([int(r["stratum_id"]) for r in rows], dtype=np.int32),
        "provenance": [r["provenance"] for r in rows],
    }


def load_manifest(corpus_dir: Path) -> dict[str, Any]:
    """Manifest columns + the reconstructed ``window_id`` the catalog joins on.

    The v2 manifest indexes tiles by ``tile_id`` (``{stem}_{row}_{col}``) and keeps the
    parts in their own columns; the catalog keys on ``window_id`` (``{stem}:{row}:{col}``).
    Rebuilding the catalog's key from the manifest's parts — rather than string-munging
    ``tile_id`` — means a COG stem containing an underscore cannot silently mis-join.
    """
    table = pq.read_table(corpus_dir / "manifest.parquet")
    out: dict[str, Any] = {name: table.column(name).to_numpy(zero_copy_only=False)
                           for name in table.schema.names if name != "split"}
    out["window_id"] = [
        f"{stem}:{row}:{col}"
        for stem, row, col in zip(out["source_cog"], out["win_row0"], out["win_col0"])
    ]
    return out


def load_ledger(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    return {
        "window_id": table.column("window_id").to_pylist(),
        "stratum_id": table.column("stratum_id").to_numpy(zero_copy_only=False),
        "provenance": table.column("provenance").to_pylist(),
        "status": np.array(table.column("status").to_pylist()),
    }


def load_catalog(catalog_dir: Path) -> dict[str, Any]:
    """The gated catalog columns this report needs — one numpy array per column.

    ``window_id`` stays a pyarrow array: it is only ever used as a join key, and
    ``pyarrow.compute.index_in`` hashes it in C++ without materializing 4.4 M Python
    strings (which is most of a gigabyte for a lookup that takes seconds).
    """
    if not (catalog_dir / catalog.PARTS_DIR).exists():
        raise SystemExit(
            f"no gated catalog at {catalog_dir}. Build it with:\n"
            "  uv run python -m dataset.catalog   # the global pass (hours)\n"
            "  uv run python -m dataset.gates     # the gates (seconds)"
        )
    wanted = ["window_id", "stem", "geocell", "lon", "lat",
              *catalog.FROZEN_METRICS, *gates.FLAG_COLUMNS]
    table = pq.read_table(catalog_dir / catalog.PARTS_DIR, columns=wanted)
    out: dict[str, Any] = {"window_id": table.column("window_id").combine_chunks()}
    for name in wanted[1:]:
        out[name] = table.column(name).to_numpy(zero_copy_only=False)
    return out


def join_rows(catalog_ids: Any, window_ids: list[str], *, label: str) -> np.ndarray:
    """Catalog row index per requested ``window_id``; a miss is fatal, not a NaN.

    Every window this report looks up was selected *from* the catalog, so a miss means
    the join key is wrong — and a wrong join that silently produced NaNs would make
    E2.1a pass by comparing nothing.
    """
    idx = pc.index_in(pc.cast(window_ids, catalog_ids.type), value_set=catalog_ids).to_numpy(
        zero_copy_only=False
    )
    missing = np.flatnonzero([v is None for v in idx])
    if missing.size:
        raise SystemExit(
            f"{missing.size} {label} window_id(s) are absent from the catalog, e.g. "
            f"{window_ids[int(missing[0])]!r} — the join key is wrong, not the data"
        )
    return idx.astype(np.int64)


# --- E2.1 metric distributions -------------------------------------------------


def _transform(name: str, values: np.ndarray) -> np.ndarray:
    return np.log1p(values) if name in LOG_METRICS else values


def _ecdf(values: np.ndarray, grid: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Empirical CDF of ``values`` on ``grid``, optionally weighted per observation."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    if weights is None:
        cum = np.arange(1, ordered.size + 1, dtype=np.float64) / ordered.size
    else:
        w = np.asarray(weights, dtype=np.float64)[order]
        cum = np.cumsum(w) / w.sum()
    return np.concatenate([[0.0], cum])[np.searchsorted(ordered, grid, side="right")]


def pool_comparison(
    corpus: dict[str, np.ndarray], pool: dict[str, np.ndarray], weights: np.ndarray
) -> dict[str, Any]:
    """E2.1c: per-metric ECDF overlays and W1 distances, corpus vs the kept pool.

    Published, never gated. The corpus is *deliberately* not pool-distributed — Neyman
    over-samples the rare rugged strata on purpose — so a large W1 here is the design
    working, not failing. The second, ``N_h/n_h``-weighted curve is the check that the
    over-sampling is the *intended* one: re-weighting each tile by its stratum's
    inclusion odds should walk the corpus back onto the pool.
    """
    out: dict[str, Any] = {"quantiles": list(QUANTILES), "ecdf_points": ECDF_POINTS}
    per_metric: dict[str, Any] = {}
    for name in catalog.FROZEN_METRICS:
        c = _transform(name, corpus[name])
        p = _transform(name, pool[name])
        # The weights ride along with the corpus filter. In practice nothing is dropped
        # (C5's invariant: no kept window carries a NaN metric), but a silent misalignment
        # between tile and weight would corrupt the reweighted curve invisibly.
        finite = np.isfinite(c)
        c, w = c[finite], weights[finite]
        p = p[np.isfinite(p)]
        lo, hi = np.quantile(p, [0.001, 0.999])
        grid = np.linspace(lo, hi, ECDF_POINTS)
        per_metric[name] = {
            "transform": "ln(x+1)" if name in LOG_METRICS else "raw",
            "w1_corpus_vs_pool": validate.wasserstein1(c, p),
            "w1_reweighted_vs_pool": _weighted_w1(c, w, p),
            "corpus_quantiles": {
                str(q): float(v) for q, v in zip(QUANTILES, np.quantile(c, QUANTILES))
            },
            "pool_quantiles": {
                str(q): float(v) for q, v in zip(QUANTILES, np.quantile(p, QUANTILES))
            },
            "grid": grid.tolist(),
            "ecdf_corpus": _ecdf(c, grid).tolist(),
            "ecdf_corpus_reweighted": _ecdf(c, grid, w).tolist(),
            "ecdf_pool": _ecdf(p, grid).tolist(),
        }
    out["per_metric"] = per_metric
    return out


def _weighted_w1(corpus: np.ndarray, weights: np.ndarray, pool: np.ndarray) -> float:
    """W1 between the ``N_h/n_h``-weighted corpus and the pool.

    Same integral as :func:`dataset.validate.wasserstein1`, with the corpus CDF weighted
    instead of counted. Kept here rather than in the module because the weighted form is
    a descriptive figure label, not something the bar gates on.
    """
    grid = np.sort(np.concatenate([corpus, pool]))
    widths = np.diff(grid)
    return float(np.sum(np.abs(_ecdf(corpus, grid[:-1], weights) - _ecdf(pool, grid[:-1])) * widths))


# --- E2.2 allocation fidelity --------------------------------------------------


def chi_square(realized: dict[int, int], expected: dict[int, int]) -> dict[str, float]:
    """Pearson's X² of realized against n_h — **descriptive only**, and here on sufferance.

    Plan §182 asked for a chi-square test; #47 dropped it from the bar and said why: the
    sampler is deterministic and consumes a committed table, so under a correct build the
    expected discrepancy is exactly zero and a p-value is decoration. It is computed
    anyway because the ticket asks for the number, and reported without a p-value so
    nobody reads a significance claim into a statistic whose null is "the code works".
    """
    strata = sorted(set(realized) | set(expected))
    obs = np.array([realized.get(h, 0) for h in strata], dtype=np.float64)
    exp = np.array([expected.get(h, 0) for h in strata], dtype=np.float64)
    scaled = exp * (obs.sum() / exp.sum()) if exp.sum() else exp
    with np.errstate(invalid="ignore", divide="ignore"):
        terms = np.where(scaled > 0, (obs - scaled) ** 2 / scaled, 0.0)
    return {
        "statistic_vs_n_h": float(np.sum(np.where(exp > 0, (obs - exp) ** 2 / exp, 0.0))),
        "statistic_vs_rescaled_n_h": float(terms.sum()),
        "dof": int(len(strata) - 1),
        "note": "descriptive; #47 dropped chi-square from the bar (deterministic sampler)",
    }


# --- E2.3 coverage -------------------------------------------------------------


def coverage(
    manifest: dict[str, Any], cat: dict[str, Any], cell_deg: int = 10
) -> dict[str, Any]:
    """Per-geo-cell corpus counts against the kept pool's occupied cells.

    Published, never gated: any geographic pass condition smuggles geo-uniformity back in
    against Neyman, and a 50 k draw over 15,834 COGs cannot touch every cell. The number
    that matters for the E3 honesty note is how many *kept-pool* cells the corpus reaches
    and how the misses are distributed.
    """
    keep = cat["keep"]
    pool_cells = Counter(cat["geocell"][keep].tolist())
    corpus_cells = Counter(manifest["geocell_id"].tolist())
    missed = sorted(set(pool_cells) - set(corpus_cells))
    lon_edges = np.arange(-180, 181, cell_deg)
    lat_edges = np.arange(-90, 91, cell_deg)
    grid, _, _ = np.histogram2d(
        manifest["lon"], manifest["lat"], bins=[lon_edges, lat_edges]
    )
    pool_grid, _, _ = np.histogram2d(
        cat["lon"][keep], cat["lat"][keep], bins=[lon_edges, lat_edges]
    )
    per_cog = Counter(manifest["source_cog"].tolist())
    return {
        "cell_deg": cell_deg,
        # Both Phase-C denominators: 472 cells hold a scored window, 425 hold a *kept*
        # one. The corpus can only be drawn from the kept pool, so 425 is the honest
        # denominator and 472 is the number the plan's coverage sentence quotes.
        "n_catalog_cells_occupied": int(np.unique(cat["geocell"]).size),
        "n_pool_cells_occupied": len(pool_cells),
        "n_corpus_cells_occupied": len(corpus_cells),
        "n_pool_cells_unreached": len(missed),
        "pool_cells_unreached": missed,
        "cell_count_quantiles": {
            str(q): float(v)
            for q, v in zip(
                QUANTILES, np.quantile(np.array(sorted(corpus_cells.values())), QUANTILES)
            )
        },
        "max_tiles_in_a_cell": max(corpus_cells.values()),
        "n_cogs_used": len(per_cog),
        "grid": grid.tolist(),
        "pool_grid": pool_grid.tolist(),
        "lon_edges": lon_edges.tolist(),
        "lat_edges": lat_edges.tolist(),
        "_per_cog": per_cog,
    }


# --- E2.5 attrition by stratum -------------------------------------------------


def would_be_strata(cat: dict[str, Any], centroids_: centroids.Centroids, scaler: frame.Scaler
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Give every catalog window the stratum it would have joined, dropped ones included.

    k-means was fit on survivors only, so the 1.33 M dropped windows carry full metrics
    and no ``stratum_id`` (#37: gates are columns, so the rows are still there). Scoring
    them against the committed centroids is what makes a per-(gate x stratum) table
    possible at all — and it is exact for the kept rows, since :func:`centroids.assign`
    is the same pure argmin the sampler labelled them with.

    Returns ``(stratum_id, assignable)``. A window whose metric vector has a NaN cannot
    be scored — that means it had no land to measure — so it is excluded and counted
    rather than imputed into somebody's terrain family.
    """
    X = frame.metric_matrix(cat)
    assignable = np.isfinite(X).all(axis=1)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels[assignable] = centroids.assign(scaler.standardize(X[assignable]), centroids_)
    return labels, assignable


def attrition_table(
    cat: dict[str, Any], labels: np.ndarray, assignable: np.ndarray, ledger: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The per-(gate x stratum) drop table, with materialization drops as a last column.

    Returns ``(published_table, rule_inputs)`` — the table as it lands in the artifact,
    and the int-keyed counts :func:`dataset.validate.gate_stratum_bias` consumes. Kept
    apart rather than smuggled into one dict under underscore keys: they are two
    different things, one for a reader and one for a rule.

    Two attritions in one table, per #50 item 4, but only the Phase-C gate columns carry
    E2.5's 3x/10% test: the materialization column's *count* effect is E2.2c's question
    and its *metric selectivity* is E2.1b's, so gating it here a third time would triple-
    count one event.

    **Two baseline populations, one of them gating.** The bar's ratio limb compares a
    within-stratum rate against "that same gate's corpus-wide drop rate", which reads on
    the whole 4,386,720-row catalog — that is ``corpus_fires``/``corpus_rows`` and it is
    what the verdict runs on. ``narrow_*`` restricts both to the 3,634,257 *assignable*
    rows, the same population the per-stratum rates use, which is arguably the cleaner
    comparison and materially changes which pairs fire for ``water`` (whose whole-catalog
    rate is dominated by the no-land windows that belong to no stratum). It is reported
    as a **sensitivity**, never as a second verdict: choosing the denominator after
    seeing which one fires is the peeking the pre-registration exists to prevent.
    """
    strata = sorted(set(labels[assignable].tolist()))
    membership = {h: int((labels == h).sum()) for h in strata}
    fires = {
        gate: {h: int((cat[gate] & (labels == h)).sum()) for h in strata}
        for gate in gates.DROP_GATES
    }
    corpus_fires = {gate: int(cat[gate].sum()) for gate in gates.DROP_GATES}
    narrow_fires = {gate: int((cat[gate] & assignable).sum()) for gate in gates.DROP_GATES}

    # The E1 half: per-stratum materialization outcomes, off the ledger.
    mat_drop = Counter(
        int(h) for h, s in zip(ledger["stratum_id"], ledger["status"]) if s != "ok"
    )
    mat_selected = Counter(int(h) for h in ledger["stratum_id"])
    kept_per_stratum = {h: int((cat["keep"] & (labels == h)).sum()) for h in strata}
    published_table = {
        "n_rows": int(labels.size),
        "n_assignable": int(assignable.sum()),
        "n_unassignable": int((~assignable).sum()),
        "n_unassignable_kept": int((~assignable & cat["keep"]).sum()),
        # Every unassignable window is a drop (a NaN metric means no land to measure, and
        # C5's invariant says no kept window carries one). Their gate fires are reported
        # here rather than folded into a stratum, so the table accounts for all 4.39 M rows.
        "unassignable_fires": {
            gate: int((cat[gate] & ~assignable).sum()) for gate in gates.DROP_GATES
        },
        "membership": {str(h): membership[h] for h in strata},
        "kept": {str(h): kept_per_stratum[h] for h in strata},
        "fires": {gate: {str(h): v for h, v in by.items()} for gate, by in fires.items()},
        "corpus_fires_whole_catalog": corpus_fires,
        "corpus_fires_assignable_only": narrow_fires,
        "materialization": {
            str(h): {
                "selected": int(mat_selected.get(h, 0)),
                "dropped": int(mat_drop.get(h, 0)),
            }
            for h in sorted(set(mat_selected) | set(mat_drop))
        },
        "materialization_causes": dict(
            Counter(s for s in ledger["status"].tolist() if s != "ok")
        ),
    }
    rule_inputs = {
        "fires": fires,
        "membership": membership,
        "corpus_fires": corpus_fires,
        "corpus_rows": int(labels.size),
        "narrow_fires": narrow_fires,
        "narrow_rows": int(assignable.sum()),
    }
    return published_table, rule_inputs


# --- Figures -------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def figure_ecdfs(comparison: dict[str, Any], path: Path) -> None:
    """2x3 of the frozen six: pool, corpus, and the N_h/n_h-reweighted corpus.

    The gap between the grey pool curve and the blue corpus curve *is* the Neyman
    over-sampling — the figure the E3 honesty note points at when it says the corpus is
    terrain-diverse rather than geographically or statistically representative. The
    dashed reweighted curve is the one that should sit back on the pool.
    """
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for ax, (name, entry) in zip(axes.ravel(), comparison["per_metric"].items()):
        grid = np.array(entry["grid"])
        ax.plot(grid, entry["ecdf_pool"], color="0.55", lw=2.4, label="kept pool")
        ax.plot(grid, entry["ecdf_corpus"], color="#1f77b4", lw=1.6, label="corpus")
        ax.plot(
            grid, entry["ecdf_corpus_reweighted"], color="#d62728", lw=1.3, ls="--",
            label="corpus x $N_h/n_h$",
        )
        unit = " [ln(x+1)]" if name in LOG_METRICS else ""
        ax.set_title(f"{name}{unit}   W1={entry['w1_corpus_vs_pool']:.3f}", fontsize=9)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "E2.1c (#52): per-metric ECDFs, corpus vs the kept pool — the gap is Neyman "
        "working, not failing (published, never a gate)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[wrote {path}]")


def figure_allocation(alloc: dict[str, Any], path: Path) -> None:
    """Selected vs realized per stratum, and the drop rate that E2.2c reads."""
    plt = _plt()
    strata = sorted(int(h) for h in alloc["per_stratum"])
    sel = [alloc["per_stratum"][str(h)]["selected"] for h in strata]
    real = [alloc["per_stratum"][str(h)]["realized"] for h in strata]
    x = np.arange(len(strata))

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), height_ratios=[2, 1])
    axes[0].bar(x - 0.2, sel, width=0.4, color="0.72", label="selected ($n_h$)")
    axes[0].bar(x + 0.2, real, width=0.4, color="#1f77b4", label="materialized")
    axes[0].axhline(
        alloc["absolute_floor"], color="#d62728", ls="--", lw=1.2,
        label=f"E2.2c floor ({alloc['absolute_floor']})",
    )
    axes[0].set_xticks(x, [str(h) for h in strata])
    axes[0].set_ylabel("windows")
    axes[0].legend(fontsize=8)
    axes[0].set_title(
        f"E2.2 (#52): allocation fidelity — {sum(real):,} of {sum(sel):,} selected windows "
        f"materialized",
        fontsize=11,
    )

    drops = [s - r for s, r in zip(sel, real)]
    axes[1].bar(x, drops, color="#d62728")
    for i, d in enumerate(drops):
        if d:
            axes[1].text(i, d, f" {d}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x, [str(h) for h in strata])
    axes[1].set_xlim(-0.6, len(strata) - 0.4)  # same columns as the panel above
    axes[1].set_xlabel("stratum_id")
    axes[1].set_ylabel("windows lost")
    axes[1].set_ylim(0, max(max(drops), 1) * 1.4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[wrote {path}]")


def figure_coverage(cov: dict[str, Any], per_cog: Counter, path: Path) -> None:
    """Where the corpus landed, and how thinly it spreads over its COGs."""
    plt = _plt()
    import matplotlib

    grid = np.array(cov["grid"])
    pool = np.array(cov["pool_grid"])
    extent = [-180, 180, -90, 90]

    fig = plt.figure(figsize=(11, 8.8))
    ax0 = fig.add_subplot(3, 1, (1, 2))
    counts = np.where(grid > 0, grid, np.nan)
    im = ax0.imshow(
        counts.T, origin="lower", extent=extent, aspect="auto", cmap="viridis",
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=np.nanmax(counts)),
    )
    unreached = (pool > 0) & (grid == 0)
    if unreached.any():
        li, la = np.nonzero(unreached)
        ax0.scatter(
            np.array(cov["lon_edges"])[li] + cov["cell_deg"] / 2,
            np.array(cov["lat_edges"])[la] + cov["cell_deg"] / 2,
            s=12, marker="x", color="#d62728",
            label=f"kept pool, no corpus tile ({int(unreached.sum())} cells)",
        )
        ax0.legend(fontsize=8, loc="lower left")
    ax0.set_title(
        f"E2.3 (#52): corpus tiles per {cov['cell_deg']}° cell (log) — "
        f"{cov['n_corpus_cells_occupied']} of {cov['n_pool_cells_occupied']} kept-pool "
        f"geo-cells reached. Published: a coverage floor would fight Neyman.",
        fontsize=10,
    )
    ax0.set_xlabel("longitude")
    ax0.set_ylabel("latitude")
    fig.colorbar(im, ax=ax0, label="corpus tiles")

    ax1 = fig.add_subplot(3, 1, 3)
    values = np.array(sorted(per_cog.values()))
    ax1.hist(values, bins=np.arange(0.5, values.max() + 1.5), color="#1f77b4")
    ax1.set_yscale("log")
    ax1.axvline(64, color="#d62728", ls="--", lw=1.2, label="E2.3 guard (64)")
    ax1.set_xlabel("tiles per source COG")
    ax1.set_ylabel("COGs")
    ax1.legend(fontsize=8)
    ax1.set_title(
        f"{len(per_cog):,} COGs; mean {values.mean():.2f}, median {np.median(values):.0f}, "
        f"p99 {np.percentile(values, 99):.0f}, max {values.max()}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[wrote {path}]")


def figure_attrition(table: dict[str, Any], path: Path) -> None:
    """Per-(gate x stratum) drop rate as a heatmap — E2.5's table, made legible.

    Rows are gates, columns are strata, cell colour is the gate's within-stratum drop
    rate. A gate that hollows out a terrain family is a bright cell in an otherwise even
    row; even attrition is a flat row, which is the pre-registered expectation.
    """
    plt = _plt()
    strata = sorted(int(h) for h in table["membership"])
    gate_names = list(table["fires"])
    rates = np.array(
        [
            [
                table["fires"][g][str(h)] / max(table["membership"][str(h)], 1)
                for h in strata
            ]
            for g in gate_names
        ]
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6), height_ratios=[3, 1.5])
    im = axes[0].imshow(rates, aspect="auto", cmap="magma_r", vmin=0, vmax=max(rates.max(), 0.01))
    axes[0].set_yticks(np.arange(len(gate_names)), gate_names)
    axes[0].set_xticks(np.arange(len(strata)), [str(h) for h in strata])
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            axes[0].text(
                j, i, f"{rates[i, j]:.0%}", ha="center", va="center", fontsize=7,
                color="white" if rates[i, j] > rates.max() * 0.55 else "black",
            )
    axes[0].set_title(
        "E2.5 (#52): quality-gate drop rate per (gate x would-be stratum) — dropped "
        "windows scored against the committed centroids",
        fontsize=11,
    )
    # The colorbar steals width from *both* axes, so the two panels stay the same width
    # and their stratum columns line up. Stealing from the heatmap alone would leave the
    # bar chart wider and the columns visibly offset between the panels.
    fig.colorbar(im, ax=axes.tolist(), label="within-stratum drop rate")

    mat = table["materialization"]
    sel = [mat.get(str(h), {}).get("selected", 0) for h in strata]
    drop = [mat.get(str(h), {}).get("dropped", 0) for h in strata]
    axes[1].bar(np.arange(len(strata)), drop, color="#d62728")
    axes[1].set_xticks(np.arange(len(strata)), [str(h) for h in strata])
    # Share the heatmap's column positions so the two panels read as one table.
    axes[1].set_xlim(-0.5, len(strata) - 0.5)
    axes[1].set_xlabel("stratum_id")
    axes[1].set_ylabel("windows")
    axes[1].set_ylim(0, max(max(drop), 1) * 1.5)
    axes[1].set_title(
        f"materialization drops per stratum (#50): {sum(drop)} of {sum(sel):,} selected "
        "— gated by E2.2c/E2.1b, not by E2.5",
        fontsize=9,
    )
    fig.savefig(path, dpi=130)  # no tight_layout: it would undo the colorbar's shared steal
    plt.close(fig)
    print(f"[wrote {path}]")


# --- Driver -------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", type=Path, default=REPO / "data" / "corpus_50k")
    p.add_argument("--catalog", type=Path, default=REPO / "data" / "catalog_gated")
    p.add_argument("--ledger", type=Path,
                   default=REPO / "artifacts" / "phase_e" / "materialization_ledger.parquet")
    p.add_argument("--selection", type=Path,
                   default=REPO / "artifacts" / "phase_d" / "selection_debug_50k.csv")
    p.add_argument("--allocation", type=Path, default=RESEARCH / "d2-allocation-nh.json")
    p.add_argument("--centroids", type=Path,
                   default=REPO / "artifacts" / "phase_d" / "centroids.json")
    p.add_argument("--scaler", type=Path, default=REPO / "artifacts" / "phase_d" / "scaler.json")
    p.add_argument("--bar", type=Path, default=RESEARCH / "e2-gate-bar.json")
    p.add_argument("--out-prefix", type=Path, default=RESEARCH / "e2")
    p.add_argument("--seed", type=int, default=52,
                   help="E2.1b permutation seed; recorded for byte-reproduction of the "
                        "printed number, not relied on for the pass/fail call")
    p.add_argument("--permutations", type=int, default=None,
                   help="override the bar's registered permutation count (smoke runs only)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bar = validate.read_bar(args.bar)
    th = bar.thresholds

    meta = json.loads((args.corpus / "corpus_meta.json").read_text())
    contract = bar.live_branch(meta["attrition_contract"])
    allocation = json.loads(args.allocation.read_text())
    n_h = {int(r["stratum_id"]): int(r["n_h"]) for r in allocation["rows"]}

    selection = load_selection(args.selection)
    manifest = load_manifest(args.corpus)
    ledger = load_ledger(args.ledger)
    cat = load_catalog(args.catalog)

    print(
        f"\n=== inputs ===\n"
        f"  bar           {args.bar.relative_to(REPO)} (contract: {contract})\n"
        f"  selection     {len(selection['window_id']):,} rows, md5 {selection['md5']}\n"
        f"  manifest      {len(manifest['window_id']):,} tiles\n"
        f"  ledger        {len(ledger['window_id']):,} rows\n"
        f"  catalog       {cat['keep'].size:,} rows, keep {int(cat['keep'].sum()):,}"
    )

    selected_rows = join_rows(cat["window_id"], selection["window_id"], label="selected")
    corpus_rows = join_rows(cat["window_id"], manifest["window_id"], label="materialized")

    outcomes: list[validate.Outcome] = []

    # E2.1a — did E1 write the metric values it was told to. The bar's tolerance is
    # "exact under the manifest's declared storage dtype", so the comparison follows the
    # manifest's own width rather than assuming today's float64: a future schema that
    # narrowed the frozen six would then report rounding as rounding, not as a Defect.
    outcomes.append(
        validate.metric_fidelity(
            {name: manifest[name] for name in catalog.FROZEN_METRICS},
            {name: cat[name][corpus_rows] for name in catalog.FROZEN_METRICS},
            storage_dtype=manifest[catalog.FROZEN_METRICS[0]].dtype,
        )
    )

    # E2.1b — were the drops metric-selective. Blind: only the drop COUNT feeds the null.
    status_by_id = dict(zip(ledger["window_id"], ledger["status"].tolist()))
    dropped = np.array([status_by_id[w] != "ok" for w in selection["window_id"]], dtype=bool)
    outcomes.append(
        validate.drop_selectivity(
            {name: cat[name][selected_rows] for name in catalog.FROZEN_METRICS},
            dropped,
            thresholds=th,
            permutations=args.permutations,
            seed=args.seed,
        )
    )

    # E2.2a — the list as this driver loaded it.
    committed = bar.committed_inputs
    outcomes.append(
        validate.selection_integrity(
            md5=selection["md5"],
            per_stratum=Counter(int(h) for h in selection["stratum_id"]),
            provenance=Counter(selection["provenance"]),
            expected_md5=committed["selection_md5"],
            expected_rows=int(committed["selection_rows"]),
            expected_per_stratum=n_h,
            expected_provenance=committed["provenance_counts"],
        )
    )

    realized = Counter(int(h) for h in manifest["stratum_id"])
    selected = Counter(int(h) for h in selection["stratum_id"])
    outcomes.append(validate.materialization_counts(realized, selected, contract=contract))
    outcomes.append(validate.thin_stratum(realized, selected, thresholds=th))

    # E2.2d — the floor as a rate. Gated on every coastline-tagged tile, because that is
    # the population #42 derived 1.337% from; the floor-provenance survivors are reported
    # beside it as the sub-diagnostic, not as the criterion.
    n_coastal = int(np.count_nonzero(manifest["coastline"]))
    outcomes.append(
        validate.coastal_floor(n_coastal, len(manifest["window_id"]), thresholds=th)
    )

    cov = coverage(manifest, cat)
    per_cog = cov.pop("_per_cog")
    outcomes.append(validate.per_cog_guard(per_cog, thresholds=th))

    # E2.5 — the per-(gate x stratum) table, off retroactive centroid assignment.
    scaler = frame.read_scaler(args.scaler)
    centroids_ = centroids.read_centroids(args.centroids)
    labels, assignable = would_be_strata(cat, centroids_, scaler)
    table, rule_inputs = attrition_table(cat, labels, assignable, ledger)
    outcomes.append(
        validate.gate_stratum_bias(
            fires=rule_inputs["fires"],
            membership=rule_inputs["membership"],
            corpus_fires=rule_inputs["corpus_fires"],
            corpus_rows=rule_inputs["corpus_rows"],
            thresholds=th,
            baseline=f"whole catalog ({rule_inputs['corpus_rows']:,} rows) — the bar's "
                     "'corpus-wide drop rate'",
        )
    )
    # The same rule against the narrower, arguably cleaner baseline. Published, never a
    # verdict — see attrition_table's docstring.
    sensitivity = validate.gate_stratum_bias(
        fires=rule_inputs["fires"],
        membership=rule_inputs["membership"],
        corpus_fires=rule_inputs["narrow_fires"],
        corpus_rows=rule_inputs["narrow_rows"],
        thresholds=th,
        baseline=f"assignable rows only ({rule_inputs['narrow_rows']:,}) — the population "
                 "the per-stratum rates use",
        criterion="E2.5_baseline_sensitivity",
        gates_verdict=False,
    )
    outcomes.append(sensitivity)

    # Published diagnostics. The reweighting is each tile's inclusion odds: a stratum
    # sampled at n_h out of N_h speaks for N_h/n_h pool windows, so weighting by that
    # should undo Neyman's deliberate over-sampling and land back on the pool.
    N_h = {int(r["stratum_id"]): int(r["N_h"]) for r in allocation["rows"]}
    weights = np.array(
        [N_h[int(h)] / n_h[int(h)] for h in manifest["stratum_id"]], dtype=np.float64
    )
    comparison = pool_comparison(
        {name: manifest[name] for name in catalog.FROZEN_METRICS},
        {name: cat[name][cat["keep"]] for name in catalog.FROZEN_METRICS},
        weights,
    )
    outcomes.append(
        validate.published(
            "E2.1c_pool_ecdf",
            {name: {k: v for k, v in entry.items() if not k.startswith(("grid", "ecdf"))}
             for name, entry in comparison["per_metric"].items()},
        )
    )
    outcomes.append(
        validate.published(
            "E2.3_coverage",
            {k: v for k, v in cov.items() if k not in ("grid", "pool_grid", "lon_edges", "lat_edges")},
        )
    )

    # --- report ---
    alloc_block = {
        "per_stratum": {
            str(h): {"selected": selected.get(h, 0), "realized": realized.get(h, 0)}
            for h in sorted(set(selected) | set(realized))
        },
        "absolute_floor": th.thin_absolute,
        "chi_square": chi_square(realized, n_h),
        "coastal": {
            "tiles_tagged_coastline": n_coastal,
            "selected_tagged_coastline": int(np.count_nonzero(cat["coastline"][selected_rows])),
            "floor_provenance_selected": int(
                sum(1 for p in selection["provenance"] if p == "floor:coastline")
            ),
            "floor_provenance_realized": int(
                sum(1 for p in manifest["provenance"].tolist() if p == "floor:coastline")
            ),
        },
    }

    # The bar's four-field handoff, one per fired criterion. A Finding that cannot name a
    # knob from the plan's table raises by design (the bar calls such a thing a Defect);
    # that is recorded rather than allowed to take the whole report down, because the
    # question it raises is for the Gate E review, not for this script.
    payloads: list[tuple[validate.Outcome, dict[str, Any]]] = []
    for outcome in outcomes:
        if outcome.fired is None:
            mark = "published"
        elif not outcome.gates:
            mark = "would-fire" if outcome.fired else "no-fire"  # a sensitivity, not a verdict
        else:
            mark = "FIRED" if outcome.fired else "pass"
        print(f"  {outcome.criterion:<32} {mark:<10} {outcome.measured_miss or ''}")
        if not (outcome.gates and outcome.fired):
            continue
        anchor = outcome.criterion.lower().replace(".", "").replace("_", "-")
        try:
            payload = outcome.payload(f"docs/research/e2-validation-report.md#{anchor}")
        except ValueError as exc:
            payload = {"payload_error": str(exc), "class": outcome.failure_class.capitalize()}
            print(f"    ! payload incomplete: {exc}")
        payloads.append((outcome, payload))

    figure_ecdfs(comparison, Path(f"{args.out_prefix}-metric-ecdfs.png"))
    figure_allocation(alloc_block, Path(f"{args.out_prefix}-allocation-fidelity.png"))
    figure_coverage(cov, per_cog, Path(f"{args.out_prefix}-coverage-map.png"))
    figure_attrition(table, Path(f"{args.out_prefix}-gate-attrition.png"))

    artifact = {
        "artifact": "E2 quantitative validation report",
        "ticket": "https://github.com/JStoschek/DEMAE/issues/52",
        "bar": str(args.bar.relative_to(REPO)),
        "verdict_owner": "https://github.com/JStoschek/DEMAE/issues/54",
        "inputs": {
            "corpus": str(args.corpus.relative_to(REPO)),
            "corpus_meta": meta,
            "selection_list": str(args.selection.relative_to(REPO)),
            "selection_md5": selection["md5"],
            "allocation_table": str(args.allocation.relative_to(REPO)),
            "gated_catalog": str(args.catalog.relative_to(REPO)),
            "attrition_contract": contract,
            "seed": args.seed,
        },
        "allocation": alloc_block,
        "coverage": cov,
        "attrition_by_stratum": table,
        "pool_comparison": comparison,
        **validate.summarize(outcomes),
        "payloads": {o.criterion: payload for o, payload in payloads},
    }
    out = Path(f"{args.out_prefix}-validation-report.json")
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"[wrote {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
