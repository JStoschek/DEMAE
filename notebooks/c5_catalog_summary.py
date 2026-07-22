"""C5 (#37) — the Phase-C summary notebook: distributions, attrition, coverage.

This is Phase C's **Definition of Done**. Everything before it produced a frame; this
reads the finished frame back and answers, from the frame alone, whether it is fit to
be the sampling frame Phase D draws from:

1. **Global metric distributions** — the frozen six over the whole catalog and over the
   kept sub-frame, at ADR-0004's transforms (``ln(x+1)`` on ``slope_mean`` and
   ``local_relief``). What matters is not the shapes themselves but the *difference*
   between the two: a gate set that reshaped terrain rather than removing non-terrain
   would show up here as a kept distribution that has lost a mode.
2. **Gate-attrition counts** — fires and sole cause per gate, recomputed from the
   persisted flag columns rather than copied from C4, so the shipped catalog is checked
   against the document that specified it instead of being assumed to match it.
3. **A coverage map** — kept windows per 10° cell, and the kept *rate* per cell, which
   is where a gate hollowing out a region would be visible as a hole rather than a
   thin spot.

It also runs the four invariants Phase D would otherwise discover the hard way: unique
window ids, no row lost or gained by gating, no kept window with a NaN metric (k-means
cannot take one), and no kept window with unread masks.

Reads ``data/catalog_gated/`` — the output of ``python -m dataset.gates``, which is
itself derived from ``data/catalog/`` in about three seconds. Neither directory is in
git (they are ~190 MB of Parquet); rebuild both with

    uv run python -m dataset.catalog      # ~6 h, the global scoring pass (#35)
    uv run python -m dataset.gates        # seconds, the gates (#37)

Zero-network, numpy + pyarrow + matplotlib only (this repo's env has no pandas).

Run:   uv run python notebooks/c5_catalog_summary.py
Writes: docs/research/c5-catalog-summary.json  — every number quoted in the doc
        docs/research/c5-metric-distributions.png
        docs/research/c5-coverage-map.png
        docs/research/c5-gate-attrition.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from dataset import catalog, gates

REPO = Path(__file__).resolve().parents[1]
GATED_DIR = REPO / "data" / "catalog_gated"
RESEARCH = REPO / "docs" / "research"

#: ADR-0004 §1's transforms — the space Phase D's k-means will actually run in.
LOG_METRICS = {"slope_mean", "local_relief"}
#: The quantiles reported per metric. The tails are what a stratifier's edge bins get.
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
#: Coverage-map resolution: the 10° geo-cell the catalog already carries as a column.
CELL_DEG = 10


def load(gated_dir: Path = GATED_DIR) -> dict[str, Any]:
    """The columns this notebook needs over the whole gated frame — numpy per column.

    Column-selective: the frame is 22 raw + 8 flag columns and this reads 20 of them,
    which is most of a gigabyte of Parquet not decoded. ``window_id`` is the one column
    left as a pyarrow array rather than numpy (hence the ``Any`` value type) — it is
    only ever counted for uniqueness, and materialising 4.4 M Python strings to do that
    is pure waste.
    """
    if not (gated_dir / catalog.PARTS_DIR).exists():
        raise SystemExit(
            f"no gated catalog at {gated_dir}. Build it with:\n"
            "  uv run python -m dataset.catalog   # the global pass (hours)\n"
            "  uv run python -m dataset.gates     # the gates (seconds)"
        )
    wanted = [
        "window_id", "stem", "lon", "lat", "band", "geocell", "land_std",
        *catalog.FROZEN_METRICS, *gates.FLAG_COLUMNS,
    ]
    table = pq.read_table(gated_dir / catalog.PARTS_DIR, columns=wanted)
    out: dict[str, Any] = {"window_id": table.column("window_id")}
    for name in wanted[1:]:
        out[name] = table.column(name).to_numpy(zero_copy_only=False)
    return out


# --- 1. Global metric distributions -------------------------------------------


def _transform(name: str, values: np.ndarray) -> np.ndarray:
    return np.log1p(values) if name in LOG_METRICS else values


def metric_summary(frame: dict[str, np.ndarray]) -> dict[str, dict]:
    """Per-metric quantiles and moments, over the whole frame and over ``keep``.

    NaNs are excluded rather than filled: a NaN metric means the window had no land to
    measure, and imputing a value would put non-terrain into a terrain distribution.
    Their count is reported per metric so the exclusion is visible, not silent.
    """
    keep = frame["keep"]
    out: dict[str, dict] = {}
    for name in catalog.FROZEN_METRICS:
        values = frame[name]
        entry: dict[str, object] = {"transform": "ln(x+1)" if name in LOG_METRICS else "raw"}
        for label, subset in (("all", values), ("kept", values[keep])):
            finite = subset[np.isfinite(subset)]
            entry[label] = {
                "n": int(finite.size),
                "n_nan": int(subset.size - finite.size),
                "mean": float(finite.mean()) if finite.size else float("nan"),
                "std": float(finite.std()) if finite.size else float("nan"),
                "quantiles": {
                    str(q): float(v)
                    for q, v in zip(QUANTILES, np.quantile(finite, QUANTILES))
                }
                if finite.size
                else {},
            }
        out[name] = entry
    return out


def figure_distributions(frame: dict[str, np.ndarray], path: Path) -> None:
    """2×3 of the frozen six: whole frame behind, kept in front, both area-normalised.

    Area-normalised (``density=True``) rather than raw counts, because the kept frame is
    30% smaller by construction and a count plot would show that gap in every panel and
    nothing else. What the reader is looking for is a *shape* change.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = frame["keep"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for ax, name in zip(axes.ravel(), catalog.FROZEN_METRICS):
        allv = _transform(name, frame[name])
        allv = allv[np.isfinite(allv)]
        kept = _transform(name, frame[name][keep])
        kept = kept[np.isfinite(kept)]
        lo, hi = np.quantile(allv, [0.001, 0.999])
        bins = np.linspace(lo, hi, 80)
        ax.hist(allv, bins=bins, density=True, color="0.75", label=f"all ({allv.size:,})")
        ax.hist(
            kept, bins=bins, density=True, histtype="step", lw=1.6, color="#1f77b4",
            label=f"kept ({kept.size:,})",
        )
        unit = " [ln(x+1)]" if name in LOG_METRICS else ""
        ax.set_title(f"{name}{unit}", fontsize=10)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)
    fig.suptitle(
        "C5 (#37): the frozen six over the candidate catalog — "
        "gating removes non-terrain, it does not reshape terrain",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[wrote {path}]")


# --- 2. Gate attrition --------------------------------------------------------


def figure_attrition(report: gates.AttritionReport, path: Path) -> None:
    """Fires against sole cause, per gate — the gap *is* C4's finding.

    A gate whose bar pair is wide apart is mostly covered by another gate; ``zero_fill``
    has no sole-cause bar at all.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [g.gate for g in report.gates]
    fires = [g.fires for g in report.gates]
    sole = [g.sole_cause for g in report.gates]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.barh(y + 0.19, fires, height=0.36, color="0.78", label="fires")
    ax.barh(y - 0.19, sole, height=0.36, color="#d62728", label="sole cause")
    for i, (f, s) in enumerate(zip(fires, sole)):
        ax.text(f, i + 0.19, f"  {f:,}", va="center", fontsize=8)
        ax.text(s, i - 0.19, f"  {s:,}", va="center", fontsize=8, color="#d62728")
    ax.set_yticks(y, names)
    ax.set_xlabel("windows")
    ax.set_xlim(0, max(fires) * 1.28)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(
        f"C5 (#37): gate attrition over {report.n_rows:,} windows — "
        f"kept {report.n_kept:,} ({report.keep_rate:.2%})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[wrote {path}]")


# --- 3. Coverage -------------------------------------------------------------


def coverage_grid(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Kept and total window counts per ``CELL_DEG`` cell, on a global lon/lat grid.

    Histogrammed from the stored window centroids rather than grouped by the
    ``geocell`` string, so the map is a picture of where the windows physically are and
    a mis-parsed cell key cannot move one.
    """
    lon_edges = np.arange(-180, 181, CELL_DEG)
    lat_edges = np.arange(-90, 91, CELL_DEG)
    total, _, _ = np.histogram2d(frame["lon"], frame["lat"], bins=[lon_edges, lat_edges])
    kept, _, _ = np.histogram2d(
        frame["lon"][frame["keep"]], frame["lat"][frame["keep"]], bins=[lon_edges, lat_edges]
    )
    return {"lon_edges": lon_edges, "lat_edges": lat_edges, "total": total, "kept": kept}


def figure_coverage(grid: dict[str, np.ndarray], path: Path) -> None:
    """Two panels: how many kept windows a cell holds, and what share of it survived.

    The rate panel is the one risk #3 is read off. A cell that is merely small shows as
    dark on the left and unremarkable on the right; a cell a gate has hollowed out shows
    as dark on *both*.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total, kept = grid["total"], grid["kept"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(total > 0, kept / total, np.nan)
    extent = [-180, 180, -90, 90]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.4))
    counts = np.where(kept > 0, kept, np.nan)
    im0 = axes[0].imshow(
        counts.T, origin="lower", extent=extent, aspect="auto", cmap="viridis",
        norm=matplotlib.colors.LogNorm(vmin=max(1.0, np.nanmin(counts)), vmax=np.nanmax(counts)),
    )
    axes[0].set_title(
        f"kept windows per {CELL_DEG}° cell (log scale) — "
        f"{int((kept > 0).sum())} of {int((total > 0).sum())} occupied cells keep something",
        fontsize=10,
    )
    fig.colorbar(im0, ax=axes[0], label="kept windows")

    im1 = axes[1].imshow(
        rate.T, origin="lower", extent=extent, aspect="auto", cmap="RdYlBu", vmin=0, vmax=1
    )
    hollow = (total > 0) & (kept == 0)
    if hollow.any():
        li, la = np.nonzero(hollow)
        axes[1].scatter(
            grid["lon_edges"][li] + CELL_DEG / 2, grid["lat_edges"][la] + CELL_DEG / 2,
            s=14, marker="x", color="k", label=f"keeps nothing ({int(hollow.sum())} cells)",
        )
        axes[1].legend(fontsize=8, loc="lower left")
    axes[1].set_title("keep rate per cell — risk #3 is read off this panel", fontsize=10)
    fig.colorbar(im1, ax=axes[1], label="kept / scored")

    for ax in axes:
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.tick_params(labelsize=8)
    fig.suptitle("C5 (#37): candidate-catalog coverage after the locked gates", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[wrote {path}]")


def coverage_summary(frame: dict[str, np.ndarray], grid: dict[str, np.ndarray]) -> dict:
    """Cell/COG/band coverage counts — the numbers the doc's coverage section quotes."""
    keep = frame["keep"]
    cells = frame["geocell"]
    kept_cells = set(np.unique(cells[keep]).tolist())
    all_cells = set(np.unique(cells).tolist())
    stems = frame["stem"]
    n_cogs = int(np.unique(stems).size)
    kept_stems, kept_per_cog = np.unique(stems[keep], return_counts=True)
    n_cogs_keeping = int(kept_stems.size)

    per_band = {}
    for band in np.unique(frame["band"]).tolist():
        sel = frame["band"] == band
        per_band[band] = {
            "n_windows": int(sel.sum()),
            "n_kept": int((sel & keep).sum()),
            "keep_rate": float((sel & keep).sum() / sel.sum()),
        }
    return {
        "cell_deg": CELL_DEG,
        "n_cells_occupied": len(all_cells),
        "n_cells_keeping": len(kept_cells),
        "hollow_cells": sorted(all_cells - kept_cells),
        "n_cells_hollow": len(all_cells - kept_cells),
        "n_cogs": n_cogs,
        "n_cogs_keeping": n_cogs_keeping,
        "median_kept_windows_per_cog": float(np.median(kept_per_cog)),
        "grid_cells_occupied": int((grid["total"] > 0).sum()),
        "grid_cells_keeping": int((grid["kept"] > 0).sum()),
        "per_band": per_band,
    }


# --- 4. Invariants ------------------------------------------------------------


def invariants(frame: dict[str, np.ndarray], meta: dict) -> dict:
    """The four checks Phase D would otherwise discover the hard way.

    Reported rather than asserted: the notebook's job is to *show* the frame's state,
    and a failing invariant is a finding to write down, not a traceback that hides the
    other three. The one that would bite hardest is ``n_kept_with_nan_metric`` — a NaN
    row in the k-means input.

    ``n_nan_metric_water_only`` is why that one holds, and it is worth having as a
    number rather than a claim. A NaN metric means the window had no land to measure, so
    what keeps NaNs out of the frame is whichever gate catches those windows. If the
    answer were "the water cap", the invariant would rest on WBM — the mask C2 caught
    calling a 3,656 m salt flat water — and would break the day that cap moved. This
    counts the windows *only* the water cap catches.
    """
    keep = frame["keep"]
    nan_metric = np.zeros(keep.size, dtype=bool)
    for name in catalog.FROZEN_METRICS:
        nan_metric |= ~np.isfinite(frame[name])
    no_land = ~np.isfinite(frame["land_std"])
    not_water = ~(frame["zero_fill"] | frame["degenerate"] | frame["edited"])
    return {
        "n_rows": int(keep.size),
        "n_unique_window_ids": len(frame["window_id"].unique()),
        "rows_match_raw_catalog": int(keep.size) == int(meta.get("n_rows", -1)),
        "n_kept_with_nan_metric": int((keep & nan_metric).sum()),
        "n_kept_with_masks_missing": int((keep & frame["masks_missing"]).sum()),
        "n_masks_missing": int(frame["masks_missing"].sum()),
        "n_nan_metric": int(nan_metric.sum()),
        "n_nan_metric_no_land": int((nan_metric & no_land).sum()),
        "n_nan_metric_water_only": int((nan_metric & not_water).sum()),
    }


# --- Driver -------------------------------------------------------------------


def main() -> int:
    frame = load()
    meta = json.loads((GATED_DIR / catalog.META_FILE).read_text())
    report = gates.attrition({name: frame[name] for name in gates.FLAG_COLUMNS})

    metrics_ = metric_summary(frame)
    grid = coverage_grid(frame)
    coverage = coverage_summary(frame, grid)
    checks = invariants(frame, meta)

    print(
        f"\n=== catalog ===\n"
        f"  {report.n_rows:,} windows; kept {report.n_kept:,} ({report.keep_rate:.2%})\n"
        f"  water_affected (kept) {report.n_water_affected:,}; "
        f"coastline (kept) {report.n_coastline:,}"
    )
    print("\n=== gate attrition ===")
    for g in report.gates:
        print(f"  {g.gate:<11} fires {g.fires:>9,}   sole cause {g.sole_cause:>9,}")
    print("\n=== metric distributions (kept frame) ===")
    for name, entry in metrics_.items():
        q = entry["kept"]["quantiles"]
        print(
            f"  {name:<24} median {q['0.5']:>10.3f}   "
            f"p01 {q['0.01']:>9.3f}   p99 {q['0.99']:>10.3f}"
        )
    print("\n=== coverage ===")
    print(
        f"  {coverage['n_cells_keeping']} of {coverage['n_cells_occupied']} "
        f"{CELL_DEG}° cells keep at least one window "
        f"({coverage['n_cells_hollow']} hollow)"
    )
    for band, entry in sorted(coverage["per_band"].items()):
        print(
            f"  band {band:<7} {entry['n_windows']:>9,} windows  "
            f"keep {entry['keep_rate']:.2%}"
        )
    print("\n=== invariants ===")
    for name, value in checks.items():
        print(f"  {name:<28} {value}")

    figure_distributions(frame, RESEARCH / "c5-metric-distributions.png")
    figure_attrition(report, RESEARCH / "c5-gate-attrition.png")
    figure_coverage(grid, RESEARCH / "c5-coverage-map.png")

    artifact = {
        "source": str(GATED_DIR.relative_to(REPO)),
        "catalog_meta": meta,
        "attrition": report.as_dict(),
        "metrics": metrics_,
        "coverage": coverage,
        "invariants": checks,
    }
    out = RESEARCH / "c5-catalog-summary.json"
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"[wrote {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
