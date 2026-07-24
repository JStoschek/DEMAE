"""E58 — characterize the stratum-19 hollowing: good alpine terrain, or corrupted tiles?

Ticket: https://github.com/JStoschek/DEMAE/issues/58 · Map: #57 (edit/fill gate).

One of the two investigations feeding the loosen-vs-accept decision. Reuses the
committed Phase-D centroids/scaler exactly as E2.5 did (dropped windows survive as
columns per #37, so they can be scored against the frozen centres), then asks three
things of the windows the ``edited`` gate removes from stratum 19:

1. **Is it edit or fill?** EDM (``edm_edited_frac``) vs FLM (``flm_fill_frac``)
   decomposed separately, because the loosen options differ (a per-mask rule only
   helps if one mask carries the firing).
2. **A cliff or a long tail?** The ``edited_max = 0.25`` cap seen against the actual
   distribution of the drops. Drops piled just over 0.25 mean the cap slices a
   continuum; a fat tail to 1.0 means genuinely mostly-synthetic tiles.
3. **Good terrain or wrecked?** The dropped surfaces rendered from the source COGs
   (``gallery.shade_tile``) beside the kept ones, plus a metric comparison, so the
   loss can be judged by eye rather than asserted.

AFK. Writes ``docs/research/e58-*`` — a JSON payload, a markdown summary, a histogram,
and three gallery sheets with a reproducible tile manifest. No pipeline state is touched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from dataset import catalog, centroids, config, frame, gallery, geometry, tiling

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "docs" / "research"
STRATUM = 19
EDITED_MAX = 0.25
DRAW_SEED = gallery.GALLERY_SEED  # reuse the registered gallery seed for keyed draws


# --- Loading + stratum assignment (mirror of E2.5's would_be_strata) -----------


def col(cat: Any, name: str) -> np.ndarray:
    return cat[name].to_numpy(zero_copy_only=False)


def would_be_strata(cat: Any, cn: centroids.Centroids, sc: frame.Scaler
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Every window's would-be stratum, dropped ones included. Copy of E2.5's routine."""
    X = frame.metric_matrix(cat)
    assignable = np.isfinite(X).all(axis=1)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels[assignable] = centroids.assign(sc.standardize(X[assignable]), cn)
    return labels, assignable


# --- Quantitative characterization ---------------------------------------------


def quantiles(x: np.ndarray) -> dict[str, float]:
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    return {f"p{int(q * 100):02d}": round(float(v), 4)
            for q, v in zip(qs, np.quantile(x, qs))} if x.size else {}


def metric_summary(cat: Any, mask: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for m in ("mean_elevation", "local_relief", "slope_mean", "texture",
              "elevation_relief_ratio", "geomorphon_entropy"):
        v = col(cat, m)[mask]
        v = v[np.isfinite(v)]
        out[m] = {"mean": round(float(v.mean()), 3), "median": round(float(np.median(v)), 3),
                  "p10": round(float(np.quantile(v, 0.1)), 3),
                  "p90": round(float(np.quantile(v, 0.9)), 3)} if v.size else {}
    return out


def characterize(cat: Any, labels: np.ndarray) -> dict[str, Any]:
    s19 = labels == STRATUM
    edited = col(cat, "edited").astype(bool)
    keep = col(cat, "keep").astype(bool)
    edm = col(cat, "edm_edited_frac")
    flm = col(cat, "flm_fill_frac")

    drop = s19 & edited                      # the windows under investigation
    kept = s19 & keep
    other_gates = np.zeros(cat.num_rows, dtype=bool)
    for g in ("zero_fill", "degenerate", "water", "masks_missing"):
        other_gates |= col(cat, g).astype(bool)

    edm_d, flm_d = edm[drop], flm[drop]
    over_edm = edm_d > EDITED_MAX
    over_flm = flm_d > EDITED_MAX

    bands = [(EDITED_MAX, 0.35), (0.35, 0.5), (0.5, 0.75), (0.75, 1.0001)]
    edm_band = {f"{lo:.2f}-{min(hi,1.0):.2f}": int(((edm_d > lo) & (edm_d <= hi)).sum())
                for lo, hi in bands}

    kept_edm = edm[kept]
    kept_edm = kept_edm[np.isfinite(kept_edm)]

    return {
        "stratum": STRATUM,
        "membership": int(s19.sum()),
        "kept": int(kept.sum()),
        "edited_drops": int(drop.sum()),
        "within_stratum_drop_rate": round(float(drop.sum() / s19.sum()), 4),
        "corpus_edited_rate": round(float(edited.sum() / cat.num_rows), 4),
        "ratio_vs_corpus": round(float((drop.sum() / s19.sum())
                                       / (edited.sum() / cat.num_rows)), 2),
        "allocation_n_h": 1106,          # d2-allocation-nh.json, stratum 19
        "kept_pool_multiple_of_allocation": round(float(kept.sum() / 1106), 1),
        "edit_vs_fill": {
            "sole_cause_edited": int((drop & ~other_gates).sum()),
            "edm_over_cap": int(over_edm.sum()),
            "flm_over_cap": int(over_flm.sum()),
            "both_over_cap": int((over_edm & over_flm).sum()),
            "edm_only_over_cap": int((over_edm & ~over_flm).sum()),
            "flm_only_over_cap": int((over_flm & ~over_edm).sum()),
        },
        "edm_frac_of_drops": {"quantiles": quantiles(edm_d), "bands_over_cap": edm_band},
        "flm_frac_of_drops": {"quantiles": quantiles(flm_d)},
        "kept_edm_frac": {
            "quantiles": quantiles(kept_edm),
            "share_in_top_quintile_of_cap": round(
                float((kept_edm > 0.20).sum() / kept_edm.size), 4) if kept_edm.size else 0.0,
        },
        "terrain_metrics": {"dropped": metric_summary(cat, drop),
                            "kept": metric_summary(cat, kept)},
        "geography": geography(cat, drop),
    }


def geography(cat: Any, mask: np.ndarray) -> dict[str, Any]:
    stems = col(cat, "stem")[mask]
    geocells = col(cat, "geocell")[mask]
    lat, lon = col(cat, "lat")[mask], col(cat, "lon")[mask]
    top = Counter(geocells.tolist()).most_common(12)
    return {
        "n_windows": int(mask.sum()),
        "n_distinct_cogs": int(len(set(stems.tolist()))),
        "n_distinct_geocells": int(len(set(geocells.tolist()))),
        "lat_range": [round(float(lat.min()), 2), round(float(lat.max()), 2)],
        "lon_range": [round(float(lon.min()), 2), round(float(lon.max()), 2)],
        "top_geocells": [{"geocell": g, "n": n} for g, n in top],
    }


# --- Figures -------------------------------------------------------------------


def histogram(cat: Any, labels: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s19 = labels == STRATUM
    edm = col(cat, "edm_edited_frac")
    keep = col(cat, "keep").astype(bool)
    edited = col(cat, "edited").astype(bool)
    kept_edm = edm[s19 & keep]
    kept_edm = kept_edm[np.isfinite(kept_edm)]
    drop_edm = edm[s19 & edited]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 1, 41)
    ax.hist(kept_edm, bins=bins, color="#4c78a8", alpha=0.85, label=f"kept (n={kept_edm.size:,})")
    ax.hist(drop_edm, bins=bins, color="#e45756", alpha=0.85,
            label=f"dropped by edited (n={drop_edm.size:,})")
    ax.axvline(EDITED_MAX, color="#333", lw=1.4, ls="--")
    ax.text(EDITED_MAX + 0.006, ax.get_ylim()[1] * 0.92, "edited_max = 0.25",
            fontsize=8, color="#333")
    ax.set_xlabel("EDM edited-pixel fraction (edm_edited_frac)")
    ax.set_ylabel("stratum-19 windows")
    ax.set_yscale("log")
    ax.set_title("Stratum 19 — EDM editing fraction, kept vs dropped (log count)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


# --- Gallery from source COGs --------------------------------------------------


def read_tiles(cat: Any, rows: list[int]) -> dict[int, np.ndarray]:
    """Read the 224x224 source window for each catalog row, batched per COG."""
    reader = tiling.RasterioCogReader()
    stems = col(cat, "stem")
    r0s, c0s = col(cat, "win_row0"), col(cat, "win_col0")
    by_stem: dict[str, list[int]] = {}
    for i in rows:
        by_stem.setdefault(str(stems[i]), []).append(i)
    out: dict[int, np.ndarray] = {}
    for stem, idxs in by_stem.items():
        ref = geometry.parse_cog_key(stem)
        origins = [(int(r0s[i]), int(c0s[i])) for i in idxs]
        for i, arr in zip(idxs, reader.read_windows(ref, origins, config.TILE_PX)):
            out[i] = arr
    return out


def caption(cat: Any, i: int) -> str:
    g = lambda m: col(cat, m)[i]  # noqa: E731
    return (f"elev {g('mean_elevation'):>5.0f} m  relief {g('local_relief'):>5.0f} m  "
            f"slope {g('slope_mean'):>4.1f}\N{DEGREE SIGN}\n"
            f"edm {g('edm_edited_frac') * 100:>3.0f}%  flm {g('flm_fill_frac') * 100:>3.0f}%  "
            f"{g('lat'):+06.2f},{g('lon'):+07.2f}")


def render_sheet(cat: Any, rows: list[int], tiles: dict[int, np.ndarray],
                 *, title: str, subtitle: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = 4
    nrow = max(1, -(-len(rows) // ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 2.95), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, i in zip(axes.ravel(), rows):
        rgb, _ve, _sp = gallery.shade_tile(tiles[i], lat=float(col(cat, "lat")[i]))
        ax.imshow(rgb, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_axis_on()
        for sp in ax.spines.values():
            sp.set_color("#999"); sp.set_linewidth(0.6)
        ax.set_xlabel(caption(cat, i), fontsize=6.5, family="monospace", labelpad=3,
                      linespacing=1.4, color="#222")
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.text(0.5, 0.965, subtitle, ha="center", fontsize=8, color="#555", family="monospace")
    fig.subplots_adjust(top=0.94, bottom=0.03, hspace=0.42, wspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  [{path.stat().st_size // 1024:>4} KB] {path.relative_to(REPO)}")


def pick_uniform(cat: Any, pool: np.ndarray, n: int) -> list[int]:
    ids = col(cat, "window_id")[pool]
    return [int(pool[p]) for p in gallery.uniform_draw(ids, n=n, seed=DRAW_SEED)]


def pick_spanning_edm(cat: Any, pool: np.ndarray, n: int) -> list[int]:
    """One tile from each of n equal-count edm-fraction bins: just-over-cap -> max."""
    edm = col(cat, "edm_edited_frac")[pool]
    order = pool[np.argsort(edm, kind="stable")]
    ids = col(cat, "window_id")
    picks = []
    for b in np.array_split(order, min(n, len(order))):
        j = min(b.tolist(), key=lambda i: gallery._shuffle_key(str(ids[i]), DRAW_SEED))
        picks.append(int(j))
    return picks


# --- Driver --------------------------------------------------------------------


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--catalog", type=Path, default=REPO / "data" / "catalog_gated")
    p.add_argument("--centroids", type=Path,
                   default=REPO / "artifacts" / "phase_d" / "centroids.json")
    p.add_argument("--scaler", type=Path, default=REPO / "artifacts" / "phase_d" / "scaler.json")
    p.add_argument("--out-prefix", type=Path, default=RESEARCH / "e58-stratum19-hollowing")
    p.add_argument("--gallery-dir", type=Path, default=RESEARCH / "e58-hollowing")
    p.add_argument("--per-sheet", type=int, default=12)
    p.add_argument("--no-gallery", action="store_true",
                   help="skip the S3 COG reads (quantitative half only)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cat = catalog.read_catalog(args.catalog)
    sc = frame.read_scaler(args.scaler)
    cn = centroids.read_centroids(args.centroids)
    labels, _assignable = would_be_strata(cat, cn, sc)

    print(f"=== catalog {cat.num_rows:,} rows; assigning would-be strata ===")
    payload = characterize(cat, labels)
    payload["provenance"] = {"code_version": git_head(),
                             "catalog": str(args.catalog.relative_to(REPO)),
                             "centroids": str(args.centroids.relative_to(REPO)),
                             "draw_seed": DRAW_SEED}
    ev = payload["edit_vs_fill"]
    print(f"  membership {payload['membership']:,}  kept {payload['kept']:,}  "
          f"edited-drops {payload['edited_drops']:,} "
          f"({payload['within_stratum_drop_rate'] * 100:.1f}%, "
          f"{payload['ratio_vs_corpus']}x corpus)")
    print(f"  edm-over-cap {ev['edm_over_cap']:,}  both {ev['both_over_cap']:,}  "
          f"edm-only {ev['edm_only_over_cap']:,}  flm-only {ev['flm_only_over_cap']:,}")

    histogram(cat, labels, args.out_prefix.with_name(args.out_prefix.name + "-edm.png"))

    manifest: dict[str, Any] = {"seed": DRAW_SEED, "sheets": {}}
    if not args.no_gallery:
        s19 = labels == STRATUM
        edited = col(cat, "edited").astype(bool)
        keep = col(cat, "keep").astype(bool)
        drop_pool = np.flatnonzero(s19 & edited)
        kept_pool = np.flatnonzero(s19 & keep)
        n = args.per_sheet

        sheets = {
            "dropped-uniform": (
                pick_uniform(cat, drop_pool, n),
                "Dropped by `edited` — keyed-random sample",
                f"{drop_pool.size:,} stratum-19 windows removed by the edit/fill gate"),
            "dropped-by-edm": (
                pick_spanning_edm(cat, drop_pool, n),
                "Dropped by `edited` — spanning EDM fraction (just-over-cap \N{RIGHTWARDS ARROW} max)",
                "one tile per equal-count edm_edited_frac bin"),
            "kept-uniform": (
                pick_uniform(cat, kept_pool, n),
                "Kept in stratum 19 — keyed-random sample (comparison)",
                f"{kept_pool.size:,} windows that cleared the gate"),
        }
        all_rows = sorted({i for rows, _, _ in sheets.values() for i in rows})
        print(f"=== reading {len(all_rows)} source windows from COGs ===")
        tiles = read_tiles(cat, all_rows)
        wid = col(cat, "window_id")
        for name, (rows, title, subtitle) in sheets.items():
            render_sheet(cat, rows, tiles, title=title, subtitle=subtitle,
                         path=args.gallery_dir / f"{name}.png")
            manifest["sheets"][name] = [
                {"window_id": str(wid[i]), "stem": str(col(cat, "stem")[i]),
                 "win_row0": int(col(cat, "win_row0")[i]), "win_col0": int(col(cat, "win_col0")[i]),
                 "edm": round(float(col(cat, "edm_edited_frac")[i]), 3),
                 "flm": round(float(col(cat, "flm_fill_frac")[i]), 3),
                 "mean_elevation": round(float(col(cat, "mean_elevation")[i]), 1)}
                for i in rows]
        (args.gallery_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  wrote {(args.gallery_dir / 'manifest.json').relative_to(REPO)}")

    payload["gallery"] = manifest if not args.no_gallery else None
    out_json = args.out_prefix.with_suffix(".json")
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"=== wrote {out_json.relative_to(REPO)} ===")
    write_markdown(args, payload)
    return 0


def write_markdown(args: argparse.Namespace, p: dict[str, Any]) -> None:
    ev, edm = p["edit_vs_fill"], p["edm_frac_of_drops"]
    tm_d, tm_k = p["terrain_metrics"]["dropped"], p["terrain_metrics"]["kept"]
    geo = p["geography"]
    over = edm["bands_over_cap"]
    total_drop = p["edited_drops"]
    near_cap = over[f"{EDITED_MAX:.2f}-0.35"]

    def row(m: str) -> str:
        d, k = tm_d[m], tm_k[m]
        return (f"| {m} | {d['median']} ({d['p10']}\N{EN DASH}{d['p90']}) "
                f"| {k['median']} ({k['p10']}\N{EN DASH}{k['p90']}) |")

    md = f"""# E58 — Stratum-19 hollowing: good alpine terrain, or corrupted tiles?

**Ticket:** [Characterize the stratum-19 hollowing (#58)](https://github.com/JStoschek/DEMAE/issues/58) ·
**Map:** [edit/fill gate (#57)](https://github.com/JStoschek/DEMAE/issues/57) ·
**Seeded by:** [Gate E FAIL (#54)](https://github.com/JStoschek/DEMAE/issues/54) ·
`{p['provenance']['code_version']}`

One of two investigations feeding the loosen-vs-accept decision ([#60](https://github.com/JStoschek/DEMAE/issues/60)).
Method mirrors E2.5 exactly: dropped windows survive as columns ([#37](https://github.com/JStoschek/DEMAE/issues/37)),
so every stratum-19 candidate — kept or dropped — is scored against the committed Phase-D
centroids (`{p['provenance']['centroids']}`) and the `edited`-gate drops are pulled out.

## Verdict in one line

The `edited` gate removes **{total_drop:,}** of stratum 19's **{p['membership']:,}** would-be members
(**{p['within_stratum_drop_rate'] * 100:.1f}%**, **{p['ratio_vs_corpus']}×** the corpus `edited` rate) — and by
the metrics and — across the bulk of the distribution — by eye these are **the same high-alpine terrain
family as the kept tiles** (§3 carries the one honest caveat, at the extreme high-EDM tail). The loss is
real terrain, but it does **not** starve selection: stratum 19 still keeps
**{p['kept']:,}** candidates against an allocation of **{p['allocation_n_h']:,}** tiles
(**{p['kept_pool_multiple_of_allocation']}×** the pool it needs). The open risk is *representativeness* —
whether the {p['within_stratum_drop_rate'] * 100:.0f}% that is removed skews which alpine terrain the 1,106 sample from —
not availability. That risk is what [#60](https://github.com/JStoschek/DEMAE/issues/60) must weigh.

## 1. Edit and fill are co-located — the knob is the combined screen, not one mask

All **{ev['sole_cause_edited']:,}** drops are **sole-cause** `edited` — none of the other four gates
(`zero_fill`/`degenerate`/`water`/`masks_missing`) co-fires, so the whole loss is attributable to this one
screen. But *within* the screen, edit and fill fire almost together:

| over the 0.25 cap | windows | share of drops |
|---|---|---|
| EDM (`edm_edited_frac` > 0.25) | **{ev['edm_over_cap']:,}** | {ev['edm_over_cap'] / total_drop * 100:.0f}% |
| FLM (`flm_fill_frac` > 0.25)  | {ev['flm_over_cap']:,} | {ev['flm_over_cap'] / total_drop * 100:.0f}% |
| **both** limbs over cap | **{ev['both_over_cap']:,}** | {ev['both_over_cap'] / total_drop * 100:.0f}% |
| EDM-only (edit fires, fill does not) | {ev['edm_only_over_cap']:,} | {ev['edm_only_over_cap'] / total_drop * 100:.0f}% |
| FLM-only (fill fires, edit does not) | {ev['flm_only_over_cap']:,} | {ev['flm_only_over_cap'] / total_drop * 100:.0f}% |

The gate is `(edm > 0.25) OR (flm > 0.25)`, so to *recover* a window both limbs must fall under their caps.
Two consequences for the loosening knob:

- **Touching fill alone recovers nothing** — FLM-only is **{ev['flm_only_over_cap']}**; fill is never the sole
  cause, exactly as C4 noted, so a fill-lenient rule is the wrong knob.
- **Touching edit alone recovers only {ev['edm_only_over_cap']:,}** ({ev['edm_only_over_cap'] / total_drop * 100:.0f}%) — because
  **{ev['both_over_cap']:,}** of the drops ({ev['both_over_cap'] / total_drop * 100:.0f}%) are over *both* caps and the FLM limb would
  still catch them. In this terrain, editing and fill land on the **same pixels** (glacier / permanent-snow /
  radar-shadow voids get interpolated *and* flagged edited), so they are not independent knobs.

The real knob is therefore the **combined edit-or-fill screen** — a single loosened cap applied to both limbs
(global or per-stratum), not a per-mask split. A per-mask rule, in either direction, is the wrong instrument here.

## 2. A continuum sliced by the cap, not a wall of synthetic tiles

EDM editing-fraction of the dropped windows (quantiles): p10 **{edm['quantiles'].get('p10')}**,
p50 **{edm['quantiles'].get('p50')}**, p90 **{edm['quantiles'].get('p90')}**, max **{edm['quantiles'].get('p100')}**.

| EDM band | dropped windows |
|---|---|
| 0.25–0.35 (just over the cap) | **{over[f'{EDITED_MAX:.2f}-0.35']:,}** |
| 0.35–0.50 | {over['0.35-0.50']:,} |
| 0.50–0.75 | {over['0.50-0.75']:,} |
| 0.75–1.00 | {over['0.75-1.00']:,} |

**{near_cap:,}** of the {total_drop:,} drops ({near_cap / total_drop * 100:.0f}%) sit in the first band just above 0.25.
The kept tiles press right up against the cap from below (kept EDM p90 **{p['kept_edm_frac']['quantiles'].get('p90')}**,
p99 **{p['kept_edm_frac']['quantiles'].get('p99')}**), so `edited_max = 0.25` is cutting through a *continuous*
editing gradient in this terrain family — not separating a clean population from a synthetic one. See
[`e58-stratum19-hollowing-edm.png`](e58-stratum19-hollowing-edm.png). This is the shape that makes a *modest*
per-stratum loosening (e.g. lifting the cap toward ~0.35–0.50 for high-alpine strata) plausible without
scooping up near-fully-synthetic tiles — though whether even a heavily-edited pixel carries trustworthy
elevation is [#59](https://github.com/JStoschek/DEMAE/issues/59)'s question, and this ticket does not pre-judge it.

## 3. The dropped terrain is the same family as the kept terrain

Median (p10–p90) over the six clustering metrics, dropped vs kept:

| metric | dropped | kept |
|---|---|---|
{row('mean_elevation')}
{row('local_relief')}
{row('slope_mean')}
{row('texture')}
{row('geomorphon_entropy')}

The dropped windows are, if anything, slightly *higher and higher-relief* — the most glaciated end of an
already high-alpine stratum, exactly where editing is heaviest. They are not a different, degenerate
population that happens to land in stratum 19.

**By eye** ([gallery](e58-hollowing/)): across the bulk of the distribution the dropped surfaces are
*indistinguishable from the kept ones* — sharp ridgelines, cirques and valley glaciers, sound
glaciated-alpine terrain. Two honest caveats, both visible in [`dropped-by-edm.png`](e58-hollowing/dropped-by-edm.png):
(a) a minority at the extreme high-EDM tail (\N{GREATER-THAN OR EQUAL TO}~70%) show visible smoothing consistent with heavy
interpolation; (b) **EDM fraction is a weak predictor of that** — several fully-edited (100%) tiles are still
crisp terrain while some ~70%-edited ones look smeared. So the hillshade cannot separate "trustworthy
elevation" from "edited but plausible" — which is exactly why *the terrain looks fine* is necessary but not
sufficient to justify loosening, and why [#59](https://github.com/JStoschek/DEMAE/issues/59) is the pivot.

- [`dropped-uniform.png`](e58-hollowing/dropped-uniform.png) — a representative random sample of the loss.
- [`dropped-by-edm.png`](e58-hollowing/dropped-by-edm.png) — spanning the cap→max EDM range; judge whether
  the *most-edited* end is where the terrain finally degrades.
- [`kept-uniform.png`](e58-hollowing/kept-uniform.png) — the kept stratum-19 tiles, for comparison.

Tile ids/coords are in [`e58-hollowing/manifest.json`](e58-hollowing/manifest.json) (seed `{p['provenance']['draw_seed']}`).

## 4. Where it is

The drops span **{geo['n_distinct_cogs']:,}** source COGs across **{geo['n_distinct_geocells']:,}** geocells,
latitude {geo['lat_range'][0]}–{geo['lat_range'][1]}, longitude {geo['lon_range'][0]}–{geo['lon_range'][1]} —
the world's glaciated ranges, not one artifact-ridden tile. Heaviest geocells:
{', '.join(f"`{g['geocell']}` ({g['n']})" for g in geo['top_geocells'][:6])}.

## What this hands to the decision

- **Loosening is a coherent option, and its knob is the combined edit-or-fill cap** (global or per-stratum) —
  edit and fill are co-located here ({ev['both_over_cap'] / total_drop * 100:.0f}% over both caps), so a per-mask split
  is the wrong instrument. Recovers up to {total_drop:,} genuine-looking alpine windows.
- **Accepting is also coherent**: the kept pool is {p['kept_pool_multiple_of_allocation']}× the allocation, so
  nothing is starved; the case for change rests on representativeness, not scarcity.
- **The pivotal unknown is [#59](https://github.com/JStoschek/DEMAE/issues/59)** — whether an edited/filled
  pixel carries trustworthy elevation. If it does, these tiles are safe to admit and loosening is well-motivated;
  if it does not, the hollowing is correct data-quality behaviour however good the terrain looks. This ticket
  establishes that the terrain is *real and wanted*; it deliberately leaves *is the elevation trustworthy* to #59.
"""
    out = args.out_prefix.with_suffix(".md")
    out.write_text(md)
    print(f"=== wrote {out.relative_to(REPO)} ===")


if __name__ == "__main__":
    raise SystemExit(main())
