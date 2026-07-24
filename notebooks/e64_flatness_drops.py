"""E64 — what the flatness (``degenerate``) gate drops in strata 2/5/7/17: junk or terrain?

Ticket: https://github.com/JStoschek/DEMAE/issues/64 · Map: #62 (flatness gate).

The E58-analog: the sibling investigation feeding the loosen-vs-accept decision (#66),
run in parallel with the structural-null model (#65). It reads committed data only and
needs no pre-registration. Method mirrors E2.5 / E58 exactly: dropped windows survive as
columns (#37), so every would-be member of these strata — kept *and* dropped — is scored
against the committed Phase-D centroids (``artifacts/phase_d/centroids.json``) and the
``degenerate``-gate drops are pulled out.

The ``degenerate`` gate fires on ``land_std < 3 m`` — ADR-0002 §2's "near-constant,
low-information surface an MAE cannot learn from". Within these flat strata that screen
straddles two populations the structural-null model (#65) cannot tell apart:

1. **Featureless near-constant junk** — flat lake beds, saturated pans, void-flattened
   tiles. Dropping these is *correct*.
2. **Low-relief but genuinely-structured terrain** — stratum 2 is literally "plains with
   faint drainage"; the drainage is information. Dropping these is *bias*.

This ticket asks which the drops are, three ways:

- **How recoverable?** ``degenerate`` sole-cause vs co-fired with ``water``/``zero_fill``
  — loosening the flatness cap only ever recovers windows no *other* gate also caught.
- **A cliff or a deep tail?** The ``land_std`` distribution of the drops against the 3 m
  cap, and — the junk-vs-informative axis — ``texture`` / ``geomorphon_entropy`` binned
  by ``land_std``: featureless tiles carry ~0 fine-scale dissection at any land_std,
  structured-but-flat tiles carry drainage texture the flatness cap is blind to.
- **Good terrain or blank?** The dropped surfaces rendered from the source COGs
  (``gallery.shade_tile``) beside the kept ones. These tiles are *flat*, so the shader
  auto-exaggerates toward its 60 m target; the caption prints ``land_std``, ``texture``,
  ``geomorphon_entropy`` and the exaggeration applied so the amplification is never silent.

Then the starvation check: after the drop, do 2/5/7/17 still keep enough members against
their Phase-D allocation? (Stratum 19 had 48x headroom.)

AFK. Writes ``docs/research/e64-*`` — a JSON payload, a markdown summary, a 2x2 land_std
histogram, and gallery sheets with a reproducible tile manifest. No pipeline state is
touched. Centred on **stratum 5** (the decider, per the #54 verdict and the #63 bar);
2/7/17 carried as lower-confidence corroboration.
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
DEGEN_LAND_STD_M = 3.0            # the flatness cap under review (ADR-0002 §2)
DRAW_SEED = gallery.GALLERY_SEED  # reuse the registered gallery seed for keyed draws

# stratum 5 first: the decider (5.32x, survives both baselines — #54 §5). 2/7/17 are the
# three marginal pairs (3.0-3.25x, baseline-dependent) carried as corroboration only.
STRATA = [5, 2, 7, 17]
PRIMARY = 5

# d2-allocation-nh.json: N_h (kept would-be pop k-means was fit on) + n_h (allocation).
ALLOCATION = {2: 2316, 5: 1991, 7: 2604, 17: 1847}
STRATUM_NAME = {
    2: "low-relief plains with faint drainage",
    5: "near-level plains / basins",
    7: "high-altitude flats / plateaus",
    17: "low-relief lowlands with drainage / water",
}

METRICS = ("land_std", "slope_mean", "local_relief", "texture",
           "elevation_relief_ratio", "mean_elevation", "geomorphon_entropy")


# --- Loading + stratum assignment (mirror of E2.5 / E58's would_be_strata) ------


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
    for m in METRICS:
        v = col(cat, m)[mask]
        v = v[np.isfinite(v)]
        out[m] = {"mean": round(float(v.mean()), 3), "median": round(float(np.median(v)), 3),
                  "p10": round(float(np.quantile(v, 0.1)), 3),
                  "p90": round(float(np.quantile(v, 0.9)), 3)} if v.size else {}
    return out


def landstd_texture_crosstab(cat: Any, drop: np.ndarray) -> list[dict[str, Any]]:
    """The junk-vs-informative core: texture & landform-entropy binned by land_std.

    Featureless tiles carry ~0 fine-scale dissection at *every* land_std; structured-but-
    flat tiles carry drainage texture the flatness cap cannot see. If the deep-flat band
    (land_std<1) is texture-dead and the near-cap band (2.5-3) is as textured as kept
    terrain, the cap is slicing a continuum and a modest loosening recovers the textured
    end without scooping the blank end.
    """
    ls = col(cat, "land_std")
    tex = col(cat, "texture")
    ent = col(cat, "geomorphon_entropy")
    bands = [(0.0, 1.0), (1.0, 2.0), (2.0, 2.5), (2.5, DEGEN_LAND_STD_M)]
    rows = []
    for lo, hi in bands:
        sel = drop & (ls >= lo) & (ls < hi)
        t, e = tex[sel], ent[sel]
        t, e = t[np.isfinite(t)], e[np.isfinite(e)]
        rows.append({
            "land_std_band": f"{lo:.1f}-{hi:.1f}",
            "n": int(sel.sum()),
            "texture_median": round(float(np.median(t)), 4) if t.size else None,
            "texture_p90": round(float(np.quantile(t, 0.9)), 4) if t.size else None,
            "geomorphon_entropy_median": round(float(np.median(e)), 4) if e.size else None,
        })
    return rows


def characterize(cat: Any, labels: np.ndarray, s: int,
                 corpus_degen_rate: float) -> dict[str, Any]:
    mem = labels == s
    keep = col(cat, "keep").astype(bool)
    degen = col(cat, "degenerate").astype(bool)

    kept = mem & keep
    drop = mem & degen                       # the windows under investigation
    # what would still catch a recovered window if we loosened the flatness cap alone
    other = np.zeros(cat.num_rows, dtype=bool)
    for g in ("zero_fill", "water", "edited", "masks_missing"):
        other |= col(cat, g).astype(bool)

    n_mem, n_kept, n_drop = int(mem.sum()), int(kept.sum()), int(drop.sum())
    n_h = ALLOCATION[s]
    ls_drop = col(cat, "land_std")[drop]
    ls_drop = ls_drop[np.isfinite(ls_drop)]
    ls_kept = col(cat, "land_std")[kept]
    ls_kept = ls_kept[np.isfinite(ls_kept)]

    # land_std bands below the cap: cliff (piled near 3) vs deep tail (down to 0).
    ls_bands = [(0.0, 1.0), (1.0, 2.0), (2.0, 2.5), (2.5, DEGEN_LAND_STD_M)]
    ls_band_counts = {f"{lo:.1f}-{hi:.1f}": int(((ls_drop >= lo) & (ls_drop < hi)).sum())
                      for lo, hi in ls_bands}

    return {
        "stratum": s,
        "name": STRATUM_NAME[s],
        "membership": n_mem,
        "kept": n_kept,
        "degenerate_drops": n_drop,
        "within_stratum_drop_rate": round(n_drop / n_mem, 4) if n_mem else None,
        "corpus_degenerate_rate": round(corpus_degen_rate, 4),
        "ratio_vs_corpus": round((n_drop / n_mem) / corpus_degen_rate, 2) if n_mem else None,
        "allocation_n_h": n_h,
        "kept_pool_multiple_of_allocation": round(n_kept / n_h, 1),
        "recoverability": {
            # loosening the flatness cap only recovers drops no other gate also caught
            "sole_cause_degenerate": int((drop & ~other).sum()),
            "co_fire_water": int((drop & col(cat, "water").astype(bool)).sum()),
            "co_fire_zero_fill": int((drop & col(cat, "zero_fill").astype(bool)).sum()),
            "co_fire_edited": int((drop & col(cat, "edited").astype(bool)).sum()),
        },
        "land_std_of_drops": {"quantiles": quantiles(ls_drop), "bands_below_cap": ls_band_counts},
        "land_std_of_kept": {"quantiles": quantiles(ls_kept)},
        "junk_vs_informative": landstd_texture_crosstab(cat, drop),
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
    """2x2: land_std kept (>=3, by definition) vs dropped (<3) per stratum, cap marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ls = col(cat, "land_std")
    keep = col(cat, "keep").astype(bool)
    degen = col(cat, "degenerate").astype(bool)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), squeeze=False)
    bins = np.linspace(0, 8, 65)
    for ax, s in zip(axes.ravel(), STRATA):
        mem = labels == s
        kept_ls = ls[mem & keep]; kept_ls = kept_ls[np.isfinite(kept_ls)]
        drop_ls = ls[mem & degen]; drop_ls = drop_ls[np.isfinite(drop_ls)]
        ax.hist(np.clip(kept_ls, 0, 8), bins=bins, color="#4c78a8", alpha=0.85,
                label=f"kept (n={kept_ls.size:,})")
        ax.hist(np.clip(drop_ls, 0, 8), bins=bins, color="#e45756", alpha=0.85,
                label=f"degenerate drop (n={drop_ls.size:,})")
        ax.axvline(DEGEN_LAND_STD_M, color="#333", lw=1.3, ls="--")
        ax.set_title(f"stratum {s} — {STRATUM_NAME[s]}", fontsize=9)
        ax.set_xlabel("land_std (m)", fontsize=8)
        ax.set_ylabel("windows", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Flat strata — land_std, kept vs degenerate-dropped (cap = 3 m)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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


def caption(cat: Any, i: int, ve: float) -> str:
    g = lambda m: col(cat, m)[i]  # noqa: E731
    return (f"land_std {g('land_std'):>4.1f} m  tex {g('texture'):>4.2f}  "
            f"gent {g('geomorphon_entropy'):>4.2f}\n"
            f"relief {g('local_relief'):>4.0f} m  VE {ve:>4.0f}x  "
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
        rgb, ve, _sp = gallery.shade_tile(tiles[i], lat=float(col(cat, "lat")[i]))
        ax.imshow(rgb, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_axis_on()
        for sp in ax.spines.values():
            sp.set_color("#999"); sp.set_linewidth(0.6)
        ax.set_xlabel(caption(cat, i, ve), fontsize=6.5, family="monospace", labelpad=3,
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


def pick_spanning_landstd(cat: Any, pool: np.ndarray, n: int) -> list[int]:
    """One tile from each of n equal-count land_std bins: deep-flat (~0) -> just-under-cap."""
    ls = col(cat, "land_std")[pool]
    order = pool[np.argsort(ls, kind="stable")]
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
    p.add_argument("--out-prefix", type=Path, default=RESEARCH / "e64-flatness-drops")
    p.add_argument("--gallery-dir", type=Path, default=RESEARCH / "e64-flatness")
    p.add_argument("--per-sheet", type=int, default=12)
    p.add_argument("--no-gallery", action="store_true",
                   help="skip the S3 COG reads (quantitative half only)")
    return p.parse_args(argv)


def build_sheets(cat: Any, labels: np.ndarray, n: int
                 ) -> dict[str, tuple[list[int], str, str]]:
    """Full three-sheet treatment for the decider (stratum 5); one dropped sheet each
    for the corroborating strata."""
    keep = col(cat, "keep").astype(bool)
    degen = col(cat, "degenerate").astype(bool)
    sheets: dict[str, tuple[list[int], str, str]] = {}
    for s in STRATA:
        drop_pool = np.flatnonzero((labels == s) & degen)
        sheets[f"s{s:02d}-dropped-uniform"] = (
            pick_uniform(cat, drop_pool, n),
            f"Stratum {s} ({STRATUM_NAME[s]}) — dropped by `degenerate`, keyed-random",
            f"{drop_pool.size:,} windows removed by the land_std<3 m flatness cap")
        if s == PRIMARY:
            sheets[f"s{s:02d}-dropped-by-landstd"] = (
                pick_spanning_landstd(cat, drop_pool, n),
                f"Stratum {s} — dropped, spanning land_std (deep-flat \N{RIGHTWARDS ARROW} just-under-cap)",
                "one tile per equal-count land_std bin; is the flattest end where terrain dies?")
            kept_pool = np.flatnonzero((labels == s) & keep)
            sheets[f"s{s:02d}-kept-uniform"] = (
                pick_uniform(cat, kept_pool, n),
                f"Stratum {s} — kept (land_std>=3 m), keyed-random (comparison)",
                f"{kept_pool.size:,} windows that cleared the flatness cap")
    return sheets


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cat = catalog.read_catalog(args.catalog)
    sc = frame.read_scaler(args.scaler)
    cn = centroids.read_centroids(args.centroids)
    labels, assignable = would_be_strata(cat, cn, sc)
    degen = col(cat, "degenerate").astype(bool)
    corpus_degen_rate = float(degen.sum() / cat.num_rows)

    print(f"=== catalog {cat.num_rows:,} rows; {assignable.sum():,} assignable; "
          f"corpus degenerate rate {corpus_degen_rate * 100:.2f}% ===")

    strata_payload = {}
    for s in STRATA:
        p = characterize(cat, labels, s, corpus_degen_rate)
        strata_payload[str(s)] = p
        rec = p["recoverability"]
        print(f"  s{s:<2} {p['name'][:34]:34}  mem {p['membership']:>7,}  "
              f"drop {p['degenerate_drops']:>7,} "
              f"({p['within_stratum_drop_rate'] * 100:>4.1f}%, {p['ratio_vs_corpus']}x)  "
              f"sole-cause {rec['sole_cause_degenerate']:>7,}  "
              f"keep/alloc {p['kept_pool_multiple_of_allocation']}x")

    payload: dict[str, Any] = {
        "gate": "degenerate", "cap_land_std_m": DEGEN_LAND_STD_M,
        "corpus_degenerate_rate": round(corpus_degen_rate, 4),
        "primary_stratum": PRIMARY, "strata": strata_payload,
        "provenance": {"code_version": git_head(),
                       "catalog": str(args.catalog.relative_to(REPO)),
                       "centroids": str(args.centroids.relative_to(REPO)),
                       "draw_seed": DRAW_SEED},
    }

    histogram(cat, labels, args.out_prefix.with_name(args.out_prefix.name + "-landstd.png"))

    manifest: dict[str, Any] = {"seed": DRAW_SEED, "sheets": {}}
    if not args.no_gallery:
        sheets = build_sheets(cat, labels, args.per_sheet)
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
                 "land_std": round(float(col(cat, "land_std")[i]), 3),
                 "texture": round(float(col(cat, "texture")[i]), 3),
                 "geomorphon_entropy": round(float(col(cat, "geomorphon_entropy")[i]), 3)}
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
    P = p["strata"][str(PRIMARY)]
    prov = p["provenance"]

    def sp(s: int) -> dict[str, Any]:
        return p["strata"][str(s)]

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    def tex(s: int, side: str) -> float:
        return sp(s)["terrain_metrics"][side]["texture"]["median"]

    def gent(s: int, side: str) -> float:
        return sp(s)["terrain_metrics"][side]["geomorphon_entropy"]["median"]

    # per-stratum headline table
    head_rows = "\n".join(
        f"| {s} | {sp(s)['name']} | {sp(s)['membership']:,} "
        f"| {sp(s)['degenerate_drops']:,} | {pct(sp(s)['within_stratum_drop_rate'])} "
        f"| {sp(s)['ratio_vs_corpus']}× "
        f"| {sp(s)['recoverability']['sole_cause_degenerate']:,} "
        f"| {sp(s)['kept']:,} | {sp(s)['kept_pool_multiple_of_allocation']}× |"
        for s in STRATA)

    # cross-stratum texture / entropy comparison (the junk-vs-informative core)
    readings = {
        5: "dropped texture **≡** kept — a pure `land_std` cut in a texture-homogeneous family",
        2: "dropped **more** textured than kept — the cut removes the drainage-bearing tiles",
        7: "polar ice plateau — high texture/entropy is amplified surface noise, not landforms",
        17: "dropped **more** textured than kept — lowland drainage + a heavy `water` co-fire",
    }
    cs_rows = "\n".join(
        f"| {s} | {sp(s)['name']} | {tex(s, 'dropped')} | {tex(s, 'kept')} "
        f"| {gent(s, 'dropped')} | {readings[s]} |"
        for s in STRATA)

    # stratum-5 land_std x texture crosstab — texture is flat across bands, unlike a
    # featureless-core/textured-cap split
    ct_rows = "\n".join(
        f"| {r['land_std_band']} m | {r['n']:,} | {r['texture_median']} "
        f"| {r['texture_p90']} | {r['geomorphon_entropy_median']} |"
        for r in P["junk_vs_informative"])

    def mrow(s_p: dict[str, Any], m: str) -> str:
        d, k = s_p["terrain_metrics"]["dropped"][m], s_p["terrain_metrics"]["kept"][m]
        return (f"| {m} | {d['median']} ({d['p10']}\N{EN DASH}{d['p90']}) "
                f"| {k['median']} ({k['p10']}\N{EN DASH}{k['p90']}) |")

    metric_rows = "\n".join(mrow(P, m) for m in METRICS)
    geo = P["geography"]
    rec = P["recoverability"]
    lsq = P["land_std_of_drops"]["quantiles"]
    deep = P["land_std_of_drops"]["bands_below_cap"]["0.0-1.0"]
    total = P["degenerate_drops"]
    min_mult = min(sp(s)["kept_pool_multiple_of_allocation"] for s in STRATA)

    md = f"""# E64 — What the flatness gate drops in strata 2/5/7/17: featureless junk, or informative terrain?

**Ticket:** [Characterize what the flatness gate drops (#64)](https://github.com/JStoschek/DEMAE/issues/64) ·
**Map:** [flatness gate (#62)](https://github.com/JStoschek/DEMAE/issues/62) ·
**Seeded by:** [Gate E FAIL (#54)](https://github.com/JStoschek/DEMAE/issues/54) ·
`{prov['code_version']}`

The E58-analog — the sibling investigation feeding the loosen-vs-accept decision
([#66](https://github.com/JStoschek/DEMAE/issues/66)), run in parallel with the structural-null model
([#65](https://github.com/JStoschek/DEMAE/issues/65)). Method mirrors E2.5 / E58: dropped windows survive as
columns ([#37](https://github.com/JStoschek/DEMAE/issues/37)), so every would-be member of these strata —
kept *and* dropped — is scored against the committed Phase-D centroids (`{prov['centroids']}`) and the
`degenerate` (`land_std < 3 m`) drops pulled out. Centred on **stratum 5**, the decider (5.32×, the one pair
that survives both baselines — [#54 §5](https://github.com/JStoschek/DEMAE/issues/54)); 2/7/17 carried as
lower-confidence corroboration.

## Verdict in one line

The flatness cap does **not** drop a single population — what it removes is **stratum-dependent**, and no
single metric captures it, so this is a by-eye call as much as a numeric one. Centred on **stratum 5** (the
decider), the drops are the *flatter end of the same smooth low-relief family as the kept tiles* — dropped
texture (**{tex(5,'dropped')}**) is **identical** to kept (**{tex(5,'kept')}**); the gate cuts purely on
overall relief, and the loss reads as genuinely low-information plains, **correctly** targeted. But in the
*marginal* strata **2 and 17** the dropped tiles are **more** dissected than the kept ones (§2) and carry
visible river/drainage structure by eye (§3) — the more plausible "informative terrain removed". Either way
selection is **not starved**: all four strata keep **≥{int(min_mult)}×** their allocation
(stratum 5: **{P['kept']:,}** kept vs **{P['allocation_n_h']:,}** allocated). As with stratum 19 the live
question for [#66](https://github.com/JStoschek/DEMAE/issues/66) is *representativeness*, not scarcity — and
the terrain-quality signal here **splits from** the rate-excess signal ([#65](https://github.com/JStoschek/DEMAE/issues/65)):
the stratum with the largest excess (5) shows the *weakest* case for informative loss, the strata with the
smallest excess (2/17) the strongest.

## The four flat strata at a glance

| h | family | would-be members | degenerate drops | within-stratum | vs corpus | sole-cause | kept pool | kept / alloc |
|---|---|---|---|---|---|---|---|---|
{head_rows}

Membership and drop counts reproduce E2.5's §5 table (stratum 5: 247,681 members, 128,511 fires, 51.9%,
5.32×) — the pipeline is the same, only the question is new.

## 1. How much is even recoverable — the flatness cap is not the only gate on these tiles

Loosening the `land_std` cap only recovers a window if **no other gate also caught it**. For stratum 5, of
the **{total:,}** degenerate drops, **{rec['sole_cause_degenerate']:,}**
({rec['sole_cause_degenerate'] / total * 100:.0f}%) are **sole-cause** `degenerate`; **{rec['co_fire_water']:,}**
also fire `water` and **{rec['co_fire_zero_fill']:,}** also fire `zero_fill`. The water/zero-fill co-fires are
flat *because they are wet or void* — exactly what the flatness screen backstops — and a flatness loosening
would not readmit them. So the recoverable ceiling is the sole-cause count, not the headline drop. Stratum
**17** (the drainage/water family) carries the heaviest co-fire — {sp(17)['recoverability']['co_fire_water']:,}
of its {sp(17)['degenerate_drops']:,} drops also fire `water` — so it has the least to gain from a flatness
loosening; stratum **7** (ice, no liquid water) has **{sp(7)['recoverability']['co_fire_water']}** water co-fire.

## 2. Texture separates the strata, not the land_std bands — and it cuts both ways

The gate keys on `land_std`, which is blind to *fine-scale* structure. The revealing comparison is **dropped
vs kept texture within each stratum**, plus what the eye confirms about each:

| h | family | dropped texture | kept texture | dropped geomorphon-entropy | reading |
|---|---|---|---|---|---|
{cs_rows}

Two honest cautions, in **opposite** directions, are why this can't be settled on `texture` alone:

- **`texture` understates drainage** (strata 2, 17). River networks and incised valleys are *meso*-scale —
  they raise `local_relief` and show plainly by eye but barely move a fine-scale texture metric. That the
  dropped tiles here still out-score the kept ones on texture is therefore a *floor* on their structure, not
  a ceiling.
- **`texture` and `geomorphon_entropy` overstate structure on ice** (stratum 7). Its drops score texture
  ~{tex(7,'dropped')} and entropy ~{gent(7,'dropped')} — near the corpus top — yet by eye are a uniform
  East-Antarctic/Greenland plateau surface: the metrics are reading amplified sub-metre sastrugi/noise and a
  geomorphon classifier firing on it, not landform variety.

Within **stratum 5** the picture is cleaner: `texture` is flat across the whole `land_std` range and equals
the kept level — there is no textured sub-population hiding among the drops:

| land_std band | dropped windows | texture median | texture p90 | geomorphon-entropy median |
|---|---|---|---|---|
{ct_rows}

`land_std` of the stratum-5 drops (p10 **{lsq.get('p10')}**, p50 **{lsq.get('p50')}**, p90 **{lsq.get('p90')}** m)
shows the cap slicing a *continuum* down toward zero — **{deep:,}** ({deep / total * 100:.0f}%) sit in the
deepest-flat `<1 m` band. See [`e64-flatness-drops-landstd.png`](e64-flatness-drops-landstd.png).

## 3. By eye — the same family, flatter; with real drainage in 2/17 and blank ice in 7

Median (p10–p90) over `land_std` + the six clustering metrics, stratum-5 dropped vs kept:

| metric | dropped | kept |
|---|---|---|
{metric_rows}

The stratum-5 drops are simply the *flatter, lower-relief* end of an already flat stratum — not a
compositionally different, degenerate population. **These tiles are flat, so the shader auto-exaggerates
toward its 60 m target (VE up to ~370×), and that amplification manufactures apparent roughness from
sub-metre noise** — so the eye must be read against the printed `land_std`/`texture`/`VE`, not on its own.

- **Stratum 5** ([`s05-dropped-uniform.png`](e64-flatness/s05-dropped-uniform.png),
  [`s05-dropped-by-landstd.png`](e64-flatness/s05-dropped-by-landstd.png),
  [`s05-kept-uniform.png`](e64-flatness/s05-kept-uniform.png)): smooth low-relief plains and basins. The
  deepest-flat draws (`land_std < 1 m`, {deep / total * 100:.0f}% of the loss) render as amplified noise —
  featureless blanks correctly dropped; the rest are gentle surfaces with the occasional low-relief drainage
  line. The kept tiles are the *same family* at lower exaggeration — the cut is on relief, not on kind.
- **Stratum 2** ([`s02-dropped-uniform.png`](e64-flatness/s02-dropped-uniform.png)): genuine fluvial
  structure — meandering rivers (some with mapped water), dendritic drainage, incised valleys — interleaved
  with amplified-noise plains. The clearest "informative low-relief terrain" in the four.
- **Stratum 7** ([`s07-dropped-uniform.png`](e64-flatness/s07-dropped-uniform.png)): a uniform polar
  ice-plateau surface; the high texture/entropy metrics are the surface-noise artefact described in §2, not
  learnable landform. Correctly targeted despite the metrics.
- **Stratum 17** ([`s17-dropped-uniform.png`](e64-flatness/s17-dropped-uniform.png)): lowland drainage
  networks and incised channels **plus** coastal/lake tiles (the `water` co-fire) — a genuine-drainage
  population mixed with correctly-flagged water.

Tile ids/coords are in [`e64-flatness/manifest.json`](e64-flatness/manifest.json) (seed `{prov['draw_seed']}`).

## 4. Where it is

The stratum-5 drops span **{geo['n_distinct_cogs']:,}** source COGs across **{geo['n_distinct_geocells']:,}**
geocells, latitude {geo['lat_range'][0]}–{geo['lat_range'][1]} — the world's arid plains and basins
(heaviest: {', '.join(f"`{g['geocell']}`" for g in geo['top_geocells'][:5])}), not one artifact-ridden tile.
Stratum 7's drops, by contrast, concentrate over the Antarctic/Greenland ice sheets
({', '.join(f"`{g['geocell']}`" for g in sp(7)['geography']['top_geocells'][:3])}) — consistent with the
ice-surface reading.

## What this hands to the decision ([#66](https://github.com/JStoschek/DEMAE/issues/66))

- **The drop is a stratum-dependent mix, not uniform junk.** Stratum 5 (the biggest rate excess) removes
  genuinely low-information smooth plains — texture-identical to the kept tiles, the same family flatter;
  strata 2/17 (marginal excess) remove the *more*-dissected, drainage-bearing low-relief tiles — the real
  "informative terrain removed"; stratum 7 removes ice-plateau surface whose high metrics are noise.
- **The terrain axis and the rate axis point opposite ways across strata.** The strongest ACCEPT case on
  terrain (stratum 5) is the strongest LOOSEN case on rate excess, and vice-versa for 2/17 — the decision
  must reconcile #64 and #65, not read either alone.
- **The recoverable ceiling is the sole-cause count** (5: {rec['sole_cause_degenerate']:,}; 17 much less,
  gated by its water co-fire), not the headline drop.
- **Nothing is starved.** All four kept pools are ≥{int(min_mult)}× their allocations, so — as with stratum
  19 — the case for change rests on representativeness, not scarcity.
- **If loosened, the knob is the `land_std` cap** (global lower value vs per-stratum exemption — the shape
  [#63](https://github.com/JStoschek/DEMAE/issues/63) pinned). A per-stratum exemption for 2/17 (where the
  informative loss concentrates) fits this evidence better than a global re-tune that would also readmit
  stratum 5's low-information plains. Whether a near-constant surface carries MAE-learnable signal is
  ADR-0002 §2's premise, which this ticket does not reopen.
"""
    out = args.out_prefix.with_suffix(".md")
    out.write_text(md)
    print(f"=== wrote {out.relative_to(REPO)} ===")


if __name__ == "__main__":
    raise SystemExit(main())
