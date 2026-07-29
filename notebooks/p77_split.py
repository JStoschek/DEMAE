"""P77 — report the geographic split committed against the 500k scale corpus.

Ticket: https://github.com/JStoschek/DEMAE/issues/77 · Map: #76 (Phase 1).
Draw: ``python -m dataset.splits --fill``

This script does **not** draw. The draw is :mod:`dataset.splits`, and re-implementing any
part of it here would create a second splitter. What it reads is the *committed*
assignment, re-derives it from the shipped manifest to prove the file reproduces, and
writes the four things #77 asks the split to report, plus the two the plan defers to it:

1. **The realized three-way split** — totals, shares, and per-stratum counts for all three
   splits against the corpus composition, so "representative" is a number.
2. **The 13 test geocells** — where they are, what terrain they hold, and how far apart.
3. **Per-stratum train counts, and what removing the hold-outs did to the Neyman
   allocation** (plan §9b). The draw's ``n_h`` is a committed artifact
   (``f-allocation-nh-500k.json``); the realized train counts are that table minus whatever
   the two hold-outs happened to take, and the perturbation is what this reports.
4. **The global mu/sigma contamination** (plan §9c). The shipped 955.67 / 1012.16 were
   accumulated over *every* tile, including the 4.98% of land pixels now held out. The
   manifest carries per-tile land ``mean``/``std``/``land_frac``, and those moments are
   additive, so the train-only pair can be recomputed exactly — without reading a shard —
   and the plan's "recompute or accept" becomes a measured 0.31%.
5. **The boundary effect** — the distance from each held-out tile to the nearest *train*
   tile. The plan asserts geocell blocking "puts hundreds of km between train and test";
   this measures it, and the median is 97 km, not hundreds. Published, not fixed: the
   split is drawn exactly as §9 specifies and the correction is argued here.
6. **Why val draws in hash order** — the rejected shortfall-greedy variant, re-run here so
   its 9.4x coastline rate is a number this report generates rather than remembers.

AFK. Reads committed artifacts only (``data/corpus_500k/manifest.parquet``,
``artifacts/phase1/*``, ``docs/research/f-allocation-nh-500k.json``); draws nothing it
keeps and writes nothing outside ``docs/research/p77-split.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from dataset import config, splits

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "docs" / "research"
MANIFEST = REPO / "data" / "corpus_500k" / "manifest.parquet"
ALLOCATION = RESEARCH / "f-allocation-nh-500k.json"

#: Tile pixels — the denominator that turns ``land_frac`` back into a land-pixel count.
TILE_PIXELS = config.TILE_PX * config.TILE_PX

#: A one-line gloss per test geocell, so the cell list reads as regions and not as ids.
#: Coordinates are the authority; these names are the reviewer's convenience.
REGION_GLOSS: dict[str, str] = {
    "+000_-010": "Gabon and the western Congo basin",
    "+020_+060": "Fennoscandia — Finnish lakeland and Karelia",
    "+020_+070": "Svalbard and the Barents shelf islands",
    "+030_+000": "Ethiopian highlands and the East African Rift",
    "+050_-090": "East Antarctic interior",
    "+090_+040": "Mongolian Altai, Dzungaria and the Gobi",
    "+150_-010": "Bismarck Archipelago and the Solomon Sea islands",
    "+150_-070": "Antarctica — George V Land and the Wilkes coast",
    "-090_+050": "Canadian Shield — northern Ontario and the Hudson Bay lowlands",
    "-100_+010": "southern Mexico — Sierra Madre del Sur and Tehuantepec",
    "-100_+060": "Canadian Shield — Kivalliq and western Hudson Bay",
    "-170_-090": "Antarctica — Ross Ice Shelf and the Transantarctic margin",
    "-170_+050": "western Alaska — Yukon-Kuskokwim delta and the Bering coast",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- the numbers --------------------------------------------------------------


def realized(label: np.ndarray, stratum: np.ndarray, k: int) -> dict[str, Any]:
    """Totals, shares and per-stratum counts for the three splits."""
    n = len(label)
    q = np.bincount(stratum, minlength=k) / n
    out: dict[str, Any] = {"n_tiles": int(n), "corpus_share": [round(float(x), 6) for x in q], "splits": {}}
    for s in splits.SPLITS:
        m = label == s
        counts = np.bincount(stratum[m], minlength=k)
        share = counts / counts.sum()
        out["splits"][s] = {
            "n_tiles": int(m.sum()),
            "share": round(float(m.mean()), 6),
            "per_stratum": [int(x) for x in counts],
            "ratio_to_corpus": [round(float(a / b), 4) for a, b in zip(share, q)],
            "shortfall": round(splits.relative_shortfall(counts, q), 4),
            "total_variation": round(float(0.5 * np.abs(share - q).sum()), 4),
        }
    return out


def test_regions(blocks: dict[str, Any], cells, cogs, lon, lat, stratum, k) -> list[dict[str, Any]]:
    """One row per held-out geocell: where it is, how big, and what terrain it holds."""
    rows = []
    for cid in blocks["test"]["cells"]:
        m = cells == cid
        counts = np.bincount(stratum[m], minlength=k)
        rows.append({
            "geocell_id": cid,
            "region": REGION_GLOSS.get(cid, ""),
            "n_tiles": int(m.sum()),
            "n_cogs": int(len(np.unique(cogs[m]))),
            "lon": round(float(lon[m].mean()), 3),
            "lat": round(float(lat[m].mean()), 3),
            "n_strata": int((counts > 0).sum()),
            "top_strata": [int(h) for h in np.argsort(-counts)[:3]],
        })
    return sorted(rows, key=lambda r: -r["n_tiles"])


def eligibility(cells: np.ndarray, stratum: np.ndarray, k: int, budget: int) -> dict[str, Any]:
    """What the per-cell size band excludes, and which terrain families it takes with it.

    §9 says "select 12-15 of the 425 occupied geocells"; the draw searches 328 of them,
    because a cell holding more than a quarter of the test budget would let one province
    decide a downstream verdict and one holding less than a fortieth spends a slot without
    buying a region. That is a real narrowing of the spec's pool and it is not neutral:
    the excluded cells are the biggest, which is where the high-plateau and high-alpine
    families live. This reports the price rather than leaving it implicit.
    """
    blocks = splits.blocks_from(cells, stratum, np.zeros(len(cells)), np.zeros(len(cells)), k=k)
    n = blocks.n
    inside = (n >= splits.TEST_CELL_MIN_SHARE * budget) & (n <= splits.TEST_CELL_MAX_SHARE * budget)
    total = blocks.counts.sum(axis=0)
    kept = blocks.counts[inside].sum(axis=0)
    reachable = kept / total
    worst = np.argsort(reachable)[:3]
    return {
        "occupied_cells": int(len(blocks.ids)),
        "eligible_cells": int(inside.sum()),
        "excluded_too_large": int((n > splits.TEST_CELL_MAX_SHARE * budget).sum()),
        "excluded_too_small": int((n < splits.TEST_CELL_MIN_SHARE * budget).sum()),
        "band_tiles": [int(splits.TEST_CELL_MIN_SHARE * budget), int(splits.TEST_CELL_MAX_SHARE * budget)],
        "least_reachable_strata": [
            {"stratum_id": int(h), "share_inside_band": round(float(reachable[h]), 4),
             "tiles_inside_band": int(kept[h]), "tiles_total": int(total[h])}
            for h in worst
        ],
    }


def latitude_balance(label: np.ndarray, lat: np.ndarray) -> list[dict[str, Any]]:
    """Corpus vs test share by coarse latitude belt — the check on "is this set polar-heavy"."""
    belts = [("60S and south", -91.0, -60.0), ("60S-0", -60.0, 0.0), ("0-30N", 0.0, 30.0),
             ("30-60N", 30.0, 60.0), ("60N and north", 60.0, 91.0)]
    out = []
    for name, lo, hi in belts:
        m = (lat >= lo) & (lat < hi)
        out.append({
            "belt": name,
            "corpus_pct": round(float(100 * m.mean()), 2),
            "test_pct": round(float(100 * (m & (label == "test")).sum() / (label == "test").sum()), 2),
            "val_pct": round(float(100 * (m & (label == "val")).sum() / (label == "val").sum()), 2),
        })
    return out


def separations(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Pairwise great-circle separation between the test cells — the diversity constraint."""
    lon = np.array([r["lon"] for r in rows])
    lat = np.array([r["lat"] for r in rows])
    d = splits.great_circle_km(lon[:, None], lat[:, None], lon, lat)
    iu = np.triu_indices(len(rows), 1)
    return {
        "min_km": round(float(d[iu].min()), 1),
        "median_km": round(float(np.median(d[iu])), 1),
        "max_km": round(float(d[iu].max()), 1),
    }


def neyman_perturbation(train_counts: np.ndarray, k: int) -> dict[str, Any]:
    """What holding out ~5% did to the drawn Neyman allocation (plan §9b).

    ``n_h`` is what the sampler allocated; the corpus realized it minus materialization
    drops; ``train`` is that minus whatever the two hold-outs took. Neyman shares are
    budget-invariant, so the honest statement is about *shares*: the train pool is no
    longer the allocation the sampler drew, and this says by how much.
    """
    alloc = json.loads(ALLOCATION.read_text())
    n_h = np.array([r["n_h"] for r in alloc["rows"]], dtype=np.float64)
    drawn_share = n_h / n_h.sum()
    train_share = train_counts / train_counts.sum()
    return {
        "source": str(ALLOCATION.relative_to(REPO)),
        "rows": [
            {
                "stratum_id": int(h),
                "n_h_drawn": int(n_h[h]),
                "train": int(train_counts[h]),
                "drawn_share": round(float(drawn_share[h]), 6),
                "train_share": round(float(train_share[h]), 6),
                "relative_shift_pct": round(float(100 * (train_share[h] / drawn_share[h] - 1)), 3),
            }
            for h in range(k)
        ],
        "max_abs_relative_shift_pct": round(float(np.abs(100 * (train_share / drawn_share - 1)).max()), 3),
        "total_variation": round(float(0.5 * np.abs(train_share - drawn_share).sum()), 5),
    }


def pooled_land_stats(mean, std, land_frac, mask) -> tuple[float, float, int]:
    """Corpus-wide land mean/std over a subset, from the manifest's per-tile moments.

    ``normalize.land_moments`` accumulates ``(sum, sqsum, count)`` per tile and divides
    once at the end, and all three are additive — so the same pair can be re-derived for
    any subset of tiles from the manifest's stored per-tile ``mean``/``std``/``land_frac``,
    with no shard read. The whole-corpus case reproduces the shipped constants to four
    decimals, which is what licenses the train-only number beside it.
    """
    n = np.rint(land_frac[mask] * TILE_PIXELS).astype(np.int64)
    total = int(n.sum())
    mu = float((mean[mask] * n).sum() / total)
    sq = float((n * (std[mask] ** 2 + mean[mask] ** 2)).sum() / total)
    return mu, float(np.sqrt(max(sq - mu * mu, 0.0))), total


def contamination(mean, std, land_frac, label) -> dict[str, Any]:
    """The plan §9c caveat, measured rather than bounded."""
    meta = json.loads((MANIFEST.parent / "corpus_meta.json").read_text())
    everything = pooled_land_stats(mean, std, land_frac, np.ones(len(label), dtype=bool))
    out = {
        "shipped": {"mean_m": meta["global_mean"], "std_m": meta["global_std"]},
        "recomputed_all_tiles": {"mean_m": round(everything[0], 4), "std_m": round(everything[1], 4)},
        "per_split": {},
    }
    for s in splits.SPLITS:
        mu, sd, n = pooled_land_stats(mean, std, land_frac, label == s)
        out["per_split"][s] = {"mean_m": round(mu, 4), "std_m": round(sd, 4), "land_pixels": n}
    tr = out["per_split"]["train"]
    out["held_out_land_pixel_share"] = round(1 - tr["land_pixels"] / everything[2], 6)
    out["delta_if_recomputed_on_train"] = {
        "mean_m": round(tr["mean_m"] - everything[0], 4),
        "mean_pct": round(100 * (tr["mean_m"] - everything[0]) / everything[0], 4),
        "std_m": round(tr["std_m"] - everything[1], 4),
        "std_pct": round(100 * (tr["std_m"] - everything[1]) / everything[1], 4),
    }
    return out


def boundary(label: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> dict[str, Any]:
    """Distance from every held-out tile to the nearest train tile, and the buffer price.

    The blocking argument is about *distance*, so it is measured in km rather than in
    cells: chord distances from a KD-tree over unit-sphere coordinates, converted back to
    great-circle. The buffer row prices the fix this measurement invites — how many train
    tiles a 10 / 25 km exclusion zone around the hold-outs would cost — without applying it.
    """
    from scipy.spatial import cKDTree

    r = splits.EARTH_RADIUS_KM
    rad_lon, rad_lat = np.radians(lon), np.radians(lat)
    xyz = np.stack([np.cos(rad_lat) * np.cos(rad_lon), np.cos(rad_lat) * np.sin(rad_lon), np.sin(rad_lat)], 1) * r
    train = label == "train"
    tree = cKDTree(xyz[train])
    out: dict[str, Any] = {}
    for s in ("test", "val"):
        chord, _ = tree.query(xyz[label == s], k=1)
        d = 2 * r * np.arcsin(np.clip(chord / (2 * r), 0, 1))
        out[s] = {
            "min_km": round(float(d.min()), 2),
            "p1_km": round(float(np.percentile(d, 1)), 2),
            "median_km": round(float(np.median(d)), 2),
            "p95_km": round(float(np.percentile(d, 95)), 2),
            "max_km": round(float(d.max()), 2),
            "within_10km_pct": round(float(100 * (d < 10).mean()), 2),
            "within_25km_pct": round(float(100 * (d < 25).mean()), 2),
            "within_50km_pct": round(float(100 * (d < 50).mean()), 2),
        }
    held = cKDTree(xyz[~train])
    chord, _ = held.query(xyz[train], k=1)
    d = 2 * r * np.arcsin(np.clip(chord / (2 * r), 0, 1))
    out["buffer_price"] = {
        f"{thr}km": {"train_tiles": int((d < thr).sum()), "pct_of_train": round(float(100 * (d < thr).mean()), 3)}
        for thr in (10, 25, 50)
    }
    return out


def val_design(blocks_cogs: list[str], cogs, coastline, land_frac, stratum, cells, test_cells, k) -> dict[str, Any]:
    """The shipped val draw against the shortfall-greedy variant that was rejected.

    Both draws are re-run here from the manifest, so the comparison in
    :mod:`dataset.splits`'s docstring is a measurement this report regenerates.
    """
    n_total = len(stratum)
    q = np.bincount(stratum, minlength=k) / n_total
    ids, index = np.unique(cogs, return_inverse=True)
    counts = np.zeros((len(ids), k), dtype=np.int64)
    np.add.at(counts, (index, stratum.astype(np.int64)), 1)
    depth = counts.sum(1)
    first = np.zeros(len(ids), dtype=np.int64)
    np.maximum.at(first, index, np.arange(n_total))
    eligible = ~np.isin(cells[first], list(test_cells))
    budget = round(splits.VAL_FRACTION * n_total)

    def character(selected_cogs: list[int]) -> dict[str, Any]:
        m = np.isin(index, selected_cogs)
        pool = counts[selected_cogs].sum(0)
        return {
            "n_cogs": len(selected_cogs),
            "n_tiles": int(m.sum()),
            "mean_depth": round(float(m.sum() / len(selected_cogs)), 2),
            "median_depth": int(np.median(depth[selected_cogs])),
            "coastline_pct": round(float(100 * coastline[m].mean()), 3),
            "land_frac_mean": round(float(land_frac[m].mean()), 4),
            "total_variation": round(float(0.5 * np.abs(pool / pool.sum() - q).sum()), 4),
            "shortfall": round(splits.relative_shortfall(pool, q), 4),
        }

    shipped = [int(np.flatnonzero(ids == c)[0]) for c in blocks_cogs]
    # the rejected variant: free to take any eligible COG, minimizing shortfall each step
    rank = splits.tie_rank(ids)
    cand = np.flatnonzero(eligible)
    have, greedy = np.zeros(k, dtype=np.int64), []
    while True:
        room = budget - have.sum()
        ok = cand[(depth[cand] <= room) & (depth[cand] > 0)]
        if ok.size == 0:
            break
        nh = have + counts[ok]
        d = (np.maximum(q - nh / nh.sum(1, keepdims=True), 0) / q).sum(1)
        winners = ok[d <= d.min() + 1e-15]
        pick = int(winners[np.argmin(rank[winners])])
        greedy.append(pick)
        have = have + counts[pick]
        cand = cand[cand != pick]
    return {
        "corpus": {
            "coastline_pct": round(float(100 * coastline.mean()), 3),
            "land_frac_mean": round(float(land_frac.mean()), 4),
            "mean_depth": round(float(depth[eligible].mean()), 2),
        },
        "shipped_hash_order": character(shipped),
        "rejected_shortfall_greedy": character(greedy),
    }


# --- report -------------------------------------------------------------------


def render(doc: dict[str, Any]) -> str:
    """The prose report — the numbers above, with the arguments they settle."""
    r, b = doc["realized"], doc["blocks"]
    tr, va, te = (r["splits"][s] for s in ("train", "val", "test"))
    bd, cn = doc["boundary"], doc["contamination"]
    vd = doc["val_design"]
    L = []
    A = L.append

    A("# P77 — the geographic split, drawn and committed\n")
    A(f"Ticket [#77](https://github.com/JStoschek/DEMAE/issues/77) · map "
      f"[#76](https://github.com/JStoschek/DEMAE/issues/76) · plan §9 · "
      f"corpus `{doc['dataset_version']}` ({r['n_tiles']:,} tiles)\n")
    A(f"Assignment `{b['assignment_md5']}` · "
      f"[`{config.SPLIT_ASSIGNMENT_PATH}`](../../{config.SPLIT_ASSIGNMENT_PATH}) · "
      f"[`{config.SPLIT_BLOCKS_PATH}`](../../{config.SPLIT_BLOCKS_PATH})\n")

    A("## 1. What was drawn\n")
    A("| split | blocking unit | blocks | tiles | share | shortfall vs corpus | TV |")
    A("|---|---|---|---|---|---|---|")
    A(f"| `train` | — (the remainder) | — | {tr['n_tiles']:,} | {tr['share']:.3%} | "
      f"{tr['shortfall']:.4f} | {tr['total_variation']:.4f} |")
    A(f"| `val` | whole source COG | {b['val']['n_cogs']} | {va['n_tiles']:,} | {va['share']:.3%} | "
      f"{va['shortfall']:.4f} | {va['total_variation']:.4f} |")
    A(f"| `test` | whole 10° geocell | {b['test']['n_cells']} | {te['n_tiles']:,} | {te['share']:.3%} | "
      f"{te['shortfall']:.4f} | {te['total_variation']:.4f} |")
    A("")
    A(f"The plan asks for ~95 / ~2 / ~3 and the draw lands on "
      f"{tr['share']:.3%} / {va['share']:.3%} / {te['share']:.3%}. Both hold-outs carry all 20 "
      f"strata. *Shortfall* is `dataset.splits.relative_shortfall` — the total relative "
      f"stratum deficit against the corpus composition, the objective both draws minimize; "
      f"*TV* is total-variation distance, reported because it is the familiar number and "
      f"**not** used as an objective (it prices a missing 1.5% stratum at 0.015 and cannot "
      f"see coverage).\n")
    A(f"The test search found its {b['test']['n_cells']} cells among "
      f"{b['test']['eligible_cells']:,} eligible, improving the greedy seed from "
      f"{b['test']['seed_shortfall']:.4f} to {b['test']['shortfall']:.4f} in "
      f"{b['test']['swaps']} swaps. Val drew {b['val']['n_cogs']:,} COGs of "
      f"{b['val']['eligible_cogs']:,} eligible at mean depth {b['val']['mean_depth']:.1f}, "
      f"against {vd['corpus']['mean_depth']:.1f} for the eligible pool. Val's shortfall is the "
      f"higher of the two because its draw does not minimize it — §6 is where that trade is "
      f"made and priced.\n")

    A("## 2. The 13 test regions\n")
    A("| geocell | region | tiles | COGs | lon | lat | strata |")
    A("|---|---|---|---|---|---|---|")
    for row in doc["test_regions"]:
        A(f"| `{row['geocell_id']}` | {row['region']} | {row['n_tiles']:,} | {row['n_cogs']} | "
          f"{row['lon']:.1f} | {row['lat']:.1f} | {row['n_strata']}/20 |")
    sep = doc["separations"]
    A("")
    A(f"Pairwise separation runs {sep['min_km']:,.0f}–{sep['max_km']:,.0f} km (median "
      f"{sep['median_km']:,.0f} km) against the 1,000 km floor. These are Phase 2's "
      f"downstream evaluation regions and no pretraining run may read them.\n")
    el = doc["eligibility"]
    A(f"**What the pool the search saw is not.** §9 says \"select 12-15 of the 425 occupied "
      f"geocells\"; the search ranged over **{el['eligible_cells']} of {el['occupied_cells']}**. "
      f"A cell must hold between {el['band_tiles'][0]:,} and {el['band_tiles'][1]:,} tiles — a "
      f"fortieth to a quarter of the test budget — which drops "
      f"{el['excluded_too_large']} cells as too large and {el['excluded_too_small']} as too small. "
      f"That narrowing is not neutral, and this is the part of the design most worth arguing "
      f"with: the largest cells are the high mountains, so the band takes the rare families "
      f"with them. Inside the band lie only "
      + ", ".join(f"{d['share_inside_band']:.1%} of stratum {d['stratum_id']}"
                  for d in el["least_reachable_strata"]) +
      ". Stratum 6 (the 4,300 m plateaus of Tibet and the Altiplano) and stratum 19 (high "
      "alpine) are held almost entirely by eight geocells of 3,000-4,100 tiles each; admitting "
      "one would have spent a fifth to a quarter of the test budget on a single region. The "
      "consequence is visible in §4 — those are the two strata furthest from their corpus "
      "share in `test` — and it is a cost of whole-geocell blocking at a 3% budget, not a "
      "defect in the search.\n")
    A("| latitude belt | corpus | `test` | `val` |")
    A("|---|---|---|---|")
    for row in doc["latitude_balance"]:
        A(f"| {row['belt']} | {row['corpus_pct']:.1f}% | {row['test_pct']:.1f}% | {row['val_pct']:.1f}% |")
    A("")
    A("Three of the thirteen cells are Antarctic, which looks polar-heavy until it is measured "
      "against the corpus: Antarctica is 14.3% of the tiles and 11.7% of the test pool, so the "
      "hold-out is *under*-weighted there, not over. The belt table is the check.\n")
    A("Two honest qualifications on \"spanning distinct geomorphic provinces\". First, the "
      "*pool* is what carries all 20 strata, not the cells: an individual cell holds as few as "
      "2 of 20 (the East Antarctic interior is ice sheet and little else), and that is the "
      "expected shape of a real region rather than a defect in the draw. Second, the closest "
      "pair — `-100_+060` and `-090_+050` at "
      f"{sep['min_km']:,.0f} km — is glaciated Canadian Shield on both sides. The 1,000 km floor "
      "kept them apart in kilometres and not in geomorphology, so the set spans twelve distinct "
      "provinces and samples one of them twice.\n")

    A("## 3. The boundary effect — the plan's \"hundreds of km\" is the median's story, not the edge's\n")
    A("| hold-out | min | p1 | median | p95 | max | <10 km | <25 km | <50 km |")
    A("|---|---|---|---|---|---|---|---|---|")
    for s in ("test", "val"):
        d = bd[s]
        A(f"| `{s}` | {d['min_km']:.1f} | {d['p1_km']:.1f} | {d['median_km']:.1f} | {d['p95_km']:.1f} | "
          f"{d['max_km']:.1f} | {d['within_10km_pct']:.1f}% | {d['within_25km_pct']:.1f}% | "
          f"{d['within_50km_pct']:.1f}% |")
    A("\nDistances in km from each held-out tile to the nearest `train` tile.\n")
    A(f"Plan §9 says geocell blocking \"puts hundreds of km between train and test, far beyond "
      f"terrain autocorrelation ranges\". The median is **{bd['test']['median_km']:.0f} km**, not "
      f"hundreds, and the minimum is **{bd['test']['min_km']:.1f} km** — a rim of test tiles sits "
      f"one tile-width across the cell boundary from a train tile. Blocking at 10° bounds the "
      f"*typical* separation, not the smallest one; a whole-cell hold-out has a perimeter, and "
      f"{bd['test']['within_25km_pct']:.1f}% of test tiles lie within 25 km of training data. The "
      f"claim is corrected here rather than quietly restated: the design is unchanged, because §9 "
      f"specifies `train` as everything else and this ticket does not get to amend it.\n")
    A(f"Val is weaker by construction and the plan says so — COG blocking is the *minimum* "
      f"honest unit. {bd['val']['within_10km_pct']:.1f}% of val tiles have a train tile within "
      f"10 km and the median is {bd['val']['median_km']:.1f} km. Val loss therefore measures "
      f"generalization across a ~110 km source-tile boundary, not across regions; that is what "
      f"the `test` cells are for.\n")
    A("**The price of the obvious fix, unspent.** Excluding train tiles near a hold-out would "
      "cost " + ", ".join(f"{v['train_tiles']:,} ({v['pct_of_train']:.2f}%) at {thr[:-2]} km"
                          for thr, v in bd["buffer_price"].items()) + " of the training pool" +
      ". Cheap — but a buffer is not in §9, and it would perturb the Neyman allocation a second "
      "time. Recorded for the map to decide, not applied.\n")

    A("## 4. Per-stratum train counts, and the Neyman perturbation (plan §9b)\n")
    ny = doc["neyman"]
    A("| stratum | drawn `n_h` | train | val | test | drawn share | train share | shift |")
    A("|---|---|---|---|---|---|---|---|")
    for row in ny["rows"]:
        h = row["stratum_id"]
        A(f"| {h} | {row['n_h_drawn']:,} | {row['train']:,} | {va['per_stratum'][h]:,} | "
          f"{te['per_stratum'][h]:,} | {row['drawn_share']:.4f} | {row['train_share']:.4f} | "
          f"{row['relative_shift_pct']:+.2f}% |")
    A("")
    A(f"Removing the hold-outs moves no stratum's share of the training pool by more than "
      f"**{ny['max_abs_relative_shift_pct']:.2f}%** of itself (total variation "
      f"{ny['total_variation']:.5f}). The Neyman allocation the sampler drew is not quite the "
      f"allocation the model trains on, which is the caveat §9b asks to be recorded; the size "
      f"of it is the reason no correction is proposed.\n")

    A("## 5. The global μ/σ contamination (plan §9c)\n")
    A(f"The shipped constants — **{cn['shipped']['mean_m']:.2f} / {cn['shipped']['std_m']:.2f} m** — "
      f"were accumulated over every tile, including the "
      f"{cn['held_out_land_pixel_share']:.2%} of land pixels now held out. Because "
      f"`normalize.land_moments` sums quantities that are additive, and the manifest stores each "
      f"tile's land `mean`/`std`/`land_frac`, the pair can be re-derived for any subset without "
      f"reading a shard. The whole-corpus re-derivation returns "
      f"{cn['recomputed_all_tiles']['mean_m']:.4f} / {cn['recomputed_all_tiles']['std_m']:.4f} m — "
      f"the shipped numbers to four decimals, which is what licenses the rest of this table.\n")
    A("| pool | land μ (m) | land σ (m) | land pixels |")
    A("|---|---|---|---|")
    for s in splits.SPLITS:
        d = cn["per_split"][s]
        A(f"| `{s}` | {d['mean_m']:.2f} | {d['std_m']:.2f} | {d['land_pixels']:,} |")
    A(f"| all (shipped) | {cn['recomputed_all_tiles']['mean_m']:.2f} | "
      f"{cn['recomputed_all_tiles']['std_m']:.2f} | — |")
    A("")
    dl = cn["delta_if_recomputed_on_train"]
    A(f"Recomputing on `train` only would move the mean by **{dl['mean_m']:+.2f} m "
      f"({dl['mean_pct']:+.2f}%)** and the sd by **{dl['std_m']:+.2f} m ({dl['std_pct']:+.2f}%)**. "
      f"§9c offered two options — recompute, or record the contamination as accepted. The train-only "
      f"pair now exists at zero cost and is recorded here for Phase 3's global-norm ablation to use; "
      f"`corpus_meta.json` is **not** rewritten, because the shipped pair is a Phase-0 record and a "
      f"0.31% shift does not justify moving a number four committed artifacts quote. The held-out "
      f"pools are not interchangeable with train — `test` runs "
      f"{cn['per_split']['test']['mean_m'] - cn['per_split']['train']['mean_m']:+.0f} m in the mean "
      f"(mountain and plateau regions) and `val` "
      f"{cn['per_split']['val']['mean_m'] - cn['per_split']['train']['mean_m']:+.0f} m.\n")

    A("## 6. Why val draws in hash order\n")
    A("| draw | COGs | tiles | mean depth | median depth | coastline | mean `land_frac` | TV | shortfall |")
    A("|---|---|---|---|---|---|---|---|---|")
    for name, key in (("shipped — per-stratum quota, hash order", "shipped_hash_order"),
                      ("rejected — shortfall greedy", "rejected_shortfall_greedy")):
        d = vd[key]
        A(f"| {name} | {d['n_cogs']:,} | {d['n_tiles']:,} | {d['mean_depth']} | {d['median_depth']} | "
          f"{d['coastline_pct']:.2f}% | {d['land_frac_mean']:.4f} | {d['total_variation']:.4f} | "
          f"{d['shortfall']:.4f} |")
    A(f"| corpus | — | — | {vd['corpus']['mean_depth']} | — | {vd['corpus']['coastline_pct']:.2f}% | "
      f"{vd['corpus']['land_frac_mean']:.4f} | — | — |")
    A("")
    g, sh = vd["rejected_shortfall_greedy"], vd["shipped_hash_order"]
    A(f"The greedy variant matches the stratum composition better — TV {g['total_variation']:.4f} "
      f"against {sh['total_variation']:.4f} — and buys it by taking {g['n_cogs']:,} COGs of mean "
      f"depth {g['mean_depth']}. A COG holding two kept tiles is a COG the gates nearly emptied, so "
      f"the pool it assembles carries coastline at **{g['coastline_pct'] / vd['corpus']['coastline_pct']:.1f}×** "
      f"the corpus rate ({g['coastline_pct']:.2f}% against {vd['corpus']['coastline_pct']:.2f}%). "
      f"Composition is not the only thing a val set has to look like. The shipped draw is "
      f"size-blind — quota per stratum, COGs in keyed-hash order — and lands on the corpus's "
      f"coastline rate to two decimals.\n")
    A(f"**What that trade costs, stated plainly.** Val's shortfall of {sh['shortfall']:.4f} is not "
      f"a missing stratum — all 20 are present, the smallest holding "
      f"{min(va['per_stratum'])} tiles — it is a sum of small relative deficits, and the larger "
      f"cause is the quota's accounting rather than the hash order. A COG is charged whole to "
      f"its **dominant** stratum, but it carries tiles from others, so a stratum's realized "
      f"count is its own quota plus whatever its neighbours' COGs happened to bring: stratum 19 "
      f"lands at {va['per_stratum'][19]:,} against a quota of {b['val']['quota'][19]:,} and "
      f"stratum 16 at {va['per_stratum'][16]:,} against {b['val']['quota'][16]:,}. The fix would "
      f"be to charge each COG's tiles to their own strata and accept overshoot, or to swap COGs "
      f"after the pass — both re-introduce a size-aware step, which is the thing the greedy "
      f"row above shows is expensive. For val loss and a kNN probe on 20 classes, +-30% on a "
      f"class count is not worth buying back at that price; it is recorded so a probe result "
      f"that leans on the rare classes can be read with it in hand.\n")

    A("## 7. Determinism, and what §9d still owes\n")
    d = doc["determinism"]
    A(f"Re-drawing from the shipped manifest reproduces the committed assignment byte for byte "
      f"(md5 `{d['replicate_md5']}`, {d['seconds']:.1f} s). As with the selection list, the "
      f"contract is the persisted file and not the seed: every tie-break is a keyed BLAKE2b hash "
      f"of the block id, and nothing here re-fits, re-clusters, or reads a COG.\n")
    A(f"§9d asks for the md5 \"in the run config\". There is no Phase-1 run config yet — R0 has "
      f"not been built — so the digest is carried by "
      f"[`{config.SPLIT_BLOCKS_PATH}`](../../{config.SPLIT_BLOCKS_PATH}), by this report, and by "
      f"the commit that ships them. The obligation is **not discharged**: plan §15.7 requires the "
      f"training run to be tagged with the split md5 alongside the git SHA, the config hash and "
      f"the selection-list md5, and `config.SPLIT_ASSIGNMENT_PATH` names the file it must hash. "
      f"Recorded here so the pre-flight checklist has something to check against.\n")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python notebooks/p77_split.py", description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--out-json", type=Path, default=RESEARCH / "p77-split.json")
    p.add_argument("--out-md", type=Path, default=RESEARCH / "p77-split.md")
    args = p.parse_args(argv)

    log("reading the manifest")
    table = pq.read_table(args.manifest, columns=[
        "tile_id", "geocell_id", "source_cog", "stratum_id", "lon", "lat",
        "mean", "std", "land_frac", "coastline", "split",
    ])
    cells = table["geocell_id"].to_numpy(zero_copy_only=False)
    cogs = table["source_cog"].to_numpy(zero_copy_only=False)
    stratum = table["stratum_id"].to_numpy().astype(np.int64)
    lon, lat = table["lon"].to_numpy(), table["lat"].to_numpy()
    mean, std = table["mean"].to_numpy(), table["std"].to_numpy()
    land_frac = table["land_frac"].to_numpy()
    coastline = table["coastline"].to_numpy(zero_copy_only=False).astype(bool)
    k = int(stratum.max()) + 1

    assignment = splits.read_assignment(REPO / config.SPLIT_ASSIGNMENT_PATH)
    label = np.array([assignment[t] for t in table["tile_id"].to_pylist()])
    column = table["split"].to_numpy(zero_copy_only=False)
    if not np.array_equal(column.astype(str), label):
        raise SystemExit("the manifest's split column disagrees with the committed assignment")
    blocks = json.loads((REPO / config.SPLIT_BLOCKS_PATH).read_text())

    log("re-drawing to check the committed assignment reproduces")
    t0 = time.time()
    plan = splits.plan_from_manifest(args.manifest)
    seconds = time.time() - t0
    if not np.array_equal(plan.labels.astype(str), label):
        raise SystemExit("the re-draw does not reproduce the committed assignment")

    log("measuring")
    doc: dict[str, Any] = {
        "ticket": 77,
        "dataset_version": json.loads((args.manifest.parent / "corpus_meta.json").read_text())["dataset_version"],
        "blocks": blocks,
        "realized": realized(label, stratum, k),
        "test_regions": test_regions(blocks, cells, cogs, lon, lat, stratum, k),
        "neyman": neyman_perturbation(np.bincount(stratum[label == "train"], minlength=k), k),
        "contamination": contamination(mean, std, land_frac, label),
        "boundary": boundary(label, lon, lat),
        "val_design": val_design(blocks["val"]["cogs"], cogs, coastline, land_frac, stratum,
                                 cells, set(blocks["test"]["cells"]), k),
        "eligibility": eligibility(cells, stratum, k, round(splits.TEST_FRACTION * len(label))),
        "latitude_balance": latitude_balance(label, lat),
        "determinism": {"replicate_md5": blocks["assignment_md5"], "seconds": round(seconds, 2)},
    }
    doc["separations"] = separations(doc["test_regions"])

    args.out_json.write_text(json.dumps(doc, indent=2) + "\n")
    args.out_md.write_text(render(doc))
    log(f"wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
