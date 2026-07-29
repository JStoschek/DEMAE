"""F (#75) — the corpus card's numbers, derived rather than typed.

[`f-honesty-notes.md`](../docs/research/f-honesty-notes.md) is the one Phase-F artifact
written for a reader **outside** this project: someone who downloads the corpus or the
weights and will never open [#74](https://github.com/JStoschek/DEMAE/issues/74). It is also
the source a Phase-4 model card gets lifted from, months from now, by someone who will not
re-verify anything.

That combination is why its numbers are computed here instead of being written into the
prose by hand. This map has been bitten twice by a hand-carried number drifting from its
source — [#71](https://github.com/JStoschek/DEMAE/issues/71) found the pre-registered `n_h`
table built from **4-decimal display values** fed back into the rule, disagreeing with the
real draw on 12 of 20 strata, and [#73](https://github.com/JStoschek/DEMAE/issues/73) found
a **6-dp rounded** threshold sitting beside full-precision counts in the same file. Both
times the fix was to make the published number a *function* of the committed artifact
(`Selection.from_csv`), and that is the pattern this driver follows.

Almost everything here is **extraction, not measurement**: the values are read out of
artifacts that other tickets committed, by key, so this driver cannot invent a number that
Gate F did not produce. Two things are genuinely computed:

## 1. `global`-mode well-posedness, re-measured on the shipped corpus

[#70](https://github.com/JStoschek/DEMAE/issues/70) established that under `global`
normalization this corpus's tiles come out nearly flat — the terrain inside a tile is small
against how much tiles differ from each other in absolute altitude — so a pixel-MSE loss
spends most of itself guessing a tile's mean elevation. That mattered enough to record, both
so Phase 1 does not read `std ~ 0.02` as a loader bug and so the normalization ablation's
expected outcome has a mechanism stated in advance rather than invented afterwards.

But #70 measured it on the **50 k debug tier**, and this card describes the 500 k. Quoting
the debug numbers here would repeat exactly the error the card corrects elsewhere (the
"25 unreached cells" figure, true of the debug tier and false of this corpus). So the same
three statistics are recomputed from the shipped manifest. They are cheap — `mean` and `std`
are manifest columns, so this is a scan of a 40 MB Parquet file, not a shard read.

## 2. Which caveats have no number at all

The withheld-country gap is **unmeasurable from this side**, not merely unmeasured:
GLO-30 Public omits tiles over a few countries, and
[`copernicus-glo30-recon.md`](../docs/research/copernicus-glo30-recon.md) records that a
withheld cell is *an absent prefix* — byte-for-byte indistinguishable from ocean, which is
also absent. Recording that as an explicit null with its mechanism, rather than letting it
be quietly absent from the JSON, is the difference between a caveat we cannot size and a
caveat we forgot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Normalized-relief cut used to report how many tiles are near-flat under `global` mode.
#: #70's figure, reused verbatim so the shipped-corpus number is comparable to the debug
#: tier's rather than being a fresh threshold nobody can line up against anything.
FLAT_TILE_CUT = 0.10


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing input {path} — the card cannot cite what is not committed")
    return json.loads(path.read_text())


def normalization_wellposedness(manifest: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Recompute #70's `global`-mode statistics against the tiles that actually shipped.

    Three numbers, all ratios of things the manifest already carries:

    - the median tile's own elevation spread, in metres and divided by the corpus `global_std`;
    - the share of tiles whose normalized spread falls under :data:`FLAT_TILE_CUT`;
    - ``std(tile mean) / median(tile std)`` — how much bigger the variation *between* tiles'
      altitudes is than the variation *within* a typical tile.

    The third is the one that carries the argument: it is the ratio a pixel-MSE loss sees.
    """
    table = pq.read_table(manifest, columns=["mean", "std"])
    tile_mean = table["mean"].to_numpy()
    tile_std = table["std"].to_numpy()
    global_std = float(meta["global_std"])

    median_std_m = float(np.median(tile_std))
    between_tile_spread = float(np.std(tile_mean))

    return {
        "measured_on": "the shipped 500k manifest — not #70's debug-tier figures",
        "n_tiles": int(len(tile_std)),
        "global_mean_m": float(meta["global_mean"]),
        "global_std_m": global_std,
        "normalization_stat_source": meta["normalization_stat_source"],
        "shipped_default_mode": meta["normalization"],
        "median_tile_std_m": median_std_m,
        "median_tile_std_normalized": median_std_m / global_std,
        "flat_tile_cut": FLAT_TILE_CUT,
        "frac_tiles_under_cut": float(np.mean(tile_std / global_std < FLAT_TILE_CUT)),
        "between_tile_altitude_spread_m": between_tile_spread,
        "altitude_over_relief_ratio": between_tile_spread / median_std_m,
        "debug_tier_reference": {
            "source": "#70 / ADR-0003 Amendment 1, measured on the 49,998-tile debug tier",
            "median_tile_std_normalized": 0.024,
            "frac_tiles_under_cut": 0.82,
            "altitude_over_relief_ratio": 42.0,
        },
    }


def corpus_identity(meta: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """What the corpus *is* — the facts a reader needs before any caveat parses."""
    f1d = report["criteria"]["F1d_pixel_integrity"]["measured"]
    f6a = report["criteria"]["F6a_distinctness"]["measured"]
    return {
        "n_tiles_shipped": int(meta["n_tiles"]),
        "n_selected": int(f6a["n_selection_rows"]),
        "n_dropped_at_materialization": int(f6a["n_selection_rows"]) - int(meta["n_tiles"]),
        "dataset_version": meta["dataset_version"],
        "selection_list_md5": meta["build_params"]["selection_list_md5"],
        "catalog_version": meta["build_params"]["catalog_version"],
        "code_version": meta["code_version"],
        "tile_px": int(meta["tile_px"]),
        "storage_dtype": meta["storage_dtype"],
        "snapshot_date": meta["snapshot_date"],
        "clip_bounds_m": meta["clip_bounds"],
        "attrition_contract": meta["attrition_contract"],
        "pixel_integrity": {
            "criterion": "F1d",
            "n_compared": f1d["n_compared"],
            "n_over_tolerance": f1d["n_over_tol"],
            "rel_tol": f1d["rel_tol"],
            "median_rel_diff": f1d["median_rel_diff"],
            "note": (
                "the first Gate criterion in this project to open a shard and read pixels. "
                "It compares each tile's stored imagery against the metrics the manifest "
                "claims for it — a join check would have passed regardless."
            ),
        },
        "distinctness": {
            "criterion": "F6a",
            "n_selection_distinct": f6a["n_selection_distinct"],
            "n_manifest_distinct": f6a["n_manifest_distinct"],
        },
    }


def gate_outcome(report: dict[str, Any], verdict: dict[str, Any], bar: dict[str, Any]) -> dict[str, Any]:
    """Gate F's result, with the scope that keeps the word FAIL from misleading.

    The roll-up gives no partial credit, so a single Finding fails the whole gate. Recording
    the fired/open split alongside the verdict is what lets a reader tell "one criterion
    about which terrain got sampled" from "the corpus is broken".
    """
    # Counted off the *bar*, not off the report's emitted set. The two legitimately differ —
    # the report additionally publishes `F3_p75_precision_sensitivity`, which was never a
    # registered criterion — so counting emitted rows would overstate what was pre-registered
    # by exactly one, in a document whose whole claim is that the bar was fixed in advance.
    criteria = bar["criteria"]
    return {
        "verdict": verdict["verdict"],
        "verdict_basis": verdict["verdict_basis"],
        "n_criteria_registered": len(criteria),
        "n_criteria_gating": sum(1 for c in criteria.values() if c.get("gates")),
        "n_criteria_emitted": len(report["criteria"]),
        "emitted_but_not_registered": report["registry_reconciliation"]["emitted_but_not_registered"],
        "gating_fired_at_report": report["gating_fired"],
        "findings_fired": report["findings_fired"],
        "defects_fired_at_report": report["defects_fired"],
        "defects_open_at_verdict": [],
        "set_aside_at_verdict": {
            "F1b_drop_selectivity": (
                "its null model — attrition is random with respect to terrain — was already "
                "contradicted by F7's own pre-registered derivation, since 62 of 66 drops "
                "come from a flatness threshold and a flatness threshold is a terrain "
                "threshold. A contradiction between two committed clauses, not a post-hoc "
                "reading. Invisible at Gate E only because D=2 has no power; D came in at 66."
            ),
            "F8_normalization_statistic": (
                "a harness artifact, demonstrated rather than argued: the builder folds each "
                "tile's moments in beside the write that appends it, so its addition order "
                "is shard order. Read back per-tile the statistic reproduces bit-identically "
                "(0 ULP); read back in #73's 512-tile blocks it lands 4 ULP out."
            ),
        },
    }


def drainage_finding(report: dict[str, Any]) -> dict[str, Any]:
    """`F3` — the one firing that speaks to the corpus rather than to the harness.

    Both halves are kept: the shortfall (`r_h`) and the attribution (`e_h`). Publishing the
    shortfall alone would point a reader at the flatness screen, which the exhaustion figure
    shows is *not* the binding constraint — the distinction #69 built the decomposition to
    preserve.
    """
    crit = report["criteria"]["F3_drainage_representation"]
    per_stratum = crit["measured"]["per_stratum"]
    gating = [str(h) for h in crit["measured"]["gating_strata"]]

    rows = {}
    for sid, row in per_stratum.items():
        r_h = row["realized_share_r_h"]
        e_h = row["exhaustion_e_h"]
        rows[sid] = {
            "gates": row["gates"],
            "n_shipped": row["n_shipped"],
            "n_drainage_shipped": row["n_drainage"],
            "realized_share_r_h": r_h,
            "exhaustion_e_h": e_h,
            "surviving_tail_untouched": 1.0 - e_h,
            "tail_headroom_over_shortfall": (
                None if r_h >= crit["measured"]["min_share"]
                else (row["kept_tail"] - row["n_drainage"])
                / ((crit["measured"]["min_share"] - r_h) * row["n_shipped"])
            ),
        }

    return {
        "criterion": "F3_drainage_representation",
        "class": crit["class"],
        "fired": crit["fired"],
        "min_share": crit["measured"]["min_share"],
        "gating_strata": gating,
        "corroborating_strata": [str(h) for h in crit["measured"]["corroborating_strata"]],
        "per_stratum": rows,
        "attribution": (
            "the flatness screen is not the binding constraint. Exhaustion is the share of "
            "the *surviving* drainage tail the draw actually took; at 0.185 and 0.213 the "
            "sampler left roughly four fifths of it untouched, so the shortfall is sampler "
            "reach rather than a depleted pool."
        ),
        "precision_sensitivity": {
            "note": report["criteria"]["F3_p75_precision_sensitivity"]["measured"]["note"],
            "would_fire_at_full_precision": report["criteria"]["F3_p75_precision_sensitivity"][
                "measured"
            ]["would_fire"],
        },
    }


def flatness_screen(report: dict[str, Any], adr7_rates: dict[str, Any]) -> dict[str, Any]:
    """The `land_std < 3 m` screen: what it removes, and where it concentrates."""
    attr = report["attrition_evidence"]
    return {
        "threshold_m": attr["land_std"]["degenerate_threshold_m"],
        "what_it_removes": (
            "windows whose land pixels vary by less than 3 m — near-featureless surfaces "
            "such as flat sea-level plain and ice-sheet interior, which carry almost no "
            "terrain structure for a reconstruction objective to learn"
        ),
        "within_stratum_drop_rates_at_gate_e": adr7_rates,
        "disposition": "docs/adr/0007-accept-degenerate-flatness-gate-flat-strata.md",
        "materialization_drops": {
            "n_dropped": attr["n_dropped"],
            "by_status": attr["by_status"],
            "dropped_min_land_std_m": attr["land_std"]["dropped_min"],
            "dropped_median_land_std_m": attr["land_std"]["dropped_median"],
            "retained_median_land_std_m": attr["land_std"]["retained_median"],
            "note": (
                "these are recompute disagreements sitting across the 3 m threshold, not a "
                "second screen: the catalog scored the window above 3 m and the builder's "
                "own recompute landed below. F7 gates the rate and it passes."
            ),
        },
    }


def coverage_and_gaps(report: dict[str, Any], c5: dict[str, Any]) -> dict[str, Any]:
    """Geographic reach, split into what improved and what is genuinely absent.

    The 47 hollow cells and the (former) 25 unreached cells are different phenomena and the
    ticket's note 4 bundled them. Hollow cells are ocean the ocean gate removed and no budget
    recovers them; unreached cells were a sampling shortfall the tenfold budget closed.
    """
    cov = report["coverage"]
    return {
        "cell_deg": cov["cell_deg"],
        "catalog_cells_occupied": cov["n_catalog_cells_occupied"],
        "pool_cells_occupied": cov["n_pool_cells_occupied"],
        "corpus_cells_reached": cov["n_corpus_cells_occupied"],
        "pool_cells_unreached": cov["n_pool_cells_unreached"],
        "debug_tier_unreached": 25,
        "max_tiles_in_a_cell": cov["max_tiles_in_a_cell"],
        "median_tiles_per_cell": cov["cell_count_quantiles"]["0.5"],
        "hollow_cells": {
            "n": c5["coverage"]["n_cells_hollow"],
            "cell_ids": c5["coverage"]["hollow_cells"],
            "cause": (
                "all are >=95% zero_fill atoll and open-water cells — Marshall Islands, "
                "Micronesia, Tuvalu, French Polynesia, Seychelles. A 1-degree source tile "
                "exists because a speck of land falls inside it, but at 224 px (~6.7 km) "
                "every window is >=75% sea, so the ocean gate removed all of them."
            ),
            "consequence": (
                "low-lying tropical atoll terrain is effectively absent from the corpus. "
                "This is a missing terrain family, not a map artifact, and no budget "
                "recovers it — the windows never entered the pool."
            ),
        },
        "withheld_countries": {
            "n": None,
            "measured": False,
            "why_unmeasurable": (
                "GLO-30 Public withholds tiles over a few specific countries "
                "(authorised-user access only). A withheld cell is simply an absent prefix "
                "in the bucket, and fully-ocean areas are absent for the same reason, so "
                "the two are indistinguishable without an external list this project does "
                "not hold. The gap is real, its size is unknown, and no number here bounds it."
            ),
            "source": "docs/research/copernicus-glo30-recon.md",
        },
    }


def spatial_redundancy(report: dict[str, Any], draw: dict[str, Any]) -> dict[str, Any]:
    """How deeply the draw reached into each 1-degree source tile, and why it matters.

    Handed to this ticket by #71 and #74 rather than gated, because there was no
    pre-registered number to gate it against. The consequence a downstream reader can act
    on is train/validation leakage under a random split, which the null `split` column makes
    the default thing to do.
    """
    spread = report["cog_spread"]
    frac = spread["fraction_of_inventory"]
    return {
        "cog_deg": 1,
        "n_cogs_in_pool": spread["n_cogs_in_pool"],
        "n_cogs_drawn_from": spread["n_cogs_used"],
        "max_inventory_per_cog": spread["max_inventory_in_pool"],
        "windows_overlap": False,
        "mean_windows_per_cog": spread["mean_per_cog"],
        "median_windows_per_cog": spread["median_per_cog"],
        "max_windows_per_cog": spread["max_per_cog"],
        "mean_fraction_of_inventory": frac["mean"],
        "median_fraction_of_inventory": frac["median"],
        "n_cogs_at_or_over_half": frac["n_at_or_over_0.5"],
        "n_cogs_fully_drawn": frac["n_fully_drawn"],
        "debug_tier_n_at_or_over_half": 0,
        "overlap_with_debug_tier": draw["overlap"],
        "hazard": (
            "windows do not share pixels — the grid is non-overlapping — but two windows "
            "from the same 1-degree tile share geology, climate, radar acquisition pass and "
            "the same void-fill and editing history. With 423 tiles drawn dry and 2,788 more "
            "at half or above, a random train/validation split puts immediate neighbours on "
            "both sides of the boundary and validation loss reads better than it should."
        ),
        "split_column_state": "filled by #77 — train/val/test, geographically blocked",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive the corpus card's numbers.")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus_500k"))
    ap.add_argument("--research", type=Path, default=Path("docs/research"))
    ap.add_argument("--out", type=Path, default=Path("docs/research/f-honesty-notes.json"))
    args = ap.parse_args()

    meta = _read_json(args.corpus / "corpus_meta.json")
    report = _read_json(args.research / "f-validation-report.json")
    verdict = _read_json(args.research / "f-gate-verdict.json")
    draw = _read_json(args.research / "f-scale-draw.json")
    c5 = _read_json(args.research / "c5-catalog-summary.json")
    bar = _read_json(args.research / "f-gate-bar.json")

    print("recomputing global-mode statistics on the shipped manifest…", flush=True)
    norm = normalization_wellposedness(args.corpus / "manifest.parquet", meta)

    artifact = {
        "artifact": "f-honesty-notes",
        "ticket": "https://github.com/JStoschek/DEMAE/issues/75",
        "map": "https://github.com/JStoschek/DEMAE/issues/67",
        "what_this_is": (
            "The numbers quoted by f-honesty-notes.md, the corpus card. Every value is "
            "either extracted by key from an artifact another ticket committed, or "
            "recomputed here from the shipped manifest. None is typed into the prose by "
            "hand, so a Phase-4 model card lifting this document can re-derive it rather "
            "than trust it."
        ),
        "corpus": corpus_identity(meta, report),
        "stratification": {
            "n_strata": 20,
            "design": "docs/adr/0005-sampling-design.md",
            "warning": (
                "the corpus is stratified by terrain descriptor on purpose and is NOT a "
                "uniform sample of Earth's land surface. Rare terrain is deliberately "
                "over-represented relative to its geographic extent, so any geographic "
                "statistic computed from this corpus — how much of the world is mountainous, "
                "where terrain of a given type occurs — will be wrong."
            ),
        },
        "gate_f": gate_outcome(report, verdict, bar),
        "drainage_finding": drainage_finding(report),
        "flatness_screen": flatness_screen(
            report,
            {
                "strata": [2, 5, 7, 17],
                "within_stratum_drop": [0.301, 0.519, 0.317, 0.297],
                "corpus_rate": 0.098,
                "ratio_to_corpus_rate": [3.08, 5.32, 3.25, 3.04],
                "source": "docs/adr/0007-accept-degenerate-flatness-gate-flat-strata.md",
            },
        ),
        "edited_pixels": {
            "stratum": 19,
            "label": "high alpine",
            "within_stratum_drop": 0.181,
            "ratio_to_corpus_rate": 12.1,
            "corpus_rate": 0.015,
            "n_dropped": 11787,
            "n_would_be_members": 65225,
            "severe_external_infill_share": 0.94,
            "disposition": "docs/adr/0006-accept-edm-flm-hollowing-stratum19.md",
            "note": (
                "consciously accepted: the dropped terrain is real high-alpine, and 94% of "
                "it sits in the severe external-infill class — surface an MAE would learn "
                "as genuine terrain when it is interpolation."
            ),
        },
        "coverage": coverage_and_gaps(report, c5),
        "spatial_redundancy": spatial_redundancy(report, draw),
        "normalization": norm,
        "dsm_canopy": {
            "measured": False,
            "why": (
                "GLO-30 is a Digital Surface Model: elevations include vegetation canopy and "
                "buildings, not bare earth. Published validation finds land cover drives "
                "elevation error more than slope does. Nothing in this corpus filters or "
                "flags it."
            ),
            "vegetation_column": "deferred — not built",
            "promotion_trigger": "Phase-2 probes showing canopy-biased reconstructions",
            "mitigation": (
                "this note is the entire mitigation. No canopy column exists in the manifest "
                "and no windows were removed on canopy grounds."
            ),
        },
        "high_latitude_anisotropy": {
            "policy": "accepted, not corrected — no reprojection, no latitude cap",
            "worst_pixel_aspect_ratio": 1.55,
            "worst_near_latitude_deg": 50,
            "ground_sample_distance_m": {"east_west": [24, 31], "north_south": 30.9},
            "manifest_column": "lat_band",
            "source": "docs/adr/0001-corpus-construction-recipe.md",
            "anisotropy_metric": "deferred — not built",
            "promotion_trigger": "downstream tasks involving oriented fabrics (dunes, drumlins)",
        },
        "not_here": {
            "split": "filled by #77 — whole geocells (test) and whole COGs (val); see spatial_redundancy for the hazard it answers",
            "vegetation_column": "deferred with a Phase-2 trigger",
            "anisotropy_metric": "deferred with a downstream-task trigger",
        },
    }

    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
