"""F (#73) — the Gate F validation report: the 500,000-window scale corpus, judged blind.

Phase F's Definition of Done on the numbers side, and the direct descendant of
[`e2_validation_report.py`](e2_validation_report.py). #72 turned the committed
500 k selection list into shards, a manifest and a ledger; this reads all three back
and asks, against a bar written *before* any of them existed, whether the scale corpus
faithfully implements its design and whether the design survives contact with it.

## It imports the bar and executes it — it does not negotiate with it

Every threshold comes off [`f-gate-bar.json`](../docs/research/f-gate-bar.json) through
[`dataset.validate_f`](../src/dataset/validate_f.py), which refuses to load a bar whose
prose has drifted from its constants. Nothing here chooses a criterion, moves a number,
or renders a verdict: the artifact carries ``verdict: null`` and the Defect-first /
any-Finding-fails roll-up is the Gate F review's
([#74](https://github.com/JStoschek/DEMAE/issues/74)).

**Gate E's module is not the one to import.** :mod:`dataset.validate` is frozen — ADR-0006
and ADR-0007 rest on its registered numbers — so this runs against
:mod:`dataset.validate_f`, which reuses E's *pure rule functions* under F's own constants
and relabels the carried-over outcomes so no Gate F artifact names ``E2.2c``.

## Where the registered phrasing met a different input at scale

Four places, named rather than silently re-interpreted (``read_bar``'s
refusal-on-drifted-prose is in force, and it is not the only way a bar can go stale):

1. **F2a's expected md5 is `null` in the bar.** #69 committed before #71 drew, and
   delegated the hash to "this file's sibling report". It is resolved from
   [`f-scale-draw.json`](../docs/research/f-scale-draw.json) and cross-checked against the
   md5 the *builder* consumed — see :func:`resolve_expected_md5`.
2. **F2a's provenance table is stated as a sum, not a table.** Gate E's bar carried
   ``{base, floor:coastline}`` explicitly; Gate F's says "both provenance tags summing to
   coast_floor = 6700", so the table is *derived* from ``budget`` and ``coast_floor``.
3. **The attrition contract is a committed input here, not a branch.** ``FBar`` has no
   ``live_branch`` on purpose — #50 settled it at Gate E. The corpus card is checked
   against the committed value rather than consulted for it.
4. **The per-COG spread tripwire named in #73's scope does not exist.** It was retired by
   #69 (a cap of 64 sits *inside* the healthy range at a 16.4 % draw) and replaced by
   ``F6b_cog_inventory_bound`` plus the published ``F6c_cog_spread_published``.

## Why this forks E2's driver where the gallery imports E2's

[`f_stratum_gallery.py`](f_stratum_gallery.py) imports its layout arithmetic from
[`e2_stratum_gallery.py`](e2_stratum_gallery.py) as a sibling module. This file deliberately
does the opposite with E2's loaders and figures, and the difference is not an oversight.

The gallery's sheets are a *presentation* of tiles, #75 will read the two tiers' sheets side
by side, and a second copy of the layout would drift into an unstated difference between
them — so there, sharing is the safe direction. Here it is not. Every helper below feeds a
number into a **pre-registered gate**, and
[`e2_validation_report.py`](e2_validation_report.py) is the producer of Gate E's committed
evidence: importing from it would mean a future edit made for Gate E's reasons could move a
Gate F measurement, which is the exact coupling
[`dataset/validate_f.py`](../src/dataset/validate_f.py) was made a sibling module to avoid.
The repo already treats driver-local loaders as the house pattern — `e2_validation_report`
and `e2_stratum_gallery` each define their own `load_manifest`, over different columns.

Where the two drivers do overlap the code is deliberately kept recognisable rather than
"improved", so a reader can diff them and see that the shared parts really are shared.

## What it reads, and what that costs

No sampling: the whole 4.39 M-row catalog, the whole 500,000-row selection list, all
499,934 tiles — and, for ``F8``, all 47 GB of shards, because "reproduces from the shipped
tiles" is a claim about pixels and there is no cheaper way to check one. Gate E never
opened a shard; this report opens every one of them twice over (``F1d`` reads the
manifest's pixel-derived ``std``, ``F8`` re-accumulates the land moments).

Zero-network; numpy + pyarrow + matplotlib only (this repo's env has no pandas).

Run:   uv run python notebooks/f_validation_report.py
Writes: docs/research/f-validation-report.json  — every number the doc quotes
        docs/research/f-metric-ecdfs.png
        docs/research/f-allocation-fidelity.png
        docs/research/f-drainage.png
        docs/research/f-pixel-integrity.png
        docs/research/f-coverage-map.png
        docs/research/f-gate-attrition.png
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dataset import catalog, centroids, frame, gates, materialize, normalize, validate_f
from dataset.shards import shard_path

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "docs" / "research"

#: ADR-0004 §1's transforms — the space the strata were cut in, so the space the ECDFs
#: are read in. A raw-space ECDF of ``local_relief`` is a spike at the origin.
LOG_METRICS = frame.LOG_METRICS
#: Quantiles printed per metric beside the W1 distances, matching E2's so the two
#: reports' columns can be read against each other.
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
#: ECDF sample points — see E2's note; 512 is past what the eye resolves on a 4-inch panel.
ECDF_POINTS = 512
#: F1b permutation batch. 64 rather than E2's 256: the membership matrix is
#: ``(batch, 500_000)`` and the working float64 array inside ``_partition_w1`` is eight
#: times that, so 256 would transiently hold a gigabyte for no gain in wall-clock.
PERMUTATION_BATCH = 64
#: Tiles per block in the F8 shard scan. 512 x 224 x 224 int16 is ~51 MB — big enough
#: that the per-block Python overhead vanishes, small enough to stay in cache-ish memory.
SHARD_BLOCK = 512


# --- Loading ------------------------------------------------------------------


def _rel(path: Path) -> str:
    """Repo-relative for the log when it can be; absolute when a smoke run points out."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_selection(path: Path) -> dict[str, Any]:
    """The #71 selection list as committed, plus the md5 of the bytes just parsed.

    The md5 is taken over the file this run actually read, never carried from a ticket:
    F2a's whole point is that "the selection list AS THE DRIVER LOADED IT" matches #71's
    commit, and a hash copied from elsewhere would check the wrong thing.
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
    out: dict[str, Any] = {
        name: table.column(name).to_numpy(zero_copy_only=False)
        for name in table.schema.names
        if name != "split"
    }
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
        "detail": table.column("detail").to_pylist(),
    }


def load_catalog(catalog_dir: Path) -> dict[str, Any]:
    """The gated catalog columns this report needs — numeric as numpy, keys as pyarrow.

    ``window_id``, ``stem`` and ``geocell`` stay pyarrow arrays. They are only ever used
    as join keys or grouped over, and ``pyarrow.compute`` hashes them in C++ without
    materializing 4.4 M Python strings apiece — which is most of a gigabyte, three times,
    for lookups that take seconds.
    """
    if not (catalog_dir / catalog.PARTS_DIR).exists():
        raise SystemExit(
            f"no gated catalog at {catalog_dir}. Build it with:\n"
            "  uv run python -m dataset.catalog   # the global pass (hours)\n"
            "  uv run python -m dataset.gates     # the gates (seconds)"
        )
    keys = ["window_id", "stem", "geocell"]
    numeric = ["lon", "lat", "land_std", *catalog.FROZEN_METRICS, *gates.FLAG_COLUMNS]
    table = pq.read_table(catalog_dir / catalog.PARTS_DIR, columns=[*keys, *numeric])
    out: dict[str, Any] = {name: table.column(name).combine_chunks() for name in keys}
    for name in numeric:
        out[name] = table.column(name).to_numpy(zero_copy_only=False)
    return out


def join_rows(catalog_ids: Any, window_ids: list[str], *, label: str) -> np.ndarray:
    """Catalog row index per requested ``window_id``; a miss is fatal, not a NaN.

    Every window this report looks up was selected *from* the catalog, so a miss means
    the join key is wrong — and a wrong join that silently produced NaNs would make F1a
    pass by comparing nothing and F1d pass by comparing nothing to nothing.
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


def counts_by(values: Any, mask: np.ndarray | None = None) -> Counter:
    """Value counts over a pyarrow string array, optionally masked — in C++, not Python."""
    array = values.filter(pa.array(np.asarray(mask, dtype=bool))) if mask is not None else values
    table = pc.value_counts(array)
    return Counter(
        {
            str(v): int(c)
            for v, c in zip(
                table.field("values").to_pylist(), table.field("counts").to_pylist()
            )
        }
    )


# --- The bar's two under-specified inputs -------------------------------------


def resolve_expected_md5(bar: validate_f.FBar, draw: dict[str, Any], built_with: str) -> dict[str, Any]:
    """F2a's expected md5, which #69 deliberately left ``null`` for #71 to supply.

    The bar was committed before the draw existed, so ``committed_inputs.selection_md5``
    is ``null`` and ``selection_md5_owner`` points at #71. That is a delegation, not a
    hole — but it means the hash this criterion gates against is read from a file the bar
    does not itself contain, so both places it appears are checked to agree before either
    is used:

    * ``f-scale-draw.json → determinism.selection_md5`` — the list #71 committed;
    * ``corpus_meta.json → build_params.selection_list_md5`` — the list #72 actually
      consumed.

    A disagreement between those two is a corpus built from a list nobody registered, and
    no gating criterion in the bar would catch it: F2a compares the *report's* load
    against #71, and would pass while the shards came from somewhere else. So it raises
    here rather than being published as a curiosity.
    """
    registered = bar.committed_inputs.get("selection_md5")
    from_draw = str(draw["determinism"]["selection_md5"])
    if registered not in (None, from_draw):
        raise SystemExit(
            f"the bar registers selection_md5 {registered!r} and #71's draw report says "
            f"{from_draw!r} — reconcile them in a dated commit before running the gate"
        )
    if built_with != from_draw:
        raise SystemExit(
            f"the corpus was built from selection list md5 {built_with!r} but #71 "
            f"committed {from_draw!r} — the shards are not the list F2a is about to check"
        )
    return {
        "expected_md5": from_draw,
        "source": "docs/research/f-scale-draw.json:determinism.selection_md5",
        "why_not_the_bar": (
            "committed_inputs.selection_md5 is null — #69 committed before #71 drew and "
            "delegated the hash to selection_md5_owner"
        ),
        # The md5 the *builder* consumed. Equal to the above or this function has raised;
        # recorded as the value rather than as a bare `true` so a reader sees the fact.
        "corpus_meta_build_params_selection_list_md5": built_with,
    }


def expected_provenance(bar: validate_f.FBar) -> dict[str, int]:
    """F2a's provenance table, derived because the bar states it as a sum.

    Gate E's bar carried ``{"base": 49330, "floor:coastline": 670}`` as a literal table.
    Gate F's says only *"both provenance tags summing to coast_floor = 6700"*, so the
    table is reconstructed from two numbers the bar does state — ``budget`` and
    ``allocation.coast_floor`` — under the vocabulary #71 committed. Written down rather
    than inlined because a derived expectation is a weaker thing than a registered one,
    and a reader is entitled to see which kind they are looking at.
    """
    budget = int(bar.committed_inputs["budget"])
    coast = int(bar.allocation["coast_floor"])
    return {"base": budget - coast, "floor:coastline": coast}


# --- F1c pool comparison (published) ------------------------------------------


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


def _weighted_w1(corpus: np.ndarray, weights: np.ndarray, pool: np.ndarray) -> float:
    """W1 between the ``N_h/n_h``-weighted corpus and the pool.

    Same integral as :func:`dataset.validate.wasserstein1`, with the corpus CDF weighted
    instead of counted. Kept in the driver rather than the module because the weighted
    form is a descriptive figure label, not something the bar gates on.
    """
    grid = np.sort(np.concatenate([corpus, pool]))
    widths = np.diff(grid)
    return float(
        np.sum(np.abs(_ecdf(corpus, grid[:-1], weights) - _ecdf(pool, grid[:-1])) * widths)
    )


def pool_comparison(
    corpus: dict[str, np.ndarray], pool: dict[str, np.ndarray], weights: np.ndarray
) -> dict[str, Any]:
    """F1c: per-metric ECDF overlays and W1 distances, corpus vs the kept pool.

    Published, never gated. The corpus is *deliberately* not pool-distributed — Neyman
    over-samples the rare rugged strata on purpose — so a large W1 here is the design
    working, not failing. The ``N_h/n_h``-weighted curve is the check that the
    over-sampling is the *intended* one, and the bar notes this reads more informatively
    at 16.4 % of the pool than at Gate E's 1.6 %.
    """
    out: dict[str, Any] = {"quantiles": list(QUANTILES), "ecdf_points": ECDF_POINTS}
    per_metric: dict[str, Any] = {}
    for name in catalog.FROZEN_METRICS:
        c = _transform(name, corpus[name])
        p = _transform(name, pool[name])
        finite = np.isfinite(c)
        c, w = c[finite], weights[finite]
        p = p[np.isfinite(p)]
        lo, hi = np.quantile(p, [0.001, 0.999])
        grid = np.linspace(lo, hi, ECDF_POINTS)
        per_metric[name] = {
            "transform": "ln(x+1)" if name in LOG_METRICS else "raw",
            "w1_corpus_vs_pool": validate_f.wasserstein1(c, p),
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


# --- Coverage and per-COG spread ----------------------------------------------


def coverage(manifest: dict[str, Any], cat: dict[str, Any], cell_deg: int = 10) -> dict[str, Any]:
    """Per-geo-cell corpus counts against the kept pool's occupied cells.

    Published, and — unlike Gate E's E2.3 — carrying **no tripwire at all**: #69 retired
    the per-COG cap rather than rescaling it, so there is no geographic gate in the Gate F
    bar for this to hang off. #73's scope asks for the coverage map and the per-geo-cell
    counts as diagnostics, so they are computed and published under no criterion number.
    """
    keep = cat["keep"]
    pool_cells = counts_by(cat["geocell"], keep)
    corpus_cells = Counter(manifest["geocell_id"].tolist())
    missed = sorted(set(pool_cells) - set(corpus_cells))
    lon_edges = np.arange(-180, 181, cell_deg)
    lat_edges = np.arange(-90, 91, cell_deg)
    grid, _, _ = np.histogram2d(manifest["lon"], manifest["lat"], bins=[lon_edges, lat_edges])
    pool_grid, _, _ = np.histogram2d(
        cat["lon"][keep], cat["lat"][keep], bins=[lon_edges, lat_edges]
    )
    return {
        "registered_as": "not a Gate F criterion — #73 asks for it as a diagnostic",
        "cell_deg": cell_deg,
        "n_catalog_cells_occupied": len(counts_by(cat["geocell"])),
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
        "grid": grid.tolist(),
        "pool_grid": pool_grid.tolist(),
        "lon_edges": lon_edges.tolist(),
        "lat_edges": lat_edges.tolist(),
    }


def cog_spread(
    realized: Counter, inventory: Counter
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """F6c: the per-COG distribution, and each COG's draw as a fraction of what it holds.

    The measurement #42 asked the scale tier to carry forward when it retired the cap as
    inert. The *fraction* column is the one that carries new information: a raw count of
    233 says nothing without knowing that a 1° COG holds up to 256 kept windows, and the
    bar's debug-tier baseline is stated in exactly those terms (among COGs with inventory
    ≥ 32, max fraction 0.312, none reaching 0.5).

    Returns ``(published, counts, fractions)``. The two 22,906-long arrays the histogram
    panels need come back as their own values rather than riding inside the published dict
    under underscore keys — that dict is written straight to JSON, and a caller who forgot
    to strip them would either bloat the artifact by a megabyte or crash the encoder.
    """
    counts = np.array([realized[c] for c in realized], dtype=np.int64)
    fractions = np.array(
        [realized[c] / inventory[c] for c in realized if inventory.get(c)], dtype=np.float64
    )
    deep = np.array(
        [realized[c] / inventory[c] for c in realized if inventory.get(c, 0) >= 32],
        dtype=np.float64,
    )
    return {
        "n_cogs_in_pool": len(inventory),
        "n_cogs_used": int(counts.size),
        "n_windows": int(counts.sum()),
        "mean_per_cog": float(counts.mean()),
        "median_per_cog": float(np.median(counts)),
        "p99_per_cog": float(np.percentile(counts, 99)),
        "max_per_cog": int(counts.max()),
        "max_inventory_in_pool": int(max(inventory.values())),
        "fraction_of_inventory": {
            "mean": float(fractions.mean()),
            "median": float(np.median(fractions)),
            "p99": float(np.percentile(fractions, 99)),
            "max": float(fractions.max()),
            "n_at_or_over_0.5": int((fractions >= 0.5).sum()),
            "n_fully_drawn": int((fractions >= 1.0).sum()),
        },
        "fraction_among_inventory_ge_32": {
            "n_cogs": int(deep.size),
            "median": float(np.median(deep)) if deep.size else float("nan"),
            "max": float(deep.max()) if deep.size else float("nan"),
            "n_at_or_over_0.5": int((deep >= 0.5).sum()),
        },
    }, counts, fractions


# --- F5 attrition by stratum (published) --------------------------------------


def would_be_strata(
    cat: dict[str, Any], centroids_: centroids.Centroids, scaler: frame.Scaler
) -> tuple[np.ndarray, np.ndarray]:
    """Give every catalog window the stratum it would have joined, dropped ones included.

    k-means was fit on survivors only, so the 1.33 M dropped windows carry full metrics
    and no ``stratum_id`` (#37: gates are columns, so the rows are still there). Scoring
    them against the committed centroids is what makes F5's per-(gate × stratum) table
    possible at all — and what gives F3 its *pre-gate* population, the one the bar cut its
    texture p75 from.

    Returns ``(stratum_id, assignable)``. A window whose metric vector has a NaN cannot be
    scored — that means it had no land to measure — so it carries ``-1`` and is counted
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
    """The per-(gate × stratum) drop table, with materialization drops as a last column.

    Returns ``(published_table, rule_inputs)`` — the table as it lands in the artifact,
    and the int-keyed counts :func:`dataset.validate_f.gate_stratum_bias` consumes.

    **At Gate F this gates nothing.** The bar's argument is that the table is *bit-
    identical* to Gate E's: it reads the catalog and the gates, neither of which has a
    budget term, so the same centroids give the same ratios and the same fired pairs. That
    claim is checkable, so the driver checks it (:func:`gate_e_bit_identity`) rather than
    reprinting it as an assertion. Its firing is dispositioned in full by ADR-0006 and
    ADR-0007, which map #67 places out of scope.
    """
    strata = sorted(set(labels[assignable].tolist()))
    membership = {h: int((labels == h).sum()) for h in strata}
    fires = {
        gate: {h: int((cat[gate] & (labels == h)).sum()) for h in strata}
        for gate in gates.DROP_GATES
    }
    corpus_fires = {gate: int(cat[gate].sum()) for gate in gates.DROP_GATES}
    narrow_fires = {gate: int((cat[gate] & assignable).sum()) for gate in gates.DROP_GATES}

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


def gate_e_bit_identity(table: dict[str, Any], e_report: Path) -> dict[str, Any]:
    """Is F5's table byte-for-byte Gate E's, as the bar claims it must be?

    The bar retires E2.5 as a gate on the grounds that it *cannot* return anything new at
    a different budget. That is a falsifiable claim about the catalog-and-gates half of
    the table, so it is tested here instead of quoted: the pre-gate membership, the
    per-(gate × stratum) fires and the corpus-wide fires must all match E's committed
    report exactly. The materialization columns are budget-dependent by construction and
    are excluded from the comparison.
    """
    if not e_report.exists():
        return {"checked": False, "reason": f"{e_report} not present"}
    previous = json.loads(e_report.read_text())["attrition_by_stratum"]
    same = {
        field: previous.get(field) == table.get(field)
        for field in ("membership", "kept", "fires", "corpus_fires_whole_catalog",
                      "corpus_fires_assignable_only", "n_rows", "n_assignable")
    }
    return {
        "checked": True,
        "against": _rel(e_report),
        "bit_identical": all(same.values()),
        "per_field": same,
        "note": (
            "the budget-dependent materialization columns are excluded — they are the "
            "only part of this table a bigger draw can move"
        ),
    }


def attrition_evidence(
    cat: dict[str, Any],
    selected_rows: np.ndarray,
    dropped: np.ndarray,
    ledger: dict[str, Any],
    *,
    degenerate_eps: float,
) -> dict[str, Any]:
    """What the drops *were*, for the two criteria that measure them without describing them.

    F1b asks whether attrition was terrain-selective and F7 asks whether there was too much
    of it; neither says what the dropped windows look like, and both handoffs to #74 are
    unreadable without that. So this reports, per frozen metric and for ``land_std``, the
    dropped set beside the retained set — and, because the bar's F7 derivation rests
    entirely on the claim that drops are int16 quantization flips across the ``degenerate``
    threshold, how close to that threshold the dropped windows' catalog ``land_std``
    actually sits.

    Descriptive. Nothing here gates, and no number in it was registered.
    """
    keep = ~dropped
    land_std = cat["land_std"][selected_rows]
    drop_std = land_std[dropped]
    bands = [0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    per_metric = {
        name: {
            "dropped_median": float(np.median(cat[name][selected_rows][dropped])),
            "retained_median": float(np.median(cat[name][selected_rows][keep])),
            "dropped_p05": float(np.quantile(cat[name][selected_rows][dropped], 0.05)),
            "dropped_p95": float(np.quantile(cat[name][selected_rows][dropped], 0.95)),
        }
        for name in catalog.FROZEN_METRICS
    }
    by_status = Counter(
        s for s in ledger["status"].tolist() if s != "ok"
    )
    return {
        "n_dropped": int(dropped.sum()),
        "by_status": {s: int(n) for s, n in sorted(by_status.items())},
        "per_metric_dropped_vs_retained": per_metric,
        "land_std": {
            "degenerate_threshold_m": degenerate_eps,
            "dropped_min": float(drop_std.min()) if drop_std.size else float("nan"),
            "dropped_median": float(np.median(drop_std)) if drop_std.size else float("nan"),
            "dropped_max": float(drop_std.max()) if drop_std.size else float("nan"),
            "retained_median": float(np.median(land_std[keep])),
            "dropped_within_band_of_threshold": {
                str(b): int(((drop_std >= degenerate_eps) & (drop_std < degenerate_eps + b)).sum())
                for b in bands
            },
            "dropped_below_threshold": int((drop_std < degenerate_eps).sum()),
            "kept_pool_in_first_band": int(
                ((cat["land_std"] >= degenerate_eps)
                 & (cat["land_std"] < degenerate_eps + 0.01)
                 & cat["keep"]).sum()
            ),
        },
    }


# --- F3's reference, checked against the bar that registered it ----------------


def registered_reference_crosscheck(
    bar: validate_f.FBar, drainage: validate_f.DrainageInputs
) -> dict[str, Any]:
    """The bar's own F3 reference table against what this report measures.

    #69 registered ``pre_gate_tail``, ``kept_tail`` and the kept-pool proportional share
    per stratum beside the p75 they were counted at. Those are *derived* numbers — the bar
    could compute them from the committed catalog before any draw existed, and did — so
    they are reproducible, and a report that did not reproduce them would be measuring a
    different population from the one the 0.25 threshold was reasoned about.

    Reproduced at the bar's own **printed** p75, because that is the number
    :func:`dataset.validate_f.read_bar` freezes and the criterion thresholds on. Where the
    two disagree, the printed p75 and the full-precision one the bar's table was evidently
    computed at are the first place to look — see ``F3_p75_precision_sensitivity``.
    """
    node = bar.doc["criteria"]["F3_drainage_representation"]
    scale = node["reference_and_scale_sensitivity"]
    tiers = node["two_tier_scope"]
    out: dict[str, Any] = {}
    for key, entry in drainage.crosscheck.items():
        registered = scale.get(f"stratum_{key}") or tiers.get(f"stratum_{key}") or {}
        row: dict[str, Any] = {}
        for name, measured in (
            ("pre_gate_tail", entry["pre_gate_tail"]),
            ("kept_tail", entry["kept_tail"]),
            ("kept_pool_share", entry["kept_pool_share"]),
        ):
            if name not in registered:
                continue
            value = registered[name]
            row[name] = {
                "registered": value,
                "measured": measured,
                # The share is registered to four decimals for legibility; the counts are
                # exact integers and are compared as such.
                "agrees": (
                    abs(float(value) - float(measured)) < 5e-5
                    if isinstance(measured, float)
                    else value == measured
                ),
            }
        out[key] = row
    return out


# --- F8's recompute -----------------------------------------------------------


def _ulps(a: float, b: float) -> int:
    """How many representable float64s separate ``a`` and ``b`` (same sign assumed).

    Reported beside F8's absolute and relative differences because "exact" is the
    registered tolerance and a reader needs to see *how* inexact a mismatch is on the
    only scale that means anything for a float: the difference between "the last bit
    landed elsewhere" and "these are different numbers".
    """
    ia = np.float64(a).view(np.int64)
    ib = np.float64(b).view(np.int64)
    return int(abs(int(ia) - int(ib)))


def recompute_global_stats(shard_dir: Path, *, limit: int | None = None) -> dict[str, Any]:
    """Re-accumulate the corpus-wide land mean/std from the shipped shards.

    F8's live branch is ``tier_specific``, whose test is that ``corpus_meta.json``'s
    ``global_mean``/``global_std`` "reproduce exactly from the shipped 500k tiles". There
    is no shortcut: the recorded pair came from an accumulator the builder carried while
    writing, and checking it against the same accumulator's saved value would be checking
    a number against itself. So all 47 GB are read back.

    The mask and the moments are :mod:`dataset.normalize`'s, and the mean/std arithmetic
    is :func:`dataset.materialize.global_stats` — the same code the builder ran, not a
    second version of it. A re-derivation that summed in a different order would report a
    Defect on a correct corpus, which is precisely the failure this criterion must not
    have.
    """
    paths = sorted(shard_dir.glob("shard_*.npy"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no shards under {shard_dir} — F8 cannot read what is not there")
    started = time.time()
    land_sum = land_sqsum = 0.0
    land_count = 0
    n_tiles = 0
    for n, path in enumerate(paths, 1):
        shard = np.load(path, mmap_mode="r")
        for start in range(0, shard.shape[0], SHARD_BLOCK):
            block = np.asarray(shard[start : start + SHARD_BLOCK])
            s, q, c = normalize.land_moments(block)
            land_sum += s
            land_sqsum += q
            land_count += c
        n_tiles += int(shard.shape[0])
        print(f"    shard {n}/{len(paths)}  {n_tiles:,} tiles  {time.time() - started:.0f}s")
    mean, std = materialize.global_stats(land_sum, land_sqsum, land_count)
    return {
        "global_mean": mean,
        "global_std": std,
        "n_shards": len(paths),
        "n_tiles": n_tiles,
        # The three additive accumulators, reported rather than only their quotient. If
        # the recorded statistic and this one disagree, the first question is whether the
        # two passes saw the *same pixels* — and these answer it, where two derived
        # floats cannot.
        "land_pixels": land_count,
        "land_sum": land_sum,
        "land_sqsum": land_sqsum,
        "block_tiles": SHARD_BLOCK,
        "elapsed_s": round(time.time() - started, 1),
        "partial": limit is not None,
    }


def builder_accumulators(corpus_dir: Path) -> dict[str, Any]:
    """The accumulators #72's build carried, for the same-pixels question F8 raises.

    ``build_state.json`` is the driver's resume checkpoint, not an independent
    measurement — it is where ``corpus_meta.json``'s statistic came *from*. That makes it
    useless as a check on the recorded numbers and exactly the right thing to compare a
    recompute's raw sums against: an identical ``land_count`` and ``land_sum`` beside a
    differing ``land_sqsum`` says the two passes read the same pixels and summed them in
    a different order, which is a different fact from a corpus that changed.
    """
    path = corpus_dir / "build_state.json"
    if not path.exists():
        return {"available": False, "path": str(path)}
    state = json.loads(path.read_text())
    return {
        "available": True,
        "source": f"{corpus_dir.name}/build_state.json — the builder's own accumulator, "
                  "not an independent measurement",
        "land_count": int(state["land_count"]),
        "land_sum": float(state["land_sum"]),
        "land_sqsum": float(state["land_sqsum"]),
    }


# --- Figures -------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def figure_ecdfs(comparison: dict[str, Any], path: Path) -> None:
    """2×3 of the frozen six: pool, corpus, and the N_h/n_h-reweighted corpus.

    The gap between the grey pool curve and the blue corpus curve *is* the Neyman
    over-sampling. The dashed reweighted curve is the one that should sit back on the pool.
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
        "F1c (#73): per-metric ECDFs, corpus vs the kept pool at 16.4% draw — the gap is "
        "Neyman working, not failing (published, never a gate)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[wrote {_rel(path)}]")


def figure_allocation(alloc: dict[str, Any], path: Path) -> None:
    """Selected vs realized per stratum, and the drop rate F2c reads."""
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
        label=f"F2c floor ({alloc['absolute_floor']:,})",
    )
    axes[0].set_xticks(x, [str(h) for h in strata])
    axes[0].set_ylabel("windows")
    axes[0].legend(fontsize=8)
    axes[0].set_title(
        f"F2 (#73): allocation fidelity — {sum(real):,} of {sum(sel):,} selected windows "
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
    print(f"[wrote {_rel(path)}]")


def figure_drainage(outcome_measured: dict[str, Any], crosscheck: dict[str, Any], path: Path) -> None:
    """F3 — the one criterion the budget actually changes, and the reading beside it.

    Two panels because the bar's ``published_decomposition`` is two questions, not one.
    The top is the criterion: ``r_h`` against the 0.25 gate and against the kept-pool
    proportional share, which is where drawing proportionally would land — the distance
    between bar and red tick is the only thing FSCS's coverage-seeking bought. The bottom
    is exhaustion, which says whether the survivors ran out; without it the top panel
    cannot distinguish "the gate depleted the pool" from "the sampler declined it".

    The exhaustion *ceiling* is deliberately not drawn on the top panel. ``tail_h / n_h``
    is 0.97-1.66 here, so as a bound on ``r_h`` it is 1.0 for three of the four strata —
    a flat line at the top of the axes that squashes the bars and says nothing.
    """
    plt = _plt()
    per = outcome_measured["per_stratum"]
    gate = float(outcome_measured["min_share"])
    strata = sorted(per, key=lambda h: (not per[h]["gates"], int(h)))
    x = np.arange(len(strata))
    r = [per[h]["realized_share_r_h"] for h in strata]
    pool = [crosscheck[h]["kept_pool_share"] for h in strata]
    exhaustion = [per[h]["exhaustion_e_h"] for h in strata]
    colours = ["#1f77b4" if per[h]["gates"] else "0.72" for h in strata]

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.4), height_ratios=[2, 1], sharex=True)
    ax = axes[0]
    ax.bar(x, r, width=0.56, color=colours)
    ax.scatter(x, pool, marker="_", s=1500, color="#d62728", lw=2.4, zorder=3,
               label="kept-pool proportional share — drawing here fires")
    ax.axhline(gate, color="black", ls="--", lw=1.3, label=f"F3 gate ({gate:g})")
    ax.bar(0, 0, color="#1f77b4", label="realized $r_h$ — gating strata")
    ax.bar(0, 0, color="0.72", label="realized $r_h$ — corroborating only")
    for i in range(len(strata)):
        ax.annotate(
            f"{r[i]:.3f}\n{r[i] / pool[i]:.2f}× proportional",
            (i, r[i]), textcoords="offset points", xytext=(0, 5),
            ha="center", fontsize=8.5,
        )
    ax.set_ylabel("$r_h$: share of shipped tiles with texture ≥ pre-gate p75")
    ax.set_ylim(0, max(max(r), gate) * 1.6)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(
        "F3 (#73): ADR-0007's drainage re-test at 500k — the criterion this map exists for",
        fontsize=11,
    )

    ax1 = axes[1]
    ax1.bar(x, exhaustion, width=0.56, color=colours)
    for i in range(len(strata)):
        ax1.annotate(f"{exhaustion[i]:.3f}", (i, exhaustion[i]), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8.5)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("$e_h$: share of the\nsurviving tail consumed")
    ax1.set_xticks(x, [f"stratum {h}" + ("" if per[h]["gates"] else "\n(corroborating)")
                       for h in strata])
    ax1.set_title(
        "exhaustion — 1.0 would mean the flatness gate's survivors genuinely ran out",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[wrote {_rel(path)}]")


def figure_pixel_integrity(measured: dict[str, Any], rel: np.ndarray, path: Path) -> None:
    """F1d — the first whole-corpus look at the pixels, and its measured separation.

    Log-histogram of the per-tile relative disagreement, with the 0.5 tolerance and the
    debug-tier mis-join signatures the bar sized it against drawn in. A healthy corpus
    piles up at the int16 quantization floor near 1e-4; a mis-join would put roughly half
    the mass to the right of the dashed line.
    """
    plt = _plt()
    finite = rel[np.isfinite(rel) & (rel > 0)]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(finite, bins=np.logspace(-8, 1, 160), color="#1f77b4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(measured["rel_tol"], color="#d62728", ls="--", lw=1.4,
               label=f"F1d tolerance ({measured['rel_tol']:g} relative)")
    ax.axvline(measured["median_rel_diff"], color="#2ca02c", lw=1.4,
               label=f"median {measured['median_rel_diff']:.2e}")
    ax.axvline(measured["max_rel_diff"], color="#ff7f0e", lw=1.2, ls=":",
               label=f"worst tile {measured['max_rel_diff']:.4f}")
    ax.set_xlabel("|manifest.std − catalog.land_std| / catalog.land_std")
    ax.set_ylabel("tiles")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(
        f"F1d (#73): whole-corpus pixel integrity — {measured['n_over_tol']:,} of "
        f"{measured['n_compared']:,} tiles ({measured['share_over_tol']:.4%}) over tolerance\n"
        f"registered max {measured['share_max']:g}; an adjacent-window mis-join measured "
        f"48% at the debug tier",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[wrote {_rel(path)}]")


def figure_coverage(
    cov: dict[str, Any], spread: dict[str, Any], counts_per_cog: np.ndarray,
    fractions: np.ndarray, path: Path,
) -> None:
    """Where the corpus landed, and how deeply it drew from each COG it touched."""
    plt = _plt()
    import matplotlib

    grid = np.array(cov["grid"])
    pool = np.array(cov["pool_grid"])
    extent = [-180, 180, -90, 90]

    fig = plt.figure(figsize=(11, 10.4))
    ax0 = fig.add_subplot(4, 1, (1, 2))
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
        f"#73 diagnostic: corpus tiles per {cov['cell_deg']}° cell (log) — "
        f"{cov['n_corpus_cells_occupied']} of {cov['n_pool_cells_occupied']} kept-pool "
        f"geo-cells reached. No Gate F criterion gates on geography.",
        fontsize=10,
    )
    ax0.set_xlabel("longitude")
    ax0.set_ylabel("latitude")
    fig.colorbar(im, ax=ax0, label="corpus tiles")

    ax1 = fig.add_subplot(4, 1, 3)
    ax1.hist(counts_per_cog, bins=np.arange(0.5, counts_per_cog.max() + 1.5), color="#1f77b4")
    ax1.set_yscale("log")
    ax1.set_xlabel("tiles per source COG")
    ax1.set_ylabel("COGs")
    ax1.set_title(
        f"F6c: {spread['n_cogs_used']:,} COGs used of {spread['n_cogs_in_pool']:,} in the "
        f"kept pool; mean {spread['mean_per_cog']:.2f}, median "
        f"{spread['median_per_cog']:.0f}, p99 {spread['p99_per_cog']:.0f}, max "
        f"{spread['max_per_cog']}",
        fontsize=9.5,
    )

    ax2 = fig.add_subplot(4, 1, 4)
    ax2.hist(fractions, bins=np.linspace(0, 1, 101), color="#7f5fb0")
    ax2.set_yscale("log")
    ax2.axvline(1.0, color="#d62728", ls="--", lw=1.2, label="F6b bound (own inventory)")
    ax2.set_xlabel("realized count ÷ that COG's own kept inventory")
    ax2.set_ylabel("COGs")
    ax2.legend(fontsize=8)
    frac = spread["fraction_of_inventory"]
    ax2.set_title(
        f"F6c: fraction of each COG's own inventory drawn — median {frac['median']:.3f}, "
        f"p99 {frac['p99']:.3f}, max {frac['max']:.3f}; {frac['n_at_or_over_0.5']:,} COGs "
        f"at or over half. Debug tier: max 0.312, none over 0.5.",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[wrote {_rel(path)}]")


def figure_attrition(table: dict[str, Any], path: Path) -> None:
    """Per-(gate × stratum) drop rate as a heatmap — F5's table, made legible.

    Rows are gates, columns are strata, cell colour is the gate's within-stratum drop
    rate. Reprinted at Gate F with a disposition pointer and no threshold: the table is
    bit-identical to Gate E's, and its firing is settled by ADR-0006 and ADR-0007.
    """
    plt = _plt()
    strata = sorted(int(h) for h in table["membership"])
    gate_names = list(table["fires"])
    rates = np.array(
        [
            [table["fires"][g][str(h)] / max(table["membership"][str(h)], 1) for h in strata]
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
        "F5 (#73): quality-gate drop rate per (gate × would-be stratum) — published, "
        "dispositioned by ADR-0006 / ADR-0007, gates nothing",
        fontsize=11,
    )
    # The colorbar steals width from *both* axes so the two panels' stratum columns line up.
    fig.colorbar(im, ax=axes.tolist(), label="within-stratum drop rate")

    mat = table["materialization"]
    sel = [mat.get(str(h), {}).get("selected", 0) for h in strata]
    drop = [mat.get(str(h), {}).get("dropped", 0) for h in strata]
    axes[1].bar(np.arange(len(strata)), drop, color="#d62728")
    axes[1].set_xticks(np.arange(len(strata)), [str(h) for h in strata])
    axes[1].set_xlim(-0.5, len(strata) - 0.5)
    axes[1].set_xlabel("stratum_id")
    axes[1].set_ylabel("windows")
    axes[1].set_ylim(0, max(max(drop), 1) * 1.5)
    axes[1].set_title(
        f"materialization drops per stratum: {sum(drop)} of {sum(sel):,} selected — "
        "gated by F7 / F2c, not here",
        fontsize=9,
    )
    fig.savefig(path, dpi=130)  # no tight_layout: it would undo the colorbar's shared steal
    plt.close(fig)
    print(f"[wrote {_rel(path)}]")


# --- Driver -------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", type=Path, default=REPO / "data" / "corpus_500k")
    p.add_argument("--catalog", type=Path, default=REPO / "data" / "catalog_gated")
    p.add_argument("--ledger", type=Path,
                   default=REPO / "artifacts" / "phase_f" / "materialization_ledger.parquet")
    p.add_argument("--selection", type=Path,
                   default=REPO / "artifacts" / "phase_f" / "selection_scale_500k.csv")
    p.add_argument("--allocation", type=Path, default=RESEARCH / "f-allocation-nh-500k.json")
    p.add_argument("--draw-report", type=Path, default=RESEARCH / "f-scale-draw.json")
    p.add_argument("--gate-e-report", type=Path, default=RESEARCH / "e2-validation-report.json")
    p.add_argument("--gallery", type=Path, default=RESEARCH / "f-gallery" / "gallery.json")
    p.add_argument("--centroids", type=Path,
                   default=REPO / "artifacts" / "phase_d" / "centroids.json")
    p.add_argument("--scaler", type=Path, default=REPO / "artifacts" / "phase_d" / "scaler.json")
    p.add_argument("--bar", type=Path, default=RESEARCH / "f-gate-bar.json")
    p.add_argument("--out-prefix", type=Path, default=RESEARCH / "f")
    p.add_argument("--seed", type=int, default=73,
                   help="F1b permutation seed; recorded for byte-reproduction of the "
                        "printed number, not relied on for the pass/fail call")
    p.add_argument("--permutations", type=int, default=None,
                   help="override the bar's registered permutation count (SMOKE RUNS ONLY)")
    p.add_argument("--shard-limit", type=int, default=None,
                   help="read only the first N shards for F8 (SMOKE RUNS ONLY — the "
                        "recomputed statistic is then not the corpus's)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    smoke = args.permutations is not None or args.shard_limit is not None
    bar = validate_f.read_bar(args.bar)
    th, ft = bar.thresholds, bar.f

    meta = json.loads((args.corpus / "corpus_meta.json").read_text())
    draw = json.loads(args.draw_report.read_text())
    allocation = json.loads(args.allocation.read_text())
    n_h = bar.expected_per_stratum
    N_h = {int(r["stratum_id"]): int(r["N_h"]) for r in allocation["rows"]}

    # The contract is a *committed input* at Gate F, not a branch — #50 settled it at
    # Gate E and #69 carried it as a constant, so `FBar` has no `live_branch`. The corpus
    # card is checked against the bar rather than consulted for the answer.
    contract = str(bar.committed_inputs["attrition_contract"])
    if meta["attrition_contract"] != contract:
        raise SystemExit(
            f"corpus card says attrition_contract {meta['attrition_contract']!r}; the bar "
            f"committed {contract!r} as an input, not a branch — reconcile before gating"
        )
    branch = bar.live_normalization_branch(meta["normalization_stat_source"])
    md5_resolution = resolve_expected_md5(
        bar, draw, str(meta["build_params"]["selection_list_md5"])
    )

    selection = load_selection(args.selection)
    manifest = load_manifest(args.corpus)
    ledger = load_ledger(args.ledger)
    cat = load_catalog(args.catalog)

    print(
        f"\n=== inputs ===\n"
        f"  bar           {_rel(args.bar)}\n"
        f"  contract      {contract} (committed input)  ·  F8 branch: {branch}\n"
        f"  selection     {len(selection['window_id']):,} rows, md5 {selection['md5']}\n"
        f"  manifest      {len(manifest['window_id']):,} tiles\n"
        f"  ledger        {len(ledger['window_id']):,} rows\n"
        f"  catalog       {cat['keep'].size:,} rows, keep {int(cat['keep'].sum()):,}"
        + ("\n  *** SMOKE RUN — not a Gate F artifact ***" if smoke else "")
    )

    selected_rows = join_rows(cat["window_id"], selection["window_id"], label="selected")
    corpus_rows = join_rows(cat["window_id"], manifest["window_id"], label="materialized")

    outcomes: list[validate_f.Outcome] = []

    # F1a — did #72 write the metric values it was told to. Compared at the manifest's own
    # declared width, which is the tolerance the bar registers.
    outcomes.append(
        validate_f.relabel(
            validate_f.metric_fidelity(
                {name: manifest[name] for name in catalog.FROZEN_METRICS},
                {name: cat[name][corpus_rows] for name in catalog.FROZEN_METRICS},
                storage_dtype=manifest[catalog.FROZEN_METRICS[0]].dtype,
            ),
            "F1a_metric_fidelity",
        )
    )

    # F1d — the pixels. The one criterion that reads something the catalog did not write.
    pixel = validate_f.pixel_integrity(
        manifest["std"], cat["land_std"][corpus_rows], thresholds=ft
    )
    outcomes.append(pixel)
    with np.errstate(invalid="ignore", divide="ignore"):
        pixel_rel = np.abs(
            np.asarray(manifest["std"], dtype=np.float64) - cat["land_std"][corpus_rows]
        ) / cat["land_std"][corpus_rows]

    # F1b — were the drops metric-selective. Blind: only the drop COUNT feeds the null.
    print("  [F1b] permutation null …")
    status_by_id = dict(zip(ledger["window_id"], ledger["status"].tolist()))
    dropped = np.array([status_by_id[w] != "ok" for w in selection["window_id"]], dtype=bool)
    outcomes.append(
        validate_f.relabel(
            validate_f.drop_selectivity(
                {name: cat[name][selected_rows] for name in catalog.FROZEN_METRICS},
                dropped,
                thresholds=th,
                permutations=args.permutations,
                seed=args.seed,
                batch=PERMUTATION_BATCH,
            ),
            "F1b_drop_selectivity",
        )
    )
    evidence = attrition_evidence(
        cat, selected_rows, dropped, ledger,
        degenerate_eps=float(bar.committed_inputs["gate_thresholds"]["degenerate_land_std_m"]),
    )

    # F2a — the list as this driver loaded it, against #71's commit.
    outcomes.append(
        validate_f.relabel(
            validate_f.selection_integrity(
                md5=selection["md5"],
                per_stratum=Counter(int(h) for h in selection["stratum_id"]),
                provenance=Counter(selection["provenance"]),
                expected_md5=md5_resolution["expected_md5"],
                expected_rows=int(bar.committed_inputs["budget"]),
                expected_per_stratum=n_h,
                expected_provenance=expected_provenance(bar),
            ),
            "F2a_selection_integrity",
        )
    )

    realized = Counter(int(h) for h in manifest["stratum_id"])
    selected = Counter(int(h) for h in selection["stratum_id"])
    outcomes.append(
        validate_f.relabel(
            validate_f.materialization_counts(realized, selected, contract=contract),
            "F2b_materialization_counts",
        )
    )
    outcomes.append(
        validate_f.relabel(
            validate_f.thin_stratum(realized, selected, thresholds=th), "F2c_thin_stratum"
        )
    )

    # F2d — the floor as the rate it always was. Gated on every coastline-tagged tile,
    # because that is the population #42 derived 1.337% from; the floor-provenance
    # survivors are reported beside it as the sub-diagnostic, not as the criterion.
    n_coastal = int(np.count_nonzero(manifest["coastline"]))
    outcomes.append(
        validate_f.relabel(
            validate_f.coastal_floor(n_coastal, len(manifest["window_id"]), thresholds=th),
            "F2d_coastal_floor",
        )
    )

    # F5 / F3 — both need every catalog window's would-be stratum, dropped ones included.
    print("  [F3/F5] scoring the whole catalog against the committed centroids …")
    scaler = frame.read_scaler(args.scaler)
    centroids_ = centroids.read_centroids(args.centroids)
    labels, assignable = would_be_strata(cat, centroids_, scaler)
    table, rule_inputs = attrition_table(cat, labels, assignable, ledger)
    table["gate_e_bit_identity"] = gate_e_bit_identity(table, args.gate_e_report)
    table["disposition"] = {
        "note": "published at Gate F; the firing is settled, not re-raised",
        "pointers": [
            "docs/adr/0006-accept-edm-flm-hollowing-stratum19.md",
            "docs/adr/0007-accept-degenerate-flatness-gate-flat-strata.md",
        ],
    }
    outcomes.append(
        validate_f.gate_stratum_bias(
            fires=rule_inputs["fires"],
            membership=rule_inputs["membership"],
            corpus_fires=rule_inputs["corpus_fires"],
            corpus_rows=rule_inputs["corpus_rows"],
            thresholds=th,
            baseline=f"whole catalog ({rule_inputs['corpus_rows']:,} rows) — Gate E's "
                     "'corpus-wide drop rate'",
            criterion="F5_gate_attrition_published",
            gates_verdict=False,
        )
    )

    # F3 — ADR-0007's revisit trigger, the criterion this map exists for. The four
    # populations are fixed by the corpus; the p75 is the only thing that varies between
    # the gated run and the sensitivity below, so it is the only free argument here.
    def drainage_at(texture_p75: Mapping[int, float]) -> validate_f.DrainageInputs:
        return validate_f.drainage_inputs(
            shipped_texture=manifest["texture"],
            shipped_stratum=manifest["stratum_id"],
            catalog_texture=cat["texture"],
            catalog_stratum=labels,
            catalog_keep=cat["keep"],
            texture_p75=texture_p75,
        )

    drainage = drainage_at(ft.drainage_texture_p75)
    f3 = validate_f.drainage_representation(
        drainage.texture_by_stratum, kept_tail=drainage.kept_tail, thresholds=ft
    )
    outcomes.append(f3)

    # A **published sensitivity, never a second verdict** — the same standing #52 gave
    # E2.5's alternative baseline. The bar freezes the pre-gate p75 and prints it to six
    # decimals, and its own registered tail counts are evidently the full-precision ones,
    # so the report re-runs the identical rule against the p75 recomputed from the catalog
    # to show whether the gap moves F3's outcome. Reported whichever way it comes out:
    # choosing which p75 to believe after seeing which one fires is the peeking the
    # pre-registration exists to prevent, and the criterion above ran on the frozen number.
    full_precision_p75 = {
        int(h): float(entry["recomputed_p75"]) for h, entry in drainage.crosscheck.items()
    }
    sens_inputs = drainage_at(full_precision_p75)
    sens = validate_f.drainage_representation(
        sens_inputs.texture_by_stratum,
        kept_tail=sens_inputs.kept_tail,
        thresholds=dataclasses.replace(ft, drainage_texture_p75=full_precision_p75),
    )
    p75_sensitivity = {
        "note": "identical rule at the full-precision p75; published, never a verdict",
        "gated_run_used": "the bar's printed six-decimal p75",
        "full_precision_p75": full_precision_p75,
        "would_fire": sens.fired,
        "hits": sens.measured["hits"],
        "per_stratum": sens.measured["per_stratum"],
        "changes_the_gated_outcome": sens.fired != f3.fired
        or sens.measured["hits"] != f3.measured["hits"],
    }
    outcomes.append(validate_f.published("F3_p75_precision_sensitivity", p75_sensitivity))
    reference_crosscheck = registered_reference_crosscheck(bar, drainage)

    # F6a / F6b — the two checks that replaced the retired per-COG cap.
    outcomes.append(
        validate_f.distinctness(
            selection_window_ids=selection["window_id"],
            manifest_tile_ids=[str(t) for t in manifest["tile_id"]],
        )
    )
    per_cog = Counter(manifest["source_cog"].tolist())
    inventory = counts_by(cat["stem"], cat["keep"])
    outcomes.append(validate_f.cog_inventory_bound(per_cog, inventory))

    # F7 / F7b — attrition, as a rate and as a vocabulary.
    statuses = Counter(ledger["status"].tolist())
    outcomes.append(
        validate_f.attrition_rate(
            len(selection["window_id"]), len(manifest["window_id"]), thresholds=ft
        )
    )
    outcomes.append(validate_f.registered_status_set(statuses))

    # F8 — the recorded normalization statistic against its live branch.
    print(f"  [F8] re-accumulating land moments over the shipped shards ({branch}) …")
    recomputed = recompute_global_stats(args.corpus / "shards", limit=args.shard_limit)
    builder = builder_accumulators(args.corpus)
    same_pixels: dict[str, Any] = {}
    if builder["available"] and not recomputed["partial"]:
        same_pixels = {
            "land_count_identical": builder["land_count"] == recomputed["land_pixels"],
            "land_sum_identical": builder["land_sum"] == recomputed["land_sum"],
            "land_sqsum_identical": builder["land_sqsum"] == recomputed["land_sqsum"],
            "land_sqsum_ulps": _ulps(builder["land_sqsum"], recomputed["land_sqsum"]),
        }
    outcomes.append(
        validate_f.normalization_statistic(
            branch=branch, recorded=meta, recomputed=recomputed, bar=bar
        )
    )

    # --- the published diagnostics ---
    weights = np.array(
        [N_h[int(h)] / n_h[int(h)] for h in manifest["stratum_id"]], dtype=np.float64
    )
    print("  [F1c] pool ECDFs …")
    comparison = pool_comparison(
        {name: manifest[name] for name in catalog.FROZEN_METRICS},
        {name: cat[name][cat["keep"]] for name in catalog.FROZEN_METRICS},
        weights,
    )
    cov = coverage(manifest, cat)
    spread, cog_counts, cog_fractions = cog_spread(per_cog, inventory)
    outcomes.append(
        validate_f.published(
            "F1c_pool_ecdf",
            {name: {k: v for k, v in entry.items() if not k.startswith(("grid", "ecdf"))}
             for name, entry in comparison["per_metric"].items()},
        )
    )
    gallery_manifest = (
        json.loads(args.gallery.read_text()) if args.gallery.exists() else None
    )
    outcomes.append(
        validate_f.published(
            "F4_gallery_published",
            {
                "status": "published, non-gating — #69's resolution comment on #73",
                "artifact": _rel(args.gallery),
                "rendered": gallery_manifest is not None,
                "seed": (gallery_manifest or {}).get("seed"),
                "n_gating_sheets": 0,
                "n_sheets": len((gallery_manifest or {}).get("strata", [])),
                "labels_owner": "https://github.com/JStoschek/DEMAE/issues/74",
                "labels_recorded": False,
                "note": (
                    "12 x 20 re-rendered from the shipped scale corpus, per #75's need "
                    "for a gallery of what shipped. The cover carries an empty "
                    "terrain-family label table for #74 to fill — the same split #53 "
                    "kept with #54, and the reason #73 does not fill it is the reason it "
                    "renders no verdict either. No coherence pass/fail: E2.4's question "
                    "was about k, which the same persisted centroids already answered "
                    "20/20 at #53/#54, and there is no mark column to fill."
                ),
            },
        )
    )
    outcomes.append(
        validate_f.published(
            "F6c_cog_spread_published",
            spread,
        )
    )
    outcomes.append(
        validate_f.published(
            "F7c_ledger_status_published",
            {
                "counts": {s: int(n) for s, n in sorted(statuses.items())},
                "details_for_dropped": dict(
                    Counter(
                        d for d, s in zip(ledger["detail"], ledger["status"].tolist())
                        if s != "ok"
                    ).most_common(12)
                ),
                "dropped_by_stratum": dict(
                    sorted(
                        Counter(
                            int(h) for h, s in zip(ledger["stratum_id"], ledger["status"])
                            if s != "ok"
                        ).items()
                    )
                ),
                "dropped_coastal": int(
                    sum(
                        1
                        for p, s in zip(ledger["provenance"], ledger["status"].tolist())
                        if s != "ok" and p == "floor:coastline"
                    )
                ),
            },
        )
    )

    # --- the report ---
    alloc_block = {
        "per_stratum": {
            str(h): {"selected": selected.get(h, 0), "realized": realized.get(h, 0)}
            for h in sorted(set(selected) | set(realized))
        },
        "absolute_floor": th.thin_absolute,
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
    # knob raises by design; that is recorded rather than allowed to take the report down,
    # because the question it raises is #74's and not this script's.
    payloads: list[tuple[validate_f.Outcome, dict[str, Any]]] = []
    print("\n=== criteria ===")
    for outcome in outcomes:
        if outcome.fired is None:
            mark = "published"
        elif not outcome.gates:
            mark = "would-fire" if outcome.fired else "no-fire"
        else:
            mark = "FIRED" if outcome.fired else "pass"
        print(f"  {outcome.criterion:<34} {mark:<10} {outcome.measured_miss or ''}")
        if not (outcome.gates and outcome.fired):
            continue
        anchor = outcome.criterion.lower().replace(".", "").replace("_", "-")
        try:
            payload = outcome.payload(f"docs/research/f-validation-report.md#{anchor}")
        except ValueError as exc:
            payload = {"payload_error": str(exc), "class": outcome.failure_class.capitalize()}
            print(f"    ! payload incomplete: {exc}")
        payloads.append((outcome, payload))

    figure_ecdfs(comparison, Path(f"{args.out_prefix}-metric-ecdfs.png"))
    figure_allocation(alloc_block, Path(f"{args.out_prefix}-allocation-fidelity.png"))
    figure_drainage(f3.measured, drainage.crosscheck, Path(f"{args.out_prefix}-drainage.png"))
    figure_pixel_integrity(pixel.measured, pixel_rel, Path(f"{args.out_prefix}-pixel-integrity.png"))
    figure_coverage(
        cov, spread, cog_counts, cog_fractions,
        Path(f"{args.out_prefix}-coverage-map.png"),
    )
    figure_attrition(table, Path(f"{args.out_prefix}-gate-attrition.png"))

    summary = validate_f.summarize(outcomes)
    summary["verdict_owner"] = "https://github.com/JStoschek/DEMAE/issues/74"
    artifact = {
        "artifact": "Gate F quantitative validation report — the 500,000-window scale tier",
        "ticket": "https://github.com/JStoschek/DEMAE/issues/73",
        "map": "https://github.com/JStoschek/DEMAE/issues/67",
        "bar": _rel(args.bar),
        "rules": "src/dataset/validate_f.py",
        "smoke_run": smoke,
        "inputs": {
            "corpus": _rel(args.corpus),
            "corpus_meta": meta,
            "selection_list": _rel(args.selection),
            "selection_md5": selection["md5"],
            "selection_md5_resolution": md5_resolution,
            "expected_provenance_derivation": {
                "table": expected_provenance(bar),
                "why": "the Gate F bar states provenance as a sum, not a table — see "
                       "notebooks/f_validation_report.py:expected_provenance",
            },
            "allocation_table": _rel(args.allocation),
            "draw_report": _rel(args.draw_report),
            "gated_catalog": _rel(args.catalog),
            "attrition_contract": contract,
            "attrition_contract_status": "committed input (#50 settled it at Gate E)",
            "normalization_branch": branch,
            "seed": args.seed,
            "elapsed_s": round(time.time() - started, 1),
        },
        "registry_reconciliation": registry_reconciliation(bar, outcomes),
        "scope_corrections": {
            "note": "two pointers in #73's scope text were stale when it was written; "
                    "#69's resolution comment corrected them and this report follows it",
            "per_cog_spread_tripwire": "retired, not executed — a cap of 64 sits inside "
                                       "the healthy range at a 16.4% draw. Replaced by "
                                       "F6b_cog_inventory_bound (gating) and "
                                       "F6c_cog_spread_published (the measurement).",
            "gallery": "published, non-gating; F1d_pixel_integrity took its former "
                       "gating role as the whole-corpus check on the imagery.",
            "module": "dataset.validate_f, not dataset.validate — Gate E's module is "
                      "frozen and its outcomes are relabelled for this report.",
        },
        "allocation": alloc_block,
        "drainage": {
            "crosscheck": drainage.crosscheck,
            "p75_rounding_moves_tiles": drainage.p75_rounding_moves_tiles,
            "vs_registered_reference": reference_crosscheck,
            "p75_precision_sensitivity": p75_sensitivity,
            "registered_reference": bar.doc["criteria"]["F3_drainage_representation"][
                "reference_and_scale_sensitivity"
            ],
        },
        "attrition_evidence": evidence,
        "coverage": cov,
        "cog_spread": spread,
        "attrition_by_stratum": table,
        "normalization": {
            "branch": branch,
            "recorded": {k: meta[k] for k in ("global_mean", "global_std")},
            "recomputed": recomputed,
            "builder_accumulators": builder,
            "same_pixels": same_pixels,
        },
        "pool_comparison": comparison,
        **summary,
        "payloads": {o.criterion: payload for o, payload in payloads},
        "open_questions": open_questions(
            outcomes,
            drainage=drainage,
            evidence=evidence,
            same_pixels=same_pixels,
            p75_sensitivity=p75_sensitivity,
            registered_reference=reference_crosscheck,
        ),
    }
    out = Path(f"{args.out_prefix}-validation-report.json")
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"[wrote {_rel(out)}]  ({time.time() - started:.0f}s total)")
    return 0


def registry_reconciliation(
    bar: validate_f.FBar, outcomes: Sequence[validate_f.Outcome]
) -> dict[str, Any]:
    """Every criterion the bar registers, against every criterion this report emitted.

    The one check on the report itself rather than on the corpus. #73's job is to execute
    *the* bar, and a criterion silently not run would look exactly like a criterion that
    passed — the scoreboard would simply be one row shorter, which nobody counts. So the
    two sets are diffed and both directions are named:

    * ``registered_but_not_emitted`` must be empty, or the report is incomplete.
    * ``emitted_but_not_registered`` is the report's own diagnostics. It is not a fault —
      Gate E's report carried ``E2.5_baseline_sensitivity`` the same way — but it is listed
      so a reader counting rows against the bar's table knows which are which rather than
      finding a criterion the pre-registration never contained.
    """
    registered = list(bar.doc["criteria"])
    emitted = [o.criterion for o in outcomes]
    return {
        "n_registered": len(registered),
        "n_emitted": len(emitted),
        "registered": registered,
        "registered_but_not_emitted": sorted(set(registered) - set(emitted)),
        "emitted_but_not_registered": sorted(set(emitted) - set(registered)),
        "complete": not set(registered) - set(emitted),
    }


def open_questions(
    outcomes: list[validate_f.Outcome],
    *,
    drainage: validate_f.DrainageInputs,
    evidence: dict[str, Any],
    same_pixels: dict[str, Any],
    p75_sensitivity: dict[str, Any],
    registered_reference: dict[str, Any],
) -> list[dict[str, Any]]:
    """Calls the bar does not settle, handed to #74 with their evidence.

    #73's remit is to execute the bar blind, which means it must also be explicit about
    where executing it leaves a question standing. Each entry below is *conditional on a
    measurement*, never on a reading — the report raises the question when the numbers
    raise it, and stops there. Resolving any of them is the Gate F review's, and none of
    them changes a fired flag above.
    """
    by = {o.criterion: o for o in outcomes}
    questions: list[dict[str, Any]] = []

    f1b = by["F1b_drop_selectivity"].measured
    f7 = by["F7_attrition_expectation"].measured
    if f1b["fired_metrics"]:
        questions.append({
            "id": "F1b-fires-on-the-mechanism-F7-predicts",
            "criterion": "F1b_drop_selectivity",
            "question": (
                "F1b fires as a **Defect** on attrition whose mechanism the same bar's F7 "
                "derivation describes in detail and treats as benign. F1b's null model is "
                "'attrition is random with respect to terrain'; F7 says the drops are "
                "int16-quantized `land_std` values flipping across the 3.0 m `degenerate` "
                "threshold — which is a terrain threshold, so drops concentrated at the "
                "flat end are the *predicted* outcome, not a departure from it. The two "
                "criteria cannot both be read literally."
            ),
            "evidence": {
                "fired_metrics": f1b["fired_metrics"],
                "observed_w1": f1b["observed_w1"],
                "null_p99": f1b["null_p99"],
                "n_dropped": f1b["n_dropped"],
                "permutations": f1b["permutations"],
                "dropped_vs_retained": evidence["per_metric_dropped_vs_retained"],
                "land_std_of_the_drops": evidence["land_std"],
                "status_composition": evidence["by_status"],
                "gate_e_comparison": (
                    "at D=2 the same rule passed with every observed W1 between a third "
                    "and a half of its null p99; the bar's own power_disclosure says D=20 "
                    "is 'only marginally' better, and D came in at "
                    f"{f1b['n_dropped']}"
                ),
            },
            "why_not_resolved_here": (
                "Reading F1b's firing as anything other than what the rule says would be "
                "this report choosing a criterion after seeing its number — the exact move "
                "the pre-registration exists to prevent. The bar assigns the Defect-first "
                "roll-up, and any re-reading of F1b's null model, to #74."
            ),
        })

    lo, hi = f7["expected_2sigma"]
    if not lo <= f7["n_dropped"] <= hi:
        questions.append({
            "id": "F7-expectation-vs-threshold",
            "criterion": "F7_attrition_expectation",
            "question": (
                "F7 passes on the rate it gates and misses the expectation it was told to "
                "be surprised against. The bar registers no rule for that combination."
            ),
            "evidence": {
                "n_dropped": f7["n_dropped"],
                "pre_registered_expectation": f7["expected_drops"],
                "expected_2sigma": [lo, hi],
                "drop_rate": f7["drop_rate"],
                "registered_max_rate": f7["max_rate"],
                "status_composition": by["F7c_ledger_status_published"].measured["counts"],
                "land_std_of_the_drops": evidence["land_std"],
            },
            "why_not_resolved_here": (
                "The bar's derivation predicts `degenerate` boundary flips and nothing "
                "else; a different composition or a different count is a claim about that "
                "derivation, and adjudicating it is the review's."
            ),
        })

    f3 = by["F3_drainage_representation"].measured
    for h in f3["gating_strata"]:
        entry = f3["per_stratum"][str(h)]
        pool_share = drainage.crosscheck[str(h)]["kept_pool_share"]
        ceiling = entry["exhaustion_e_h"]
        questions.append({
            "id": f"F3-attribution-stratum-{h}",
            "criterion": "F3_drainage_representation",
            "question": (
                f"Where stratum {h}'s r_h sits between the kept-pool proportional share "
                "and the exhaustion ceiling is the attribution the bar's unsatisfiable "
                "conjunction could not carry. It is published, never gated."
            ),
            "evidence": {
                "realized_share_r_h": entry["realized_share_r_h"],
                "kept_pool_share": pool_share,
                "exhaustion_e_h": ceiling,
                "gate": f3["min_share"],
                "fired": h in f3["hits"],
            },
            "why_not_resolved_here": (
                "The bar states how to read it and assigns the reading to nobody; the "
                "fresh map the consequence clause opens is #74's to declare."
            ),
        })

    if drainage.p75_rounding_moves_tiles:
        questions.append({
            "id": "F3-p75-display-rounding",
            "criterion": "F3_drainage_representation",
            "question": (
                "The registered pre-gate p75 no longer reproduces from the catalog closely "
                "enough to leave the drainage tail unchanged — the same display-rounding "
                "trap the n_h table fell into — and the bar's own registered tail counts "
                "are the full-precision ones, so the committed file disagrees with itself."
            ),
            "evidence": {
                "crosscheck": drainage.crosscheck,
                "vs_registered_reference": registered_reference,
                "p75_precision_sensitivity": p75_sensitivity,
            },
            "why_not_resolved_here": (
                "The bar freezes the p75, so the rule ran against the registered number as "
                "committed, and the sensitivity above says whether that choice mattered. "
                "Whether the reference itself should be re-evaluated is a bar question, "
                "and #69's own post_commit_corrections is the procedure for it."
            ),
        })

    f8 = by["F8_normalization_statistic"].measured
    if f8["branch"] == "inherit_debug":
        questions.append({
            "id": "F8-inherited-divergence",
            "criterion": "F8_normalization_statistic",
            "question": "How far the inherited statistic sits from the corpus it normalizes.",
            "evidence": f8.get("divergence_diagnostic", {}),
            "why_not_resolved_here": "The bar says this diagnostic carries no threshold.",
        })
    if f8["mismatches"] and f8["branch"] == "tier_specific":
        questions.append({
            "id": "F8-exact-on-a-float",
            "criterion": "F8_normalization_statistic",
            "question": (
                "F8's registered tolerance is **exact** and the rule compares two float64s "
                "with `!=`. Whether a recompute that reads the same pixels in a different "
                "block order can satisfy that at all is a question about the criterion, "
                "not about the corpus — and it is not one this report may answer, because "
                "any tolerance it chose would be chosen after seeing the number."
            ),
            "evidence": {
                "recorded": f8["recorded"],
                "recomputed": f8["reference"],
                "mismatches": {
                    key: {
                        **values,
                        "abs_diff": abs(values["recorded"] - values["expected"]),
                        "rel_diff": (
                            abs(values["recorded"] - values["expected"]) / values["expected"]
                            if values["expected"]
                            else float("inf")
                        ),
                        "ulps": _ulps(values["recorded"], values["expected"]),
                    }
                    for key, values in f8["mismatches"].items()
                },
                "same_pixels": same_pixels,
                "float64_epsilon": float(np.finfo(np.float64).eps),
                "what_differs_between_the_two_passes": (
                    "the builder accumulated land moments one tile at a time as it wrote; "
                    "this report accumulates them over 512-tile blocks read back from the "
                    "shards, so the additions associate differently. dataset.materialize."
                    "global_stats' own docstring puts the variance at four digits of "
                    "cancellation out of sixteen."
                ),
            },
            "why_not_resolved_here": (
                "The bar registered the tolerance; #74 owns the roll-up and any reading of "
                "what 'exact' means for a streamed float64 accumulator."
            ),
        })

    return questions


if __name__ == "__main__":
    raise SystemExit(main())
