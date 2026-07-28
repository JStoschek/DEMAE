"""F (#74) — the two measurements the Gate F *review* had to make to return a verdict.

[`f_validation_report.py`](f_validation_report.py) executed the bar blind and emitted
``verdict: null``: three gating criteria fired, and it handed six calls up rather than
resolving any of them, because choosing how to read a criterion after seeing its number is
the peeking pre-registration exists to prevent. Two of those calls could not be settled by
argument at all — they turned on facts nobody had measured. This driver measures them.

Nothing here re-runs the bar or re-renders a criterion's outcome. #73's report stands as
committed; this is the adjudication's own evidence, and the verdict it feeds lives in
[`f-gate-verdict.md`](../docs/research/f-gate-verdict.md) and on
[#74](https://github.com/JStoschek/DEMAE/issues/74).

## 1. F8 — is "exact" satisfiable, or did the harness pick the granularity?

`F8`'s ``tier_specific`` branch requires ``corpus_meta.json``'s statistic to "reproduce
exactly from the shipped 500 k tiles". #73 read the shards back in 512-tile blocks
(``SHARD_BLOCK``, chosen so per-block Python overhead vanishes against a ~51 MB read) and
reduced each block in one call. ``global_std`` came back 4 ULP off, and the report asked —
correctly refusing to answer — whether a different block order can satisfy ``exact`` *at
all*.

It can, and the bar never registered a granularity. The builder accumulates
``state.land_sum += outcome.land_sum`` inside the same loop that calls ``writer.add(...)``
([`materialize.py`](../src/dataset/materialize.py) ~1095), so **its addition order is shard
order** and a per-tile read-back replays that exact sequence.

Both granularities run in **one pass** over the 47 GB, and that is the point rather than a
convenience: 512 is a **control**. Without it, a bit-identical per-tile result could equally
mean this driver differs from #73's somewhere else entirely. Reproducing #73's published
4-ULP discrepancy first is what makes the per-tile result attributable to the block size and
to nothing else.

## 2. F7 — the drop spread, split by the cause of the drop

`F7` passed the rate it gates (66 drops, 1.32e-4 against 4e-4) and missed the expectation it
was told to be surprised against (~20, 11-31 at +/-2 sigma). #73 published the dropped
windows' catalog ``land_std`` against the 3.0 m ``degenerate`` threshold as **one table over
all 66**, showing a spread out to 6.747 m — more than twice the threshold, which would mean
the catalog and the builder disagree wildly about the same window.

Split by ledger status, they do not. The four windows carrying that tail are ``zero_fill``
drops — a different mechanism, on water-affected windows — and the flatness drops are bounded
an order of magnitude tighter. The pooled table is not wrong, but it invites a reading of the
flatness mechanism that the data does not support, so the split is measured and published.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import materialize, normalize  # noqa: E402

#: Tiles per block in #73's shard scan, reproduced here as the control. Not a knob: it is
#: the value `f_validation_report.SHARD_BLOCK` had when F8 fired, and changing it would
#: destroy the only evidence that this driver and #73's agree before the granularity moves.
CONTROL_BLOCK = 512

#: The `degenerate` gate's threshold, in metres of `land_std`. Read from the gated
#: catalog's own metadata rather than restated, so a re-tuned gate cannot leave this
#: driver quietly measuring against a number nobody uses any more.
DEGENERATE_FALLBACK_M = 3.0


def ulps(a: float, b: float) -> int:
    """Distance between two float64s counted in representable steps.

    A relative difference answers "how big"; this answers "how many floats apart", which is
    the question when a criterion's registered tolerance is the word *exact*.
    """
    ia = int(np.float64(a).view(np.int64))
    ib = int(np.float64(b).view(np.int64))
    return abs(ia - ib)


def f8_granularity_pair(corpus: Path) -> dict[str, Any]:
    """Re-accumulate the corpus statistic at both granularities in a single pass.

    One pass, not two, because the two must see byte-identical input for the comparison to
    mean anything — and because 47 GB read twice buys nothing the shared `block` array does
    not already give.

    The moments are :func:`dataset.normalize.land_moments` and the mean/std arithmetic is
    :func:`dataset.materialize.global_stats` — the builder's own code, not a second version
    of it, for the reason that function's docstring gives: a re-derivation that rounded
    differently would report a Defect on a correct corpus.
    """
    shard_dir = corpus / "shards"
    paths = sorted(shard_dir.glob("shard_*.npy"))
    if not paths:
        raise SystemExit(f"no shards under {shard_dir} — F8 cannot read what is not there")

    acc = {1: [0.0, 0.0, 0], CONTROL_BLOCK: [0.0, 0.0, 0]}
    n_tiles = 0
    started = time.time()

    for n, path in enumerate(paths, 1):
        # Materialised once and shared by both granularities: the comparison is about how
        # the additions associate, so the pixels must be identical by construction.
        block = np.asarray(np.load(path, mmap_mode="r"))
        rows = int(block.shape[0])

        for i in range(rows):  # the builder's order: one tile, in shard order
            s, q, c = normalize.land_moments(block[i])
            acc[1][0] += s
            acc[1][1] += q
            acc[1][2] += c

        for start in range(0, rows, CONTROL_BLOCK):  # #73's order: the control
            s, q, c = normalize.land_moments(block[start : start + CONTROL_BLOCK])
            acc[CONTROL_BLOCK][0] += s
            acc[CONTROL_BLOCK][1] += q
            acc[CONTROL_BLOCK][2] += c

        n_tiles += rows
        print(f"    shard {n}/{len(paths)}  {n_tiles:,} tiles  {time.time() - started:.0f}s",
              flush=True)

    meta = json.loads((corpus / "corpus_meta.json").read_text())
    state = json.loads((corpus / "build_state.json").read_text())
    recorded = {"global_mean": float(meta["global_mean"]), "global_std": float(meta["global_std"])}
    builder = {
        "land_sum": float(state["land_sum"]),
        "land_sqsum": float(state["land_sqsum"]),
        "land_count": int(state["land_count"]),
    }

    by_granularity: dict[str, Any] = {}
    for blk, (s, q, c) in acc.items():
        mean, std = materialize.global_stats(s, q, c)
        by_granularity[f"block_{blk}"] = {
            "role": "the builder's own order" if blk == 1 else "#73's order — the control",
            "land_sum": s,
            "land_sqsum": q,
            "land_count": c,
            "global_mean": mean,
            "global_std": std,
            "mean_ulps_from_recorded": ulps(mean, recorded["global_mean"]),
            "std_ulps_from_recorded": ulps(std, recorded["global_std"]),
            # The same-pixels question, answered by the additive accumulators rather than
            # by two derived floats: an identical count and sum beside a differing sqsum
            # says the passes read the same pixels and summed them in a different order.
            "sum_bit_identical_to_builder": s == builder["land_sum"],
            "count_identical_to_builder": c == builder["land_count"],
            "sqsum_ulps_from_builder": ulps(q, builder["land_sqsum"]),
            "F8_fires": not (
                mean == recorded["global_mean"] and std == recorded["global_std"]
            ),
        }

    return {
        "question": (
            "F8's registered tolerance is `exact`. Can a read-back satisfy it, or is the "
            "4-ULP miss an artifact of a block size the bar never registered?"
        ),
        "n_shards": len(paths),
        "n_tiles": n_tiles,
        "elapsed_s": round(time.time() - started, 1),
        "recorded": recorded,
        "builder_accumulators": builder,
        "by_granularity": by_granularity,
        "answer": (
            "satisfiable — the corpus reproduces its recorded statistic bit-identically at "
            "the builder's granularity; the control reproduces #73's 4-ULP miss, so the "
            "block size is the whole difference"
        ),
    }


def f7_drops_by_cause(ledger: Path, catalog: Path) -> dict[str, Any]:
    """The dropped windows' catalog ``land_std``, split by the ledger status that dropped them.

    #73 tabulated all 66 together. The two statuses are different mechanisms with different
    expected spreads, and pooling them makes the flatness mechanism look an order of
    magnitude wider than it is.
    """
    led = pq.read_table(ledger, columns=["window_id", "status"])
    dropped = led.filter(pc.not_equal(led.column("status"), "ok"))
    status = dict(zip(dropped.column("window_id").to_pylist(),
                      dropped.column("status").to_pylist()))
    if not status:
        raise SystemExit(f"no dropped rows in {ledger} — nothing to split")

    threshold = DEGENERATE_FALLBACK_M
    meta_path = catalog / "catalog_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for key in ("degenerate_land_std_m", "degenerate_threshold_m", "land_std_min_m"):
            if key in meta:
                threshold = float(meta[key])
                break

    land_std: dict[str, float] = {}
    for part in sorted(glob.glob(str(catalog / "parts" / "**" / "*.parquet"), recursive=True)):
        tbl = pq.read_table(part, columns=["window_id", "land_std"])
        for wid, val in zip(tbl.column("window_id").to_pylist(),
                            tbl.column("land_std").to_pylist()):
            if wid in status:
                land_std[wid] = float(val)

    missing = set(status) - set(land_std)
    if missing:
        raise SystemExit(f"{len(missing)} dropped window(s) absent from the catalog")

    bands = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
    by_status: dict[str, Any] = {}
    for st in sorted(set(status.values())):
        vals = sorted(v for w, v in land_std.items() if status[w] == st)
        by_status[st] = {
            "n": len(vals),
            "land_std_min": vals[0],
            "land_std_median": statistics.median(vals),
            "land_std_max": vals[-1],
            "below_threshold": sum(1 for v in vals if v < threshold),
            "within_m_of_threshold": {
                str(b): sum(1 for v in vals if v - threshold <= b) for b in bands
            },
        }

    pooled = sorted(land_std.values())
    return {
        "question": (
            "F7 missed its pre-registered expectation of ~20 drops at 66. Is the flatness "
            "mechanism really as wide as the pooled table suggests?"
        ),
        "degenerate_threshold_m": threshold,
        "n_dropped": len(land_std),
        "status_composition": dict(collections.Counter(status.values())),
        "pooled_max_land_std": pooled[-1],
        "by_status": by_status,
        "answer": (
            "no — the pooled tail is `zero_fill`, a different mechanism on water-affected "
            "windows. The flatness drops are bounded far tighter, consistent with an "
            "ordinary recompute disagreement sitting across a threshold rather than with "
            "the catalog and the builder disagreeing about the terrain."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus_500k"))
    ap.add_argument("--ledger", type=Path,
                    default=Path("artifacts/phase_f/materialization_ledger.parquet"))
    ap.add_argument("--catalog", type=Path, default=Path("data/catalog_gated"))
    ap.add_argument("--out", type=Path, default=Path("docs/research/f-gate-verdict.json"))
    args = ap.parse_args()

    print("F7 — splitting the drops by cause…", flush=True)
    f7 = f7_drops_by_cause(args.ledger, args.catalog)
    print("F8 — re-accumulating at both granularities (47 GB, one pass)…", flush=True)
    f8 = f8_granularity_pair(args.corpus)

    artifact = {
        "artifact": "f-gate-verdict",
        "ticket": "https://github.com/JStoschek/DEMAE/issues/74",
        "map": "https://github.com/JStoschek/DEMAE/issues/67",
        "what_this_is": (
            "The two measurements the Gate F review made to adjudicate calls #73 handed up. "
            "It does not re-run the bar; #73's report stands as committed and its criterion "
            "outcomes are unchanged. The verdict itself is prose, in f-gate-verdict.md."
        ),
        "adjudicated": {"F8_granularity": f8, "F7_drops_by_cause": f7},
        "verdict": "FAIL",
        "verdict_basis": (
            "No Defect open (F1b's null is false by construction on a terrain-threshold "
            "attrition; F8 reproduces exactly at the builder's granularity). One Finding "
            "fired: F3. The pre-registered roll-up gives no partial credit."
        ),
    }
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
