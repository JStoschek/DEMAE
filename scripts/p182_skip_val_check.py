"""#182's val check: A4-R5's registered head against the skip head, one seed, on this laptop.

Trains each arm through :func:`benchmark.train.train_head` — the pass `run` uses, so the tiles,
the bank, the seed and §4.1's frozen recipe are the ones a cell would get — and then scores the
trained head over §3.1's **val** draw, one void per val tile, through the same
:func:`benchmark.evaluate.score_filled` the eval pass uses. The test split is never opened:
:class:`benchmark.train.TrainingTiles` cannot reach it.

What this is not: a cell. Nothing here writes to ``results.parquet``, and the size rung it runs at
is chosen for this laptop rather than by §7 — the skip arms are refused below 20k by
`launch.resolve` (they are not in the label sweep), and this script reaches the pass directly to
run them at 1k. It prints per-band means over the val instances for ``rmse``, ``slope_rmse`` and
``seam_gap``, the paired difference with a bootstrap interval over val tiles, and the wall clock.

    PYTHONPATH=src uv run python scripts/p182_skip_val_check.py --size 1k --epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark import arms, evaluate, masks, metrics, train, tune
from benchmark.config import BenchmarkConfig

REPORTED = ("rmse", "slope_rmse", "seam_gap")


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arms", default="A4-R5,A4-R5-SKIP")
    parser.add_argument("--size", default="1k", choices=("1k", "5k", "20k"))
    parser.add_argument("--epochs", type=int, default=None, help="default: the recipe's own")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--micro-batch", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="data/task1/heads/p182")
    parser.add_argument("--out", default="data/task1/p182_val_check.json")
    return parser.parse_args(argv)


def _config(args: argparse.Namespace, arm: str) -> BenchmarkConfig:
    base = BenchmarkConfig(arm=arm, size=args.size, seed=args.seed)
    return base.replace(
        runtime=base.runtime.replace(
            device=args.device,
            loader_workers=args.workers,
            micro_batch=args.micro_batch,
            checkpoint_dir=args.checkpoint_dir,
        )
    )


def score_val(cfg: BenchmarkConfig, trained: train.Trained) -> list[dict]:
    """Every val instance, scored the way the eval pass scores a committed one."""
    inputs = train._training_inputs(cfg)
    bar = metrics.bar_for(cfg)
    bands = evaluate.bands_for(cfg)
    store = evaluate.TileStore(cfg.paths.corpus_dir)
    rows: list[dict] = []
    # Chunks of 32 rather than §4.1's 128: every fill is a per-tile function of its own inputs
    # (GroupNorm is per sample, LayerNorm per token), so the chunk moves no number, and 128
    # skip-head instances at 224 px under no_grad was the allocation that paged this laptop out.
    batch = min(32, cfg.head.batch_size)
    for start in range(0, len(inputs.val), batch):
        chunk = inputs.val[start : start + batch]
        tiles = [store.read(inputs.tiles.allow(item.tile_id)) for item in chunk]
        voids = [masks.decode_rle(item.mask_rle) for item in chunk]
        filled = trained.predict([t.raw_i16 for t in tiles], voids, cfg)
        for tile, item, void, out in zip(tiles, chunk, voids, filled):
            instance = evaluate.Instance(
                tile_id=item.tile_id, family=item.family, bucket=item.bucket,
                mask_rle=item.mask_rle, instance=0,
            )
            scored = evaluate.score_filled(tile, instance, void, out, bands=bands, bar=bar)
            rows.append({
                "tile_id": scored.tile_id, "family": scored.family, "bucket": scored.bucket,
                "band": scored.band, **{k: float(v) for k, v in scored.metrics.items()},
            })
    return rows


def run_arm(args: argparse.Namespace, name: str) -> dict:
    cfg = _config(args, name)
    arm = arms.get(name)
    # The one gate `launch.resolve` applies that matters here: the recipe is the frozen one.
    tune.assert_frozen(cfg)
    print(f"\n=== {name}: size {args.size}, seed {args.seed}, device {cfg.runtime.device}")
    started = time.perf_counter()
    trained = train.train_head(
        cfg, arm, epochs=args.epochs, on_epoch=lambda line: print(f"  {line.line()}", flush=True)
    )
    for line in train.format_trained(trained):
        print(line)
    train_seconds = time.perf_counter() - started
    if trained.stack.device == "mps":
        # Hand the training pass's cached blocks back before scoring; the MPS allocator keeps
        # them otherwise, and this box has 16 GB shared with everything else that is open.
        import torch

        torch.mps.empty_cache()
    started = time.perf_counter()
    rows = score_val(cfg, trained)
    score_seconds = time.perf_counter() - started
    print(f"scored {len(rows)} val instances in {score_seconds:.0f}s")
    return {
        "arm": name, "config_hash": cfg.config_hash, "train_tiles": trained.tiles,
        "epochs_run": len(trained.history), "best_epoch": trained.best_epoch,
        "best_val_rmse": trained.best_val_rmse, "parameters": trained.parameters,
        "device": trained.stack.device, "precision_used": trained.precision_used,
        "train_seconds": train_seconds, "score_seconds": score_seconds,
        "epoch_seconds": [e.seconds for e in trained.history],
        "history": [e.to_dict() if hasattr(e, "to_dict") else e.__dict__ for e in trained.history],
        "rows": rows,
    }


def _band_means(rows: list[dict]) -> dict[str, dict[str, float]]:
    by_band: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_band[row["band"]].append(row)
    return {
        band: {m: float(np.mean([r[m] for r in items])) for m in REPORTED}
        for band, items in sorted(by_band.items())
    }


def _paired(a: list[dict], b: list[dict], metric: str, band: str | None, *, resamples=10_000):
    """Mean of (b - a) over the val tiles in ``band``, with a bootstrap 95% interval."""
    key = lambda r: (r["tile_id"], r["family"], r["bucket"])  # noqa: E731
    lookup = {key(r): r for r in b}
    diffs = np.array([
        lookup[key(r)][metric] - r[metric]
        for r in a
        if (band is None or r["band"] == band) and key(r) in lookup
    ])
    rng = np.random.default_rng(0)
    draws = rng.choice(diffs, size=(resamples, diffs.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi), int(diffs.size)


def report(results: list[dict]) -> None:
    base, *others = results
    print(f"\n{'arm':<12} {'tiles':>6} {'epochs':>6} {'best':>5} {'val_rmse':>9} "
          f"{'train s':>8} {'s/epoch':>8} {'score s':>8} {'params':>10}")
    for r in results:
        per_epoch = float(np.mean(r["epoch_seconds"])) if r["epoch_seconds"] else float("nan")
        print(f"{r['arm']:<12} {r['train_tiles']:>6} {r['epochs_run']:>6} {r['best_epoch']:>5} "
              f"{r['best_val_rmse']:>9.3f} {r['train_seconds']:>8.0f} {per_epoch:>8.1f} "
              f"{r['score_seconds']:>8.0f} {r['parameters']:>10,}")
    means = {r["arm"]: _band_means(r["rows"]) for r in results}
    bands = sorted(means[base["arm"]])
    for metric in REPORTED:
        unit = "m" if metric == "rmse" else "deg"
        print(f"\n{metric} ({unit}), val, band means over instances")
        print(f"{'band':<6} {'n':>5} " + "".join(f"{r['arm']:>13}" for r in results)
              + "".join(f"{'delta ' + o['arm'][:6]:>14} {'95% CI':>20}" for o in others))
        for band in [*bands, None]:
            n = sum(1 for row in base["rows"] if band is None or row["band"] == band)
            cells = []
            for r in results:
                rows = [x for x in r["rows"] if band is None or x["band"] == band]
                cells.append(float(np.mean([x[metric] for x in rows])))
            line = f"{band or 'all':<6} {n:>5} " + "".join(f"{c:>13.3f}" for c in cells)
            for o in others:
                d, lo, hi, _ = _paired(base["rows"], o["rows"], metric, band)
                line += f"{d:>+14.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>20}"
            print(line)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    names = [n.strip() for n in args.arms.split(",") if n.strip()]
    results = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name in names:
        results.append(run_arm(args, name))
        out.write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
        print(f"wrote {out}")
    report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
