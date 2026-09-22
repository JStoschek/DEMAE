#!/usr/bin/env python3
"""Write §9's hillshade-panel fill exhibits for the arms whose weights are on this box.

    PYTHONPATH=src python scripts/panel_exhibits.py                 # A1 and A7, from local weights
    PYTHONPATH=src python scripts/panel_exhibits.py --arm A1        # one arm

`benchmark.report` draws the panel from whatever pixels the box holds and **never computes a
fill itself** (`report.panel_exhibit_dir`): a fill is a 224x224 raster and the results parquet
holds metrics, so the four arm columns can only come from an exhibit saved beside the file. No
`run` saves one — the memo (#170) shipped with all four columns reading *no fill exhibit* — and
this script is the smaller honest fix: the three panel tiles' fills, for the arms whose weights
are here, with the weights' md5 recorded beside the rasters so the panel is answerable to them.

What it does **not** do is score anything or touch the parquet. It reads the same committed eval
mask list `run` reads, picks the same three tiles `report` picks (`report.panel_tiles`, so the
exhibits are for the tiles the figure will actually draw), and fills the same blob void the
`voided` column shows. A1 is `benchmark.fill.fill` on the arm's declared recipe. A7 is §4.4's
UNet rebuilt by `benchmark.unet.build` and loaded from the seed-0 `best.pt` its run left in
`data/task1/heads/`, through `benchmark.train.predict` — the eval path's own composition, not a
second one.

A3-R5 and A5 are not written here because their weights are not on this box (A3's fine-tune and
A5's head were trained on a rented box that was torn down, and `p159_deploy.sh` brings no
`best.pt` home). Their columns keep rendering as the labelled absence they are.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark import fill, report, train, unet  # noqa: E402
from benchmark.config import BenchmarkConfig  # noqa: E402
from benchmark.evaluate import TileStore  # noqa: E402

A7_WEIGHTS = Path("data/task1/heads/A7-20k-seed0/best.pt")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def a1_filler(cfg: BenchmarkConfig):
    def run(raw_i16: np.ndarray, void: np.ndarray) -> np.ndarray:
        return fill.fill(np.asarray(raw_i16, dtype=np.float64), void, recipe=cfg.fill)

    return run, {"weights": None}


def a7_filler(cfg: BenchmarkConfig, weights: Path):
    import torch

    if not weights.is_file():
        raise FileNotFoundError(f"{weights}: A7's seed-0 best.pt is not on this box")
    device = train.resolve_device(cfg)
    built = unet.build(cfg.unet, seed=0)
    state = torch.load(weights, map_location="cpu", weights_only=False)
    built.module.load_state_dict(state["state_dict"])
    module = built.module.to(device).eval()
    stack = train.ScratchStack(module=module, arm="A7", device=device)

    def run(raw_i16: np.ndarray, void: np.ndarray) -> np.ndarray:
        return train.predict([raw_i16], [void], cfg, stack)[0]

    return run, {
        "weights": str(weights),
        "weights_md5": _md5(weights),
        "epoch": int(state["epoch"]),
        "val_rmse": float(state["val_rmse"]),
        "device": device,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=None, help="results parquet (default: the config's)")
    parser.add_argument("--arm", action="append", choices=["A1", "A7"], default=None)
    parser.add_argument("--a7-weights", default=str(A7_WEIGHTS))
    args = parser.parse_args(argv)

    cfg = BenchmarkConfig()
    results = Path(args.results or cfg.paths.results_path)
    frame = report.read(results)
    tiles = report.panel_tiles(frame)
    voids = report._panel_voids(cfg, tiles)
    missing = [t for t in tiles if t not in voids]
    if missing:
        raise SystemExit(f"no committed blob mask for panel tile(s) {missing}")
    store = TileStore(cfg.paths.corpus_dir)
    out_dir = report.panel_exhibit_dir(results)
    out_dir.mkdir(parents=True, exist_ok=True)

    fillers = {}
    for arm in args.arm or ["A1", "A7"]:
        fillers[arm] = a1_filler(cfg) if arm == "A1" else a7_filler(cfg, Path(args.a7_weights))

    sidecar_path = out_dir / "exhibits.json"
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
    for arm, (run, provenance) in fillers.items():
        for tile_id in tiles:
            tile = store.read(tile_id)
            void = voids[tile_id]
            filled = np.asarray(run(tile.raw_i16, void), dtype=np.float32)
            assert filled.shape == void.shape and np.isfinite(filled[void]).all()
            np.testing.assert_array_equal(filled[~void], tile.raw_i16[~void].astype(np.float32))
            path = out_dir / f"{arm}--{tile_id}.npy"
            np.save(path, filled)
            err = float(np.sqrt(np.mean((filled[void] - tile.raw_i16[void]) ** 2)))
            sidecar[path.name] = {
                "arm": arm,
                "tile_id": tile_id,
                "void_px": int(void.sum()),
                "rmse_m_on_this_void": round(err, 3),
                "written": date.today().isoformat(),
                "script": "scripts/panel_exhibits.py",
                **provenance,
            }
            print(f"{arm:<6} {tile_id}  void {int(void.sum()):>6} px  rmse {err:8.3f} m  -> {path}")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(f"sidecar          {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
