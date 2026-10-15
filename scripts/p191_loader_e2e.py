"""#191: end-to-end DataLoader throughput, Task 1's Examples vs a Task 2-shaped dataset.

Same corpus, same tile ids, same worker count, same batch; what differs is the per-tile work and
the bytes each item carries. Run as a module path so spawn workers can import the dataset class.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from benchmark import train  # noqa: E402
from benchmark.config import BenchmarkConfig  # noqa: E402
from benchmark.evaluate import TileStore  # noqa: E402

LABELS = Path(__file__).parent / "labels"


class Task2Examples:
    def __init__(self, cfg, ids, index):
        self.cfg, self.ids, self.index, self._store = cfg, list(ids), index, None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self._store is None:
            self._store = TileStore(self.cfg.paths.corpus_dir, index=self.index)
        t = self.ids[i]
        raw = np.asarray(self._store.read(t).raw_i16)
        lab = np.load(LABELS / f"{t}.npy")
        k = i % 8
        raw, lab = np.rot90(raw, k % 4), np.rot90(lab, k % 4)
        if k >= 4:
            raw, lab = raw[:, ::-1], lab[:, ::-1]
        return np.ascontiguousarray(raw), np.ascontiguousarray(lab)


def collate2(items):
    return np.stack([a for a, _ in items]), np.stack([b for _, b in items])


def main():
    from torch.utils.data import DataLoader

    n, workers, batch = int(sys.argv[1]), int(sys.argv[2]), 128
    cfg = BenchmarkConfig()
    tiles = train.TrainingTiles.read(cfg)
    ids = list(tiles.head_train)[:n]
    index = TileStore.index_for(cfg.paths.corpus_dir, ids)
    rng = np.random.default_rng(0)
    LABELS.mkdir(exist_ok=True)
    for t in ids:
        p = LABELS / f"{t}.npy"
        if not p.exists():
            np.save(p, rng.integers(0, 10, (224, 224), dtype=np.uint8))
    banked = train.read_bank(cfg, ids)
    items = [(t, banked.rle(t, 0)) for t in ids]
    out = {"tiles": n, "workers": workers, "batch": batch}
    for name, ds, col in (
        ("task1", train.Examples(cfg, items, tiles, index), train.collate),
        ("task2", Task2Examples(cfg, ids, index), collate2),
    ):
        for rep in range(2):  # second rep is the warm-cache number
            t0 = time.perf_counter()
            for _ in DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=col):
                pass
            out[f"{name}_ms_per_tile_rep{rep}"] = (time.perf_counter() - t0) / n * 1e3
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
