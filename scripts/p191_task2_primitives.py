"""#191: Task 2's cost primitives, measured on the laptop beside Task 1's box-measured anchor.

Two halves:
  loader  -- CPU ms/tile for Task 1's getitem (shard read + A1 fill) vs a Task 2-shaped getitem
             (shard read + uint8 label read + dihedral). Box-measured: Task 1's loader feeds
             ~2.1 ms/tile at 12 workers on a 5080.
  gpu     -- MPS ms/tile for Task 1's frozen-head step (the anchor: box-measured fwd+bwd 23.0 s
             over 20k on a 5080) and for each Task 2 step, so each Task 2 step is priced as
             anchor_box * (task2_mps / anchor_mps).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from benchmark import extract, head, masks, train  # noqa: E402
from benchmark.config import BenchmarkConfig  # noqa: E402

OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("p191_out.json")


def loader(n: int = 160) -> dict:
    cfg = BenchmarkConfig()
    tiles = train.TrainingTiles.read(cfg)
    ids = list(tiles.head_train)[:n]
    from benchmark.evaluate import TileStore

    store = TileStore(cfg.paths.corpus_dir, index=TileStore.index_for(cfg.paths.corpus_dir, ids))
    banked = train.read_bank(cfg, ids)
    rles = [banked.rle(t, 0) for t in ids]
    scratch = OUT.parent / "labels"
    scratch.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for t in ids:
        np.save(scratch / f"{t}.npy", rng.integers(0, 10, (224, 224), dtype=np.uint8))

    store.read(ids[0])  # warm handles
    t0 = time.perf_counter()
    raws = [np.array(store.read(t).raw_i16) for t in ids]
    read = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    for raw, rle in zip(raws, rles):
        train.compose_one(raw, masks.decode_rle(rle), cfg)
    task1_compose = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    for i, (t, raw) in enumerate(zip(ids, raws)):
        lab = np.load(scratch / f"{t}.npy")
        k = i % 8
        a, b = np.rot90(raw, k % 4), np.rot90(lab, k % 4)
        if k >= 4:
            a, b = a[:, ::-1], b[:, ::-1]
        np.ascontiguousarray(a), np.ascontiguousarray(b)
    task2_extra = (time.perf_counter() - t0) / n
    return {
        "tiles": n,
        "read_ms": read * 1e3,
        "task1_compose_ms": task1_compose * 1e3,
        "task2_label_dihedral_ms": task2_extra * 1e3,
        "task1_getitem_ms": (read + task1_compose) * 1e3,
        "task2_getitem_ms": (read + task2_extra) * 1e3,
    }


def gpu(batch: int = 16, reps: int = 6) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = "mps"
    cfg = BenchmarkConfig()
    ex = extract.Extractor(extract.load_spec(extract.spec_path(cfg)), cfg, device=dev)
    entry = ex.spec.encoder_for("A4-R5")
    vit = ex.encoder("A4-R5")
    rng = np.random.default_rng(0)
    raw = (rng.normal(800, 300, (batch, 224, 224))).astype(np.int16)
    valid = np.ones_like(raw, dtype=bool)
    x = torch.from_numpy(extract.normalize(*extract._as_batch(raw, valid), normalization=entry.normalization)).unsqueeze(1).to(dev)
    labels = torch.from_numpy(rng.integers(0, 10, (batch, 224, 224))).long().to(dev)

    def sync():
        torch.mps.synchronize()

    def timeit(step) -> float:
        step(); sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            step()
        sync()
        return (time.perf_counter() - t0) / (reps * batch) * 1e3

    def blocks_of(f):
        return [f.intermediates[i].transpose(1, 2).reshape(batch, 768, 14, 14).contiguous() for i in (2, 5, 8, 11)]

    res = {"batch": batch}
    ac = dict(device_type="mps", dtype=torch.bfloat16)

    # forward only (eval / §5 passes)
    res["vitb_forward_ms"] = timeit(lambda: ex.extract(raw, valid, arm="A4-R5"))

    # anchor: Task 1's §4.1 head on R5 output
    h = head.build(768).module.to(dev)
    opt = torch.optim.AdamW(h.parameters(), 1e-3)
    void = torch.zeros(batch, 1, 224, 224, device=dev)
    void[:, :, 80:140, 80:140] = 1

    def anchor():
        f = ex.extract(raw, valid, arm="A4-R5")
        with torch.autocast(**ac):
            out = h(f.output.to(dev), void)
        loss = (out.float() * void).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    res["anchor_task1_head_ms"] = timeit(anchor); print("anchor_task1_head_ms", res["anchor_task1_head_ms"], flush=True)

    # B1: LN + 1x1 conv on output, bilinear to 224, CE
    lin = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 10)).to(dev)
    optl = torch.optim.AdamW(lin.parameters(), 1e-3)

    def linear():
        f = ex.extract(raw, valid, arm="A4-R5")
        with torch.autocast(**ac):
            logits = lin(f.output.to(dev)).transpose(1, 2).reshape(batch, 10, 14, 14)
            logits = F.interpolate(logits.float(), size=224, mode="bilinear")
        loss = F.cross_entropy(logits, labels)
        optl.zero_grad(); loss.backward(); optl.step()
    res["linear_probe_ms"] = timeit(linear); print("linear_probe_ms", res["linear_probe_ms"], flush=True)

    # depth sweep: six linear probes trained off one forward
    lins = nn.ModuleList(nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 10)) for _ in range(7)).to(dev)
    optd = torch.optim.AdamW(lins.parameters(), 1e-3)

    def sweep():
        f = ex.extract(raw, valid, arm="A4-R5")
        feats = [f.intermediates[i] for i in (1, 3, 5, 7, 9, 11)] + [f.output]
        loss = 0
        with torch.autocast(**ac):
            for m, t in zip(lins, feats):
                lg = m(t.to(dev)).transpose(1, 2).reshape(batch, 10, 14, 14)
                loss = loss + F.cross_entropy(F.interpolate(lg.float(), size=224, mode="bilinear"), labels)
        optd.zero_grad(); loss.backward(); optd.step()
    res["depth_sweep_7probes_one_forward_ms"] = timeit(sweep); print("depth_sweep_7probes_one_forward_ms", res["depth_sweep_7probes_one_forward_ms"], flush=True)

    # UperNet: PANGAEA-style neck (4x, 2x, 1x, 0.5x) + terratorch UperNetDecoder, 512 channels
    from terratorch.models.decoders import UperNetDecoder

    class Upn(nn.Module):
        def __init__(self):
            super().__init__()
            self.neck = nn.ModuleList([
                nn.Sequential(nn.ConvTranspose2d(768, 768, 2, 2), nn.BatchNorm2d(768), nn.GELU(), nn.ConvTranspose2d(768, 768, 2, 2)),
                nn.ConvTranspose2d(768, 768, 2, 2),
                nn.Identity(),
                nn.MaxPool2d(2),
            ])
            self.dec = UperNetDecoder([768] * 4, channels=512, scale_modules=False, pool_scales=(1, 7))
            self.cls = nn.Conv2d(512, 10, 1)

        def forward(self, feats):
            out = self.dec([m(f).contiguous() for m, f in zip(self.neck, feats)])
            return F.interpolate(self.cls(out), size=224, mode="bilinear")

    up = Upn().to(dev)
    res["upernet_params"] = sum(p.numel() for p in up.parameters())
    optu = torch.optim.AdamW(up.parameters(), 6e-4)

    def upernet():
        f = ex.extract(raw, valid, arm="A4-R5")
        with torch.autocast(**ac):
            logits = up([b.to(dev) for b in blocks_of(f)])
        loss = F.cross_entropy(logits.float(), labels)
        optu.zero_grad(); loss.backward(); optu.step()
    res["upernet_probe_ms"] = timeit(upernet); print("upernet_probe_ms", res["upernet_probe_ms"], flush=True)

    # B3: encoder unfrozen + UperNet
    for p in vit.parameters():
        p.requires_grad_(True)
    vit.train()
    optf = torch.optim.AdamW(list(vit.parameters()) + list(up.parameters()), 1e-4)

    def finetune():
        with torch.autocast(**ac):
            _, inter = extract._forward_blocks(vit, x)
            feats = [inter[i][:, 1:].transpose(1, 2).reshape(batch, 768, 14, 14).contiguous() for i in (2, 5, 8, 11)]
            logits = up(feats)
        loss = F.cross_entropy(logits.float(), labels)
        optf.zero_grad(); loss.backward(); optf.step()
    res["finetune_vitb_upernet_ms"] = timeit(finetune); print("finetune_vitb_upernet_ms", res["finetune_vitb_upernet_ms"], flush=True)
    for p in vit.parameters():
        p.requires_grad_(False)
    vit.eval()
    del optf
    torch.mps.empty_cache()

    # B9: terramind_v1_base forward on raw DEM (weights random: the cost is the architecture)
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_base

    tm = terramind_v1_base(pretrained=False, modalities=["DEM"], merge_method=None).to(dev).eval()

    def b9():
        with torch.no_grad(), torch.autocast(**ac):
            tm({"DEM": x})
    res["b9_forward_ms"] = timeit(b9); print("b9_forward_ms", res["b9_forward_ms"], flush=True)

    def r5_raw():
        with torch.no_grad(), torch.autocast(**ac):
            extract._forward_blocks(vit, x)
    res["r5_raw_forward_ms"] = timeit(r5_raw); print("r5_raw_forward_ms", res["r5_raw_forward_ms"], flush=True)
    return res


if __name__ == "__main__":
    part = sys.argv[1]
    out = loader() if part == "loader" else gpu()
    print(json.dumps(out, indent=1))
    OUT.write_text(json.dumps(out, indent=1))
