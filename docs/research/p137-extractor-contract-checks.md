# P137 — the two measurements ADR-0012 rests on

[ADR-0012](../adr/0012-frozen-feature-extractor-contract.md) takes two decisions on numbers rather
than on argument, and the numbers are small enough to be re-run in under a minute. This file is
here so they are reproducible rather than quoted — the corpus card's rule (`derived, never typed`)
applied one layer down.

Both were run on 2026-08-11, macOS, Apple MPS, against `data/corpus_500k` and the committed R5
release encoder at `.release-work/release-R5-v2/encoder.pt`
(`encoder_md5 3ec5ebc72fa28c43fe88f67692d9b41a`, 400 epochs completed, `config_hash 6e3f6a8d29e30ef6`).

## 1. Recomputed per-tile stats against the manifest — ADR-0012 Decision 2

**Question.** ADR-0012 makes the extractor recompute per-tile mean/std from the passed valid-pixel
mask, on one code path, rather than reading the manifest's stored columns when it could. Phase 1's
loader ([`src/dataset/loader.py`](../../src/dataset/loader.py)) reads the stored columns, so the
question is what that costs in continuity with the features R5 was pretrained under.

**Result**, over 500 tiles drawn from three shards, 18 of them containing ocean:

| | max | median |
|---|---:|---:|
| `mean` absolute difference | **3.637e-4 m** | 8.719e-6 m |
| `std` absolute difference | 2.773e-5 m | 5.588e-7 m |
| `std` relative difference | **1.059e-7** | — |

Float32 accumulation order, not a definitional gap: recomputing from `raw != 0` *is* the ETL's land
definition. This is the equivalence ADR-0012 ships as a test — so that if the ETL's land definition
ever drifts, Task 1 finds it rather than Task 2.

```python
# PYTHONPATH=src uv run python
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
t = pq.read_table('data/corpus_500k/manifest.parquet',
                  columns=['shard_id', 'row_index', 'mean', 'std', 'land_frac'])
sid, row = t.column('shard_id').to_numpy(), t.column('row_index').to_numpy()
mm, ss = t.column('mean').to_numpy(), t.column('std').to_numpy()
rng = np.random.default_rng(0)
shards = sorted(set(sid[:20000].tolist()))[:3]
sel = rng.choice(np.flatnonzero(np.isin(sid, shards)), size=500, replace=False)
sd, cache = Path('data/corpus_500k/shards'), {}
def arr(s):
    if s not in cache:
        cache[s] = np.load(next(sd.glob(f'*{s:05d}*.npy')), mmap_mode='r')
    return cache[s]
dm, ds = [], []
for i in sel:
    a = arr(int(sid[i]))[int(row[i])].astype(np.float64)
    land = a[a != 0]
    dm.append(land.mean() - mm[i]); ds.append(land.std() - ss[i])
dm, ds = np.array(dm), np.array(ds)
print(np.abs(dm).max(), np.median(np.abs(dm)), np.abs(ds).max(), np.abs(ds / ss[sel]).max())
```

## 2. fp32 against bf16 autocast on the R5 encoder — ADR-0012 Decision 6

**Question.** ADR-0012 fixes fp32 as the contract's *reference* tensor but lets each task declare
its own compute dtype. That is only defensible if the difference is immaterial by the bar this map
already set: [#134](https://github.com/JStoschek/DEMAE/issues/134) fixed **mean per-token cosine
≥ 0.99** as immaterial, and rejected the bicubic `pos_emb` at 0.9708.

**Result**, 64 corpus tiles, 12,544 patch tokens (CLS excluded), `forward_features` under
`torch.autocast('mps', bfloat16)` against plain fp32:

| | value |
|---|---:|
| per-token cosine, mean | **0.999968** |
| per-token cosine, median | 0.999980 |
| per-token cosine, p01 | 0.999809 |
| per-token cosine, min | 0.999431 |
| relative RMS | 0.81% |

Three orders of magnitude the safe side of the 0.99 bar. Note also that p120's **0.179 ms** frozen
forward was itself measured under bf16 autocast (p120 §Device), so "fp32 costs nothing" had never
actually been priced — it is cents either way on a program whose GPU line is ~$1, which is why this
is decided on materiality rather than on cost.

**What this does not measure, and the build session owes.** It is the **DEMAE family only** —
terratorch is not installed in any live worktree, so the TerraMind arm's fp32-vs-bf16 agreement does
not exist — and it is **MPS, not CUDA**. ADR-0012 names both gaps rather than generalizing this
number across them.

```python
# PYTHONPATH=src uv run python
import torch, timm, numpy as np, pyarrow.parquet as pq
from pathlib import Path
state = torch.load('.release-work/release-R5-v2/encoder.pt', map_location='cpu')
m = timm.create_model('vit_base_patch16_224', pretrained=False, in_chans=1, num_classes=0)
m.load_state_dict(state, strict=True); m.eval()
t = pq.read_table('data/corpus_500k/manifest.parquet',
                  columns=['shard_id', 'row_index', 'mean', 'std'])
sid, row = t.column('shard_id').to_numpy(), t.column('row_index').to_numpy()
mu, sg = t.column('mean').to_numpy(), t.column('std').to_numpy()
sd, cache = Path('data/corpus_500k/shards'), {}
def arr(s):
    if s not in cache:
        cache[s] = np.load(next(sd.glob(f'*{s:05d}*.npy')), mmap_mode='r')
    return cache[s]
idx = np.flatnonzero(np.isin(sid, sorted(set(sid[:20000].tolist()))[:2]))[:64]
x = np.stack([(arr(int(sid[i]))[int(row[i])].astype(np.float32) - np.float32(mu[i]))
              / np.float32(max(sg[i], 1e-6)) for i in idx])[:, None].astype(np.float32)
x, dev = torch.from_numpy(x), 'mps'
m, x = m.to(dev), x.to(dev)
with torch.no_grad():
    f32 = m.forward_features(x).float()
    with torch.autocast(dev, dtype=torch.bfloat16):
        fbf = m.forward_features(x).float()
a, b = f32[:, 1:].cpu(), fbf[:, 1:].cpu()
cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).flatten()
print(cos.mean(), cos.min(), cos.quantile(0.01), cos.median(),
      (a - b).pow(2).mean().sqrt() / a.pow(2).mean().sqrt())
```

Note the harness detail: the released `encoder.pt` loads **strictly** into timm's
`vit_base_patch16_224` at `in_chans=1` (`<All keys matched successfully>`), which is the property
[`pretrain.release`](../../src/pretrain/release.py) exists to guarantee, and timm's
`forward_features` is CLS + twelve blocks + final norm — the same composition as
`MaskedAutoencoder.encode`. That is what makes this a measurement of the shipped encoder rather
than of a re-implementation of it.
