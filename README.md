# DEMAE

A self-supervised **ViT-MAE** pretrained on global elevation (Copernicus DEM
GLO-30), fine-tuned for terrain-analysis tasks. Solo applied-research / portfolio
project. See [`DEMAE_Project_Plan.md`](DEMAE_Project_Plan.md) for the full plan and
[`CONTEXT.md`](CONTEXT.md) for the vocabulary.

**Status:** Phase 0 — building the reproducible DEM data pipeline. Progress is
charted on the wayfinder map (GitHub issue #1).

## Quickstart

```bash
uv sync                                   # build the venv from uv.lock
uv run python -c "import rasterio; import dataset; print('ok')"
uv run jupyter lab                         # dev tools (viz notebook lands in Phase 0)
```

Requires [`uv`](https://docs.astral.sh/uv/) (`brew install uv`). No system GDAL is
needed — `rasterio` wheels bundle it.

## Layout

```
src/dataset/      # the Phase-0 pipeline package (stage-per-module skeleton)
  config.py       # settled constants (source bucket, tile size, normalization)
  discovery.py    # stage 1 — COG discovery + land masking
  tiling.py       # stage 2 — 224x224 window extraction
  stratify.py     # stage 3 — TRI stratification
  normalize.py    # stage 4 — per-tile standardization
  manifest.py     # stage 5a — tile manifest
  shards.py       # stage 5b — corpus storage
  loader.py       # stage 6 — dataloader
docs/adr/         # architecture decision records
```

Most stage modules are intentionally stubs — their design is still being resolved
on the wayfinder map. Each stub names the ticket that fills it in.
