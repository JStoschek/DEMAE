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
```

Requires [`uv`](https://docs.astral.sh/uv/) (`brew install uv`). No system GDAL is
needed — `rasterio` wheels bundle it.

## Build a corpus

The pipeline is driven by one smoke-gated CLI (`python -m dataset.build`, aka
`demae-build`): it runs a tiny tier end-to-end to validate the run, then ships the
full corpus.

```bash
# Reproduce a sample end-to-end with ZERO network (deterministic synthetic bucket):
uv run python -m dataset.build --source synthetic      # -> data/sample/

# Ship the ~50k-tile Phase-0 debug corpus from the live Copernicus bucket:
uv run python -m dataset.build --source live           # -> data/corpus/

# Before a live run, project throughput and get a local-vs-cloud-burst verdict:
uv run python -m dataset.throughput
```

Each build writes `manifest.parquet`, `corpus_meta.json` (with a `dataset_version`
fingerprint), and int16 `shards/`. The corpus itself is gitignored — it's large
and fully reproducible from the CLI + `uv.lock`.

## Visualize

```bash
uv run jupyter lab notebooks/visualize_corpus.ipynb
```

[`notebooks/visualize_corpus.ipynb`](notebooks/visualize_corpus.ipynb) renders
sampled terrain tiles, the TRI/stratification distribution, and before/after
per-tile normalization. It auto-builds the synthetic sample if no corpus exists,
so it runs top-to-bottom with no network.

## Reproduce on a rented box

The [`Dockerfile`](Dockerfile) pins the ETL environment (uv + Python 3.12 +
rasterio wheels, everything else from `uv.lock`) so a live build reproduces on a
same-region vast.ai box — the cloud-burst escape hatch for transatlantic COG reads.

```bash
docker build -t demae-etl .
docker run --rm -v "$PWD/data:/data" demae-etl --source synthetic --out-dir /data/sample
```

## Layout

```
src/dataset/      # the Phase-0 pipeline package (stage-per-module)
  config.py       # settled constants (source bucket, tile size, normalization)
  discovery.py    # stage 1 — COG discovery
  geometry.py     # pure key / lat-band / geocell / window-grid math
  tiling.py       # stage 2 — 224x224 window reads (swappable CogReader)
  stratify.py     # stage 3 — ocean screen, TRI, quantile strata, quota fill
  normalize.py    # stage 4 — clip, land-only stats, degenerate drop, int16
  manifest.py     # stage 5a — Parquet manifest + corpus_meta.json fingerprint
  shards.py       # stage 5b — sharded int16 memmap storage
  loader.py       # stage 6 — torch read path (imports torch lazily)
  pipeline.py     # end-to-end orchestrator
  throughput.py   # smoke-sample measurement + cloud-burst decision
  synthetic.py    # deterministic zero-network sample bucket
  build.py        # the smoke-gated corpus build CLI (python -m dataset.build)
notebooks/        # visualize_corpus.ipynb — the Phase-0 viz artifact
docs/adr/         # architecture decision records
Dockerfile        # pinned ETL environment for a rented vast.ai box
```
