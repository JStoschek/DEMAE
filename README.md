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

## Build the corpus

The corpus is built by the ADR-0004 metric-basis pipeline, one stage per module —
each a `python -m dataset.<stage>` entry point that reads the previous stage's
committed artifact:

```bash
uv run python -m dataset.catalog     # global x8-overview scoring pass -> data/catalog/
uv run python -m dataset.gates       # apply the locked quality gates -> data/catalog_gated/
uv run python -m dataset.frame       # fit + persist the standardization scaler (#40)
uv run python -m dataset.centroids   # fit + persist the k=20 stratum centroids (#41)
uv run python -m dataset.select      # allocate + FSCS -> artifacts/phase_d/selection_debug_50k.csv
```

`dataset.select` is **budget-parameterized** (`--budget`, default 50,000): it re-loads
the committed scaler and centroids and draws a deterministic selection list of
`window_id, stratum_id, provenance`, so Phase F re-runs the same code at 200–500k against
identical strata. The selection list *is* the determinism contract — two runs from the
committed artifacts reproduce it byte-for-byte (`docs/research/d4-sampler.md`).

The **end-to-end materialize** — reading the selected windows at full resolution into
int16 `shards/` + `manifest.parquet` + `corpus_meta.json` — is **Phase E** (consumes the
selection list; not yet built). The ADR-0001 TRI-quintile build CLI it replaces
(`dataset.build` / `dataset.pipeline`) was retired with the metric-basis sampler.

## Visualize

```bash
uv run jupyter lab notebooks/visualize_corpus.ipynb
```

[`notebooks/visualize_corpus.ipynb`](notebooks/visualize_corpus.ipynb) renders
sampled terrain tiles, the stratification distribution, and before/after per-tile
normalization. (It predates the ADR-0004 metric basis and the retirement of the
TRI-quintile build path, and is pending a refresh to the current pipeline.)

## Reproduce on a rented box

The [`Dockerfile`](Dockerfile) pins the ETL environment (uv + Python 3.12 +
rasterio wheels, everything else from `uv.lock`) so the live-read stages reproduce on a
same-region vast.ai box — the cloud-burst escape hatch for transatlantic COG reads. Each
pipeline stage is a `python -m dataset.<stage>` command passed to the image.

```bash
docker build -t demae-etl .
docker run --rm -v "$PWD/data:/data" demae-etl -m dataset.catalog   # the global scoring pass
```

## Layout

```
src/dataset/      # the Phase-0 pipeline package (stage-per-module)
  config.py       # settled constants (source bucket, tile size, normalization)
  discovery.py    # stage 1 — COG discovery
  geometry.py     # pure key / lat-band / geocell / window-grid math
  tiling.py       # stage 2 — 224x224 window reads (swappable CogReader)
  screen.py       # the exact-zero land / zero-fill pixel screen (ADR-0001 §3)
  catalog.py      # stage 3a — global metric scoring pass -> raw candidate catalog
  gates.py        # stage 3b — the locked quality gates as filter columns
  frame.py        # stage 3c — the stratification frame k-means fits + the scaler (#40)
  centroids.py    # stage 3d — the k=20 stratum centroids (#41)
  select.py       # stage 3e — allocate + FSCS + emit the selection list (#44)
  normalize.py    # stage 4 — clip, land-only stats, degenerate drop, int16
  manifest.py     # stage 5a — Parquet manifest + corpus_meta.json fingerprint
  shards.py       # stage 5b — sharded int16 memmap storage
  loader.py       # stage 6 — torch read path (imports torch lazily)
  concurrency.py  # the bounded, order-preserving parallel-read primitive (#38)
  provenance.py   # git SHA + snapshot date stamped into every artifact
notebooks/        # visualize_corpus.ipynb — the Phase-0 viz artifact
docs/adr/         # architecture decision records
Dockerfile        # pinned ETL environment for a rented vast.ai box
```
