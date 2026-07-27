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
uv run python -m dataset.materialize # full-res read of the selected windows -> data/corpus_50k/
```

`dataset.select` is **budget-parameterized** (`--budget`, default 50,000): it re-loads
the committed scaler and centroids and draws a deterministic selection list of
`window_id, stratum_id, provenance`, so a bigger tier is a bigger `--budget` against
identical strata. The selection list *is* the determinism contract — two runs from the
committed artifacts reproduce it byte-for-byte (`docs/research/d4-sampler.md`).

Phase F spent that design: the **500,000-window scale tier** is the same command at
`--budget 500000 --out artifacts/phase_f/selection_scale_500k.csv`, drawn from the same
two artifacts with no re-clustering, two runs byte-identical
(`docs/research/f-scale-draw.md`).

`dataset.materialize` is the **end-to-end build** (Phase E1) and replaces the retired
ADR-0001 TRI-quintile CLI (`dataset.build` / `dataset.pipeline`). It groups the selection
list by COG so one open serves all of that COG's windows, reads them at full resolution
across 16 worker processes, and writes:

| artifact | where | committed? |
|---|---|---|
| int16 shards | `data/corpus_50k/shards/` | no — ~5 GB, gitignored, rebuildable |
| tile manifest (v2 schema) | `data/corpus_50k/manifest.parquet` | yes |
| corpus card + fingerprint | `data/corpus_50k/corpus_meta.json` | yes |
| materialization ledger | `artifacts/phase_e/materialization_ledger.parquet` | yes |

The **ledger** carries one row per *selected* window with the cause of every drop
(`ok / zero_fill / no_land / degenerate / read_error`). It is committed because, unlike
the shards and the manifest, it cannot be re-derived from committed inputs — knowing
which windows failed requires the full-res read. Under the `accept_shortfall` attrition
contract (#50) a window that fails is recorded and **never backfilled**, so the corpus is
"50,000 selected, N materialized" and the manifest has no null-`shard_id` rows.

The pass is **resumable at COG granularity** — re-run the same command to continue from
the last checkpoint, or pass `--restart` to discard the attempt. A checkpoint is the only
commit: it flushes the open shard, rewrites the manifest and ledger, and writes
`build_state.json` last, so a kill at any instant costs at most one checkpoint of work and
can never double-book a window. `--limit-cogs N` runs a prefix as a live-path smoke check
(it leaves the build incomplete and writes no corpus card).

Two runs from the same committed inputs reproduce the manifest and the `dataset_version`
fingerprint byte-for-byte, at any executor and worker count: `ordered_futures` yields COGs
in input order while keeping the pool fed, and the shard writer stays serial in the parent
(measured in [`e1-read-cost.md`](docs/research/e1-read-cost.md) §Determinism).

## Visualize

```bash
uv run jupyter lab notebooks/visualize_corpus.ipynb
```

[`notebooks/visualize_corpus.ipynb`](notebooks/visualize_corpus.ipynb) renders the
drawn selection: the k=20 stratum centroids in raw metric units, per-stratum occupancy
against the Neyman allocation, the selection list (geo-cell spread, windows per COG,
where the 50k land on the globe), and before/after per-tile normalization through
`dataset.loader`.

It runs top-to-bottom on a fresh clone with **no network and no corpus**: the strata and
the selection come from the committed artifacts (`artifacts/phase_d/`,
`docs/research/d4-selection.json`), and the read-path figures come from a seeded 9-tile
shard fixture the notebook builds itself. Point `CORPUS_DIR` at a corpus built by
`dataset.materialize` to render that instead.

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
  shards.py       # stage 5b — sharded int16 memmap storage (resumable writer)
  materialize.py  # stage 5 — the driver: selection list -> shards + manifest + ledger (#51)
  loader.py       # stage 6 — torch read path (imports torch lazily)
  concurrency.py  # the bounded, order-preserving parallel-read primitive (#38)
  provenance.py   # git SHA + snapshot date stamped into every artifact
notebooks/        # visualize_corpus.ipynb — the Phase-0 viz artifact
docs/adr/         # architecture decision records
Dockerfile        # pinned ETL environment for a rented vast.ai box
```
