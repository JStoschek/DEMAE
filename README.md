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
uv run python -m dataset.splits      # geographic train/val/test split -> artifacts/phase1/ (#77)
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

`dataset.splits` is the one Phase-1 stage in this list: it fills the manifest's `split`
column once, geographically blocked and **never randomly** (the corpus card's §4.6
warning), and commits the assignment as `artifacts/phase1/split_scale_500k.csv` plus the
geocell/COG block lists. `--fill` is opt-in; the draw itself is safe to repeat and
reproduces byte-for-byte (`docs/research/p77-split.md`).

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

## Pretrain

`python -m pretrain.launch` is **the way a run is started** — on a laptop for a smoke test,
on the rented box for the real thing:

```bash
uv run python -m pretrain.launch --corpus data/corpus_500k --epochs 400
```

It assembles the run config, prints the header a checkpoint records, confirms §7's
efficiency levers against the model it just built, and then trains — reporting samples/s,
step time and the **loader wait fraction** for every step, every epoch and the run as a
whole. Asked for no length it runs the config's own schedule to its end; `--steps N` and
`--epochs N` shorten it.

It is **the only command that trains**, and that is the point of it: one command owns
§11's checkpoints and resume, §10's whole reporting panel — val loss under a fixed mask
seed, the three alarms, the per-patch loss distribution, reconstruction grids every 10
epochs, a kNN probe every 20 — and the W&B run they all go to. Checkpoints land in
`runs/<run_id>` on both of §11's triggers (every epoch, and every 45 minutes), keeping the
last 3 plus the milestone set. **Resume is the default**: vast.ai instances vanish
permanently, so a first launch and a relaunch on a fresh box are the same command, and the
launcher prints what it found before spending anything. `--checkpoint-dir` moves the
directory, `--no-checkpoints` and `--no-resume` opt out.

Every recipe knob is reachable without a file. A run is the plan's defaults plus the knobs
it moves, and `--set` is how it moves them (repeatable, JSON-valued, applied last):

```bash
uv run python -m pretrain.launch --corpus data/corpus_500k --set masking.ratio=0.9 --set run_id=R2
```

`--config run.json` supplies the same thing as a committed partial config, `--vit-base`
selects the ViT-B/16 release shape, and **`--dry-run` does all of the assembling and none
of the spending** — the header, the data fingerprints and the step count, with nothing
trained. That is the form the pre-flight checklist is walked in before a GPU is rented.

Two knobs are worth knowing by name. `--set runtime.compile=false` drops `torch.compile`
for debugging (eager is a supported fallback, ~1.1–1.4× slower), and
`--set runtime.attention=math` drops SDPA the same way. Both are *confirmed* rather than
assumed: the run refuses to start if the attention path it reads off the built blocks is
not the one the recipe asked for, and it warns in the footer if the compiled graph was
rebuilt after warmup.

`python -m pretrain` prints the same header and trains nothing;
`python -m pretrain.train --steps N` is the minimal loop underneath the launcher — a tracer
bullet for watching a step go by, not a way to start a run: it has no monitoring, and its
checkpoints are opt-in.
`python -m pretrain.panel` draws and commits §10.2's fixed 40-tile val panel, once per
corpus tier, and `python -m pretrain.recon` renders that panel on an untrained model and
writes the two contact sheets to look at — *garbage in, garbage rendered*, which is item 5
of the pre-flight checklist. `python -m pretrain.probe --draw` commits §10.3's probe set
the same once-per-tier way, and without `--draw` takes a single probe point off a
checkpoint — or off an untrained encoder, which is the epoch-0 baseline later points are a
delta against.

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
src/pretrain/     # the Phase-1 MAE pretraining package (torch + timm live here)
  config.py       # the frozen RunConfig — every plan §2-§8 knob + the config hash
  fingerprint.py  # dataset_version / selection md5 / split md5 / git SHA
  data.py         # split-aware Dataset; asserts no test geocell reaches a train batch
  model.py        # the MAE forward pass — ViT encoder, shallow decoder, in-model masking
  augment.py      # §6's dihedral group, wrapped around the read path
  loss.py         # §4's objective — masked MSE, norm_pix, ocean/invalid weighting
  init.py         # §8's warm start: an ImageNet MAE encoder, RGB patch embed collapsed
  optim.py        # AdamW off the config, and §6's bias/norm no-decay grouping
  schedule.py     # the LR curve — linear warmup from 0, cosine to exactly 0
  train.py        # the step and the loop around it (bf16, accumulation, the seed)
  runtime.py      # §7's levers — `torch.compile`, and SDPA confirmed on the built blocks
  throughput.py   # §7.5's counters — samples/s, step time, loader wait fraction
  monitor.py      # §10 — what a run reports every epoch, and the loop that composes it all
  alarms.py       # §10.1's failure conditions, as pure functions over loss trajectories
  artifact.py     # the committed `tile_id,stratum_id` contract — one format, two artifacts
  panel.py        # §10.2's fixed 40-tile val panel — drawn once, committed, re-read by runs
  recon.py        # §10.2's grids — masked -> predicted -> truth in metres, plus hillshade
  probe.py        # §10.3's kNN probe over a frozen encoder — trends, never an absolute bar
  sink.py         # §11's metrics sink — every scalar, histogram and grid goes through here
  checkpoint.py   # §11's exact resume — weights, optimizer, five RNG streams, the config
  header.py       # `python -m pretrain` — the config + fingerprint header a run records
  launch.py       # `python -m pretrain.launch` — the one command a run is started with
notebooks/        # visualize_corpus.ipynb — the Phase-0 viz artifact
docs/adr/         # architecture decision records
Dockerfile        # pinned ETL environment for a rented vast.ai box
```
