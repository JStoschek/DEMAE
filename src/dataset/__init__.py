"""DEMAE Phase-0 data pipeline.

Turns the global Copernicus DEM GLO-30 collection into a terrain-diverse,
per-tile-normalized corpus of 224x224 elevation tiles for ViT-MAE pretraining.

The pipeline implements the locked Phase-0 design (ADRs 0001-0004). Stage map
(data flow, top to bottom):

    discovery -> tiling -> catalog -> gates -> frame -> centroids -> select
              -> normalize -> manifest/shards -> loader

- :mod:`dataset.discovery`  — list released COGs from the public bucket (§stage 1)
- :mod:`dataset.geometry`   — pure key/lat-band/geocell/window-grid math
- :mod:`dataset.tiling`     — COG reads behind a swappable ``CogReader`` (§stage 2)
- :mod:`dataset.screen`     — the exact-zero land / zero-fill pixel screen (ADR-0001 §3)
- :mod:`dataset.sampling`   — deterministic global COG sample (lat bands + geo-cells)
- :mod:`dataset.masks`      — resolve/read/cache the EDM/FLM/WBM quality masks
- :mod:`dataset.prevalence` — measure EDM/FLM/WBM prevalence on a global sample (#19)
- :mod:`dataset.calibration`— overview-vs-full-res metric ρ, Gate A3 (#20)
- :mod:`dataset.pilot`      — the 15k-window pilot scoring frame, Phase B1 (#24)
- :mod:`dataset.catalog`    — global scoring pass -> raw candidate catalog (#35)
- :mod:`dataset.gates`      — the locked quality gates as filter columns (#37)
- :mod:`dataset.frame`      — the stratification frame k-means fits + the scaler (#40)
- :mod:`dataset.centroids`  — the k=20 stratum centroids (#41)
- :mod:`dataset.select`     — allocate + FSCS + emit the selection list (Phase D, #44)
- :mod:`dataset.normalize`  — clip, land-only stats, degenerate drop, int16 (§4)
- :mod:`dataset.shards`     — sharded int16 memmap storage (§5b)
- :mod:`dataset.manifest`   — Parquet index + corpus_meta.json fingerprint (§5a)
- :mod:`dataset.concurrency`— the bounded, order-preserving parallel-read primitive (#38)
- :mod:`dataset.provenance` — git SHA + snapshot date stamped into every artifact
- :mod:`dataset.loader`     — Phase-1 torch read path (imports torch lazily)

Standing constraint across every stage: **elevation-only output, but architected
channel-ready** — adding slope/curvature later is a config flag, not a rewrite.
"""

__version__ = "0.1.0"
