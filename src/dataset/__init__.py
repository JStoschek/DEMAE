"""DEMAE Phase-0 data pipeline.

Turns the global Copernicus DEM GLO-30 collection into a terrain-diverse,
per-tile-normalized corpus of 224x224 elevation tiles for ViT-MAE pretraining.

The pipeline implements the locked Phase-0 design (ADRs 0001-0003). Stage map
(data flow, top to bottom):

    discovery  ->  tiling  ->  stratify  ->  normalize  ->  manifest/shards  ->  loader

- :mod:`dataset.discovery`  — list released COGs from the public bucket (§stage 1)
- :mod:`dataset.geometry`   — pure key/lat-band/geocell/window-grid math
- :mod:`dataset.tiling`     — COG reads behind a swappable ``CogReader`` (§stage 2)
- :mod:`dataset.stratify`   — ocean screen, TRI, quantile strata, quota fill (§3)
- :mod:`dataset.sampling`   — deterministic global COG sample (lat bands + geo-cells)
- :mod:`dataset.masks`      — resolve/read/cache the EDM/FLM/WBM quality masks
- :mod:`dataset.prevalence` — measure EDM/FLM/WBM prevalence on a global sample (#19)
- :mod:`dataset.calibration`— overview-vs-full-res metric ρ, Gate A3 (#20)
- :mod:`dataset.pilot`      — the 15k-window pilot scoring frame, Phase B1 (#24)
- :mod:`dataset.catalog`    — global scoring pass -> raw candidate catalog (#35)
- :mod:`dataset.gates`      — the locked quality gates as filter columns (#37)
- :mod:`dataset.normalize`  — clip, land-only stats, degenerate drop, int16 (§4)
- :mod:`dataset.shards`     — sharded int16 memmap storage (§5b)
- :mod:`dataset.manifest`   — Parquet index + corpus_meta.json fingerprint (§5a)
- :mod:`dataset.pipeline`   — end-to-end orchestrator
- :mod:`dataset.throughput` — smoke-sample measurement + cloud-burst decision
- :mod:`dataset.loader`     — Phase-1 torch read path (imports torch lazily)
- :mod:`dataset.synthetic`  — deterministic zero-network sample bucket
- :mod:`dataset.build`      — smoke-gated corpus build CLI (``python -m dataset.build``)

Standing constraint across every stage: **elevation-only output, but architected
channel-ready** — adding slope/curvature later is a config flag, not a rewrite.
"""

__version__ = "0.1.0"
