"""DEMAE Phase-0 data pipeline.

Turns the global Copernicus DEM GLO-30 collection into a terrain-diverse,
per-tile-normalized corpus of 224x224 elevation tiles for ViT-MAE pretraining.

The pipeline is deliberately a *skeleton* at this stage: the modules below name
the stages, but their design decisions are still open wayfinder tickets on the
Phase-0 map (see CONTEXT.md and the repo issues). Each stub module documents the
ticket that will fill it in. Nothing here pre-commits an undecided design.

Stage map (data flow, top to bottom):

    discovery  ->  tiling  ->  stratify  ->  normalize  ->  manifest/shards  ->  loader

Standing constraint across every stage: **elevation-only output, but architected
channel-ready** — adding slope/curvature later is a config flag, not a rewrite.
"""

__version__ = "0.1.0"
