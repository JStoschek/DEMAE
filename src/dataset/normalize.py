"""Stage 4 — per-tile normalization.

Per-tile standardization (subtract mean / divide std), storing mean+std for
inversion (see config.NORMALIZATION). Robust-clip percentile and near-constant
(std~=0) handling are open: resolved by "Decide the corpus-construction recipe".
"""
