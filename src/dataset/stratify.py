"""Stage 3 — terrain-ruggedness stratification.

Compute a cheap TRI (or elevation std) per candidate window and oversample
rugged terrain while keeping a representative flat tail. Strata thresholds and
oversampling ratios are open: resolved by "Decide the corpus-construction recipe".
"""
