"""The synthetic sample bucket — the zero-network corpus a fresh reader builds."""

from __future__ import annotations

import numpy as np

from dataset import stratify
from dataset.geometry import geocell_id, lat_band
from dataset.synthetic import synthetic_reader


def test_reader_lists_multiple_distinct_cogs():
    reader = synthetic_reader()
    cogs = reader.list_cogs()
    assert len(cogs) >= 8
    # every stem parses and is unique
    assert len({c.stem for c in cogs}) == len(cogs)


def test_corpus_spreads_geocells_and_lat_bands():
    reader = synthetic_reader()
    cogs = reader.list_cogs()
    geocells = {geocell_id(c.origin_lon + 0.5, c.origin_lat + 0.5) for c in cogs}
    bands = {lat_band(c.origin_lat + 0.5) for c in cogs}
    # the sample must exercise the round-robin geo spread and >1 resolution band
    assert len(geocells) >= 6
    assert len(bands) >= 3


def test_terrain_has_ruggedness_variation_across_cogs():
    reader = synthetic_reader()
    cogs = reader.list_cogs()
    tris = []
    for c in cogs:
        ov = reader.read_overview(c, 8)
        tris.append(float(stratify.tri_riley(ov.array).mean()))
    # varied ruggedness is what makes the TRI strata non-degenerate
    assert max(tris) > 2 * (min(tris) + 1e-6)


def test_reader_is_deterministic():
    a = synthetic_reader(seed=7)
    b = synthetic_reader(seed=7)
    ca, cb = a.list_cogs()[0], b.list_cogs()[0]
    assert np.array_equal(
        a.read_overview(ca, 8).array, b.read_overview(cb, 8).array
    )


def test_some_cogs_carry_ocean():
    reader = synthetic_reader()
    saw_ocean_window = False
    for c in reader.list_cogs():
        ov = reader.read_overview(c, 8)
        if (ov.array == 0.0).mean() > 0.1:
            saw_ocean_window = True
    # coastlines/ocean must appear so the land mask + ocean drop are exercised
    assert saw_ocean_window
