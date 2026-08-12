"""The real-void family's shapes, and where their band comes from (#152).

Four things in this module are load-bearing, and each of them has a way of quietly coming
back wrong. They are the four this file exists to pin.

**The predicate.** Plan §3.3's *original* rule — "pixels with scene count 0 are true voids" —
is wrong for this product and produces an **empty mask**. There is no value 0 in SRTMGL1N's
class table; 0 means void in `SRTMIMGM`'s differently-encoded `.num`, a different product.
An empty mask is not an error anything downstream would notice: a family with no instances
prints as a family with no instances. So the rule is refused *here*, out loud, and a test
asserts the refusal rather than the corrected constants.

**The band's source.** Inferring the ruggedness band from the SRTM pixels *around* a void
measures the wrong terrain and biases systematically flat, because radar shadow occurs
exactly where the surroundings are tamer than the hole. The band comes from GLO-30 at the
void's own coordinates. That circularity was found and fixed in #126; the tests below are
what stop it being reintroduced, and they are behavioural — the same granule over two
different GLO-30 surfaces must give two different bands, and two different granules with the
same void shape must give the same one.

**The same-lattice premise.** The whole transplant is a crop-and-paste with no resampling,
which is true only where SRTM 1 arc-second and GLO-30 are the same lattice. GLO-30 widens
its longitude spacing poleward, so the premise fails silently at `|lat| >= 50`. It is
asserted, not discovered — and asserted *before* anything is read.

**The size sub-rows.** Real-small is the headline, real-medium+ is reported separately so it
stays size-comparable to synthetic medium, and 128-160 px is **an instance count with no
metric attached** — publishing a mean over eleven instances would invite a significance the
number cannot carry.

Nothing here touches the network: the granule reader and the GLO-30 reader are both injected,
following `tests/test_p127_fetch_srtm_voids.py`'s rule for the fetch path it tests without an
account. The one test that reads a real granule skips when the 24 are not on this box.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from benchmark import voids
from benchmark.config import Masks
from dataset import config as dataset_config
from dataset.geometry import pixel_spacing_m
from dataset.metrics import slope_mean

REPO = Path(__file__).resolve().parent.parent
GRANULES = REPO / "data" / "srtm_voids" / "num"

#: A whole cell, scaled down. The lattice relation is what the code checks — GLO-30 samples
#: one fewer post per degree than SRTM's overlapping 3601 — so a test cell can be small as
#: long as it keeps that relation and stays wider than the 224 px stratum window.
SRTM_TEST_PX = 501
GLO_TEST_PX = SRTM_TEST_PX - 1

#: An observed SRTM post: 201-224 is the swath count (§3.3).
OBSERVED = 210
#: A void class inside `num > 10 AND num < 201`, and clear of the two arguable values.
VOID = 50

#: Interior edges only, so the arithmetic below is readable: 0 deg is R1, ~3 deg is R2,
#: ~30 deg is R4. The real edges are quartiles of the eligible test pool (§5).
BANDS = voids.Bands(edges=(0.0, 1.0, 5.0, 20.0, 90.0), pool_n=13_538)


# ---------------------------------------------------------------------------------------
# Fixture rasters
# ---------------------------------------------------------------------------------------


def _num(*, blobs=(), size: int = SRTM_TEST_PX, background: int = OBSERVED) -> np.ndarray:
    """A `.num` class raster: observed everywhere, with ``blobs`` of a void class punched in.

    ``blobs`` are ``(row0, col0, height, width, value)``; the value defaults to :data:`VOID`.
    """
    num = np.full((size, size), background, dtype=np.uint8)
    for blob in blobs:
        r0, c0, h, w = blob[:4]
        value = blob[4] if len(blob) > 4 else VOID
        num[r0 : r0 + h, c0 : c0 + w] = value
    return num


def _square_of_diameter(d: float) -> int:
    """The side of a square whose equivalent diameter ``2*sqrt(area/pi)`` is ``d``."""
    return max(1, int(round(math.sqrt(math.pi * (d / 2.0) ** 2))))


def _surface(slope_deg: float, *, size: int = GLO_TEST_PX, base: float = 1000.0) -> np.ndarray:
    """A GLO-30 cell that is a plane of exactly ``slope_deg``, tilted along the row axis.

    ``base`` keeps every pixel off Copernicus's exact-``0.0`` ocean fill, which
    :func:`dataset.metrics.slope_mean` masks out before it differences.
    """
    dy = 111_320.0 / size
    rise = math.tan(math.radians(slope_deg)) * dy
    rows = np.arange(size, dtype=np.float32).reshape(-1, 1)
    return np.broadcast_to(base + rows * rise, (size, size)).astype(np.float32).copy()


def _readers(num: np.ndarray, surface: np.ndarray):
    """``(read_num, read_surface, calls)`` — injected readers that record what was asked for."""
    calls: list[tuple[str, str]] = []

    def read_num(cell: str) -> np.ndarray:
        calls.append(("num", cell))
        return num

    def read_surface(cell: str) -> np.ndarray:
        calls.append(("surface", cell))
        return surface

    return read_num, read_surface, calls


def _stack(diameters, *, col: int = 20, row: int = 20):
    """Square blobs of the given equivalent diameters, stacked down one column of the cell."""
    blobs = []
    for diameter in diameters:
        side = _square_of_diameter(diameter)
        blobs.append((row, col, side, side))
        row += side + 20
    return blobs


def _extract(num, surface, *, cell="N28E086", **kwargs):
    read_num, read_surface, calls = _readers(num, surface)
    catalog = voids.extract_cell(
        cell,
        bands=BANDS,
        read_num_fn=read_num,
        read_surface_fn=read_surface,
        **kwargs,
    )
    return catalog, calls


# ---------------------------------------------------------------------------------------
# The predicate, and the rule it replaced
# ---------------------------------------------------------------------------------------


def test_the_predicate_is_the_class_table_and_not_a_typed_range():
    """§3.3: observed is 201-224, water is 1/2/5, and 10 < num < 201 is fill from elsewhere."""
    predicate = voids.PLAN_PREDICATE

    assert predicate.selects(50) and predicate.selects(200) and predicate.selects(11)
    for observed in voids.OBSERVED_SCENE_COUNTS:
        assert not predicate.selects(observed)
    for water in voids.WATER_CLASSES:
        assert not predicate.selects(water)
    assert not predicate.selects(0), "0 is not in this product's class table at all"


def test_the_predicate_yields_a_non_empty_mask():
    num = _num(blobs=[(100, 100, 20, 30)])

    mask = voids.PLAN_PREDICATE.mask(num)

    assert mask.sum() == 20 * 30
    assert mask.dtype == bool


def test_the_zero_scene_count_rule_yields_an_empty_mask_and_is_refused():
    """The rule plan §3.3 originally carried. It selects nothing, and nothing is not shippable."""
    num = _num(blobs=[(100, 100, 20, 30)])

    assert (num == 0).sum() == 0, "the fixture has to be honest about the class table too"

    with pytest.raises(voids.EmptyVoidMask) as excinfo:
        voids.REFUTED_ZERO_SCENE_PREDICATE.mask(num)

    message = str(excinfo.value)
    assert "SRTMIMGM" in message, "say which product's `.num` the 0 rule belongs to"
    assert "num > 10" in message, "and which rule to use instead"


def test_the_strict_flag_drops_the_two_arguable_classes():
    """11 is NGA-interpolated SRTMv1 filigree, 25 is GDEM fill that was itself SRTM-derived."""
    num = _num(blobs=[(100, 100, 20, 30), (200, 200, 20, 30, 11), (300, 300, 20, 30, 25)])

    assert voids.PLAN_PREDICATE.mask(num).sum() == 3 * 20 * 30
    assert voids.STRICT_PREDICATE.mask(num).sum() == 20 * 30
    assert voids.STRICT_PREDICATE.strict and not voids.PLAN_PREDICATE.strict


def test_a_granule_with_no_voids_is_refused_rather_than_returned_empty():
    """The 24 granules were drawn for being void-rich. An empty one is a bad read, not a result."""
    with pytest.raises(voids.EmptyVoidMask) as excinfo:
        voids.PLAN_PREDICATE.mask(_num(), where="N28E086")

    assert "N28E086" in str(excinfo.value)


# ---------------------------------------------------------------------------------------
# The same-lattice premise — asserted, and asserted before anything is read
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("cell", ["N28E086", "S33W070", "N49E010", "S50W070"])
def test_the_draws_own_cells_are_on_the_shared_lattice(cell):
    """N49 and S50 both stop exactly at the band edge; their pixel centres stay inside it."""
    voids.assert_same_lattice(cell)


@pytest.mark.parametrize("cell", ["N50E086", "N60E010", "S51W070"])
def test_a_source_cell_at_fifty_degrees_is_rejected_naming_the_premise(cell):
    with pytest.raises(voids.OffLattice) as excinfo:
        voids.assert_same_lattice(cell)

    message = str(excinfo.value)
    assert "same lattice" in message
    assert "50" in message
    assert "3600" in message, "name the spacing that stops matching, not just the latitude"


def test_a_poleward_cell_is_refused_before_a_single_byte_is_read():
    """Ordering is the point: the premise fails silently, so it is checked ahead of the reads."""
    num, surface = _num(blobs=[(100, 100, 20, 30)]), _surface(3.0)
    read_num, read_surface, calls = _readers(num, surface)

    with pytest.raises(voids.OffLattice):
        voids.extract_cell(
            "N50E086", bands=BANDS, read_num_fn=read_num, read_surface_fn=read_surface
        )

    assert calls == [], "the lattice check has to come before the 13 MB read, not after"


def test_a_surface_that_is_not_the_granules_lattice_minus_one_is_refused():
    """SRTM's 3601 overlaps its neighbours by a post; GLO-30's 3600 does not. Off by one, always."""
    num, surface = _num(blobs=[(100, 100, 20, 30)]), _surface(3.0, size=400)

    with pytest.raises(voids.LatticeMismatch) as excinfo:
        _extract(num, surface)

    assert "400" in str(excinfo.value) and "501" in str(excinfo.value)


# ---------------------------------------------------------------------------------------
# Component labelling and the size sub-rows
# ---------------------------------------------------------------------------------------


def test_components_are_eight_connected():
    """P125 §10.2's extraction path. Two blobs touching at a corner are one void, not two."""
    num = _num(blobs=[(100, 100, 20, 20), (120, 120, 20, 20)])

    found = voids.components("N28E086", voids.PLAN_PREDICATE.mask(num))

    assert len(found) == 1
    assert found[0].area_px == 2 * 20 * 20
    assert found[0].eqdiam_px == pytest.approx(2.0 * math.sqrt(800 / math.pi))


def test_the_size_sub_rows_are_derived_from_the_mask_config_not_retyped():
    """§3.2's buckets and §3.3's sub-rows are the same boundaries; a second copy is a drift risk."""
    masks = Masks()
    rows = {row.name: row for row in voids.size_rows(masks)}

    assert (rows["real-small"].lo_px, rows["real-small"].hi_px) == masks.small_px
    assert (rows["real-medium+"].lo_px, rows["real-medium+"].hi_px) == masks.medium_px
    # 128-160, not 128-158: those bands describe sizes real voids *have*, and the 40% area cap
    # that lowered the synthetic large bucket is not a constraint on a measurement.
    assert rows["count-only-128-160"].hi_px == masks.real_band_top_px
    assert rows["count-only-128-160"].lo_px == masks.large_px[0]

    moved = {row.name: row for row in voids.size_rows(masks.replace(small_px=(16, 40)))}
    assert moved["real-small"].hi_px == 40
    assert moved["gap-40-64"].lo_px == 40, "the gap rows follow the buckets they sit between"


def test_the_sub_rows_do_not_tile_the_range_and_the_gaps_are_named():
    rows = voids.size_rows()

    assert [row.name for row in rows] == [
        "real-small",
        "gap-48-64",
        "real-medium+",
        "gap-96-128",
        "count-only-128-160",
    ]
    assert [row.reporting for row in rows] == [
        "metric",
        "excluded",
        "metric",
        "excluded",
        "count",
    ]


def test_the_128_160_row_carries_a_count_and_not_a_metric():
    """§3.3: an instance count with no metric attached. Eleven instances cannot carry a mean."""
    rows = {row.name: row for row in voids.size_rows()}

    assert rows["real-small"].carries_metric
    assert not rows["count-only-128-160"].carries_metric
    assert rows["count-only-128-160"].published, "counted is not the same as excluded"
    assert not rows["gap-48-64"].published


def test_components_are_bucketed_into_the_size_sub_rows():
    """One void per row, sized by equivalent diameter, including one in each published gap."""
    wanted = {
        "real-small": 30.0,
        "gap-48-64": 55.0,
        "real-medium+": 80.0,
        "gap-96-128": 110.0,
        "count-only-128-160": 140.0,
    }

    catalog, _ = _extract(_num(blobs=_stack(wanted.values())), _surface(30.0))

    assert catalog.instances_by_row() == {name: 1 for name in wanted}
    assert len(catalog.instances) == 3, "the two gap rows are counted, never published"
    assert catalog.scored_components == 5, "§3.3 counts the gaps among the scored components"


def test_a_component_above_the_top_is_excluded_and_counted():
    """The >160 px tail is real and abundant; the tile geometry excludes it, so say how much."""
    side = _square_of_diameter(200.0)
    catalog, _ = _extract(_num(blobs=[(20, 20, side, side), (400, 400, 20, 20)]), _surface(30.0))

    assert catalog.excluded_above_top == 1
    assert catalog.instances_by_row()["real-small"] == 1


def test_a_component_that_cannot_be_placed_on_a_tile_is_counted_as_attrition():
    """§3.2's two placement tests: inside a 208 px interior box, and at most 40% of the tile."""
    margin = Masks().edge_margin_px
    interior = dataset_config.TILE_PX - 2 * margin
    # A 4 px ribbon far longer than the interior box: its diameter sits in the scored window,
    # its bounding box cannot.
    catalog, _ = _extract(_num(blobs=[(100, 40, 4, interior + 40)]), _surface(30.0))

    assert catalog.instances == ()
    assert catalog.failed_placement == 1


# ---------------------------------------------------------------------------------------
# The band — from GLO-30 at the void's own coordinates, and from nowhere else
# ---------------------------------------------------------------------------------------


def test_the_band_comes_from_glo30_at_the_voids_own_coordinates():
    """One granule, two GLO-30 surfaces. If the band moves, it is reading the surface."""
    num = _num(blobs=[(240, 240, 20, 30)])

    flat, _ = _extract(num, _surface(0.0))
    steep, _ = _extract(num, _surface(30.0))

    assert flat.instances[0].band == 0 and BANDS.label(flat.instances[0].band) == "R1"
    assert steep.instances[0].band == 3 and BANDS.label(steep.instances[0].band) == "R4"
    assert flat.instances[0].slope_mean == pytest.approx(0.0, abs=1e-6)
    assert steep.instances[0].slope_mean == pytest.approx(30.0, abs=0.05)


def test_the_srtm_granule_contributes_the_shape_and_nothing_else():
    """#126's circularity, as a test: the surround is not what the band is read from.

    Two granules with the identical void and *different* surroundings — one all-observed, one
    laced with water and with a second, far steeper-looking class pattern. Nothing about the
    surround can reach the band, because nothing about the surround is passed to the code that
    assigns it: :func:`benchmark.voids.band_for` takes the GLO-30 surface and the component's
    own geometry, and has no parameter a `.num` raster could arrive through.
    """
    surface = _surface(30.0)
    plain = _num(blobs=[(240, 240, 20, 30)])
    laced = _num(blobs=[(240, 240, 20, 30)], background=voids.WATER_CLASSES[0])
    laced[0:100, 0:100] = OBSERVED
    laced[300:400, 300:400] = 224

    from_plain, _ = _extract(plain, surface)
    from_laced, _ = _extract(laced, surface)

    assert from_plain.instances[0].slope_mean == from_laced.instances[0].slope_mean
    assert from_plain.instances[0].band == from_laced.instances[0].band

    # And the same answer with no granule in the room at all.
    direct = voids.band_for(surface, from_plain.instances[0].component, BANDS)
    assert (direct.slope_mean, direct.band) == (
        from_plain.instances[0].slope_mean,
        from_plain.instances[0].band,
    )


def test_the_granule_is_read_once_and_the_surface_is_read_once():
    """A per-component COG read would be 24 cells x hundreds of voids, and the same answer."""
    blobs = [(100, 100, 20, 30), (300, 300, 20, 30), (400, 100, 20, 30)]

    catalog, calls = _extract(_num(blobs=blobs), _surface(30.0))

    assert len(catalog.instances) == 3
    assert calls == [("num", "N28E086"), ("surface", "N28E086")]


def test_a_void_near_the_cell_edge_has_its_window_clamped_and_says_so():
    """A centred window would run off the cell; the shift is recorded rather than hidden."""
    catalog, _ = _extract(_num(blobs=[(2, 2, 20, 30)]), _surface(30.0))

    instance = catalog.instances[0]
    assert instance.clamped is True
    assert (instance.window_row0, instance.window_col0) == (0, 0)


def test_an_instance_carries_the_coordinates_its_band_was_read_at():
    """C4 is answered on these rows, so a row has to say which ground it was scored against."""
    catalog, _ = _extract(_num(blobs=[(240, 240, 20, 30)]), _surface(30.0), cell="S33W070")

    instance = catalog.instances[0]
    assert -33.0 <= instance.lat <= -32.0
    assert -70.0 <= instance.lon <= -69.0
    assert instance.cell == "S33W070"


# ---------------------------------------------------------------------------------------
# The summary the histogram is published from
# ---------------------------------------------------------------------------------------


def test_the_summary_states_the_excluded_counts_beside_the_scored_ones():
    blobs = _stack([30.0, 30.0, 55.0, 80.0, 140.0])
    # The excluded tail, out of the stack's way: 200 px is a size real voids abundantly have
    # and this benchmark's 224 px tile cannot hold.
    big = _square_of_diameter(200.0)
    blobs.append((300, 300, big, big))

    catalog, _ = _extract(_num(blobs=blobs), _surface(30.0))

    summary = voids.summarise([catalog])

    assert summary["granules"] == 1
    assert summary["attrition"]["excluded_above_top"] == 1
    # §3.3's two counts, kept apart: the gap component is a scored component and not an
    # instance, exactly as the section's own 9,978-versus-8,824+438+11 arithmetic has it.
    assert summary["attrition"]["scored_components"] == 5
    assert summary["attrition"]["excluded_in_gaps"] == 1
    assert summary["attrition"]["instances"] == 4
    assert summary["rows"]["real-small"]["instances"] == 2
    assert summary["rows"]["gap-48-64"] == {"instances": 1, "reporting": "excluded"}
    assert summary["predicate"]["strict"] is False


def test_the_summarys_count_only_row_is_a_bare_count():
    """No mean, no band breakdown — the two shapes that would give eleven instances authority."""
    side = _square_of_diameter(140.0)
    catalog, _ = _extract(_num(blobs=[(20, 20, side, side)]), _surface(30.0))

    row = voids.summarise([catalog])["rows"]["count-only-128-160"]

    assert row == {"instances": 1, "reporting": "count"}


def test_the_summary_breaks_the_metric_rows_out_by_band():
    """§5's stratification, kept in the artifact even though p135 measured it 99.94% R4."""
    catalog, _ = _extract(_num(blobs=[(240, 240, 20, 30)]), _surface(30.0))

    row = voids.summarise([catalog])["rows"]["real-small"]

    assert row["bands"] == [0, 0, 0, 1]
    assert row["reporting"] == "metric"


# ---------------------------------------------------------------------------------------
# The GLO-30 reader's lattice gate
# ---------------------------------------------------------------------------------------


class _FakeCog:
    """A COG's geometry without a COG. What is under test is the gate, not the read."""

    def __init__(self, *, width=voids.GLO_PX, height=voids.GLO_PX, origin=None, step=None):
        step = 1.0 / voids.GLO_PX if step is None else step
        origin = (86 - 0.5 / voids.GLO_PX, 29 + 0.5 / voids.GLO_PX) if origin is None else origin
        self.width, self.height = width, height
        self.transform = type(
            "_Affine", (), {"a": step, "e": -step, "c": origin[0], "f": origin[1]}
        )()
        self.array = np.zeros((2, 2), dtype=np.float32)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, _band):
        return self.array


def test_the_glo30_reader_accepts_a_cell_on_whole_arc_second_centres():
    cog = _FakeCog()

    assert voids.read_surface("N28E086", open_fn=lambda _url: cog) is cog.array


def test_the_glo30_reader_refuses_a_cell_that_is_not_one_post_per_arc_second():
    """Poleward COGs are narrower. Reaching one here means the lattice assertion was skipped."""
    with pytest.raises(voids.LatticeMismatch) as excinfo:
        voids.read_surface("N28E086", open_fn=lambda _url: _FakeCog(width=2400))

    assert "2400" in str(excinfo.value)


def test_the_glo30_reader_refuses_a_half_pixel_shift():
    """Half a post off is a mask cut from ground it never covered, and every array still slices."""
    shifted = _FakeCog(origin=(86.0, 29.0))

    with pytest.raises(voids.LatticeMismatch) as excinfo:
        voids.read_surface("N28E086", open_fn=lambda _url: shifted)

    assert "pixel centres" in str(excinfo.value)


def test_the_glo30_reader_refuses_a_wrong_pixel_size():
    with pytest.raises(voids.LatticeMismatch) as excinfo:
        voids.read_surface("N28E086", open_fn=lambda _url: _FakeCog(step=2.0 / voids.GLO_PX))

    assert "arc-second" in str(excinfo.value)


# ---------------------------------------------------------------------------------------
# The bands, cut on the pool a void is transplanted onto
# ---------------------------------------------------------------------------------------


def test_the_bands_are_quartiles_of_the_eligible_test_pool(tmp_path, corpus_factory):
    """§5 cuts the bands on the *pool*, so a band index means what it means for every tile."""
    corpus_factory(
        tmp_path,
        [
            ("Copernicus_DSM_COG_10_N45_00_E006_00_DEM", "+000_+040", "test"),
            ("Copernicus_DSM_COG_10_N45_00_E007_00_DEM", "+000_+040", "train"),
        ],
        tiles_per_cog=8,
        metrics_fn=lambda cog, tile: {"slope_mean": float(tile) if cog == 0 else 100.0},
    )

    bands = voids.Bands.from_manifest(tmp_path)

    assert bands.pool_n == 8, "the train COG's tiles are not ground a test void lands on"
    assert bands.count == voids.N_BANDS
    assert bands.edges[0] == 0.0 and bands.edges[-1] == 7.0
    assert bands.of(0.0) == 0 and bands.label(0) == "R1"
    assert bands.of(7.0) == voids.N_BANDS - 1 and bands.label(3) == "R4"


def test_an_empty_eligible_pool_is_refused_rather_than_cut(tmp_path, corpus_factory):
    corpus_factory(
        tmp_path,
        [("Copernicus_DSM_COG_10_N45_00_E006_00_DEM", "+000_+040", "train")],
        tiles_per_cog=2,
    )

    with pytest.raises(ValueError) as excinfo:
        voids.Bands.from_manifest(tmp_path)

    assert "no eligible test tiles" in str(excinfo.value)


# ---------------------------------------------------------------------------------------
# The band is read through the corpus's own code path, against committed catalog rows
# ---------------------------------------------------------------------------------------

CORPUS = REPO / "data" / "corpus_500k"
SOURCE_COG = "Copernicus_DSM_COG_10_N28_00_E086_00_DEM"

# The manifest is committed and the 47 GB of shards are not, so the pixels are the guard, not
# the catalog: a checkout with the manifest alone would fail this test rather than skip it.
requires_corpus = pytest.mark.skipif(
    not (CORPUS / "shards").is_dir(), reason="data/corpus_500k's shards are not on this box"
)


def _committed_rows(limit: int) -> list[dict]:
    """The first ``limit`` committed catalog rows of `N28E086`, with their tile pixels."""
    import pyarrow.parquet as pq

    from dataset.shards import shard_path

    table = pq.read_table(
        CORPUS / "manifest.parquet",
        columns=[
            "source_cog", "win_row0", "win_col0", "lon", "lat", "slope_mean",
            "shard_id", "row_index",
        ],
    )
    column = {n: table.column(n).to_numpy(zero_copy_only=False) for n in table.column_names}
    wanted = np.nonzero(column["source_cog"].astype(str) == SOURCE_COG)[0][:limit]
    out = []
    for i in wanted:
        row = {n: column[n][i] for n in column}
        path = shard_path(CORPUS / "shards", int(row["shard_id"]))
        if not path.is_file():
            pytest.skip(f"{path.name} is not on this box; the shards are gitignored")
        shard = np.load(path, mmap_mode="r")
        row["tile"] = np.asarray(shard[int(row["row_index"])], dtype=np.float32)
        out.append(row)
    return out


@requires_corpus
def test_the_band_is_read_through_the_corpus_own_scoring_call():
    """§3.3: the stratum comes from GLO-30 "through the same code path that assigns strata to
    every other tile in the corpus", and that path is *checked rather than asserted*.

    Five committed catalog rows of `N28E086`, each one's materialized tile dropped into a cell
    at the grid window the manifest records it at. What is asserted to **0.0** is that
    :func:`benchmark.voids.band_for` reproduces the window's own centre coordinates and the
    catalog's own `slope_mean` call on the identical pixels — i.e. that this module reaches the
    corpus's number rather than a look-alike of it. The residual against the manifest's *stored*
    value is the shard's int16 quantization and nothing else; it measures 0.010-0.044 deg over
    these five rows, and it is reported rather than asserted, because a bound invented here
    would be a number the plan never registered.
    """
    bands = voids.Bands(edges=(0.0, 5.0, 10.0, 20.0, 90.0))
    tile_px = dataset_config.TILE_PX

    for row in _committed_rows(5):
        surface = np.zeros((voids.GLO_PX, voids.GLO_PX), dtype=np.float32)
        r0, c0 = int(row["win_row0"]), int(row["win_col0"])
        surface[r0 : r0 + tile_px, c0 : c0 + tile_px] = row["tile"]
        # A void whose centred window is exactly this catalog row's grid window.
        component = voids.Component(
            cell="N28E086",
            label=1,
            area_px=400,
            eqdiam_px=22.6,
            row=r0 + tile_px / 2,
            col=c0 + tile_px / 2,
            bbox_h=20,
            bbox_w=20,
            fits_interior=True,
            fits_area=True,
        )

        reading = voids.band_for(surface, component, bands)

        assert (reading.window_row0, reading.window_col0) == (r0, c0)
        assert reading.lon == pytest.approx(float(row["lon"]), abs=0.0)
        assert reading.lat == pytest.approx(float(row["lat"]), abs=0.0)
        expected = slope_mean(
            row["tile"],
            pixel_spacing_m(
                float(row["lat"]),
                rows_per_degree=voids.GLO_PX,
                cols_per_degree=voids.GLO_PX,
            ),
        )
        assert reading.slope_mean == pytest.approx(expected, abs=0.0)
        assert abs(reading.slope_mean - float(row["slope_mean"])) < 0.1


# ---------------------------------------------------------------------------------------
# The real granules, when this box has them
# ---------------------------------------------------------------------------------------

requires_granules = pytest.mark.skipif(
    not (GRANULES / "N28E086.SRTMGL1N.num.zip").is_file(),
    reason="the 24 fetched SRTMGL1N granules are not on this box (#127)",
)


@requires_granules
def test_the_predicate_yields_a_non_empty_mask_over_a_fetched_granule():
    """The acceptance criterion, against the real data rather than a fixture of it."""
    num = voids.read_num("N28E086", granule_dir=GRANULES)

    assert num.shape == (voids.SRTM_PX, voids.SRTM_PX)
    mask = voids.PLAN_PREDICATE.mask(num, where="N28E086")
    # P125 measured 3.7-22.5% over the Himalaya cells; the assertion is only that it is a
    # real, non-degenerate fraction, because the magnitude is #135's number, not this test's.
    assert 0.001 < mask.mean() < 0.5


@requires_granules
def test_the_zero_rule_finds_nothing_in_a_real_granule_and_is_refused():
    """The class table, checked against the product rather than against the User Guide's PDF."""
    num = voids.read_num("N28E086", granule_dir=GRANULES)

    assert (num == 0).sum() == 0

    with pytest.raises(voids.EmptyVoidMask):
        voids.REFUTED_ZERO_SCENE_PREDICATE.mask(num, where="N28E086")


@requires_granules
def test_every_fetched_cell_is_on_the_shared_lattice():
    """24 cells, all equatorward of 50 deg. The premise holds for the draw that exists."""
    cells = voids.cells(GRANULES)

    assert len(cells) == 24
    for cell in cells:
        voids.assert_same_lattice(cell)


@requires_granules
def test_the_promotion_reproduces_p135s_measured_count_for_a_cell():
    """The prototype this module was promoted from measured `N28E086` at 221 components.

    A promotion that changes the measurement silently is the failure worth catching, and the
    committed p135 artifact is the only thing that can catch it. Bands need a COG read and this
    test has no network, so it stops at the seam the count is a property of: the predicate, the
    labelling, §3.2's placement tests and §3.3's size rows. One cell, because 24 is 20 seconds —
    run over all of them the same code reproduces p135's whole attrition table (10,291 at or
    above 16 px, 182 above the top, 131 failing placement, 9,978 in a published row).
    """
    p135 = json.loads((REPO / "docs" / "research" / "p135-void-ruggedness.json").read_text())
    expected = next(row["scored"] for row in p135["per_cell"] if row["cell"] == "N28E086")

    num = voids.read_num("N28E086", granule_dir=GRANULES)
    found = voids.components("N28E086", voids.PLAN_PREDICATE.mask(num, where="N28E086"))

    rows = voids.size_rows()
    scored_components = [
        component
        for component in found
        if component.eqdiam_px <= rows[-1].hi_px and component.placeable
    ]
    assert len(scored_components) == expected == 221


@requires_granules
def test_a_truncated_granule_is_refused_rather_than_reshaped(tmp_path):
    """P125 §13.2's failure mode is a login page named `.zip`; a short member is its cousin."""
    path = tmp_path / "N28E086.SRTMGL1N.num.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("N28E086.num", b"\x00" * 1024)

    with pytest.raises(ValueError) as excinfo:
        voids.read_num("N28E086", granule_dir=tmp_path)

    assert "12967201" in str(excinfo.value).replace(",", "")
