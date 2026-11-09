"""Geomorphon labels behind one WBT call (#201, ADR-0014) — tested without the WBT binary.

Three seams, the spec's (#197 "Testing Decisions"):

* ``benchmark.launch.main(["freeze", "--task", "landform", ...])`` with a **fake labeller**
  (substitution point A) — a deterministic function of the window and the four parameters — so
  labels, fingerprints, the ignore rule, the crop, resume and the fixed sample are checked end to
  end through the one command.
* :func:`benchmark.geomorphon.call_wbt` against a **fake binary**: a script that records the
  command line it was given and the header of the raster it was handed, so "all four parameters
  reach the call", "an unpinned build is refused" and "an EPSG:4326 window never reaches it" are
  checked on the real subprocess path.
* :func:`benchmark.geomorphon.read_buffered` against **fixture COGs** (substitution point B):
  small GeoTIFFs on the true GLO-30 geographic grid, so the mosaic, the anisotropic column
  buffer, a column-width band crossing, the antimeridian wrap and absent data are checked on the
  same ``rasterio.merge`` path the live read takes.

The fixture world is one elevation function of (lat, lon) sampled at each COG's pixel centres;
the corpus's shard tiles are the same function at the tile's pixels, so the label pass's centre
check holds exactly when the mosaic is on the shard grid.
"""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmark import geomorphon as gm
from benchmark import geomorphon_pass, landform, launch
from benchmark.landform_config import Geomorphon
from dataset.geometry import COLS_PER_DEGREE, format_cog_stem, lat_band
from pretrain.artifact import write_tile_strata

# --- The fixture world ---------------------------------------------------------------------------


def elevation(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Integer-valued metres, never 0 — except a 5-pixel coastal strip just east of lon 40°.

    It varies along both axes at the pixel scale, so a read one row or column off comes back
    different everywhere.
    """
    lat, lon = np.broadcast_arrays(lat, lon)
    lon = np.where(lon >= 180.0, lon - 360.0, lon)
    z = 1000 + 200 * np.sin(lat * 3600 / 37.0) + 150 * np.cos(lon * 3600 / 23.0)
    coast = (lon >= 40.0) & (lon < 40.0 + 4.5 / 3600) & (lat < 12)
    return np.where(coast, 0.0, np.round(z)).astype(np.float32)


def stem(lat: int, lon: int) -> str:
    return format_cog_stem(lon, lat)


def width(lat: int) -> int:
    return COLS_PER_DEGREE[lat_band(lat + 0.5)]


def write_cog(root: Path, lat: int, lon: int, rows: tuple[int, int], cols: tuple[int, int]):
    """A fixture COG: the block ``rows`` x ``cols`` of the real one, at its true geo-transform."""
    import rasterio
    from rasterio.transform import Affine

    w = width(lat)
    r = np.arange(*rows)
    c = np.arange(*cols)
    z = elevation((lat + 1 - r / 3600)[:, None], (lon + c / w)[None, :])
    path = root / stem(lat, lon) / f"{stem(lat, lon)}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = Affine(1 / w, 0, lon - 0.5 / w + cols[0] / w, 0, -1 / 3600,
                       lat + 1 + 0.5 / 3600 - rows[0] / 3600)
    with rasterio.open(path, "w", driver="GTiff", width=z.shape[1], height=z.shape[0], count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform,
                       compress="deflate") as d:
        d.write(z, 1)


def write_world(root: Path, tile_cogs, *, absent=(), extra=()) -> Path:
    """Fixture COGs for tiles at the west edge (col 0) of ``tile_cogs``, rows 0 and 224.

    Each tile COG gets its own block and its W, N and NW neighbours theirs — every pixel an L =
    100 window can reach — unless the neighbour is in ``absent``. Returns the frame file.
    """
    present: set[str] = set()
    for lat, lon in tile_cogs:
        blocks = {
            (lat, lon): ((0, 700), (0, 500)),
            (lat, lon - 1): ((0, 700), (width(lat) - 250, width(lat))),
            (lat + 1, lon): ((3400, 3600), (0, 500)),
            (lat + 1, lon - 1): ((3400, 3600), (width(lat + 1) - 250, width(lat + 1))),
        }
        for (la, lo), (rows, cols) in blocks.items():
            if (la, lo) not in absent:
                write_cog(root, la, lo, rows, cols)
                present.add(stem(la, lo))
    for la, lo, rows, cols in extra:
        write_cog(root, la, lo, rows, cols)
        present.add(stem(la, lo))
    frame = root / "cog_frame.json"
    frame.write_text(json.dumps(sorted(present)))
    return frame


def tile_pixels(tile_id: str) -> np.ndarray:
    s, r0, c0 = gm.parse_tile_id(tile_id)
    cog = gm.parse_cog_key(s)
    w = gm.width_of(cog.origin_lat)
    r = r0 + np.arange(224)
    c = c0 + np.arange(224)
    return elevation((cog.origin_lat + 1 - r / 3600)[:, None], (cog.origin_lon + c / w)[None, :])


# A train COG with every neighbour present; a val COG whose west neighbour is land absent from
# the release; a test COG on a coast whose west neighbours are verified open ocean.
TRAIN, VAL, TEST = (10, 20), (10, 30), (10, 40)
ABSENT_LAND = (10, 29)
OCEAN = ((10, 39), (11, 39))


def tile_id(cog, j: int) -> str:
    return f"{stem(*cog)}_{j * 224}_0"


def fake_labeller(window: gm.Window, params: gm.WbtParams) -> np.ndarray:
    """A deterministic stand-in for WBT: a function of each pixel's elevation and the four
    parameters, WBT's nodata (0) where the window is absent."""
    e = np.where(window.absent, 0, np.rint(window.elevation)).astype(np.int64)
    shift = params.search + round(params.threshold * 100) + 3 * params.skip + 7 * params.fdist
    forms = (e + shift) % 10 + 1
    forms[window.absent] = 0
    return forms.astype(np.uint8)


class Recorder:
    """The fake labeller, counting which tiles and window shapes reach it, optionally failing."""

    def __init__(self, fn=fake_labeller, fail_after: int | None = None):
        self.fn = fn
        self.calls: list[tuple[str, gm.WbtParams, tuple[int, int]]] = []
        self.fail_after = fail_after
        self.build = "fake-wbt"

    def __call__(self, window, params):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise ConnectionError("the network dropped")
        self.calls.append((window.tile_id, params, window.elevation.shape))
        return self.fn(window, params)

    @property
    def tiles(self) -> set[str]:
        return {t for t, _, _ in self.calls}


@pytest.fixture
def world(tmp_path, corpus_factory):
    """The fixture COGs, a corpus whose shards are the same terrain, and Task 1's lists."""
    cogs_dir = tmp_path / "cogs"
    frame = write_world(cogs_dir, [TRAIN, VAL, TEST], absent=(ABSENT_LAND, *OCEAN))
    sequence = iter(tile_pixels(tile_id(cog, j)) for cog in (TRAIN, VAL, TEST) for j in (0, 1))
    corpus_factory(
        tmp_path / "corpus",
        [(stem(*TRAIN), "+000_+000", "train"), (stem(*VAL), "+000_+010", "val"),
         (stem(*TEST), "+000_+020", "test")],
        tile_fn=lambda rng, base: next(sequence),
    )
    task1 = tmp_path / "task1"
    for name, cog in (("head_train_tiles.csv", TRAIN), ("val_tiles.csv", VAL),
                      ("test_tiles.csv", TEST)):
        write_tile_strata([(tile_id(cog, j), j) for j in (0, 1)], task1 / name)
    (tmp_path / "task2").mkdir()
    return tmp_path, frame


@pytest.fixture
def flags(world):
    box, frame = world
    return [
        "--task", "landform",
        "--set", f'paths.artifacts_dir="{box / "task2"}"',
        "--set", f'paths.task1_artifacts_dir="{box / "task1"}"',
        "--set", f'paths.corpus_dir="{box / "corpus"}"',
        "--set", f'paths.cog_frame="{frame}"',
        "--set", f'paths.cog_root="{box / "cogs"}"',
        "--set", f'paths.labels_dir="{box / "labels"}"',
        "--set", f"geomorphon.verified_ocean={json.dumps(sorted(stem(*c) for c in OCEAN))}",
        "--set", f"geomorphon.absent_land={json.dumps([stem(*ABSENT_LAND)])}",
        "--set", "geomorphon.fixed_sample=2",
        "--set", "labelling.threads=1",
    ]


@pytest.fixture
def only_labels(monkeypatch):
    """``freeze`` with only this ticket's stage, so its siblings' stages neither run nor refuse."""
    stage = next(s for s in landform.FREEZE_STAGES if s.label == "geomorphon labels")
    monkeypatch.setattr(landform, "FREEZE_STAGES", (stage,))


@pytest.fixture
def labeller(monkeypatch):
    """Substitution point A: the fake in place of the pinned binary."""
    rec = Recorder()
    monkeypatch.setattr(geomorphon_pass, "make_labeller", lambda cfg: rec)
    return rec


def freeze(flags) -> int:
    return launch.main(["freeze", *flags])


def committed(box: Path) -> dict:
    return json.loads((box / "task2" / geomorphon_pass.CONFIG_NAME).read_text())


def raster(box: Path, variant: str, tid: str) -> np.ndarray:
    (hashdir,) = (box / "labels").iterdir()
    return np.load(hashdir / variant / f"{tid}.npy")


VARIANTS = ("primary", "L25", "L100", "flat0.75", "flat1.25")


# --- Seam 1: freeze --task landform, through the fake labeller ------------------------------------


def test_a_dry_run_says_the_geomorphon_stage_is_built(flags, capsys):
    assert launch.main(["freeze", "--dry-run", *flags]) == 0
    out = capsys.readouterr().out
    assert "geomorphon labels — built (#201)" in out
    assert "not frozen yet (`freeze --task landform`, #201/#202)" in out


def test_freeze_labels_every_listed_tile_and_commits_fingerprints_per_variant(
    world, flags, only_labels, labeller, capsys
):
    box, _ = world
    assert freeze(flags) == 0
    out = capsys.readouterr().out
    fp = box / "task2" / geomorphon_pass.FINGERPRINT_DIR
    assert sorted(p.stem for p in fp.iterdir()) == sorted(VARIANTS)
    primary = geomorphon_pass.read_fingerprints(fp / "primary.csv")
    assert sorted(primary) == sorted(tile_id(c, j) for c in (TRAIN, VAL, TEST) for j in (0, 1))
    for variant in VARIANTS[1:]:  # D7: the sensitivity variants label the test tiles only
        assert sorted(geomorphon_pass.read_fingerprints(fp / f"{variant}.csv")) == [
            tile_id(TEST, 0), tile_id(TEST, 1)
        ]
    body = committed(box)
    assert set(body["variants"]) == set(VARIANTS)
    assert body["variants"]["primary"]["tiles"] == 6
    assert len(body["label_config_hash"]) == 16
    assert body["label_config"]["params"] == {"search": 50, "threshold": 1.0, "skip": 0,
                                              "fdist": 0}
    # D6: weights over the train list's non-ignore pixels, mean 1, rarer classes weigh more.
    weights, counts = body["class_weights"], body["class_pixel_counts_train"]
    assert sum(counts.values()) == 2 * 224 * 224  # the train COG has no ocean and no absence
    assert math.isclose(sum(weights.values()) / 10, 1.0)
    rare, common = min(counts, key=counts.get), max(counts, key=counts.get)
    assert weights[rare] > weights[common]
    # The census names each absence it found and how D5 treats it.
    absent = body["census"]["absent"]
    assert absent[stem(*ABSENT_LAND)]["treated_as"].startswith("land")
    assert all(absent[stem(*c)]["treated_as"].startswith("verified ocean") for c in OCEAN)
    assert "byte-identical" in out


def test_a_label_is_the_labellers_output_on_the_tiles_own_pixels_cropped_after_the_call(
    world, flags, only_labels, labeller
):
    box, _ = world
    freeze(flags)
    # The train COG has every neighbour, so no pixel is ignored: every label is the fake's
    # value at that pixel's own shard elevation, i.e. the crop landed on the tile (D4).
    for j in (0, 1):
        label = raster(box, "primary", tile_id(TRAIN, j))
        shard = tile_pixels(tile_id(TRAIN, j)).astype(np.int64)
        assert label.dtype == np.uint8 and label.shape == (224, 224)
        np.testing.assert_array_equal(label, (shard + 50 + 100) % 10)
        np.testing.assert_array_equal(
            raster(box, "flat1.25", tile_id(TEST, j))[:, 5:],
            (tile_pixels(tile_id(TEST, j))[:, 5:].astype(np.int64) + 50 + 125) % 10,
        )
    # Each call saw the buffered window, never a bare tile: L rows and ceil(L*dy/dx) columns
    # per side, the columns wider than the rows at 10.9°N.
    for tid, params, shape in labeller.calls:
        rows, cols = (shape[0] - 224) // 2, (shape[1] - 224) // 2
        assert rows == params.search
        assert cols == math.ceil(params.search / math.cos(math.radians(10.95)))
        assert cols > rows


def test_ocean_and_the_reach_of_absent_land_are_255_and_verified_ocean_reads_as_zero(
    world, flags, only_labels, labeller
):
    box, _ = world
    freeze(flags)
    # The val COG's west neighbour is absent land: every pixel whose L-row x ceil(L*dy/dx)-
    # column window reaches it is ignored, and no other.
    reach = math.ceil(50 / math.cos(math.radians(10.95)))  # 51 at 10.9°N
    for j in (0, 1):
        label = raster(box, "primary", tile_id(VAL, j))
        assert (label[:, :reach] == 255).all()
        assert (label[:, reach:] != 255).all()
    # The test COG's west neighbours are verified ocean: they read as 0 m, so only the 5-pixel
    # strip of ocean inside the tile is ignored — at every variant, whatever its L.
    for variant in VARIANTS:
        for j in (0, 1):
            label = raster(box, variant, tile_id(TEST, j))
            assert (label[:, :5] == 255).all() and (label[:, 5:] != 255).all(), variant
    rows = geomorphon_pass.read_fingerprints(
        box / "task2" / geomorphon_pass.FINGERPRINT_DIR / "primary.csv"
    )
    assert rows[tile_id(TEST, 0)]["counts"][-1] == 5 * 224
    assert rows[tile_id(VAL, 1)]["counts"][-1] == reach * 224


def test_a_second_freeze_is_idempotent_and_regenerates_the_fixed_sample(
    world, flags, only_labels, labeller, capsys
):
    box, _ = world
    freeze(flags)
    before = {p: p.read_bytes() for p in (box / "task2").rglob("*") if p.is_file()}
    first_calls = len(labeller.calls)
    capsys.readouterr()
    assert freeze(flags) == 0
    out = capsys.readouterr().out
    assert {p: p.read_bytes() for p in (box / "task2").rglob("*") if p.is_file()} == before
    # Nothing relabelled but the fixed sample: two test tiles, every variant, from the COGs.
    again = labeller.calls[first_calls:]
    assert len(again) == 2 * len(VARIANTS)
    assert {t for t, _, _ in again} == {tile_id(TEST, 0), tile_id(TEST, 1)}
    assert "2 test tiles x 5 variants regenerated from the COGs, byte-identical" in out
    assert "6 already fingerprinted, 0 to read" in out


def test_a_moved_pipeline_is_caught_by_the_fixed_sample_before_it_labels_a_tile(
    world, flags, only_labels, labeller, monkeypatch
):
    box, _ = world
    freeze(flags)
    shutil_rmtree(box / "labels")
    moved = Recorder(fn=lambda w, p: np.where(w.absent, 0, fake_labeller(w, p) % 10 + 1)
                     .astype(np.uint8))
    monkeypatch.setattr(geomorphon_pass, "make_labeller", lambda cfg: moved)
    with pytest.raises(ValueError, match="labels the same window differently"):
        freeze(flags)
    assert moved.tiles <= {tile_id(TEST, 0), tile_id(TEST, 1)}
    assert not (box / "labels").exists() or not any((box / "labels").rglob("*.npy"))


def test_an_interrupted_pass_resumes_and_skips_tiles_already_fingerprinted(
    world, flags, only_labels, monkeypatch, capsys
):
    box, _ = world
    dropped = Recorder(fail_after=3)  # the 4th WBT call loses the network
    monkeypatch.setattr(geomorphon_pass, "make_labeller", lambda cfg: dropped)
    with pytest.raises(ConnectionError):
        freeze(flags)
    assert not (box / "task2" / geomorphon_pass.CONFIG_NAME).exists()
    written = dropped.tiles
    assert len(written) == 3  # train and val tiles take one call each
    capsys.readouterr()

    resumed = Recorder()
    monkeypatch.setattr(geomorphon_pass, "make_labeller", lambda cfg: resumed)
    assert freeze(flags) == 0
    assert "3 already fingerprinted, 3 to read" in capsys.readouterr().out
    pass_calls = resumed.calls[: len(resumed.calls) - 2 * len(VARIANTS)]  # minus the sample
    assert not {t for t, _, _ in pass_calls} & written


def test_labels_from_a_different_build_are_never_resumed_into(
    world, flags, only_labels, monkeypatch
):
    first = Recorder(fail_after=2)
    monkeypatch.setattr(geomorphon_pass, "make_labeller", lambda cfg: first)
    with pytest.raises(ConnectionError):
        freeze(flags)
    other = Recorder()
    other.build = "another-wbt-sha256"
    monkeypatch.setattr(geomorphon_pass, "make_labeller", lambda cfg: other)
    with pytest.raises(ValueError, match="never mixed unchecked"):
        freeze(flags)


def test_a_failed_buffered_read_stops_the_pass_with_nothing_committed(
    world, flags, only_labels, labeller
):
    box, _ = world
    # The frame says the val COG's north neighbour is in the release; its file is gone.
    (box / "cogs" / stem(11, 30) / f"{stem(11, 30)}.tif").unlink()
    with pytest.raises(Exception, match=stem(11, 30)):
        freeze(flags)
    assert not (box / "task2" / geomorphon_pass.CONFIG_NAME).exists()
    assert not (box / "task2" / geomorphon_pass.FINGERPRINT_DIR).exists()
    assert tile_id(VAL, 0) not in labeller.tiles


def test_a_window_off_the_shard_grid_stops_the_pass(world, flags, only_labels, labeller):
    box, _ = world
    # Shift the train COG's fixture one column west: the mosaic still succeeds and the labels
    # would look normal, but the centre is no longer the shard tile.
    import rasterio

    path = box / "cogs" / stem(*TRAIN) / f"{stem(*TRAIN)}.tif"
    with rasterio.open(path, "r+") as d:
        t = d.transform
        d.transform = t * t.translation(-1, 0)
    with pytest.raises(ValueError, match="differs from the shard tile"):
        freeze(flags)


def test_the_census_refuses_an_absence_no_list_classifies(world, flags, only_labels, labeller):
    unlisted = [*flags, "--set", "geomorphon.absent_land=[]"]
    with pytest.raises(ValueError, match=stem(*ABSENT_LAND)) as exc:
        freeze(unlisted)
    assert "No absence is assumed" in str(exc.value)
    assert labeller.calls == []


def test_without_a_pinned_binary_freeze_refuses_before_reading_a_tile(world, flags, only_labels):
    with pytest.raises(FileNotFoundError, match="paths.wbt_binary is not set"):
        freeze(flags)


# --- Substitution point A: the one narrow WBT call ------------------------------------------------

FAKE_WBT = """#!{python}
import json, sys
import numpy as np
import rasterio
args = sys.argv[1:]
opt = dict(a.split("=", 1) for a in args if "=" in a)
with rasterio.open(opt["--dem"]) as d:
    dem = d.read(1)
    profile = d.profile
    seen = {{"args": args, "crs": d.crs.to_string(), "transform": list(d.transform)[:6],
            "nodata": d.nodata, "absent": int((dem == d.nodata).sum())}}
open({record!r}, "w").write(json.dumps(seen))
forms = (np.arange(dem.size).reshape(dem.shape) % 10 + 1).astype("float32")
forms[dem == profile["nodata"]] = -32768
profile.update(dtype="float32", nodata=-32768)
with rasterio.open(opt["--output"], "w", **profile) as d:
    d.write(forms, 1)
"""


@pytest.fixture
def fake_wbt(tmp_path):
    """An executable that answers like ``whitebox_tools -r=Geomorphons`` and records its call."""
    record = tmp_path / "wbt_call.json"
    binary = tmp_path / "whitebox_tools"
    binary.write_text(FAKE_WBT.format(python=sys.executable, record=str(record)))
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return binary, record, gm.binary_sha256(binary)


def a_window(header: gm.SpacingHeader | None = None, rows: int = 4, cols: int = 6) -> gm.Window:
    shape = (224 + 2 * rows, 224 + 2 * cols)
    absent = np.zeros(shape, dtype=bool)
    absent[:, :2] = True
    return gm.Window(
        tile_id="fixture",
        elevation=np.full(shape, 100.0, dtype=np.float32),
        absent=absent,
        header=header or gm.SpacingHeader(dx=20.5, dy=30.9),
        rows=rows,
        cols=cols,
    )


def test_all_four_parameters_and_the_spacing_header_reach_the_binary(fake_wbt):
    binary, record, sha = fake_wbt
    params = gm.WbtParams(search=50, threshold=1.0, skip=0, fdist=0)
    forms = gm.call_wbt(a_window(), params, binary=binary, pins=(sha,))
    seen = json.loads(record.read_text())
    for flag in ("--search=50", "--threshold=1.0", "--skip=0", "--fdist=0", "--forms",
                 "-r=Geomorphons"):
        assert flag in seen["args"], flag
    # D3: a projected header carrying the ground spacing, absent data declared as nodata.
    assert seen["crs"] == "EPSG:3857"
    assert seen["transform"] == [20.5, 0.0, 0.0, 0.0, -30.9, 0.0]
    assert seen["absent"] == 232 * 2
    assert forms.dtype == np.uint8 and forms.shape == (232, 236)
    assert set(np.unique(forms[:, 2:])) <= set(range(1, 11)) and (forms[:, :2] == 0).all()


def test_a_moved_parameter_reaches_the_binary_as_moved(fake_wbt):
    binary, record, sha = fake_wbt
    gm.call_wbt(a_window(), gm.WbtParams(search=25, threshold=0.75, skip=3, fdist=7),
                binary=binary, pins=(sha,))
    args = json.loads(record.read_text())["args"]
    assert {"--search=25", "--threshold=0.75", "--skip=3", "--fdist=7"} <= set(args)


def test_an_unpinned_binary_is_refused_before_it_runs(fake_wbt):
    binary, record, sha = fake_wbt
    params = gm.WbtParams(search=50, threshold=1.0, skip=0, fdist=0)
    with pytest.raises(ValueError, match=f"sha256 {sha}, which is not in"):
        gm.call_wbt(a_window(), params, binary=binary, pins=("0" * 64,))
    with pytest.raises(ValueError, match="not in"):
        gm.WbtLabeller(str(binary), ())
    assert not record.exists()


def test_an_epsg_4326_window_cannot_be_built_or_reach_the_call(fake_wbt):
    binary, record, sha = fake_wbt
    with pytest.raises(ValueError, match="reads the cell size in degrees as metres"):
        gm.SpacingHeader(dx=1 / 3600, dy=1 / 3600, crs="EPSG:4326")
    forged = a_window()
    object.__setattr__(forged.header, "crs", "EPSG:4326")  # past the constructor, somehow
    with pytest.raises(ValueError, match="cannot reach WBT"):
        gm.call_wbt(forged, gm.WbtParams(50, 1.0, 0, 0), binary=binary, pins=(sha,))
    assert not record.exists()


def test_wbt_nodata_on_a_pixel_the_ignore_rule_keeps_stops_the_pass():
    window = a_window(rows=50, cols=76)
    window.absent[:] = False
    shard = np.full((224, 224), 100, dtype=np.int16)

    def holes(w, p):
        forms = np.full(w.elevation.shape, 3, dtype=np.uint8)
        forms[w.rows + 10, w.cols + 10] = 0
        return forms

    with pytest.raises(ValueError, match="WBT wrote nodata at 1 pixels the ignore rule keeps"):
        gm.label_tile(window, gm.WbtParams(50, 1.0, 0, 0), holes, shard_raw_i16=shard)


def test_a_variant_needing_more_buffer_than_was_read_is_refused():
    window = a_window(rows=50, cols=60)
    with pytest.raises(ValueError, match="read the tile at its widest variant's L, once"):
        window.crop(100)


# --- Substitution point B: the buffered re-read on fixture COGs ----------------------------------


def read(root: Path, frame: Path, tid: str, search: int, ex=gm.Exceptions()) -> gm.Window:
    header = gm.spacing(tid)
    source = gm.RasterioCogSource(str(frame), str(root))
    return gm.read_buffered(source, tid, search, header.column_buffer(search), ex, header)


def expected(tid: str, window: gm.Window, *, wrap: bool = False) -> np.ndarray:
    """The fixture terrain at every pixel centre of ``window``, on the tile COG's grid."""
    s, r0, c0 = gm.parse_tile_id(tid)
    cog = gm.parse_cog_key(s)
    w = gm.width_of(cog.origin_lat)
    r = r0 - window.rows + np.arange(window.elevation.shape[0])
    c = c0 - window.cols + np.arange(window.elevation.shape[1])
    return elevation((cog.origin_lat + 1 - r / 3600)[:, None], (cog.origin_lon + c / w)[None, :])


def test_the_column_buffer_is_anisotropic_and_the_mosaic_lands_on_the_tile_grid(tmp_path):
    # 49°N: 3600 columns a degree, so a column is ~1.55x narrower on the ground than a row.
    frame = write_world(tmp_path, [(49, 10)])
    tid = f"{stem(49, 10)}_0_0"
    header = gm.spacing(tid)
    assert header.column_buffer(50) == 78  # ceil(50 * 30.90 m / 19.93 m) on the ellipsoid
    assert abs(header.column_buffer(50) - 50 / math.cos(math.radians(49.97))) < 1.5
    window = read(tmp_path, frame, tid, 50)
    assert window.elevation.shape == (224 + 100, 224 + 156)
    assert not window.absent.any()
    want = expected(tid, window)
    # The tile's own COG and its west neighbour share the grid: exact, pixel for pixel.
    np.testing.assert_array_equal(window.elevation[50:], want[50:])
    np.testing.assert_array_equal(window.elevation[window.centre], tile_pixels(tid))
    # North of 50°N is the 2400-column band: each of this tile's columns reads the 2400-band
    # column whose centre is nearest its own, every column filled (none dropped at the edge).
    lat = 50 + np.arange(50, 0, -1) / 3600
    lon = 10 + (np.arange(window.elevation.shape[1]) - window.cols) / 3600
    nearest = np.round(lon * 2400) / 2400
    np.testing.assert_array_equal(window.elevation[:50], elevation(lat[:, None], nearest[None, :]))
    assert len(np.unique(nearest)) < len(lon)  # coarser: neighbouring columns share a source


def test_absent_data_is_masked_and_written_as_nodata_never_zero_filled(tmp_path):
    frame = write_world(tmp_path, [(10, 30)], absent=((10, 29),))
    tid = f"{stem(10, 30)}_224_0"
    window = read(tmp_path, frame, tid, 50)
    cols = window.cols
    assert window.absent[:, :cols].all() and not window.absent[:, cols:].any()
    assert (window.elevation[:, :cols] == gm.WBT_NODATA).all()
    assert not (window.elevation == 0).any()


def test_a_verified_ocean_entry_bordering_land_is_refused(tmp_path):
    # Same absent neighbour, but listed as ocean: the tile beside it is land, so the entry is
    # wrong, and reading it as 0 m would invent a coastline.
    frame = write_world(tmp_path, [(10, 30)], absent=((10, 29),))
    ex = gm.Exceptions(verified_ocean=frozenset({stem(10, 29)}))
    with pytest.raises(ValueError, match="move the entry to absent_land"):
        read(tmp_path, frame, f"{stem(10, 30)}_224_0", 50, ex)


@pytest.mark.parametrize("wrap", [True, False])
def test_the_listed_polar_tiles_wrap_across_the_antimeridian(tmp_path, wrap):
    # S85 E179, 720 columns a degree: a tile at the east edge reaches past 180°, which only the
    # W180 COG holds.
    w = width(-85)
    frame = write_world(tmp_path, [], extra=[
        (-85, 179, (900, 1500), (w - 500, w)),
        (-85, -180, (900, 1500), (0, 400)),
    ])
    tid = f"{stem(-85, 179)}_1000_{w - 224}"
    ex = gm.Exceptions(antimeridian_wrap=frozenset({stem(-85, 179)}) if wrap else frozenset())
    window = read(tmp_path, frame, tid, 50, ex)
    east = slice(window.cols + 224, None)
    assert window.cols > 2 * 50  # the 80-85° band's columns are ~15 m on the ground
    if wrap:
        assert not window.absent.any()
        np.testing.assert_array_equal(window.elevation, expected(tid, window))
    else:
        assert window.absent[:, east].all() and not window.absent[:, : window.cols + 224].any()


def test_the_census_counts_the_absences_and_the_wraps_each_window_reaches(tmp_path):
    g = Geomorphon(
        verified_ocean=(stem(-45, 172),), absent_land=(stem(10, 29),),
        antimeridian_wrap=(stem(-85, 179),),
    )
    frame = frozenset({stem(10, 30), stem(11, 30), stem(11, 29), stem(-85, 179),
                       stem(-85, -180), stem(-44, 171), stem(-44, 172), stem(-45, 171)})
    lists = {
        "train": (f"{stem(10, 30)}_0_0", f"{stem(10, 30)}_224_0"),
        "val": (f"{stem(-44, 171)}_3376_{3600 - 224}",),
        "test": (f"{stem(-85, 179)}_1000_{width(-85) - 224}",),
    }
    result = geomorphon_pass.census(lists, frame, g)
    assert result.search == 100 and result.tiles == 4 and result.wrapped == 1
    assert result.absent == {stem(-45, 172): 1, stem(10, 29): 2}


# --- The live binary ------------------------------------------------------------------------------

#: The committed label hash of :func:`live_window` under the primary parameters, from the pinned
#: macOS build. ``None`` until #202 fetches and pins WBT v2.4.0 and runs this test once.
LIVE_LABEL_HASH: str | None = None


def live_window() -> gm.Window:
    """A committed fixture window: analytic relief with ridges, valleys and a flat floor."""
    rows, cols = 50, 77
    shape = (224 + 2 * rows, 224 + 2 * cols)
    y, x = np.mgrid[: shape[0], : shape[1]].astype(np.float64)
    z = 400 + 120 * np.sin(x / 23.0) * np.cos(y / 31.0) + 0.8 * x + 60 * np.sin((x + y) / 57.0)
    z = np.maximum(z, 380.0)  # a flat valley floor, so "flat" has somewhere to appear
    return gm.Window(tile_id="live-fixture", elevation=z.astype(np.float32),
                     absent=np.zeros(shape, dtype=bool),
                     header=gm.SpacingHeader(dx=20.2, dy=30.9), rows=rows, cols=cols)


def test_the_pinned_binary_labels_the_committed_window_to_the_committed_hash():
    binary = os.environ.get("DEMAE_WBT_BINARY")
    if not binary or not Path(binary).is_file():
        pytest.skip("WhiteboxTools v2.4.0 is not on this box (set DEMAE_WBT_BINARY; #202)")
    g = Geomorphon()
    if gm.binary_sha256(binary) not in g.wbt_sha256 or LIVE_LABEL_HASH is None:
        label = gm.label_tile(
            live_window(), gm.WbtParams(g.search, g.threshold, g.skip, g.fdist),
            lambda w, p: gm.call_wbt(w, p, binary=binary, pins=(gm.binary_sha256(binary),)),
            shard_raw_i16=np.rint(live_window().elevation[50:274, 77:301]).astype(np.int16),
        )[0]
        pytest.fail(
            f"#202: pin sha256 {gm.binary_sha256(binary)} in Geomorphon.wbt_sha256 and commit "
            f"LIVE_LABEL_HASH = {gm.content_hash(label)!r} (classes present: "
            f"{sorted(np.unique(label).tolist())})"
        )
    label, _ = gm.label_tile(
        live_window(), gm.WbtParams(g.search, g.threshold, g.skip, g.fdist),
        gm.WbtLabeller(binary, g.wbt_sha256),
        shard_raw_i16=np.rint(live_window().elevation[50:274, 77:301]).astype(np.int16),
    )
    assert gm.content_hash(label) == LIVE_LABEL_HASH


def shutil_rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path)
