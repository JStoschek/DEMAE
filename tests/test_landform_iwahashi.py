"""Plan §3.2's Iwahashi & Pike tile labels, rebuilt at ``freeze --task landform`` (#203).

Driven through :func:`benchmark.launch.main`, the spec's primary seam, against substitution
point B: a small synthetic GeoTIFF with the 15-group legend, zipped under the distributor's
member names, over a fixture corpus whose tile footprints land on whole label pixels. Every
expected label below is worked by hand from the paint plan in :func:`_paint`, not recomputed
the way the module computes it.

The label rasters' geometry: origin (6 E, 47 N), 1/1125 degree per pixel, so a 224 px GLO-30
tile in the 0-50 band (224/3600 = 70/1125 degrees each way) is exactly 70 x 70 label pixels,
4,900 of them. 60% of that is 42 whole rows and 50% is 35. The raster stops at 46.8 N (225
rows), so each COG's fourth tile (rows 210-280) is 15 rows on the map and 55 off it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmark import iwahashi, landform, launch
from benchmark.landform_config import LandformPaths
from dataset import manifest
from tests.conftest import build_corpus

PX = 1125  # label pixels per degree
TILE = 70  # label pixels per tile side

COGS = [
    ("Copernicus_DSM_COG_10_N46_00_E006_00_DEM", "+000_+040", "train"),
    ("Copernicus_DSM_COG_10_N46_00_E007_00_DEM", "+000_+040", "val"),
    ("Copernicus_DSM_COG_10_N46_00_E008_00_DEM", "+010_+040", "test"),
    ("Copernicus_DSM_COG_10_N46_00_E009_00_DEM", "+010_+040", "test"),
]
TILES_PER_COG = 4

#: The distributor's legend, transcribed from ``Landform_legend.csv`` in the pinned zip (its
#: first three columns), so the synthetic zip carries the same file the real one does.
LEGEND = """Value,GROUP,Overview
1,1a,Bedrock mountain
2,1b,Bedrock mountain
3,2a,Bedrock mountain
4,2b,Bedrock mountain
5,3a,Hills
6,3b,Hills
7,4,Large highland slope
8,5,Large highland slope
9,6,"Plateau, terrace, large lowland slope"
10,7,"Plateau, terrace, large lowland slope"
11,8,"Plateau, terrace, large lowland slope"
12,9,"Plateau, terrace, large lowland slope"
13,10,Plain
14,11,Plain
15,12,Plain
"""


def _tid(cog: int, j: int) -> str:
    return manifest.make_tile_id(COGS[cog][0], j * 224, 0)


#: The one test tile whose manifest says it is not all land: #150's filter drops it.
WET = _tid(3, 2)


def _paint() -> np.ndarray:
    """The synthetic label raster. Background 5 (group 3a); each test tile painted on purpose."""
    raster = np.full((225, 4 * PX), 5, dtype=np.uint8)

    def tile(cog: int, j: int) -> np.ndarray:
        c0 = (cog) * PX  # COG E00(6+cog) starts at column (6 + cog - 6) * PX
        return raster[j * TILE : (j + 1) * TILE, c0 : c0 + TILE]

    t = tile(2, 0)  # 42 rows 2a, 28 rows 2b: exactly 60%
    t[:42], t[42:] = 3, 4
    t = tile(2, 1)  # 35 rows 1a, 35 rows nodata: exactly 50%, nodata against the majority
    t[:35], t[35:] = 1, 255
    t = tile(2, 2)  # 34 rows 1b, 36 rows of the unlegended 0: 0 counts against, 34/70
    t[:34], t[34:] = 2, 0
    raster[210:225, 2 * PX : 2 * PX + TILE] = 15  # tile (2, 3): 15 rows of 12 on the map
    tile(3, 0)[:] = 255  # all nodata: no label at all
    t = tile(3, 1)  # 41 rows of 4, 29 of 5: 41/70, between the thresholds
    t[:41], t[41:] = 7, 8
    return raster


#: tile -> (group, majority fraction in parts per million), worked from the paint plan.
EXPECTED = {
    _tid(0, 0): ("3a", 1_000_000),
    _tid(0, 1): ("3a", 1_000_000),
    _tid(0, 2): ("3a", 1_000_000),
    _tid(0, 3): ("3a", 214_286),  # 15/70 on the map, the rest south of its edge
    _tid(1, 0): ("3a", 1_000_000),
    _tid(1, 1): ("3a", 1_000_000),
    _tid(1, 2): ("3a", 1_000_000),
    _tid(1, 3): ("3a", 214_286),
    _tid(2, 0): ("2a", 600_000),
    _tid(2, 1): ("1a", 500_000),
    _tid(2, 2): ("1b", 485_714),
    _tid(2, 3): ("12", 214_286),
    _tid(3, 0): ("none", 0),
    _tid(3, 1): ("4", 585_714),
    _tid(3, 2): ("3a", 1_000_000),
    _tid(3, 3): ("3a", 214_286),
}

#: Land-filtered test survivors at 60% and 50%: (2,0) at both; (2,1) and (3,1) at 50% only.
#: (3,2) clears both but is the wet tile.
TEST_SURVIVORS = (1, 3)


def _write_zip(path: Path, raster: np.ndarray, legend: str = LEGEND) -> str:
    import rasterio
    from rasterio.transform import from_origin

    buf = io.BytesIO()
    with rasterio.MemoryFile() as mem:
        with mem.open(
            driver="GTiff", width=raster.shape[1], height=raster.shape[0], count=1,
            dtype="uint8", crs="EPSG:4326", nodata=255, compress="deflate",
            transform=from_origin(6.0, 47.0, 1 / PX, 1 / PX),
        ) as dst:
            dst.write(raster, 1)
        tif = mem.read()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(iwahashi.TIF_MEMBER, tif)
        z.writestr(iwahashi.LEGEND_MEMBER, legend)
    path.write_bytes(buf.getvalue())
    return hashlib.md5(buf.getvalue()).hexdigest()


def _corpus(root: Path) -> Path:
    build_corpus(
        root, COGS, tiles_per_cog=TILES_PER_COG,
        metrics_fn=lambda c, j: {"slope_mean": 0.1 + c + 0.25 * j},
    )
    # build_corpus places tiles at lon=c, lat=j; put each at its footprint's true centre, the
    # way the shipped manifest does, and wet one test tile.
    table = pq.read_table(root / "manifest.parquet")
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    for i, tid in enumerate(cols["tile_id"]):
        cog = next(k for k, (stem, _, _) in enumerate(COGS) if tid.startswith(stem))
        j = cols["win_row0"][i] // 224
        cols["lon"][i] = 6 + cog + 112 / 3600
        cols["lat"][i] = 47 - (j * 224 + 112) / 3600
        if tid == WET:
            cols["land_frac"][i] = 0.5
    pq.write_table(pa.table(cols, schema=table.schema), root / "manifest.parquet")
    return root


def _write_lists(task1: Path) -> None:
    by_split = {"train": "head_train_tiles.csv", "val": "val_tiles.csv", "test": "test_tiles.csv"}
    for split, name in by_split.items():
        ids = sorted(
            _tid(c, j) for c, cog in enumerate(COGS) if cog[2] == split
            for j in range(TILES_PER_COG)
        )
        (task1 / name).write_text("tile_id,stratum_id\n" + "".join(f"{t},0\n" for t in ids))


@pytest.fixture
def box(tmp_path):
    for d in ("task1", "task2", "release", "labels"):
        (tmp_path / d).mkdir()
    _corpus(tmp_path / "corpus")
    _write_lists(tmp_path / "task1")
    md5 = _write_zip(tmp_path / "iwahashi.zip", _paint())
    return tmp_path, md5


@pytest.fixture
def flags(box):
    root, md5 = box
    return [
        "--task", "landform",
        "--set", f'paths.artifacts_dir="{root / "task2"}"',
        "--set", f'paths.task1_artifacts_dir="{root / "task1"}"',
        "--set", f'paths.results_path="{root / "results.parquet"}"',
        "--set", f'paths.corpus_dir="{root / "corpus"}"',
        "--set", f'paths.iwahashi_zip="{root / "iwahashi.zip"}"',
        "--set", f'paths.iwahashi_table="{root / "labels" / "iwahashi.csv"}"',
        "--set", f'iwahashi.zip_md5="{md5}"',
        "--set", f"iwahashi.test_survivors={list(TEST_SURVIVORS)}",
    ]


@pytest.fixture
def only_this_stage(monkeypatch):
    """Every freeze stage but this ticket's made a no-op, so the test reads one stage's output
    and stays true as the sibling tickets (#199, #200, #201, #218) fill theirs."""
    stages = tuple(
        stage if stage.ticket == "#203" else replace(stage, write=lambda resolution: [])
        for stage in landform.FREEZE_STAGES
    )
    monkeypatch.setattr(landform, "FREEZE_STAGES", stages)


def _table(root: Path) -> dict[str, tuple[str, int]]:
    return {
        row.tile_id: (row.group, row.frac_ppm)
        for row in iwahashi.read_table(root / "labels" / "iwahashi.csv").values()
    }


# --- The footprint majority, both thresholds, nodata and 0 against it ---------


def test_freeze_labels_every_split_by_the_whole_footprint_majority(box, flags, only_this_stage):
    root, _ = box
    assert launch.main(["freeze", *flags]) == 0
    assert _table(root) == EXPECTED


def test_both_thresholds_read_the_same_table_with_nodata_and_zero_against_the_majority(
    box, flags, only_this_stage
):
    """Exactly 60% survives both; exactly 50% with the rest nodata survives only the
    sensitivity row; 34/70 with the rest 0 survives neither, though every labelled pixel in it
    is 1b."""
    root, _ = box
    launch.main(["freeze", *flags])
    table = iwahashi.read_table(root / "labels" / "iwahashi.csv")
    at = {tid: (row.at(0.60), row.at(0.50)) for tid, row in table.items()}
    assert at[_tid(2, 0)] == ("2a", "2a")
    assert at[_tid(2, 1)] == (None, "1a")
    assert at[_tid(2, 2)] == (None, None)
    assert at[_tid(3, 0)] == (None, None)
    assert at[_tid(3, 1)] == (None, "4")


# --- freeze prints the survivors and checks them against #193's ---------------


def test_freeze_prints_survivors_per_split_and_band_and_checks_the_test_counts(
    box, flags, only_this_stage, capsys
):
    launch.main(["freeze", *flags])
    out = capsys.readouterr().out
    assert "survivors after #150's land filter at 60% / 50%" in out
    rows = {
        tuple(line.split()[:2]): line.split()[2:]
        for line in out.splitlines()
        if line.startswith(" ") and line.split()[:1] in (["train"], ["val"], ["test"])
    }
    # test, every band: 8 tiles, 7 on land; 1 (2) at 60%, 3 (4) at 50%.
    assert rows[("test", "all")] == ["8", "7", "1", "(2)", "3", "(4)"]
    assert rows[("train", "all")] == ["4", "4", "3", "(3)", "3", "(3)"]
    assert {band for split, band in rows if split == "test"} == {"all", "R1", "R2", "R3", "R4"}
    assert "test 1 / 3 at 60% / 50% after the land filter: #193's counts, reproduced" in out


def test_freeze_refuses_a_rebuild_that_misses_193s_test_counts_and_writes_nothing(
    box, flags, only_this_stage
):
    root, _ = box
    with pytest.raises(ValueError, match="does not reproduce #193's test counts.*1 / 3 at 60%"):
        launch.main(["freeze", *flags, "--set", "iwahashi.test_survivors=[1985, 2712]"])
    assert not (root / "task2" / iwahashi.FINGERPRINT_NAME).exists()
    assert not (root / "labels" / "iwahashi.csv").exists()


def test_the_real_counts_are_the_default_check():
    from benchmark.landform_config import LandformConfig

    assert LandformConfig().iwahashi.test_survivors == (1_985, 2_712)


# --- The zip's md5 and the committed fingerprint ------------------------------


def test_a_zip_whose_md5_is_not_the_committed_one_is_refused(box, flags, only_this_stage):
    """The default md5 is the distributor's zip, which the synthetic one is not."""
    root, md5 = box
    pinned = "5ae98dc7782c13384a1df4d8d609418f"
    with pytest.raises(ValueError, match=f"has md5 {md5}, and the committed one is {pinned}"):
        launch.main(["freeze", *flags, "--set", f'iwahashi.zip_md5="{pinned}"'])
    assert not (root / "task2" / iwahashi.FINGERPRINT_NAME).exists()
    assert not (root / "labels" / "iwahashi.csv").exists()


def test_a_missing_zip_names_where_to_get_it(box, flags, only_this_stage):
    root, _ = box
    with pytest.raises(FileNotFoundError, match="gisstar.gsi.go.jp/terrain2018"):
        launch.main(["freeze", *flags, "--set", f'paths.iwahashi_zip="{root / "absent.zip"}"'])


def test_freeze_commits_the_fingerprint_and_not_the_table(box, flags, only_this_stage):
    root, md5 = box
    launch.main(["freeze", *flags])
    assert sorted(p.name for p in (root / "task2").iterdir()) == [iwahashi.FINGERPRINT_NAME]
    fp = json.loads((root / "task2" / iwahashi.FINGERPRINT_NAME).read_text())
    table = (root / "labels" / "iwahashi.csv").read_bytes()
    assert fp["table_md5"] == hashlib.md5(table).hexdigest()
    assert fp["source_zip_md5"] == md5
    assert fp["rows"] == len(EXPECTED)
    assert fp["survivors"]["test"]["all"]["land>=60%"] == 1


def test_freeze_is_idempotent_and_refuses_a_moved_fingerprint(box, flags, only_this_stage):
    root, _ = box
    launch.main(["freeze", *flags])
    path = root / "task2" / iwahashi.FINGERPRINT_NAME
    before = path.read_bytes()
    assert launch.main(["freeze", *flags]) == 0
    assert path.read_bytes() == before
    # The same layout with one train tile repainted: the table moves, so its fingerprint would.
    raster = _paint()
    raster[0:TILE, 0:TILE] = 6
    md5 = _write_zip(root / "iwahashi.zip", raster)
    with pytest.raises(FileExistsError, match="Iwahashi table fingerprint"):
        launch.main(["freeze", *flags, "--set", f'iwahashi.zip_md5="{md5}"'])
    assert path.read_bytes() == before


# --- run refuses a table the fingerprint does not describe --------------------


def _run_iwahashi(flags):
    return launch.main(["run", "--dry-run", "--label-set", "iwahashi", *flags])


def test_run_resolves_the_rebuilt_table_by_its_fingerprint(box, flags, only_this_stage, capsys):
    root, _ = box
    launch.main(["freeze", *flags])
    capsys.readouterr()
    assert _run_iwahashi(flags) == 0
    out = capsys.readouterr().out
    md5 = hashlib.md5((root / "labels" / "iwahashi.csv").read_bytes()).hexdigest()
    assert f"iwahashi.csv  md5 {md5}" in out
    assert not any(line.startswith("labels ") for line in out.splitlines())


def test_run_refuses_a_rebuilt_table_whose_fingerprint_does_not_match(
    box, flags, only_this_stage, capsys
):
    root, _ = box
    launch.main(["freeze", *flags])
    capsys.readouterr()
    table = root / "labels" / "iwahashi.csv"
    table.write_text(table.read_text().replace(",2a,600000", ",2b,600000"))
    with pytest.raises(ValueError, match="does not match the committed fingerprint"):
        _run_iwahashi(flags)
    assert capsys.readouterr().out == ""


def test_run_refuses_a_table_committed_under_another_rule(box, flags, only_this_stage):
    launch.main(["freeze", *flags])
    with pytest.raises(ValueError, match="committed under a different rule"):
        _run_iwahashi([*flags, "--set", "iwahashi.threshold=0.7"])


def test_a_geomorphon_cell_does_not_read_the_iwahashi_table(box, flags, only_this_stage, capsys):
    root, _ = box
    launch.main(["freeze", *flags])
    (root / "labels" / "iwahashi.csv").write_text("tampered\n")
    capsys.readouterr()
    assert launch.main(["run", "--dry-run", *flags]) == 0
    assert "iwahashi" not in capsys.readouterr().out


# --- The overview mapping: a function of the 15, a/b apart --------------------


def test_the_15_groups_stay_apart_and_the_overview_is_a_function_of_them(
    box, flags, only_this_stage
):
    root, _ = box
    launch.main(["freeze", *flags])
    table = iwahashi.read_table(root / "labels" / "iwahashi.csv")
    one_a, one_b = table[_tid(2, 1)], table[_tid(2, 2)]
    # 1a and 1b are two labels at the headline level ...
    assert one_a.at(0.4) == "1a" and one_b.at(0.4) == "1b"
    # ... and one overview class, read off the 15-group label and nothing else.
    assert one_a.overview_at(0.4) == one_b.overview_at(0.4) == "Bedrock mountain"
    assert one_a.overview_at(0.6) is None
    assert len(set(iwahashi.GROUPS)) == 15
    assert set(iwahashi.OVERVIEW) == set(iwahashi.GROUPS)
    assert len(iwahashi.OVERVIEW_CLASSES) == 5
    for a, b in (("1a", "1b"), ("2a", "2b"), ("3a", "3b")):
        assert iwahashi.OVERVIEW[a] == iwahashi.OVERVIEW[b]


REPO = Path(__file__).resolve().parents[1]
REAL_ZIP = REPO / LandformPaths.iwahashi_zip


@pytest.mark.skipif(not REAL_ZIP.is_file(), reason="the distributor's zip is not on this box")
def test_the_real_zip_reproduces_193s_counts_and_the_committed_fingerprint(
    tmp_path, only_this_stage, capsys
):
    """The acceptance run, live: ~1 min over all 27,000 listed tiles. ``freeze`` against a copy
    of the committed fingerprint is a no-op only if the rebuilt table's md5 is the committed
    one, and it refuses unless the land-filtered test counts are §3.2's 1,985 and 2,712."""
    committed = REPO / "artifacts" / "task2" / iwahashi.FINGERPRINT_NAME
    (tmp_path / "task2").mkdir()
    (tmp_path / "task2" / iwahashi.FINGERPRINT_NAME).write_bytes(committed.read_bytes())
    table = tmp_path / "iwahashi.csv"
    assert launch.main([
        "freeze", "--task", "landform",
        "--set", f'paths.artifacts_dir="{tmp_path / "task2"}"',
        "--set", f'paths.task1_artifacts_dir="{REPO / "artifacts" / "task1"}"',
        "--set", f'paths.corpus_dir="{REPO / "data" / "corpus_500k"}"',
        "--set", f'paths.iwahashi_zip="{REAL_ZIP}"',
        "--set", f'paths.iwahashi_table="{table}"',
    ]) == 0
    assert "test 1,985 / 2,712 at 60% / 50% after the land filter" in capsys.readouterr().out
    fp = json.loads(committed.read_text())
    assert hashlib.md5(table.read_bytes()).hexdigest() == fp["table_md5"]
    assert fp["rows"] == 27_000
    assert fp["survivors"]["test"]["all"]["tiles>=60%"] == 2_240  # §3.2, before the filter
    assert fp["survivors"]["test"]["all"]["tiles>=50%"] == 3_053


def test_a_legend_that_disagrees_with_the_committed_mapping_is_refused(box, flags, only_this_stage):
    root, _ = box
    md5 = _write_zip(root / "iwahashi.zip", _paint(), LEGEND.replace("2,1b,Bedrock", "2,1b,Hills"))
    with pytest.raises(ValueError, match="does not match the committed groups and overview"):
        launch.main(["freeze", *flags, "--set", f'iwahashi.zip_md5="{md5}"'])
