"""Plan §3.2's Iwahashi & Pike tile labels, rebuilt from the distributor's zip (#203).

The independent label set: Iwahashi et al. (2018)'s 15 terrain groups, as Hengl's 7.5 arc-second
rasterization of the authors' polygons, reduced to one label per Task 1 tile under the rule
#193 fixed before any probe was scored. Promoted from #186's measurement script
(``scripts/p186_iwahashi_survival.py``); :func:`footprint`, :func:`overlap_weights` and
:func:`group_areas` are that script's, unchanged, because §3.2's counts came from them.

## The rule

- A tile's footprint is its 224 x 224 GLO-30 window in degrees, from the COG stem's SW corner,
  ``win_row0``/``win_col0`` and the band's columns per degree. Each label pixel is weighted by
  its overlap with that rectangle. (No half-pixel pixel-is-point shift: about 15 m against
  232 m label pixels, and #193's counts were produced without one.)
- The label is the group with the most area, and its **fraction is of the whole footprint**:
  nodata (255: water, and everything south of the map's 59.46 S edge) and the unlegended value
  0 (Arctic ice) count **against** the majority. A tie goes to the lower raster value.
- **≥ 60% is primary**, and C3 is read there only. **≥ 50% is a sensitivity row**. One rule
  labels train, val and test, and no survivor is re-weighted.
- The 15 groups are the headline and **a/b are not pooled** there. The 5-way overview is a
  function of the 15 by :data:`OVERVIEW`, which the zip's own legend is checked against.

## What is committed and what is rebuilt (§3.4)

The table is **not committed**: the distributor's grant is hedged and MERIT's upstream terms
carry a share-alike clause. What is committed is this code, the zip's md5
(``iwahashi.zip_md5``) and :data:`FINGERPRINT_NAME`, which holds the md5 of the table this code
rebuilds — the **table fingerprint** — with the rule and the survivor counts it produced. The
table is canonical text (sorted by tile id, LF line endings, the fraction as an integer number
of parts per million), so its md5 does not depend on the platform, and a fraction computed as
0.5999999999 on one numpy and 0.6000000001 on another is 600,000 on both.

``freeze --task landform`` rebuilds it, refuses a zip whose md5 is not the committed one,
refuses a rebuild whose test survivors are not #193's, and refuses (through
:func:`benchmark.artifacts.write_committed`) a rebuild whose fingerprint has moved. A ``run``
on the Iwahashi label set refuses a table on this box that does not match the fingerprint.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from . import artifacts
from .artifacts import Resolved
from .config import _to_dict

if TYPE_CHECKING:
    from .landform import LandformResolution
    from .landform_config import LandformConfig

#: The 15 groups, in raster order: value ``i + 1`` is ``GROUPS[i]``.
GROUPS: Final = (
    "1a", "1b", "2a", "2b", "3a", "3b", "4", "5", "6", "7", "8", "9", "10", "11", "12",
)

#: The legend's five-way overview, from ``Landform_legend.csv`` in the pinned zip. The
#: secondary table only: the headline keeps all 15 groups, a and b apart.
OVERVIEW: Final = {
    "1a": "Bedrock mountain",
    "1b": "Bedrock mountain",
    "2a": "Bedrock mountain",
    "2b": "Bedrock mountain",
    "3a": "Hills",
    "3b": "Hills",
    "4": "Large highland slope",
    "5": "Large highland slope",
    "6": "Plateau, terrace, large lowland slope",
    "7": "Plateau, terrace, large lowland slope",
    "8": "Plateau, terrace, large lowland slope",
    "9": "Plateau, terrace, large lowland slope",
    "10": "Plain",
    "11": "Plain",
    "12": "Plain",
}

#: The five overview classes, in legend order.
OVERVIEW_CLASSES: Final = tuple(dict.fromkeys(OVERVIEW.values()))

#: A tile with no labelled area at all: its whole footprint is nodata or 0.
NO_GROUP: Final = "none"

NODATA: Final = 255
UNLEGENDED: Final = 0

#: The distributor's member names inside the zip.
TIF_MEMBER: Final = (
    "7.5_arc_second_compressedGeoTIFF/GlobalTerrainClassification_Iwahashi_etal_2018.tif"
)
LEGEND_MEMBER: Final = "7.5_arc_second_compressedGeoTIFF/Landform_legend.csv"

#: Where the zip comes from. Plain HTTPS, no login; the page links the file relatively.
DISTRIBUTOR: Final = "https://gisstar.gsi.go.jp/terrain2018/ (7.5_arc_second_compressedGeoTIFF.zip)"

#: The committed fingerprint, in Task 2's artifacts directory.
FINGERPRINT_NAME: Final = "iwahashi_fingerprint.json"

TABLE_COLUMNS: Final = ("tile_id", "split", "group", "frac_ppm")

#: Fractions are stored as integer parts per million, and thresholds compared in the same unit.
PPM: Final = 1_000_000

#: Task 1's three lists, in the order the table's splits are reported. §3.3 reuses them.
SPLITS: Final = (
    ("train", artifacts.TILE_LIST_NAMES["20k"]),
    ("val", artifacts.VAL_TILES_NAME),
    ("test", artifacts.TEST_TILES_NAME),
)

_MANIFEST_COLUMNS: Final = (
    "tile_id", "source_cog", "win_row0", "win_col0", "lon", "lat", "split",
    "land_frac", "water_affected", "slope_mean",
)


def ppm(fraction: float) -> int:
    """A fraction (or a threshold) in the table's unit."""
    return round(fraction * PPM)


@dataclass(frozen=True)
class Label:
    """One row of the table: a tile, its split, its majority group and that group's fraction."""

    tile_id: str
    split: str
    #: One of :data:`GROUPS`, or :data:`NO_GROUP`.
    group: str
    #: The majority group's share of the whole footprint, in parts per million.
    frac_ppm: int

    def at(self, threshold: float) -> str | None:
        """The tile's 15-group label under ``threshold``, or ``None`` if it does not survive."""
        if self.group == NO_GROUP or self.frac_ppm < ppm(threshold):
            return None
        return self.group

    def overview_at(self, threshold: float) -> str | None:
        """The 5-way overview label — a function of :meth:`at`, never a second draw."""
        group = self.at(threshold)
        return None if group is None else OVERVIEW[group]


# --- The footprint and its majority (#186, unchanged) -------------------------


def footprint(source_cog: str, row0: int, col0: int, lat: float) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a tile window, in degrees."""
    from dataset.config import TILE_PX
    from dataset.geometry import COLS_PER_DEGREE, ROWS_PER_DEGREE, lat_band, parse_cog_key

    ref = parse_cog_key(source_cog)
    width = COLS_PER_DEGREE[lat_band(lat)]
    west = ref.origin_lon + col0 / width
    east = ref.origin_lon + (col0 + TILE_PX) / width
    north = ref.origin_lat + 1 - row0 / ROWS_PER_DEGREE
    south = ref.origin_lat + 1 - (row0 + TILE_PX) / ROWS_PER_DEGREE
    return west, south, east, north


def overlap_weights(lo: float, hi: float, start: float, step: float, n: int) -> np.ndarray:
    """Length of overlap of [lo, hi] with each of n cells [start + i*step, start + (i+1)*step]."""
    edges = start + step * np.arange(n + 1)
    return np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0.0, None)


def group_areas(src, bounds) -> np.ndarray:
    """Each raster value 0..255's share of the footprint; off-map area is filed under nodata."""
    from rasterio.windows import Window

    west, south, east, north = bounds
    t = src.transform
    c0 = math.floor((west - t.c) / t.a)
    c1 = math.ceil((east - t.c) / t.a)
    r0 = math.floor((north - t.f) / t.e)
    r1 = math.ceil((south - t.f) / t.e)
    areas = np.zeros(256)
    total = (east - west) * (north - south)
    # Rows/cols outside the raster (south of 59.46 S) are nodata by construction.
    rr0, rr1 = max(r0, 0), min(r1, src.height)
    cc0, cc1 = max(c0, 0), min(c1, src.width)
    if rr1 > rr0 and cc1 > cc0:
        arr = src.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
        wx = overlap_weights(west, east, t.c + cc0 * t.a, t.a, cc1 - cc0)
        # Rows run north to south: the cell edges in latitude descend, so work in -lat.
        wy = overlap_weights(-north, -south, -(t.f + rr0 * t.e), -t.e, rr1 - rr0)
        w = np.outer(wy, wx)
        np.add.at(areas, arr.ravel(), w.ravel())
    areas[NODATA] += max(total - areas.sum(), 0.0)
    return areas / total


def majority(areas: np.ndarray) -> tuple[str, int]:
    """#193's rule on one footprint: the top group, and its share of the **whole** footprint.

    Nodata and 0 are in the denominator and never the winner. ``np.argmax`` breaks a tie toward
    the lower raster value.
    """
    groups = areas[1 : len(GROUPS) + 1]
    if groups.sum() <= 0:
        return NO_GROUP, 0
    g = int(np.argmax(groups))
    return GROUPS[g], ppm(float(groups[g]))


# --- The one reader -----------------------------------------------------------


@contextmanager
def open_source(zip_path: Path | str, md5: str) -> Iterator:
    """The distributor's zip, opened in place — refused unless its md5 is ``md5``.

    The one place the label raster is read from (the spec's "one reader each"): the md5 is
    checked before a byte of the raster is, the legend inside the zip is checked against
    :data:`GROUPS` and :data:`OVERVIEW`, and the GeoTIFF is read through GDAL's ``/vsizip/``
    without being unpacked.
    """
    import rasterio

    path = Path(zip_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"the Iwahashi & Pike zip is not at {path}. Download it from {DISTRIBUTOR} "
            f"(md5 {md5}) and point paths.iwahashi_zip at it; it is read in place and never "
            f"committed (§3.4)"
        )
    got = artifacts.file_md5(path)
    if got != md5:
        raise ValueError(
            f"the Iwahashi & Pike zip at {path} has md5 {got}, and the committed one is {md5} "
            f"(iwahashi.zip_md5, §3.4). A different file is a different label source; the "
            f"rebuild refuses it rather than label against it"
        )
    with zipfile.ZipFile(path) as z:
        check_legend(z.read(LEGEND_MEMBER).decode("utf-8"))
    with rasterio.open(f"/vsizip/{path.resolve()}/{TIF_MEMBER}") as src:
        yield src


def check_legend(text: str) -> None:
    """Refuse a legend whose groups or overview disagree with the committed mapping."""
    rows = list(csv.DictReader(io.StringIO(text)))
    got = {int(r["Value"]): (r["GROUP"], r["Overview"]) for r in rows}
    want = {i + 1: (g, OVERVIEW[g]) for i, g in enumerate(GROUPS)}
    if got != want:
        raise ValueError(
            f"the zip's {LEGEND_MEMBER} does not match the committed groups and overview "
            f"mapping: it has {got}, the code has {want}"
        )


# --- The rebuild --------------------------------------------------------------


def rebuild(cfg: LandformConfig) -> tuple[list[Label], dict]:
    """Every tile of Task 1's three lists, labelled — and the manifest columns it read."""
    from pretrain.artifact import read_tile_strata

    from . import voids

    if cfg.iwahashi.groups != len(GROUPS):
        raise ValueError(
            f"iwahashi.groups is {cfg.iwahashi.groups}, but the map holds {len(GROUPS)} groups "
            f"and §3.2 pools none of them"
        )
    task1 = Path(cfg.paths.task1_artifacts_dir)
    wanted = {
        split: [tid for tid, _ in read_tile_strata(task1 / name, what=f"{split} list")]
        for split, name in SPLITS
    }
    cols = voids.read_manifest_columns(cfg.paths.corpus_dir, _MANIFEST_COLUMNS)
    index = {t: i for i, t in enumerate(cols["tile_id"])}
    todo = []
    for split, ids in wanted.items():
        for tid in ids:
            i = index.get(tid)
            if i is None:
                raise ValueError(f"{split} tile {tid} is not in {cfg.paths.corpus_dir}'s manifest")
            if cols["split"][i] != split:
                raise ValueError(
                    f"{tid} is on Task 1's {split} list but the manifest says "
                    f"{cols['split'][i]!r}; the lists and the corpus disagree"
                )
            todo.append((split, tid, i))
    labels = []
    with open_source(cfg.paths.iwahashi_zip, cfg.iwahashi.zip_md5) as src:
        # North to south, so consecutive reads share the raster's strips.
        for split, tid, i in sorted(todo, key=lambda item: -cols["lat"][item[2]]):
            bounds = footprint(
                str(cols["source_cog"][i]), int(cols["win_row0"][i]),
                int(cols["win_col0"][i]), float(cols["lat"][i]),
            )
            # The manifest's lon/lat is the window centre; a footprint that missed it is a bug.
            if (
                abs((bounds[0] + bounds[2]) / 2 - cols["lon"][i]) > 1e-6
                or abs((bounds[1] + bounds[3]) / 2 - cols["lat"][i]) > 1e-6
            ):
                raise ValueError(
                    f"{tid}'s footprint {bounds} is not centred on its manifest lon/lat "
                    f"({cols['lon'][i]}, {cols['lat'][i]})"
                )
            group, frac = majority(group_areas(src, bounds))
            labels.append(Label(tile_id=tid, split=split, group=group, frac_ppm=frac))
    labels.sort(key=lambda label: label.tile_id)
    return labels, {"columns": cols, "index": index}


def render_table(labels: list[Label]) -> bytes:
    """The canonical table: sorted by tile id, LF-terminated, integers only past the id."""
    lines = [",".join(TABLE_COLUMNS)]
    lines += [f"{x.tile_id},{x.split},{x.group},{x.frac_ppm}" for x in labels]
    return ("\n".join(lines) + "\n").encode("utf-8")


def read_table(path: Path | str) -> dict[str, Label]:
    """The rebuilt table, by tile id. Check its md5 against the fingerprint before trusting it."""
    rows = list(csv.DictReader(io.StringIO(Path(path).read_text())))
    if not rows or tuple(rows[0]) != TABLE_COLUMNS:
        raise ValueError(f"{path} is not an Iwahashi tile-label table (columns {TABLE_COLUMNS})")
    return {
        r["tile_id"]: Label(r["tile_id"], r["split"], r["group"], int(r["frac_ppm"]))
        for r in rows
    }


# --- Survivors, the fingerprint and the freeze stage --------------------------


def _pct(threshold: float) -> str:
    return f"{threshold:.0%}"


def survivors(cfg: LandformConfig, labels: list[Label], read: dict) -> dict:
    """Surviving tiles per split and per ruggedness band, at both thresholds.

    ``land`` is #150's filter (:func:`benchmark.voids.eligible`), the population §3.2 counts
    on; ``tiles`` is every tile on the list. Bands are §5's quartiles of ``slope_mean`` over the
    eligible test pool, applied to every split.
    """
    from . import voids

    cols, index = read["columns"], read["index"]
    bands = voids.Bands.from_manifest(cfg.paths.corpus_dir, split="test")
    thresholds = (cfg.iwahashi.threshold, cfg.iwahashi.sensitivity_threshold)
    keys = [f"{k}>={_pct(t)}" for t in thresholds for k in ("land", "tiles")]
    out: dict = {}
    for split, _ in SPLITS:
        table = {
            name: {"tiles": 0, "land": 0, **dict.fromkeys(keys, 0)}
            for name in ("all", *(bands.label(b) for b in range(bands.count)))
        }
        for label in labels:
            if label.split != split:
                continue
            i = index[label.tile_id]
            land = bool(voids.eligible(
                np.array([cols["land_frac"][i]]), np.array([cols["water_affected"][i]])
            )[0])
            for name in ("all", bands.label(bands.of(float(cols["slope_mean"][i])))):
                cell = table[name]
                cell["tiles"] += 1
                cell["land"] += land
                for t in thresholds:
                    if label.at(t) is not None:
                        cell[f"tiles>={_pct(t)}"] += 1
                        cell[f"land>={_pct(t)}"] += land
        out[split] = table
    return out


def _survivor_lines(cfg: LandformConfig, counts: dict) -> list[str]:
    hi, lo = _pct(cfg.iwahashi.threshold), _pct(cfg.iwahashi.sensitivity_threshold)
    lines = [
        f"{'iwahashi':<16} survivors after #150's land filter at {hi} / {lo} "
        f"(every listed tile in brackets)",
        f"{'':<16} {'split':<6}{'band':<5}{'tiles':>8}{'land':>8}{'>=' + hi:>17}{'>=' + lo:>17}",
    ]

    def pair(c: dict, pct: str) -> str:
        return f"{c['land>=' + pct]:>8,}{'(' + format(c['tiles>=' + pct], ',') + ')':>9}"

    for split, table in counts.items():
        for band, c in table.items():
            lines.append(
                f"{'':<16} {split:<6}{band:<5}{c['tiles']:>8,}{c['land']:>8,}"
                f"{pair(c, hi)}{pair(c, lo)}"
            )
    return lines


def fingerprint_path(cfg: LandformConfig) -> Path:
    return Path(cfg.paths.artifacts_dir) / FINGERPRINT_NAME


def committed(cfg: LandformConfig) -> dict | None:
    """The committed fingerprint, or ``None`` before ``freeze`` has written one."""
    path = fingerprint_path(cfg)
    return json.loads(path.read_text()) if path.is_file() else None


def inputs(cfg: LandformConfig, *, verb: str) -> tuple[Resolved, Resolved]:
    """A cell on the Iwahashi label set reads two things: the committed fingerprint, and the
    table rebuilt on this box. Both are resolved and printed; ``run`` refuses a table the
    fingerprint does not describe (story 24), upstream of ``--dry-run``, because a pre-flight
    that printed a drifted table as a normal one would be worse than none."""
    fp, table = fingerprint_path(cfg), Path(cfg.paths.iwahashi_table)
    resolved = (
        Resolved(
            key="iwahashi_fingerprint",
            label="iwahashi fprint",
            path=str(fp),
            md5=artifacts.file_md5(fp),
            absent_note="not frozen yet (`python -m benchmark.launch freeze --task landform`)",
        ),
        Resolved(
            key="iwahashi_table",
            label="iwahashi table",
            path=str(table),
            md5=artifacts.file_md5(table),
            absent_note="not rebuilt on this box (`freeze --task landform` rebuilds it from "
            "the zip)",
        ),
    )
    if verb == "run":
        check(cfg, resolved[1].md5)
    return resolved


def check(cfg: LandformConfig, table_md5: str | None) -> None:
    """Refuse a fingerprint made under another rule, and a table whose md5 is not the one the
    fingerprint holds. Absence is a state, not a refusal, as for every input."""
    fp = committed(cfg)
    if fp is None:
        return
    rule = _to_dict(cfg.iwahashi)
    if fp["rule"] != rule:
        raise ValueError(
            f"{fingerprint_path(cfg)} was committed under a different rule ({fp['rule']}) from "
            f"this cell's iwahashi group ({rule}); labels drawn under one cannot be scored as "
            f"the other"
        )
    if table_md5 is not None and table_md5 != fp["table_md5"]:
        raise ValueError(
            f"the Iwahashi table at {cfg.paths.iwahashi_table} has md5 {table_md5}, which does "
            f"not match the committed fingerprint {fp['table_md5']} ({fingerprint_path(cfg)}). "
            f"Rebuild it with `freeze --task landform` rather than train on labels that cannot "
            f"be shown to be the committed ones"
        )


def freeze(resolution: LandformResolution) -> list[str]:
    """``freeze --task landform``'s Iwahashi stage: rebuild, check, fingerprint, then write.

    In that order, so nothing is committed about a table that fails a check, and the table on
    this box is never one the committed fingerprint disagrees with: the zip's md5 first (inside
    :func:`open_source`), then #193's test counts, then the fingerprint through
    :func:`artifacts.write_committed` (a no-op if unchanged, a refusal if moved), and only then
    the table itself, atomically.
    """
    cfg = resolution.config
    rule = cfg.iwahashi
    labels, read = rebuild(cfg)
    counts = survivors(cfg, labels, read)
    lines = _survivor_lines(cfg, counts)
    hi, lo = _pct(rule.threshold), _pct(rule.sensitivity_threshold)
    test = counts["test"]["all"]
    got = (test[f"land>={hi}"], test[f"land>={lo}"])
    if got != tuple(rule.test_survivors):
        raise ValueError(
            "the Iwahashi rebuild does not reproduce #193's test counts after #150's land "
            f"filter: {got[0]:,} / {got[1]:,} at {hi} / {lo}, against iwahashi.test_survivors "
            f"{rule.test_survivors[0]:,} / {rule.test_survivors[1]:,}. A footprint or "
            "land-filter bug stops here, before any classifier is fitted; nothing was written.\n"
            + "\n".join(lines)
        )
    table = render_table(labels)
    table_md5 = hashlib.md5(table).hexdigest()
    body = {
        "what": "The Iwahashi & Pike (2018) tile-label table's fingerprint (plan §3.2, #193's "
        "rule, #203). The table itself is not committed (§3.4): `freeze --task landform` "
        "rebuilds it from the zip, and `run` refuses one whose md5 is not table_md5.",
        "source_zip_md5": rule.zip_md5,
        "table_md5": table_md5,
        "table_columns": list(TABLE_COLUMNS),
        "rows": len(labels),
        "rule": _to_dict(rule),
        "overview": OVERVIEW,
        "survivors": counts,
    }
    fp_path, fp_md5 = artifacts.write_committed(
        fingerprint_path(cfg),
        lambda: (json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
        what="Iwahashi table fingerprint",
    )
    out = Path(cfg.paths.iwahashi_table)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(table)
    os.replace(tmp, out)
    return [
        f"{'iwahashi zip':<16} {cfg.paths.iwahashi_zip}  md5 {rule.zip_md5}",
        f"{'iwahashi table':<16} {out}  md5 {table_md5}  ({len(labels):,} tiles; rebuilt, "
        f"not committed)",
        f"{'iwahashi fprint':<16} {fp_path}  md5 {fp_md5}",
        *lines,
        f"{'iwahashi':<16} test {got[0]:,} / {got[1]:,} at {hi} / {lo} after the land filter: "
        f"#193's counts, reproduced",
    ]
