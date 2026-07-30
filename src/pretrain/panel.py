"""The fixed val panel: which 40 tiles §10.2's grids render, committed (#98).

Plan §10.2 asks for reconstruction grids "on a fixed panel: 2 val tiles from each of the 20
strata (40 tiles)", and §15's checklist item 3 makes committing that panel a gate for R1. This
module draws it, writes it, reads it back, and refuses it when it no longer describes the
corpus in front of it.

## Why the tiles are an artifact and not a query

A reconstruction grid is never read on its own. It is read against *the last one* — has epoch
200 sharpened the drainage epoch 100 smeared? — and against the other arm's, and against the
same panel under a different mask ratio. Every one of those comparisons is between two
pictures, so the tiles under them have to be the same tiles. A panel re-drawn per run (or
per process, or per corpus rebuild) would keep rendering perfectly good pictures of a
*different* 40 tiles, and nothing in the picture would say so.

So this joins the family the selection list and the split assignment are in:
**the committed file is what later runs re-read, never this code re-run**
(:data:`dataset.config.VAL_PANEL_PATH`). The draw below is how the file was produced once; a
run consumes :func:`read_panel` and :func:`resolve_panel`.

## Two per stratum, and only from val

The strata are the project's own terrain families (ADR-0005's k=20), which is what makes the
panel a *diagnostic* rather than 40 pretty pictures: the failure §10.2 is built to catch —
"reconstructs alpine ridges but smears low-relief drainage", the corpus card §3.1 prediction
that [ADR-0008](../../docs/adr/0008-accept-f3-drainage-shortfall-strata-2-17.md) accepted a
known shortfall on — is a statement about *which strata* look wrong. Two tiles per stratum is
what plan §10.2 asks for and is the smallest number that shows whether a bad reconstruction is
the stratum or the tile.

**Val only, and enforced twice** — once in the draw, once at every load. A train tile in this
panel would be the model shown reconstructing something it was fit on, in the one place a
human looks to judge whether it generalizes, and it would look *better* than the rest of the
panel for exactly the wrong reason.

## The draw is uniform, deliberately

Ordered by the house keyed BLAKE2b (:func:`dataset.sampling._shuffle_key`) over the tile id,
so the panel is reproducible from the manifest and :data:`PANEL_SEED` alone and does not depend
on the row order the manifest happens to arrive in. :mod:`dataset.gallery` also offers a
*spanning* draw (stratum interior to edge) and this does not use it: the gallery was gating a
question about whether the strata are coherent families, where the boundary is where the
question lives, and this is monitoring a model, where the typical member is what a trend should
be read off.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np

from dataset.config import VAL_PANEL_PATH
from dataset.sampling import _shuffle_key  # the house determinism rule, as `gallery` imports it

__all__ = [
    "PANEL_SEED",
    "PANEL_SPLIT",
    "TILES_PER_STRATUM",
    "VAL_PANEL_PATH",
    "PanelTile",
    "ResolvedPanel",
    "draw_panel",
    "main",
    "read_panel",
    "resolve_from_manifest",
    "resolve_panel",
    "write_panel",
]

#: The committed draw seed. A named constant for the same reason
#: :data:`dataset.gallery.GALLERY_SEED` is one: the artifact has to be reconstructible from the
#: manifest after the fact, and a seed that lives in a shell invocation is not committed. The
#: digits read as "DEAE §10.2", beside :data:`pretrain.config.VAL_MASK_SEED`'s §10.1.
PANEL_SEED: Final = 0xDEAE_10_2

#: Plan §10.2's "2 val tiles from each of the 20 strata". The stratum count is *not* fixed
#: here — it is read off the corpus, because k is ADR-0005's decision and this module should
#: fail loudly against a corpus that disagrees rather than carry a second copy of 20.
TILES_PER_STRATUM: Final = 2

#: The only split a panel tile may come from. See the module docstring.
PANEL_SPLIT: Final = "val"

_HEADER: Final = "tile_id,stratum_id"


class PanelTile(NamedTuple):
    """One committed panel entry: a tile, and the stratum it is standing in for.

    The stratum rides in the artifact rather than being looked up, because it is what the
    grid is *captioned* with — "stratum 14 smears" is the finding — and because carrying it
    lets :func:`resolve_panel` check the committed label against the manifest's own. A tile
    whose stratum has been re-fit under the panel is not the tile the panel drew.
    """

    tile_id: str
    stratum_id: int


@dataclass(frozen=True)
class ResolvedPanel:
    """The committed panel joined to a corpus: rows to read, and the geography to shade with."""

    tiles: tuple[PanelTile, ...]
    #: Manifest row index per tile, in panel order.
    rows: np.ndarray
    #: Tile-centre latitude per tile. The hillshade needs true ground spacing, and Copernicus
    #: columns are up to 1.55x wider on the ground than its rows toward the poles
    #: (:func:`dataset.geometry.band_pixel_spacing_m`) — shading a polar tile as if its pixels
    #: were square draws it far too steep across.
    lat: np.ndarray

    def __len__(self) -> int:
        return len(self.tiles)

    @property
    def strata(self) -> list[int]:
        """The strata the panel covers, in order — 0..19 on the shipped corpus."""
        return sorted({t.stratum_id for t in self.tiles})


# --- The draw -----------------------------------------------------------------


def draw_panel(
    tile_id: np.ndarray,
    stratum_id: np.ndarray,
    split: np.ndarray,
    *,
    per_stratum: int = TILES_PER_STRATUM,
    seed: int = PANEL_SEED,
) -> list[PanelTile]:
    """Draw ``per_stratum`` val tiles from every stratum the val split holds.

    Ordered by a keyed hash of the tile id, so the result is a function of the *contents* of
    the manifest and the seed — not of the order the rows arrived in, which is not a contract
    anyone has made.

    A stratum too thin to fill its quota raises. It should be impossible on a real corpus —
    :func:`dataset.splits.plan_splits` already refuses a val pool missing a stratum — and if it
    ever happens the honest outcome is a panel that is not §10.2's rather than one that quietly
    covers 19 terrain families and is still called 40 tiles.
    """
    tile_id = np.asarray(tile_id)
    stratum_id = np.asarray(stratum_id)
    split = np.asarray(split)
    if not (len(tile_id) == len(stratum_id) == len(split)):
        raise ValueError(
            f"columns disagree on length: {len(tile_id)} ids, {len(stratum_id)} strata, "
            f"{len(split)} split labels"
        )

    eligible = np.flatnonzero(split == PANEL_SPLIT)
    if eligible.size == 0:
        raise ValueError(
            f"no tile carries split {PANEL_SPLIT!r}; §10.2's panel is drawn from the val "
            f"split alone, and this corpus holds {sorted(set(split.tolist()))}"
        )

    drawn: list[PanelTile] = []
    for stratum in sorted({int(s) for s in stratum_id[eligible]}):
        candidates = sorted(
            (str(tile_id[i]) for i in eligible if int(stratum_id[i]) == stratum),
            key=lambda t: _shuffle_key(t, seed),
        )
        if len(candidates) < per_stratum:
            raise ValueError(
                f"stratum {stratum} has {len(candidates)} val tile(s), too few for a panel of "
                f"{per_stratum} per stratum (plan §10.2)"
            )
        drawn += [PanelTile(t, stratum) for t in candidates[:per_stratum]]
    return sorted(drawn, key=lambda t: (t.stratum_id, t.tile_id))


# --- The artifact -------------------------------------------------------------


def write_panel(tiles: list[PanelTile], path: Path | str) -> Path:
    """Write the committed ``tile_id,stratum_id`` CSV, sorted and LF-terminated.

    Sorted by ``(stratum_id, tile_id)`` — the order the grids render in — so the file is
    byte-identical whatever order the draw handed the tiles over in, which is what makes its
    md5 a contract rather than a timestamp. Same rule as
    :func:`dataset.splits.write_assignment`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(tiles, key=lambda t: (t.stratum_id, t.tile_id))
    lines = [_HEADER] + [f"{t.tile_id},{t.stratum_id}" for t in ordered]
    path.write_text("\n".join(lines) + "\n")
    return path


def read_panel(path: Path | str) -> list[PanelTile]:
    """Read the committed panel back. Neither column ever contains a comma."""
    lines = Path(path).read_text().splitlines()
    if not lines or lines[0] != _HEADER:
        raise ValueError(f"{path} is not a val panel (header {lines[:1]}, expected {_HEADER!r})")
    tiles = []
    for line in lines[1:]:
        tile_id, stratum = line.rsplit(",", 1)
        tiles.append(PanelTile(tile_id, int(stratum)))
    return tiles


# --- Resolving it against a corpus --------------------------------------------

#: The manifest columns a resolve reads. Nothing else is touched, and no pixel is read.
MANIFEST_COLUMNS: Final = ("tile_id", "stratum_id", "split", "lat")

#: How many offending tiles an error message names before it stops listing.
_MAX_NAMED: Final = 5


def resolve_panel(
    tiles: list[PanelTile],
    *,
    tile_id: np.ndarray,
    stratum_id: np.ndarray,
    split: np.ndarray,
    lat: np.ndarray,
    per_stratum: int = TILES_PER_STRATUM,
) -> ResolvedPanel:
    """Join the committed panel to a corpus's manifest columns, refusing four kinds of drift.

    Each refusal is a way the panel can stop being the panel while every grid it produces
    still looks fine, which is why they are checked at load rather than trusted:

    - **A tile this corpus does not hold** — a panel from another tier, or from a rebuild that
      dropped the tile. Silently skipping it would shrink the panel and re-caption the grid.
    - **A tile that is no longer val** — a re-drawn split. The model is then shown
      reconstructing its own training data in the panel built to judge generalization.
    - **A committed stratum the manifest disagrees with** — re-fit centroids. The caption is
      the finding ("stratum 14 smears"), so a wrong label is a wrong conclusion.
    - **A panel that is not ``per_stratum`` from every stratum** — a hand-edited file. §10.2's
      claim is about terrain *families*, and it needs all of them present in equal measure.
    """
    tile_id = np.asarray(tile_id)
    stratum_id = np.asarray(stratum_id)
    split = np.asarray(split)
    lat = np.asarray(lat)
    row_of = {str(t): i for i, t in enumerate(tile_id.tolist())}

    absent = [t.tile_id for t in tiles if t.tile_id not in row_of]
    if absent:
        raise ValueError(
            f"the corpus does not hold {len(absent)} of {len(tiles)} panel tile(s) — a panel "
            f"drawn against another corpus, or a rebuild that dropped them. First: "
            f"{absent[:_MAX_NAMED]}"
        )

    rows = np.array([row_of[t.tile_id] for t in tiles], dtype=np.int64)
    off_split = [
        f"{t.tile_id} ({split[r]})" for t, r in zip(tiles, rows) if split[r] != PANEL_SPLIT
    ]
    if off_split:
        raise ValueError(
            f"{len(off_split)} panel tile(s) no longer carry split {PANEL_SPLIT!r} — the split "
            f"has been re-drawn under the panel, and the grids would show the model "
            f"reconstructing tiles it trained on. First: {off_split[:_MAX_NAMED]}"
        )

    relabelled = [
        f"{t.tile_id} (panel {t.stratum_id}, manifest {stratum_id[r]})"
        for t, r in zip(tiles, rows)
        if int(stratum_id[r]) != t.stratum_id
    ]
    if relabelled:
        raise ValueError(
            f"{len(relabelled)} panel tile(s) sit in a different stratum than the panel "
            f"records — re-fit centroids under a committed panel. First: "
            f"{relabelled[:_MAX_NAMED]}"
        )

    # Only one direction needs checking. A panel stratum the val split does not hold is
    # already impossible here: every tile has been shown to exist, to be val, and to carry the
    # stratum the manifest gives it — so its stratum is some val tile's stratum by
    # construction. What is left is the count.
    expected = sorted({int(s) for s in stratum_id[split == PANEL_SPLIT]})
    counts = Counter(t.stratum_id for t in tiles)
    wrong = {s: counts.get(s, 0) for s in expected if counts.get(s, 0) != per_stratum}
    if wrong:
        raise ValueError(
            f"§10.2's panel is {per_stratum} tiles from every stratum; this one has "
            f"{dict(sorted(wrong.items()))} against the {len(expected)} strata the val split "
            f"holds ({len(tiles)} tiles in total)"
        )

    return ResolvedPanel(tiles=tuple(tiles), rows=rows, lat=lat[rows].astype(float))


def resolve_from_manifest(
    tiles: list[PanelTile], manifest_path: Path | str, **kwargs
) -> ResolvedPanel:
    """:func:`resolve_panel` against a corpus's ``manifest.parquet``, reading four columns."""
    import pyarrow.parquet as pq

    table = pq.read_table(Path(manifest_path), columns=list(MANIFEST_COLUMNS))
    columns = {c: table[c].to_numpy(zero_copy_only=False) for c in MANIFEST_COLUMNS}
    return resolve_panel(tiles, **columns, **kwargs)


# --- The entrypoint -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python -m pretrain.panel`` — draw §10.2's panel and commit it. Run once per tier."""
    import pyarrow.parquet as pq

    from dataset.materialize import file_md5

    p = argparse.ArgumentParser(
        prog="python -m pretrain.panel",
        description="Draw and commit the fixed 40-tile val panel §10.2's grids render (#98).",
    )
    p.add_argument("--manifest", type=Path, default=Path("data/corpus_500k/manifest.parquet"))
    p.add_argument("--out", type=Path, default=Path(VAL_PANEL_PATH))
    p.add_argument("--per-stratum", type=int, default=TILES_PER_STRATUM)
    p.add_argument("--seed", type=int, default=PANEL_SEED)
    args = p.parse_args(argv)

    table = pq.read_table(args.manifest, columns=list(MANIFEST_COLUMNS))
    columns = {c: table[c].to_numpy(zero_copy_only=False) for c in MANIFEST_COLUMNS}
    drawn = draw_panel(
        columns["tile_id"],
        columns["stratum_id"],
        columns["split"],
        per_stratum=args.per_stratum,
        seed=args.seed,
    )
    # Resolved before it is written: the checks are the artifact's contract, and a panel that
    # would be refused at load is not one to commit.
    resolved = resolve_panel(drawn, **columns, per_stratum=args.per_stratum)
    out = write_panel(drawn, args.out)

    print(f"Panel of {len(resolved)} val tiles over {len(resolved.strata)} strata:")
    for stratum in resolved.strata:
        ids = [t.tile_id for t in resolved.tiles if t.stratum_id == stratum]
        print(f"  stratum {stratum:2d}  " + "  ".join(ids))
    print(f"  panel: {out}  md5 {file_md5(out)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
