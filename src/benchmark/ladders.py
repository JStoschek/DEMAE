"""§3.3's three geocell ladders: exact tile budgets over whole geocells, one per seed (#200).

[Plan §3.3](../../docs/research/task2-landform-plan.md) is the whole specification, and #189
committed no code, so this is new. What it says, in the order this module does it:

* **One walk per ladder.** Ladder *i* is a single order over Task 1's 20,000-tile head-train
  list: :func:`dataset.splits.tie_rank` keyed with ladder *i*'s draw seed over the geocell ids,
  then over the tile ids within each cell. The key is a per-id BLAKE2b, so ranking one cell's
  tiles gives the order the whole list would give them. No RNG, so a redraw is byte-identical.
* **A rung is a prefix of the walk**, of exactly its tile budget (200, 2,000, 20,000). Cells
  are taken whole until the next would pass the budget; that cell contributes just enough of
  its own tiles, in keyed order, to land on the budget. A prefix is precisely that rule, so a
  rung holds at most one partial cell, and its id is recorded. A rung ending on a cell boundary
  has none, and records ``null`` rather than the last whole cell.
* **Nested by construction, asserted anyway** (:func:`check`), on every ladder ``freeze``
  writes, so a later edit to the walk cannot break it quietly.
* **Not stratified, never redrawn.** Each rung's strata and band coverage is committed beside
  it instead (:func:`coverage`). Per-class train-pixel counts need the geomorphon labels, so
  #201/#202 add them; until then the report says so rather than omitting the field.

## The files

``ladder_<i>.csv`` is :mod:`pretrain.artifact`'s ``tile_id,stratum_id`` family format, written
in **walk order** rather than sorted, which that format allows (order is the caller's), because
walk order is what makes a rung a prefix: rung *r* of ladder *i* is the first
``ladders.budgets[r]`` rows. ``ladder_<i>_coverage.json`` records the ladder's index, its draw
seed, the md5 of the walk it describes and of the head-train list it walked, and each rung's
coverage. The resolver reads the JSON back to refuse a walk filed under the wrong ladder.

Both go through :func:`benchmark.artifacts.write_committed`, so a second ``freeze`` is a no-op
and one that would write a different walk refuses rather than overwrite it.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from pretrain.artifact import read_tile_strata, render_tile_strata

from . import artifacts
from .artifacts import Resolved
from .config import STRATUM_COUNT
from .landform_config import RUNGS, LandformConfig

if TYPE_CHECKING:
    from .landform import LandformResolution

#: The split every head-train tile must carry. Task 1 drew the list from it.
TRAIN_SPLIT: Final = "train"

#: What the coverage report says about the per-class pixel counts it does not carry yet.
CLASS_PIXELS_NOTE: Final = (
    "not computed yet: per-class train-pixel counts need the geomorphon labels, which the "
    "label pass adds (#201, #202)"
)

#: How the walk is ordered, in the plan's words, so the report is readable without this module.
ORDER: Final = (
    "plan §3.3: tie_rank keyed with the draw seed over geocell ids, then over tile ids within "
    "a cell; a rung is the first `tiles` rows of the walk"
)


def ladder_name(index: int) -> str:
    """The committed walk's file name. One function, so writer and resolver cannot drift."""
    return f"ladder_{index}.csv"


def coverage_name(index: int) -> str:
    """The coverage report committed beside ladder ``index``."""
    return f"ladder_{index}_coverage.json"


# --- The walk -----------------------------------------------------------------


@dataclass(frozen=True)
class Rung:
    """One rung: a prefix of a walk, and which cells it holds whole and in part."""

    name: str
    tile_id: tuple[str, ...]
    stratum_id: tuple[int, ...]
    #: Cells the rung holds every tile of, in walk order.
    whole_cells: tuple[str, ...]
    #: The one cell it holds only some tiles of, or ``None`` when it ends on a cell boundary.
    partial_cell: str | None
    partial_tiles: int = 0
    partial_of: int = 0


@dataclass(frozen=True)
class Ladder:
    """One ladder: the walk, the seed it was walked under, and the rung budgets cut from it."""

    index: int
    draw_seed: int
    #: The walk: every head-train tile, in walk order.
    tile_id: tuple[str, ...]
    stratum_id: tuple[int, ...]
    geocell_id: tuple[str, ...]
    #: One per rung in :data:`benchmark.landform_config.RUNGS` order, the last the whole walk.
    budgets: tuple[int, ...]

    def rung_at(self, k: int) -> Rung:
        """Rung ``k``: the first ``budgets[k]`` tiles of the walk."""
        n = self.budgets[k]
        held = Counter(self.geocell_id[:n])
        size = Counter(self.geocell_id)
        cells = list(dict.fromkeys(self.geocell_id[:n]))
        partial = [cell for cell in cells if held[cell] < size[cell]]
        return Rung(
            name=RUNGS[k],
            tile_id=self.tile_id[:n],
            stratum_id=self.stratum_id[:n],
            whole_cells=tuple(cell for cell in cells if cell not in partial),
            partial_cell=partial[0] if partial else None,
            partial_tiles=held[partial[0]] if partial else 0,
            partial_of=size[partial[0]] if partial else 0,
        )

    @property
    def rungs(self) -> tuple[Rung, ...]:
        return tuple(self.rung_at(k) for k in range(len(self.budgets)))


def walk(
    tile_id: Sequence[str],
    stratum_id: Sequence[int],
    geocell_id: Sequence[str],
    *,
    index: int,
    draw_seed: int,
    budgets: Sequence[int],
) -> Ladder:
    """Walk the head-train list into ladder ``index``, and check its rungs before returning it.

    The input order does not matter: the walk is a function of the ids and the seed alone.
    """
    from dataset.splits import tie_rank

    if not len(tile_id) == len(stratum_id) == len(geocell_id):
        raise ValueError(
            f"{len(tile_id)} tiles carry {len(stratum_id)} strata and {len(geocell_id)} geocells"
        )
    if len(set(tile_id)) != len(tile_id):
        raise ValueError("the head-train list names a tile twice; a ladder walks a set")
    budgets = tuple(int(b) for b in budgets)
    if budgets[-1] != len(tile_id):
        raise ValueError(
            f"the top rung's budget is {budgets[-1]:,} tiles and the whole head-train list is "
            f"{len(tile_id):,}. §3.3: at 100% every ladder is the whole list, so a list of any "
            f"other length is not the one the rungs were set against"
        )

    by_cell: dict[str, list[int]] = {}
    for position, cell in enumerate(geocell_id):
        by_cell.setdefault(str(cell), []).append(position)
    cells = np.array(sorted(by_cell))
    order: list[int] = []
    for c in np.argsort(tie_rank(cells, draw_seed), kind="stable"):
        members = by_cell[str(cells[c])]
        ids = np.array([str(tile_id[m]) for m in members])
        order += [members[k] for k in np.argsort(tie_rank(ids, draw_seed), kind="stable")]

    ladder = Ladder(
        index=index,
        draw_seed=draw_seed,
        tile_id=tuple(str(tile_id[m]) for m in order),
        stratum_id=tuple(int(stratum_id[m]) for m in order),
        geocell_id=tuple(str(geocell_id[m]) for m in order),
        budgets=budgets,
    )
    check(ladder)
    return ladder


def check(ladder: Ladder) -> None:
    """Raise unless every rung is §3.3's: a prefix of the walk, exactly its budget, at most one
    partial cell — and so nested in the next.

    Every one of these holds by construction today. That is the reason to assert them: the
    next edit to :func:`walk` or :meth:`Ladder.rung_at` is the one that would break them, and
    the tables would print normally.
    """
    rungs = ladder.rungs
    for rung, budget in zip(rungs, ladder.budgets):
        where = f"ladder {ladder.index}'s {rung.name} rung"
        if rung.tile_id != ladder.tile_id[: len(rung.tile_id)]:
            raise AssertionError(
                f"{where} is not a prefix of the walk, so the rungs are not nested and the "
                f"rung is not §3.3's cells-then-tiles order"
            )
        if len(rung.tile_id) != budget:
            raise AssertionError(f"{where} holds {len(rung.tile_id):,} tiles, not its {budget:,}")
        partial = [cell for cell in set(ladder.geocell_id[:budget]) if cell not in rung.whole_cells]
        if len(partial) > 1:
            raise AssertionError(f"{where} holds {len(partial)} partial cells, not at most one")
    for smaller, larger in zip(rungs, rungs[1:]):
        stray = set(smaller.tile_id) - set(larger.tile_id)
        if stray:
            raise AssertionError(
                f"ladder {ladder.index}'s {smaller.name} rung is not nested in its "
                f"{larger.name} rung ({len(stray)} tiles, first {sorted(stray)[:3]})"
            )


def render(ladder: Ladder) -> bytes:
    """The committed walk: the family's two columns, in walk order, LF-terminated."""
    return render_tile_strata(zip(ladder.tile_id, ladder.stratum_id)).encode()


# --- The coverage report ------------------------------------------------------


def coverage(
    ladder: Ladder,
    *,
    slope: Mapping[str, float],
    bands,
    ladder_md5: str,
    head_train_md5: str,
) -> dict:
    """What region luck each rung carries: its cells, strata and ruggedness bands (story 30).

    ``bands`` is :class:`benchmark.voids.Bands`, cut over the eligible test pool and not refit
    on train, so a band here means what it means in every Task 1 and Task 2 table.
    """
    rungs = []
    for rung in ladder.rungs:
        per_band = Counter(bands.label(bands.of(slope[t])) for t in rung.tile_id)
        strata = sorted(set(rung.stratum_id))
        rungs.append(
            {
                "rung": rung.name,
                "tiles": len(rung.tile_id),
                "whole_cells": list(rung.whole_cells),
                "partial_cell": None
                if rung.partial_cell is None
                else {
                    "geocell_id": rung.partial_cell,
                    "tiles": rung.partial_tiles,
                    "of": rung.partial_of,
                },
                "strata": {"present": len(strata), "ids": strata},
                "bands": {
                    "present": len(per_band),
                    "tiles": {bands.label(b): per_band.get(bands.label(b), 0)
                              for b in range(bands.count)},
                },
                "class_pixels": None,
            }
        )
    return {
        "ladder": ladder.index,
        "draw_seed": ladder.draw_seed,
        "ladder_md5": ladder_md5,
        "head_train_md5": head_train_md5,
        "order": ORDER,
        "strata_k": STRATUM_COUNT,
        "bands": {
            "edges": list(bands.edges),
            "cut_over": "slope_mean quartiles of the eligible test pool (benchmark.voids.Bands)",
            "pool_n": bands.pool_n,
        },
        "class_pixels_note": CLASS_PIXELS_NOTE,
        "rungs": rungs,
    }


def _json(body: dict) -> bytes:
    return (json.dumps(body, sort_keys=True, indent=2) + "\n").encode()


def summary(rung: Mapping) -> str:
    """One rung's coverage, as ``freeze`` prints it."""
    partial = rung["partial_cell"]
    part = (
        "no partial cell"
        if partial is None
        else f"{partial['tiles']} of {partial['of']} tiles of {partial['geocell_id']}"
    )
    return (
        f"{rung['rung']:>4} {rung['tiles']:>6,} tiles = {len(rung['whole_cells'])} whole cells "
        f"+ {part}; strata {rung['strata']['present']}/{STRATUM_COUNT}; bands "
        f"{rung['bands']['present']}/{len(rung['bands']['tiles'])} "
        f"({', '.join(f'{b} {n}' for b, n in rung['bands']['tiles'].items())})"
    )


# --- freeze -------------------------------------------------------------------


def head_train_path(cfg: LandformConfig) -> Path:
    """Task 1's committed 20,000-tile head-train list, which every ladder walks unchanged."""
    return Path(cfg.paths.task1_artifacts_dir) / artifacts.TILE_LIST_NAMES["20k"]


def _read_head_train(cfg: LandformConfig) -> tuple[list[str], list[int], list[str], dict]:
    """The head-train list, joined to the manifest for each tile's geocell and slope.

    The list was containment-checked when Task 1 drew it (:func:`benchmark.tiles._pool`); what
    is re-checked here is what a hand edit since could break: every tile still a train tile of
    this corpus, under the stratum it was committed with.
    """
    from .voids import read_manifest_columns

    path = head_train_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: Task 1's head-train list is not frozen, and every ladder walks it "
            f"(`python -m benchmark.launch freeze`)"
        )
    rows = read_tile_strata(path, what="head-train list")
    cols = read_manifest_columns(
        cfg.paths.corpus_dir, ["tile_id", "geocell_id", "stratum_id", "slope_mean", "split"]
    )
    at = {t: k for k, t in enumerate(cols["tile_id"].astype(str))}
    tile_id, stratum, geocell, slope = [], [], [], {}
    for t, s in rows:
        k = at.get(t)
        if k is None or cols["split"][k] != TRAIN_SPLIT:
            where = "absent from the manifest" if k is None else f"in {cols['split'][k]!r}"
            raise ValueError(
                f"{path} names {t}, which is not a train tile of {cfg.paths.corpus_dir} "
                f"({where}). A ladder rung holding it would train on held-out ground"
            )
        if int(cols["stratum_id"][k]) != s:
            raise ValueError(
                f"{path} commits {t} under stratum {s} and the manifest now says "
                f"{int(cols['stratum_id'][k])}: the manifest was re-fit under the list"
            )
        tile_id.append(t)
        stratum.append(s)
        geocell.append(str(cols["geocell_id"][k]))
        slope[t] = float(cols["slope_mean"][k])
    return tile_id, stratum, geocell, slope


def freeze(resolution: LandformResolution) -> list[str]:
    """``freeze --task landform``'s ladder stage: walk, check and commit all three ladders."""
    from .voids import Bands

    cfg = resolution.config
    spec = cfg.ladders
    tile_id, stratum, geocell, slope = _read_head_train(cfg)
    bands = Bands.from_manifest(cfg.paths.corpus_dir, split="test")
    head_train_md5 = artifacts.file_md5(head_train_path(cfg))
    directory = Path(cfg.paths.artifacts_dir)
    lines = []
    for i in range(spec.count):
        ladder = walk(
            tile_id, stratum, geocell, index=i, draw_seed=spec.draw_seed(i), budgets=spec.budgets
        )
        path, md5 = artifacts.write_committed(
            directory / ladder_name(i), lambda ladder=ladder: render(ladder), what="ladder"
        )
        report = coverage(
            ladder, slope=slope, bands=bands, ladder_md5=md5, head_train_md5=head_train_md5
        )
        _, report_md5 = artifacts.write_committed(
            directory / coverage_name(i),
            lambda report=report: _json(report),
            what="ladder coverage report",
        )
        lines += [
            f"{'committed':<16} ladder {i} (draw seed {ladder.draw_seed}) {path}  md5 {md5}",
            f"{'':<16} coverage {directory / coverage_name(i)}  md5 {report_md5}",
            *(f"{'':<16} {summary(rung)}" for rung in report["rungs"]),
        ]
    return lines


# --- The resolver -------------------------------------------------------------


def bind(cfg: LandformConfig) -> int:
    """The ladder this cell trains on, refusing any pairing but seed *i* with ladder *i*.

    Two places a config can name a ladder: the cell's seed, which *is* its ladder, and §4.5's
    tuning pass, which names its seed and its ladder separately. Both must agree with L3.
    """
    schedule = cfg.schedule
    if schedule.tune_seed != schedule.tune_ladder:
        raise ValueError(
            f"schedule.tune_seed {schedule.tune_seed} and schedule.tune_ladder "
            f"{schedule.tune_ladder}: seed {schedule.tune_seed} trains on ladder "
            f"{schedule.tune_seed}, not ladder {schedule.tune_ladder} (ADR-0015 L3). Every arm "
            f"trains on the same three ladders so pairs stay paired"
        )
    for seed in (cfg.seed, schedule.tune_seed):
        if seed >= cfg.ladders.count:
            raise ValueError(
                f"seed {seed} has no ladder: there is no ladder {seed}, and seed i trains on "
                f"ladder i (ADR-0015 L3); freeze writes ladders 0..{cfg.ladders.count - 1}"
            )
    return cfg.ladder


def resolved(
    cfg: LandformConfig, index: int, *, key: str, label: str, verify: bool = True
) -> Resolved:
    """Ladder ``index`` as this box has it, refusing a walk filed under the wrong ladder.

    The check reads the coverage report committed beside the walk: it must be there, name this
    ladder and the draw seed this config gives it, and quote the walk's md5. A copy of another
    ladder over this one fails the md5, a copy of both files fails the index and seed, and a walk
    whose report was deleted is refused rather than trusted.

    ``freeze`` passes ``verify=False``: it re-walks every ladder and
    :func:`benchmark.artifacts.write_committed` refuses any file that differs from the walk, which
    is the stronger check and the one whose message is about freezing.
    """
    directory = Path(cfg.paths.artifacts_dir)
    path = directory / ladder_name(index)
    md5 = artifacts.file_md5(path)
    report_path = directory / coverage_name(index)
    if verify and md5 is not None:
        if not report_path.is_file():
            raise ValueError(
                f"{report_path} is absent, so nothing says {path} (md5 {md5}) is ladder {index}: "
                f"a walk is checked against the coverage report committed beside it, and a seed "
                f"paired with another ladder's walk is refused. Re-freeze rather than copy "
                f"(`python -m benchmark.launch freeze --task landform`)"
            )
        report = json.loads(report_path.read_text())
        seed = cfg.ladders.draw_seed(index)
        if (report.get("ladder"), report.get("draw_seed")) != (index, seed):
            raise ValueError(
                f"{report_path} records ladder {report.get('ladder')} under draw seed "
                f"{report.get('draw_seed')}, and this cell reads ladder {index} under draw seed "
                f"{seed}. Seed {index} trains on ladder {index}; re-freeze rather than rename"
            )
        if report.get("ladder_md5") != md5:
            raise ValueError(
                f"{path} (md5 {md5}) does not hold ladder {index}: its coverage report quotes "
                f"md5 {report.get('ladder_md5')}. A seed paired with another ladder's walk is "
                f"refused; re-freeze rather than copy"
            )
    return Resolved(
        key=key,
        label=label,
        path=str(path),
        md5=md5,
        absent_note="not frozen yet (`python -m benchmark.launch freeze --task landform`)",
    )


def rung_line(cfg: LandformConfig) -> str:
    """Story 5: the ladder id and its tile count, before the run takes a step."""
    i = cfg.ladder
    return (
        f"{'ladder':<16} rung {cfg.rung} = the first {cfg.train_tiles:,} tiles of ladder {i} "
        f"(draw seed {cfg.ladders.draw_seed(i)}), bound to seed {cfg.seed}"
    )
