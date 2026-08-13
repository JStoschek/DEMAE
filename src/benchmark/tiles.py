"""§3.1's four committed tile draws, and the leak they are checked against (#149).

Four draws in **five files**: §3.1's head-train, val and test draws, plus the label-efficiency
one, whose top rung *is* the head-train list and whose 1k and 5k rungs are their own files. The
benchmark thereby joins the contract family the selection list, the split assignment, the val
panel and the probe set are in. The family's rule is one sentence: **the committed file is what
later phases re-read**, not this code re-run (ADR-0005 §5). So the format is
:mod:`pretrain.artifact`'s two columns, the order is sorted, and a redraw is byte-identical —
an md5 the run record can quote is the whole point.

## What the draw is, and what it is not

It is an apportionment, not a sampler. Every choice this module makes is either a quota from
:func:`dataset.sampling.band_quotas` — the project's one largest-remainder apportionment — or a
position in :func:`dataset.splits.tie_rank`'s keyed-BLAKE2b order. There is no ``Generator``
anywhere in it, on the house rule :mod:`dataset.sampling` states: NumPy guarantees its
bit-generator streams across releases but not the distributions layered on them. ``draw_seed``
is what the *file* records having been drawn under, and the file is the contract.

Three properties, each of which is a way a benchmark number stops meaning what it says:

- **Proportional over the k=20 strata.** A head-train set that over-weighted a stratum trains
  the head on a terrain mix the corpus does not have, and no column of the results table would
  say so. All four draws are stratified for the same reason, including val — §3.1 only asks it
  of head-train and test, but val is what early stopping and the head LR pick read, and picking
  a stopping point against a differently-composed population is the same error one step earlier.
- **Nested: 1k ⊂ 5k ⊂ 20k.** So C3's label-efficiency sweep measures data volume rather than
  draw luck. Nesting *falls out* of taking prefixes of one rank order, and is nonetheless
  **asserted** (:func:`assert_nested`) rather than assumed: largest-remainder apportionment is
  not monotone in the house size — the Alabama paradox is exactly a stratum losing a seat as the
  budget grows — so the property belongs to the numbers, not to the construction.
- **Contained.** No list may name a tile sitting in a block held out for another split. This is
  :mod:`dataset.containment`'s check, unchanged, against the same union of two block lists a
  Phase-1 Dataset is built through (#89): the blocks derived from the manifest's ``split``
  column, and #77's committed block lists, because neither subsumes the other. It fails loudly.
  A benchmark scoring on ground the encoder pretrained on prints a perfectly normal table, and
  is unrepairable without re-pretraining.

A draw runs :class:`~pretrain.data.SplitDataset`'s whole refusal set and not merely that one,
because "exactly as a Phase-1 Dataset asserts it" is the requirement: a partly filled ``split``
column, a split label no tile carries, a block list belonging to another corpus, a manifest
column that disagrees with the committed assignment, and containment itself. The blocks defend
the *regions*; the assignment check defends the *claim* a table makes about which split it was
scored against.

## Two places this deliberately differs from the probe set

:func:`pretrain.probe.draw_probe_set` is the nearest prior art — the apportionment is literally
the same function — with two changes that are both about what a *benchmark* is:

- **A thin pool is refused, not accepted short.** The probe returns fewer than asked when a
  debug tier has fewer val tiles, so the probe stays exercisable where it is cheapest to run. A
  "20,000-tile head-train set" holding 12,000 is a different benchmark reported under the same
  name, and every label-efficiency point would be off its own axis.
- **Containment is asserted here rather than downstream.** The probe leans on
  :class:`~pretrain.data.SplitDataset` for it. `freeze` is torch-free and never constructs one,
  so the assertion attaches at the draw — which is also the last moment before the leak becomes
  a committed file.

The corpus is read for five columns and no pixel; a whole `freeze` of the tile lists is about a
second on the shipped 500k tier.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from dataset import containment
from dataset.sampling import stratified_draw
from pretrain.artifact import write_tile_strata
from pretrain.fingerprint import COMMITTED, split_state

from .artifacts import TEST_TILES_NAME, TILE_LIST_NAMES, VAL_TILES_NAME, file_md5
from .config import LABEL_EFFICIENCY_SIZES, SIZE_TILES, BenchmarkConfig, Tiles

#: The manifest columns a draw reads. Containment's four, plus the stratum it apportions over.
DRAW_COLUMNS: Final = (*containment.SPLIT_COLUMNS, "stratum_id")


@dataclass(frozen=True)
class TileList:
    """One committed list: what it is called, where it was drawn from, and its rows.

    ``stratum_id`` rides beside ``tile_id`` for the reason the probe set carries it — it is what
    lets a later phase *catch* a manifest re-fit underneath a committed list, rather than
    silently comparing arms under two different label vectors.
    """

    #: The committed file's name, so a list knows what it is called wherever it is passed.
    name: str
    #: The split it was drawn from — ``train``, ``val`` or ``test``.
    split: str
    tile_id: tuple[str, ...]
    stratum_id: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.tile_id) != len(self.stratum_id):
            raise ValueError(f"{len(self.tile_id)} tiles carry {len(self.stratum_id)} strata")
        if len(set(self.tile_id)) != len(self.tile_id):
            raise ValueError(f"{self.name} names a tile twice; it is a set, not a sample")

    def __len__(self) -> int:
        return len(self.tile_id)

    @property
    def rows(self) -> list[tuple[str, int]]:
        """The committed rows, sorted by ``tile_id`` — the order the file is written in."""
        return sorted(zip(self.tile_id, self.stratum_id))


@dataclass(frozen=True)
class TileDraw:
    """Every list one `freeze` commits, drawn together so the nesting is one draw's property."""

    #: The head-train ladder, keyed by :data:`benchmark.config.LABEL_EFFICIENCY_SIZES`.
    head_train: Mapping[str, TileList]
    val: TileList
    test: TileList

    @property
    def lists(self) -> tuple[TileList, ...]:
        """All five, smallest ladder rung first, then val and test — the write order."""
        return (*(self.head_train[size] for size in LABEL_EFFICIENCY_SIZES), self.val, self.test)


# --- The apportionment --------------------------------------------------------


def _apportion(stratum: np.ndarray, rank: np.ndarray, size: int) -> np.ndarray:
    """:func:`dataset.sampling.stratified_draw`, refusing the short answer it may return.

    Shared with the probe set down to the apportionment; what differs is what the two do with a
    frame that cannot supply the budget. Here the pool has already been checked against it
    (:func:`_pool`), so a short draw means the quota logic moved underneath this module rather
    than that the corpus is thin — which is worth a message rather than a silently smaller list.
    """
    picked = stratified_draw(stratum, rank, size)
    if picked.size != size:
        raise ValueError(f"apportioned {picked.size:,} tiles of the {size:,} asked for")
    return picked


def assert_nested(ladder: Iterable[TileList]) -> None:
    """Raise unless each rung of the label-efficiency ladder is a subset of the next.

    Asserted rather than assumed — see the module docstring on the Alabama paradox. The message
    names the two rungs and one tile that broke it, which is enough to tell a quota that moved
    from a draw that was assembled from two different orders.
    """
    rungs = list(ladder)
    for smaller, larger in zip(rungs, rungs[1:]):
        stray = sorted(set(smaller.tile_id) - set(larger.tile_id))
        if stray:
            raise ValueError(
                f"§3.1's label-efficiency subsets are not nested: {len(stray):,} of "
                f"{len(smaller):,} tiles in {smaller.name} are absent from {larger.name} "
                f"(first {stray[:containment.MAX_NAMED]}). C3's sweep would then measure draw "
                f"luck rather than data volume, which is what the nesting exists to rule out"
            )


def _check_ladder(head_train_sizes: Mapping[str, int], head_train: int) -> tuple[int, ...]:
    """Refuse a ladder that is not §3.1's shape, before the corpus is opened."""
    if tuple(head_train_sizes) != LABEL_EFFICIENCY_SIZES:
        raise ValueError(
            f"the label-efficiency ladder is {list(head_train_sizes)}; §3.1's rungs are "
            f"{list(LABEL_EFFICIENCY_SIZES)}, smallest first"
        )
    sizes = tuple(head_train_sizes[size] for size in LABEL_EFFICIENCY_SIZES)
    if not all(a < b for a, b in zip(sizes, sizes[1:])) or sizes[-1] != head_train:
        raise ValueError(
            f"the label-efficiency ladder {sizes} must ascend to tiles.head_train "
            f"({head_train:,}): §3.1's sweep is nested inside the one committed draw"
        )
    return sizes


# --- The draw -----------------------------------------------------------------


def _pool(
    cols: Mapping[str, np.ndarray],
    split: str,
    blocks: containment.SplitBlocks,
    budget: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One split's ``(tile_id, stratum_id, rank)``, containment-checked against its budget.

    Containment is checked over the **whole** split rather than over the tiles drawn from it,
    exactly as :class:`~pretrain.data.SplitDataset` checks it: a pool holding a held-out tile is
    a broken corpus whether or not this particular draw happened to reach that tile.

    The rank is :func:`dataset.splits.tie_rank`'s keyed-BLAKE2b order over the pool's tile ids,
    computed once per pool because the ladder's three rungs must walk the *same* queue.
    """
    from dataset.splits import tie_rank

    indices = containment.contained_indices(split, cols=cols, blocks=blocks)
    if indices.size < budget:
        raise ValueError(
            f"the {split} pool holds {indices.size:,} tiles and §3.1 draws {budget:,} from it. "
            f"A list short of its budget is a different benchmark reported under the same name, "
            f"so it is refused rather than drawn shallow"
        )
    tile_id = cols["tile_id"][indices].astype(str)
    return tile_id, cols["stratum_id"][indices].astype(np.int64), tie_rank(tile_id, seed)


def draw(
    manifest_path: Path | str,
    tiles: Tiles,
    *,
    blocks_path: Path | str | None,
    assignment_path: Path | str | None = None,
    head_train_sizes: Mapping[str, int] = SIZE_TILES,
) -> TileDraw:
    """Draw §3.1's four draws — five lists — off one corpus, refusing an uncontained split.

    Run once per tier and its output committed. Reads :data:`DRAW_COLUMNS` and no pixel.

    ``blocks_path`` is #77's committed block list, or ``None`` when a corpus has none — the
    derived blocks then stand alone, which is weaker in exactly one way (a wholesale
    relabelling stops being visible) and is why every run records the artifact's md5.
    ``assignment_path`` is #77's committed ``tile_id,split`` CSV; absent or unset, it is skipped
    on the same terms :class:`~pretrain.data.SplitDataset` skips it.
    """
    import pyarrow.parquet as pq

    sizes = _check_ladder(head_train_sizes, tiles.head_train)

    table = pq.read_table(Path(manifest_path), columns=list(DRAW_COLUMNS))
    # `split_state` is where that rule and its wording live — one of the two answering "is this
    # corpus split?" differently from the other is precisely the disagreement it exists to stop.
    state, note = split_state(table.num_rows, table["split"].null_count)
    if state != COMMITTED:
        raise ValueError(
            f"{manifest_path}: {note}. Its nulls read as 'not yet split' while the rows they "
            f"mark are perfectly drawable, so a draw would sample tiles nothing decided the fate "
            f"of — commit the split first (#77: `python -m dataset.splits --fill`)"
        )
    cols = {c: table[c].to_numpy(zero_copy_only=False) for c in DRAW_COLUMNS}

    if assignment_path is not None and Path(assignment_path).exists():
        containment.assert_assignment_agrees(
            assignment_path, tile_id=cols["tile_id"], labels=cols["split"]
        )
    blocks = containment.load_blocks(
        blocks_path,
        geocell_id=cols["geocell_id"],
        source_cog=cols["source_cog"],
        labels=cols["split"],
    )

    def committed(name: str, split: str, budget: int) -> TileList:
        """One standalone list — val and test, each drawn out of its own pool."""
        tile_id, stratum, rank = _pool(cols, split, blocks, budget, tiles.draw_seed)
        return _list(name, split, tile_id, stratum, _apportion(stratum, rank, budget))

    # The head-train ladder is drawn first, so a leak in the train pool is the message a broken
    # corpus produces whatever else is wrong with it. All three rungs take prefixes of **one**
    # rank order over **one** pool, which is where the nesting comes from.
    train_id, train_stratum, rank = _pool(
        cols, tiles.train_split, blocks, tiles.head_train, tiles.draw_seed
    )
    head_train = {
        size: _list(
            TILE_LIST_NAMES[size],
            tiles.train_split,
            train_id,
            train_stratum,
            _apportion(train_stratum, rank, budget),
        )
        for size, budget in zip(LABEL_EFFICIENCY_SIZES, sizes)
    }
    assert_nested(head_train[size] for size in LABEL_EFFICIENCY_SIZES)

    return TileDraw(
        head_train=head_train,
        val=committed(VAL_TILES_NAME, tiles.val_split, tiles.val),
        test=committed(TEST_TILES_NAME, tiles.test_split, tiles.test),
    )


def _list(
    name: str, split: str, tile_id: np.ndarray, stratum: np.ndarray, picked: np.ndarray
) -> TileList:
    """A committed list from a pool and the positions apportioned out of it."""
    return TileList(
        name=name,
        split=split,
        tile_id=tuple(tile_id[picked].tolist()),
        stratum_id=tuple(int(s) for s in stratum[picked]),
    )


# --- The committed files ------------------------------------------------------


def write(drawn: TileDraw, directory: Path | str) -> dict[str, str]:
    """Write every list into ``directory`` and return ``{name: md5}``.

    Sorted by ``tile_id`` and LF-terminated, through :func:`pretrain.artifact.write_tile_strata`
    — the family's one writer. Sorted for the reason the probe set is: two runs that agree on
    the set produce a byte-identical file whatever order the rows arrived in, which is what
    makes the md5 a contract rather than a timestamp.

    It **overwrites**, and §3.4's "refuse an artifact whose content hash has moved" is `freeze`'s
    to add over the whole set of committed inputs rather than this function's to invent for one
    of them. Nothing is lost meanwhile: a redraw of the same corpus at the same seed reproduces
    the same bytes, so the only way to overwrite with different content is to have moved a knob.
    """
    directory = Path(directory)
    written = {}
    for lst in drawn.lists:
        path = write_tile_strata(lst.rows, directory / lst.name)
        written[lst.name] = file_md5(path)
    return written


def freeze(cfg: BenchmarkConfig) -> dict[str, str]:
    """`freeze`'s tile stage: draw §3.1's five lists off this box's corpus and commit them."""
    drawn = draw(
        Path(cfg.paths.corpus_dir) / "manifest.parquet",
        cfg.tiles,
        blocks_path=cfg.paths.split_blocks_path,
        assignment_path=cfg.paths.split_assignment_path,
    )
    return write(drawn, cfg.paths.artifacts_dir)


def format_written(written: Mapping[str, str], directory: Path | str) -> list[str]:
    """What `freeze` says it committed — by content, since that is what the run record quotes.

    Its own block rather than a second resolution line: the resolution says what was *there*
    before the command took a step, and this says what is there now. Printing them in one
    layout would invite reading a written md5 as a resolved one.
    """
    return [
        f"{'committed':<16} {len(written)} tile lists into {directory}",
        *(f"{'':<16} {name:<21} md5 {md5}" for name, md5 in written.items()),
    ]
