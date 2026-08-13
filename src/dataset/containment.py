"""Split containment — the refusals every phase that *reads* the split shares (#89, #149).

:mod:`dataset.splits` draws the split and commits it; this is the code that reads it back and
refuses it when it no longer means what it says. The two halves live in one package because
the artifacts are one artifact family: :func:`read_blocks` is the reader for the JSON
:func:`dataset.splits.write_blocks` writes, and :func:`assert_assignment_agrees` reads the CSV
:func:`dataset.splits.write_assignment` writes, on the rule :mod:`pretrain.artifact` states for
its own two columns — *the format is known in one place*.

It is worth being blunt about why it fails hard. A test geocell that leaks into training does
not show up in the loss, the reconstruction grids, or the kNN probe; the run looks perfect.
It surfaces in Phase 2, as a downstream evaluation on regions the encoder has already seen,
which invalidates the headline claim after the fact and cannot be repaired without re-running
pretraining. So: refused at construction, with a ``ValueError``, never a warning.

## The two block lists, and why both are consulted

A split is contaminated in two distinguishable ways, and neither check subsumes the other:

- **Derived blocks** — the geocells that hold a ``test`` tile and the COGs that hold a ``val``
  tile, read off the manifest's own column. This catches the common accident: a relabelled
  tile inside a region the rest of which is still held out.
- **Committed blocks** — :data:`dataset.config.SPLIT_BLOCKS_PATH`, the artifact #77 commits
  naming what was held out *as a block*. This catches what the column alone cannot: a whole
  test geocell relabelled ``train``, where nothing left in the manifest looks held out.

The union of the two is what a split is checked against. The artifact is also checked against
the corpus — a block list naming cells this corpus does not hold is a stale or mismatched
artifact, and it would otherwise pass while protecting nothing. When the artifact is absent
the derived lists stand alone, which is weaker and is why every run records the artifact's md5.

On top of the blocks, the manifest column is checked against the committed ``tile_id,split``
CSV itself (:func:`assert_assignment_agrees`). The blocks defend the regions; that check
defends the claim — a Phase-1 run records the assignment's md5 as the split it trained under,
and a column edited out from under that file would make the record a lie.

## Why this is here rather than in :mod:`pretrain`

It shipped inside ``pretrain.data`` in #89, when the Phase-1 Dataset was the only reader. #149
made Task 1's ``freeze`` the second one, and ``freeze`` is CPU-only and torch-free by
construction (§3.4) while ``pretrain.data`` imports ``torch.utils.data`` at module scope. The
alternative was a second copy of the same refusals in :mod:`benchmark`, and two copies of a
check that fails loudly is one copy that eventually does not. :mod:`pretrain.data` re-exports
every name here, so #89's callers are unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

#: The manifest columns the check reads. Nothing else is touched, and no pixel is read.
SPLIT_COLUMNS: Final = ("tile_id", "geocell_id", "source_cog", "split")

#: How many offending tiles an error message names before it stops listing.
MAX_NAMED: Final = 5


@dataclass(frozen=True)
class SplitBlocks:
    """What is held out, as blocks: whole geocells for ``test``, whole source COGs for ``val``."""

    test_geocells: frozenset[str]
    val_cogs: frozenset[str]
    #: Where these came from, for the error message: an artifact path, or the manifest.
    source: str

    def union(self, other: SplitBlocks) -> SplitBlocks:
        return SplitBlocks(
            test_geocells=self.test_geocells | other.test_geocells,
            val_cogs=self.val_cogs | other.val_cogs,
            source=f"{self.source} + {other.source}",
        )

    def forbidden_for(self, split: str) -> tuple[frozenset[str], frozenset[str]]:
        """The (geocells, COGs) a tile labelled ``split`` may not sit in.

        A tile is contaminated iff it sits in a block held out for a *different* split, so
        each list drops out of its own split's check and applies to every other one.
        """
        return (
            frozenset() if split == "test" else self.test_geocells,
            frozenset() if split == "val" else self.val_cogs,
        )


def _member(values: np.ndarray, allowed: frozenset[str]) -> np.ndarray:
    """Elementwise "is this string in that set", as a bool array.

    A hash lookup per row rather than :func:`numpy.isin`, which on a string (object) column
    sorts both sides: over the 475k-row train split that costs ~2 s, all of it on the startup
    path of every run.
    """
    items = values.tolist()
    return np.fromiter((v in allowed for v in items), dtype=bool, count=len(items))


def derive_blocks(
    geocell_id: np.ndarray, source_cog: np.ndarray, labels: np.ndarray
) -> SplitBlocks:
    """The blocks implied by the manifest column itself."""
    return SplitBlocks(
        test_geocells=frozenset(np.unique(geocell_id[labels == "test"]).tolist()),
        val_cogs=frozenset(np.unique(source_cog[labels == "val"]).tolist()),
        source="the manifest's split column",
    )


def read_blocks(path: Path | str) -> SplitBlocks:
    """Read #77's committed block lists (``split_blocks_*.json``)."""
    path = Path(path)
    doc = json.loads(path.read_text())
    try:
        cells, cogs = doc["test"]["cells"], doc["val"]["cogs"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path} is not a split-blocks artifact (missing {exc})") from exc
    return SplitBlocks(
        test_geocells=frozenset(map(str, cells)),
        val_cogs=frozenset(map(str, cogs)),
        source=str(path),
    )


def load_blocks(
    path: Path | str | None,
    *,
    geocell_id: np.ndarray,
    source_cog: np.ndarray,
    labels: np.ndarray,
) -> SplitBlocks:
    """The derived blocks, unioned with the committed ones when the artifact is there.

    A committed block the corpus does not hold raises: it means the artifact belongs to
    another tier (or another draw), and a hold-out list that matches nothing would pass every
    containment check while protecting nothing at all. Same rule, and the same reason, as
    :func:`dataset.splits.assign_splits` refusing a block absent from the corpus.
    """
    derived = derive_blocks(geocell_id, source_cog, labels)
    if path is None or not Path(path).exists():
        return derived
    committed = read_blocks(path)
    absent_cells = sorted(committed.test_geocells - set(geocell_id.tolist()))
    absent_cogs = sorted(committed.val_cogs - set(source_cog.tolist()))
    if absent_cells or absent_cogs:
        raise ValueError(
            f"{path} holds out blocks this corpus does not contain — a stale or "
            f"cross-tier artifact: geocells {absent_cells[:MAX_NAMED]}, "
            f"COGs {absent_cogs[:MAX_NAMED]} "
            f"({len(absent_cells)} cells and {len(absent_cogs)} COGs unmatched)"
        )
    return derived.union(committed)


def assert_contained(
    split: str,
    *,
    indices: np.ndarray,
    tile_id: np.ndarray,
    geocell_id: np.ndarray,
    source_cog: np.ndarray,
    blocks: SplitBlocks,
) -> None:
    """Raise unless every selected tile sits outside every block held out for another split.

    The message names the count first and then up to :data:`MAX_NAMED` tiles with the block
    each one sits in — enough to tell a one-tile relabelling from a whole region rejoining
    training, which are very different problems.

    Vectorized because it runs at construction over the whole train split (475k rows on the
    shipped tier): a Python loop would put a visible pause in front of every run, and a check
    people are tempted to skip is a check that eventually gets skipped.
    """
    cells, cogs = blocks.forbidden_for(split)
    in_cell = _member(geocell_id[indices], cells)
    in_cog = _member(source_cog[indices], cogs)
    offending = np.flatnonzero(in_cell | in_cog)
    if offending.size == 0:
        return
    named = []
    for j in offending[:MAX_NAMED]:
        i = indices[j]
        named.append(
            f"{tile_id[i]} (geocell {geocell_id[i]}, held out as test)"
            if in_cell[j]
            else f"{tile_id[i]} (COG {source_cog[i]}, held out as val)"
        )
    raise ValueError(
        f"split containment violated: {offending.size} {split} tile(s) sit inside blocks held "
        f"out for another split, per {blocks.source}. Plan §15.2 requires that no test geocell "
        f"is reachable from a train batch. First: " + "; ".join(named)
    )


def contained_indices(
    split: str, *, cols: Mapping[str, np.ndarray], blocks: SplitBlocks
) -> np.ndarray:
    """Rows of ``split``, having asserted that none of them sits in a held-out block.

    The sequence both readers of a split perform — select, refuse an empty selection, assert
    containment — in one place, so a caller cannot obtain the rows without the check having run.
    That is exactly what :class:`~pretrain.data.SplitDataset` is built around ("an assertion a
    caller can route around is not one"), and #149's ``freeze`` needs the same guarantee without
    constructing a Dataset.

    ``cols`` holds :data:`SPLIT_COLUMNS`; extra columns are ignored, so a caller reading the
    stratum beside them passes one dict rather than unpacking four arrays at every call site.
    """
    labels = cols["split"]
    indices = np.flatnonzero(labels == split)
    if indices.size == 0:
        raise ValueError(
            f"no tile carries split {split!r}; the corpus holds "
            f"{sorted(set(map(str, labels.tolist())))}"
        )
    assert_contained(
        split,
        indices=indices,
        tile_id=cols["tile_id"],
        geocell_id=cols["geocell_id"],
        source_cog=cols["source_cog"],
        blocks=blocks,
    )
    return indices


def assert_assignment_agrees(
    path: Path | str, *, tile_id: np.ndarray, labels: np.ndarray
) -> None:
    """Raise unless the manifest's ``split`` column is what the committed assignment says.

    The block lists above defend the *regions*; this defends the claim a run makes about
    itself. A Phase-1 run records the assignment's md5 (:mod:`pretrain.fingerprint`) as the
    split it trained under, and a manifest column edited out from under that file would make
    the record a lie no downstream reader could catch. Cheap — one pass over the committed CSV,
    once, at construction — so it is checked rather than trusted.

    Reads it with :func:`dataset.splits.read_assignment`, the writer's own reader, so the
    format is known in one place. Imported lazily to keep this module's import cost the cost
    of ``numpy`` and nothing else — it is on the startup path of every run.
    """
    from .splits import read_assignment

    committed = read_assignment(path)
    disagree = [
        f"{t}: manifest {label!r} vs assignment {committed.get(t, 'absent')!r}"
        for t, label in zip(tile_id.tolist(), labels.tolist())
        if committed.get(t) != label
    ]
    if disagree:
        raise ValueError(
            f"the manifest's split column disagrees with {path} on {len(disagree):,} of "
            f"{len(labels):,} tiles — the committed assignment is the contract every later "
            f"phase re-reads. First: " + "; ".join(disagree[:MAX_NAMED])
        )
