"""Split-aware reading — and the assertion that no test geocell reaches a train batch (#89).

Plan §15.2's pre-flight gate reads: *"Split column filled and committed; ... no test geocell in
any train batch (asserted in the Dataset)."* This module is where that assertion attaches to
the read path: :class:`SplitDataset` cannot be constructed without it having run.

The refusals themselves live in :mod:`dataset.containment`, which is where they moved in #149
when Task 1's ``freeze`` became their second caller — that path is CPU-only and torch-free by
construction (§3.4), and this module imports ``torch.utils.data`` at module scope. Every name
is re-exported here, so #89's callers are unchanged and ``pretrain.data.read_blocks`` still
resolves. What each check is *for* — the two block lists, why neither subsumes the other, and
why a leak is unrepairable rather than merely wrong — is that module's docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from dataset import config as dataset_config
from dataset.containment import (
    SPLIT_COLUMNS,
    SplitBlocks,
    assert_assignment_agrees,
    assert_contained,
    contained_indices,
    derive_blocks,
    load_blocks,
    read_blocks,
)
from dataset.loader import TileDataset

from .fingerprint import COMMITTED, split_state

__all__ = [
    "SPLIT_COLUMNS",
    "SplitBlocks",
    "SplitDataset",
    "assert_assignment_agrees",
    "assert_contained",
    "contained_indices",
    "derive_blocks",
    "load_blocks",
    "read_blocks",
]


class SplitDataset(Subset):
    """One split of a corpus, with containment asserted **by construction**.

    A ``Subset`` and not a new Dataset: ADR-0003's read path is consumed unchanged (lazy
    per-worker memmaps, the sample dict, the normalizer), and a split is a choice of *which
    rows*, which is exactly what a Subset expresses. :func:`dataset.loader.build_dataloader`
    wraps it like any other.

    It builds itself from the corpus directory rather than accepting a dataset and a set of
    indices, and that is the whole design: there is no way to obtain one of these without the
    check having run. Plan §15.2 asks for the assertion *in the Dataset*, and an assertion a
    caller can route around is not one.

    Five refusals, all at construction: an unfilled or partly filled ``split`` column, a split
    label no tile carries, a block list that does not belong to this corpus, a manifest column
    that disagrees with the committed assignment, and any tile of this split sitting inside a
    block held out for another one. The partial-fill case is refused rather than filtered
    because its nulls are ambiguous in the worst direction — they read as "not yet split"
    while the rows they mark are perfectly trainable, so a run would quietly train on tiles
    nothing decided the fate of.

    :func:`benchmark.tiles.draw` makes the same five, deliberately and in the same order (#149):
    "asserted exactly as a Phase-1 Dataset asserts it" is the requirement a benchmark drawing
    its test set out of this corpus has to meet, and four out of five would meet it silently.
    """

    def __init__(
        self,
        corpus_dir: Path | str,
        split: str = "train",
        *,
        blocks_path: Path | str | None = dataset_config.SPLIT_BLOCKS_PATH,
        assignment_path: Path | str | None = dataset_config.SPLIT_ASSIGNMENT_PATH,
        normalization: str = "per_tile",
        channels: tuple[str, ...] = dataset_config.INPUT_CHANNELS,
    ) -> None:
        import pyarrow.parquet as pq

        corpus_dir = Path(corpus_dir)
        table = pq.read_table(corpus_dir / "manifest.parquet", columns=list(SPLIT_COLUMNS))
        cols = {c: table[c].to_numpy(zero_copy_only=False) for c in SPLIT_COLUMNS}
        labels = cols["split"]

        state, note = split_state(table.num_rows, table["split"].null_count)
        if state != COMMITTED:
            raise ValueError(
                f"{corpus_dir}/manifest.parquet: {note}. Draw and commit the split first "
                f"(#77: `python -m dataset.splits --fill`); training on an unsplit corpus is "
                f"what plan §15.2 exists to prevent"
            )

        if assignment_path is not None and Path(assignment_path).exists():
            assert_assignment_agrees(assignment_path, tile_id=cols["tile_id"], labels=labels)

        blocks = load_blocks(
            blocks_path,
            geocell_id=cols["geocell_id"],
            source_cog=cols["source_cog"],
            labels=labels,
        )
        indices = contained_indices(split, cols=cols, blocks=blocks)

        super().__init__(
            TileDataset(corpus_dir, normalization=normalization, channels=channels),
            indices.tolist(),
        )
        self.split = split
        self.blocks = blocks

    def __repr__(self) -> str:
        return f"SplitDataset(split={self.split!r}, n={len(self)}, of={len(self.dataset)})"
