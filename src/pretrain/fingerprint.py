"""Data fingerprints — what a run records about the data it read (issue #89, plan §6/§15.7).

Every checkpoint and every W&B run carries this block. It answers one question a reader of a
released encoder will actually ask — *which corpus, which selection, which split, which
code?* — with four values that are all **content hashes of committed artifacts**, never
descriptions:

- ``dataset_version`` — ADR-0002's fingerprint over sorted ``(tile_id, content_hash)`` plus
  ``build_params``. It pins the tiles themselves.
- ``selection_list_md5`` — the committed selection list, quoted from the corpus's own
  ``corpus_meta.json`` rather than recomputed. That file *is* the determinism contract
  (ADR-0005 §5), and the value recorded there is the one the corpus was actually built from;
  re-hashing whatever CSV happens to sit at the tier path today would answer a different,
  weaker question.
- ``split_md5`` — the committed ``tile_id,split`` assignment (#77).
- ``git_sha`` — :func:`dataset.provenance.git_sha`, ``-dirty`` and all.

## The split may be absent, and absence is recorded rather than raised

This module ships before the split does, so an unfilled split is a state, not an error. It is
recorded the way the corpus card records an unmeasurable caveat: an explicit null **with a
stated reason**, because an absent entry and a forgotten one look identical otherwise. Two
things have to hold for :data:`COMMITTED` — the assignment artifact exists, and the manifest's
``split`` column is filled on **every** row. A partially filled column is the dangerous shape
(its nulls read as "not yet split" and would quietly rejoin held-out tiles to training), so it
reads as absent and says how many rows are missing.

Nothing here imports :mod:`dataset.splits`. The fingerprint reads artifacts, not the code that
wrote them, which is what lets it work before that code lands and after it changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from dataset import config as dataset_config
from dataset.provenance import git_sha

#: ``split_state``: the assignment is committed *and* the manifest column is filled.
COMMITTED: Final = "committed"
#: ``split_state``: one of those two is missing. ``split_note`` says which.
ABSENT: Final = "absent"


@dataclass(frozen=True)
class DataFingerprint:
    """The block a run records — one corpus, one selection, one split, one commit."""

    corpus_dir: str
    dataset_version: str
    n_tiles: int
    selection_list_md5: str
    split_md5: str | None
    split_blocks_md5: str | None
    split_state: str
    split_note: str | None
    git_sha: str

    def to_dict(self) -> dict:
        """Plain JSON types, for the checkpoint payload and the W&B config."""
        return asdict(self)

    def lines(self) -> list[str]:
        """The block as a run header prints it — see :mod:`pretrain.header`."""
        split = self.split_md5 or "-"
        if self.split_state != COMMITTED:
            split = f"{split}  ({self.split_state}: {self.split_note})"
        return [
            f"corpus           {self.corpus_dir}  ({self.n_tiles:,} tiles)",
            f"dataset_version  {self.dataset_version}",
            f"selection md5    {self.selection_list_md5}",
            f"split md5        {split}",
            f"split blocks md5 {self.split_blocks_md5 or '-'}",
            f"git sha          {self.git_sha}",
        ]


def _md5(path: Path | str | None) -> str | None:
    """md5 of a committed artifact, or ``None`` if it is not there.

    :func:`dataset.materialize.file_md5` is the project's one implementation of "hash a
    committed artifact" — the corpus card, the ledger and the split all pin their inputs with
    it. Imported inside the function so importing a fingerprint never drags in the ETL.
    """
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    from dataset.materialize import file_md5

    return file_md5(path)


def split_state(n_rows: int, n_missing: int) -> tuple[str, str | None]:
    """Is a manifest's ``split`` column filled? The one place that rule and its wording live.

    :mod:`pretrain.data` asks the same question of the same column and must answer it
    identically — one of them records the corpus as unsplit while the other trains on it is
    exactly the disagreement that would make the fingerprint a lie.
    """
    if n_missing == n_rows:
        return ABSENT, f"the manifest's split column is not filled ({n_rows:,} rows)"
    if n_missing:
        return ABSENT, (
            f"the manifest's split column is filled on all but {n_missing:,} of {n_rows:,} rows"
        )
    return COMMITTED, None


def _manifest_split_state(manifest_path: Path) -> tuple[str, str | None]:
    """Read the manifest's ``split`` column alone and put it to :func:`split_state`."""
    import pyarrow.parquet as pq

    column = pq.read_table(manifest_path, columns=["split"])["split"]
    return split_state(len(column), column.null_count)


def collect_fingerprint(
    corpus_dir: Path | str,
    *,
    split_assignment: Path | str | None = dataset_config.SPLIT_ASSIGNMENT_PATH,
    split_blocks: Path | str | None = dataset_config.SPLIT_BLOCKS_PATH,
) -> DataFingerprint:
    """Resolve the fingerprint block for a corpus on disk.

    Reads three files and one git call: ``corpus_meta.json`` for the corpus's own recorded
    fingerprints, the manifest's ``split`` column (that column and nothing else), and the two
    committed split artifacts for their md5s. A corpus with no ``corpus_meta.json`` raises —
    the block is a run's claim about its data, and guessing at it is worse than stopping.
    """
    corpus_dir = Path(corpus_dir)
    meta_path = corpus_dir / "corpus_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} is missing: {corpus_dir} is not a built corpus")
    meta = json.loads(meta_path.read_text())

    state, note = _manifest_split_state(corpus_dir / "manifest.parquet")
    assignment_md5 = _md5(split_assignment)
    if assignment_md5 is None and state == COMMITTED:
        state, note = ABSENT, f"no committed assignment at {split_assignment}"
    elif assignment_md5 is None and split_assignment is not None:
        note = f"{note}; and no committed assignment at {split_assignment}"

    return DataFingerprint(
        corpus_dir=str(corpus_dir),
        dataset_version=meta["dataset_version"],
        n_tiles=int(meta["n_tiles"]),
        selection_list_md5=meta["build_params"]["selection_list_md5"],
        split_md5=assignment_md5,
        split_blocks_md5=_md5(split_blocks),
        split_state=state,
        split_note=note,
        git_sha=git_sha(),
    )


def fingerprint_for(config) -> DataFingerprint:
    """:func:`collect_fingerprint` against a :class:`~pretrain.config.RunConfig`'s data paths."""
    return collect_fingerprint(
        config.data.corpus_dir,
        split_assignment=config.data.split_assignment_path,
        split_blocks=config.data.split_blocks_path,
    )
