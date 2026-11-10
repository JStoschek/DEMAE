"""Task 2's results parquet, its run record, and the write-once rule (#206).

Its own file (``paths.results_path``, ``data/task2/results.parquet``), so Task 1's schema and
its cast-on-read (:mod:`benchmark.results`) are not touched. The run records sit beside it under
Task 1's naming rule (:func:`benchmark.results.run_record_path`), one JSON line per run.

## The rows

One per test tile per cell, keyed by ``(arm, head, rung, seed, ladder, label_set, variant, tile_id)``
plus ``attempt`` (below). Each row holds the tile's 10x10 confusion counts over non-ignore
pixels (``confusion``, truth-major, length 100), boundary-F1's four counts at 2 px, and the
tile's ruggedness band and stratum. mIoU, per-class IoU, macro-F1, accuracy, per-band numbers
and the L2 bootstrap are all sums over these rows; none is stored (story 56).

## Appended once (L20)

A cell's rows are written once and never replaced. A second write of a cell that has rows is
refused, unless the run names an infrastructure failure (crash, OOM, corrupt checkpoint, guard
refusal) as its reason, on the same seed and ladder. Then its rows are appended beside the
first ones under the next ``attempt`` and the reason goes in its run record. Class collapse is
a finished run, and not a reason.

A table scores a cell's attempt 0 (:func:`scored_rows`). A later attempt is a disclosed row
beside it (:func:`disclosed_rows`), never pooled into the core table: L20 says a finished run is
never replaced, and a run that fails before it writes leaves no rows, so its rerun is attempt 0.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from . import results

#: What identifies a cell's rows, before the tile.
#: The head is in it because an arm may train two (B6, and B9 with #211): B6's linear cell and
#: its UperNet cell are different cells (#208).
CELL_COLUMNS: Final = ("arm", "head", "rung", "seed", "ladder", "label_set", "variant")

#: L20's four infrastructure failures, the only reasons a cell may be run again.
RERUN_REASONS: Final = ("crash", "oom", "corrupt-checkpoint", "guard-refusal")

BOUNDARY_COLUMNS: Final = (
    "bf1_pred_boundary", "bf1_pred_matched", "bf1_true_boundary", "bf1_true_matched",
)


def schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("arm", pa.string()),
            ("head", pa.string()),
            ("rung", pa.string()),
            ("seed", pa.int64()),
            ("ladder", pa.int64()),
            ("label_set", pa.string()),
            ("variant", pa.string()),
            ("attempt", pa.int64()),
            ("tile_id", pa.string()),
            ("band", pa.int64()),
            ("stratum", pa.int64()),
            ("confusion", pa.list_(pa.int64())),
            *((name, pa.int64()) for name in BOUNDARY_COLUMNS),
            ("run_id", pa.string()),
            ("config_hash", pa.string()),
        ]
    )


def _existing(path: Path):
    import pyarrow.parquet as pq

    if not path.is_file():
        return None
    table = pq.read_table(path)
    if table.schema.names != schema().names:
        raise ValueError(
            f"{path} has columns {table.schema.names}, not Task 2's {schema().names}: it is "
            f"another file, perhaps Task 1's. Task 2 rows go to their own parquet"
        )
    return table.cast(schema())


def attempt_for(path: Path | str, cell: Mapping[str, Any], rerun_reason: str | None) -> int:
    """The attempt this run's rows go under, or the refusal L20 makes.

    A cell is ``(arm, head, rung, seed, label_set, variant)`` with its ladder: rows for the same
    cell under another ladder are a seed paired with another ladder, refused whatever the reason.
    """
    if rerun_reason is not None and rerun_reason not in RERUN_REASONS:
        raise ValueError(f"rerun reason {rerun_reason!r} is not one of L20's {RERUN_REASONS}")
    table = _existing(Path(path))
    if table is None:
        return 0
    same = [
        row for row in table.select([*CELL_COLUMNS, "attempt", "run_id"]).to_pylist()
        if all(row[k] == cell[k] for k in CELL_COLUMNS if k != "ladder")
    ]
    # The config ties ladder to seed, so a command-built cell cannot reach this; it guards a
    # hand-built cell (or a hand-edited table) pairing a seed with another ladder.
    ladders = {row["ladder"] for row in same} - {cell["ladder"]}
    if ladders:
        raise ValueError(
            f"{path} holds rows for this cell on ladder(s) {sorted(ladders)}, and this run is "
            f"seed {cell['seed']} on ladder {cell['ladder']}: a rerun keeps the same seed and "
            f"ladder (L20)"
        )
    if not same:
        return 0
    name = ", ".join(f"{k} {cell[k]}" for k in CELL_COLUMNS)
    if rerun_reason is None:
        raise ValueError(
            f"this cell is finished: {path} already holds its rows ({name}; run "
            f"{same[0]['run_id']}). A finished run is never replaced (L20); a rerun names an "
            f"infrastructure failure with --rerun-reason {{{','.join(RERUN_REASONS)}}}"
        )
    return 1 + max(row["attempt"] for row in same)


def append(rows: Sequence[Mapping[str, Any]], path: Path | str) -> tuple[Path, int]:
    """Append ``rows``; nothing already in the file is dropped or changed."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    fresh = pa.Table.from_pylist(list(rows), schema=schema())
    existing = _existing(path)
    if existing is not None:
        fresh = pa.concat_tables([existing, fresh])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(fresh, tmp, compression="zstd")
    tmp.replace(path)
    return path, fresh.num_rows


def scored_rows(path: Path | str):
    """The rows a table scores: each cell's attempt 0 (L20; plan §4.4)."""
    import pyarrow.compute as pc

    table = _read(path)
    return table.filter(pc.equal(table["attempt"], 0))


def disclosed_rows(path: Path | str):
    """Every later attempt: printed beside its cell's scored row, never pooled with it."""
    import pyarrow.compute as pc

    table = _read(path)
    return table.filter(pc.greater(table["attempt"], 0))


def _read(path: Path | str):
    table = _existing(Path(path))
    if table is None:
        raise FileNotFoundError(f"no Task 2 results at {path}")
    return table


def run_id(record: Mapping[str, Any]) -> str:
    """A name for this run: the cell, the config and every input it read, plus the attempt."""
    payload = json.dumps(
        {k: record[k] for k in ("cell", "config_hash", "git_sha", "inputs", "recipe_line")},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def write(rows: Sequence[Mapping[str, Any]], record: dict, path: Path | str,
          *, rerun_reason: str | None) -> tuple[Path, int, dict]:
    """Land one cell: the write-once check again (the file may have moved since pre-flight),
    the rows, then the record. Returns the record with ``attempt`` and ``run_id`` filled in, as
    a new dict: the caller's is left as it was."""
    path = Path(path)
    attempt = attempt_for(path, record["cell"], rerun_reason)
    record = {
        **record,
        "cell": {**record["cell"], "attempt": attempt},
        **({"rerun": {"reason": rerun_reason, "attempt": attempt}}
           if rerun_reason is not None else {}),
    }
    record["run_id"] = run_id(record)
    record["written_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["produced"] = {"rows": len(rows), **record.get("produced", {})}
    stamped = [{**row, "attempt": attempt, "run_id": record["run_id"]} for row in rows]
    path, total = append(stamped, path)
    results.append_run_record(record, results.run_record_path(path))
    return path, total, record
