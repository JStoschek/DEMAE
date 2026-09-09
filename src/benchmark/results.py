"""The results parquet and the run record beside it (#158).

One row per ``(arm, init, size, seed, family, bucket, band, tile_id, instance)``, carrying
**every** metric in the registered set — headline and appendix alike — plus the eroded-mask
slope, the void's area and equivalent diameter, and the mask's RLE hash.

## Computed and printed are different questions

That sentence is the whole design of this file, and it is #136's. Nothing that collapses an axis
for print discards a number here: the size buckets are pooled in the printed table and are a
column here; the appendix metrics are printed nowhere and are columns here; seed spread reaches
the page as one printed maximum and is a row per seed here. What that buys is that
:mod:`benchmark.report` is a **pure function of this file** — re-rendering the table never
re-runs an arm, and a reviewer asking a question the table's shape did not anticipate is asking
it of a file that already holds the answer.

The inverse is what it costs, and it is worth stating: a metric that is not computed here cannot
be recovered without re-running 2.0–5.2 CPU-hours of eval scoring. So the roster is R1's whole
registered set rather than the six that print, read off the bar (:attr:`benchmark.prereg.Bar.
registered_metrics`) rather than listed again here — a second home for the metric names is what
produced [#146](https://github.com/JStoschek/DEMAE/issues/146).

## The `init` column is the encoder, not the name's suffix

§1's table splits A2, A3 and A4 by init, and the registry spells that split into the arm's own
name — ``A4-R5`` and ``A4-R4``. So an ``init`` column looks like it should be the bit after the
hyphen, and it must not be: ``A4-S`` is §2.3's ViT-S appendix cell, whose suffix is a **capacity**
and not an init. Splitting on the name would file it as a third init and put a capacity aside
into an axis three C-claims are read along.

So ``init`` is the arm's **encoder key** — ``demae-r5``, ``demae-r4``, ``demae-vits`` — which the
registry already declares and which is what "which checkpoint was this initialized from" actually
means. ``None`` for the two arms that run no encoder. The registry key stays in ``arm``, because
that is what the CLI takes and what :data:`benchmark.arms.PRINTED_ROWS` makes a printed row.

## Re-running a cell replaces it

``run`` executes exactly one ``(arm, size, seed)`` cell, so that triple is what
:func:`append_rows` replaces. A cell re-run after a bug fix should leave one set of rows behind,
not two — and "two" is the quiet failure, because every aggregate downstream would silently
double-count the tiles the two passes agree on. The run record is append-only against that, so
the *fact* that a cell was scored twice, and under which git SHA and which artifact hashes,
survives the replacement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from .artifacts import Resolution

#: The row key, in the order #158 states it. ``instance`` disambiguates the real family, where
#: ~7 voids can share one host tile (§3.3) and ``(tile_id, family, bucket)`` therefore does not
#: identify a row; it is 0 for every synthetic instance, which have one void per tile per bucket.
KEY_COLUMNS: Final = (
    "arm",
    "init",
    "size",
    "seed",
    "family",
    "bucket",
    "band",
    "tile_id",
    "instance",
)

#: What the *mask* was, beside what the fill scored. Here rather than in
#: :func:`benchmark.metrics.score` because these are properties of the void and that function
#: scores a fill — and in the parquet at all because C2 is a claim about void **size**, so a
#: results file that recorded only the bucket could not answer it at any finer grain than three.
#:
#: ``mask_rle_md5`` is the committed mask's identity, not its bytes: it is what lets a table be
#: proved to have been scored against the eval mask list that was frozen, without the RLE column
#: itself — 22 MB uncompressed — being copied into every arm's rows.
GEOMETRY_COLUMNS: Final = ("void_area_px", "void_eqdiam_px", "mask_rle_md5")

#: The provenance a row carries on its own, so a rows-only reader can still tell one run's rows
#: from another's. Everything else about the run is in the run record, joined on ``run_id``.
PROVENANCE_COLUMNS: Final = ("run_id", "config_hash")

#: The run record's file, beside the parquet. One JSON object per line, appended — see the
#: module docstring on why it is append-only where the parquet is replace-by-cell.
RUN_RECORD_SUFFIX: Final = "-runs.jsonl"


def run_record_path(results_path: Path | str) -> Path:
    """Where the run records for this results file live.

    Derived rather than configured: the two are one artifact in two files, and a box that could
    point them at different directories is a box that can produce a table whose provenance is
    somewhere else.
    """
    results_path = Path(results_path)
    return results_path.with_name(results_path.stem + RUN_RECORD_SUFFIX)


@dataclass(frozen=True)
class ResultRow:
    """One scored instance — the key, the mask's geometry, and every registered metric."""

    arm: str
    init: str | None
    size: str
    seed: int
    family: str
    bucket: str
    band: str
    tile_id: str
    instance: int
    void_area_px: int
    void_eqdiam_px: float
    mask_rle_md5: str
    #: :func:`run_id`'s, so a rows-only reader can separate two runs' rows without the record.
    run_id: str
    #: The recipe this row was scored under, denormalized onto the row for the same reason.
    config_hash: str
    #: R1's roster, by name. Keys are checked against the schema's at write time rather than
    #: trusted, so a pass that computed nine of the ten cannot land a column of nulls.
    metrics: Mapping[str, float] = field(default_factory=dict)

    def cell(self) -> tuple[str, str, int]:
        """The ``(arm, size, seed)`` triple ``run`` executes and :func:`append_rows` replaces."""
        return (self.arm, self.size, self.seed)


def mask_rle_md5(mask_rle: str) -> str:
    """The committed mask's identity — see :data:`GEOMETRY_COLUMNS`."""
    return hashlib.md5(mask_rle.encode("utf-8")).hexdigest()


def columns(metrics: Sequence[str]) -> tuple[str, ...]:
    """Every column, in the committed order: the key, the geometry, provenance, then metrics."""
    return (*KEY_COLUMNS, *GEOMETRY_COLUMNS, *PROVENANCE_COLUMNS, *metrics)


def schema(metrics: Sequence[str]):
    """The parquet schema for a given metric roster.

    A *function* of the roster rather than a constant, because the roster is R1's and lives in
    the registered bar. A results file whose columns were typed here could disagree with the bar
    the run was scored under, and the disagreement would look like a missing metric rather than
    like a contract violation.
    """
    import pyarrow as pa

    integer = {"seed", "instance", "void_area_px"}
    text = {*KEY_COLUMNS, *PROVENANCE_COLUMNS, "mask_rle_md5"} - integer
    fields = []
    for name in columns(metrics):
        if name in integer:
            fields.append((name, pa.int64()))
        elif name in text:
            fields.append((name, pa.string()))
        else:
            fields.append((name, pa.float64()))
    return pa.schema(fields)


def _table(rows: Sequence[ResultRow], metrics: Sequence[str]):
    """The rows as an Arrow table, refusing a row whose metric keys are not the roster."""
    import pyarrow as pa

    for row in rows:
        missing = sorted(set(metrics) - set(row.metrics))
        extra = sorted(set(row.metrics) - set(metrics))
        if missing or extra:
            raise ValueError(
                f"{row.arm} {row.tile_id} {row.family}/{row.bucket} scored "
                f"{sorted(row.metrics)}, and this file's roster is {list(metrics)} "
                f"(missing {missing}, unregistered {extra}). R1's roster is what the results "
                f"parquet's metric columns are, and a row short of it would land a column of "
                f"nulls that reads as a metric this arm scored badly"
            )
    data: dict[str, list] = {}
    for name in columns(metrics):
        if name in metrics:
            data[name] = [float(row.metrics[name]) for row in rows]
        else:
            data[name] = [getattr(row, name) for row in rows]
    return pa.table(data, schema=schema(metrics))


def append_rows(
    rows: Sequence[ResultRow],
    path: Path | str,
    *,
    metrics: Sequence[str],
) -> tuple[Path, int]:
    """Write ``rows`` into the results parquet, replacing whatever this cell had there.

    Returns ``(path, total_rows)``. The replacement is by ``(arm, size, seed)`` and the reason is
    the module docstring's: two passes over one cell would double-count every tile they agree on,
    silently, in every aggregate ``report`` computes.

    The whole file is rewritten rather than appended to as a second row group. That is the honest
    shape for a replace — parquet has no delete — and it is affordable at this file's size: one
    arm's pass is ~40,000 rows and the full table is eleven printed rows over three seeds, which
    is ~1.3M rows of mostly-repeated short strings.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    fresh = _table(rows, metrics)
    if path.is_file():
        committed = pq.read_table(path)
        if committed.schema != fresh.schema:
            raise ValueError(
                f"{path} was written with columns {committed.schema.names} and this run scores "
                f"{fresh.schema.names}. The metric roster comes from the registered bar (R1), so "
                f"a mismatch means this file and this run were scored under two different "
                f"pre-registrations — which is not something to merge. Move the old file aside "
                f"deliberately; every table already rendered from it quotes its own run records"
            )
        replaced = {row.cell() for row in rows}
        # A boolean filter rather than `take` on a list of indices: re-running the *only* cell a
        # file holds keeps nothing, and an empty Python list has no Arrow type for `take` to
        # match against — which surfaces as a kernel error about `(string, null)` rather than as
        # anything to do with cells. `filter` is typed by the mask and is happy to keep zero rows.
        keep = pa.array(
            [
                cell not in replaced
                for cell in zip(
                    committed.column("arm").to_pylist(),
                    committed.column("size").to_pylist(),
                    committed.column("seed").to_pylist(),
                )
            ],
            type=pa.bool_(),
        )
        fresh = pa.concat_tables([committed.filter(keep), fresh])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(fresh, path, compression="zstd")
    return path, fresh.num_rows


def read_rows(path: Path | str):
    """The results parquet as an Arrow table — what ``report`` is a pure function of."""
    import pyarrow.parquet as pq

    return pq.read_table(Path(path))


def rows_for_cell(
    path: Path | str, *, arm: str, size: str, seed: int, columns: Sequence[str]
) -> list[dict]:
    """One cell's rows as plain dicts, or ``[]`` when the file holds none.

    Here rather than at the caller because this module owns the file: which columns key a cell
    is :meth:`ResultRow.cell`'s answer, and a second filter written beside a consumer is one that
    can disagree with :func:`append_rows`' replacement rule about what "this cell" means.

    ``[]`` for a missing file rather than a refusal, on :func:`read_run_records`' rule: a results
    parquet that does not exist yet is a state, and the caller — #167's R12 comparison, which
    needs A1's rows to read A2's row — has something specific to say about it that an exception
    here would pre-empt.
    """
    path = Path(path)
    if not path.is_file():
        return []
    table = read_rows(path)
    missing = [name for name in (*columns, *KEY_COLUMNS) if name not in table.schema.names]
    if missing:
        raise ValueError(
            f"{path} has no {missing} column(s); it holds {table.schema.names}. The metric "
            f"roster is the registered bar's (R1), so a file written under a different roster "
            f"is a file this run cannot read a cell out of"
        )
    # Filtered in Arrow and *then* decoded, rather than the other way round. The full table is
    # ~1.3M rows by the end of the sweep (eleven printed rows over three seeds) and one cell is
    # ~40,000 of them, so decoding first would turn a per-run read into eight million Python
    # objects to throw away.
    import pyarrow.compute as pc

    cell = table.filter(
        pc.and_(
            pc.and_(pc.equal(table["arm"], arm), pc.equal(table["size"], size)),
            pc.equal(table["seed"], seed),
        )
    )
    column = {name: cell.column(name).to_pylist() for name in columns}
    return [{name: column[name][i] for name in columns} for i in range(cell.num_rows)]


# --- The run record -----------------------------------------------------------


def run_id(resolution: Resolution) -> str:
    """A stable name for *this run of this cell against the inputs it actually read*.

    A hash of the provenance payload rather than a timestamp or a counter, so that re-running an
    unchanged cell against unchanged artifacts under unchanged code produces the **same** id.
    That is what makes ``run`` idempotent in the way that matters: the rows are replaced and the
    record they point at is the one that was already there, rather than a second identity for a
    computation that did not change.

    It moves when anything a results row is answerable to moves — the recipe, the checkpoint, the
    git SHA, and the md5 of every committed input **this arm reads**.

    That last qualifier is load-bearing rather than tidy. :meth:`Resolution.to_dict` records every
    artifact on the box, read or not, because that is what provenance is; hashing all of them
    would mean that the day #151 first writes a train mask bank, every A1 row's ``run_id`` moves
    — announcing a change to a computation whose every input was identical. An id that moves for
    a reason the rows do not share is worse than no id, because it is quoted in a table.
    """
    resolved = resolution.to_dict()
    resolved["artifacts"] = {
        key: entry
        for key, entry in resolved["artifacts"].items()
        if entry.get("required", True)
    }
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def run_record(
    resolution: Resolution,
    *,
    rows: int,
    seconds: float,
    parameters: Mapping[str, int | None],
    metrics: Sequence[str],
    gates: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    register: Mapping[str, Any] | None = None,
) -> dict:
    """The provenance payload a results table is answerable to.

    :meth:`benchmark.artifacts.Resolution.to_dict` is embedded whole rather than restated: it
    already carries the recipe, the config hash, §3.4's artifact hash, the arm, the checkpoint,
    every committed input by md5 — the extractor spec's and the mask bank's among them — and the
    git SHA. What this function adds is what the run *produced*, which is the other half of the
    split :func:`pretrain.header.run_record` and :mod:`pretrain.fingerprint` have for the same
    reason: what a run read and what it produced are two questions.

    ``parameters`` is §6 item 4's, and it is a mapping with two keys rather than two ints so that
    "not built yet" stays expressible — see :func:`benchmark.arms.parameter_counts`.

    ``register`` is a **disposition**, not a rule: how this cell's row is to be read under
    whichever registered rule governs it, computed rather than asserted. ADR-0013's register is
    already committed in `task1-prereg-bar.json` and is not restated here. What a run can add is
    the outcome — R12's majority-of-bands comparison and its verdict (:func:`benchmark.zeroshot.
    disposition`) — for the rules whose trigger is a *result* rather than a threshold, and which
    therefore have nothing a bar file could hold. ``None`` for a cell no such rule governs, and
    the key is then absent rather than null: a block on every row is one a reviewer stops
    reading, and an empty one would say "R12 found nothing" where nothing asked.
    """
    return {
        "run_id": run_id(resolution),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "resolved": resolution.to_dict(),
        "cell": {
            "arm": resolution.config.arm,
            "size": resolution.config.size,
            "seed": resolution.config.seed,
        },
        "parameters": dict(parameters),
        "produced": {
            "rows": rows,
            "seconds": round(float(seconds), 3),
            "metrics": list(metrics),
        },
        "gates": dict(gates or {}),
        # The training half, for the arms that have one (#159): the head's parameter count, the
        # epoch val selected, the checkpoint holding it, and the committed lists the pass read.
        # `{}` for A1 and every inference-only arm, which is a different fact from "not recorded"
        # and reads as one because `parameters` above is already `None`-for-unknown.
        "training": dict(training or {}),
        # R12's disposition, for the arm that has one (#167). Omitted rather than empty, which
        # is that ticket's own choice and is left alone.
        **({"register": dict(register)} if register else {}),
    }


def append_run_record(record: Mapping[str, Any], path: Path | str) -> Path:
    """Append one run record, as one JSON line.

    Append-only, where the parquet is replace-by-cell: a cell scored twice is a fact about how
    this table was produced, and the second pass overwriting the first one's rows is exactly when
    that fact stops being visible anywhere else.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def read_run_records(path: Path | str) -> list[dict]:
    """Every run record written beside a results file, oldest first."""
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def format_written(
    path: Path, *, rows: int, total: int, record: Mapping[str, Any]
) -> list[str]:
    """What ``run`` says it produced — by content, since that is what a reader has to trust."""
    parameters = record["parameters"]

    def count(key: str) -> str:
        value = parameters.get(key)
        return "not built yet" if value is None else f"{value:,}"

    return [
        f"{'scored':<16} {rows:,} rows into {path} ({total:,} in the file)",
        f"{'run':<16} {record['run_id']} — config {record['resolved']['config_hash']}, "
        f"artifacts {record['resolved']['artifact_hash']}, git {record['resolved']['git_sha']}",
        f"{'params':<16} {count('total')} total / {count('trainable')} trainable",
        f"{'record':<16} appended to {run_record_path(path)}",
    ]


def write(
    rows: Sequence[ResultRow],
    resolution: Resolution,
    *,
    metrics: Sequence[str],
    seconds: float,
    parameters: Mapping[str, int | None],
    gates: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    register: Mapping[str, Any] | None = None,
) -> tuple[Path, int, dict]:
    """Land one cell: the rows into the parquet, the record beside it. Returns both plus totals.

    One call rather than two, because a pass that wrote rows and then failed to write its record
    would leave a table nothing is answerable to — which is the state this whole module exists to
    make unreachable. The record goes second: a record naming rows that are not there is a
    louder, more debuggable failure than rows no record explains.
    """
    path, total = append_rows(rows, resolution.config.paths.results_path, metrics=metrics)
    record = run_record(
        resolution,
        rows=len(rows),
        seconds=seconds,
        parameters=parameters,
        metrics=metrics,
        gates=gates,
        training=training,
        register=register,
    )
    append_run_record(record, run_record_path(path))
    return path, total, record


def rows_from(
    scored: Iterable[Any],
    *,
    arm: str,
    init: str | None,
    size: str,
    seed: int,
    run: str,
    config_hash: str,
) -> list[ResultRow]:
    """Turn a pass's :class:`benchmark.evaluate.Scored` instances into rows.

    The cell's constant columns — the arm, its init, the size, the seed and the run's identity —
    are filled in once here rather than carried on every instance the pass produced. That is the
    split the two modules are for: the pass scores a fill and knows nothing about which cell
    asked for it, and this module owns what a row *is*.
    """
    return [
        ResultRow(
            arm=arm,
            init=init,
            size=size,
            seed=seed,
            family=item.family,
            bucket=item.bucket,
            band=item.band,
            tile_id=item.tile_id,
            instance=item.instance,
            void_area_px=item.void_area_px,
            void_eqdiam_px=item.void_eqdiam_px,
            mask_rle_md5=item.mask_rle_md5,
            run_id=run,
            config_hash=config_hash,
            metrics=item.metrics,
        )
        for item in scored
    ]
