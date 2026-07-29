"""Stage 5 — the materialization driver: selection list in, corpus on disk out (Phase E1).

This is the map's central execution ticket ([#51](https://github.com/JStoschek/DEMAE/issues/51)):
it reads a committed selection list at **full resolution** and turns it into int16
shards, a v2 manifest, a corpus card, and a materialization ledger. Both shipped tiers
are this one driver over two lists — the 50,000-window debug tier (#51) and the
500,000-window scale tier ([#72](https://github.com/JStoschek/DEMAE/issues/72)), which
renegotiated no contract and so needed only a second set of paths to point at.
It orchestrates components that already exist and adds no new recipe — :mod:`dataset.tiling`
reads, :mod:`dataset.normalize` screens and prepares, :mod:`dataset.shards` stores,
:mod:`dataset.manifest` indexes, :mod:`dataset.concurrency` schedules. What lives *here*
is the five things none of them can own alone.

**1. Grouping.** The unit of network cost is a COG *block*, not a window (E1,
``docs/research/e1-read-cost.md``), and opening a COG is ~30% of serial cost. So the
selection list is grouped by COG and each COG is one task, which is the whole reason
:func:`~dataset.tiling.CogReader.read_windows` takes a *list* of origins.

**2. Attrition, under `accept_shortfall`** ([#50](https://github.com/JStoschek/DEMAE/issues/50)).
A selected window that fails at full res is **recorded, never backfilled**. The manifest
keeps one row per tile *on disk*; the dropped windows live only in the
:data:`LEDGER_SCHEMA` ledger, which carries one row per **selected** window and so is the
only artifact here that cannot be re-derived from committed inputs. Hence it is committed
too. The invariant :func:`_assert_ledger_matches_manifest` enforces: the ledger's ``ok``
window ids are exactly the manifest's.

**3. The corpus-wide land mean/std** ADR-0003's ``global`` normalization mode needs,
accumulated **streaming** as tiles are prepared — a second pass over 5 GB to compute two
floats would be absurd.

**4. Resumability.** A ~1-hour S3 pass that dies at 80% must not restart from zero. The
checkpoint granularity is **the COG**, because the COG is the unit of work, and
:meth:`~dataset.shards.ShardWriter.checkpoint` makes the half-written shard a resume point
rather than something to discard. A checkpoint is the **only** commit: it flushes the
shard, rewrites the ledger and manifest, and writes :data:`STATE_FILE` **last**. Anything
after the last state write — rows in the shard, rows in either Parquet — is disowned on
resume and re-done, so a kill at any instant costs at most one checkpoint of work and can
never double-book a window.

**5. Determinism.** The manifest and the ``dataset_version`` fingerprint reproduce across
runs *and across executors*: :func:`~dataset.concurrency.ordered_futures` yields COGs in
input order while keeping the pool fed, and the shard writer stays serial in the parent.
E1 demonstrated this over 199 tiles at serial / 16 threads / 8 processes — one digest
(``docs/research/e1-read-cost.md`` §Determinism). Shard *ordering* is not free from a
parallel executor; it is paid for here, once.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Sequence

import numpy as np

from . import config, manifest, normalize
from .catalog import FROZEN_METRICS, META_FILE, PARTS_DIR
from .concurrency import ExecutorKind, default_workers, make_executor
from .geometry import make_window_id, parse_cog_key, parse_window_id
from .shards import ShardState, ShardWriter
from .tiling import CogReader

# --- Committed paths ----------------------------------------------------------

#: Phase D's committed selection list — the determinism contract (#44). The literal
#: lives in `config` so this stage and `dataset.select`, which writes it, cannot drift.
SELECTION_PATH: Final = Path(config.SELECTION_PATH)
#: The Phase-D provenance chain the corpus card records (#48 / ADR-0002 §3a). Shared by
#: both tiers: the scale draw came from the *same* persisted centroids and scaler (#71).
CENTROIDS_PATH: Final = Path("artifacts/phase_d/centroids.json")
SCALER_PATH: Final = Path("artifacts/phase_d/scaler.json")
#: One row per *selected* window, with the cause of every drop. **Committed** — unlike
#: the shards and the manifest it cannot be re-derived from committed inputs, because
#: knowing which windows failed requires the full-res read (#50 item 1).
LEDGER_PATH: Final = Path("artifacts/phase_e/materialization_ledger.parquet")
#: The corpus itself: gitignored, ~5 GB at the debug tier.
CORPUS_DIR: Final = Path("data/corpus_50k")

#: The Phase-F scale tier (#72): the same driver over ten times the budget. Nothing in
#: the recipe changes — the ticket renegotiated no contract — so what a tier *is*, here,
#: is exactly these three paths.
SCALE_SELECTION_PATH: Final = Path(config.SCALE_SELECTION_PATH)
SCALE_CORPUS_DIR: Final = Path("data/corpus_500k")
SCALE_LEDGER_PATH: Final = Path("artifacts/phase_f/materialization_ledger.parquet")

MANIFEST_FILE: Final = "manifest.parquet"
CORPUS_META_FILE: Final = "corpus_meta.json"
SHARDS_DIR: Final = "shards"
#: Resume bookkeeping. Written **last** at every checkpoint, so its existence and
#: contents define exactly what a resumed run may trust.
STATE_FILE: Final = "build_state.json"
STATE_VERSION: Final = 1

#: Per-window outcomes the ledger books (#50 item 1). Four come straight from
#: :func:`~dataset.normalize.prepare_tile`; ``read_error`` is this driver's, for a COG
#: whose full-res read would not come back. ``no_land`` is in the vocabulary but is
#: structurally unreachable while :data:`config.LAND_ZERO_FRAC_DROP` < 1.0 — a window
#: with no land pixels is 100% zero-fill, so the zero-fill screen claims it first.
LEDGER_STATUSES: Final = ("ok", "zero_fill", "no_land", "degenerate", "read_error")

#: Catalog columns the manifest denormalizes (ADR-0002 §3a "Denormalize, not join").
#: ``band``/``geocell`` are the catalog's names for the manifest's ``lat_band``/``geocell_id``.
CATALOG_COLUMNS: Final = (
    "window_id",
    "band",
    "geocell",
    "lon",
    "lat",
    *FROZEN_METRICS,
    "edm_edited_frac",
    "flm_fill_frac",
    "wbm_water_frac",
    "water_affected",
    "coastline",
)

#: COGs between checkpoints. ~250 is ~1 minute of either tier's projected pass (15,834
#: COGs at the debug tier, 22,909 at the scale tier), against a checkpoint that costs two
#: small Parquet rewrites — the cheap end of a cheap trade.
CHECKPOINT_EVERY: Final = 250
#: Whole-COG read attempts before the windows are booked ``read_error``. A transient
#: ``/vsis3`` blip is the common failure and it is not worth a re-run for.
READ_ATTEMPTS: Final = 3
RETRY_BACKOFF_S: Final = 0.5
#: Consecutive all-failed COGs that mean the transport is gone, not the data. Booking a
#: whole tier's COGs dead in a tight loop during a network outage is the failure this stops.
MAX_CONSECUTIVE_READ_ERRORS: Final = 50


class ReadOutage(RuntimeError):
    """Too many consecutive COGs failed — the transport is down, not the corpus.

    Raised before *this* streak is committed, so at most the failures a checkpoint
    happened to catch mid-streak survive the abort — and those stay retryable, which
    is what :data:`BuildState.read_error_cogs` is for. Without the breaker a dropped
    connection would book every remaining COG dead in a tight loop.
    """


# --- Tiers --------------------------------------------------------------------


@dataclass(frozen=True)
class Tier:
    """A tier's three paths, which move together or not at all.

    :mod:`dataset.select` deliberately does *not* derive its ``--out`` from its
    ``--budget`` — a tier is a budget and where it is spent, and the operator names
    both. That reasoning holds for one output; this stage has three, and they are not
    independent. Passing ``--selection`` and ``--out-dir`` for the scale tier while
    forgetting ``--ledger`` writes 500,000 rows over
    ``artifacts/phase_e/materialization_ledger.parquet`` — the **committed** debug
    ledger, the one artifact in the repo that cannot be re-derived from committed
    inputs (#50 item 1). The ``selection_md5`` guard in :func:`run` does not catch it:
    that guard lives in ``build_state.json``, which sits beside the *shards*, so a
    fresh ``--out-dir`` has no state to disagree with and the stale ledger is simply
    truncated.

    Naming the tier once, rather than the three paths, is what closes that. But the
    *type* alone would only close it for the CLI, and the hazard is not the CLI's — it
    is any caller that names two tiers' artifacts in one run. So the check that
    actually holds the line is :func:`_assert_tier_paths_agree`, in :func:`run`, below
    every override: paths that belong to a shipped tier must all belong to the **same**
    one. Overrides stay unrestricted otherwise, because a path that is nobody's tier —
    a scratch directory for a ``--limit-cogs`` smoke — claims nothing and crosses
    nothing.
    """

    name: str
    selection_path: Path
    out_dir: Path
    ledger_path: Path


#: The two shipped tiers — the debug corpus (#51) and the scale corpus (#72).
TIERS: Final[dict[str, Tier]] = {
    "debug": Tier(
        name="debug",
        selection_path=SELECTION_PATH,
        out_dir=CORPUS_DIR,
        ledger_path=LEDGER_PATH,
    ),
    "scale": Tier(
        name="scale",
        selection_path=SCALE_SELECTION_PATH,
        out_dir=SCALE_CORPUS_DIR,
        ledger_path=SCALE_LEDGER_PATH,
    ),
}


def resolve_tier(
    name: str,
    *,
    selection_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    ledger_path: Path | str | None = None,
) -> Tier:
    """The named tier, with any explicitly-given path substituted in.

    ``is not None`` rather than truthiness, so an empty override is a caller error the
    filesystem reports rather than a silent fall-back to a *committed* default — the
    one direction this stage must never fail in (#50 item 1).
    """
    if name not in TIERS:
        raise KeyError(f"unknown tier {name!r}; shipped tiers: {sorted(TIERS)}")
    tier = TIERS[name]
    return dataclasses.replace(
        tier,
        **{
            field: Path(value)
            for field, value in (
                ("selection_path", selection_path),
                ("out_dir", out_dir),
                ("ledger_path", ledger_path),
            )
            if value is not None
        },
    )


def _assert_tier_paths_agree(tier: Tier) -> None:
    """Refuse a run that names one tier's artifact beside another's.

    The failure this exists for is silent and unrecoverable: a scale-tier selection
    list written into the debug tier's **committed** ledger truncates the only record
    of which debug windows failed at full res, and no committed input can rebuild it
    (#50 item 1). A path belonging to no shipped tier — a scratch directory — claims
    nothing and is left alone, so ``--limit-cogs`` smoke runs stay unrestricted.
    """
    claimed = {
        field: shipped.name
        for field in ("selection_path", "out_dir", "ledger_path")
        for shipped in TIERS.values()
        if getattr(tier, field) == getattr(shipped, field)
    }
    if len(set(claimed.values())) > 1:
        named = ", ".join(f"{field}={name}" for field, name in sorted(claimed.items()))
        raise ValueError(
            f"these paths belong to different tiers ({named}) — refusing to run. "
            "One tier's artifacts must not be written from another's inputs; pass "
            "--tier alone, or override every path away from a shipped tier."
        )


# --- The selection list, grouped ----------------------------------------------


@dataclass(frozen=True)
class SelectedWindow:
    """One row of the selection list, with its ``window_id`` already parsed."""

    window_id: str
    stem: str
    win_row0: int
    win_col0: int
    stratum_id: int
    provenance: str


@dataclass(frozen=True)
class CogJob:
    """Every selected window of one COG — the driver's unit of work and of resume."""

    stem: str
    windows: tuple[SelectedWindow, ...]

    @property
    def origins(self) -> list[tuple[int, int]]:
        return [(w.win_row0, w.win_col0) for w in self.windows]


def window_id_of(manifest_row: dict) -> str:
    """The window id a manifest row came from.

    The v2 manifest stores the window's *parts* (``source_cog``, ``win_row0``,
    ``win_col0``) rather than the composite id, so recovering the id is this one
    function's job — it is how the ledger and the manifest are checked against each
    other, and how the E2 notebook joins them.
    """
    return make_window_id(
        manifest_row["source_cog"], manifest_row["win_row0"], manifest_row["win_col0"]
    )


def load_selection(path: Path | str = SELECTION_PATH) -> list[CogJob]:
    """Read the committed selection list and group it by source COG.

    Grouping is not a convenience: opening a COG costs ~0.43 s against ~0.24 s per
    block read, so a driver that saw the windows one at a time would pay the open once
    per *window* instead of once per COG — 22,909 opens against 500,000 at the scale
    tier. COGs are ordered by stem and windows by pixel origin, which
    makes the job list — and therefore the shard row order — a pure function of the
    committed file, independent of executor, worker count and filesystem order.
    """
    import csv

    grouped: dict[str, list[SelectedWindow]] = {}
    seen: set[str] = set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            window_id = row["window_id"]
            if window_id in seen:
                raise ValueError(f"duplicate window_id in {path}: {window_id}")
            seen.add(window_id)
            stem, r0, c0 = parse_window_id(window_id)
            grouped.setdefault(stem, []).append(
                SelectedWindow(
                    window_id=window_id,
                    stem=stem,
                    win_row0=r0,
                    win_col0=c0,
                    stratum_id=int(row["stratum_id"]),
                    provenance=row["provenance"],
                )
            )
    return [
        CogJob(stem=stem, windows=tuple(sorted(ws, key=lambda w: (w.win_row0, w.win_col0))))
        for stem, ws in sorted(grouped.items())
    ]


# --- The catalog side table ---------------------------------------------------


class CatalogSideTable:
    """The gated catalog's per-window facts, narrowed to the selected windows.

    ADR-0002 §3a decided the manifest **denormalizes** these columns rather than
    carrying a join key, so a corpus consumer needs one self-contained file instead of
    a 4.4M-row sampling frame. This is the narrowing: the catalog is read part by part
    (bounding memory to one part, not the whole frame) and everything outside the
    selection is dropped on the way through.
    """

    def __init__(self, columns: dict[str, list], index: dict[str, int]) -> None:
        self._columns = columns
        self._index = index

    def __len__(self) -> int:
        return len(self._index)

    @classmethod
    def load(cls, catalog_dir: Path | str, window_ids: Sequence[str]) -> "CatalogSideTable":
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        parts = sorted((Path(catalog_dir) / PARTS_DIR).glob("part-*.parquet"))
        if not parts:
            raise FileNotFoundError(f"no catalog parts under {Path(catalog_dir) / PARTS_DIR}")
        wanted = pa.array(sorted(set(window_ids)), type=pa.string())

        kept = []
        for part in parts:
            table = pq.read_table(part, columns=list(CATALOG_COLUMNS))
            kept.append(table.filter(pc.is_in(table.column("window_id"), value_set=wanted)))
        table = pa.concat_tables(kept)

        columns = {name: table.column(name).to_pylist() for name in CATALOG_COLUMNS}
        index = {wid: i for i, wid in enumerate(columns["window_id"])}

        missing = [w for w in window_ids if w not in index]
        if missing:
            raise KeyError(
                f"{len(missing)} selected window(s) not in the gated catalog at "
                f"{catalog_dir} — the selection list and the catalog disagree; "
                f"first: {missing[:3]}"
            )
        return cls(columns, index)

    def row(self, window_id: str) -> dict[str, Any]:
        i = self._index[window_id]
        return {name: values[i] for name, values in self._columns.items()}


# --- The unit of work (runs in a worker) --------------------------------------


@dataclass
class WindowOutcome:
    """What became of one selected window, plus what the parent needs to write it."""

    window_id: str
    status: str
    detail: str | None = None
    tile: np.ndarray | None = None  # (224, 224) int16, only when status == "ok"
    mean: float = 0.0
    std: float = 0.0
    land_frac: float = 0.0
    # Streaming accumulators for the ADR-0003 corpus-wide land mean/std.
    land_sum: float = 0.0
    land_sqsum: float = 0.0
    land_count: int = 0


def read_cog(
    cog_reader: CogReader,
    tile_px: int,
    attempts: int,
    backoff_s: float,
    job: CogJob,
) -> list[WindowOutcome]:
    """Read and prepare every selected window of one COG. **The driver's inner loop.**

    Prepared int16 tiles come back rather than raw float32 windows: preparation is
    cheap and local (297 µs against a ~240 ms block read), and doing it in the worker
    halves what a process pool ships home over its pipe — 100 KB int16 against 200 KB
    float32 per tile. The shard write stays in the parent, where it is serial and so
    deterministic.

    A read that raises is retried ``attempts`` times with a linear backoff, because the
    common failure is a transient ``/vsis3`` blip and a re-run of a 1-hour pass is a
    poor way to absorb one. What survives that is booked ``read_error`` for every window
    of the COG with the exception text as ``detail`` — under ``accept_shortfall`` the
    failure is *recorded*, not retried forever and not backfilled (#50).

    Module-level and taken bound by :func:`functools.partial` rather than a closure,
    because a process pool has to pickle the task.
    """
    cog = parse_cog_key(job.stem)
    last: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            windows = cog_reader.read_windows(cog, job.origins, tile_px)
            break
        except Exception as exc:  # noqa: BLE001 — booked with its cause, never swallowed
            last = exc
            if attempt + 1 < attempts and backoff_s:
                time.sleep(backoff_s * (attempt + 1))
    else:
        windows = None
    if windows is None:
        detail = f"{type(last).__name__}: {last}"
        return [
            WindowOutcome(window_id=w.window_id, status="read_error", detail=detail)
            for w in job.windows
        ]

    out: list[WindowOutcome] = []
    for window, selected in zip(windows, job.windows):
        if window.shape != (tile_px, tile_px):
            out.append(
                WindowOutcome(
                    window_id=selected.window_id,
                    status="read_error",
                    detail=f"window shape {window.shape} != ({tile_px}, {tile_px})",
                )
            )
            continue
        prepared, reason = normalize.prepare_tile(window)
        if prepared is None:
            out.append(WindowOutcome(window_id=selected.window_id, status=reason))
            continue
        land_sum, land_sqsum, land_count = normalize.land_moments(prepared.raw_i16)
        out.append(
            WindowOutcome(
                window_id=selected.window_id,
                status="ok",
                tile=prepared.raw_i16,
                mean=prepared.mean,
                std=prepared.std,
                land_frac=prepared.land_frac,
                land_sum=land_sum,
                land_sqsum=land_sqsum,
                land_count=land_count,
            )
        )
    return out


# --- Build fingerprints -------------------------------------------------------


def file_md5(path: Path | str) -> str:
    """md5 of a committed artifact — how the corpus card pins its inputs (#48)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Fingerprints:
    """The four Phase-D/Phase-C pins that feed :func:`dataset.manifest.build_params`.

    They travel as one value because they are one claim: *this* corpus was built from
    *these* committed artifacts. ``catalog_version`` is the md5 of the gated catalog's
    ``catalog_meta.json``, which is content-derived (it records the row counts, the gate
    thresholds and the source catalog's code version) and therefore stable across two
    runs, unlike a build timestamp.
    """

    selection_list_md5: str
    catalog_version: str
    centroids_md5: str
    scaler_md5: str

    @classmethod
    def collect(
        cls,
        *,
        selection_path: Path,
        catalog_dir: Path,
        centroids_path: Path,
        scaler_path: Path,
    ) -> "Fingerprints":
        return cls(
            selection_list_md5=file_md5(selection_path),
            catalog_version=file_md5(Path(catalog_dir) / META_FILE),
            centroids_md5=file_md5(centroids_path),
            scaler_md5=file_md5(scaler_path),
        )

    def as_kwargs(self) -> dict[str, str]:
        return {
            "selection_list_md5": self.selection_list_md5,
            "catalog_version": self.catalog_version,
            "centroids_md5": self.centroids_md5,
            "scaler_md5": self.scaler_md5,
        }


# --- Checkpoint state ---------------------------------------------------------


@dataclass
class BuildState:
    """What a previous run committed — everything a resume is allowed to trust.

    ``n_cogs_done`` is an *index into the job list*, not a set of stems, and that is
    only sound because completion is a strict prefix: :func:`ordered_futures` consumes
    COGs in input order and every COG reaches a terminal state when it is consumed
    (``read_error`` included), so there is never a hole behind the frontier.
    ``selection_md5`` and ``n_cogs_total`` guard the assumption — a job list that
    changed underneath the state makes the index meaningless, so the resume refuses
    instead of writing a corpus that is half one selection and half another.

    ``read_error_cogs`` is the exception the frontier cannot express: the job indices
    of COGs that were booked failed. They sit *behind* the frontier, so a plain resume
    steps over them, and ``--retry-read-errors`` is what brings them back. Keeping them
    named is the difference between "this window failed" and "this window is lost":
    ``accept_shortfall`` (#50) forbids backfilling a *different* window, and says
    nothing about re-reading the one that was selected.

    ``elapsed_s`` accumulates across invocations so the wall clock a resumed build
    reports is the pass's, not its last leg's.
    """

    selection_md5: str = ""
    n_cogs_total: int = 0
    n_cogs_done: int = 0
    n_windows_done: int = 0
    shards: ShardState = ShardState()
    land_sum: float = 0.0
    land_sqsum: float = 0.0
    land_count: int = 0
    read_error_cogs: tuple[int, ...] = ()
    elapsed_s: float = 0.0
    complete: bool = False

    def as_dict(self) -> dict:
        return {
            "state_version": STATE_VERSION,
            "selection_md5": self.selection_md5,
            "n_cogs_total": self.n_cogs_total,
            "n_cogs_done": self.n_cogs_done,
            "n_windows_done": self.n_windows_done,
            "shards": self.shards.as_dict(),
            "land_sum": self.land_sum,
            "land_sqsum": self.land_sqsum,
            "land_count": self.land_count,
            "read_error_cogs": sorted(self.read_error_cogs),
            "elapsed_s": self.elapsed_s,
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BuildState":
        return cls(
            selection_md5=d["selection_md5"],
            n_cogs_total=int(d["n_cogs_total"]),
            n_cogs_done=int(d["n_cogs_done"]),
            n_windows_done=int(d["n_windows_done"]),
            shards=ShardState.from_dict(d["shards"]),
            land_sum=float(d["land_sum"]),
            land_sqsum=float(d["land_sqsum"]),
            land_count=int(d["land_count"]),
            read_error_cogs=tuple(d.get("read_error_cogs", ())),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
            complete=bool(d["complete"]),
        )


def read_state(out_dir: Path) -> BuildState | None:
    """Replay :data:`STATE_FILE`, or ``None`` if there is nothing to resume from.

    A torn or unreadable file is treated as absent rather than raising: it can only be
    a crash mid-write, and the honest reading of a half-written commit record is that
    the commit did not happen.
    """
    path = out_dir / STATE_FILE
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
        if int(d.get("state_version", -1)) != STATE_VERSION:
            return None
        return BuildState.from_dict(d)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _replace_atomic(tmp: Path, path: Path) -> None:
    """fsync then rename — so the file at ``path`` is either the old one or a whole new one.

    The checkpoint's crash-safety rests on this: a torn manifest or ledger would make a
    state file that claims those rows a lie, and the resume would have to be abandoned.
    """
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


# --- The ledger ---------------------------------------------------------------


def _ledger_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("window_id", pa.string()),
            ("stratum_id", pa.int32()),
            ("provenance", pa.string()),
            ("status", pa.string()),
            ("detail", pa.string()),  # read errors only; null everywhere else
        ]
    )


#: One row per **selected** window: what it was allocated as, and what became of it
#: (#50 item 1). The manifest is unchanged and keeps one row per tile *on disk*, so a
#: dropped window appears here and nowhere else.
LEDGER_SCHEMA = _ledger_schema()


def write_ledger(rows: list[dict], path: Path | str) -> Path:
    """Write the materialization ledger, atomically."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {name: [r[name] for r in rows] for name in LEDGER_SCHEMA.names}
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.table(cols, schema=LEDGER_SCHEMA), tmp)
    _replace_atomic(tmp, path)
    return path


def read_ledger(path: Path | str) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


# --- The build report ---------------------------------------------------------


@dataclass(frozen=True)
class BuildResult:
    """What the pass realized — the numbers #51 asks to be reported."""

    out_dir: Path
    ledger_path: Path
    complete: bool
    n_selected: int  # rows in the committed selection list
    n_tiles: int  # rows in the manifest = tiles on disk
    n_cogs: int  # COGs read by *this* invocation
    n_skipped: int  # COGs a previous invocation had already finished
    n_cogs_total: int  # COGs the selection list groups into
    status_counts: dict[str, int]
    per_stratum: dict[int, int]
    provenance_counts: dict[str, int]
    n_shards: int
    shard_bytes: int
    global_mean: float
    global_std: float
    dataset_version: str
    wall_s: float

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / MANIFEST_FILE

    @property
    def meta_path(self) -> Path:
        return self.out_dir / CORPUS_META_FILE

    @property
    def n_dropped(self) -> int:
        return self.n_selected - self.n_tiles

    @property
    def s_per_cog(self) -> float:
        """Wall clock per COG — directly comparable to E1's measured 0.24 s at process:16."""
        return self.wall_s / max(1, self.n_cogs_total)

    @property
    def n_attempted(self) -> int:
        """Selected windows this build has actually reached — all of them once complete."""
        return sum(self.status_counts.values())

    def summary_lines(self) -> list[str]:
        drops = ", ".join(
            f"{status} {self.status_counts.get(status, 0):,}"
            for status in LEDGER_STATUSES
            if status != "ok" and self.status_counts.get(status)
        )
        if self.complete:
            rate = self.n_dropped / self.n_selected if self.n_selected else 0.0
            headline = (
                f"Materialized {self.n_tiles:,} tiles of {self.n_selected:,} selected "
                f"({rate:.3%} dropped under accept_shortfall)."
            )
        else:
            # A partial build's drop rate against the full budget would read as
            # catastrophic attrition; the honest denominator is what it has reached.
            headline = (
                f"Materialized {self.n_tiles:,} tiles of {self.n_attempted:,} windows "
                f"attempted so far ({self.n_selected:,} selected in total)."
            )
        return [
            headline,
            f"  drops: {drops or 'none'}",
            f"  strata: {len(self.per_stratum)} occupied, "
            f"min {min(self.per_stratum.values()) if self.per_stratum else 0:,}, "
            f"max {max(self.per_stratum.values()) if self.per_stratum else 0:,}",
            # Printed, not just returned: the ticket asks the *report* to carry the
            # per-stratum counts, and 20 of them fit on two lines.
            "  n_h realized: "
            + " ".join(f"{h}:{n}" for h, n in sorted(self.per_stratum.items())),
            "  provenance: "
            + ", ".join(f"{k} {v:,}" for k, v in sorted(self.provenance_counts.items())),
            f"  storage: {self.n_shards} shard(s), {self.shard_bytes / 1e9:.2f} GB",
            f"  land mean/std: {self.global_mean:.2f} / {self.global_std:.2f} m",
            # s/COG, because that is the unit E1's projection is stated in and it is
            # budget-independent — a tier-specific "hours" figure would not compare.
            f"  wall clock: {self.wall_s / 3600:.2f} h = {self.s_per_cog:.3f} s/COG "
            f"over {self.n_cogs_total:,} COGs "
            f"({self.n_cogs:,} read this run, {self.n_skipped:,} resumed past)",
            *(
                [f"  dataset_version: {self.dataset_version}"]
                if self.dataset_version
                else []
            ),
        ]


# --- The driver ---------------------------------------------------------------


def _tile_record(
    outcome: WindowOutcome,
    selected: SelectedWindow,
    catalog_row: dict,
    ref,
) -> manifest.TileRecord:
    """Assemble one v2 manifest row from its three sources (ADR-0002 §3a).

    Materialization owns the storage join and the land stats, the *selection list* owns
    ``stratum_id`` and ``provenance``, and the *catalog* owns everything else — copied
    in under the catalog's own column names, so the manifest lines up column-for-column
    with the frame it came from.
    """
    return manifest.TileRecord(
        tile_id=manifest.make_tile_id(selected.stem, selected.win_row0, selected.win_col0),
        shard_id=ref.shard_id,
        row_index=ref.row_index,
        source_cog=selected.stem,
        win_row0=selected.win_row0,
        win_col0=selected.win_col0,
        lon=catalog_row["lon"],
        lat=catalog_row["lat"],
        lat_band=catalog_row["band"],
        geocell_id=catalog_row["geocell"],
        mean=outcome.mean,
        std=outcome.std,
        land_frac=outcome.land_frac,
        stratum_id=selected.stratum_id,
        provenance=selected.provenance,
        slope_mean=catalog_row["slope_mean"],
        local_relief=catalog_row["local_relief"],
        texture=catalog_row["texture"],
        elevation_relief_ratio=catalog_row["elevation_relief_ratio"],
        mean_elevation=catalog_row["mean_elevation"],
        geomorphon_entropy=catalog_row["geomorphon_entropy"],
        edm_edited_frac=catalog_row["edm_edited_frac"],
        flm_fill_frac=catalog_row["flm_fill_frac"],
        wbm_water_frac=catalog_row["wbm_water_frac"],
        water_affected=catalog_row["water_affected"],
        coastline=catalog_row["coastline"],
        content_hash=ref.content_hash,
        split=None,  # filled after the build by `dataset.splits` (#77)
    )


def _read_manifest_records(path: Path, limit: int) -> list[manifest.TileRecord]:
    """Read committed manifest rows back as :class:`TileRecord`s, truncated to ``limit``.

    ``limit`` is the shard watermark's row count: anything past it was written after the
    last state commit and is disowned.
    """
    import pyarrow.parquet as pq

    if not path.exists():
        if limit:
            raise FileNotFoundError(f"state claims {limit} manifest rows but {path} is gone")
        return []
    rows = pq.read_table(path).to_pylist()
    if len(rows) < limit:
        raise ValueError(
            f"{path} holds {len(rows)} rows but the state commits {limit} — "
            "the manifest is behind its own checkpoint"
        )
    return [manifest.TileRecord(**row) for row in rows[:limit]]


def _assert_ledger_matches_manifest(
    ledger_rows: list[dict], records: list[manifest.TileRecord]
) -> None:
    """#50's invariant: ``status == 'ok'`` in the ledger *is* the manifest, exactly."""
    ok = {r["window_id"] for r in ledger_rows if r["status"] == "ok"}
    written = {make_window_id(r.source_cog, r.win_row0, r.win_col0) for r in records}
    # (composed via the same geometry helper `window_id_of` uses — one definition)
    if ok != written:
        raise AssertionError(
            "ledger/manifest disagree: "
            f"{len(ok - written)} ok row(s) with no tile, "
            f"{len(written - ok)} tile(s) with no ok row"
        )


def global_stats(land_sum: float, land_sqsum: float, land_count: int) -> tuple[float, float]:
    """Corpus-wide land mean/std from the streamed accumulators (ADR-0003 §3).

    Public because Gate F's ``F8_normalization_statistic`` has to *reproduce* the
    recorded statistic from the shipped tiles, and reproducing it means running this
    arithmetic rather than a second version of it — a re-derivation that rounded
    differently would report a Defect on a correct corpus.

    Plain sums rather than a pooled Welford merge: the sum of squares peaks near 2e17 at
    the debug tier and 4.9e16 at the scale tier — both well inside float64's exact-integer
    range times its 1e-16 relative error — and the variance is ~1e4 against a mean square
    of ~8e7, four digits of cancellation out of sixteen. Ten times the tiles moves that
    by one digit, so the simplicity keeps outrunning the twelve digits Welford would buy.
    """
    if land_count == 0:
        return 0.0, 0.0
    mean = land_sum / land_count
    var = max(0.0, land_sqsum / land_count - mean * mean)
    return mean, float(np.sqrt(var))


def _shard_stats(shard_dir: Path) -> tuple[int, int]:
    paths = sorted(shard_dir.glob("shard_*.npy"))
    return len(paths), sum(p.stat().st_size for p in paths)


def _tqdm(total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="cog", dynamic_ncols=True, smoothing=0.02)
    except ImportError:  # pragma: no cover
        return None


def run(
    *,
    cog_reader: CogReader | None = None,
    tier: str = "debug",
    selection_path: Path | str | None = None,
    catalog_dir: Path | str = Path("data/catalog_gated"),
    out_dir: Path | str | None = None,
    ledger_path: Path | str | None = None,
    centroids_path: Path | str = CENTROIDS_PATH,
    scaler_path: Path | str = SCALER_PATH,
    workers: int | None = None,
    executor: ExecutorKind = "thread",
    rows_per_shard: int = config.ROWS_PER_SHARD,
    checkpoint_every: int = CHECKPOINT_EVERY,
    read_attempts: int = READ_ATTEMPTS,
    retry_backoff_s: float = RETRY_BACKOFF_S,
    max_consecutive_read_errors: int = MAX_CONSECUTIVE_READ_ERRORS,
    limit_cogs: int | None = None,
    retry_read_errors: bool = False,
    restart: bool = False,
    code_version: str = "unknown",
    snapshot_date: str = "unknown",
    progress: Callable[[int, str], Any] | None = None,
) -> BuildResult:
    """Materialize the selection list into a corpus at ``out_dir``. Resumable.

    Re-running the same command after an interruption continues from the last
    checkpoint; re-running after a *complete* build is a no-op that just re-reports.
    ``restart=True`` throws the previous attempt away.

    ``executor`` defaults to ``"thread"`` for the same reason
    :func:`dataset.catalog.scan`'s does: this entry point takes an **injected** reader,
    and a process pool constrains it in ways an in-process caller has no reason to
    expect — it must pickle, and anything it mutates while reading mutates in the worker
    where the parent cannot see it. :func:`main` opts into ``"process"``, which is what
    E1 measured as 2.4x faster (``docs/research/e1-read-cost.md``).

    ``limit_cogs`` runs a prefix of the job list — the live-path smoke check. It leaves
    the build **incomplete**: no ``corpus_meta.json`` is written, because a partial
    corpus has no honest corpus-wide mean/std and no honest ``dataset_version``.

    ``retry_read_errors`` re-reads the COGs a previous run booked ``read_error`` and
    replaces their ledger rows. It is opt-in because the two cases are different
    claims: a COG that is genuinely gone should stay booked (that is
    ``accept_shortfall``), while one that lost a race with the network should not be
    written off silently. The operator reads the ledger's ``detail`` column and decides.

    ``tier`` names which corpus is being built and supplies the three paths that go
    with it; each may still be overridden individually. The resolution runs *here*
    rather than in :func:`main` so the cross-tier guard covers in-process callers too —
    see :class:`Tier`.
    """
    from functools import partial

    from .concurrency import ordered_futures

    started = time.perf_counter()
    elapsed_at_start = 0.0  # replaced from the state below; a resume keeps accumulating
    resolved = resolve_tier(
        tier, selection_path=selection_path, out_dir=out_dir, ledger_path=ledger_path
    )
    _assert_tier_paths_agree(resolved)
    selection_path = resolved.selection_path
    out_dir = Path(resolved.out_dir)
    ledger_path = Path(resolved.ledger_path)
    shard_dir = out_dir / SHARDS_DIR
    manifest_path = out_dir / MANIFEST_FILE
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_selection(selection_path)
    n_selected = sum(len(j.windows) for j in jobs)
    fingerprints = Fingerprints.collect(
        selection_path=Path(selection_path),
        catalog_dir=Path(catalog_dir),
        centroids_path=Path(centroids_path),
        scaler_path=Path(scaler_path),
    )

    state = None if restart else read_state(out_dir)
    if state is not None:
        if state.selection_md5 != fingerprints.selection_list_md5:
            raise ValueError(
                f"{out_dir / STATE_FILE} was written for a different selection list "
                f"(md5 {state.selection_md5[:8]}… vs {fingerprints.selection_list_md5[:8]}…). "
                "Point --out-dir somewhere else, or pass --restart to discard it."
            )
        if state.n_cogs_total != len(jobs):
            raise ValueError(
                f"{out_dir / STATE_FILE} covers {state.n_cogs_total} COGs but the "
                f"selection list groups into {len(jobs)} — refusing to resume."
            )
    if state is None:
        state = BuildState(
            selection_md5=fingerprints.selection_list_md5, n_cogs_total=len(jobs)
        )
        if restart:
            _clear_build(out_dir, ledger_path)

    elapsed_at_start = state.elapsed_s
    records = _read_manifest_records(manifest_path, state.shards.total_rows)
    ledger_rows = (
        read_ledger(ledger_path)[: state.n_windows_done]
        if state.n_windows_done and ledger_path.exists()
        else []
    )
    if len(ledger_rows) != state.n_windows_done:
        raise ValueError(
            f"{ledger_path} holds {len(ledger_rows)} rows but the state commits "
            f"{state.n_windows_done} — the ledger is behind its own checkpoint"
        )

    # Work is (job index, job) pairs so a retried COG — which sits *behind* the
    # frontier — can be told apart from one in front of it, and so neither the frontier
    # nor the outstanding-failure set has to be inferred from position alone.
    retry_indices: list[int] = []
    if retry_read_errors and state.read_error_cogs:
        retry_indices = sorted(state.read_error_cogs)
        retried_ids = {w.window_id for i in retry_indices for w in jobs[i].windows}
        # Their old ledger rows go, or the re-read would book each window twice.
        ledger_rows = [r for r in ledger_rows if r["window_id"] not in retried_ids]

    frontier = jobs[state.n_cogs_done :]
    if limit_cogs is not None:
        frontier = frontier[:limit_cogs]
    todo_pairs = [(i, jobs[i]) for i in retry_indices] + [
        (state.n_cogs_done + k, job) for k, job in enumerate(frontier)
    ]
    todo = [job for _, job in todo_pairs]
    n_skipped = state.n_cogs_done - len(retry_indices)
    n_done = 0
    outstanding_errors = set(state.read_error_cogs) - set(retry_indices)

    # Built whenever there is work *or* the corpus is unfinished — a run that died after
    # its last COG but before finalizing has an empty `todo` and still owns a shard that
    # has to be closed. `sealed()` covers the other direction: a retry into a build that
    # already finalized cannot append to its truncated trailing shard, so it starts a
    # fresh one rather than reopening that.
    writer = (
        ShardWriter(
            shard_dir,
            rows_per_shard=rows_per_shard,
            resume=state.shards.sealed() if state.complete else state.shards,
        )
        if todo or not state.complete
        else None
    )

    if todo:
        side = CatalogSideTable.load(
            catalog_dir, [w.window_id for j in todo for w in j.windows]
        )
        workers = default_workers(executor) if workers is None else workers
        selected_by_id = {w.window_id: w for j in todo for w in j.windows}
        bar = progress(len(todo), "materialize") if progress else None
        consecutive_failures = 0
        since_checkpoint = 0

        def commit() -> None:
            """The one commit point. State is written **last** — see the module docstring."""
            assert writer is not None
            state.shards = writer.checkpoint()
            state.n_cogs_done = frontier_done
            state.n_windows_done = len(ledger_rows)
            state.read_error_cogs = tuple(sorted(outstanding_errors))
            state.elapsed_s = elapsed_at_start + (time.perf_counter() - started)
            write_ledger(ledger_rows, ledger_path)
            _write_manifest_atomic(records, manifest_path)
            _write_json_atomic(out_dir / STATE_FILE, state.as_dict())

        work = partial(read_cog, cog_reader or _live_reader(), config.TILE_PX,
                       read_attempts, retry_backoff_s)
        frontier_done = state.n_cogs_done
        try:
            with make_executor(executor, workers) as ex:
                for k, (job, fut) in enumerate(
                    ordered_futures(ex=ex, fn=work, items=todo, in_flight=max(1, workers * 2))
                ):
                    # ordered_futures yields in input order, so the k-th result is the
                    # k-th pair — which is where its job index comes from.
                    job_index = todo_pairs[k][0]
                    outcomes = fut.result()
                    if all(o.status == "read_error" for o in outcomes):
                        outstanding_errors.add(job_index)
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_read_errors:
                            raise ReadOutage(
                                f"{consecutive_failures} consecutive COGs failed to read "
                                f"(last: {job.stem} — {outcomes[0].detail}). Nothing since "
                                "the last checkpoint is committed; fix the transport and "
                                "re-run to resume."
                            )
                    else:
                        outstanding_errors.discard(job_index)
                        consecutive_failures = 0

                    for outcome in outcomes:
                        selected = selected_by_id[outcome.window_id]
                        if outcome.status == "ok":
                            assert outcome.tile is not None
                            ref = writer.add(outcome.tile)
                            records.append(
                                _tile_record(outcome, selected, side.row(outcome.window_id), ref)
                            )
                            state.land_sum += outcome.land_sum
                            state.land_sqsum += outcome.land_sqsum
                            state.land_count += outcome.land_count
                        ledger_rows.append(
                            {
                                "window_id": outcome.window_id,
                                "stratum_id": selected.stratum_id,
                                "provenance": selected.provenance,
                                "status": outcome.status,
                                "detail": outcome.detail,
                            }
                        )
                    n_done += 1
                    since_checkpoint += 1
                    # Retried COGs are behind the frontier and must not push it.
                    frontier_done = max(frontier_done, job_index + 1)
                    if bar:
                        bar.update(1)
                    if since_checkpoint >= checkpoint_every:
                        commit()
                        since_checkpoint = 0
        finally:
            if bar:
                bar.close()
        # Deliberately outside the `finally`: a checkpoint is the only commit, so an
        # abort leaves the last one standing and the resumed run re-does the tail.
        commit()

    complete = state.n_cogs_done == len(jobs)
    meta_path = out_dir / CORPUS_META_FILE
    global_mean, global_std = global_stats(state.land_sum, state.land_sqsum, state.land_count)
    version = ""

    if complete:
        _assert_ledger_matches_manifest(ledger_rows, records)
        if state.complete and meta_path.exists() and n_done == 0:
            # Nothing changed, so nothing is rewritten — a no-op re-run must not
            # restamp a committed artifact with a fresh build timestamp. A retry that
            # actually re-read something does fall through and re-finalize.
            version = json.loads(meta_path.read_text())["dataset_version"]
        else:
            _finalize(out_dir, ledger_path, records, ledger_rows, state, writer=writer)
            version = manifest.write_corpus_meta(
                meta_path,
                rows=records,
                global_mean=global_mean,
                global_std=global_std,
                snapshot_date=snapshot_date,
                build_timestamp=_utc_now(),
                code_version=code_version,
                attrition_contract=config.ATTRITION_CONTRACT,
                normalization_stat_source=config.NORMALIZATION_STAT_SOURCE,
                **fingerprints.as_kwargs(),
            )["dataset_version"]

    n_shards, shard_bytes = _shard_stats(shard_dir)
    status_counts = Counter(r["status"] for r in ledger_rows)
    ok_rows = [r for r in ledger_rows if r["status"] == "ok"]
    return BuildResult(
        out_dir=out_dir,
        ledger_path=ledger_path,
        complete=complete,
        n_selected=n_selected,
        n_tiles=len(records),
        n_cogs=n_done,
        n_skipped=n_skipped,
        n_cogs_total=len(jobs),
        status_counts=dict(status_counts),
        per_stratum=dict(sorted(Counter(r["stratum_id"] for r in ok_rows).items())),
        provenance_counts=dict(sorted(Counter(r["provenance"] for r in ok_rows).items())),
        n_shards=n_shards,
        shard_bytes=shard_bytes,
        global_mean=global_mean,
        global_std=global_std,
        dataset_version=version,
        wall_s=elapsed_at_start + (time.perf_counter() - started),
    )


def _live_reader() -> CogReader:
    from .tiling import RasterioCogReader

    return RasterioCogReader()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_manifest_atomic(records: list[manifest.TileRecord], path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    manifest.write_manifest(records, tmp)
    _replace_atomic(tmp, path)


def _clear_build(out_dir: Path, ledger_path: Path) -> None:
    """Throw a previous attempt away — ``--restart``."""
    import shutil

    shutil.rmtree(out_dir / SHARDS_DIR, ignore_errors=True)
    for path in (
        out_dir / MANIFEST_FILE,
        out_dir / CORPUS_META_FILE,
        out_dir / STATE_FILE,
        ledger_path,
    ):
        path.unlink(missing_ok=True)


def _finalize(
    out_dir: Path,
    ledger_path: Path,
    records: list[manifest.TileRecord],
    ledger_rows: list[dict],
    state: BuildState,
    *,
    writer: ShardWriter | None,
) -> None:
    """Close the trailing shard and write the final artifacts — the last commit.

    Closing is what truncates the trailing shard from its write-time capacity to its
    real length, so no zero-padded rows masquerade as ocean tiles. It happens exactly
    once, at the end, which is why an interrupted run's shard stays appendable.
    """
    if writer is not None:
        writer.close()
        state.shards = writer.final_state()
    state.complete = True
    # Sorted only here, at the last write. The manifest's row order *is* the shard's
    # and cannot move, but the ledger's is free — so sorting it makes a corpus healed
    # by --retry-read-errors ship the same ledger as one built in a single clean pass.
    ledger_rows.sort(key=lambda r: r["window_id"])
    write_ledger(ledger_rows, ledger_path)
    _write_manifest_atomic(records, out_dir / MANIFEST_FILE)
    _write_json_atomic(out_dir / STATE_FILE, state.as_dict())


# --- Live driver --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python -m dataset.materialize`` — build the corpus from the selection list.

    The CLI the retired ``dataset.build`` / ``dataset.__main__`` left behind (#44), on
    the metric-basis pipeline instead of the TRI one. Re-run it to resume.
    """
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log = logging.getLogger("materialize")

    p = argparse.ArgumentParser(
        prog="python -m dataset.materialize",
        description="Read a tier's selected windows at full resolution into int16 "
        "shards + manifest + corpus card + materialization ledger (Phase E1 issue #51 "
        "at --tier debug, Phase F issue #72 at --tier scale). Resumable: re-run to "
        "continue.",
    )
    p.add_argument("--tier", choices=sorted(TIERS), default="debug",
                   help="which corpus to build: 'debug' (50k, #51) or 'scale' "
                        "(500k, #72). Sets --selection, --out-dir and --ledger as a "
                        "set — see the Tier docstring for why they cannot half-move")
    p.add_argument("--selection", type=Path, default=None,
                   help="override the tier's selection list")
    p.add_argument("--catalog-dir", type=Path, default=Path("data/catalog_gated"))
    p.add_argument("--out-dir", type=Path, default=None,
                   help="override the tier's corpus directory")
    p.add_argument("--ledger", type=Path, default=None,
                   help="override the tier's materialization ledger")
    p.add_argument("--centroids", type=Path, default=CENTROIDS_PATH)
    p.add_argument("--scaler", type=Path, default=SCALER_PATH)
    p.add_argument("--executor", choices=("process", "thread"), default="process",
                   help="read COGs in worker processes (default) or threads; "
                        "rasterio.open is GIL-serialized, so threads cap at ~2.8x "
                        "against processes' ~6.6x (docs/research/e1-read-cost.md)")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel whole-COG readers (default: 16 on this box — the "
                        "measured knee, see catalog.default_workers)")
    p.add_argument("--rows-per-shard", type=int, default=config.ROWS_PER_SHARD)
    p.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                   help="COGs between commits — also the resume granularity")
    p.add_argument("--read-attempts", type=int, default=READ_ATTEMPTS,
                   help="whole-COG read attempts before booking read_error")
    p.add_argument("--limit-cogs", type=int, default=None,
                   help="read only the first N COGs of the job list — the live-path "
                        "smoke check; leaves the build incomplete (no corpus card)")
    p.add_argument("--retry-read-errors", action="store_true",
                   help="re-read the COGs a previous run booked read_error and replace "
                        "their ledger rows; opt-in, because a COG that is genuinely "
                        "gone should stay booked (accept_shortfall, #50)")
    p.add_argument("--restart", action="store_true",
                   help="discard any previous attempt in --out-dir and start over")
    p.add_argument("--snapshot-date", default=None)
    args = p.parse_args(argv)

    from .provenance import git_sha, today

    # Resolved once here for the log line; `run` resolves again from the same inputs,
    # which is where the cross-tier guard actually fires.
    tier = resolve_tier(
        args.tier,
        selection_path=args.selection,
        out_dir=args.out_dir,
        ledger_path=args.ledger,
    )
    workers = args.workers if args.workers is not None else default_workers(args.executor)
    jobs = load_selection(tier.selection_path)
    log.info(
        "tier %s: %s windows over %s COGs -> %s (%d %s workers), ledger %s",
        tier.name, f"{sum(len(j.windows) for j in jobs):,}", f"{len(jobs):,}",
        tier.out_dir, workers, args.executor, tier.ledger_path,
    )

    result = run(
        tier=args.tier,
        selection_path=args.selection,
        catalog_dir=args.catalog_dir,
        out_dir=args.out_dir,
        ledger_path=args.ledger,
        centroids_path=args.centroids,
        scaler_path=args.scaler,
        workers=workers,
        executor=args.executor,
        rows_per_shard=args.rows_per_shard,
        checkpoint_every=args.checkpoint_every,
        read_attempts=args.read_attempts,
        limit_cogs=args.limit_cogs,
        retry_read_errors=args.retry_read_errors,
        restart=args.restart,
        code_version=git_sha(),
        snapshot_date=args.snapshot_date or today(),
        progress=_tqdm,
    )

    print()
    for line in result.summary_lines():
        print(line)
    print(f"  manifest: {result.manifest_path}")
    print(f"  ledger:   {result.ledger_path}")
    if result.complete:
        print(f"  card:     {result.meta_path}")
    else:
        print("  INCOMPLETE — re-run to resume (no corpus card written)")
    if result.status_counts.get("read_error"):
        print(
            f"  NOTE: {result.status_counts['read_error']:,} window(s) failed to read. "
            "Check the ledger's `detail` column; re-run with --retry-read-errors to "
            "re-read those COGs, or leave them booked."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
