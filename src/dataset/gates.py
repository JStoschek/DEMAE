"""The quality gates as filter columns over the raw catalog — Phase C5 (issue #37).

C3 ([#35](https://github.com/JStoschek/DEMAE/issues/35)) put 4,386,720 raw scored
windows on disk; C4 ([#36](https://github.com/JStoschek/DEMAE/issues/36)) locked the
thresholds against them. This module is the third step and the smallest: it evaluates
those thresholds over the stored columns and **appends the answers as boolean columns**,
producing the frame Phase D draws from without re-scoring a byte.

**Gating here is additive, never destructive.** Every raw row and every raw column
survives into the gated catalog; the only difference is eight more columns. That is the
standing preference the Phase-C map made binding ("gates are re-runnable filter
columns"), and it is what makes a threshold re-tunable: moving the water cap is a
re-run of this module over the same parts, not six hours of S3 reads. The `keep` column
is a convenience over the four drop gates, not a filtered frame.

**The thresholds are C4's, and they are inherited rather than re-derived**
([`docs/research/gate-attrition.md`](../../docs/research/gate-attrition.md) §3). Two of
them are worth knowing the provenance of before touching:

- ``degenerate`` at 3 m is **ADR-0002-bound**, not a free knob: §48–49 fix ε ≥ 3 m as
  what neutralises int16 quantisation banding, so lowering it needs that ADR amended.
- ``zero_fill`` at 0.75 is **inert at these settings** — all 790,152 windows it fires on
  are also caught by the water cap (100.00%, 0 exceptions in 4.39 M), so no value
  between 0.50 and 0.999 moves a single row. It is kept as the one ocean test that does
  not trust WBM — the mask C2 caught classifying a 3,656 m salt flat as water — and a
  future reader must not mistake the 0.75 for evidence. ``flm_fill_frac`` is inert
  identically (0 sole cause) and kept on the same reasoning.

**The two tags are not drops.** ``water_affected`` and ``coastline`` are computed over
the *whole* frame and describe windows rather than removing them; C4's headline counts
(159,011 and 40,815) are those tags intersected with ``keep``, which is how
:func:`attrition` reports them.

**NaN is not a pass** — the one place this module's arithmetic is not simply C4's.
``NaN > 0.30`` is ``False``, so a window whose masks could not be read would clear the
water and edited caps without either ever having been evaluated. Those windows are
flagged ``masks_missing`` and excluded from ``keep``. The shipped catalog has **zero**
of them (its two anomalies were whole-COG read failures, and an unmarked COG is
re-scanned rather than half-written), so this changes no count C4 published — it stops
a future run from quietly seeding Phase D's sampling frame with unknown quality. A NaN
``land_std`` is a different animal and is *not* missing data: it means the window has no
land pixel at all, which is why ``degenerate`` cannot fire on an ocean window and why
``zero_fill`` is the gate that catches it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:  # pyarrow is imported lazily — only the writer needs it
    import pyarrow as pa

from . import catalog

# --- The locked thresholds (C4 / #36, docs/research/gate-attrition.md §3) ------

#: ``exact_zero_frac >= 0.75`` — ADR-0001 §3 + Amendment 2. Inert at these settings
#: (0 sole cause); retained as the WBM-independent ocean backstop, not a tuned value.
ZERO_FILL_MIN: Final = 0.75
#: ``land_std < 3.0`` m — ADR-0002 §48–49, the int16 banding floor. The largest
#: marginal gate (8.37% sole cause) and the one that is not a free knob.
DEGENERATE_LAND_STD_M: Final = 3.0
#: ``wbm_water_frac > 0.30`` drops — plan §9 [project]. The operative ocean gate.
WATER_DROP_MAX: Final = 0.30
#: ``wbm_water_frac > 0.05`` tags ``water_affected`` — non-destructive, plan §9.
WATER_AFFECTED_MIN: Final = 0.05
#: ``edm_edited_frac > 0.25`` **or** ``flm_fill_frac > 0.25`` drops — plan §9 [project].
#: EDM carries it; FLM's half is inert (0 sole cause), kept as a provenance backstop.
EDITED_MAX: Final = 0.25

#: The four gates that remove a window from ``keep``, in C4 §3 table order.
DROP_GATES: Final = ("zero_fill", "degenerate", "water", "edited")
#: The two non-destructive tags. Computed frame-wide; reported against ``keep``.
TAG_COLUMNS: Final = ("water_affected", "coastline")
#: Every column gating adds. ``masks_missing`` is this module's own (see the docstring).
FLAG_COLUMNS: Final = (*DROP_GATES, "masks_missing", "keep", *TAG_COLUMNS)

#: The raw catalog columns every predicate reads. All six are stored (C3), which is
#: what makes gating a re-runnable filter rather than a re-score.
GATE_INPUTS: Final = (
    "exact_zero_frac",
    "land_std",
    "edm_edited_frac",
    "flm_fill_frac",
    "wbm1_frac",
    "wbm_water_frac",
)

#: Columns whose NaN means *unread*, not *no land*. See the module docstring.
_MASK_INPUTS: Final = ("edm_edited_frac", "flm_fill_frac", "wbm1_frac", "wbm_water_frac")

#: The gated catalog's column order: the raw frame unchanged, then the flags.
GATED_COLUMNS: Final = (*catalog.COLUMNS, *FLAG_COLUMNS)


# --- The predicates -----------------------------------------------------------


def gate_flags(columns: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Evaluate every gate and tag over a column frame — the whole of the arithmetic.

    ``columns`` maps at least :data:`GATE_INPUTS` to equal-length arrays (a dict of
    numpy arrays, or anything ``np.asarray`` accepts per column — a pyarrow column
    included). Returns one boolean array per :data:`FLAG_COLUMNS`, aligned row-for-row
    with the input.

    Comparisons are written so NaN falls to ``False`` on its own — no ``nan_to_num``,
    which would have to pick a direction per column and would silently invert a gate
    if a threshold ever moved past the sentinel. ``keep`` then subtracts
    ``masks_missing`` explicitly, so an unevaluated gate never reads as a passed one.

    Raises :class:`KeyError` naming the first missing input: a gate evaluated over a
    frame that lacks its column is a gate that has not run, and the cost of noticing
    that after a 4.4 M-row write is the whole pass.
    """
    missing = [name for name in GATE_INPUTS if name not in columns]
    if missing:
        raise KeyError(f"gate inputs missing from the frame: {', '.join(missing)}")
    col = {name: np.asarray(columns[name], dtype=np.float64) for name in GATE_INPUTS}

    with np.errstate(invalid="ignore"):
        zero_fill = col["exact_zero_frac"] >= ZERO_FILL_MIN
        degenerate = col["land_std"] < DEGENERATE_LAND_STD_M
        water = col["wbm_water_frac"] > WATER_DROP_MAX
        edited = (col["edm_edited_frac"] > EDITED_MAX) | (col["flm_fill_frac"] > EDITED_MAX)
        water_affected = col["wbm_water_frac"] > WATER_AFFECTED_MIN
        # C2 / #34: sea-touching, and no lower edge (C4 §3.5 — `lo` bought no purity).
        coastline = (col["exact_zero_frac"] < ZERO_FILL_MIN) & (col["wbm1_frac"] > 0.0)

    masks_missing = np.zeros_like(zero_fill)
    for name in _MASK_INPUTS:
        masks_missing |= np.isnan(col[name])

    dropped = zero_fill | degenerate | water | edited
    return {
        "zero_fill": zero_fill,
        "degenerate": degenerate,
        "water": water,
        "edited": edited,
        "masks_missing": masks_missing,
        "keep": ~dropped & ~masks_missing,
        "water_affected": water_affected,
        "coastline": coastline,
    }


def apply_gates(table: pa.Table) -> pa.Table:
    """Append the :data:`FLAG_COLUMNS` to a catalog table — additive and idempotent.

    Row count and raw column values are untouched; re-gating an already-gated table
    recomputes the flags in place rather than duplicating them, so the module can be
    re-run over its own output after a threshold moves.
    """
    import pyarrow as pa

    flags = gate_flags({name: table.column(name) for name in GATE_INPUTS})
    out = table
    for name in FLAG_COLUMNS:
        column = pa.array(flags[name], type=pa.bool_())
        index = out.schema.get_field_index(name)
        if index >= 0:
            out = out.set_column(index, pa.field(name, pa.bool_()), column)
        else:
            out = out.append_column(pa.field(name, pa.bool_()), column)
    return out.select(list(GATED_COLUMNS))


# --- Attrition accounting -----------------------------------------------------


@dataclass(frozen=True)
class GateAttrition:
    """What one gate costs: how often it fires, and how often it is the only cause.

    ``sole_cause`` is the number the threshold discussion turns on — remove the gate
    and exactly these windows return. C4's headline finding is a gate with 790,152
    fires and 0 sole cause.
    """

    gate: str
    fires: int
    sole_cause: int


@dataclass(frozen=True)
class AttritionReport:
    """The attrition table for one frame: per-gate cost, keep-rate, tag counts."""

    n_rows: int
    n_kept: int
    n_masks_missing: int
    n_water_affected: int  # over the kept frame
    n_coastline: int  # over the kept frame
    gates: tuple[GateAttrition, ...]

    @property
    def keep_rate(self) -> float:
        """Kept fraction of the **whole** frame (the raw catalog is the denominator)."""
        return self.n_kept / self.n_rows if self.n_rows else float("nan")

    def as_dict(self) -> dict[str, Any]:
        """JSON-shaped, for the gated catalog's meta sidecar."""
        return {
            "n_rows": self.n_rows,
            "n_kept": self.n_kept,
            "keep_rate": self.keep_rate,
            "n_masks_missing": self.n_masks_missing,
            "n_water_affected": self.n_water_affected,
            "n_coastline": self.n_coastline,
            "gates": [
                {"gate": g.gate, "fires": g.fires, "sole_cause": g.sole_cause}
                for g in self.gates
            ],
        }


def attrition(flags: Mapping[str, np.ndarray]) -> AttritionReport:
    """Count fires and sole causes per drop gate, plus the tags over the kept frame.

    A gate is the *sole cause* of a window when it fires and no other drop gate does —
    ``masks_missing`` excluded, since it is not one of C4's four and treating it as a
    cover would understate a real gate's cost.

    The tags are counted **against ``keep``**, matching how C4 reports them (159,011
    ``water_affected`` = 5.21% *of kept*), because a tag on a dropped window describes
    nothing that will ever be sampled.
    """
    fired = {name: np.asarray(flags[name], dtype=bool) for name in DROP_GATES}
    keep = np.asarray(flags["keep"], dtype=bool)
    n_rows = int(keep.size)

    rows = []
    for name in DROP_GATES:
        others = np.zeros(n_rows, dtype=bool)
        for other in DROP_GATES:
            if other != name:
                others |= fired[other]
        rows.append(
            GateAttrition(
                gate=name,
                fires=int(fired[name].sum()),
                sole_cause=int((fired[name] & ~others).sum()),
            )
        )

    return AttritionReport(
        n_rows=n_rows,
        n_kept=int(keep.sum()),
        n_masks_missing=int(np.asarray(flags["masks_missing"], dtype=bool).sum()),
        n_water_affected=int((np.asarray(flags["water_affected"], dtype=bool) & keep).sum()),
        n_coastline=int((np.asarray(flags["coastline"], dtype=bool) & keep).sum()),
        gates=tuple(rows),
    )


def _merge(reports: list[AttritionReport]) -> AttritionReport:
    """Sum per-part reports into one. Every count is additive over disjoint parts."""
    per_gate = {name: [0, 0] for name in DROP_GATES}
    for report in reports:
        for gate in report.gates:  # keyed by name, not position — a report that
            per_gate[gate.gate][0] += gate.fires  # ordered its gates differently
            per_gate[gate.gate][1] += gate.sole_cause  # would otherwise sum crosswise
    return AttritionReport(
        n_rows=sum(r.n_rows for r in reports),
        n_kept=sum(r.n_kept for r in reports),
        n_masks_missing=sum(r.n_masks_missing for r in reports),
        n_water_affected=sum(r.n_water_affected for r in reports),
        n_coastline=sum(r.n_coastline for r in reports),
        gates=tuple(
            GateAttrition(gate=name, fires=fires, sole_cause=sole)
            for name, (fires, sole) in per_gate.items()
        ),
    )


# --- The persisted gated catalog ----------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """What one :func:`write_gated` produced, and what the gated frame now holds.

    The row and keep counts are read off :attr:`report` rather than stored beside it —
    two fields holding the same count are two fields that can disagree.
    """

    out_dir: Path
    n_parts: int
    report: AttritionReport

    @property
    def n_rows(self) -> int:
        return self.report.n_rows

    @property
    def n_kept(self) -> int:
        return self.report.n_kept


def thresholds() -> dict[str, float]:
    """The locked threshold set, as the meta sidecar records it."""
    return {
        "zero_fill_min": ZERO_FILL_MIN,
        "degenerate_land_std_m": DEGENERATE_LAND_STD_M,
        "water_drop_max": WATER_DROP_MAX,
        "water_affected_min": WATER_AFFECTED_MIN,
        "edited_max": EDITED_MAX,
    }


def _source_meta(in_dir: Path) -> dict[str, Any]:
    """The raw catalog's fingerprint, echoed forward so the gated frame is traceable."""
    path = in_dir / catalog.META_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_gated(
    in_dir: Path | str,
    out_dir: Path | str,
    *,
    progress: bool = False,
) -> GateResult:
    """Gate the raw catalog at ``in_dir`` into a new gated catalog at ``out_dir``.

    **Part for part, one at a time.** The raw frame is ~190 MB of zstd Parquet that
    expands past 700 MB in memory, and there is no reason to hold all of it: the gates
    are row-local, so a part gated in isolation is identical to the same rows gated as
    part of the whole. That also keeps the output's part boundaries aligned with the
    input's, so a row's part index means the same thing on both sides of the pass.

    The output directory is **cleared of stale parts first**. Re-running after a
    threshold moves would otherwise leave the previous run's files to be concatenated
    with the new ones by :func:`dataset.catalog.read_catalog`, which reads a directory
    rather than a manifest — a frame silently holding both answers.

    Writing in place is refused. The raw catalog cost a six-hour global pass and is the
    denominator every future threshold is measured against; the gated frame is derived
    and can be rebuilt from it in minutes.
    """
    import pyarrow.parquet as pq

    in_dir, out_dir = Path(in_dir), Path(out_dir)
    if in_dir.resolve() == out_dir.resolve():
        raise ValueError(
            f"refusing to gate {in_dir} in place — the raw catalog is the denominator "
            "every threshold is measured against; write the gated frame elsewhere"
        )

    parts = sorted((in_dir / catalog.PARTS_DIR).glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no part files under {in_dir / catalog.PARTS_DIR}")

    out_parts = out_dir / catalog.PARTS_DIR
    out_parts.mkdir(parents=True, exist_ok=True)
    for stale in out_parts.glob("part-*"):  # the glob covers a crashed run's *.tmp too
        stale.unlink()

    reports: list[AttritionReport] = []
    for i, part in enumerate(parts):
        gated = apply_gates(pq.read_table(part))
        reports.append(attrition({name: gated.column(name) for name in FLAG_COLUMNS}))
        target = catalog.part_path(out_dir, i)
        # Temp file + rename, as `catalog.write_part` does it: a half-written Parquet
        # under its final name would be read as data by the next `read_catalog`.
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        try:
            pq.write_table(gated, tmp, compression="zstd")
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        if progress:
            print(f"  {part.name}: {gated.num_rows:,} rows, {reports[-1].n_kept:,} kept")

    result = GateResult(out_dir=out_dir, n_parts=len(parts), report=_merge(reports))
    write_meta(out_dir, result, source=_source_meta(in_dir))
    return result


def write_meta(out_dir: Path, result: GateResult, *, source: Mapping[str, Any]) -> Path:
    """Write the gated catalog's fingerprint — thresholds, counts, source provenance.

    Mirrors :func:`dataset.catalog.write_meta`, with the fields that only make sense
    once the frame is gated: the locked threshold set, the attrition table, and the raw
    catalog's own meta echoed under ``source_catalog`` so the gated frame names the
    scoring run it derives from.
    """
    from datetime import datetime, timezone

    meta = {
        "schema_version": source.get("schema_version"),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gated": True,
        "gate_source": "docs/research/gate-attrition.md (C4, issue #36)",
        "thresholds": thresholds(),
        "columns": list(GATED_COLUMNS),
        "flag_columns": list(FLAG_COLUMNS),
        "n_rows": result.n_rows,
        "n_kept": result.n_kept,
        "n_parts": result.n_parts,
        "attrition": result.report.as_dict(),
        "source_catalog": dict(source),
    }
    path = out_dir / catalog.META_FILE
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


# --- Live driver --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python -m dataset.gates`` — apply the locked gates to the raw catalog.

    Minutes, not hours: this reads Parquet the global pass already wrote and touches
    no COG. Re-run it whenever a threshold moves.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m dataset.gates",
        description="Apply the C4-locked quality gates to the raw candidate catalog as "
        "boolean filter columns (Phase C5, issue #37). Non-destructive: every raw row "
        "and column survives.",
    )
    p.add_argument("--in-dir", type=Path, default=Path("data/catalog"))
    p.add_argument("--out-dir", type=Path, default=Path("data/catalog_gated"))
    args = p.parse_args(argv)

    result = write_gated(args.in_dir, args.out_dir, progress=True)
    report = result.report
    print(
        f"\nGated catalog at {result.out_dir}: {result.n_rows:,} rows in "
        f"{result.n_parts} part(s).\n"
        f"  kept {result.n_kept:,} ({report.keep_rate:.2%}); "
        f"water_affected {report.n_water_affected:,}; coastline {report.n_coastline:,}"
    )
    for gate in report.gates:
        print(f"    {gate.gate:<11} fires {gate.fires:>9,}  sole cause {gate.sole_cause:>9,}")
    if report.n_masks_missing:
        print(f"    masks_missing {report.n_masks_missing:,} (excluded from keep)")
    print(f"  meta: {result.out_dir / catalog.META_FILE}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
