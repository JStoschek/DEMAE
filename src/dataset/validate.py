"""Gate E's pre-registered criteria, as executable rules — Phase E2 (issue #52).

The bar itself is not here. It was written before the corpus existed and committed to
[`docs/research/e2-gate-bar.json`](../../docs/research/e2-gate-bar.json) by issue #47,
precisely so the review could not rationalize whatever the numbers turned out to be.
This module is the other half of that arrangement: the rules that read that file and
turn a finished corpus into a pass or a fail against it. Nothing here decides a
threshold, and nothing here renders the verdict — the roll-up from criteria to a Gate E
verdict is #54's, not this module's.

The split against ``notebooks/e2_validation_report.py`` is the usual one: this module
is pure and takes arrays and counts, the notebook does the Parquet reads, the joins and
the figures. That is what lets the rules be tested against hand-built four-row frames
instead of against the corpus they are meant to judge, and what lets Phase F re-run
them at 500 k without a single 50 k constant in the way.

## Where the numbers come from

The bar states some thresholds structurally (``permutations: 10000``) and some only in
prose (``"every realized n_h >= 500"``). :func:`read_bar` extracts the first kind and,
for the second, checks that the committed prose still contains the number
:class:`Thresholds` carries. So the constants live in one place a reader can see, and
an edit to the committed bar that this code has not followed raises on load rather than
quietly measuring against a threshold nobody registered.

## Two failure classes, never merged

Every criterion is pre-assigned a class by the bar. A **Defect** means the corpus does
not implement its own committed inputs — fix the code, re-run, stay in the map. A
**Finding** means the corpus is faithful and the *design* is wrong — record it, stop,
open a fresh map with the implicated knob. A Finding therefore has to name a knob from
the plan's §215-228 table; :meth:`Outcome.payload` refuses to emit one that does not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from .catalog import FROZEN_METRICS

#: The committed bar, relative to the repo root. The notebook passes an explicit path.
BAR_PATH: Final = Path("docs/research/e2-gate-bar.json")

# --- The registered numbers ---------------------------------------------------
#
# Every threshold below is #47's, and every one carries the bar text it was registered
# in. They are module constants in the shape ``gates.py`` and ``centroids.py`` use for
# their locked values, so a reader sees each number beside its provenance rather than
# buried in a function body — and :func:`read_bar` refuses to load a bar whose prose no
# longer states them (see :data:`_PROSE_ANCHORS`).

#: E2.1b: percentile of the permutation null a metric's observed W1 must exceed to fire.
#: Bar: ``"any one of the 6 metrics' observed W1 exceeds the 99th percentile"``. The
#: permutation *count* is structural in the bar and is read from it, not from here.
NULL_PERCENTILE: Final = 99.0

#: E2.2c absolute limb. Bar: ``"every realized n_h >= 500"`` — #42's ``max(500, 1%)``
#: clamp, documented structurally inert; a realized count below it means attrition made
#: that inert clamp retroactively binding.
THIN_ABSOLUTE: Final = 500

#: E2.2c relative limb, both halves. Bar: ``"a stratum trips if BOTH its drop rate >= 3x
#: the corpus-wide drop rate AND it loses >= 2% of its allocation"``.
THIN_RATE_RATIO: Final = 3.0
THIN_LOSS_FRAC: Final = 0.02

#: E2.2d under ``accept_shortfall``. Bar: ``"realized_coastal / realized_total >=
#: 0.01337"`` — coastline's within-frame rate (40,815 / 3,052,735), not the count 670.
COASTAL_MIN_RATE: Final = 0.01337

#: E2.3's one tripwire. Bar: ``"no single COG exceeds 64 windows"`` — #42's retired
#: ADR-0001-era per-COG cap, inert at debug tier and kept only to catch concentration.
PER_COG_CAP: Final = 64

#: E2.5's conjunction. Bar: ``"fire if BOTH: the gate's within-stratum drop rate >= 3x
#: that same gate's corpus-wide drop rate, AND that gate removes >= 10% of the stratum's
#: would-be (pre-gate) membership"``.
BIAS_RATE_RATIO: Final = 3.0
BIAS_LOSS_FRAC: Final = 0.10

#: The frozen metric vector, re-exported so a caller needs one import (ADR-0004).
FROZEN: Final = FROZEN_METRICS

#: Quality gate -> the plan §215-228 knob a firing implicates, verbatim from that table.
#: Only the two *cap* gates appear there, which is what the bar's E2.5 clause anticipated
#: ("WBM water cap OR EDM/FLM cap, whichever gate fired"). ``zero_fill`` and
#: ``degenerate`` are pixel screens, not caps, and have no row in the knob table — so a
#: firing on one of them is a Finding the bar cannot hand a knob to. That gap is surfaced
#: as ``pairs_without_a_named_knob`` rather than papered over with an invented knob.
GATE_KNOBS: Final = {"water": "WBM water cap", "edited": "EDM/FLM cap"}


# --- The committed bar --------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """Every number a criterion gates on, with the bar text each was registered in.

    Held as a frozen dataclass rather than read ad hoc out of the JSON at each call
    site so that a rule function is a pure function of numbers — testable against
    values the bar never mentions, which is how the conjunctions get exercised at all.
    """

    permutations: int
    null_percentile: float
    thin_absolute: int
    thin_rate_ratio: float
    thin_loss_frac: float
    coastal_min_rate: float
    per_cog_cap: int
    bias_rate_ratio: float
    bias_loss_frac: float


#: ``(json path into the bar, phrase that must be present)`` for every threshold the bar
#: states only in prose. Checked on load — see the module docstring. Matched with word
#: boundaries, not as bare substrings: ``"500"`` must not be satisfied by a bar edited to
#: say ``5000``, nor ``"2%"`` by one edited to say ``12%``, which is precisely the edit
#: the tripwire exists to catch.
_PROSE_ANCHORS: Final = (
    (("criteria", "E2.1b_drop_selectivity", "branch", "accept_shortfall", "threshold", "fire_if"),
     "99th percentile"),
    (("criteria", "E2.2c_thin_stratum", "conditions", "absolute"), "500"),
    (("criteria", "E2.2c_thin_stratum", "conditions", "relative"), "3x"),
    (("criteria", "E2.2c_thin_stratum", "conditions", "relative"), "2%"),
    (("criteria", "E2.2d_coastal_floor", "branch", "accept_shortfall", "test"), "0.01337"),
    (("criteria", "E2.3_coverage", "guard", "test"), "64"),
    (("criteria", "E2.5_gate_attrition_bias", "test"), "3x"),
    (("criteria", "E2.5_gate_attrition_bias", "test"), "10%"),
)


@dataclass(frozen=True)
class Bar:
    """The parsed pre-registered bar: its thresholds, its inputs, its raw text."""

    doc: dict[str, Any]
    thresholds: Thresholds
    path: Path

    @property
    def committed_inputs(self) -> dict[str, Any]:
        return self.doc["committed_inputs"]

    @property
    def contracts(self) -> tuple[str, ...]:
        """The attrition-contract branches the bar itself pre-committed."""
        return tuple(self.doc["attrition_contract_branch"]["branches"])

    def live_branch(self, contract: str) -> str:
        """Resolve the corpus card's ``attrition_contract`` against the bar's branches.

        Refuses a contract string the bar never pre-committed, and reads the permitted
        set out of the committed file rather than from a constant here — that is the
        whole point of the conditional pre-registration: #50 was allowed to *select* a
        branch, not to invent one after the fact.
        """
        if contract not in self.contracts:
            raise ValueError(
                f"attrition contract {contract!r} is unregistered — {self.path} "
                f"pre-committed {self.contracts}. A new branch is a new pre-registration, "
                "not a report detail."
            )
        return contract


def read_bar(path: Path | str = BAR_PATH) -> Bar:
    """Load the committed bar, refusing one whose prose no longer states these numbers.

    The anchor check is deliberately crude — one word-bounded phrase per registered
    number. It cannot verify that the *sentence* still means what it meant, but it does
    catch the failure that matters: someone moves a threshold in the bar and this module
    keeps measuring against the old one, which would make the pre-registration
    decorative.
    """
    path = Path(path)
    doc = json.loads(path.read_text())
    thresholds = Thresholds(
        permutations=int(
            doc["criteria"]["E2.1b_drop_selectivity"]["branch"]["accept_shortfall"]["threshold"][
                "permutations"
            ]
        ),
        null_percentile=NULL_PERCENTILE,
        thin_absolute=THIN_ABSOLUTE,
        thin_rate_ratio=THIN_RATE_RATIO,
        thin_loss_frac=THIN_LOSS_FRAC,
        coastal_min_rate=COASTAL_MIN_RATE,
        per_cog_cap=PER_COG_CAP,
        bias_rate_ratio=BIAS_RATE_RATIO,
        bias_loss_frac=BIAS_LOSS_FRAC,
    )
    for keys, anchor in _PROSE_ANCHORS:
        node: Any = doc
        for key in keys:
            node = node[key]
        if not re.search(rf"(?<![\w.]){re.escape(anchor)}(?![\w.])", str(node)):
            raise ValueError(
                f"{path}: {'.'.join(keys)} no longer states {anchor!r} — the committed "
                "bar and dataset.validate's registered constants have drifted. Reconcile "
                "them in a dated commit before running the report."
            )
    return Bar(doc=doc, thresholds=thresholds, path=path)


# --- What a criterion returns -------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """One criterion's result: did its fail condition fire, and on what numbers.

    ``fired`` is ``None`` for the bar's *published* criteria (the pool ECDF, the
    coverage map), which carry no threshold at all. Reporting them through the same
    type keeps the published diagnostics beside the gating ones in the artifact, rather
    than in a second shape a reader has to reconcile.
    """

    criterion: str
    gates: bool
    failure_class: str
    fired: bool | None
    measured: dict[str, Any] = field(default_factory=dict)
    measured_miss: str | None = None
    implicated_knob: str | None = None

    @property
    def passed(self) -> bool | None:
        return None if self.fired is None else not self.fired

    def payload(self, evidence_pointer: str) -> dict[str, str | None] | None:
        """The bar's four-field handoff, or ``None`` if the criterion did not fire.

        A Finding must name a knob — the bar says a criterion that cannot name one from
        the plan's table is a Defect, not a Finding — so an unnamed knob raises here
        rather than reaching a fresh map as an empty field.
        """
        if not self.fired:
            return None
        klass = self.failure_class.capitalize()
        if klass == "Finding" and not self.implicated_knob:
            raise ValueError(
                f"{self.criterion} fired as a Finding without an implicated knob; the bar "
                "says such a criterion is a Defect, not a Finding"
            )
        return {
            "implicated_knob": self.implicated_knob,
            "measured_miss": self.measured_miss or "",
            "evidence_pointer": evidence_pointer,
            "class": klass,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "gates": self.gates,
            "class": self.failure_class,
            "fired": self.fired,
            "passed": self.passed,
            "measured_miss": self.measured_miss,
            "implicated_knob": self.implicated_knob,
            "measured": self.measured,
        }


# --- Small shared shapes ------------------------------------------------------


def _strata(*maps: Mapping[int, int]) -> list[int]:
    """The union of stratum ids across several count maps, as sorted ints.

    Every count rule takes two or more sparse per-stratum maps and has to iterate their
    union — a stratum missing from one side is a zero there, never a skipped row.
    """
    keys: set[int] = set()
    for m in maps:
        keys |= {int(h) for h in m}
    return sorted(keys)


def _ratio(rate: float, baseline: float) -> float:
    """``rate / baseline``, with the two degenerate cases named rather than raised.

    A zero baseline means the gate (or attrition) fired nowhere corpus-wide. A nonzero
    rate against it is unboundedly concentrated, so ``inf`` — which trips the ratio limb
    and leaves the conjunction's second limb to decide whether the bite is substantial.
    A zero rate against it is 0: nothing happened anywhere, including here.
    """
    if baseline > 0:
        return rate / baseline
    return float("inf") if rate > 0 else 0.0


# --- Wasserstein-1 ------------------------------------------------------------


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """The 1-D earth-mover distance ``∫|F_a - F_b| dx`` between two empirical samples.

    Hand-rolled rather than taken from ``scipy.stats``: scipy is only a transitive
    dependency here (via scikit-learn, which the reproducible read path does not use),
    and the vectorized form below is what :func:`drop_selectivity` needs to run 10,000
    permutations without 10,000 sorts.

    Non-finite values are dropped — a NaN metric means the window had no land to
    measure, and there is no sensible place to put it on the real line.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.sort(np.concatenate([a, b]))
    # Step widths between consecutive support points; the CDFs are constant on each.
    widths = np.diff(grid)
    cdf_a = np.searchsorted(np.sort(a), grid[:-1], side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), grid[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * widths))


def _partition_w1(order_values: np.ndarray, member: np.ndarray) -> np.ndarray:
    """W1 between a subset and its complement, for many subsets of one sorted sample.

    ``order_values`` is the sample sorted ascending; ``member`` is a ``(p, n)`` boolean
    matrix, one row per subset, indexed in that sorted order. Returns ``p`` distances.

    The two empirical CDFs are step functions on the *same* support (the subset and its
    complement partition the sample), so both cumulative counts come from one cumsum
    and the whole permutation batch is three array ops. This is the form that makes the
    bar's 10,000-permutation null affordable: the alternative sorts 10,000 times per
    metric.
    """
    widths = np.diff(order_values)
    cum_in = np.cumsum(member, axis=1, dtype=np.int32)[:, :-1]
    n_in = cum_in[:, -1:] + member[:, -1:].astype(np.int32)
    total = np.arange(1, order_values.size, dtype=np.int32)[None, :]
    n_out = order_values.size - n_in
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.abs(cum_in / n_in - (total - cum_in) / n_out)
    return (gap * widths[None, :]).sum(axis=1)


# --- E2.1a metric fidelity ----------------------------------------------------


def metric_fidelity(
    manifest: Mapping[str, np.ndarray],
    catalog_: Mapping[str, np.ndarray],
    *,
    storage_dtype: np.dtype | str = np.float64,
) -> Outcome:
    """Did E1 write the rows it was told to write (bar E2.1a, Defect, exact).

    Both frames are already joined and row-aligned by ``window_id`` — the join is the
    notebook's job. Equality is exact **after the manifest's declared storage cast**,
    which is the tolerance the bar registers: the catalog carries float64, and if a
    manifest schema ever narrows the frozen six to float32, the honest comparison is at
    that width rather than one that reports every rounded value as a Defect. Today's v2
    schema (#48) stores float64, so ``storage_dtype`` is the identity and the comparison
    is bitwise; the caller passes the manifest's own dtype so a future narrowing is
    followed rather than discovered.

    An empty comparison **fires**. A criterion this cheap has no honest reason to see
    zero rows, and "nothing mismatched because nothing was compared" is exactly how a
    broken join would sneak a pass through a Defect gate.
    """
    dtype = np.dtype(storage_dtype)
    per_metric: dict[str, int] = {}
    mismatched = np.zeros(0, dtype=bool)
    for name in FROZEN:
        left = np.asarray(manifest[name]).astype(dtype, copy=False)
        right = np.asarray(catalog_[name]).astype(dtype, copy=False)
        # A NaN copied faithfully is a faithful copy.
        differs = (left != right) & ~(np.isnan(left) & np.isnan(right))
        per_metric[name] = int(differs.sum())
        mismatched = differs if mismatched.size == 0 else (mismatched | differs)
    n_rows = int(mismatched.size)
    n_bad = int(mismatched.sum())
    problems = []
    if n_rows == 0:
        problems.append("no rows were compared — the manifest/catalog join produced nothing")
    if n_bad:
        problems.append(
            f"{n_bad} of {n_rows} materialized rows carry a frozen-metric value that differs "
            f"from the catalog's for the same window_id"
        )
    return Outcome(
        criterion="E2.1a_metric_fidelity",
        gates=True,
        failure_class="defect",
        fired=bool(problems),
        measured={
            "n_rows_compared": n_rows,
            "n_mismatched_rows": n_bad,
            "storage_dtype": dtype.name,
            "per_metric_mismatches": per_metric,
        },
        measured_miss="; ".join(problems) or None,
    )


# --- E2.1b drop selectivity ---------------------------------------------------


def drop_selectivity(
    values: Mapping[str, np.ndarray],
    dropped: np.ndarray,
    *,
    thresholds: Thresholds,
    permutations: int | None = None,
    seed: int = 0,
    batch: int = 256,
) -> Outcome:
    """Were the materialization drops metric-selective (bar E2.1b, Defect).

    ``values`` holds the *committed catalog* metrics for the selected windows and
    ``dropped`` marks which of them failed at full res. Per the bar: the observed
    statistic is the per-metric W1 between the dropped set and the retained set, and the
    null draws random subsets of the same size from the same selected set. It fires if
    any one of the six exceeds its own null's 99th percentile.

    The rule is blind by construction and stays that way here: only ``dropped.sum()``
    enters the null, never which windows dropped, so the threshold this returns is the
    one the bar fixed before anybody had seen an attrition. ``seed`` reproduces the
    printed number byte for byte; at 10,000 permutations the quantile is smooth enough
    that the verdict does not turn on it.

    A non-finite metric raises rather than being dropped. Every selected window is a
    *kept* window and kept windows carry no NaN metric (C5's invariant), so one here
    means the join is wrong — and a NaN would propagate into ``observed``, make
    ``observed > null_p99`` false, and report **pass** on a Defect gate. Unknown is not
    passing, the same stance ``gates.py`` takes on an unread mask.
    """
    dropped = np.asarray(dropped, dtype=bool)
    n = int(dropped.size)
    for name in FROZEN:
        column = np.asarray(values[name], dtype=np.float64)
        if column.size != n:
            raise ValueError(
                f"{name} has {column.size} values for {n} selected windows — misaligned"
            )
        if not np.isfinite(column).all():
            raise ValueError(
                f"{int((~np.isfinite(column)).sum())} non-finite {name} value(s) among the "
                "selected windows. Every selected window is a kept window and kept windows "
                "carry no NaN metric, so this is a bad join, not data — refusing to report "
                "a pass on a Defect gate whose statistic would silently be NaN."
            )
    n_dropped = int(dropped.sum())
    permutations = int(thresholds.permutations if permutations is None else permutations)

    observed: dict[str, float] = {}
    null_p: dict[str, float] = {}
    fired_metrics: list[str] = []
    if 0 < n_dropped < n:
        sorted_by_metric = {}
        for name in FROZEN:
            column = np.asarray(values[name], dtype=np.float64)
            order = np.argsort(column, kind="stable")
            sorted_by_metric[name] = (order, column[order])
            observed[name] = float(_partition_w1(column[order], dropped[order][None, :])[0])

        # Drawn in batches and consumed by all six metrics before the next batch, so a
        # 10,000 x 50,000 membership matrix never exists (it would be 500 MB) and the
        # seed still means one thing: the same draws test every metric.
        rng = np.random.default_rng(seed)
        null: dict[str, list[np.ndarray]] = {name: [] for name in FROZEN}
        for start in range(0, permutations, batch):
            size = min(batch, permutations - start)
            draws = np.zeros((size, n), dtype=bool)
            for i in range(size):
                draws[i, rng.choice(n, n_dropped, replace=False)] = True
            for name in FROZEN:
                order, sorted_values = sorted_by_metric[name]
                null[name].append(_partition_w1(sorted_values, draws[:, order]))
        for name in FROZEN:
            null_p[name] = float(
                np.percentile(np.concatenate(null[name]), thresholds.null_percentile)
            )
            if observed[name] > null_p[name]:
                fired_metrics.append(name)

    fired = bool(fired_metrics)
    return Outcome(
        criterion="E2.1b_drop_selectivity",
        gates=True,
        failure_class="defect",
        fired=fired,
        measured={
            "n_selected": n,
            "n_dropped": n_dropped,
            "permutations": permutations if 0 < n_dropped < n else 0,
            "seed": seed,
            "percentile": thresholds.null_percentile,
            "observed_w1": observed,
            "null_p99": null_p,
            "fired_metrics": fired_metrics,
        },
        measured_miss=(
            "; ".join(
                f"{m}: W1 {observed[m]:.4g} > null p99 {null_p[m]:.4g}" for m in fired_metrics
            )
            if fired
            else None
        ),
    )


# --- E2.2a selection integrity ------------------------------------------------


def selection_integrity(
    *,
    md5: str,
    per_stratum: Mapping[int, int],
    provenance: Mapping[str, int],
    expected_md5: str,
    expected_rows: int,
    expected_per_stratum: Mapping[int, int],
    expected_provenance: Mapping[str, int],
) -> Outcome:
    """Did the driver load Phase D's list, unaltered (bar E2.2a, Defect, zero tolerance).

    The dull failure nobody plans for: a stale file, or rows silently dropped on parse.
    Costs milliseconds and is independent of every other decision, which is why the bar
    puts it first.
    """
    n_rows = int(sum(per_stratum.values()))
    strata_off = sorted(
        int(h)
        for h in set(map(int, per_stratum)) | set(map(int, expected_per_stratum))
        if int(per_stratum.get(h, 0)) != int(expected_per_stratum.get(h, 0))
    )
    provenance_off = {
        key: [int(provenance.get(key, 0)), int(expected_provenance.get(key, 0))]
        for key in set(provenance) | set(expected_provenance)
        if int(provenance.get(key, 0)) != int(expected_provenance.get(key, 0))
    }
    problems: list[str] = []
    if md5 != expected_md5:
        problems.append(f"selection-list md5 {md5} != committed {expected_md5}")
    if n_rows != int(expected_rows):
        problems.append(f"{n_rows} rows loaded, committed list has {expected_rows}")
    if strata_off:
        problems.append(f"per-stratum counts differ from n_h for strata {strata_off}")
    if provenance_off:
        problems.append(f"provenance counts differ: {provenance_off}")
    return Outcome(
        criterion="E2.2a_selection_integrity",
        gates=True,
        failure_class="defect",
        fired=bool(problems),
        measured={
            "md5": md5,
            "expected_md5": expected_md5,
            "n_rows": n_rows,
            "expected_rows": int(expected_rows),
            "strata_off_allocation": strata_off,
            "provenance": {k: int(v) for k, v in provenance.items()},
            "provenance_mismatches": provenance_off,
        },
        measured_miss="; ".join(problems) or None,
    )


# --- E2.2b materialization counts ---------------------------------------------


def materialization_counts(
    realized: Mapping[int, int], selected: Mapping[int, int], *, contract: str
) -> Outcome:
    """Realized per-stratum counts against the allocation (bar E2.2b, Defect).

    Branches on the live attrition contract, both of which #47 pre-committed. Under
    ``accept_shortfall`` the only defect is a stratum with *more* tiles than it was
    allocated — that is duplication, and there is no benign way to get there; how much
    a stratum lost is E2.2c's question, not this one.
    """
    strata = _strata(realized, selected)
    over = [h for h in strata if int(realized.get(h, 0)) > int(selected.get(h, 0))]
    under = [h for h in strata if int(realized.get(h, 0)) < int(selected.get(h, 0))]
    if contract == "accept_shortfall":
        problems = (
            [f"strata {over} materialized more tiles than were selected for them"] if over else []
        )
    elif contract == "backfill_to_budget":
        problems = (
            [f"strata {sorted(set(over) | set(under))} != n_h exactly under backfill"]
            if (over or under)
            else []
        )
    else:
        raise ValueError(f"unregistered attrition contract {contract!r}")
    return Outcome(
        criterion="E2.2b_materialization_counts",
        gates=True,
        failure_class="defect",
        fired=bool(problems),
        measured={
            "contract": contract,
            "n_selected": int(sum(selected.values())),
            "n_realized": int(sum(realized.values())),
            "strata_over_allocation": over,
            "strata_under_allocation": under,
            "per_stratum": {
                str(h): {"selected": int(selected.get(h, 0)), "realized": int(realized.get(h, 0))}
                for h in strata
            },
        },
        measured_miss="; ".join(problems) or None,
    )


# --- E2.2c thin stratum -------------------------------------------------------


def thin_stratum(
    realized: Mapping[int, int], selected: Mapping[int, int], *, thresholds: Thresholds
) -> Outcome:
    """The plan's 'a stratum is empty' condition, made reachable (bar E2.2c, Finding).

    Two limbs, either of which fires. The **absolute** one is #42's ``max(500, 1%)``
    floor, documented structurally inert at selection (the smallest Neyman share is
    1.47%) — if attrition pushes a realized count under it, that inert clamp has become
    retroactively binding, which is the event worth naming. The **relative** one is a
    conjunction on purpose: a bare drop-rate ratio detonates on a small denominator, so
    the loss must also be substantial in its own right.
    """
    strata = _strata(realized, selected)
    total_selected = int(sum(selected.values()))
    total_realized = int(sum(realized.values()))
    corpus_drop_rate = (
        (total_selected - total_realized) / total_selected if total_selected else 0.0
    )

    per_stratum: dict[str, dict[str, float | int]] = {}
    below_absolute: list[int] = []
    relative_hits: list[int] = []
    for h in strata:
        sel = int(selected.get(h, 0))
        real = int(realized.get(h, 0))
        drop_rate = (sel - real) / sel if sel else 0.0
        loss_frac = drop_rate  # lost share of the stratum's own allocation
        ratio = _ratio(drop_rate, corpus_drop_rate)
        per_stratum[str(h)] = {
            "selected": sel,
            "realized": real,
            "drop_rate": drop_rate,
            "drop_rate_ratio": ratio,
            "loss_frac_of_allocation": loss_frac,
        }
        if real < thresholds.thin_absolute:
            below_absolute.append(h)
        if ratio >= thresholds.thin_rate_ratio and loss_frac >= thresholds.thin_loss_frac:
            relative_hits.append(h)

    fired = bool(below_absolute or relative_hits)
    misses = []
    for h in below_absolute:
        misses.append(f"stratum {h}: realized {per_stratum[str(h)]['realized']} < {thresholds.thin_absolute}")
    for h in relative_hits:
        entry = per_stratum[str(h)]
        misses.append(
            f"stratum {h}: drop rate {entry['drop_rate']:.2%} vs corpus-wide "
            f"{corpus_drop_rate:.2%} ({entry['drop_rate_ratio']:.1f}x), losing "
            f"{entry['loss_frac_of_allocation']:.2%} of its allocation"
        )
    return Outcome(
        criterion="E2.2c_thin_stratum",
        gates=True,
        failure_class="finding",
        fired=fired,
        measured={
            "corpus_drop_rate": corpus_drop_rate,
            "absolute_floor": thresholds.thin_absolute,
            "rate_ratio": thresholds.thin_rate_ratio,
            "loss_frac": thresholds.thin_loss_frac,
            "below_absolute": below_absolute,
            "relative_hits": relative_hits,
            "min_realized": min((int(realized.get(h, 0)) for h in strata), default=0),
            "per_stratum": per_stratum,
        },
        measured_miss="; ".join(misses) or None,
        implicated_knob="allocation" if fired else None,
    )


# --- E2.2d coastal floor ------------------------------------------------------


def coastal_floor(
    realized_coastal: int, realized_total: int, *, thresholds: Thresholds
) -> Outcome:
    """The coastal floor as the rate it always was (bar E2.2d, Finding under shortfall).

    670 is a rate wearing a count's clothes: #42 derived it from coastline's within-frame
    rate (40,815 / 3,052,735 = 1.337%). If the denominator shrinks, honoring the count
    would over-represent coast, so the criterion is the rate. The realized count is
    reported as itself — never rounded up to 670, never described as "met".
    """
    rate = realized_coastal / realized_total if realized_total else 0.0
    fired = rate < thresholds.coastal_min_rate
    return Outcome(
        criterion="E2.2d_coastal_floor",
        gates=True,
        failure_class="finding",
        fired=fired,
        measured={
            "realized_coastal": int(realized_coastal),
            "realized_total": int(realized_total),
            "realized_rate": rate,
            "min_rate": thresholds.coastal_min_rate,
        },
        measured_miss=(
            f"coastal share {rate:.4%} ({realized_coastal}/{realized_total}) below the "
            f"registered frame rate {thresholds.coastal_min_rate:.4%} (1.337%)"
            if fired
            else None
        ),
        implicated_knob="allocation (coastal floor sub-knob)" if fired else None,
    )


# --- E2.3 per-COG guard -------------------------------------------------------


def per_cog_guard(counts: Mapping[str, int], *, thresholds: Thresholds) -> Outcome:
    """The one tripwire on an otherwise published analysis (bar E2.3 guard, Defect).

    Reuses #42's retired ADR-0001-era per-COG cap of 64, which is inert at debug tier
    (mean 3.16 windows/COG, measured max 44). As a tripwire it never fires on a healthy
    corpus; it fires if materialization concentrated windows where selection never
    intended — a broken round-robin, a looping COG read, a collapsed stratum.
    """
    offending = sorted(cog for cog, n in counts.items() if int(n) > thresholds.per_cog_cap)
    values = np.array([int(n) for n in counts.values()], dtype=np.int64) if counts else np.zeros(0, dtype=np.int64)
    return Outcome(
        criterion="E2.3_coverage_per_cog_guard",
        gates=True,
        failure_class="defect",
        fired=bool(offending),
        measured={
            "cap": thresholds.per_cog_cap,
            "n_cogs": int(values.size),
            "max_windows_per_cog": int(values.max()) if values.size else 0,
            "mean_windows_per_cog": float(values.mean()) if values.size else 0.0,
            "median_windows_per_cog": float(np.median(values)) if values.size else 0.0,
            "p99_windows_per_cog": float(np.percentile(values, 99)) if values.size else 0.0,
            "offending_cogs": offending,
        },
        measured_miss=(
            f"{len(offending)} COG(s) exceed the {thresholds.per_cog_cap}-window cap: "
            f"{offending[:5]}"
            if offending
            else None
        ),
    )


# --- E2.5 gate x stratum bias -------------------------------------------------


def gate_stratum_bias(
    *,
    fires: Mapping[str, Mapping[int, int]],
    membership: Mapping[int, int],
    corpus_fires: Mapping[str, int],
    corpus_rows: int,
    thresholds: Thresholds,
    baseline: str = "whole catalog",
    criterion: str = "E2.5_gate_attrition_bias",
    gates_verdict: bool = True,
) -> Outcome:
    """Did a quality gate hollow out a terrain family (bar E2.5, Finding).

    ``membership`` is each stratum's *would-be, pre-gate* size — dropped windows scored
    against the committed centroids to give them the stratum they would have joined.
    Both limbs must trip: the gate has to be concentrated (``>= 3x`` its own corpus-wide
    rate) *and* to take a substantial bite (``>= 10%`` of that stratum's pre-gate
    membership). The pre-registered expectation is even attrition, so a firing here is a
    genuine surprise, and the one that graduates the map's per-stratum gate-loosening fog.

    ``corpus_fires``/``corpus_rows`` define the *baseline population* the ratio limb
    compares against, and the choice matters enough to be named. The bar says "that same
    gate's corpus-wide drop rate", which reads on the **whole catalog** — so that is the
    default and the only population whose result gates. ``baseline`` labels it in the
    artifact, and a caller may re-run the same rule over a narrower population as a
    *published sensitivity* (``gates_verdict=False``) — never as a second verdict. A
    denominator chosen after the numbers are visible is exactly the peeking the
    pre-registration exists to prevent.
    """
    per_pair: dict[str, dict[str, dict[str, float | int]]] = {}
    fired_pairs: list[dict[str, Any]] = []
    for gate, by_stratum in fires.items():
        corpus_rate = (int(corpus_fires.get(gate, 0)) / corpus_rows) if corpus_rows else 0.0
        per_pair[gate] = {}
        for h in _strata(membership, by_stratum):
            size = int(membership.get(h, 0))
            if size == 0:
                continue
            n_fires = int(by_stratum.get(h, 0))
            rate = n_fires / size
            ratio = _ratio(rate, corpus_rate)
            per_pair[gate][str(h)] = {
                "fires": n_fires,
                "membership": size,
                "within_stratum_rate": rate,
                "corpus_wide_rate": corpus_rate,
                "rate_ratio": ratio,
            }
            if ratio >= thresholds.bias_rate_ratio and rate >= thresholds.bias_loss_frac:
                fired_pairs.append(
                    {
                        "gate": gate,
                        "stratum_id": h,
                        "within_stratum_rate": rate,
                        "corpus_wide_rate": corpus_rate,
                        "rate_ratio": ratio,
                        "membership": size,
                        "fires": n_fires,
                        "knob": GATE_KNOBS.get(gate),
                    }
                )

    fired = bool(fired_pairs)
    knobs = sorted({GATE_KNOBS[p["gate"]] for p in fired_pairs if p["gate"] in GATE_KNOBS})
    unnamed = sorted({p["gate"] for p in fired_pairs if p["gate"] not in GATE_KNOBS})
    return Outcome(
        criterion=criterion,
        gates=gates_verdict,
        failure_class="finding" if gates_verdict else "published",
        fired=fired,
        measured={
            "rate_ratio": thresholds.bias_rate_ratio,
            "min_within_stratum_rate": thresholds.bias_loss_frac,
            "baseline_population": baseline,
            "corpus_rows": int(corpus_rows),
            "corpus_wide_rates": {
                gate: (int(corpus_fires.get(gate, 0)) / corpus_rows) if corpus_rows else 0.0
                for gate in fires
            },
            "fired_pairs": fired_pairs,
            "pairs_without_a_named_knob": unnamed,
            "per_pair": per_pair,
        },
        measured_miss="; ".join(
            f"stratum {p['stratum_id']}: {p['gate']} within-stratum drop "
            f"{p['within_stratum_rate']:.1%}, corpus-wide {p['corpus_wide_rate']:.1%}, "
            f"ratio {p['rate_ratio']:.1f}x"
            for p in fired_pairs
        )
        or None,
        implicated_knob=(", ".join(knobs) if knobs else None) if gates_verdict else None,
    )


# --- Published diagnostics ----------------------------------------------------


def published(criterion: str, measured: dict[str, Any]) -> Outcome:
    """Wrap a non-gating diagnostic (E2.1c, E2.3) in the same shape as a gated one.

    The bar is explicit that these can never gate: 'matches the pool' is the *wrong*
    success criterion for a deliberately terrain-diverse corpus, and any geographic pass
    condition smuggles geo-uniformity back in against Neyman. They are published so the
    E3 honesty note has a number instead of an adjective.
    """
    return Outcome(
        criterion=criterion, gates=False, failure_class="published", fired=None, measured=measured
    )


def summarize(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    """Per-criterion pass/fail plus the payloads — but no verdict.

    Deliberately stops one step short of the roll-up. The bar assigns the Defect-first /
    any-Finding-fails state machine to #54; emitting it here would let the report that
    produces the numbers also render the judgement on them.
    """
    return {
        "criteria": {o.criterion: o.as_dict() for o in outcomes},
        "gating_fired": sorted(o.criterion for o in outcomes if o.gates and o.fired),
        "defects_fired": sorted(
            o.criterion for o in outcomes if o.fired and o.failure_class == "defect"
        ),
        "findings_fired": sorted(
            o.criterion for o in outcomes if o.fired and o.failure_class == "finding"
        ),
        "verdict": None,
        "verdict_owner": "https://github.com/JStoschek/DEMAE/issues/54",
    }
