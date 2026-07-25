"""Gate F's pre-registered criteria, as executable rules — the 500 k scale tier (issue #69).

The bar itself is not here. It was written before the 500 k draw existed and committed to
[`docs/research/f-gate-bar.json`](../../docs/research/f-gate-bar.json) by issue #69, for
the same reason Gate E's was: a criterion chosen after seeing the numbers can be argued
either way. This module is the other half — the rules that read that file and turn a
finished scale corpus into a pass or a fail against it. Nothing here decides a threshold,
and nothing here renders the verdict; the roll-up is #74's.

## Why a sibling module and not a `tier=` flag on `dataset.validate`

:mod:`dataset.validate` is **frozen**. ADR-0006 and ADR-0007 rest on the numbers it has
registered, so rescaling ``THIN_ABSOLUTE`` from 500 to 5,000 in place would silently
re-point *Gate E's* measurements at a threshold nobody registered for Gate E — and the
prose anchors would not catch it (see below). What is genuinely shared is the *rule
functions*: they are pure, they take arrays and counts, and re-running them at a new
budget without a 50 k constant in the way is the design win the E module was built for.
So this module imports them unchanged and supplies its own numbers.

## The anchors interpolate their constants here

Gate E's ``_PROSE_ANCHORS`` hard-code the literal ``"500"`` rather than deriving it from
``THIN_ABSOLUTE``, so the tie between the committed prose and the measured constant is by
convention, not by code — an edit to the constant alone would leave ``read_bar`` passing
while measuring against an unregistered number. Every anchor below is built by
interpolating the constant it guards, so that gap is closed: the prose and the code
cannot drift apart without raising on load.

## What changed from Gate E, and why it is not just a bigger number

Three of Gate E's criteria are **structurally inert** at 500 k and are carried as
published diagnostics instead of gates — the bar's ``what_changed_from_gate_e`` block
holds the full argument, but in short: E2.5 is *bit-identical* at any budget (it reads the
catalog and the gates, which have no budget term), the gallery's question is about ``k``
and was answered by the same centroids at 50 k, and the per-COG cap of 64 sits *inside*
the healthy range once the average draw is 16.4 % of the pool.

Three are new. :func:`pixel_integrity` closes the fact that **Gate E never read a shard**
— ``E2.1a`` compares the manifest's frozen metrics to the catalog's, but
:mod:`dataset.manifest` *copies* them from the catalog, so it verifies the join and not
the imagery. :func:`distinctness` closes a duplicate-inside-the-selection-list hole that
``E2.2a`` and ``E2.2b`` both pass. And :func:`drainage_representation` is ADR-0007's
Phase-F revisit trigger — the one criterion here that measures something the *budget*
changes.

## Two failure classes, never merged

Unchanged from Gate E. A **Defect** means the corpus does not implement its own committed
inputs — fix the code, re-run, stay in the map. A **Finding** means the corpus is faithful
and the *design* is wrong — record it, stop, open a fresh map with the implicated knob.
:meth:`Outcome.payload` still refuses a Finding that cannot name a knob from the plan's
§215-228 table.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from .validate import (  # the pure rules, reused unchanged — see the module docstring
    Outcome,
    Thresholds,
    coastal_floor,
    drop_selectivity,
    gate_stratum_bias,
    materialization_counts,
    metric_fidelity,
    published,
    selection_integrity,
    summarize,
    thin_stratum,
    wasserstein1,
)

#: The committed Gate F bar, relative to the repo root. The report passes an explicit path.
F_BAR_PATH: Final = Path("docs/research/f-gate-bar.json")

#: The scale-tier budget this bar was registered against (map #67's upper-bound choice).
BUDGET: Final = 500_000

# --- The registered numbers ---------------------------------------------------
#
# Every threshold below is #69's, and every one carries the bar text it was registered
# in. Numbers the bar states *structurally* (the drainage p75 thresholds, the permutation
# count) are read off the JSON instead — see :func:`read_bar`.

#: F1b: percentile of the permutation null a metric's observed W1 must exceed to fire.
#: Bar: ``"exceeds the 99th percentile of its own null distribution"``. Carried from #50
#: unchanged, because the rule is keyed only on the realized drop count.
NULL_PERCENTILE: Final = 99.0

#: F2c absolute limb. Bar: ``"every realized n_h >= 5000"`` — #42's ``max(500, 1%·budget)``
#: clamp evaluated at this budget, confirmed by ``select.allocate(budget=500_000)``
#: reporting ``floor_clamp=5000, floor_clamp_strata_below=0``. Still structurally inert
#: (smallest ``n_h`` is 7,343); a realized count below it means attrition made that inert
#: clamp retroactively binding.
THIN_ABSOLUTE: Final = 5_000

#: F2c relative limb, both halves. Bar: ``"a stratum trips if BOTH its drop rate >= 3x the
#: corpus-wide drop rate AND it loses >= 2% of its allocation"``. Carried from #47.
THIN_RATE_RATIO: Final = 3.0
THIN_LOSS_FRAC: Final = 0.02

#: F2d. Bar: ``"realized_coastal / realized_total >= 0.01337"``. Budget-invariant — it is
#: coastline's within-frame *rate* (40,815 / 3,052,735), not the count 6,700.
COASTAL_MIN_RATE: Final = 0.01337

#: F1d, both halves. Bar: ``"the share of tiles with ... > 0.5 must be at most 0.001"``.
#: The debug tier measured 0.000 over 0.5 as-built against 0.481 under an adjacent-window
#: mixup, so the separation is ~480x. Zero tolerance was rejected as brittle against the
#: known benign causes (the plausibility clip and the differing land-mask definition).
PIXEL_REL_TOL: Final = 0.5
PIXEL_SHARE_MAX: Final = 0.001

#: F3. Bar: ``"r_h ... must be at least 0.25"`` — the drainage share the terrain itself has
#: before the gate, which is what thresholding at the *pre-gate* p75 makes it by
#: construction. The per-stratum texture thresholds are structural and read from the bar.
DRAINAGE_MIN_SHARE: Final = 0.25

#: F3's two tiers, copying #63's structure. Strata 2/17 gate; 5/7 are computed identically
#: and reported as corroboration only, never independent triggers — ADR-0007 scoped its
#: revisit trigger to 2/17, stratum 5's drops are genuine junk, and stratum 7's texture is
#: sastrugi noise running the *opposite* way (kept 0.098 > dropped 0.091).
DRAINAGE_GATING: Final = (2, 17)
DRAINAGE_CORROBORATING: Final = (5, 7)

#: F7. Bar: ``"must not exceed 0.0004 (200 windows at 500,000)"`` — 10x the debug-tier
#: rate, and ~40 sigma from the pre-registered expectation of ~20 drops.
DROP_RATE_MAX: Final = 0.0004

#: F7's pre-registered expectation, for the report to be surprised against. Derived from
#: the boundary-band density, not by scaling: the kept pool holds 1,449 windows in
#: ``land_std ∈ [3.0, 3.01)``, and the two debug-tier drops imply a flip band ~0.001 m
#: wide, so attrition is linear in budget.
EXPECTED_DROPS: Final = 20

#: F7b. The only statuses ``normalize.prepare_tile`` and the driver can produce. Anything
#: else is an unregistered failure mode and surfaces as itself.
REGISTERED_STATUSES: Final = frozenset({"ok", "zero_fill", "degenerate", "no_land"})

#: F8's ``inherit_debug`` branch reference — ``data/corpus_50k/corpus_meta.json``.
DEBUG_TIER_STATS: Final = {
    "global_mean": 951.1396151875607,
    "global_std": 1012.9799372491552,
}

#: F5's constants, retained only so the *published* attrition table can be reprinted with
#: the same ratios Gate E used. They gate nothing here.
BIAS_RATE_RATIO: Final = 3.0
BIAS_LOSS_FRAC: Final = 0.10


# --- The committed bar --------------------------------------------------------


@dataclass(frozen=True)
class FThresholds:
    """The Gate-F-only numbers, beside the shared :class:`Thresholds` the E rules take.

    Split in two because the carried-over rules (:func:`thin_stratum`,
    :func:`coastal_floor`) take :class:`dataset.validate.Thresholds` by contract, and
    re-typing them to reach F's new numbers would mean editing the frozen module.
    """

    pixel_rel_tol: float
    pixel_share_max: float
    drainage_min_share: float
    drainage_texture_p75: Mapping[int, float]
    drop_rate_max: float
    expected_drops: int


#: ``(json path into the bar, phrase that must be present)`` for every threshold the bar
#: states only in prose. Each phrase is *interpolated from the constant it guards* — the
#: fix for the convention-only tie described in the module docstring. Matched with word
#: boundaries, so ``"0.25"`` is not satisfied by a bar edited to say ``0.255``.
_PROSE_ANCHORS: Final = (
    (("criteria", "F1b_drop_selectivity", "threshold", "fire_if"),
     f"{NULL_PERCENTILE:g}th percentile"),
    (("criteria", "F2c_thin_stratum", "conditions", "absolute"), f"{THIN_ABSOLUTE}"),
    (("criteria", "F2c_thin_stratum", "conditions", "relative"), f"{THIN_RATE_RATIO:g}x"),
    (("criteria", "F2c_thin_stratum", "conditions", "relative"), f"{THIN_LOSS_FRAC:.0%}"),
    (("criteria", "F2d_coastal_floor", "test"), f"{COASTAL_MIN_RATE:g}"),
    (("criteria", "F1d_pixel_integrity", "test"), f"{PIXEL_REL_TOL:g}"),
    (("criteria", "F1d_pixel_integrity", "test"), f"{PIXEL_SHARE_MAX:g}"),
    (("criteria", "F3_drainage_representation", "test"), f"{DRAINAGE_MIN_SHARE:g}"),
    (("criteria", "F7_attrition_expectation", "test"), f"{DROP_RATE_MAX:g}"),
)


@dataclass(frozen=True)
class FBar:
    """The parsed Gate F bar: its thresholds, its inputs, its raw text."""

    doc: dict[str, Any]
    thresholds: Thresholds
    f: FThresholds
    path: Path

    @property
    def committed_inputs(self) -> dict[str, Any]:
        return self.doc["committed_inputs"]

    @property
    def allocation(self) -> dict[str, Any]:
        return self.doc["committed_inputs"]["allocation"]

    @property
    def expected_per_stratum(self) -> dict[int, int]:
        """The committed ``n_h`` table, keyed by int — what F2a checks the list against."""
        return {int(h): int(n) for h, n in self.allocation["n_h"].items()}

    @property
    def normalization_branches(self) -> tuple[str, ...]:
        """The branches the bar itself pre-committed for #70 to *select* between."""
        return tuple(self.doc["normalization_branch"]["branches"])

    def live_normalization_branch(self, source: str) -> str:
        """Resolve the corpus card's ``normalization_stat_source`` against the bar.

        Refuses a branch string the bar never pre-committed, and reads the permitted set
        out of the committed file rather than from a constant here — the whole point of a
        conditional pre-registration is that the later ticket may *select* a branch, not
        invent one after the fact. The same discipline
        :meth:`dataset.validate.Bar.live_branch` holds #50 to.
        """
        if source not in self.normalization_branches:
            raise ValueError(
                f"normalization statistic source {source!r} is unregistered — "
                f"{self.path} pre-committed {self.normalization_branches}. A new branch "
                "is a new pre-registration, not a report detail."
            )
        return source


def read_bar(path: Path | str = F_BAR_PATH) -> FBar:
    """Load the committed Gate F bar, refusing one whose prose has drifted from this module.

    Same crude, deliberate anchor check Gate E uses — one word-bounded phrase per
    registered number — except each phrase is derived from the constant rather than
    written out beside it, so the two cannot part company silently.
    """
    path = Path(path)
    doc = json.loads(path.read_text())
    thresholds = Thresholds(
        permutations=int(
            doc["criteria"]["F1b_drop_selectivity"]["threshold"]["permutations"]
        ),
        null_percentile=NULL_PERCENTILE,
        thin_absolute=THIN_ABSOLUTE,
        thin_rate_ratio=THIN_RATE_RATIO,
        thin_loss_frac=THIN_LOSS_FRAC,
        coastal_min_rate=COASTAL_MIN_RATE,
        # Unused: Gate F retired the absolute per-COG cap (see the module docstring).
        # F6b bounds each COG by its own kept inventory instead. A negative value is
        # self-evidently not a threshold, so a stray call cannot quietly measure anything.
        per_cog_cap=-1,
        bias_rate_ratio=BIAS_RATE_RATIO,
        bias_loss_frac=BIAS_LOSS_FRAC,
    )
    scale = doc["criteria"]["F3_drainage_representation"]["reference_and_scale_sensitivity"]
    tiers = doc["criteria"]["F3_drainage_representation"]["two_tier_scope"]
    p75: dict[int, float] = {}
    for h in (*DRAINAGE_GATING, *DRAINAGE_CORROBORATING):
        node = scale.get(f"stratum_{h}") or tiers.get(f"stratum_{h}")
        if node is None:
            raise ValueError(
                f"{path}: no pre-gate texture p75 registered for stratum {h}; the bar and "
                "dataset.validate_f's two-tier scope have drifted."
            )
        p75[h] = float(node["texture_p75_pre_gate"])
    f = FThresholds(
        pixel_rel_tol=PIXEL_REL_TOL,
        pixel_share_max=PIXEL_SHARE_MAX,
        drainage_min_share=DRAINAGE_MIN_SHARE,
        drainage_texture_p75=p75,
        drop_rate_max=DROP_RATE_MAX,
        expected_drops=EXPECTED_DROPS,
    )
    for keys, anchor in _PROSE_ANCHORS:
        node: Any = doc
        for key in keys:
            node = node[key]
        if not re.search(rf"(?<![\w.]){re.escape(anchor)}(?![\w.])", str(node)):
            raise ValueError(
                f"{path}: {'.'.join(keys)} no longer states {anchor!r} — the committed "
                "bar and dataset.validate_f's registered constants have drifted. "
                "Reconcile them in a dated commit before running the report."
            )
    return FBar(doc=doc, thresholds=thresholds, f=f, path=path)


# --- Re-labelling the carried-over rules --------------------------------------


def relabel(outcome: Outcome, criterion: str) -> Outcome:
    """Re-tag a reused Gate E rule with its Gate F criterion name.

    The E rules hard-code their own ``criterion`` string, which is right for E's report
    and wrong for F's — a Gate F artifact naming ``E2.2c_thin_stratum`` invites a reader to
    look the number up in the wrong committed bar. Only the label changes; the measured
    values and the fired flag are the rule's.
    """
    return dataclasses.replace(outcome, criterion=criterion)


# --- F1d pixel integrity ------------------------------------------------------


def pixel_integrity(
    manifest_std: np.ndarray,
    catalog_land_std: np.ndarray,
    *,
    thresholds: FThresholds,
) -> Outcome:
    """Do the stored pixels belong to the window the manifest claims (bar F1d, Defect)?

    The only whole-corpus check on the *imagery* rather than the metadata. ``manifest_std``
    is computed by :func:`dataset.normalize.prepare_tile` on the stored int16 array;
    ``catalog_land_std`` was computed independently at full res in Phase C. Agreement is at
    the quantization floor for a correctly-built corpus (debug-tier median relative
    disagreement 8.4e-5) and collapses under a mis-join (48 % of tiles over the tolerance
    for an adjacent-window mixup, 76 % for a full shuffle).

    The share, not the max, is what gates: a handful of tiles legitimately disagree by
    more than the tolerance because the plausibility clip and the land-mask definition
    differ between the two computations, and a zero-tolerance rule would fail a healthy
    corpus on them.
    """
    a = np.asarray(manifest_std, dtype=np.float64)
    b = np.asarray(catalog_land_std, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"pixel_integrity got {a.size} manifest rows against {b.size} catalog rows — "
            "the join dropped or duplicated tiles, which is itself the defect"
        )
    usable = np.isfinite(a) & np.isfinite(b) & (b > 0)
    rel = np.full(a.shape, np.nan, dtype=np.float64)
    rel[usable] = np.abs(a[usable] - b[usable]) / b[usable]
    n = int(usable.sum())
    over = int((rel[usable] > thresholds.pixel_rel_tol).sum())
    share = over / n if n else 0.0
    fired = share > thresholds.pixel_share_max
    return Outcome(
        criterion="F1d_pixel_integrity",
        gates=True,
        failure_class="defect",
        fired=fired,
        measured={
            "n_compared": n,
            "n_unusable": int((~usable).sum()),
            "rel_tol": thresholds.pixel_rel_tol,
            "share_max": thresholds.pixel_share_max,
            "n_over_tol": over,
            "share_over_tol": share,
            "median_rel_diff": float(np.median(rel[usable])) if n else float("nan"),
            "p99_rel_diff": float(np.quantile(rel[usable], 0.99)) if n else float("nan"),
            "max_rel_diff": float(np.max(rel[usable])) if n else float("nan"),
        },
        measured_miss=(
            f"{over} of {n} tiles ({share:.2%}) disagree by more than "
            f"{thresholds.pixel_rel_tol:g} relative, above the registered "
            f"{thresholds.pixel_share_max:g}"
            if fired
            else None
        ),
        implicated_knob=None,
    )


# --- F3 drainage representation -----------------------------------------------


def drainage_representation(
    texture_by_stratum: Mapping[int, np.ndarray],
    *,
    kept_tail: Mapping[int, int],
    thresholds: FThresholds,
    gating: Sequence[int] = DRAINAGE_GATING,
    corroborating: Sequence[int] = DRAINAGE_CORROBORATING,
) -> Outcome:
    """ADR-0007's Phase-F revisit trigger: does the shipped corpus still carry drainage?

    (Bar F3, Finding, knob = the ``degenerate`` flatness screen.)

    For each stratum, ``r_h`` is the share of *shipped* tiles whose ``texture`` reaches the
    stratum's **pre-gate p75** — so the reference is 0.25 by construction, and it is the
    drainage share the terrain itself has before the flatness gate touches it. Strata 2 and
    17 gate; 5 and 7 are computed identically and reported as corroboration only.

    **Single-limb on purpose.** The obvious design is a conjunction — fire only if the draw
    both exhausts the surviving tail (so the gate, not the budget, binds) *and* leaves the
    corpus deficient. At this budget that is unsatisfiable: ``n_h`` is approximately the
    tail size, so ``r_h = e_h · tail_h / n_h`` and requiring ``e_h >= 0.5`` forces ``r_h``
    to 0.49-0.57 — far above any deficit threshold. The exhaustion ``e_h`` is therefore
    *measured and reported* for the attribution it carries (``r_h`` near the kept-pool
    proportional share means the sampler declined an available tail; ``r_h`` near the
    ceiling means the survivors ran out) but it does not gate.
    """
    per_stratum: dict[str, dict[str, float | int]] = {}
    hits: list[int] = []
    for h in (*gating, *corroborating):
        values = np.asarray(texture_by_stratum.get(h, np.empty(0)), dtype=np.float64)
        values = values[np.isfinite(values)]
        p75 = float(thresholds.drainage_texture_p75[h])
        n = int(values.size)
        drainage = int((values >= p75).sum())
        tail = int(kept_tail.get(h, 0))
        per_stratum[str(h)] = {
            "n_shipped": n,
            "texture_p75_pre_gate": p75,
            "n_drainage": drainage,
            "realized_share_r_h": drainage / n if n else 0.0,
            "kept_tail": tail,
            "exhaustion_e_h": drainage / tail if tail else float("nan"),
            "gates": h in gating,
        }
        if h in gating and n and (drainage / n) < thresholds.drainage_min_share:
            hits.append(h)

    fired = bool(hits)
    misses = [
        f"stratum {h}: realized drainage share "
        f"{per_stratum[str(h)]['realized_share_r_h']:.3f} < "
        f"{thresholds.drainage_min_share:g} (pre-gate p75 texture "
        f"{per_stratum[str(h)]['texture_p75_pre_gate']:g}, exhaustion "
        f"{per_stratum[str(h)]['exhaustion_e_h']:.3f})"
        for h in hits
    ]
    return Outcome(
        criterion="F3_drainage_representation",
        gates=True,
        failure_class="finding",
        fired=fired,
        measured={
            "min_share": thresholds.drainage_min_share,
            "gating_strata": list(gating),
            "corroborating_strata": list(corroborating),
            "hits": hits,
            "per_stratum": per_stratum,
        },
        measured_miss="; ".join(misses) or None,
        implicated_knob=(
            "degenerate flatness screen (`land_std` cap)" if fired else None
        ),
    )


# --- F6a / F6b the checks that replaced the per-COG cap -----------------------


def distinctness(
    *, selection_window_ids: Sequence[str], manifest_tile_ids: Sequence[str]
) -> Outcome:
    """No window drawn twice, no tile written twice (bar F6a, Defect, zero tolerance).

    Closes a hole both count criteria pass through: a duplicate *inside* the selection
    list leaves F2a's md5, row count, per-stratum counts and provenance all matching, and
    leaves F2b's ``realized <= selected`` true. ``tile_id`` is derived from
    ``source_cog`` + ``win_row0`` + ``win_col0``, so a repeat is definitionally a bug
    rather than a judgement call.
    """
    sel_dupes = {k: v for k, v in Counter(selection_window_ids).items() if v > 1}
    man_dupes = {k: v for k, v in Counter(manifest_tile_ids).items() if v > 1}
    fired = bool(sel_dupes or man_dupes)
    misses = []
    if sel_dupes:
        misses.append(
            f"{len(sel_dupes)} duplicated window_id(s) in the selection list, e.g. "
            f"{sorted(sel_dupes)[:3]}"
        )
    if man_dupes:
        misses.append(
            f"{len(man_dupes)} duplicated tile_id(s) in the manifest, e.g. "
            f"{sorted(man_dupes)[:3]}"
        )
    return Outcome(
        criterion="F6a_distinctness",
        gates=True,
        failure_class="defect",
        fired=fired,
        measured={
            "n_selection_rows": len(selection_window_ids),
            "n_selection_distinct": len(set(selection_window_ids)),
            "n_manifest_rows": len(manifest_tile_ids),
            "n_manifest_distinct": len(set(manifest_tile_ids)),
            "n_selection_duplicated": len(sel_dupes),
            "n_manifest_duplicated": len(man_dupes),
        },
        measured_miss="; ".join(misses) or None,
        implicated_knob=None,
    )


def cog_inventory_bound(
    realized: Mapping[str, int], inventory: Mapping[str, int]
) -> Outcome:
    """No COG yields more windows than it has (bar F6b, Defect, zero tolerance).

    The mechanical failure the retired per-COG cap of 64 was actually watching for — a
    looping read, a collapsed stratum — stated in the one form that survives a budget
    change. An absolute cap cannot: a 1° COG holds at most 256 kept windows, and at a
    16.4 % average draw rate the healthy per-COG count runs into the tens, so any fixed
    number either sits inside the healthy range or above the physical ceiling.

    A COG absent from ``inventory`` is itself a breach: the draw produced a source the
    kept pool does not contain.
    """
    offenders: dict[str, dict[str, int]] = {}
    for cog, count in realized.items():
        available = int(inventory.get(cog, 0))
        if int(count) > available:
            offenders[cog] = {"realized": int(count), "inventory": available}
    fired = bool(offenders)
    return Outcome(
        criterion="F6b_cog_inventory_bound",
        gates=True,
        failure_class="defect",
        fired=fired,
        measured={
            "n_cogs_used": len(realized),
            "n_offending": len(offenders),
            "offenders": dict(sorted(offenders.items())[:10]),
            "max_realized": max((int(v) for v in realized.values()), default=0),
        },
        measured_miss=(
            f"{len(offenders)} COG(s) exceed their own kept inventory, e.g. "
            f"{sorted(offenders)[:3]}"
            if fired
            else None
        ),
        implicated_knob=None,
    )


# --- F7 attrition -------------------------------------------------------------


def attrition_rate(
    n_selected: int, n_realized: int, *, thresholds: FThresholds
) -> Outcome:
    """Is full-res attrition still a boundary artifact (bar F7, Defect)?

    Gated on the *rate*, not the count, so the rule reads the same at any budget. A 10x
    excursion above the debug-tier rate means a read path, mask-cache or clip-bound
    regression rather than a design flaw — the re-screens recompute the same quantities on
    the same full-res arrays Phase C scored, so drops should stay pinned to the handful of
    windows sitting within the int16 quantization of the 3.0 m threshold.

    ``expected_drops`` is reported beside the verdict so the report has a pre-registered
    number to be surprised against rather than only a threshold to clear.
    """
    n_selected = int(n_selected)
    n_realized = int(n_realized)
    dropped = n_selected - n_realized
    rate = dropped / n_selected if n_selected else 0.0
    fired = rate > thresholds.drop_rate_max
    return Outcome(
        criterion="F7_attrition_expectation",
        gates=True,
        failure_class="defect",
        fired=fired,
        measured={
            "n_selected": n_selected,
            "n_realized": n_realized,
            "n_dropped": dropped,
            "drop_rate": rate,
            "max_rate": thresholds.drop_rate_max,
            "expected_drops": thresholds.expected_drops,
            "expected_2sigma": [
                max(0, round(thresholds.expected_drops - 2 * thresholds.expected_drops**0.5)),
                round(thresholds.expected_drops + 2 * thresholds.expected_drops**0.5),
            ],
        },
        measured_miss=(
            f"{dropped} of {n_selected} dropped (rate {rate:.2e}), above the registered "
            f"{thresholds.drop_rate_max:g}"
            if fired
            else None
        ),
        implicated_knob=None,
    )


def registered_status_set(statuses: Mapping[str, int]) -> Outcome:
    """Every ledger status is one the bar registered (bar F7b, Defect, zero tolerance).

    An unregistered failure mode should surface as itself rather than be absorbed into
    :func:`attrition_rate`'s count — a handful of I/O errors and a handful of boundary
    re-screen fires are the same number and completely different news.
    """
    unregistered = {
        s: int(n) for s, n in statuses.items() if s not in REGISTERED_STATUSES
    }
    fired = bool(unregistered)
    return Outcome(
        criterion="F7b_registered_status_set",
        gates=True,
        failure_class="defect",
        fired=fired,
        measured={
            "registered": sorted(REGISTERED_STATUSES),
            "observed": {s: int(n) for s, n in sorted(statuses.items())},
            "unregistered": unregistered,
        },
        measured_miss=(
            f"unregistered ledger status(es): {unregistered}" if fired else None
        ),
        implicated_knob=None,
    )


# --- F8 the normalization statistic ------------------------------------------


def normalization_statistic(
    *,
    branch: str,
    recorded: Mapping[str, float],
    recomputed: Mapping[str, float],
    bar: FBar,
) -> Outcome:
    """Is the corpus card's global statistic the one it claims (bar F8, Defect)?

    ``dataset_version`` is Phase 1's reproducibility contract, so the recorded statistic is
    gated either way — but *which* check is correct depends on #70, which was unresolved
    when the bar was committed. Under ``tier_specific`` the recorded values must reproduce
    from the shipped 500 k tiles; under ``inherit_debug`` they must equal the debug tier's
    committed values, and the recomputed 500 k values are reported as a divergence
    diagnostic carrying no threshold.

    The branch is resolved through :meth:`FBar.live_normalization_branch`, which refuses a
    branch the bar never pre-committed — #70 may select, not invent.
    """
    live = bar.live_normalization_branch(branch)
    keys = ("global_mean", "global_std")
    if live == "tier_specific":
        reference = {k: float(recomputed[k]) for k in keys}
        gated_against = "recomputed from the shipped 500k tiles"
    else:
        reference = {k: float(DEBUG_TIER_STATS[k]) for k in keys}
        gated_against = "the debug tier's committed values"
    mismatches = {
        k: {"recorded": float(recorded[k]), "expected": reference[k]}
        for k in keys
        if float(recorded[k]) != reference[k]
    }
    fired = bool(mismatches)
    measured: dict[str, Any] = {
        "branch": live,
        "gated_against": gated_against,
        "recorded": {k: float(recorded[k]) for k in keys},
        "reference": reference,
        "mismatches": mismatches,
    }
    if live == "inherit_debug":
        # Published, not gated: how far the inherited statistic sits from the corpus it
        # is actually normalizing. The bar says this carries no threshold.
        measured["divergence_diagnostic"] = {
            k: {
                "inherited": float(DEBUG_TIER_STATS[k]),
                "recomputed_500k": float(recomputed[k]),
                "relative": (
                    abs(float(recomputed[k]) - DEBUG_TIER_STATS[k]) / DEBUG_TIER_STATS[k]
                ),
            }
            for k in keys
        }
    return Outcome(
        criterion="F8_normalization_statistic",
        gates=True,
        failure_class="defect",
        fired=fired,
        measured=measured,
        measured_miss=(
            f"corpus_meta global stats disagree with {gated_against}: {mismatches}"
            if fired
            else None
        ),
        implicated_knob=None,
    )


__all__ = [
    "BUDGET",
    "DEBUG_TIER_STATS",
    "DRAINAGE_CORROBORATING",
    "DRAINAGE_GATING",
    "F_BAR_PATH",
    "REGISTERED_STATUSES",
    "FBar",
    "FThresholds",
    "Outcome",
    "attrition_rate",
    "coastal_floor",
    "cog_inventory_bound",
    "distinctness",
    "drainage_representation",
    "drop_selectivity",
    "gate_stratum_bias",
    "materialization_counts",
    "metric_fidelity",
    "normalization_statistic",
    "pixel_integrity",
    "published",
    "read_bar",
    "registered_status_set",
    "relabel",
    "selection_integrity",
    "summarize",
    "thin_stratum",
    "wasserstein1",
]
