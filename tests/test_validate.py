"""The Gate E criterion rules — Phase E2 (issue #52), bar pre-registered by #47.

Zero-network, zero-corpus: every case is a hand-built frame of a few rows, so this
suite pins the *rules* [`e2-gate-bar.json`](../docs/research/e2-gate-bar.json) committed
before the corpus existed. The 4.39 M-row / 50 k-tile measurement lives in
``notebooks/e2_validation_report.py``.

The load-bearing checks are the ones a wrong report would pass silently:

1. **The thresholds come off the committed bar**, and a bar whose prose no longer
   carries the number this code implements raises rather than measuring against a
   threshold nobody registered.
2. **Each conjunction is a conjunction.** E2.2c and E2.5 both fire only when *both*
   limbs trip; a rule that fired on either would detonate on small denominators, which
   is exactly what the bar's number_provenance says the second limb is there to stop.
3. **The blind rule stays blind.** E2.1b's null depends on the drop *count*, never on
   which windows dropped — so the same count against the same committed metrics gives
   the same threshold whatever the drops were.
4. **A fired criterion carries its payload.** Every Finding emits the bar's four
   fields; a criterion that fires without naming a knob is a bug in the handoff.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset import validate

REPO = Path(__file__).resolve().parents[1]
BAR_PATH = REPO / "docs" / "research" / "e2-gate-bar.json"


@pytest.fixture(scope="module")
def bar() -> validate.Bar:
    return validate.read_bar(BAR_PATH)


@pytest.fixture(scope="module")
def thresholds(bar: validate.Bar) -> validate.Thresholds:
    return bar.thresholds


# --- The bar itself -----------------------------------------------------------


def test_thresholds_are_the_committed_numbers(thresholds: validate.Thresholds) -> None:
    """#47's registered values, read off the file rather than typed in here twice."""
    assert thresholds.permutations == 10000
    assert thresholds.null_percentile == 99.0
    assert thresholds.thin_absolute == 500
    assert thresholds.thin_rate_ratio == 3.0
    assert thresholds.thin_loss_frac == 0.02
    assert thresholds.coastal_min_rate == 0.01337
    assert thresholds.per_cog_cap == 64
    assert thresholds.bias_rate_ratio == 3.0
    assert thresholds.bias_loss_frac == 0.10


def test_a_bar_whose_prose_lost_the_number_raises(tmp_path: Path) -> None:
    """The anchor check is the tripwire: prose and code cannot drift apart quietly.

    The bar states several thresholds only in prose (``"every realized n_h >= 500"``),
    so this module has to carry the numbers. Carrying them is safe exactly as long as
    an edit to the committed file cannot pass unnoticed.
    """
    doc = json.loads(BAR_PATH.read_text())
    doc["criteria"]["E2.2c_thin_stratum"]["conditions"]["absolute"] = "every realized n_h >= 400"
    path = tmp_path / "moved-bar.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="500"):
        validate.read_bar(path)


def test_live_attrition_branch_is_read_not_assumed(bar: validate.Bar) -> None:
    """#50 selected the branch; the bar records both. The corpus card says which is live.

    The permitted set is read out of the committed file, so an unregistered contract is
    refused against the bar rather than against a constant that could drift from it.
    """
    assert set(bar.contracts) == {"backfill_to_budget", "accept_shortfall"}
    assert bar.live_branch("accept_shortfall") == "accept_shortfall"
    with pytest.raises(ValueError, match="unregistered"):
        bar.live_branch("backfill_if_short")


# --- Wasserstein-1 ------------------------------------------------------------


def test_w1_of_a_distribution_against_itself_is_zero() -> None:
    x = np.array([0.0, 1.0, 4.0, 9.0])
    assert validate.wasserstein1(x, x) == pytest.approx(0.0)


def test_w1_equal_sizes_is_the_mean_sorted_gap() -> None:
    a = np.array([0.0, 2.0])
    b = np.array([1.0, 3.0])
    assert validate.wasserstein1(a, b) == pytest.approx(1.0)


def test_w1_handles_unequal_sizes() -> None:
    """One point against two: the CDFs differ by 1/2 over a width-2 interval."""
    assert validate.wasserstein1(np.array([0.0]), np.array([0.0, 2.0])) == pytest.approx(1.0)


def test_w1_is_symmetric_and_shift_equivariant() -> None:
    a = np.array([0.0, 1.0, 5.0])
    b = np.array([2.0, 3.0])
    assert validate.wasserstein1(a, b) == pytest.approx(validate.wasserstein1(b, a))
    assert validate.wasserstein1(a + 7.0, b + 7.0) == pytest.approx(validate.wasserstein1(a, b))


def test_the_batched_partition_form_agrees_with_the_plain_one() -> None:
    """The null runs on a vectorized rewrite; this is what keeps it honest.

    ``_partition_w1`` exists only because 10,000 permutations x 6 metrics cannot afford
    10,000 sorts. It has to give exactly what the direct definition gives, including on
    a sample with ties (zero-width support steps) and an unbalanced split.
    """
    rng = np.random.default_rng(11)
    column = np.round(rng.normal(size=200) * 3, 1)  # rounding manufactures ties
    order = np.argsort(column, kind="stable")
    sorted_values = column[order]
    for n_in in (1, 7, 100, 199):
        member = np.zeros(200, dtype=bool)
        member[rng.choice(200, n_in, replace=False)] = True
        fast = validate._partition_w1(sorted_values, member[order][None, :])[0]
        slow = validate.wasserstein1(column[member], column[~member])
        assert fast == pytest.approx(slow)


# --- E2.1b drop selectivity ---------------------------------------------------


def _selected(n: int = 400) -> dict[str, np.ndarray]:
    """A selected set whose metrics are a plain ramp — order is the only structure."""
    ramp = np.arange(n, dtype=np.float64)
    return {name: ramp.copy() for name in validate.FROZEN}


def test_random_attrition_does_not_fire(thresholds: validate.Thresholds) -> None:
    """A drop set drawn the same way the null draws is, by construction, unremarkable."""
    values = _selected()
    dropped = np.zeros(400, dtype=bool)
    dropped[np.random.default_rng(7).choice(400, 8, replace=False)] = True
    outcome = validate.drop_selectivity(
        values, dropped, thresholds=thresholds, permutations=400, seed=0
    )
    assert outcome.fired is False
    assert outcome.measured["n_dropped"] == 8


def test_terrain_selective_attrition_fires(thresholds: validate.Thresholds) -> None:
    """Every drop taken from one tail — the bias the criterion exists to catch."""
    values = _selected()
    dropped = np.zeros(400, dtype=bool)
    dropped[-8:] = True
    outcome = validate.drop_selectivity(
        values, dropped, thresholds=thresholds, permutations=400, seed=0
    )
    assert outcome.fired is True
    assert outcome.failure_class == "defect"
    assert set(outcome.measured["fired_metrics"]) == set(validate.FROZEN)


def test_the_null_depends_only_on_the_drop_count(thresholds: validate.Thresholds) -> None:
    """The bar's blindness clause, as an executable claim.

    Two different drop sets of the same size must produce the *same* null, hence the
    same threshold — the rule can be evaluated without anyone having seen which windows
    dropped.
    """
    values = _selected()
    tail = np.zeros(400, dtype=bool)
    tail[-8:] = True
    head = np.zeros(400, dtype=bool)
    head[:8] = True
    a = validate.drop_selectivity(values, tail, thresholds=thresholds, permutations=200, seed=3)
    b = validate.drop_selectivity(values, head, thresholds=thresholds, permutations=200, seed=3)
    for name in validate.FROZEN:
        assert a.measured["null_p99"][name] == pytest.approx(b.measured["null_p99"][name])


def test_a_nan_metric_raises_rather_than_reporting_a_pass(
    thresholds: validate.Thresholds,
) -> None:
    """"Unknown is not passing" — the repo's stance on an unread mask, applied here.

    A NaN would propagate into the observed W1, make ``observed > null_p99`` false, and
    hand a Defect gate a silent pass. Every selected window is a kept window and kept
    windows carry no NaN metric, so one here is a bad join, not data.
    """
    values = _selected()
    values["texture"][3] = np.nan
    dropped = np.zeros(400, dtype=bool)
    dropped[-8:] = True
    with pytest.raises(ValueError, match="non-finite"):
        validate.drop_selectivity(values, dropped, thresholds=thresholds, permutations=50, seed=0)


def test_zero_drops_is_vacuous_not_an_error(thresholds: validate.Thresholds) -> None:
    """Under backfill, and under a shortfall run that happened to lose nothing."""
    outcome = validate.drop_selectivity(
        _selected(), np.zeros(400, dtype=bool), thresholds=thresholds, permutations=50, seed=0
    )
    assert outcome.fired is False
    assert outcome.measured["n_dropped"] == 0


# --- E2.1a metric fidelity ----------------------------------------------------


def test_metric_fidelity_passes_on_an_exact_copy() -> None:
    cat = {name: np.linspace(0, 1, 5) for name in validate.FROZEN}
    outcome = validate.metric_fidelity({k: v.copy() for k, v in cat.items()}, cat)
    assert outcome.fired is False
    assert outcome.measured["n_mismatched_rows"] == 0


def test_metric_fidelity_fires_when_nothing_was_compared() -> None:
    """A broken join must not pass a Defect gate by having nothing to disagree about."""
    empty = {name: np.empty(0) for name in validate.FROZEN}
    outcome = validate.metric_fidelity(empty, empty)
    assert outcome.fired is True
    assert "no rows were compared" in (outcome.measured_miss or "")


def test_metric_fidelity_compares_at_the_manifests_declared_width() -> None:
    """The bar's tolerance is exact *after* the documented cast, not exact in float64.

    Today's v2 schema stores the frozen six as float64, so the cast is the identity. If a
    later schema narrowed them, comparing in float64 would report every stored value as a
    Defect; comparing at the declared width reports rounding as rounding.
    """
    cat = {name: np.array([1 / 3, 2 / 3]) for name in validate.FROZEN}
    narrowed = {k: v.astype(np.float32) for k, v in cat.items()}
    assert validate.metric_fidelity(narrowed, cat).fired is True  # float64 comparison
    assert validate.metric_fidelity(narrowed, cat, storage_dtype=np.float32).fired is False


def test_metric_fidelity_catches_one_wrong_value() -> None:
    """A mis-join or a recompute, not a knob — hence Defect, hence exact tolerance."""
    cat = {name: np.linspace(0, 1, 5) for name in validate.FROZEN}
    man = {k: v.copy() for k, v in cat.items()}
    man["texture"][3] += 1e-12
    outcome = validate.metric_fidelity(man, cat)
    assert outcome.fired is True
    assert outcome.failure_class == "defect"
    assert outcome.measured["per_metric_mismatches"]["texture"] == 1


def test_metric_fidelity_treats_matching_nans_as_equal() -> None:
    """A NaN copied faithfully is a faithful copy; ``nan != nan`` is not the question."""
    cat = {name: np.array([1.0, np.nan]) for name in validate.FROZEN}
    outcome = validate.metric_fidelity({k: v.copy() for k, v in cat.items()}, cat)
    assert outcome.fired is False


# --- E2.2a selection integrity ------------------------------------------------


def test_selection_integrity_passes_on_the_committed_list() -> None:
    outcome = validate.selection_integrity(
        md5="abc",
        per_stratum={0: 3, 1: 2},
        provenance={"base": 4, "floor:coastline": 1},
        expected_md5="abc",
        expected_rows=5,
        expected_per_stratum={0: 3, 1: 2},
        expected_provenance={"base": 4, "floor:coastline": 1},
    )
    assert outcome.fired is False


@pytest.mark.parametrize(
    "override",
    [
        {"md5": "stale"},
        {"per_stratum": {0: 3, 1: 1}},
        {"provenance": {"base": 5, "floor:coastline": 0}},
    ],
)
def test_selection_integrity_catches_the_dull_failures(override: dict) -> None:
    """A stale file, a dropped row, a provenance the parse mangled."""
    kwargs = {
        "md5": "abc",
        "per_stratum": {0: 3, 1: 2},
        "provenance": {"base": 4, "floor:coastline": 1},
        "expected_md5": "abc",
        "expected_rows": 5,
        "expected_per_stratum": {0: 3, 1: 2},
        "expected_provenance": {"base": 4, "floor:coastline": 1},
        **override,
    }
    outcome = validate.selection_integrity(**kwargs)
    assert outcome.fired is True
    assert outcome.failure_class == "defect"


# --- E2.2b materialization counts ---------------------------------------------


def test_shortfall_allows_fewer_but_never_more() -> None:
    ok = validate.materialization_counts({0: 9, 1: 5}, {0: 10, 1: 5}, contract="accept_shortfall")
    assert ok.fired is False
    assert ok.measured["n_realized"] == 14

    dup = validate.materialization_counts({0: 11}, {0: 10}, contract="accept_shortfall")
    assert dup.fired is True
    assert dup.measured["strata_over_allocation"] == [0]


def test_backfill_demands_the_allocation_exactly() -> None:
    """The branch #50 retired, kept live in code because the bar committed both."""
    outcome = validate.materialization_counts(
        {0: 9, 1: 5}, {0: 10, 1: 5}, contract="backfill_to_budget"
    )
    assert outcome.fired is True


# --- E2.2c thin stratum -------------------------------------------------------


def test_a_healthy_corpus_trips_neither_limb(thresholds: validate.Thresholds) -> None:
    outcome = validate.thin_stratum({0: 999, 1: 700}, {0: 1000, 1: 700}, thresholds=thresholds)
    assert outcome.fired is False


def test_the_inert_500_clamp_becomes_binding(thresholds: validate.Thresholds) -> None:
    """The event the bar named worth naming: attrition pushes a stratum under the floor."""
    outcome = validate.thin_stratum({0: 480}, {0: 734}, thresholds=thresholds)
    assert outcome.fired is True
    assert outcome.measured["below_absolute"] == [0]
    assert outcome.implicated_knob == "allocation"


def test_the_relative_limb_needs_both_conditions(thresholds: validate.Thresholds) -> None:
    """A 4x drop rate that costs the stratum 1% of its allocation does not fire.

    This is the conjunction the bar's number_provenance insists on: a pure ratio
    detonates on a small denominator, so the 2% floor has to be able to veto it.
    """
    # Both strata stay well clear of the 500 floor, so only the relative limb is in play.
    selected = {0: 1000, 1: 99_000}
    realized = {0: 960, 1: 98_010}  # corpus-wide drop 1.03%; stratum 0 drops 4% (3.9x)
    outcome = validate.thin_stratum(realized, selected, thresholds=thresholds)
    assert outcome.measured["per_stratum"]["0"]["drop_rate_ratio"] > 3.0
    assert outcome.fired is True  # 4% loss clears the 2% floor

    realized_mild = {0: 990, 1: 98_010}  # stratum 0 drops 1% of its allocation, ratio ~1
    mild = validate.thin_stratum(realized_mild, selected, thresholds=thresholds)
    assert mild.fired is False


def test_zero_corpus_wide_attrition_does_not_divide_by_zero(
    thresholds: validate.Thresholds,
) -> None:
    """A perfect run: no drops anywhere, so no ratio exists to compare against."""
    outcome = validate.thin_stratum({0: 1000}, {0: 1000}, thresholds=thresholds)
    assert outcome.fired is False


# --- E2.2d coastal floor ------------------------------------------------------


def test_coastal_floor_is_the_rate_not_the_count(thresholds: validate.Thresholds) -> None:
    """670/50,000 honoured as 1.337% survives a shrunken denominator; 670 would not."""
    outcome = validate.coastal_floor(669, 49_998, thresholds=thresholds)
    assert outcome.fired is False
    assert outcome.measured["realized_rate"] == pytest.approx(669 / 49_998)
    assert outcome.measured["realized_coastal"] == 669  # reported as itself, never as 670


def test_a_breached_coastal_rate_is_a_finding(thresholds: validate.Thresholds) -> None:
    outcome = validate.coastal_floor(600, 49_998, thresholds=thresholds)
    assert outcome.fired is True
    assert outcome.failure_class == "finding"
    assert "coastal" in (outcome.implicated_knob or "")


# --- E2.3 per-COG guard -------------------------------------------------------


def test_per_cog_guard_passes_a_healthy_spread(thresholds: validate.Thresholds) -> None:
    outcome = validate.per_cog_guard({"A": 44, "B": 2, "C": 1}, thresholds=thresholds)
    assert outcome.fired is False
    assert outcome.measured["max_windows_per_cog"] == 44


def test_per_cog_guard_fires_on_a_looping_read(thresholds: validate.Thresholds) -> None:
    """The tripwire's only job: materialization concentrating where selection did not."""
    outcome = validate.per_cog_guard({"A": 65, "B": 2}, thresholds=thresholds)
    assert outcome.fired is True
    assert outcome.failure_class == "defect"
    assert outcome.measured["offending_cogs"] == ["A"]


# --- E2.5 gate x stratum bias -------------------------------------------------


def test_even_attrition_does_not_fire(thresholds: validate.Thresholds) -> None:
    outcome = validate.gate_stratum_bias(
        fires={"edited": {0: 100, 1: 100}},
        membership={0: 1000, 1: 1000},
        corpus_fires={"edited": 200},
        corpus_rows=2000,
        thresholds=thresholds,
    )
    assert outcome.fired is False


def test_a_hollowed_stratum_fires_with_its_knob(thresholds: validate.Thresholds) -> None:
    """The plan-risk-#3 case: one gate takes a big, concentrated bite of one stratum."""
    outcome = validate.gate_stratum_bias(
        fires={"edited": {0: 400, 1: 20}},
        membership={0: 1000, 1: 9000},
        corpus_fires={"edited": 420},
        corpus_rows=10_000,
        thresholds=thresholds,
    )
    assert outcome.fired is True
    assert outcome.failure_class == "finding"
    assert outcome.implicated_knob == "EDM/FLM cap"  # the plan §215-228 row, verbatim
    assert outcome.measured["fired_pairs"][0]["stratum_id"] == 0


def test_a_gate_with_no_row_in_the_knob_table_is_surfaced(
    thresholds: validate.Thresholds,
) -> None:
    """The bar's E2.5 clause names only the two cap gates; the screens have no knob.

    A ``degenerate`` firing is a real firing that the pre-registered payload cannot hand
    a knob to. It is reported as such rather than given an invented one — the gap is for
    the Gate E review to adjudicate, not for this module to close.
    """
    outcome = validate.gate_stratum_bias(
        fires={"degenerate": {0: 400, 1: 20}},
        membership={0: 1000, 1: 9000},
        corpus_fires={"degenerate": 420},
        corpus_rows=10_000,
        thresholds=thresholds,
    )
    assert outcome.fired is True
    assert outcome.measured["pairs_without_a_named_knob"] == ["degenerate"]
    assert outcome.implicated_knob is None
    with pytest.raises(ValueError, match="without an implicated knob"):
        outcome.payload("docs/research/e2-validation-report.md")


def test_concentration_without_a_substantial_bite_does_not_fire(
    thresholds: validate.Thresholds,
) -> None:
    """5x the corpus rate, but the gate removes 5% of the stratum — not 'hollowed'."""
    outcome = validate.gate_stratum_bias(
        fires={"edited": {0: 50, 1: 90}},
        membership={0: 1000, 1: 9000},
        corpus_fires={"edited": 140},
        corpus_rows=10_000,
        thresholds=thresholds,
    )
    assert outcome.measured["per_pair"]["edited"]["0"]["rate_ratio"] > 3.0
    assert outcome.fired is False


def test_the_baseline_population_is_named_and_a_sensitivity_never_gates(
    thresholds: validate.Thresholds,
) -> None:
    """Which denominator the ratio limb used is recorded, and only one of them gates.

    The bar's ratio limb reads on the whole catalog. A narrower baseline is a defensible
    alternative and it changes which pairs fire — which is exactly why re-running the
    rule over it must be a *published sensitivity*, not a second verdict. Picking the
    denominator after seeing which one fires is the peeking the pre-registration exists
    to prevent.
    """
    kwargs = {
        "fires": {"water": {0: 300, 1: 20}},
        "membership": {0: 1000, 1: 9000},
        "thresholds": thresholds,
    }
    gating = validate.gate_stratum_bias(
        corpus_fires={"water": 6_000}, corpus_rows=50_000, baseline="whole catalog", **kwargs
    )
    assert gating.gates is True and gating.failure_class == "finding"
    assert gating.fired is False  # 30% against a 12% baseline is 2.5x — under the 3x limb
    assert gating.measured["baseline_population"] == "whole catalog"
    assert gating.measured["corpus_wide_rates"]["water"] == pytest.approx(0.12)

    sensitivity = validate.gate_stratum_bias(
        corpus_fires={"water": 320}, corpus_rows=10_000, baseline="assignable only",
        criterion="E2.5_baseline_sensitivity", gates_verdict=False, **kwargs
    )
    assert sensitivity.fired is True  # 30% vs 3.2% is 9.4x — the same pair, other baseline
    assert sensitivity.gates is False
    assert sensitivity.failure_class == "published"
    assert sensitivity.implicated_knob is None
    assert sensitivity.criterion == "E2.5_baseline_sensitivity"


def test_an_empty_stratum_is_skipped_not_divided_by(thresholds: validate.Thresholds) -> None:
    outcome = validate.gate_stratum_bias(
        fires={"edited": {0: 0}},
        membership={0: 0},
        corpus_fires={"edited": 0},
        corpus_rows=0,
        thresholds=thresholds,
    )
    assert outcome.fired is False


# --- The payload handoff ------------------------------------------------------


def test_a_fired_finding_emits_the_bars_four_fields(thresholds: validate.Thresholds) -> None:
    """#47's finding_payload_schema, emitted mechanically so the fresh map is usable."""
    outcome = validate.coastal_floor(600, 49_998, thresholds=thresholds)
    payload = outcome.payload("docs/research/e2-validation-report.md#e22d")
    assert payload is not None
    assert set(payload) == {"implicated_knob", "measured_miss", "evidence_pointer", "class"}
    assert payload["class"] == "Finding"
    assert "1.337" in payload["measured_miss"] or "0.01337" in payload["measured_miss"]


def test_a_passing_criterion_emits_no_payload(thresholds: validate.Thresholds) -> None:
    assert validate.coastal_floor(700, 49_998, thresholds=thresholds).payload("x") is None
