"""The Gate F criterion rules — the 500 k scale tier, bar pre-registered by #69.

Zero-network, zero-corpus, exactly like ``test_validate.py``: every case is a hand-built
handful of rows, so this suite pins the *rules*
[`f-gate-bar.json`](../docs/research/f-gate-bar.json) committed before the 500 k draw
existed. The 500 k measurement lives in the Phase-F report (#73).

The load-bearing checks are the ones a wrong report would pass silently:

1. **The thresholds come off the committed bar**, and a bar whose prose has drifted from
   this module raises on load. Gate F strengthens this: the anchors *interpolate* their
   constants, so unlike Gate E an edit to a constant alone cannot pass.
2. **Gate E's bar is untouched.** ADR-0006 and ADR-0007 rest on it, so the two modules
   must disagree about ``thin_absolute`` — 500 there, 5,000 here — and both must load.
3. **F3 is single-limb and two-tier.** The corroborating strata never fire on their own,
   and the gating ones fire on the realized share alone; exhaustion is reported, not
   gated. The bar records *why* the conjunction was unsatisfiable, and a later edit that
   reinstates it should break this suite.
4. **The pixel check discriminates.** A shuffled join must fire; the benign quantization
   tail must not.
5. **A fired criterion carries its payload.** Every Finding emits the bar's four fields;
   a Finding that cannot name a knob is a bug in the handoff.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset import validate, validate_f

REPO = Path(__file__).resolve().parents[1]
F_BAR_PATH = REPO / "docs" / "research" / "f-gate-bar.json"
E_BAR_PATH = REPO / "docs" / "research" / "e2-gate-bar.json"


@pytest.fixture(scope="module")
def bar() -> validate_f.FBar:
    return validate_f.read_bar(F_BAR_PATH)


@pytest.fixture(scope="module")
def ft(bar: validate_f.FBar) -> validate_f.FThresholds:
    return bar.f


# --- The bar itself -----------------------------------------------------------


def test_thresholds_are_the_committed_numbers(bar: validate_f.FBar) -> None:
    """#69's registered values, read off the file rather than typed in here twice."""
    assert bar.thresholds.permutations == 10000
    assert bar.thresholds.null_percentile == 99.0
    assert bar.thresholds.thin_absolute == 5000
    assert bar.thresholds.thin_rate_ratio == 3.0
    assert bar.thresholds.thin_loss_frac == 0.02
    assert bar.thresholds.coastal_min_rate == 0.01337
    assert bar.f.pixel_rel_tol == 0.5
    assert bar.f.pixel_share_max == 0.001
    assert bar.f.drainage_min_share == 0.25
    assert bar.f.drop_rate_max == 0.0004
    assert bar.f.expected_drops == 20


def test_the_committed_allocation_is_the_scale_budget(bar: validate_f.FBar) -> None:
    """Neyman shares are budget-invariant, so F's table is the same rule at 500 k.

    Pinned here because F2a checks the drawn list against it and F2c's absolute floor is
    only inert *because* the smallest share clears 5,000.
    """
    n_h = bar.expected_per_stratum
    assert sum(n_h.values()) == validate_f.BUDGET == 500_000
    assert min(n_h.values()) == 7343
    assert min(n_h.values()) > bar.thresholds.thin_absolute
    assert bar.allocation["coast_floor"] == 6700
    assert bar.allocation["floor_clamp"] == 5000
    assert bar.allocation["floor_clamp_strata_below"] == 0


def test_per_cog_cap_is_retired_not_rescaled(bar: validate_f.FBar) -> None:
    """A 1° COG holds at most 256 windows, so no absolute cap survives a 16.4 % draw.

    The sentinel is negative so a stray :func:`dataset.validate.per_cog_guard` call cannot
    quietly measure against something plausible-looking; F6b bounds by inventory instead.
    """
    assert bar.thresholds.per_cog_cap < 0
    assert "E2.3_coverage.guard_per_cog_cap" in bar.doc["what_changed_from_gate_e"]["retired_as_gating"]


def test_a_bar_whose_prose_lost_the_number_raises(tmp_path: Path) -> None:
    """The anchor check is the tripwire: prose and code cannot drift apart quietly."""
    doc = json.loads(F_BAR_PATH.read_text())
    doc["criteria"]["F2c_thin_stratum"]["conditions"]["absolute"] = "every realized n_h >= 4000"
    path = tmp_path / "moved-bar.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="5000"):
        validate_f.read_bar(path)


def test_word_boundaries_stop_a_substring_from_satisfying_an_anchor(tmp_path: Path) -> None:
    """``5000`` must not be satisfied by a bar edited to say ``50000``."""
    doc = json.loads(F_BAR_PATH.read_text())
    doc["criteria"]["F2c_thin_stratum"]["conditions"]["absolute"] = "every realized n_h >= 50000"
    path = tmp_path / "inflated-bar.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="5000"):
        validate_f.read_bar(path)


def test_gate_e_bar_still_loads_against_its_own_module() -> None:
    """The whole reason this is a sibling module: E's registration must not move.

    ADR-0006 and ADR-0007 rest on Gate E's numbers. If rescaling the floor for F had been
    done in place, E's bar would now be measured against 5,000.
    """
    e = validate.read_bar(E_BAR_PATH)
    f = validate_f.read_bar(F_BAR_PATH)
    assert e.thresholds.thin_absolute == 500
    assert f.thresholds.thin_absolute == 5000
    assert e.thresholds.per_cog_cap == 64


def test_normalization_branch_is_read_not_assumed(bar: validate_f.FBar) -> None:
    """#70 may select a branch; it may not invent one."""
    assert bar.normalization_branches == ("tier_specific", "inherit_debug")
    assert bar.live_normalization_branch("tier_specific") == "tier_specific"
    with pytest.raises(ValueError, match="unregistered"):
        bar.live_normalization_branch("recompute_per_shard")


# --- F1d pixel integrity ------------------------------------------------------


def test_pixel_integrity_passes_the_quantization_tail(ft: validate_f.FThresholds) -> None:
    """Agreement at the int16 floor, plus a couple of benign clip/mask disagreements.

    The share gates, not the max, precisely so a healthy corpus is not failed by the few
    tiles where the plausibility clip and the land-mask definition genuinely differ.
    """
    catalog = np.full(2000, 10.0)
    manifest = catalog * (1.0 + 8.4e-5)
    manifest[0] = catalog[0] * 1.46  # the debug tier's worst benign case, 0.46 relative
    out = validate_f.pixel_integrity(manifest, catalog, thresholds=ft)
    assert out.fired is False
    # 0.46 sits *below* the tolerance — which is where 0.5 was placed, above the whole
    # benign tail the debug tier measured and below the mixup signature.
    assert out.measured["n_over_tol"] == 0
    assert out.measured["max_rel_diff"] == pytest.approx(0.46)

    # And the share allowance absorbs a couple that do clear it: 1 in 2,000 is 0.0005,
    # inside the registered 0.001, so a lone outlier is not a Defect.
    manifest[1] = catalog[1] * 1.8
    lone = validate_f.pixel_integrity(manifest, catalog, thresholds=ft)
    assert lone.fired is False
    assert lone.measured["share_over_tol"] == pytest.approx(1 / 2000)

    # Five is 0.0025, outside it.
    manifest[2:6] = catalog[2:6] * 1.8
    many = validate_f.pixel_integrity(manifest, catalog, thresholds=ft)
    assert many.fired is True


def test_pixel_integrity_fires_on_a_mis_join(ft: validate_f.FThresholds) -> None:
    """A wrong-window shard changes land-std wholesale; half the corpus trips."""
    rng = np.random.default_rng(0)
    catalog = rng.uniform(3.0, 200.0, size=2000)
    manifest = catalog[rng.permutation(2000)]
    out = validate_f.pixel_integrity(manifest, catalog, thresholds=ft)
    assert out.fired is True
    assert out.measured["share_over_tol"] > 0.5
    assert out.payload("docs/research/f-validation-report.md")["class"] == "Defect"


def test_pixel_integrity_refuses_a_ragged_join(ft: validate_f.FThresholds) -> None:
    """Unequal lengths mean the join itself dropped rows — the defect, not an input bug."""
    with pytest.raises(ValueError, match="join"):
        validate_f.pixel_integrity(np.ones(5), np.ones(4), thresholds=ft)


# --- F3 drainage representation -----------------------------------------------


def _texture(n: int, n_drainage: int, p75: float) -> np.ndarray:
    """``n`` shipped tiles of which ``n_drainage`` reach the pre-gate p75."""
    out = np.full(n, p75 / 2.0)
    out[:n_drainage] = p75 * 1.5
    return out


def test_drainage_passes_when_the_corpus_carries_the_terrains_share(
    ft: validate_f.FThresholds,
) -> None:
    """Coverage-seeking selection reaching ~0.26 clears the pre-gate reference of 0.25."""
    p75 = ft.drainage_texture_p75
    shipped = {
        2: _texture(23159, 6021, p75[2]),
        17: _texture(18472, 4803, p75[17]),
        5: _texture(19909, 3000, p75[5]),
        7: _texture(26043, 8000, p75[7]),
    }
    out = validate_f.drainage_representation(
        shipped, kept_tail={2: 26271, 17: 17930, 5: 21250, 7: 38591}, thresholds=ft
    )
    assert out.fired is False
    assert out.measured["per_stratum"]["2"]["realized_share_r_h"] > 0.25
    assert out.payload("x") is None


def test_drainage_fires_on_proportional_drawing(ft: validate_f.FThresholds) -> None:
    """Drawing at the kept pool's own 0.174 share is the failure mode the bar predicts."""
    p75 = ft.drainage_texture_p75
    shipped = {
        2: _texture(23159, round(0.174 * 23159), p75[2]),
        17: _texture(18472, round(0.170 * 18472), p75[17]),
    }
    out = validate_f.drainage_representation(
        shipped, kept_tail={2: 26271, 17: 17930}, thresholds=ft
    )
    assert out.fired is True
    assert out.measured["hits"] == [2, 17]
    payload = out.payload("docs/research/f-validation-report.md")
    assert payload["class"] == "Finding"
    assert payload["implicated_knob"] == "degenerate flatness screen (`land_std` cap)"
    assert "0.25" in payload["measured_miss"] or "0.174" in payload["measured_miss"]


def test_corroborating_strata_never_fire_alone(ft: validate_f.FThresholds) -> None:
    """#63's two-tier rule, carried forward: 5 and 7 are reported, never triggers.

    ADR-0007 scoped its revisit trigger to 2/17 — stratum 5 drops genuine junk and
    stratum 7's texture is sastrugi noise running the opposite way.
    """
    p75 = ft.drainage_texture_p75
    shipped = {
        2: _texture(23159, 6021, p75[2]),
        17: _texture(18472, 4803, p75[17]),
        5: _texture(19909, 0, p75[5]),  # catastrophic in a corroborating stratum
        7: _texture(26043, 0, p75[7]),
    }
    out = validate_f.drainage_representation(
        shipped, kept_tail={2: 26271, 17: 17930, 5: 21250, 7: 38591}, thresholds=ft
    )
    assert out.fired is False
    assert out.measured["per_stratum"]["5"]["realized_share_r_h"] == 0.0
    assert out.measured["per_stratum"]["5"]["gates"] is False


def test_exhaustion_is_reported_but_does_not_gate(ft: validate_f.FThresholds) -> None:
    """The conjunction the bar records as unsatisfiable must not have been reinstated.

    High exhaustion with an adequate share is a *pass*: the survivors were consumed, but
    the corpus still carries the terrain's drainage share. Only ``r_h`` gates.
    """
    p75 = ft.drainage_texture_p75
    shipped = {2: _texture(23159, 20000, p75[2]), 17: _texture(18472, 15000, p75[17])}
    out = validate_f.drainage_representation(
        shipped, kept_tail={2: 26271, 17: 17930}, thresholds=ft
    )
    assert out.fired is False
    assert out.measured["per_stratum"]["2"]["exhaustion_e_h"] > 0.75
    assert out.measured["per_stratum"]["17"]["exhaustion_e_h"] > 0.8


# --- F6a / F6b ----------------------------------------------------------------


def test_distinctness_catches_a_duplicate_the_count_rules_miss() -> None:
    """The hole F2a and F2b both pass: a repeat inside the selection list.

    md5, row count, per-stratum counts and ``realized <= selected`` all stay true.
    """
    out = validate_f.distinctness(
        selection_window_ids=["a:1:1", "b:1:1", "a:1:1"],
        manifest_tile_ids=["a_1_1", "b_1_1"],
    )
    assert out.fired is True
    assert out.measured["n_selection_duplicated"] == 1
    assert out.payload("x")["class"] == "Defect"


def test_distinctness_passes_a_clean_draw() -> None:
    out = validate_f.distinctness(
        selection_window_ids=["a:1:1", "b:1:1"], manifest_tile_ids=["a_1_1", "b_1_1"]
    )
    assert out.fired is False
    assert out.measured["n_manifest_distinct"] == 2


def test_cog_inventory_bound_fires_past_a_cogs_own_supply() -> None:
    out = validate_f.cog_inventory_bound({"cog_a": 300, "cog_b": 40}, {"cog_a": 256, "cog_b": 122})
    assert out.fired is True
    assert out.measured["offenders"]["cog_a"] == {"realized": 300, "inventory": 256}


def test_cog_inventory_bound_allows_a_deep_but_legal_draw() -> None:
    """A COG fully consumed is legal at 16.4 % average — which is why the cap had to go."""
    out = validate_f.cog_inventory_bound({"cog_a": 256, "cog_b": 122}, {"cog_a": 256, "cog_b": 122})
    assert out.fired is False


def test_an_unknown_cog_is_itself_a_breach() -> None:
    """The draw produced a source the kept pool does not contain."""
    out = validate_f.cog_inventory_bound({"cog_ghost": 1}, {"cog_a": 256})
    assert out.fired is True


# --- F7 -----------------------------------------------------------------------


def test_attrition_passes_the_expected_handful(ft: validate_f.FThresholds) -> None:
    """~20 drops at 500 k is the pre-registered expectation, derived from the flip band."""
    out = validate_f.attrition_rate(500_000, 499_980, thresholds=ft)
    assert out.fired is False
    assert out.measured["n_dropped"] == 20
    assert out.measured["expected_drops"] == 20
    assert out.measured["expected_2sigma"] == [11, 29]


def test_attrition_fires_on_a_ten_fold_rate_excursion(ft: validate_f.FThresholds) -> None:
    """A read-path or clip-bound regression, not a design flaw — hence Defect."""
    out = validate_f.attrition_rate(500_000, 499_700, thresholds=ft)
    assert out.fired is True
    assert out.measured["drop_rate"] > ft.drop_rate_max
    assert out.payload("x")["class"] == "Defect"


def test_attrition_is_a_rate_so_it_reads_the_same_at_any_budget(
    ft: validate_f.FThresholds,
) -> None:
    """The same proportional loss must give the same verdict at 50 k and 500 k."""
    small = validate_f.attrition_rate(50_000, 49_998, thresholds=ft)
    large = validate_f.attrition_rate(500_000, 499_980, thresholds=ft)
    assert small.fired is large.fired is False


def test_unregistered_ledger_status_surfaces_as_itself() -> None:
    """A handful of I/O errors and a handful of boundary re-screens are different news."""
    out = validate_f.registered_status_set({"ok": 499_980, "degenerate": 18, "io_error": 2})
    assert out.fired is True
    assert out.measured["unregistered"] == {"io_error": 2}


def test_the_debug_tiers_own_status_set_passes() -> None:
    out = validate_f.registered_status_set({"ok": 49_998, "degenerate": 2})
    assert out.fired is False


# --- F8 -----------------------------------------------------------------------


def test_tier_specific_gates_against_the_recomputed_statistic(bar: validate_f.FBar) -> None:
    recorded = {"global_mean": 900.5, "global_std": 1100.25}
    out = validate_f.normalization_statistic(
        branch="tier_specific", recorded=recorded, recomputed=recorded, bar=bar
    )
    assert out.fired is False
    out_bad = validate_f.normalization_statistic(
        branch="tier_specific",
        recorded=recorded,
        recomputed={"global_mean": 900.5, "global_std": 1100.26},
        bar=bar,
    )
    assert out_bad.fired is True


def test_inherit_debug_gates_the_recorded_values_and_publishes_the_divergence(
    bar: validate_f.FBar,
) -> None:
    """Under this branch the 500 k recompute is a diagnostic, not a threshold."""
    out = validate_f.normalization_statistic(
        branch="inherit_debug",
        recorded=dict(validate_f.DEBUG_TIER_STATS),
        recomputed={"global_mean": 900.5, "global_std": 1100.25},
        bar=bar,
    )
    assert out.fired is False
    assert out.measured["divergence_diagnostic"]["global_mean"]["relative"] > 0.0


# --- Reuse of the frozen module -----------------------------------------------


def test_carried_over_rules_are_relabelled_for_f(bar: validate_f.FBar) -> None:
    """A Gate F artifact naming ``E2.2c`` would send a reader to the wrong bar."""
    out = validate_f.thin_stratum(
        {0: 32200, 6: 7343}, {0: 32200, 6: 7343}, thresholds=bar.thresholds
    )
    assert out.criterion == "E2.2c_thin_stratum"
    assert validate_f.relabel(out, "F2c_thin_stratum").criterion == "F2c_thin_stratum"
    assert validate_f.relabel(out, "F2c_thin_stratum").fired is out.fired


def test_the_rescaled_floor_actually_bites_at_f(bar: validate_f.FBar) -> None:
    """500 would have passed a stratum that 5,000 catches — the point of rescaling."""
    realized = {0: 32200, 6: 4000}
    e_bar = validate.read_bar(E_BAR_PATH)
    assert validate_f.thin_stratum(realized, {0: 32200, 6: 7343}, thresholds=e_bar.thresholds).measured[
        "below_absolute"
    ] == []
    assert validate_f.thin_stratum(realized, {0: 32200, 6: 7343}, thresholds=bar.thresholds).measured[
        "below_absolute"
    ] == [6]
