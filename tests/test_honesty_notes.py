"""The corpus card (#75) — its prose must not drift from its derived numbers.

[`f-honesty-notes.md`](../docs/research/f-honesty-notes.md) is written for a reader outside
this project, and a Phase-4 model card will be lifted from it months after anyone here last
looked at the evidence. That is the worst possible place for a stale number: the downstream
reader has no way to check it, and the two hand-carried-number bugs this map already found
([#71](https://github.com/JStoschek/DEMAE/issues/71)'s 4-dp ``n_h`` table,
[#73](https://github.com/JStoschek/DEMAE/issues/73)'s 6-dp threshold) both looked exactly
like correct documents until someone recomputed them.

So the card quotes [`f-honesty-notes.json`](../docs/research/f-honesty-notes.json), the driver
derives that JSON from committed artifacts, and this suite asserts the two agree. The load-
bearing checks are the ones a wrong card would otherwise pass silently:

1. **Every headline figure in the prose is present in the JSON**, at the precision the prose
   prints. Editing a number in the Markdown alone breaks this.
2. **The JSON's own numbers are consistent with their sources** — the card cannot claim a
   tile count, a drop count or a coverage figure that the validation report and the corpus
   metadata do not agree on.
3. **The caveats that have no number say so explicitly.** The withheld-country gap is
   unmeasurable, and an *absent* entry looks identical to a forgotten one — so it is pinned
   as an explicit null with a stated reason.
4. **The recomputed normalization figures reproduce the debug tier's**, which is the claim
   §4.8 makes in prose and the reason re-measuring was worth doing at all.
5. **The drainage attribution keeps both halves.** A card publishing the shortfall without
   the exhaustion would point every reader at the wrong knob.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "docs" / "research"
CARD_MD = RESEARCH / "f-honesty-notes.md"
CARD_JSON = RESEARCH / "f-honesty-notes.json"
REPORT_JSON = RESEARCH / "f-validation-report.json"
VERDICT_JSON = RESEARCH / "f-gate-verdict.json"
META_JSON = REPO / "data" / "corpus_500k" / "corpus_meta.json"


@pytest.fixture(scope="module")
def card() -> dict:
    if not CARD_JSON.exists():
        pytest.skip(f"{CARD_JSON} not built — run notebooks/f_honesty_notes.py")
    return json.loads(CARD_JSON.read_text())


@pytest.fixture(scope="module")
def prose() -> str:
    if not CARD_MD.exists():
        pytest.skip(f"{CARD_MD} not written")
    return CARD_MD.read_text()


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT_JSON.read_text())


@pytest.fixture(scope="module")
def meta() -> dict:
    return json.loads(META_JSON.read_text())


def _thousands(n: int) -> str:
    return f"{n:,}"


# ---------------------------------------------------------------------------
# 1. the prose quotes the JSON
# ---------------------------------------------------------------------------


def test_headline_counts_appear_in_prose(card: dict, prose: str) -> None:
    """The counts a reader will cite must be in the prose exactly as derived."""
    corpus = card["corpus"]
    for value in (
        corpus["n_tiles_shipped"],
        corpus["n_selected"],
        corpus["n_dropped_at_materialization"],
    ):
        assert _thousands(value) in prose, f"{value:,} missing from the card's prose"


def test_fingerprints_appear_in_prose(card: dict, prose: str) -> None:
    """A card whose `dataset_version` has drifted identifies the wrong corpus."""
    corpus = card["corpus"]
    assert corpus["dataset_version"] in prose
    assert corpus["selection_list_md5"] in prose


def test_coverage_figures_appear_in_prose(card: dict, prose: str) -> None:
    cov = card["coverage"]
    assert str(cov["catalog_cells_occupied"]) in prose
    assert str(cov["corpus_cells_reached"]) in prose
    assert str(cov["hollow_cells"]["n"]) in prose
    assert str(cov["debug_tier_unreached"]) in prose


def test_redundancy_figures_appear_in_prose(card: dict, prose: str) -> None:
    """§4.6 is the caveat a reader can act on, so its numbers are pinned hardest."""
    red = card["spatial_redundancy"]
    assert _thousands(red["n_cogs_at_or_over_half"]) in prose
    assert str(red["n_cogs_fully_drawn"]) in prose
    assert str(red["max_windows_per_cog"]) in prose
    assert str(red["max_inventory_per_cog"]) in prose


def test_drainage_shares_appear_in_prose(card: dict, prose: str) -> None:
    finding = card["drainage_finding"]
    for sid in finding["gating_strata"]:
        row = finding["per_stratum"][sid]
        assert f"{row['realized_share_r_h']:.4f}" in prose
        assert f"{row['exhaustion_e_h']:.3f}" in prose


def test_normalization_figures_appear_in_prose(card: dict, prose: str) -> None:
    norm = card["normalization"]
    assert f"{norm['median_tile_std_m']:.1f}" in prose
    assert f"{norm['median_tile_std_normalized']:.3f}" in prose
    assert f"{norm['frac_tiles_under_cut'] * 100:.1f}%" in prose
    assert f"{norm['altitude_over_relief_ratio']:.1f}" in prose


def test_edited_pixel_figures_appear_in_prose(card: dict, prose: str) -> None:
    edited = card["edited_pixels"]
    assert _thousands(edited["n_dropped"]) in prose
    assert _thousands(edited["n_would_be_members"]) in prose
    assert f"{edited['within_stratum_drop'] * 100:.1f}%" in prose
    assert f"{edited['ratio_to_corpus_rate']:.1f}" in prose


# ---------------------------------------------------------------------------
# 2. the JSON agrees with its sources
# ---------------------------------------------------------------------------


def test_tile_count_matches_corpus_metadata(card: dict, meta: dict) -> None:
    assert card["corpus"]["n_tiles_shipped"] == meta["n_tiles"]
    assert card["corpus"]["dataset_version"] == meta["dataset_version"]


def test_drop_count_is_the_arithmetic_not_an_assertion(card: dict, report: dict) -> None:
    """The 66 must be selected-minus-shipped, and agree with the attrition evidence.

    Two independent routes to the same number: if the card ever quotes a drop count that
    is not the difference between the two counts it also quotes, it is internally
    inconsistent in a way a reader could catch and we could not.
    """
    corpus = card["corpus"]
    derived = corpus["n_selected"] - corpus["n_tiles_shipped"]
    assert corpus["n_dropped_at_materialization"] == derived
    assert derived == report["attrition_evidence"]["n_dropped"]


def test_verdict_matches_the_committed_verdict(card: dict) -> None:
    verdict = json.loads(VERDICT_JSON.read_text())
    assert card["gate_f"]["verdict"] == verdict["verdict"] == "FAIL"


def test_no_defect_open_at_verdict(card: dict) -> None:
    """The card's scoping claim — a Finding failed the gate, not a build defect."""
    gate = card["gate_f"]
    assert gate["defects_open_at_verdict"] == []
    assert gate["findings_fired"] == ["F3_drainage_representation"]
    for defect in gate["defects_fired_at_report"]:
        assert defect in gate["set_aside_at_verdict"], f"{defect} fired but was never dispositioned"


def test_criterion_counts_come_off_the_bar_not_the_report(card: dict, prose: str) -> None:
    """The card claims a bar fixed in advance, so it must count what was *registered*.

    The report emits one more row than the bar registers — `F3_p75_precision_sensitivity`,
    a published sensitivity that was never a criterion. Counting emitted rows would overstate
    the pre-registered set by one in the very document whose claim is that it was pinned
    before the corpus existed.
    """
    bar = json.loads((RESEARCH / "f-gate-bar.json").read_text())
    gate = card["gate_f"]
    assert gate["n_criteria_registered"] == len(bar["criteria"]) == 18
    assert gate["n_criteria_gating"] == sum(1 for c in bar["criteria"].values() if c.get("gates"))
    assert gate["n_criteria_emitted"] > gate["n_criteria_registered"]
    assert gate["emitted_but_not_registered"] == ["F3_p75_precision_sensitivity"]
    assert f"{gate['n_criteria_registered']} criteria" in prose
    assert f"{gate['n_criteria_gating']} of them gating" in prose


def test_pixel_integrity_claim_matches_the_report(card: dict, report: dict) -> None:
    measured = report["criteria"]["F1d_pixel_integrity"]["measured"]
    integrity = card["corpus"]["pixel_integrity"]
    assert integrity["n_compared"] == measured["n_compared"]
    assert integrity["n_over_tolerance"] == measured["n_over_tol"] == 0


# ---------------------------------------------------------------------------
# 3. the caveats without numbers stay explicit
# ---------------------------------------------------------------------------


def test_withheld_country_gap_is_an_explicit_null(card: dict) -> None:
    """An absent entry and a forgotten one look identical — so this one is pinned.

    The gap cannot be sized from this side: a withheld cell is an absent prefix, and ocean
    is absent for the same reason. The card must carry that as a stated null with a reason,
    never as a number and never as silence.
    """
    withheld = card["coverage"]["withheld_countries"]
    assert withheld["n"] is None
    assert withheld["measured"] is False
    assert withheld["why_unmeasurable"]


def test_dsm_canopy_note_states_it_is_the_whole_mitigation(card: dict) -> None:
    """The vegetation column was deliberately not built; the note must not imply otherwise."""
    dsm = card["dsm_canopy"]
    assert dsm["measured"] is False
    assert "not built" in dsm["vegetation_column"]
    assert dsm["promotion_trigger"]


def test_unbuilt_items_each_carry_a_trigger(card: dict) -> None:
    for key, state in card["not_here"].items():
        assert state, f"{key} declared unbuilt with no stated reason"


# ---------------------------------------------------------------------------
# 4. the re-measurement reproduces what it was checked against
# ---------------------------------------------------------------------------


def test_normalization_reproduces_the_debug_tier(card: dict) -> None:
    """§4.8 claims the finding reproduces at scale — this is that claim, as a test.

    Loose tolerances on purpose: the two tiers are independent draws, so the numbers should
    be *close*, not equal. A tight assertion here would be asserting something false.
    """
    norm = card["normalization"]
    ref = norm["debug_tier_reference"]
    assert norm["median_tile_std_normalized"] == pytest.approx(
        ref["median_tile_std_normalized"], abs=0.005
    )
    assert norm["frac_tiles_under_cut"] == pytest.approx(ref["frac_tiles_under_cut"], abs=0.05)
    assert norm["altitude_over_relief_ratio"] == pytest.approx(
        ref["altitude_over_relief_ratio"], rel=0.10
    )


def test_normalization_stats_are_tier_specific(card: dict, meta: dict) -> None:
    """The card tells the reader these constants are not portable — ADR-0003 Amendment 1."""
    norm = card["normalization"]
    assert norm["normalization_stat_source"] == meta["normalization_stat_source"] == "tier_specific"
    assert norm["global_mean_m"] == meta["global_mean"]
    assert norm["global_std_m"] == meta["global_std"]


# ---------------------------------------------------------------------------
# 5. the drainage finding keeps its attribution
# ---------------------------------------------------------------------------


def test_drainage_finding_publishes_exhaustion_beside_the_shortfall(card: dict) -> None:
    """Shortfall alone points at the flatness screen; exhaustion is what refutes that."""
    finding = card["drainage_finding"]
    assert finding["fired"] is True
    assert finding["gating_strata"] == ["2", "17"]
    for sid in finding["gating_strata"]:
        row = finding["per_stratum"][sid]
        assert row["realized_share_r_h"] < finding["min_share"]
        assert 0.0 < row["exhaustion_e_h"] < 0.5, "the surviving tail was mostly not drawn"
        assert row["tail_headroom_over_shortfall"] > 1.0, (
            "the untouched tail must be able to cover the shortfall — that is the "
            "attribution the card rests on"
        )


def test_prose_names_sampler_reach_not_the_screen(prose: str) -> None:
    """The card must not leave a reader with the obvious-but-wrong fix.

    §3.1's whole point is that the flatness screen is not the binding constraint. If the
    attribution sentence is ever edited away, the numbers above stay true and the document
    starts misleading.
    """
    section = prose.split("### 3.1")[1].split("### 3.2")[0]
    assert re.search(r"never drawn|not reach|did not reach", section), (
        "§3.1 no longer states that the drainage tail was left undrawn"
    )
    assert "exhaustion" in section.lower()


def test_split_hazard_is_stated_where_the_redundancy_is(prose: str) -> None:
    """#75 folded the null `split` column into §4.6 so hook and hazard are in one place."""
    section = prose.split("### 4.6")[1].split("### 4.7")[0]
    assert "split" in section.lower()
    assert "geocell_id" in section, "the card must name the column to split on, not just warn"
