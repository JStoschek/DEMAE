"""`benchmark.prereg` — the registered bar and its §6-phrase refusal (#148).

Prior art is `tests/test_validate.py`, and deliberately: Gate E's bar and this one are the
same arrangement — a file committed before the numbers exist, and code that refuses to load it
once the prose it was registered in has moved. What is new here is the third link. Gate E ties
*code constant* to *file prose*; this ties **value → the §6 phrase it was registered in → §6
itself**, because the file is written by `freeze` rather than by hand, so a value and a phrase
that agree with each other prove nothing unless the phrase is still in the plan.

The load-bearing checks are the ones a silently-moved threshold would pass:

1. **A phrase edited out of §6 refuses the load**, and the refusal names the entry that lost
   its anchor rather than the file.
2. **A value moved in the file refuses the load too**, because the anchor it renders no longer
   reads in its own phrase. This is the edit the register exists to catch.
3. **The refusal fires at load**, not at the first call that reads a threshold.
4. **The five discursive rules and R9b are present and unmechanised**, not absent — the
   register makes them countable without claiming to enforce them (ADR-0013 Decision 8).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import prereg

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / prereg.PLAN_PATH
COMMITTED = REPO / "artifacts" / "task1" / prereg.BAR_NAME


@pytest.fixture
def bar_path(tmp_path: Path) -> Path:
    """The bar as `freeze` writes it, on a box of its own."""
    return prereg.write_bar(tmp_path / prereg.BAR_NAME, plan=PLAN)


@pytest.fixture
def bar(bar_path: Path) -> prereg.Bar:
    return prereg.read_bar(bar_path, plan=PLAN)


# --- What is in it ------------------------------------------------------------


def test_the_bar_carries_the_seven_mechanised_entries_with_their_values(bar: prereg.Bar) -> None:
    """ADR-0013's register, numeric subset: seven of the twelve rules carry a number."""
    assert tuple(entry.rule for entry in bar.mechanised) == (
        "R1", "R4", "R5", "R6", "R7", "R8", "R11",
    )
    assert bar.seam_break_deg == 0.15
    assert bar.cell_floor == 100
    assert bar.value("R4") == 20
    assert bar.value("R11") == 0.99


# --- The refusal --------------------------------------------------------------


def _plan_without(tmp_path: Path, phrase: str) -> Path:
    """The plan with one registered §6 phrase edited out, and §6 otherwise intact."""
    text = PLAN.read_text()
    section = text[text.index(prereg.SECTION):]
    edited = section.replace(phrase, "a seam gap above +0.25° counts as a seam break", 1)
    assert edited != section, f"the fixture did not find {phrase!r} in §6"
    path = tmp_path / "edited-plan.md"
    path.write_text(text[: text.index(prereg.SECTION)] + edited)
    return path


def test_a_plan_that_lost_a_registered_phrase_refuses_the_load(bar_path: Path, tmp_path) -> None:
    """Gate E's mechanism, on §6: the bar and the copy that ships cannot drift quietly.

    The phrase is edited in the *plan*, which is the direction that matters — §6 goes in the
    writeup verbatim, so its drift is the externally visible kind.
    """
    plan = _plan_without(tmp_path, "a seam gap above **+0.15°** counts as a seam break")
    with pytest.raises(ValueError, match="R7 lost its anchor"):
        prereg.read_bar(bar_path, plan=plan)


def test_the_refusal_names_the_entry_rather_than_the_file(bar_path: Path, tmp_path) -> None:
    # "the bar failed to load" sends a build session to the JSON; the entry that lost its
    # anchor sends it to the §6 sentence someone edited, which is where the decision was made.
    plan = _plan_without(tmp_path, "a seam gap above **+0.15°** counts as a seam break")
    with pytest.raises(ValueError) as exc:
        prereg.read_bar(bar_path, plan=plan)
    assert "R7" in str(exc.value)
    assert "counts as a seam break" in str(exc.value)
    assert prereg.ADR in str(exc.value)


def test_a_threshold_moved_in_the_file_refuses_too(bar_path: Path) -> None:
    """The link `dataset.validate` does not need and this one does.

    Gate E's bar was written by hand and its numbers live in code, so prose-versus-constant is
    the whole drift surface. This bar is *written by* `freeze`, so its value and its phrase come
    from one source: an edit to the file alone would otherwise leave both agreeing and neither
    right. The value renders its own anchor, so a moved threshold stops matching its phrase.
    """
    doc = json.loads(bar_path.read_text())
    for entry in doc["entries"]:
        if entry["rule"] == "R7":
            entry["value"] = 0.25
    bar_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=r"R7: the registered value renders 'above \+0.25°'"):
        prereg.read_bar(bar_path, plan=PLAN)


def test_a_rule_dropped_from_the_index_refuses_the_load(bar_path: Path, tmp_path) -> None:
    """§6 item 8 is the register's index, and the index is what makes the set countable."""
    text = PLAN.read_text()
    head, section = text[: text.index(prereg.SECTION)], text[text.index(prereg.SECTION):]
    path = tmp_path / "unindexed-plan.md"
    path.write_text(head + section.replace("**R12**", "**R13**"))
    with pytest.raises(ValueError, match="§6 no longer names R12"):
        prereg.read_bar(bar_path, plan=path)


def test_a_bar_missing_an_entry_refuses_rather_than_loading_a_partial_register(bar_path) -> None:
    """The register is whole or it is nothing.

    Checking only the entries a file happens to carry is this module's own failure mode: every
    anchor R7 no longer has still passes, the bar loads clean, and the first caller to ask for
    the seam threshold is the one who finds out. That is "at first use" wearing a load check.
    """
    doc = json.loads(bar_path.read_text())
    doc["entries"] = [entry for entry in doc["entries"] if entry["rule"] != "R7"]
    bar_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=r"is not the register: \['R7'\] missing"):
        prereg.read_bar(bar_path, plan=PLAN)


def test_a_bar_carrying_an_unregistered_entry_refuses_too(bar_path: Path) -> None:
    # A rule nobody registered is a threshold with no ADR behind it, which is the arrangement
    # inverted: the register admits entries, the file does not.
    doc = json.loads(bar_path.read_text())
    doc["entries"].append({"rule": "R13", "statement": "…", "home": "§5", "source": "nowhere"})
    bar_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=r"\['R13'\] unregistered"):
        prereg.read_bar(bar_path, plan=PLAN)


def test_the_refusal_fires_at_load_rather_than_at_first_use(bar_path: Path, tmp_path) -> None:
    """A threshold checked only where it is read is one that passes on any run that misses it.

    R8's floor is read by `report` and by nothing else, so a load that returned a `Bar` and
    refused later would let every `run` proceed against a bar the plan no longer states.
    """
    plan = _plan_without(tmp_path, "clears\n     **100 per band**")
    with pytest.raises(ValueError, match="R8 lost its anchor"):
        prereg.read_bar(bar_path, plan=plan)


def test_freeze_refuses_to_write_a_bar_that_would_not_load(tmp_path) -> None:
    # Checked before the write, not after: a `freeze` that leaves the file on disk and then
    # raises has committed the drift it was refusing.
    plan = _plan_without(tmp_path, "a seam gap above **+0.15°** counts as a seam break")
    path = tmp_path / prereg.BAR_NAME
    with pytest.raises(ValueError, match="R7 lost its anchor"):
        prereg.write_bar(path, plan=plan)
    assert not path.exists()


# --- What is deliberately not mechanised --------------------------------------


def test_the_discursive_rules_and_the_refusal_are_present_and_unmechanised(bar: prereg.Bar) -> None:
    """Recorded rather than omitted: the register makes them countable, not enforceable.

    ADR-0013 rejected mechanising them — symmetry is a property of how a sentence is used in an
    argument, and a tick-box asserting it would look like enforcement while checking nothing.
    Leaving them out of the file would have looked like the register held seven rules.
    """
    assert tuple(entry.rule for entry in bar.unmechanised) == (
        "R2", "R3", "R9", "R9b", "R10", "R12",
    )
    for entry in bar.unmechanised:
        assert entry.unmechanised_because, f"{entry.rule} is unmechanised without saying why"
        assert entry.value is None


def test_the_register_reads_twelve_rules(bar: prereg.Bar) -> None:
    # Thirteen entries, twelve rules: R9b is a refusal within R9's topic, not a thirteenth rule,
    # and ADR-0013's register, §6's index and the counts all read twelve.
    assert len(bar.entries) == 13
    assert bar.rules == 12
    assert len(bar.mechanised) == 7


def test_asking_a_discursive_rule_for_a_number_refuses_rather_than_returning_none(bar) -> None:
    """`None` would let a caller measure against nothing, which is the failure R9b is about."""
    with pytest.raises(ValueError, match="R9b carries no registered value"):
        bar.value("R9b")


def test_an_unregistered_rule_is_refused_by_name(bar: prereg.Bar) -> None:
    with pytest.raises(KeyError, match="R13"):
        bar.rule("R13")


# --- What reads it ------------------------------------------------------------


def test_r1s_roster_is_the_one_list_the_metric_gate_reads(bar: prereg.Bar) -> None:
    """Six printed columns — five graded plus the derived seam gap — and four appendix (#146).

    The seam gap is *in* the roster rather than carved out by name beside it: a second home for
    the metric set is what produced #146 in the first place.
    """
    assert bar.headline_metrics == (
        "rmse", "nmad", "bias_abs", "slope_rmse", "aspect_error", "seam_gap",
    )
    assert bar.appendix_metrics == ("mae", "p95", "profile_curvature", "slope_rmse_eroded")
    assert len(bar.registered_metrics) == 10


def test_r5s_printed_grid_is_r1s_roster_times_the_cells(bar: prereg.Bar) -> None:
    # 66 (row, band) cells x 6 columns = 396 printed numbers. §6 states the axes and not the
    # arithmetic, so this is the one tie the phrase check cannot make — and it is #146's defect
    # exactly: a count in one place disagreeing with a set in another.
    table = bar.value("R5")
    assert table["printed_columns"] == len(bar.headline_metrics)
    assert table["printed_numbers"] == table["printed_cells"] * table["printed_columns"] == 396


def test_a_roster_that_no_longer_matches_its_registered_count_refuses(bar_path: Path) -> None:
    doc = json.loads(bar_path.read_text())
    for entry in doc["entries"]:
        if entry["rule"] == "R1":
            entry["value"]["appendix"].pop()
    bar_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="R1: the registered value renders 'three appendix'"):
        prereg.read_bar(bar_path, plan=PLAN)


def test_the_four_marked_pairs_are_the_ones_the_claims_name(bar: prereg.Bar) -> None:
    """R6, named in advance of the 55 computed — which is the pre-registration doing its job."""
    assert bar.bootstrap_pairs == (
        ("A4-R5", "A5"), ("A4-R5", "A6"), ("A3", "A1"), ("A4-R5", "A7"),
    )
    assert bar.value("R6")["computed"] == 55


# --- The committed copy -------------------------------------------------------


def test_the_committed_bar_is_what_freeze_writes_and_still_reads_in_the_plan(tmp_path) -> None:
    """The coupling ADR-0013 states out loud, as the thing that fails when §6 moves.

    §6 was already "goes in the writeup verbatim"; it is now also the anchor a committed file
    validates against, so editing a registered phrase breaks this test. That is the intended
    coupling rather than a hazard — and it fires here, in the suite, rather than on a rented box
    halfway through an arm.

    Byte equality is also what makes `freeze` idempotent for this artifact: the bar is a pure
    function of :data:`benchmark.prereg.REGISTER`, so re-running it can never re-draw an input
    a half-finished run is already scoring against.
    """
    fresh = prereg.write_bar(tmp_path / prereg.BAR_NAME, plan=PLAN)
    assert COMMITTED.read_text() == fresh.read_text(), (
        "the committed bar and the register have drifted — re-run "
        "`python -m benchmark.launch freeze` and commit the result"
    )
    assert prereg.read_bar(COMMITTED, plan=PLAN).rules == 12
