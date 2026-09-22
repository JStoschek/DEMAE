"""The results memo — §9 item 3's deliverable, as checks a reviewer can run (#170).

The memo is prose, and most of what makes it honest is not mechanisable. Four things are, and
they are exactly the four a later edit could quietly undo:

1. **§6 ships verbatim.** :func:`benchmark.prereg.section_text` already normalizes and anchors
   §6 for the committed bar; pointing it at the memo instead is the same check against the same
   text, so the copy that ships and the copy the bar is anchored in cannot drift apart. §6 is
   the one section the plan itself marks *goes in the writeup verbatim*.
2. **Every registered rule's disposition is stated.** ADR-0013's argument for the register is
   that a reviewer can audit whether the pre-registration did work; a memo that states eight of
   twelve dispositions is one where the four it skipped are the four to ask about. The register
   is iterated rather than listed here, so a thirteenth rule fails this test rather than
   silently going unreported.
3. **ADR-0011's divergence sentence is present, with its arithmetic.** That ADR's Consequences
   name the debt in the writeup's own words — *one sentence, with 1.1429-vs-1.0 attached*.
4. **R12's struck clause stays struck.** #167 asserted the five wordings over A2's run record;
   the memo is the other surface where the clause could reappear, and it is the one a reader
   actually reads.

What is deliberately *not* checked: whether the readings are right. A test cannot tell whether
C2 partially failed, and one that pattern-matched for the word "fails" would be satisfied by a
memo that used it about the wrong claim.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Final

import pytest

from benchmark import arms, prereg

#: The six printed (section, band) cells, as :func:`benchmark.report.printed_cells` orders them.
_PRINTED_A4_CELLS: Final = (
    ("blob", "R1"),
    ("blob", "R2"),
    ("blob", "R3"),
    ("blob", "R4"),
    ("real-small", "R4"),
    ("real-medium+", "R4"),
)


def _seed_rows(frame, arm: str, section: str, band: str, seed: int):
    """One seed's rows for a printed cell — `report.cell_rows` does not split by seed."""
    family = {"family": "blob"} if section == "blob" else {"family": "real", "bucket": section}
    return frame.select(arm=arm, size="20k", band=band, seed=seed, **family)

#: Anchored at the repo root rather than at the working directory, on
#: `test_benchmark_prereg`'s rule: a committed file is found from the test's own location, so a
#: PYTHONPATH-isolated worktree resolves it the same way a working checkout does.
REPO: Final = Path(__file__).resolve().parents[1]

#: The deliverable, where §9 item 3 puts it — ``docs/research/``.
MEMO: Final = REPO / "docs" / "research" / "task1-results-memo.md"

#: The plan §6 is compared against. `prereg.PLAN_PATH` is a repo-relative default, so it is
#: anchored here too — `test_benchmark_prereg`'s `PLAN`. Every `prereg` entry point below takes
#: this explicitly rather than letting the default resolve against the working directory.
PLAN: Final = REPO / prereg.PLAN_PATH

#: The committed bar, composed from `prereg.BAR_NAME` rather than retyped — the artifact and the
#: code that refuses it cannot be renamed apart, and that property should reach the tests too.
BAR: Final = REPO / "artifacts" / "task1" / prereg.BAR_NAME

#: The canonical results file. Gitignored (``data/*``), so the tests that read it skip in a
#: throwaway worktree rather than failing there — `test_benchmark_a9`'s rule for the release
#: weights, applied to the rows.
RESULTS: Final = REPO / "data" / "task1" / "results.parquet"

#: R12's struck clause, in #167's five wordings. Carried here rather than imported from
#: `tests.test_benchmark_a2` because the two surfaces are independent: a test module is not an
#: API, and the clause has to be struck in the memo whether or not that file still exists.
STRUCK: Final = (
    "quantifies precisely what the task adaptation bought",
    "what task adaptation bought",
    "attributable to task adaptation",
    "due to task adaptation",
    "what the task adaptation bought",
)


@pytest.fixture(scope="module")
def memo() -> str:
    """The memo's raw text. Read once — every test below is a read."""
    assert MEMO.exists(), (
        f"{MEMO} does not exist. §9 item 3 names a short results memo in `docs/research/` "
        f"answering C1, C2 and C3 in numbers and C4 in ranks (#170)."
    )
    return MEMO.read_text()


def _outside_section_6(text: str) -> str:
    """The memo with §6 cut out, flattened — the part that speaks in the memo's own voice.

    §6 is quoted, not written: it *states* the twelve rules, so its wording is evidence about
    the pre-registration rather than about this run. A check on how the memo reports a result
    has to exclude it, or the rule's own statement reads as the memo asserting the outcome.
    """
    start = text.index(prereg.SECTION)
    end = text.find("\n## ", start + 1)
    without = text[:start] + (text[end:] if end >= 0 else "")
    return re.sub(r"\s+", " ", without.replace("*", ""))


@pytest.fixture(scope="module")
def flat(memo: str) -> str:
    """The memo with emphasis stripped and whitespace collapsed — `prereg.section_text`'s form.

    Phrase assertions run against this so that hard wrapping and a bold marker are not part of
    what a sentence must match, for the reason `section_text` gives: neither can turn one number
    into another, which is the only failure these anchors are for.
    """
    return re.sub(r"\s+", " ", memo.replace("*", ""))


# --- §0's paragraph, in the opening position ----------------------------------


def test_it_opens_on_the_incomparability_paragraph(memo: str, flat: str):
    """§9 item 3's amendment: the memo opens on §0's paragraph, not on a comparison.

    Checked as a *position* and not only as a presence, because the amendment is about where the
    paragraph sits. A reader who meets the table first has already read the missing competitor
    column as an omission by the time the explanation arrives, which is the failure #129
    described.
    """
    assert "internally controlled comparison" in flat
    assert "nothing external to stand it beside" in flat

    body = memo[memo.index("\n") :]
    incomparability = body.index("internally controlled comparison")
    for later in ("## C1", "## C2", "## C3", "## C4"):
        assert body.index(later) > incomparability, (
            f"{later} is answered before the incomparability paragraph. §9 item 3 puts that "
            f"paragraph in the opening position (#129)."
        )


def test_the_reason_is_the_leak_and_not_access(flat: str):
    """Both candidates were opened and read, and the memo says which and why.

    #170's own framing: *the reason is not access*. A memo that said only "no comparable public
    benchmark exists" would be the weak version of the claim that §0 explicitly rejects in
    favour of *incomparable, and we looked*.
    """
    assert "Diff-DEM" in flat and "GeomorPM" in flat
    assert "void interior included" in flat
    assert "mean rank" in flat.lower() and "Reuter" in flat


def test_the_struck_selection_premise_is_struck_in_the_memo_that_carries_it(flat: str):
    """§9 item 3: void filling was selected partly on the premise §0 later destroyed.

    *"That reason did not survive #124 and #128, and is struck in the memo that carries it."*
    The memo is the only place that strike can happen, and the task has to be left standing on
    the reasons that did survive — otherwise the strike reads as a retraction of the task.
    """
    assert "downstream-task-selection" in flat
    assert "inpainter" in flat.lower()


# --- §6, verbatim -------------------------------------------------------------


def test_section_6_appears_verbatim(memo: str):
    """The plan's §6 and the memo's §6 are the same text, through the bar's own normalizer.

    Reusing :func:`benchmark.prereg.section_text` rather than comparing raw bytes is the whole
    point: that function is what the committed bar is anchored through, so a phrase that would
    still satisfy the bar cannot fail here and vice versa. One normalizer, two documents.
    """
    assert prereg.SECTION in memo, (
        f"the memo carries no {prereg.SECTION!r} heading. §6 is the section the plan marks "
        f"'goes in the writeup verbatim', and the bar is anchored in it ({prereg.ADR})."
    )
    assert prereg.section_text(MEMO) == prereg.section_text(PLAN)


def test_the_committed_bar_still_loads_against_the_memos_copy(tmp_path: Path):
    """The stronger form: every registered phrase reads in the memo's §6 too.

    `read_bar` refuses a bar whose phrase has moved out of §6 and names the entry that lost its
    anchor. Pointing it at the memo asserts that the shipped copy would satisfy the same
    refusal — so the seven mechanised values are anchored in the document a reader holds, not
    only in the plan they may never open.
    """
    written = prereg.write_bar(tmp_path / prereg.BAR_NAME, plan=PLAN)
    bar = prereg.read_bar(written, plan=MEMO)

    assert bar.rules == sum(1 for r in prereg.REGISTER if r.counted)


# --- Every rule's disposition -------------------------------------------------


def _disposition_rows(memo: str) -> dict[str, str]:
    """The dispositions table, as ``{rule: the rest of its row}``.

    A table rather than prose, and parsed rather than searched for, because "R8 appears
    somewhere in the memo" is satisfied by a sentence that mentions R8 while disposing of
    nothing. A row with cells is a claim that something was decided.
    """
    rows: dict[str, str] = {}
    for line in memo.splitlines():
        match = re.match(r"^\|\s*\*\*(R\d+b?)\*\*\s*\|(.+)$", line.strip())
        if match:
            rows[match.group(1)] = match.group(2)
    return rows


def test_every_registered_rule_has_a_disposition(memo: str):
    """#170's third criterion, iterated over the register rather than over a typed list.

    ADR-0013's *"a twelfth site is now a defect, not an accident"* has a counterpart here: a
    thirteenth rule added to the register must fail this test, because a memo that reports the
    twelve it knew about is exactly the unauditable pre-registration the register exists to
    prevent.
    """
    rows = _disposition_rows(memo)
    expected = {r.rule for r in prereg.REGISTER}

    assert set(rows) == expected, (
        f"dispositions missing for {sorted(expected - set(rows))}, unexpected rows for "
        f"{sorted(set(rows) - expected)}. All twelve rules and R9b's refusal need one (#170)."
    )
    for name, row in rows.items():
        cells = [c.strip() for c in row.split("|") if c.strip()]
        assert len(cells) >= 2, f"{name}'s row disposes of nothing: {row!r}"


def test_the_two_rules_that_fired_say_what_they_returned(memo: str):
    """R8 and R11, named in #170 because they are the audit's whole evidence.

    A rule that held against a bar nobody could move is the only kind worth having registered,
    and the way a reader checks that is the measured value beside the bar. Both are asserted
    with their numbers rather than with the word "held".
    """
    rows = _disposition_rows(memo)

    assert "438" in rows["R8"] and "100" in rows["R8"]
    assert "0.9708" in rows["R11"] and "0.99" in rows["R11"]


def test_r9b_is_disposed_of_as_a_refusal_and_not_given_a_threshold(memo: str):
    """R9b is irreducibly unmechanised, and a memo that reported a number for it would be a bug.

    #139 refused a threshold on the FSQ diagnostic because the statistic has no natural scale.
    The disposition therefore has to be *that there is no threshold*, which is a stateable
    outcome rather than a blank.
    """
    row = _disposition_rows(memo)["R9b"]

    assert "no threshold" in row.lower()


# --- R3, and what C1's number is ----------------------------------------------


def test_r5_is_the_headline_and_the_r4_win_is_not_c1s_number(flat: str):
    """#170's fifth criterion. Both halves, because either alone is satisfiable dishonestly.

    A memo that suppressed the R4 result would satisfy "R5 is the headline" and violate R3's
    other half — *an R4 win is reported prominently*. One that led with R4 would do the
    reverse. The arm names are the check: C1's number is A4-R5's, and R4's row is present.
    """
    assert "A4-R5" in flat and "A4-R4" in flat
    assert "R5 is the headline" in flat


def test_the_c1_answer_names_the_r5_arm_before_the_r4_one(memo: str):
    """R3 as an ordering in the document, not only as a sentence about one.

    C1's section has to reach its number through A4-R5. Finding A4-R4 first inside C1 would be
    the failure R3 describes even if the sentence "R5 is the headline" appeared elsewhere.
    """
    c1 = memo[memo.index("## C1") : memo.index("## C2")]

    assert c1.index("A4-R5") < c1.index("A4-R4")


# --- ADR-0011's sentence ------------------------------------------------------


def test_the_terratorch_divergence_sentence_is_present_with_its_arithmetic(flat: str):
    """#170's sixth criterion, and ADR-0011's Consequences in their own words.

    *"We now diverge from the library's default inference path, and the writeup must say so
    plainly — one sentence, with 1.1429-vs-1.0 attached."* Both spacings are asserted, because
    the sentence without them is the claim with its evidence removed.
    """
    assert "diverge" in flat and "terratorch" in flat
    assert str(arms.BICUBIC_TOKEN_SPACING) in flat
    assert str(arms.CROP_TOKEN_SPACING) in flat


# --- R12's dispositions -------------------------------------------------------


def test_no_claim_that_the_a2_to_a3_difference_measures_task_adaptation(flat: str):
    """#170's seventh criterion, over the memo as #167 asserted it over the record.

    The clause is struck in five wordings rather than one, because the thing R12 forbids is a
    reading and not a phrase — and a memo is where that reading is most naturally reached for.
    """
    for struck in STRUCK:
        assert struck not in flat, f"R12's struck clause reappears: {struck!r}"


@pytest.mark.skipif(not RESULTS.exists(), reason="results.parquet is gitignored (data/*)")
def test_r12s_verdict_in_the_memo_is_the_one_band_comparison_computes(memo: str, flat: str):
    """R12's disposition, recomputed from the parquet rather than read off the prose.

    #173 registered §4.3's per-patch scale estimator and A2 was scored through it, so the
    contingency is computable: an ordering against A1 over §5's four ruggedness bands on RMSE,
    ties losing, over the instances both arms scored. The memo's R12 row has to state *that*
    verdict — the number of bands A2 won and the register's own word for the outcome — and the
    memo must say in its own voice that the estimator was registered before the row was read,
    because "chosen after seeing the ordering" is the tunability R12 exists to prevent.
    """
    from benchmark import results, zeroshot

    a1 = results.rows_for_cell(RESULTS, arm="A1", size="20k", seed=0, columns=zeroshot.COMPARISON_COLUMNS)
    verdicts = {}
    for arm in ("A2-R5", "A2-R4"):
        a2 = results.rows_for_cell(RESULTS, arm=arm, size="20k", seed=0, columns=zeroshot.COMPARISON_COLUMNS)
        assert a2, f"{arm} has no rows in the parquet; #173 scored it"
        verdicts[arm] = zeroshot.band_comparison(a1_rows=a1, a2_rows=a2)

    row = _disposition_rows(memo)["R12"]
    headline = verdicts["A2-R5"]
    assert headline["bands_compared"] == headline["bands_registered"] == 4
    assert f"{headline['bands_won']} of" in row, (
        f"R12's row does not state the computed bands won ({headline['bands_won']} of 4): {row!r}"
    )
    if headline["negative_result"]:
        assert "negative result" in row.lower()
    else:
        assert "ordering against A1" in row
        assert "negative result" not in row.lower().replace("not a negative result", "")
    # R4's secondary reading is stated too, at R3's prominence and never as the headline.
    assert f"{verdicts['A2-R4']['bands_won']} of" in row or "A2-R4" in row

    #: And every band mean the memo prints for the comparison is the computed one.
    for arm, verdict in verdicts.items():
        for band, entry in verdict["per_band"].items():
            for value in (entry["a1"], entry["a2"]):
                assert f"{value:.3f}" in flat or f"{value:.2f}" in flat, (
                    f"{arm} {band}: the memo does not print {value:.3f}"
                )

    assert "#173" in flat and "norm_pix" in flat
    assert "before" in _outside_section_6(memo).lower()  # registered before the row was read


# --- The narrative numbers, which are the ones a hand edit reaches -------------
#
# The table and the rank block are recomputed wholesale below. These two are the *prose* figures
# that no such sweep covers, and both are places this memo was already wrong once: the "what ran"
# counts, and the seed-spread split R3's reading rests on. A claim a reader would have to open the
# parquet to falsify is exactly the claim worth a test.


@pytest.mark.skipif(not RESULTS.exists(), reason="results.parquet is gitignored (data/*)")
def test_the_what_ran_table_counts_the_rows_the_parquet_holds(memo: str):
    """Every row count in the opening table is the arm's real row count.

    A2's zero is the one that carries an argument — it is why R12 is pending — so it is checked
    like any other number rather than read as an absence.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(RESULTS, columns=["arm"])
    actual = Counter(table.column("arm").to_pylist())

    rows = re.findall(r"^\|\s*(A[0-9][\w-]*)\s*\|[^|]*\|\s*\*{0,2}([\d,]+|0)\*{0,2}\s*\|", memo, re.M)
    assert rows, "no arm row counts found in the 'what ran' table"

    for arm, printed in rows:
        assert int(printed.replace(",", "")) == actual.get(arm, 0), (
            f"{arm}: the memo says {printed} rows, the parquet holds {actual.get(arm, 0):,}"
        )


@pytest.mark.skipif(not RESULTS.exists(), reason="results.parquet is gitignored (data/*)")
def test_the_r4_secondarys_seed_spread_split_is_the_measured_one(flat: str):
    """R3's reading: which A4-R5-vs-A4-R4 cells separate beyond seed noise, and which way.

    The memo claims all four cells R4 wins sit inside a seed spread and both cells R5 wins sit
    outside one. That is the whole support for not treating the R4 win as meaningful, and it is
    a claim about six subtractions and twelve spreads that nobody checks by eye — this memo
    asserted the false universal ("all six are inside") on the first pass.
    """
    from benchmark import report

    frame = report.read(RESULTS)
    inside_r4_wins, outside_r5_wins = 0, 0
    for section, band in _PRINTED_A4_CELLS:
        means, spreads = {}, {}
        for arm in ("A4-R5", "A4-R4"):
            rows = report.cell_rows(frame, arm, section, band, size=report.HEADLINE_SIZE)
            means[arm] = report.finite_mean(frame.columns["rmse"][rows])[0]
            per_seed = [
                report.finite_mean(frame.columns["rmse"][sel])[0]
                for sel in (_seed_rows(frame, arm, section, band, seed) for seed in (0, 1, 2))
            ]
            spreads[arm] = max(per_seed) - min(per_seed)

        gap = abs(means["A4-R4"] - means["A4-R5"])
        if means["A4-R4"] < means["A4-R5"]:
            assert gap < max(spreads.values()), f"{section}/{band}: an R4 win outside seed spread"
            inside_r4_wins += 1
        else:
            assert gap > max(spreads.values()), f"{section}/{band}: an R5 win inside seed spread"
            outside_r5_wins += 1

    assert (inside_r4_wins, outside_r5_wins) == (4, 2), (
        f"the split is {inside_r4_wins} R4 wins inside and {outside_r5_wins} R5 wins outside; "
        f"the memo says four and two"
    )
    assert "four cells where R4 edges R5 are inside a seed spread" in flat
    assert "both cells where R5 is ahead are outside one" in flat


# --- The numbers are the report's numbers -------------------------------------


#: §5's six printed columns, in the order the memo's table prints them.
COLUMNS: Final = ("rmse", "nmad", "bias_abs", "slope_rmse", "aspect_error", "seam_gap")


@pytest.mark.skipif(not RESULTS.exists(), reason="results.parquet is gitignored (data/*)")
def test_every_printed_cell_in_the_memo_is_the_cell_the_report_computes(flat: str):
    """The whole printed table, cell by cell, against :mod:`benchmark.report`'s own machinery.

    Not a spot check on the headline column: a memo is a document someone edits by hand under
    time pressure, and a transposed digit in `bias_abs` is exactly the error no reader would
    catch and no other test would either. Recomputed rather than re-derived, so a mean taken
    over a different row set fails here even when it looks plausible.

    A cell with no rows is skipped rather than special-cased — `cell_rows` returns nothing for
    it, which is the same fact the table would print as *not scored*. A2's twelve were that until
    #173 scored them; now they are checked like every other cell.
    """
    from benchmark import report

    frame = report.read(RESULTS)
    missing = []
    for arm, section, band in report.printed_cells():
        rows = report.cell_rows(frame, arm, section, band, size=report.HEADLINE_SIZE)
        if not rows.any():
            continue
        for column in COLUMNS:
            mean, _ = report.finite_mean(frame.columns[column][rows])
            if f"{mean:.3f}" not in flat:
                missing.append(f"{arm} {section}/{band} {column} = {mean:.3f}")

    assert not missing, (
        "these computed cells do not appear in the memo's table:\n  "
        + "\n  ".join(missing)
        + "\nThe memo's numbers are the printed table's numbers (#169)."
    )


@pytest.fixture(scope="module")
def ranks() -> dict[tuple[str, str, str], int]:
    """C4's rank block as ``{(set, metric, arm): rank}``, computed once for both readers.

    A fixture rather than a helper called twice, on `test_benchmark_prereg`'s `bar_path`/`bar`
    pattern: the block costs a full read of the parquet, and the two tests below want the same
    one. Module-scoped for the same reason :func:`memo` is.
    """
    from benchmark import report
    from benchmark.config import BenchmarkConfig

    if not RESULTS.exists():  # pragma: no cover - the skipif on each reader covers this
        pytest.skip("results.parquet is gitignored (data/*)")

    frame = report.read(RESULTS)
    bar = prereg.read_bar(BAR, plan=PLAN)
    return {
        (cell.set_name, cell.metric, cell.arm): cell.rank
        for cell in report.rank_block(frame, bar, BenchmarkConfig(arm="A1")).cells
    }


@pytest.mark.skipif(not RESULTS.exists(), reason="results.parquet is gitignored (data/*)")
def test_c4_is_answered_in_ranks_and_every_rank_is_the_computed_one(memo: str, ranks):
    """C4 in ranks, never in metres — §0 makes it an ordering claim for a stated reason.

    The real family is drawn at its natural size distribution and the synthetic families are
    bucket-balanced, so a difference in metres between the two rows would confound void *shape*
    with void *size*. So the check is two things: that the answer is given in ranks at all, and
    that all 66 of them are :func:`benchmark.report.rank_block`'s own. A rank table is where a
    hand edit does the most damage for the least visibility — one transposed pair invents a
    swap, or hides one, and the swaps are the entire finding.
    """
    computed = ranks

    c4 = memo[memo.index("## C4") :]
    assert re.search(r"#\d", c4), "C4 is answered without a rank token"
    assert "real-medium+" in c4 and "real-small" in c4

    order = list(arms.PRINTED_ROWS)
    pattern = re.compile(
        r"^\|\s*(blob/small|real-small|real-medium\+)\s*/\s*(rmse|slope_rmse)\s*\|(.+?)\|\s*$"
    )
    checked = 0
    for line in c4.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        printed = [c.strip() for c in match.group(3).split("|")]
        assert len(printed) == len(order), (
            f"{match.group(1)}/{match.group(2)} has {len(printed)} columns for "
            f"{len(order)} printed rows"
        )
        for arm, token in zip(order, printed):
            want = computed.get((match.group(1), match.group(2), arm))
            got = None if token == "—" else int(token.lstrip("#"))
            assert want == got, (
                f"{match.group(1)}/{match.group(2)} {arm}: memo says {token!r}, "
                f"`report.rank_block` computes {want!r}"
            )
            checked += 1

    assert checked == len(computed) == 66, (
        f"checked {checked} of {len(computed)} rank cells; R5 registers a 66-cell block"
    )


@pytest.mark.skipif(not RESULTS.exists(), reason="results.parquet is gitignored (data/*)")
def test_c4s_swapping_pairs_are_the_computed_ones_and_none_is_a_c1_pair(
    memo: str, flat: str, ranks
):
    """C4's whole finding, recomputed: which pairs swap, how many, and whether C1 is among them.

    This is the assertion the prose is most likely to get wrong, because the swaps are counted by
    eye off a rank table and a miscount reads as authoritative either way. §0's condition is
    specific — a pair swapping between the *synthetic* row and a *real-void* row within the same
    band — so a pair that only reorders between the two real sets does not qualify, and one that
    does qualify must be named.

    The second half is the one that matters for the conclusion: C1's comparison is A4-R5 against
    A6 and A5, so C4 failing is only *not* evidence against C1 if that triple keeps its order
    everywhere. Asserted rather than asserted-about, because "the swaps do not touch C1" is the
    convenient half of the sentence.
    """
    from itertools import combinations

    scored = [a for a in arms.PRINTED_ROWS if ranks.get(("blob/small", "rmse", a)) is not None]
    real_sets = ("real-small", "real-medium+")

    swaps = set()
    for metric in ("rmse", "slope_rmse"):
        for left, right in combinations(scored, 2):
            synthetic = ranks[("blob/small", metric, left)] < ranks[("blob/small", metric, right)]
            for real in real_sets:
                if (ranks[(real, metric, left)] < ranks[(real, metric, right)]) != synthetic:
                    swaps.add(frozenset((left, right)))

    for pair in swaps:
        left, right = sorted(pair)
        c4 = memo[memo.index("## C4") :]
        assert left in c4 and right in c4, f"{left}/{right} swaps rank and C4 does not name it"

    count = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}[len(swaps)]
    assert f"{count} pairs swap" in flat or f"{count} pairs\ndo" in memo, (
        f"{len(swaps)} pairs swap rank and the memo does not say so in words. A miscounted "
        f"heading reads as authoritative either way."
    )

    c1_triple = ("A4-R5", "A6", "A5")
    assert not any(
        frozenset(p) in swaps for p in combinations(c1_triple, 2)
    ), "a C1 pair swaps rank, so C4's failure does reach C1 and the memo must say that instead"
    for metric in ("rmse", "slope_rmse"):
        for group in ("blob/small", *real_sets):
            ordered = [ranks[(group, metric, arm)] for arm in c1_triple]
            assert ordered == sorted(ordered), (
                f"C1's ordering A4-R5 < A6 < A5 breaks at {group}/{metric}: {ordered}"
            )
