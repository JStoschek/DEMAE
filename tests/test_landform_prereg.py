"""Task 2's registered bar — ADR-0015's numeric subset, anchored in plan §7 (#199).

The same mechanism as `tests/test_benchmark_prereg.py`, pointed at the other task: one module
loads either task's bar, and the phrase refusal is the one Task 1's §6 already has. What these
tests pin is that Task 2 gets that refusal *through the shared path* — a §7 phrase edited away,
a value moved in the file, or a rule id dropped from §7's index all refuse the load, with the
same wording Task 1's refusals use — and that the two registers agree where they overlap
(L6 and R4, the TerraMind DEM tokenizer's 20 epochs).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmark import launch, prereg
from benchmark.landform_config import RUNGS, LandformConfig

REPO = Path(__file__).resolve().parents[1]
TASK2 = prereg.TASK2
PLAN = REPO / TASK2.plan_path
COMMITTED = REPO / "artifacts" / "task2" / TASK2.bar_name

#: ADR-0015's "In bar file" column, ticked: nine rules.
BAR_RULES = ("L1", "L2", "L3", "L6", "L12", "L15", "L16", "L17", "L18")


@pytest.fixture
def bar_path(tmp_path: Path) -> Path:
    return TASK2.write_bar(tmp_path / TASK2.bar_name, plan=PLAN)


@pytest.fixture
def bar(bar_path: Path) -> prereg.LandformBar:
    return TASK2.read_bar(bar_path, plan=PLAN)


def _plan_with(tmp_path: Path, old: str, new: str) -> Path:
    """The plan with one §7 phrase replaced, and everything else intact."""
    text = PLAN.read_text()
    head, section = text[: text.index(TASK2.section)], text[text.index(TASK2.section):]
    edited = section.replace(old, new, 1)
    assert edited != section, f"the fixture did not find {old!r} in §7"
    path = tmp_path / "edited-plan.md"
    path.write_text(head + edited)
    return path


# --- freeze --task landform writes it -----------------------------------------


def test_freeze_writes_the_task_two_bar_with_the_nine_rules_and_their_section_seven_phrases(
    tmp_path, capsys, monkeypatch
):
    # #200's ladders walk Task 1's head-train list, which this empty box does not hold; they are
    # tested with their own fixture. What is under test here is the bar stage alone.
    from benchmark import landform

    stages = tuple(stage for stage in landform.FREEZE_STAGES if stage.label == "prereg bar")
    monkeypatch.setattr(landform, "FREEZE_STAGES", stages)
    task2 = tmp_path / "task2"
    flags = [
        "--task", "landform",
        "--set", f'paths.artifacts_dir="{task2}"',
        "--set", f'paths.task1_artifacts_dir="{tmp_path / "task1"}"',
        "--set", f'paths.results_path="{tmp_path / "results.parquet"}"',
    ]
    launch.main(["freeze", *flags])
    doc = json.loads((task2 / "task2-prereg-bar.json").read_text())
    mechanised = [e for e in doc["entries"] if e.get("registered_in")]
    assert tuple(e["rule"] for e in mechanised) == BAR_RULES
    assert doc["rules"] == 20 and doc["mechanised"] == 9
    section = prereg.TASK2.section_text(PLAN)
    for entry in mechanised:
        assert entry["registered_in"] in section, entry["rule"]
    assert "L15 a margin of 1.0 mIoU point" in {e["registered_in"] for e in mechanised}
    out = capsys.readouterr().out
    assert "task2-prereg-bar.json" in out and "prereg bar — built (#199)" in out


# --- The refusal, by Task 1's path --------------------------------------------


def test_a_section_seven_that_lost_a_registered_phrase_refuses_the_load(bar_path, tmp_path):
    """δ edited in the plan: the bar and the copy that ships cannot drift quietly."""
    plan = _plan_with(tmp_path, "L15 a margin of 1.0 mIoU", "L15 a margin of 2.0 mIoU")
    with pytest.raises(ValueError, match="L15 lost its anchor: §7 no longer states") as exc:
        TASK2.read_bar(bar_path, plan=plan)
    assert TASK2.adr in str(exc.value)


def test_a_section_seven_that_stops_naming_a_rule_refuses_the_load(bar_path, tmp_path):
    """§7 item 10 is the index; a discursive rule leaving it refuses as a numeric one would."""
    plan = _plan_with(tmp_path, "L20 reruns", "reruns")
    with pytest.raises(
        ValueError, match="§7 no longer names L20 — the register's index in §7 item 10"
    ):
        TASK2.read_bar(bar_path, plan=plan)


def test_a_threshold_moved_in_the_file_refuses_too(bar_path):
    doc = json.loads(bar_path.read_text())
    for entry in doc["entries"]:
        if entry["rule"] == "L2":
            entry["value"]["resamples"] = 1_000
    bar_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="L2: the registered value renders '1,000 resamples'"):
        TASK2.read_bar(bar_path, plan=PLAN)


def test_freeze_refuses_to_write_a_bar_that_would_not_load(tmp_path):
    plan = _plan_with(tmp_path, "L3 3 seeds", "L3 2 seeds")
    path = tmp_path / TASK2.bar_name
    with pytest.raises(ValueError, match="L3 lost its anchor"):
        TASK2.write_bar(path, plan=plan)
    assert not path.exists()


def test_a_bar_missing_an_entry_or_carrying_task_ones_refuses(bar_path, tmp_path):
    doc = json.loads(bar_path.read_text())
    doc["entries"] = [e for e in doc["entries"] if e["rule"] != "L15"]
    bar_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=r"is not the register: \['L15'\] missing.*ADR-0015"):
        TASK2.read_bar(bar_path, plan=PLAN)
    task1_bar = prereg.write_bar(tmp_path / prereg.BAR_NAME, plan=REPO / prereg.PLAN_PATH)
    with pytest.raises(ValueError, match="is not the register"):
        TASK2.read_bar(task1_bar, plan=PLAN)


# --- What is in it ------------------------------------------------------------


def test_the_bar_reads_twenty_rules_nine_with_a_number(bar):
    """ADR-0015's table ticks nine for the bar file; its prose says ten. The file follows the
    table, and the eleven it does not tick are recorded as discursive with a reason each."""
    assert len(bar.entries) == bar.rules == 20
    assert tuple(e.rule for e in bar.mechanised) == BAR_RULES
    assert tuple(e.rule for e in bar.unmechanised) == (
        "L4", "L5", "L7", "L8", "L9", "L10", "L11", "L13", "L14", "L19", "L20",
    )
    for entry in bar.unmechanised:
        assert entry.unmechanised_because and entry.value is None
    with pytest.raises(ValueError, match="L11 carries no registered value"):
        bar.value("L11")


def test_the_readers_return_section_seven_and_adr_0015s_numbers(bar):
    # Typed from §7 item 10 and ADR-0015's table, not from the register in code.
    assert bar.bootstrap_resamples == 10_000
    assert bar.test_geocells == 13
    assert bar.seeds == 3
    assert bar.margin == 1.0
    assert bar.c1_rung == "100%"
    assert (bar.ladders_required, bar.ladders) == (2, 3)
    assert (bar.iwahashi_rule, bar.iwahashi_groups) == (0.60, 15)
    assert bar.claim_metrics == {"geomorphon": "miou", "iwahashi": "macro_f1"}


def test_the_eight_marked_pairs_are_the_ones_the_claims_name(bar):
    pairs = [(tuple(p["pair"]), p["claim"], tuple(p["rungs"])) for p in bar.marked_pairs]
    t = prereg.STRONGER_TERRAMIND
    assert pairs == [
        (("B1", "B6"), "C1", ("100%", "100%")),
        (("B1", t), "C1", ("100%", "100%")),
        (("B2", "B6"), "C1", ("100%", "100%")),
        (("B2", t), "C1", ("100%", "100%")),
        (("B2", "B7"), "C2", ("1%", "10%")),
        (("demae-r5", "imagenet-mae"), "C3a", ("100%", "100%")),
        (("demae-r5", t), "C3a", ("100%", "100%")),
        (("B3", "B7"), "C3b", ("100%", "100%")),
    ]
    for p in bar.marked_pairs:
        assert set(p["rungs"]) <= set(RUNGS), "a marked pair sits at a rung §3.3 does not draw"


def test_l6_and_task_ones_r4_agree():
    """One tokenizer, one per-modality citation, in both registers."""
    assert prereg.rule("R4").value == TASK2.rule("L6").value == 20


def test_a_task_two_bar_whose_l6_disagrees_with_r4_refuses(bar_path, tmp_path):
    # Moving L6 together with its §7 phrase passes all three links; the cross-register tie is
    # what still refuses, because Task 1's arms cite R4's figure for the same tokenizer.
    doc = json.loads(bar_path.read_text())
    for entry in doc["entries"]:
        if entry["rule"] == "L6":
            entry["value"], entry["registered_in"] = 100, "L6 100 tokenizer epochs"
    bar_path.write_text(json.dumps(doc))
    plan = _plan_with(tmp_path, "L6 20 tokenizer", "L6 100 tokenizer")
    with pytest.raises(ValueError, match="L6 registers 100 .* Task 1's R4 registers 20"):
        TASK2.read_bar(bar_path, plan=plan)


# --- The values §7 does not state ---------------------------------------------


def test_the_values_section_seven_does_not_state_are_declared_unanchored(bar):
    """Recorded in the file, not passed off as anchored (see `prereg.TASK2`'s section note)."""
    unanchored = {e.rule: e.unanchored for e in bar.mechanised if e.unanchored}
    assert unanchored == {
        "L1": ["claim_metrics"],
        "L16": ["pair"],
        "L17": ["iwahashi_rule", "iwahashi_groups"],
    }
    section = TASK2.section_text(PLAN)
    for absent in ("macro-F1", "60%", "15 groups", "1%", "10%"):
        assert absent not in section, f"§7 now states {absent!r}: anchor it and drop `unanchored`"


def test_an_unanchored_field_is_still_tied_to_the_others(bar_path):
    """L1's metric names are in no §7 phrase, so the marked pairs' metrics are what hold them."""
    doc = json.loads(bar_path.read_text())
    for entry in doc["entries"]:
        if entry["rule"] == "L1":
            entry["value"]["claim_metrics"]["geomorphon"] = "overall_accuracy"
    bar_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="L1 reads geomorphon claims on 'overall_accuracy'"):
        TASK2.read_bar(bar_path, plan=PLAN)


def test_l18_must_be_exactly_the_pairs_the_claims_name(bar_path):
    doc = json.loads(bar_path.read_text())
    for entry in doc["entries"]:
        if entry["rule"] == "L18":
            entry["value"]["marked_pairs"][4]["rungs"] = ["1%", "1%"]
    bar_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="L18's marked pairs are not L12's, L16's and L17's"):
        TASK2.read_bar(bar_path, plan=PLAN)


def test_l16s_ladders_are_l3s_seeds_even_when_both_links_pass(bar_path, tmp_path):
    # Seed i trains on ladder i, so a fourth ladder moved with its §7 phrase still refuses.
    doc = json.loads(bar_path.read_text())
    for entry in doc["entries"]:
        if entry["rule"] == "L16":
            entry["value"]["ladders"] = 4
            entry["registered_in"] = "L16 C2's reading, 2 of 4 ladders"
    bar_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="L16 lost its anchor"):
        TASK2.read_bar(bar_path, plan=PLAN)
    plan = _plan_with(tmp_path, "2 of 3 ladders", "2 of 4 ladders")
    with pytest.raises(ValueError, match="L16 counts 4 ladders and L3 registers 3 seeds"):
        TASK2.read_bar(bar_path, plan=plan)


def test_a_stale_unanchored_declaration_refuses(bar_path):
    doc = json.loads(bar_path.read_text())
    for entry in doc["entries"]:
        if entry["rule"] == "L17":
            entry["unanchored"].append("iwahashi_threshold")
    bar_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="L17 declares 'iwahashi_threshold' unanchored"):
        TASK2.read_bar(bar_path, plan=PLAN)


# --- Where Task 2 reads them --------------------------------------------------


def test_the_iwahashi_rule_the_labels_are_built_at_is_l17s(bar):
    """The 60% rule and the 15 groups are L17's; the label config takes them from the register
    rather than typing a second copy. (The 50% sensitivity row is not a registered rule.)"""
    cfg = LandformConfig().iwahashi
    assert (cfg.threshold, cfg.groups) == (bar.iwahashi_rule, bar.iwahashi_groups)


def test_a_dry_run_resolves_the_bar_by_path_and_md5_once_frozen(tmp_path, capsys):
    task2 = tmp_path / "task2"
    flags = [
        "--task", "landform",
        "--set", f'paths.artifacts_dir="{task2}"',
        "--set", f'paths.task1_artifacts_dir="{tmp_path / "task1"}"',
        "--set", f'paths.results_path="{tmp_path / "results.parquet"}"',
    ]

    def bar_line() -> str:
        launch.main(["run", "--dry-run", *flags])
        out = capsys.readouterr().out.splitlines()
        return next(line for line in out if line.startswith("prereg bar"))

    assert "not frozen yet" in bar_line()
    written = TASK2.write_bar(task2 / TASK2.bar_name, plan=PLAN)
    md5 = hashlib.md5(written.read_bytes()).hexdigest()
    line = bar_line()
    assert str(written) in line and md5 in line


# --- The committed copy -------------------------------------------------------


def test_the_committed_bar_is_what_freeze_writes_and_still_reads_in_the_plan(tmp_path):
    fresh = TASK2.write_bar(tmp_path / TASK2.bar_name, plan=PLAN)
    assert COMMITTED.read_text() == fresh.read_text(), (
        "the committed Task 2 bar and the register have drifted — re-run "
        "`python -m benchmark.launch freeze --task landform` and commit the result"
    )
    assert TASK2.read_bar(COMMITTED, plan=PLAN).rules == 20
