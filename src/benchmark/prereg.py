"""The registered bar — ADR-0013's numeric subset, anchored in plan §6 (#148).

[ADR-0013](../../docs/adr/0013-task1-pre-registration-register.md) registers twelve rules
governing how this benchmark's evidence may be used. **Seven of them carry a number** — R1's
metric roster, R4's `20`, R5's axes, R6's four pairs, R7's `0.15`, R8's `100` and R11's `0.99` —
and those seven are committed by `freeze` as `task1-prereg-bar.json` and re-checked at load.
Past this module **no threshold in the benchmark is typed as a literal**: `benchmark.metrics`
reads R7's seam bar and R1's roster from here, `benchmark.report` reads R5, R6 and R8, and the
one place any of them is written down is :data:`REGISTER` below.

## The three links, and why there are three

`dataset.validate` ties a **code constant** to the **committed bar's prose** and refuses a bar
whose prose no longer states the number (its :func:`~dataset.validate.read_bar`). That is Gate
E's mechanism and it is the mechanism here — but it needs one more link, because this file is
*written by* `freeze` rather than committed by hand, so its value and its phrase come out of
the same source and agreeing with each other proves nothing. The chain is:

1. **value → anchor.** Every registered value renders into a phrase fragment (:data:`_ANCHORS`)
   — `0.15` renders `"above +0.15°"`, a five-name roster renders `"five headline columns"` —
   and the fragment must read in the entry's own registered phrase. Move a value in the file
   and its anchor stops matching its phrase.
2. **anchor → §6.** The registered phrase must appear in plan §6. Edit the phrase to follow the
   moved value and it stops matching the plan.
3. **§6 → the register.** §6 must still name every rule id, so a rule cannot leave the index
   quietly. That is the *countability* ADR-0013 Decision 4 asks of §6, and it is all this link
   claims: it does not check what the rule says.

It binds to **§6 rather than to the ADR** because §6 is the copy that leaves the building — it
goes in the writeup verbatim, so its drift is the externally visible kind. §6 is a marked
quotation of the register, so pinning the shipped copy pins all three transitively.

A fourth check is not a link but a floor: the file's entries must be **the whole register**.
Checking only the entries a file happens to carry is the failure this module exists against — a
bar with R7 deleted would pass every anchor it still had, and the first caller to ask for the
seam threshold would be the one to find out.

Amending a registered rule therefore takes a plan edit, an ADR amendment and a change here, in
that order, and no two of them alone will load. That is the intended coupling (ADR-0013's
Consequences say so out loud) rather than a hazard discovered later.

**"Bar" here is not Gate E's bar**, and `CONTEXT.md` keeps the two terms apart deliberately:
Gate E's is *pass/fail* criteria with Defect-versus-Finding classes and a stop condition, and
this register has no pass/fail at all. The name is the plan's own — §6 and ADR-0013 D8 both call
this artifact the registered bar — and the package prefix is what keeps `prereg.Bar` and
`validate.Bar` from reading as one object.

## What is deliberately not mechanised

**R2, R3, R9, R10 and R12 are discursive, and R9b is a refusal.** They are entries here — with
their statements and the reason each has no constant — and they are *not* checked. "The caveat
is symmetric" has no machine-checkable form; symmetry is a property of how a sentence is used
in an argument, and a tick-box asserting it would look like enforcement while checking nothing.
That is the error [#139](https://github.com/JStoschek/DEMAE/issues/139) refused when it declined
a threshold on a scale-free statistic, and ADR-0013's Alternatives section refuses it again.
Recording them as *present and unmechanised* rather than omitting them is what keeps the
register countable: twelve rules, seven with a number, and the file says which is which.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final

#: What `freeze` writes into ``paths.artifacts_dir``. Named here rather than at the call site
#: so the artifact and the code that refuses it cannot be renamed apart.
BAR_NAME: Final = "task1-prereg-bar.json"

#: The plan, relative to the repo root — `dataset.validate.BAR_PATH`'s shape. Every entry point
#: takes an explicit path (``paths.plan_path``); this is the default a bare call resolves.
PLAN_PATH: Final = Path("docs/research/task1-void-fill-plan.md")

#: §6's heading, verbatim. The anchored text is this heading through to the next ``## ``.
SECTION: Final = "## 6. Fairness checklist"

#: Where the rules are asserted, for a reader who arrives at the artifact rather than at the ADR.
ADR: Final = "docs/adr/0013-task1-pre-registration-register.md"

#: Counts as §6 spells them. Only what the register renders is here; a count word §6 does not
#: use has no business in an anchor.
_COUNT_WORDS: Final = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _word(n: int) -> str:
    """A count as §6 writes it — spelled below seven, digits above, which is §6's own habit."""
    return _COUNT_WORDS.get(n, str(n))


# --- The register -------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """One register row: what it says, where it lives, and — for seven of them — its value."""

    #: ``R1`` … ``R12``, and ``R9b`` for the refusal.
    rule: str
    #: ADR-0013's one-line statement of the rule, so the artifact reads without the ADR open.
    statement: str
    #: The plan sections that carry the rule's own reasoning. ``R11``'s home is ADR-0011.
    home: str
    #: The source ticket, which is where the rule was actually decided.
    source: str
    #: One of the twelve? ``R9b`` is a refusal *within* R9's topic and is not counted again.
    counted: bool = True
    #: The registered value, or ``None`` for the six that carry no number.
    value: Any = None
    #: The exact §6 phrase this value was registered in — whitespace-collapsed and stripped of
    #: Markdown emphasis, which is the form :func:`section_text` produces.
    registered_in: str | None = None
    #: What the value is in, for a reader of the file. Not checked; nothing states it in §6.
    units: str | None = None
    #: Why this rule has no constant. Set for exactly the six that are not mechanised.
    unmechanised_because: str | None = None

    @property
    def mechanised(self) -> bool:
        return self.registered_in is not None


#: The twelve rules and the refusal, in register order. **This is the single home for every
#: number in the benchmark**, and each row carries the ticket that decided it — the shape
#: `dataset.validate` and `dataset.gates` use for their locked values, so a reader sees a number
#: beside its provenance rather than buried in a function body.
REGISTER: Final = (
    Rule(
        rule="R1",
        statement=(
            "Metric set: ten candidates graded, of which five are headline columns; the derived "
            "seam gap is the sixth printed column and was never one of the ten; four appendix, "
            "nothing else."
        ),
        home="§5",
        source="#123, #146",
        # The roster is R1's *single* home. A second list of metric names is what produced #146
        # — a count in one place and a set in another — so `benchmark.metrics` gates on this and
        # the seam gap is not carved out by name anywhere.
        value={
            "graded": ["rmse", "nmad", "bias_abs", "slope_rmse", "aspect_error"],
            "derived": ["seam_gap"],
            "appendix": ["mae", "p95", "profile_curvature", "slope_rmse_eroded"],
        },
        registered_in=(
            "the metric set is five headline columns plus the derived seam gap — six printed — "
            "and four appendix ones, nothing else"
        ),
        units="metric keys, as the results parquet and the printed table name them",
    ),
    Rule(
        rule="R2",
        statement=(
            "The pretraining-corpus caveat is symmetric — unavailable to explain a DEMAE loss "
            "unless also priced as a discount on a DEMAE win."
        ),
        home="§2.3, §6.4",
        source="#122",
        unmechanised_because=(
            "symmetry is a property of how a sentence is used in an argument, not a quantity; a "
            "tick-box asserting it would look like enforcement while checking nothing"
        ),
    ),
    Rule(
        rule="R3",
        statement=(
            "R5 is the headline unconditionally; an R4 win is reported prominently but never "
            "becomes C1's number."
        ),
        home="§0, §2.3",
        source="#122",
        unmechanised_because=(
            "the rule governs which number a sentence in the memo is about; both rows are "
            "computed either way, so there is no constant a run could read"
        ),
    ),
    Rule(
        rule="R4",
        statement=(
            "TerraMind's 20 DEM-tokenizer epochs carry their per-modality citation wherever "
            "stated — not the paper's 100, which is its S-2/S-1 figure."
        ),
        home="§2.3, §6.4, §10",
        source="#138",
        value=20,
        registered_in=(
            "TerraMind's 20 DEM-tokenizer epochs carry their per-modality citation wherever "
            "stated, not the paper's 100"
        ),
        units="epochs over the DEM modality",
    ),
    Rule(
        rule="R5",
        statement=(
            "The printed table's shape: the size buckets pool, the blob family is headline, C4 "
            "prints as a rank block, and 66 (row, band) cells × 6 columns = 396 printed numbers."
        ),
        home="§5, §9",
        source="#136, #146",
        value={
            "pooled_axis": "size buckets",
            "headline_family": "blob",
            "c4_form": "rank block",
            # §6 states the axes and not the arithmetic, so these three are anchored against R1
            # rather than against a phrase — see :func:`_consistency`.
            "printed_cells": 66,
            "printed_columns": 6,
            "printed_numbers": 396,
        },
        registered_in=(
            "the size buckets pool, the blob family is headline, and C4 prints as a rank block"
        ),
        units="axes, and the printed grid they leave",
    ),
    Rule(
        rule="R6",
        statement=(
            "Exactly four arm pairs carry a bootstrap CI mark, named in advance, of the 55 "
            "computed."
        ),
        home="§5, §9",
        source="#136",
        value={
            "marked_pairs": [
                {"pair": ["A4-R5", "A5"], "claim": "C1", "size": None},
                {"pair": ["A4-R5", "A6"], "claim": "C1", "size": None},
                {"pair": ["A3", "A1"], "claim": "C2", "size": None},
                {"pair": ["A4-R5", "A7"], "claim": "C3", "size": "20k"},
            ],
            "computed": 55,
        },
        registered_in=(
            "exactly four arm pairs carry a bootstrap CI mark, named in advance, of the 55 "
            "computed"
        ),
        units="arm pairs marked in print, of C(11,2) computed into the parquet",
    ),
    Rule(
        rule="R7",
        statement="A seam gap above +0.15° is a seam break; |gap| ≤ 0.15° is not.",
        home="§5",
        source="#136",
        value=0.15,
        registered_in="a seam gap above +0.15° counts as a seam break",
        units="degrees of full-minus-eroded slope RMSE",
    ),
    Rule(
        rule="R8",
        statement=(
            "Real-medium+ is unstratified unless the realised histogram clears 100 per band; a "
            "(row, band) cell below it prints a count and no metric."
        ),
        home="§3.3, §5",
        source="#121",
        value=100,
        registered_in=(
            "real-medium+ is unstratified unless the realised histogram clears 100 per band"
        ),
        units="instances per (row, band) cell",
    ),
    Rule(
        rule="R9",
        statement=(
            "The GSD caveat is symmetric, and is stated as a bandwidth shift rather than a "
            "resolution handicap."
        ),
        home="§2.2, §6.3",
        source="#139",
        unmechanised_because=(
            "same shape as R2 — the rule constrains how a caveat may be deployed in argument, "
            "and the shift it is about is already measured on the tokenizer's own FSQ quantizer"
        ),
    ),
    Rule(
        rule="R9b",
        statement=(
            "A refusal: the FSQ diagnostic carries no threshold, because the statistic has no "
            "natural scale and one would imply it settles what the shift costs."
        ),
        home="§2.2",
        source="#139, #140",
        counted=False,  # a refusal within R9's topic; the register still reads twelve rules
        unmechanised_because=(
            "irreducibly — the entry *is* the absence of a threshold, and #140 already committed "
            "the enforcement it can have: `threshold`, `verdict` and `ablation_row` are "
            "forbidden as record keys and the limit rides as a record field"
        ),
    ),
    Rule(
        rule="R10",
        statement=(
            "C3's low-label sweep is the pre-registered fallback claim; nulls are reported with "
            "the prominence of wins."
        ),
        home="§6.7",
        source="plan §6",
        unmechanised_because=(
            "it fixes which claim the memo falls back to, not a number the sweep must clear"
        ),
    ),
    Rule(
        rule="R11",
        statement=(
            "The bicubic-vs-re-derived ablation row, A5 at 20k, is dropped iff the materiality "
            "check returns ≥ 0.99. It returned 0.9708, so the row stays."
        ),
        home="ADR-0011 D4-5",
        source="#134",
        value=0.99,
        registered_in="dropped iff the materiality check returns ≥ 0.99",
        units="cosine similarity between the bicubic and re-derived pos_emb",
    ),
    Rule(
        rule="R12",
        statement=(
            "A2 carries no C-claim and is insulated from C1 in both directions; the A2→A3 "
            "difference is a joint quantity; A2 is a negative result if it does not beat A1 in "
            "a majority of the four bands."
        ),
        home="§4.3, §6.8",
        source="#142",
        unmechanised_because=(
            "its trigger is an *ordering* against A1 rather than a level, so there is no "
            "constant to hold — the same reason its template R10 is discursive"
        ),
    ),
)

#: value → the §6 fragment it renders. **The first of the module docstring's three links**, and
#: the one that makes the other two bite: an anchor is computed from whatever value the file
#: carries, so a threshold edited in `task1-prereg-bar.json` renders a fragment its own
#: registered phrase no longer contains. Keyed by rule so the check survives the round trip
#: through JSON, exactly as `dataset.validate` keeps its anchors in code and its data in the file.
_ANCHORS: Final[dict[str, Callable[[Any], tuple[str, ...]]]] = {
    "R1": lambda v: (
        f"{_word(len(v['graded']))} headline columns",
        f"{_word(len(v['graded']) + len(v['derived']))} printed",
        f"{_word(len(v['appendix']))} appendix",
    ),
    "R4": lambda v: (f"{v} DEM-tokenizer epochs",),
    "R5": lambda v: (
        f"{v['pooled_axis']} pool",
        f"{v['headline_family']} family is headline",
        f"prints as a {v['c4_form']}",
    ),
    "R6": lambda v: (
        f"{_word(len(v['marked_pairs']))} arm pairs",
        f"of the {v['computed']} computed",
    ),
    "R7": lambda v: (f"above +{v}°",),
    "R8": lambda v: (f"clears {v} per band",),
    "R11": lambda v: (f"returns ≥ {v}",),
}


# --- §6, as the anchor sees it ------------------------------------------------


def section_text(plan: Path | str = PLAN_PATH) -> str:
    """§6 of the plan, from its heading to the next one, in the form an anchor matches.

    Normalization is two crudenesses, both deliberate and both bounded: Markdown emphasis is
    stripped, so a rule that gains or loses a bold marker does not read as a moved threshold;
    and whitespace is collapsed, so §6's hard wrapping is not part of what a phrase must match.
    Neither can turn one number into another, which is the only failure the anchor is for.
    """
    text = Path(plan).read_text()
    try:
        start = text.index(SECTION)
    except ValueError:  # pragma: no cover - a plan without §6 is not a plan
        raise ValueError(
            f"{plan}: no {SECTION!r} — the registered bar is anchored in §6, which is the copy "
            f"that goes in the writeup verbatim (see {ADR})."
        ) from None
    end = text.find("\n## ", start + 1)
    section = text[start:] if end < 0 else text[start:end]
    return re.sub(r"\s+", " ", section.replace("*", ""))


def _reads(haystack: str, phrase: str) -> bool:
    """Is ``phrase`` in ``haystack`` on word boundaries?

    Word-bounded rather than a bare substring, for `dataset.validate`'s reason: ``"100"`` must
    not be satisfied by a plan edited to say ``1000``, which is precisely the edit this catches.
    """
    return re.search(rf"(?<![\w.]){re.escape(phrase)}(?![\w.])", haystack) is not None


# --- The refusals -------------------------------------------------------------


def _consistency(values: dict[str, Any]) -> None:
    """The ties between entries that no §6 phrase states — #146's defect, as a check.

    #146 was a *count* in one place disagreeing with a *set* in another, caught only because
    someone read both. These three are that class of drift, and they are checked here because
    §6 states the axes and R1's counts but never the arithmetic they imply.
    """
    r1, r5, r6 = values["R1"], values["R5"], values["R6"]
    printed = len(r1["graded"]) + len(r1["derived"])
    if printed != r5["printed_columns"]:
        raise ValueError(
            f"R1's roster prints {printed} columns and R5 registers {r5['printed_columns']} — "
            "the register cannot gate a build on a count that contradicts itself (#146)."
        )
    expected = r5["printed_cells"] * r5["printed_columns"]
    if expected != r5["printed_numbers"]:
        raise ValueError(
            f"R5: {r5['printed_cells']} cells × {r5['printed_columns']} columns is {expected} "
            f"printed numbers, not {r5['printed_numbers']}."
        )
    if len(r6["marked_pairs"]) != len(set(tuple(m["pair"]) for m in r6["marked_pairs"])):
        raise ValueError("R6: the marked pairs are not distinct, so fewer than four are marked.")


def _check_against_section(doc: dict, section: str) -> None:
    """Every refusal, in one place, so `freeze` and `read_bar` refuse identically.

    Called by :func:`write_bar` before a byte is written and by :func:`read_bar` before a value
    is handed out — which is what makes the refusal fire **at load rather than at first use**.
    A threshold that is only checked when something reads it is a threshold that passes on every
    run that happens not to reach the line.
    """
    entries = doc["entries"]
    seen = {entry["rule"]: entry for entry in entries}
    if len(seen) != len(entries):
        raise ValueError(f"{doc['artifact']}: a rule id appears twice in the register.")

    # The register is *whole* or it is nothing. Checking only the entries the file happens to
    # carry is the failure mode this whole module is against: a bar with R7 deleted would load
    # clean, every anchor it still had would pass, and the first caller to ask for the seam bar
    # would be the one to discover it — which is precisely "at first use".
    registered = {rule.rule for rule in REGISTER}
    if set(seen) != registered:
        missing = sorted(registered - set(seen))
        extra = sorted(set(seen) - registered)
        raise ValueError(
            f"{doc['artifact']} is not the register: "
            + (f"{missing} missing. " if missing else "")
            + (f"{extra} unregistered. " if extra else "")
            + f"ADR-0013 registers {len(registered)} entries and a bar carrying any other set "
            f"is not the one this benchmark is answerable to. Re-freeze, and if a rule really "
            f"has entered or left, amend {ADR} and plan §6 first."
        )

    for entry in entries:
        rule = entry["rule"]
        # Link 3 — §6 is the index, and every rule is named on it. This says nothing about what
        # the rule *states*; it is the countability ADR-0013 Decision 4 asks of §6, and it is the
        # only tie the six unmechanised entries have, on purpose (see the module docstring).
        if not _reads(section, rule):
            raise ValueError(
                f"§6 no longer names {rule} — the register's index in §6 item 8 and "
                f"{doc['artifact']} have drifted. A rule leaving the index is a rule leaving "
                f"the pre-registration; amend {ADR} in the same change."
            )
        phrase = entry.get("registered_in")
        if phrase is None:
            if not entry.get("unmechanised_because"):
                raise ValueError(
                    f"{rule} carries neither a registered phrase nor a reason it has none. "
                    "An unmechanised rule is recorded as unmechanised, never omitted."
                )
            continue
        # Link 1 — the value renders an anchor, and the anchor must read in the phrase the value
        # was registered in. This is what a threshold moved in the file fails.
        if rule not in _ANCHORS:
            raise ValueError(
                f"{rule} carries a registered value and no way to render it into a §6 fragment. "
                f"A mechanised entry without an anchor is a number nothing checks — give it one "
                f"in `_ANCHORS` or record it as unmechanised and say why."
            )
        for anchor in _ANCHORS[rule](entry["value"]):
            if not _reads(phrase, anchor):
                raise ValueError(
                    f"{rule}: the registered value renders {anchor!r}, which no longer reads in "
                    f"the phrase it was registered in ({phrase!r}). A value and its phrase "
                    f"cannot be moved apart — amend §6 and {ADR}, then re-freeze."
                )
        # Link 2 — and that phrase must still be in the copy that ships.
        if not _reads(section, phrase):
            raise ValueError(
                f"{rule} lost its anchor: §6 no longer states {phrase!r}. The committed bar and "
                f"the plan have drifted, and §6 is the copy that leaves the building. Reconcile "
                f"them — and {ADR} — in one dated commit before running an arm."
            )

    _consistency({rule: entry["value"] for rule, entry in seen.items()})


# --- The artifact -------------------------------------------------------------


@dataclass(frozen=True)
class Bar:
    """The loaded bar: the register as the file carries it, checked against §6.

    Every accessor reads the *file*, never :data:`REGISTER`. That is the whole arrangement — the
    committed artifact is what a run is answerable to, and the module constant is only what
    `freeze` wrote it from.
    """

    doc: dict[str, Any]
    path: Path

    @property
    def entries(self) -> tuple[Rule, ...]:
        """The register as the file carries it — converted once, so nothing below reads a dict."""
        return tuple(Rule(**entry) for entry in self.doc["entries"])

    @property
    def mechanised(self) -> tuple[Rule, ...]:
        """The seven that carry a number, in register order."""
        return tuple(entry for entry in self.entries if entry.mechanised)

    @property
    def unmechanised(self) -> tuple[Rule, ...]:
        """The five discursive rules and the refusal — present, and recorded as unenforced."""
        return tuple(entry for entry in self.entries if not entry.mechanised)

    @property
    def rules(self) -> int:
        """How many rules the register reads. Twelve; R9b is a refusal within R9's topic."""
        return sum(1 for entry in self.entries if entry.counted)

    def rule(self, rule: str) -> Rule:
        for entry in self.entries:
            if entry.rule == rule:
                return entry
        raise KeyError(
            f"{rule} is not in the register — {self.path} carries "
            f"{[entry.rule for entry in self.entries]}."
        )

    def value(self, rule: str) -> Any:
        """The registered value, refusing the six that have none rather than returning ``None``.

        An unmechanised rule asked for its number is a caller that has misread the register, and
        ``None`` would let it measure against nothing. The refusal says why there is no number.
        """
        entry = self.rule(rule)
        if not entry.mechanised:
            raise ValueError(
                f"{rule} carries no registered value: {entry.unmechanised_because}. It is in the "
                f"register so it can be counted and attributed, not so it can be enforced."
            )
        return entry.value

    # --- the four the build spec names as read-from-the-bar

    @property
    def headline_metrics(self) -> tuple[str, ...]:
        """R1's six printed columns: the five graded plus the derived seam gap."""
        r1 = self.value("R1")
        return tuple(r1["graded"]) + tuple(r1["derived"])

    @property
    def appendix_metrics(self) -> tuple[str, ...]:
        """R1's four appendix columns — computed into the parquet, printed nowhere."""
        return tuple(self.value("R1")["appendix"])

    @property
    def registered_metrics(self) -> tuple[str, ...]:
        """The whole registered set. A metric outside it is not one `score` may compute."""
        return self.headline_metrics + self.appendix_metrics

    @property
    def seam_break_deg(self) -> float:
        """R7's +0.15°, read rather than typed."""
        return self.value("R7")

    @property
    def cell_floor(self) -> int:
        """R8's 100 instances: below it a (row, band) cell prints a count and no metric."""
        return self.value("R8")

    @property
    def bootstrap_pairs(self) -> tuple[tuple[str, str], ...]:
        """R6's four marked pairs, in the order they were named."""
        return tuple(tuple(m["pair"]) for m in self.value("R6")["marked_pairs"])

    @property
    def bootstrap_claims(self) -> tuple[str, ...]:
        """Which C-claim each marked pair is for, in :attr:`bootstrap_pairs`' order."""
        return tuple(m["claim"] for m in self.value("R6")["marked_pairs"])

    @property
    def computed_pairs(self) -> int:
        """R6's 55: C(11, 2), computed into the parquet and unmarked in print."""
        return self.value("R6")["computed"]

    # --- R5's printed shape, which `benchmark.report` checks its own render against

    @property
    def headline_family(self) -> str:
        """R5's headline family. The other synthetic one is scored and never printed."""
        return self.value("R5")["headline_family"]

    @property
    def printed_shape(self) -> tuple[int, int, int]:
        """R5's ``(cells, columns, numbers)`` — 66, 6 and 396.

        Returned together because they are one registered fact and their product is the check:
        a render matching two of the three has not matched R5. Reading them one at a time is how
        [#146](https://github.com/JStoschek/DEMAE/issues/146) happened — a count in one place and
        a set in another.
        """
        r5 = self.value("R5")
        return (r5["printed_cells"], r5["printed_columns"], r5["printed_numbers"])


def bar_document() -> dict[str, Any]:
    """:data:`REGISTER` as the committed artifact — plain JSON, and self-describing.

    The prose fields are here for the reader who arrives at the file rather than at the ADR: an
    artifact that says only ``{"R7": 0.15}`` is one whose reader has to be told what it is.

    A pure function of :data:`REGISTER`, and deliberately not of where the plan sits on this
    box: ``anchored_in`` names §6's canonical home rather than the path this freeze happened to
    read, on `benchmark.config`'s rule that a rented box mounting a file somewhere else must not
    change what the artifact says. It is also what makes `freeze` byte-idempotent here.
    """
    return {
        "artifact": BAR_NAME,
        "asserted_in": ADR,
        "anchored_in": f"{PLAN_PATH.as_posix()} {SECTION}",
        "purpose": (
            "The numeric subset of ADR-0013's register of twelve rules, committed by "
            "`benchmark.launch freeze` before any arm runs. Every entry carries its value and "
            "the exact §6 phrase it was registered in; `benchmark.prereg.read_bar` refuses a "
            "bar whose phrase no longer appears in §6, and names the entry that lost its "
            "anchor. Five discursive rules and one refusal are recorded here without values, "
            "because mechanising them would look like enforcement while checking nothing."
        ),
        "rules": sum(1 for rule in REGISTER if rule.counted),
        "mechanised": sum(1 for rule in REGISTER if rule.mechanised),
        "entries": [
            {k: v for k, v in vars(rule).items() if v is not None or k == "value"}
            for rule in REGISTER
        ],
    }


def write_bar(path: Path | str, *, plan: Path | str = PLAN_PATH) -> Path:
    """Write the bar, having refused to write one that would not load.

    Checked *before* the write rather than after it: a `freeze` that leaves a bar on disk and
    then raises has committed the drift it was refusing.

    This artifact carries **no §3.4 hash guard**, unlike the four that will join it here, and the
    difference is not an omission. §3.4's refusal protects a *draw* — a re-run that silently
    re-rolled the masks a half-finished run was already scoring against would be unrecoverable.
    The bar is a pure function of :data:`REGISTER`, so it has no draw to protect: re-freezing
    either rewrites the same bytes or reflects an amendment that has already passed all three
    links, and refusing the second case would block the one path an amendment is supposed to take.
    """
    path = Path(path)
    doc = bar_document()
    _check_against_section(doc, section_text(plan))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return path


def read_bar(path: Path | str, *, plan: Path | str = PLAN_PATH) -> Bar:
    """Load the committed bar, refusing one whose §6 phrases have moved.

    Gate E's arrangement (:func:`dataset.validate.read_bar`), one link longer — see the module
    docstring. The refusal is here, at load, so a run that reaches a threshold and a run that
    does not fail the same way.
    """
    path = Path(path)
    doc = json.loads(path.read_text())
    _check_against_section(doc, section_text(plan))
    return Bar(doc=doc, path=path)
