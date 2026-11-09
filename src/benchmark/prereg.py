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

## Two tasks, one mechanism (#199)

Task 2 has its own register — [ADR-0015](../../docs/adr/0015-task2-pre-registration-register.md)'s
twenty rules, L1-L20, nine of them in the bar file — anchored in the Task 2 plan's §7 the way
Task 1's is anchored in its §6. ADR-0015's Consequences ask that the two bars "load through one
module and can't drift apart in mechanism", so everything above is a :class:`Register`: one
object per task holding its artifact name, plan, section, ADR, rule tuple, anchors and
cross-entry checks, and one :meth:`Register.check` that both bars pass through. :data:`TASK1`
is the register this module always had, and its module-level names (:data:`BAR_NAME`,
:data:`PLAN_PATH`, :data:`SECTION`, :data:`ADR`, :data:`REGISTER`, :func:`rule`,
:func:`write_bar`, :func:`read_bar`) are kept as Task 1's, so no Task 1 caller and no Task 1
byte moved. :data:`TASK2` is the new one, and its section further down says where §7 does not
state a value the ADR put in the bar.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Mapping

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
    #: Keys of a mechanised value that **no anchored-section phrase states** — recorded in the
    #: file rather than passed off as anchored. Task 1 has none. Task 2 has a few, because
    #: ADR-0015 put values in the bar that plan §7's index does not spell out (see :data:`TASK2`).
    #: Such a field is tied to the other entries by the register's cross-checks only.
    unanchored: list[str] | None = None

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


# --- The anchored section, as the anchor sees it -------------------------------


def _reads(haystack: str, phrase: str) -> bool:
    """Is ``phrase`` in ``haystack`` on word boundaries?

    Word-bounded rather than a bare substring, for `dataset.validate`'s reason: ``"100"`` must
    not be satisfied by a plan edited to say ``1000``, which is precisely the edit this catches.
    """
    return re.search(rf"(?<![\w.]){re.escape(phrase)}(?![\w.])", haystack) is not None


# --- Task 1's cross-entry ties ------------------------------------------------


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


# --- The loaded artifact ------------------------------------------------------


@dataclass(frozen=True)
class RegisteredBar:
    """A loaded bar, either task's: the register as the file carries it, checked at load.

    Every accessor reads the *file*, never the register in code. That is the whole arrangement —
    the committed artifact is what a run is answerable to, and the module constant is only what
    `freeze` wrote it from. :class:`Bar` and :class:`LandformBar` add each task's named readers.
    """

    doc: dict[str, Any]
    path: Path

    @property
    def entries(self) -> tuple[Rule, ...]:
        """The register as the file carries it — converted once, so nothing below reads a dict."""
        return tuple(Rule(**entry) for entry in self.doc["entries"])

    @property
    def mechanised(self) -> tuple[Rule, ...]:
        """The entries that carry a number, in register order."""
        return tuple(entry for entry in self.entries if entry.mechanised)

    @property
    def unmechanised(self) -> tuple[Rule, ...]:
        """The discursive rules (and Task 1's refusal) — present, and recorded as unenforced."""
        return tuple(entry for entry in self.entries if not entry.mechanised)

    @property
    def rules(self) -> int:
        """How many rules the register reads. Task 1's R9b is a refusal within R9's topic."""
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
        """The registered value, refusing a rule that has none rather than returning ``None``.

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


@dataclass(frozen=True)
class Bar(RegisteredBar):
    """Task 1's loaded bar, checked against §6 — :class:`RegisteredBar` with R1-R11's readers."""

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


# --- One task's register, and the refusals every bar passes through -----------


@dataclass(frozen=True)
class Register:
    """One task's pre-registration: what its bar is called, where it is anchored, what it holds.

    Everything task-specific is a field; everything that refuses is a method. So the two tasks
    share the *mechanism* — the three links, the whole-register floor, the at-load timing and the
    wording of every refusal — and differ only in the data the mechanism is pointed at.
    """

    #: What `freeze` writes into the task's ``paths.artifacts_dir``.
    bar_name: str
    #: The plan, relative to the repo root. Every entry point takes an explicit path too.
    plan_path: Path
    #: The anchored section's heading, verbatim. Its text runs to the next ``## ``.
    section: str
    #: How the refusals name the section: ``"§6"``, ``"§7"``.
    mark: str
    #: Where in the section the register's index is, for the refusal that a rule left it.
    index: str
    #: Where the rules are asserted, and the ADR's short name.
    adr: str
    adr_name: str
    #: The ``purpose`` paragraph the artifact carries for a reader who arrives at the file.
    purpose: str
    #: The rules, in register order — the single home for every number the task registers.
    entries: tuple[Rule, ...]
    #: value → the section fragments it renders. The first of the three links.
    anchors: Mapping[str, Callable[[Any], tuple[str, ...]]]
    #: The ties between entries that no section phrase states.
    consistency: Callable[[dict[str, Any]], None]
    #: What :meth:`read_bar` hands back.
    bar_type: type[RegisteredBar] = RegisteredBar

    def rule(self, name: str) -> Rule:
        """One register row, by name, from the register *in code* — see :func:`rule`."""
        for entry in self.entries:
            if entry.rule == name:
                return entry
        raise KeyError(
            f"{name} is not in the register — it holds {[e.rule for e in self.entries]}"
        )

    def section_text(self, plan: Path | str | None = None) -> str:
        """The anchored section, from its heading to the next one, in the form an anchor matches.

        Normalization is two crudenesses, both deliberate and both bounded: Markdown emphasis is
        stripped, so a rule that gains or loses a bold marker does not read as a moved threshold;
        and whitespace is collapsed, so the section's hard wrapping is not part of what a phrase
        must match. Neither can turn one number into another, which is the only failure the
        anchor is for.
        """
        plan = self.plan_path if plan is None else plan
        text = Path(plan).read_text()
        try:
            start = text.index(self.section)
        except ValueError:  # pragma: no cover - a plan without its checklist is not a plan
            raise ValueError(
                f"{plan}: no {self.section!r} — the registered bar is anchored in {self.mark}, "
                f"which is the copy that goes in the writeup verbatim (see {self.adr})."
            ) from None
        end = text.find("\n## ", start + 1)
        section = text[start:] if end < 0 else text[start:end]
        return re.sub(r"\s+", " ", section.replace("*", ""))

    def check(self, doc: dict, section: str) -> None:
        """Every refusal, in one place, so `freeze` and `read_bar` refuse identically.

        Called by :meth:`write_bar` before a byte is written and by :meth:`read_bar` before a
        value is handed out — which is what makes the refusal fire **at load rather than at first
        use**. A threshold that is only checked when something reads it is a threshold that
        passes on every run that happens not to reach the line.
        """
        mark, adr = self.mark, self.adr
        entries = doc["entries"]
        seen = {entry["rule"]: entry for entry in entries}
        if len(seen) != len(entries):
            raise ValueError(f"{doc['artifact']}: a rule id appears twice in the register.")

        # The register is *whole* or it is nothing. Checking only the entries the file happens to
        # carry is the failure mode this whole module is against: a bar with a rule deleted would
        # load clean, every anchor it still had would pass, and the first caller to ask for the
        # missing threshold would be the one to discover it — which is precisely "at first use".
        registered = {rule.rule for rule in self.entries}
        if set(seen) != registered:
            missing = sorted(registered - set(seen))
            extra = sorted(set(seen) - registered)
            raise ValueError(
                f"{doc['artifact']} is not the register: "
                + (f"{missing} missing. " if missing else "")
                + (f"{extra} unregistered. " if extra else "")
                + f"{self.adr_name} registers {len(registered)} entries and a bar carrying any "
                f"other set is not the one this benchmark is answerable to. Re-freeze, and if a "
                f"rule really has entered or left, amend {adr} and plan {mark} first."
            )

        for entry in entries:
            rule = entry["rule"]
            # Link 3 — the section is the index, and every rule is named on it. This says
            # nothing about what the rule *states*; it is the countability the ADR asks of the
            # section, and it is the only tie the unmechanised entries have, on purpose.
            if not _reads(section, rule):
                raise ValueError(
                    f"{mark} no longer names {rule} — the register's index in {self.index} and "
                    f"{doc['artifact']} have drifted. A rule leaving the index is a rule leaving "
                    f"the pre-registration; amend {adr} in the same change."
                )
            phrase = entry.get("registered_in")
            if phrase is None:
                if not entry.get("unmechanised_because"):
                    raise ValueError(
                        f"{rule} carries neither a registered phrase nor a reason it has none. "
                        "An unmechanised rule is recorded as unmechanised, never omitted."
                    )
                continue
            # Link 1 — the value renders an anchor, and the anchor must read in the phrase the
            # value was registered in. This is what a threshold moved in the file fails.
            if rule not in self.anchors:
                raise ValueError(
                    f"{rule} carries a registered value and no way to render it into a {mark} "
                    f"fragment. A mechanised entry without an anchor is a number nothing checks "
                    f"— give it one in the register's anchors or record it as unmechanised and "
                    f"say why."
                )
            for anchor in self.anchors[rule](entry["value"]):
                if not _reads(phrase, anchor):
                    raise ValueError(
                        f"{rule}: the registered value renders {anchor!r}, which no longer reads "
                        f"in the phrase it was registered in ({phrase!r}). A value and its phrase "
                        f"cannot be moved apart — amend {mark} and {adr}, then re-freeze."
                    )
            # Link 2 — and that phrase must still be in the copy that ships.
            if not _reads(section, phrase):
                raise ValueError(
                    f"{rule} lost its anchor: {mark} no longer states {phrase!r}. The committed "
                    f"bar and the plan have drifted, and {mark} is the copy that leaves the "
                    f"building. Reconcile them — and {adr} — in one dated commit before running "
                    f"an arm."
                )
            # A field declared unanchored must exist: a stale declaration would let the next
            # reader believe a value is disclosed as unanchored when the value has moved shape.
            value = entry["value"]
            for key in entry.get("unanchored") or ():
                if not isinstance(value, dict) or key not in value:
                    raise ValueError(
                        f"{rule} declares {key!r} unanchored in {mark}, and its value has no such "
                        f"field. The declaration and the value have drifted; re-freeze."
                    )

        self.consistency({rule: entry.get("value") for rule, entry in seen.items()})

    def document(self) -> dict[str, Any]:
        """The register as the committed artifact — plain JSON, and self-describing.

        The prose fields are here for the reader who arrives at the file rather than at the ADR:
        an artifact that says only ``{"R7": 0.15}`` is one whose reader has to be told what it is.

        A pure function of the register, and deliberately not of where the plan sits on this
        box: ``anchored_in`` names the section's canonical home rather than the path this freeze
        happened to read, on `benchmark.config`'s rule that a rented box mounting a file
        somewhere else must not change what the artifact says. It is also what makes `freeze`
        byte-idempotent here.
        """
        return {
            "artifact": self.bar_name,
            "asserted_in": self.adr,
            "anchored_in": f"{self.plan_path.as_posix()} {self.section}",
            "purpose": self.purpose,
            "rules": sum(1 for rule in self.entries if rule.counted),
            "mechanised": sum(1 for rule in self.entries if rule.mechanised),
            "entries": [
                {k: v for k, v in vars(rule).items() if v is not None or k == "value"}
                for rule in self.entries
            ],
        }

    def write_bar(self, path: Path | str, *, plan: Path | str | None = None) -> Path:
        """Write the bar, having refused to write one that would not load.

        Checked *before* the write rather than after it: a `freeze` that leaves a bar on disk and
        then raises has committed the drift it was refusing.

        **No content-hash guard, for either task**, unlike the drawn artifacts beside it, and the
        difference is not an omission. That refusal protects a *draw* — a re-run that silently
        re-rolled masks or a ladder a half-finished run was already reading would be
        unrecoverable. A bar is a pure function of its register, so it has no draw to protect:
        re-freezing either rewrites the same bytes or reflects an amendment that has already
        passed all three links, and refusing the second case would block the one path an
        amendment is supposed to take.
        """
        path = Path(path)
        doc = self.document()
        self.check(doc, self.section_text(plan))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        return path

    def read_bar(self, path: Path | str, *, plan: Path | str | None = None) -> Any:
        """Load the committed bar, refusing one whose section phrases have moved.

        Gate E's arrangement (:func:`dataset.validate.read_bar`), one link longer — see the
        module docstring. The refusal is here, at load, so a run that reaches a threshold and a
        run that does not fail the same way.
        """
        path = Path(path)
        doc = json.loads(path.read_text())
        self.check(doc, self.section_text(plan))
        return self.bar_type(doc=doc, path=path)


# --- Task 1: ADR-0013's register, anchored in plan §6 --------------------------

#: Task 1's register. The module-level names below are its own, kept so no caller moves.
TASK1: Final = Register(
    bar_name=BAR_NAME,
    plan_path=PLAN_PATH,
    section=SECTION,
    mark="§6",
    index="§6 item 8",
    adr=ADR,
    adr_name="ADR-0013",
    purpose=(
        "The numeric subset of ADR-0013's register of twelve rules, committed by "
        "`benchmark.launch freeze` before any arm runs. Every entry carries its value and "
        "the exact §6 phrase it was registered in; `benchmark.prereg.read_bar` refuses a "
        "bar whose phrase no longer appears in §6, and names the entry that lost its "
        "anchor. Five discursive rules and one refusal are recorded here without values, "
        "because mechanising them would look like enforcement while checking nothing."
    ),
    entries=REGISTER,
    anchors=_ANCHORS,
    consistency=_consistency,
    bar_type=Bar,
)


def rule(name: str) -> Rule:
    """One Task 1 register row, by name — :data:`REGISTER`'s :func:`benchmark.arms.get`.

    Distinct from :meth:`Bar.rule`, which reads the register a *committed bar file* carries. This
    reads the one in code, which is what a caller wants when it needs a rule's statement or its
    home rather than a value a run must clear — a discursive rule has no value either way.
    Task 2's is ``TASK2.rule``.
    """
    return TASK1.rule(name)


def section_text(plan: Path | str = PLAN_PATH) -> str:
    """Task 1's §6, as :meth:`Register.section_text` normalizes it."""
    return TASK1.section_text(plan)


def bar_document() -> dict[str, Any]:
    """Task 1's :data:`REGISTER` as the committed artifact (:meth:`Register.document`)."""
    return TASK1.document()


def write_bar(path: Path | str, *, plan: Path | str = PLAN_PATH) -> Path:
    """Write Task 1's bar, having refused to write one that would not load."""
    return TASK1.write_bar(path, plan=plan)


def read_bar(path: Path | str, *, plan: Path | str = PLAN_PATH) -> Bar:
    """Load Task 1's committed bar, refusing one whose §6 phrases have moved."""
    return TASK1.read_bar(path, plan=plan)


# --- Task 2: ADR-0015's register, anchored in plan §7 --------------------------
#
# ADR-0015's "In bar file" column ticks **nine** rules: L1, L2, L3, L6, L12 and L15-L18. Its own
# prose says "ten numeric rules" (the register's preamble and its Consequences); the table is the
# itemised statement and the one this follows, so the file says ``"mechanised": 9``.
#
# **Where §7 does not state a value the ADR put in the bar.** §7 item 10 is the index, and it
# names L1's metrics only as "claim metrics", L16's rungs not at all, and L17 only as "C3's two
# halves". So L1's metric names (mIoU, macro-F1), L16's pair at 1% and 10%, and L17's 60% rule
# and 15 groups have no §7 phrase to be anchored in. They are in the file, they are listed in
# each entry's ``unanchored``, and they are tied to the other entries by
# :func:`_task2_consistency` — the marked pairs' metrics to L1's, the L12 pairs' rungs to L12's —
# and to nothing in the plan. Closing that takes an amendment to §7 item 10 and ADR-0015 that
# states them, which is ADR-0013's prescribed route; inventing a phrase here would be the
# workaround the spec rules out.

TASK2_BAR_NAME: Final = "task2-prereg-bar.json"
TASK2_PLAN_PATH: Final = Path("docs/research/task2-landform-plan.md")
TASK2_SECTION: Final = "## 7. Fairness checklist (verbatim into the writeup)"
TASK2_ADR: Final = "docs/adr/0015-task2-pre-registration-register.md"

#: L11's opponent, resolved by `report` and never by a run: whichever of the TerraMind tokenizer
#: (B4/B5) and backbone (B9) encoders has the higher seed-mean test score, per head and per label
#: set. A marked pair names it as a token because which encoder it is cannot be known in advance.
STRONGER_TERRAMIND: Final = "stronger-terramind"

#: L1's claim metric for each label set: mIoU on the dense geomorphon labels, macro-F1 over the
#: Iwahashi tile labels. The marked pairs carry theirs, and :func:`_task2_consistency` ties them.
_CLAIM_METRICS: Final = {"geomorphon": "miou", "iwahashi": "macro_f1"}

#: §3.3's full rung, and C2's two (L16).
_FULL: Final = "100%"
_C2_RUNGS: Final = ("1%", "10%")

#: Counts as §7 spells them. §7 spells pairs and halves ("four pairs", "two halves", "eight
#: marked pairs") and writes seeds and ladders as digits ("3 seeds", "2 of 3 ladders"), so the
#: anchors choose per rule; only what an anchor renders spelled is here.
_SPELLED: Final = {2: "two", 4: "four", 8: "eight"}


def _spelled(n: int) -> str:
    return _SPELLED.get(n, str(n))


def _pair(a: str, b: str, *, claim: str, rungs: tuple[str, str], label_set: str) -> dict:
    """One marked pair: first arm minus second, at those rungs, on that label set's claim metric.

    A pair names B-arms, except C3a's, which names encoders: the Iwahashi tile classifier is a
    fitted probe on each encoder's pooled embeddings (§4.5), not a trained B-row.
    """
    return {
        "pair": [a, b],
        "claim": claim,
        "rungs": list(rungs),
        "label_set": label_set,
        "metric": _CLAIM_METRICS[label_set],
    }


#: L12's four: B1 and B2, each vs B6 and vs the stronger TerraMind, at 100%.
_C1_PAIRS: Final = [
    _pair("B1", "B6", claim="C1", rungs=(_FULL, _FULL), label_set="geomorphon"),
    _pair("B1", STRONGER_TERRAMIND, claim="C1", rungs=(_FULL, _FULL), label_set="geomorphon"),
    _pair("B2", "B6", claim="C1", rungs=(_FULL, _FULL), label_set="geomorphon"),
    _pair("B2", STRONGER_TERRAMIND, claim="C1", rungs=(_FULL, _FULL), label_set="geomorphon"),
]
#: L16's one: B2 at 1% vs B7 at 10%.
_C2_PAIR: Final = _pair("B2", "B7", claim="C2", rungs=_C2_RUNGS, label_set="geomorphon")
#: L17's three: C3a's Iwahashi classifier, R5 vs ImageNet-MAE and vs the stronger TerraMind;
#: C3b's fine-tune, B3 vs B7 at 100%.
_C3_PAIRS: Final = [
    _pair("demae-r5", "imagenet-mae", claim="C3a", rungs=(_FULL, _FULL), label_set="iwahashi"),
    _pair("demae-r5", STRONGER_TERRAMIND, claim="C3a", rungs=(_FULL, _FULL),
          label_set="iwahashi"),
    _pair("B3", "B7", claim="C3b", rungs=(_FULL, _FULL), label_set="geomorphon"),
]

TASK2_REGISTER: Final = (
    Rule(
        rule="L1",
        statement=(
            "Claims are read on mIoU (geomorphons) and macro-F1 (Iwahashi tiles), on the pooled "
            "test set only. Dense macro-F1, per-class IoU, overall accuracy, boundary-F1, "
            "per-band metrics and C1's 1%/10% rows are printed and carry no claim."
        ),
        home="§6",
        source="#192",
        value={"claim_metrics": dict(_CLAIM_METRICS), "population": "pooled"},
        registered_in="L1 claim metrics, pooled only",
        units="metric keys per label set, and the test population claims are read on",
        unanchored=["claim_metrics"],
    ),
    Rule(
        rule="L2",
        statement=(
            "Paired bootstrap over test tiles, 10,000 resamples, confusion matrices summed per "
            "resample, seed-paired differences averaged over seeds on one shared resample. "
            "Registered caveat: the test set spans 13 geocells, so the tile CI is narrower than "
            "a spatially honest one."
        ),
        home="§6",
        source="#192",
        value={"unit": "tile", "resamples": 10_000, "test_geocells": 13},
        registered_in="L2 tile bootstrap, 10,000 resamples, 13 test geocells",
        units="resamples of the test tiles; test geocells the caveat counts",
    ),
    Rule(
        rule="L3",
        statement=(
            "3 seeds for every core cell, B3 and B7 included; seed i trains on ladder i. The "
            "seed spread prints as mean ± std, separate from the CI and never folded into it."
        ),
        home="§3.3, §4.3, §6",
        source="#189, #192",
        value={"seeds": 3, "including": ["B3", "B7"]},
        registered_in="L3 3 seeds, B3 and B7 included",
        units="seeds per core cell",
    ),
    Rule(
        rule="L4",
        statement=(
            "The pretraining-corpus caveat is symmetric: it can't explain a DEMAE loss unless "
            "also priced as a discount on a DEMAE win (Task 1 R2)."
        ),
        home="§2, §7.4",
        source="#122",
        unmechanised_because=(
            "symmetry is a property of how a sentence is used in an argument, not a quantity — "
            "Task 1's R2 reason, unchanged"
        ),
    ),
    Rule(
        rule="L5",
        statement=(
            "R5 is C1's DEMAE number unconditionally; an R4 win is reported prominently and "
            "never becomes C1's number (Task 1 R3)."
        ),
        home="§0, §1",
        source="#122, #190",
        unmechanised_because=(
            "it governs which number a sentence is about; both rows are computed either way, so "
            "there is no constant a run could read"
        ),
    ),
    Rule(
        rule="L6",
        statement=(
            "TerraMind's DEM tokenizer had 20 epochs, cited per modality, not the paper's 100 "
            "(Task 1 R4)."
        ),
        home="§2",
        source="#138",
        value=20,
        registered_in="L6 20 tokenizer epochs",
        units="epochs over the DEM modality",
    ),
    Rule(
        rule="L7",
        statement=(
            "The GSD caveat is symmetric, and stated as a bandwidth shift rather than a "
            "resolution handicap (Task 1 R9)."
        ),
        home="§2, §7.3",
        source="#139",
        unmechanised_because=(
            "same shape as L4 — it constrains how a caveat may be deployed in argument"
        ),
    ),
    Rule(
        rule="L8",
        statement="A null is reported with the prominence of a win (Task 1 R10).",
        home="§0",
        source="plan §0",
        unmechanised_because="prominence is a property of the memo's prose, not of a number",
    ),
    Rule(
        rule="L9",
        statement=(
            "Insulated rows (B4-ALT/B5-ALT, B8, the concatenation row, B1-S/B2-S, B1-R4/B2-R4) "
            "carry no claim and never stand in for a core row; an ALT row beating DEMAE at 100% "
            "with lower > 0 is said in C1's paragraph."
        ),
        home="§1, §5.4",
        source="#190, #192",
        unmechanised_because=(
            "it fixes what a row may stand in for in prose; its one trigger reads L2's CI and "
            "adds no constant of its own"
        ),
    ),
    Rule(
        rule="L10",
        statement=(
            "A band's macro metric averages over the classes with ground truth in that band. "
            "Class collapse (a class present in test truth with test IoU = 0) is flagged and "
            "counted per cell, never excluded or rerun."
        ),
        home="§6",
        source="ADR-0014, #192",
        unmechanised_because=(
            "it is a definition — how a macro metric averages and what counts as collapse — and "
            "its one number, IoU = 0, is the definition rather than a threshold anyone could tune"
        ),
    ),
    Rule(
        rule="L11",
        statement=(
            "The stronger TerraMind encoder is whichever of the tokenizer (B4/B5) and the "
            "backbone (B9) has the higher seed-mean test score, per head and per label set."
        ),
        home="§0, §1",
        source="#190, #192",
        unmechanised_because=(
            "its trigger is an ordering between two encoders, not a level, so there is no "
            "constant to hold; the marked pairs name it as a token `report` resolves"
        ),
    ),
    Rule(
        rule="L12",
        statement=(
            "C1 is read at the 100% rung on 4 pairs — B1 and B2, each vs B6 and vs the stronger "
            "TerraMind — and holds if all four have lower > 0; one head's two clearing reports "
            "C1 for that head only. Iwahashi gate: C1 is claimed only if R5's Iwahashi macro-F1 "
            "point estimate exceeds both B6's and the stronger TerraMind's."
        ),
        home="§0",
        source="#190, #192",
        value={"rung": _FULL, "pairs": _C1_PAIRS},
        registered_in="L12 C1's four pairs at 100% and the Iwahashi gate",
        units="the rung C1 is read at, and its marked pairs",
    ),
    Rule(
        rule="L13",
        statement=(
            "Depth-sweep reversal: a competitor's best point beating DEMAE's best with a CI "
            "excluding zero qualifies C1's linear half to \"at the registered layer\" and never "
            "withdraws it."
        ),
        home="§5.3",
        source="#190, #192",
        unmechanised_because=(
            "its trigger is a CI excluding zero, which is not a tunable level, and what it "
            "changes is a sentence rather than a number"
        ),
    ),
    Rule(
        rule="L14",
        statement=(
            "If B1 and B4 swap order under any of ADR-0014's sensitivity label sets, C1's "
            "paragraph says so; C1 is read on the primary labels only."
        ),
        home="§7.7",
        source="ADR-0014, #192",
        unmechanised_because="its trigger is an ordering swap, not a level",
    ),
    Rule(
        rule="L15",
        statement=(
            "δ = 1.0 mIoU point, the non-inferiority margin for C2's \"matches\" and C3b's "
            "\"competitive\"."
        ),
        home="§0",
        source="#192",
        value=1.0,
        registered_in="L15 a margin of 1.0 mIoU point",
        units="mIoU points",
    ),
    Rule(
        rule="L16",
        statement=(
            "C2: B2 at the 1% rung vs B7 at the 10% rung — beats if lower > 0, matches if lower "
            "> −δ. The per-ladder difference must be ≥ −δ in at least 2 of 3 ladders, otherwise "
            "C2 is region-dependent and not claimed."
        ),
        home="§0",
        source="#189, #192",
        value={"pair": _C2_PAIR, "ladders_required": 2, "ladders": 3},
        registered_in="L16 C2's reading, 2 of 3 ladders",
        units="C2's marked pair, and ladders that must clear −δ of those drawn",
        unanchored=["pair"],
    ),
    Rule(
        rule="L17",
        statement=(
            "C3a: R5's Iwahashi classifier vs B6's and the stronger TerraMind's, macro-F1 over "
            "the 15 groups at the 60% rule, holding if both have lower > 0. C3b: B3 vs B7 at "
            "100%, mIoU, competitive if lower > −δ."
        ),
        home="§0, §3.2",
        source="#193, #192",
        value={"pairs": _C3_PAIRS, "iwahashi_rule": 0.60, "iwahashi_groups": 15},
        registered_in="L17 C3's two halves",
        units="C3's marked pairs; the Iwahashi tile-majority fraction and group count",
        unanchored=["iwahashi_rule", "iwahashi_groups"],
    ),
    Rule(
        rule="L18",
        statement=(
            "Eight marked bootstrap pairs, named in L12 (4), L16 (1) and L17 (3); every other "
            "pair is computed and left unmarked in print."
        ),
        home="§6",
        source="#192",
        value={"marked_pairs": [*_C1_PAIRS, _C2_PAIR, *_C3_PAIRS]},
        registered_in="L18 eight marked pairs",
        units="arm pairs marked in print",
    ),
    Rule(
        rule="L19",
        statement=(
            "Val realism caveat: every rung selects its checkpoint on all 2,000 val tiles, which "
            "a real 200-label user would not have; it costs realism, not fairness."
        ),
        home="§4.5",
        source="#196",
        unmechanised_because=(
            "a caveat that applies equally to every arm; there is nothing for a run to clear"
        ),
    ),
    Rule(
        rule="L20",
        statement=(
            "A run may be rerun only for an infrastructure failure, with the same seed and "
            "ladder and the reason in the run record; a finished run is never replaced, and "
            "class collapse counts as finished."
        ),
        home="§4.4",
        source="#192",
        unmechanised_because=(
            "it governs which runs may exist, which the results and the run record enforce; "
            "there is no constant to hold"
        ),
    ),
)

#: Task 2's first link: value → the §7 fragments it renders.
_TASK2_ANCHORS: Final[dict[str, Callable[[Any], tuple[str, ...]]]] = {
    "L1": lambda v: (f"claim metrics, {v['population']} only",),
    "L2": lambda v: (
        f"{v['unit']} bootstrap",
        f"{v['resamples']:,} resamples",
        f"{v['test_geocells']} test geocells",
    ),
    "L3": lambda v: (f"{v['seeds']} seeds", f"{' and '.join(v['including'])} included"),
    "L6": lambda v: (f"{v} tokenizer epochs",),
    "L12": lambda v: (f"{_spelled(len(v['pairs']))} pairs at {v['rung']}",),
    "L15": lambda v: (f"a margin of {v} mIoU point",),
    "L16": lambda v: (f"{v['ladders_required']} of {v['ladders']} ladders",),
    "L17": lambda v: (f"C3's {_spelled(len({p['claim'] for p in v['pairs']}))} halves",),
    "L18": lambda v: (f"{_spelled(len(v['marked_pairs']))} marked pairs",),
}


def _task2_consistency(values: dict[str, Any]) -> None:
    """The ties between Task 2's entries that §7 does not state — and the tie to Task 1.

    These are what hold the unanchored fields: a marked pair's metric must be L1's for its label
    set, so L1's metric names and the pairs cannot move apart; L12's pairs must sit at L12's
    rung; L18 must be exactly L12's, L16's and L17's pairs; L16's ladders must be L3's seeds
    (seed i trains on ladder i). And L6 must be Task 1's R4: one tokenizer, one citation.
    """
    l1, l3, l12, l16, l17, l18 = (values[k] for k in ("L1", "L3", "L12", "L16", "L17", "L18"))
    named = [*l12["pairs"], l16["pair"], *l17["pairs"]]
    if l18["marked_pairs"] != named:
        raise ValueError(
            "L18's marked pairs are not L12's, L16's and L17's, in that order — the register "
            "names its pairs once, in the claims, and L18 counts them."
        )
    if len({(tuple(p["pair"]), tuple(p["rungs"]), p["label_set"]) for p in named}) != len(named):
        raise ValueError("L18: the marked pairs are not distinct, so fewer than eight are marked.")
    for owner, pairs, claims in (
        ("L12", l12["pairs"], {"C1"}),
        ("L16", [l16["pair"]], {"C2"}),
        ("L17", l17["pairs"], {"C3a", "C3b"}),
    ):
        for p in pairs:
            if p["claim"] not in claims:
                raise ValueError(f"{owner} marks {p['pair']} for {p['claim']}, not {sorted(claims)}.")
    for p in named:
        want = l1["claim_metrics"].get(p["label_set"])
        if p["metric"] != want:
            raise ValueError(
                f"L1 reads {p['label_set']} claims on {want!r}, and {p['claim']}'s pair "
                f"{p['pair']} is marked on {p['metric']!r}."
            )
    for p in l12["pairs"]:
        if p["rungs"] != [l12["rung"], l12["rung"]]:
            raise ValueError(f"L12 reads C1 at {l12['rung']}, and {p['pair']} sits at {p['rungs']}.")
    if l16["ladders"] != l3["seeds"]:
        raise ValueError(
            f"L16 counts {l16['ladders']} ladders and L3 registers {l3['seeds']} seeds — seed i "
            "trains on ladder i, so the two are one number."
        )
    if not 0 < l16["ladders_required"] <= l16["ladders"]:
        raise ValueError(
            f"L16: {l16['ladders_required']} of {l16['ladders']} ladders is not a condition."
        )
    r4 = rule("R4").value
    if values["L6"] != r4:
        raise ValueError(
            f"L6 registers {values['L6']} DEM-tokenizer epochs and Task 1's R4 registers {r4}. "
            "It is one tokenizer and one per-modality citation; amend both registers together."
        )


@dataclass(frozen=True)
class LandformBar(RegisteredBar):
    """Task 2's loaded bar, checked against §7 — the readers a run and the report use.

    Past :mod:`benchmark.prereg` no Task 2 threshold is typed: δ, the resample count, the seed
    and ladder counts and the marked pairs are read here, off the committed file.
    """

    @property
    def claim_metrics(self) -> dict[str, str]:
        """L1: the metric each label set's claims are read on."""
        return dict(self.value("L1")["claim_metrics"])

    @property
    def bootstrap_resamples(self) -> int:
        """L2's 10,000 tile resamples."""
        return self.value("L2")["resamples"]

    @property
    def test_geocells(self) -> int:
        """L2's registered caveat: the test set's 13 geocells."""
        return self.value("L2")["test_geocells"]

    @property
    def seeds(self) -> int:
        """L3's seeds per core cell, B3 and B7 included."""
        return self.value("L3")["seeds"]

    @property
    def terramind_dem_epochs(self) -> int:
        """L6's 20, the per-modality figure Task 1's R4 also carries."""
        return self.value("L6")

    @property
    def margin(self) -> float:
        """L15's δ, in mIoU points."""
        return self.value("L15")

    @property
    def c1_rung(self) -> str:
        """L12's rung."""
        return self.value("L12")["rung"]

    @property
    def ladders(self) -> int:
        """L16's ladder count — L3's seeds, one ladder per seed."""
        return self.value("L16")["ladders"]

    @property
    def ladders_required(self) -> int:
        """L16's 2: ladders whose difference must be ≥ −δ for C2 to be claimed."""
        return self.value("L16")["ladders_required"]

    @property
    def iwahashi_rule(self) -> float:
        """L17's tile-majority fraction C3a is read at (unanchored in §7; see :data:`TASK2`)."""
        return self.value("L17")["iwahashi_rule"]

    @property
    def iwahashi_groups(self) -> int:
        """L17's Iwahashi group count (unanchored in §7; see :data:`TASK2`)."""
        return self.value("L17")["iwahashi_groups"]

    @property
    def marked_pairs(self) -> tuple[dict[str, Any], ...]:
        """L18's eight, in the order the claims name them."""
        return tuple(dict(p) for p in self.value("L18")["marked_pairs"])


#: Task 2's register, through the same :class:`Register` Task 1's bar loads through.
TASK2: Final = Register(
    bar_name=TASK2_BAR_NAME,
    plan_path=TASK2_PLAN_PATH,
    section=TASK2_SECTION,
    mark="§7",
    index="§7 item 10",
    adr=TASK2_ADR,
    adr_name="ADR-0015",
    purpose=(
        "The numeric subset of ADR-0015's register of twenty rules, committed by "
        "`benchmark.launch freeze --task landform` before any head trains. Every entry carries "
        "its value and the exact §7 phrase it was registered in; `benchmark.prereg` refuses a "
        "bar whose phrase no longer appears in §7, by the same check Task 1's bar passes "
        "against its §6, and names the entry that lost its anchor. Eleven discursive rules are "
        "recorded here without values. A field listed under an entry's `unanchored` is a value "
        "§7 does not state: it is tied to the other entries by the register's cross-checks and "
        "to nothing in the plan, until §7 item 10 and ADR-0015 are amended to state it."
    ),
    entries=TASK2_REGISTER,
    anchors=_TASK2_ANCHORS,
    consistency=_task2_consistency,
    bar_type=LandformBar,
)

#: Each task's register, keyed by `benchmark.launch`'s ``--task`` value.
REGISTERS: Final = {"void-fill": TASK1, "landform": TASK2}
