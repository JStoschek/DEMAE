# ADR 0013 — The Task-1 pre-registration is a register of twelve rules, asserted here and anchored in §6

- **Status:** Accepted
- **Date:** 2026-08-11
- **Amended by:** Task 1 map, 2026-08-12 ([issue #142](https://github.com/JStoschek/DEMAE/issues/142)) — **R12** added, the
  first entry admitted by the tunability test after this ADR was written: A2's row carries no C-claim, is insulated
  from C1 in both directions, its difference against A3 is a joint quantity, and a contingency fires if A2 does not
  beat A1 in a majority of bands. No existing entry changes. The **register, the §6 index and the counts read
  twelve**; [Context](#context) and [Alternatives considered](#alternatives-considered) keep their as-accepted
  figures, because those are measurements taken at `46d90ca` and arguments made against them, and editing them
  would rewrite the reasoning rather than amend the record. This is the Consequences section's procedure working
  as specified — one register row and one marked site, with the tunability test deciding admission.
- **Context ticket:** "Decide whether the pre-registration gets its own ADR, now that it carries seven rules" ([issue #141](https://github.com/JStoschek/DEMAE/issues/141)), part of the Task-1 void-fill map ([#117](https://github.com/JStoschek/DEMAE/issues/117))
- **Evidence:** the duplication and drift counts in [Context](#context) below, measured over [`task1-void-fill-plan.md`](../research/task1-void-fill-plan.md) at `46d90ca`; Gate E's precedent in [ADR-0005 §5](0005-sampling-design.md) and `dataset.validate`; the twelve rules' own source tickets, cited per row in [The register](#the-register)
- **Amends:** [`docs/research/task1-void-fill-plan.md`](../research/task1-void-fill-plan.md) §6 item 8 (becomes the register's index), and marks the registered rule at each site in §0, §2.2, §2.3, §3.3, §5 and §9 *(2026-08-12 — [#142](https://github.com/JStoschek/DEMAE/issues/142) adds §4.3, R12's home, and §8's risk bullet as a marked site)*
- **Depends on:** [ADR-0011](0011-terramind-posemb-rederived-at-14x14.md) — its Decisions 4–5 are register entry **R11**, asserted there and pointed at from here, never restated

## Context

[#126](https://github.com/JStoschek/DEMAE/issues/126) declined an ADR for **C4** on the grounds that
C1–C3 have none, and that formally justifying the newest and weakest claim reads backwards. That
ruling was correct and still is — but it was about *claims*, and the thing that has accumulated
since is not claims. [#137](https://github.com/JStoschek/DEMAE/issues/137) already carved the first
non-claim category off from it, writing [ADR-0012](0012-frozen-feature-extractor-contract.md) on the
explicit ground that "that was a *claim*, this is an *interface*". This is the third kind: **rules
governing how the benchmark's evidence may be used.**

#141 framed the exposure as rules "held together by prose in four different sections", discoverable
only by reading all of them. **Measured, the premise is inverted.** The rules are not under-stated;
they are stated repeatedly and owned nowhere. Every one appears in two to four places, `pre-registered`
appears **18 times** in the plan, and §6 items 3, 4, 7 and 8 already collect six of them:

| Rule | Sites at `46d90ca` |
|---|---|
| +0.15° seam break | §5 ×2, §6 item 8 |
| Four bootstrap pairs | §5, §9, §6 item 8 |
| R5 headline / R4 never C1's number | §0, §2.3 ×2 |
| 20-epoch citation | §2.3 ×3, §6 item 4, §10 |
| Symmetric GSD caveat | §2.2 ×4, §6 item 3 |
| Symmetric corpus caveat | §2.3 ×2, §6 item 4 |
| Printed table shape | §5, §9, §6 item 8 |

So the risk is **drift between copies, not obscurity** — and in this effort that risk is not
hypothetical. It is the single most frequently observed defect on this map:
[#121](https://github.com/JStoschek/DEMAE/issues/121) swept two stale halves (§4.5's "precompute
frozen features", §8's "features are memmapped"); [#122](https://github.com/JStoschek/DEMAE/issues/122)
found §4.2's init line stale twice over, naming a ViT-S run as the ViT-B checkpoint;
[#123](https://github.com/JStoschek/DEMAE/issues/123) found §5's stratification bullet describing an
operation that does not exist; [#128](https://github.com/JStoschek/DEMAE/issues/128) found **ten
sites** where the plan still carried an arm ruled out at charting;
[#139](https://github.com/JStoschek/DEMAE/issues/139) found §8 and §2.2 stating opposite rulings on
the 10 m regime and called it "the third time a parameter has outlived its governing ruling here".
A twelfth copy of the rules would extend that list, not close it.

**The set is also larger than #141 could count, which is the defect in its clearest form.** The
ticket lists seven. The register below held eleven at acceptance *(twelve since
[#142](https://github.com/JStoschek/DEMAE/issues/142))*. The four it missed are §5's metric set — the
whole of #123's 24,000-instance grading — §5's 100-per-band floor (the ticket cites
[#135](https://github.com/JStoschek/DEMAE/issues/135) as its worked example without counting the
rule that example fired), §6 item 7's C3 fallback claim, and ADR-0011's own pre-registered ablation
row. That last one matters most: **a pre-registered rule already lives in an ADR on this map**, with
[#134](https://github.com/JStoschek/DEMAE/issues/134)'s ≥ 0.99 threshold fixed before the numbers
existed and fired at 0.9708. The precedent #141 asks whether to set was set at ADR-0011.

## Decision

**1. The pre-registration is one ADR for the collection, not one per rule and not prose.** All three
of this repo's ADR criteria pass, and the first passes at its strongest: "hard to reverse" is not a
side effect of a pre-registered rule but its definition — a rule revisable after the numbers exist is
worth nothing. Twelve ADRs would multiply the drift surface twelvefold to record a set whose entire
content is that it *is* a set.

**2. This ADR asserts the rules; the plan sections mark and point.** Each site in §0, §2.2, §2.3,
§3.3, §5 and §9 keeps its local reasoning — which is unique to it and is what makes the rule
defensible — but stops independently *committing* to the rule, carrying instead a registration marker
naming the entry. Removing the reasoning was never the intent and would destroy the argument; removing
the *duplicate assertion* is the whole point.

**3. §6 stays a full restatement, explicitly marked as a quotation of this register.** §6 goes in the
writeup verbatim, so it has an external reader this ADR does not. It is the one deliberate duplicate,
and it is deliberate because it ships.

**4. §6 item 8 becomes the register's index, and the rules stay beside their topics.** A bare list of
all of them at item 8 would strip the context that makes each legible — the GSD caveat means something next
to the GSD discussion and little in a list. But the index is what repairs the defect #141 exposed:
the set becomes **countable**.

**5. Membership is the tunability test.** A rule belongs iff **both**: (a) it constrains how results
are read, reported or compared; and (b) a different choice, made after seeing the numbers, could
change which conclusion the table supports. "Fixed before the numbers existed" does **not**
discriminate — this map is plan-only, so it is true of every decision in it. The tunability test
admits all twelve and correctly excludes the 224 px regime (ADR-0011 Decision 1), which is a setup
decision, not an evidence-use one. The plan already reaches for this test without naming it: §6 item
8's "worth nothing decided after the numbers existed" *is* this test.

*(Amended 2026-08-12 — [#142](https://github.com/JStoschek/DEMAE/issues/142).* The test was turned on the map's own
fog and returned **R12**, so it has now done prospective work rather than only classifying rules that already
existed. Two properties of that first application are worth recording, because they are what a future ticket will
copy. **The test found the rule where no claim pointed at one**: A2 is named by none of C1–C4, which is exactly why
its 60 printed numbers had no fixed reading — absence of a claim is a *reason to check*, not an exemption.
And **it separated the tunable half from the untunable one**: the tail risk #142 was chartered on — a large
A2–A3 gap being reframed after the fact — turned out to be the smaller half, because §4.3's "quantifies precisely
what the task-adaptation bought" is unsupportable at *every* outcome, the A2→A3 difference confounding task
adaptation with mask-distribution adaptation and no proposed arm separating them. A test that only guards tails
would have missed it.)

**6. C1–C4 stay in §0 and do not enter the register.** They are the one part of this complex that is
**not** duplicated — stated once, in the writeup's own "what this benchmark must show". Pulling them
in would create exactly the duplication Decision 2 exists to remove. #126's ruling stands unamended;
it simply never applied to rules.

**7. The refusal is a first-class entry (R9b).** #139 declined a threshold on the FSQ diagnostic
because the statistic has no natural scale, and a threshold would imply the instrument settles whether
the shift *costs* TerraMind anything — which it cannot. Reasoning that declines to commit is worth as
much as reasoning that commits, and it is already enforced in code:
[#140](https://github.com/JStoschek/DEMAE/issues/140) committed a test forbidding `threshold`,
`verdict` and `ablation_row` as record keys, and carries the limit as a record field. The refusal had
enforcement and no document; this is the document.

**8. The numeric subset is committed as `task1-prereg-bar.json`, checked against §6 at load.**
Specified here, built by the build session — ADR-0012's precedent for `extractor_spec.json`, and
permitted by this map's plan-only rule for the same reason. Each entry carries its value **and the
exact §6 phrase it is registered in**; the loader refuses a bar whose registered phrase no longer
appears in §6. This is Gate E's mechanism verbatim (`dataset.validate` "refuses to load a bar whose
prose no longer states it"), and it binds to §6 rather than to this ADR because §6 is the copy that
leaves the building — the one whose drift is externally visible. Since §6 is a marked quotation of
this register, pinning the shipped copy pins all three transitively.

Discursive rules are **not** mechanised and the ADR says so: "the caveat is symmetric" has no
machine-checkable form, and inventing one would be the same error #139 refused.

## The register

Twelve rules. **R11 is asserted in ADR-0011 and pointed at here**, per the header's `Depends on`.

| # | Rule | Home | In bar file | Source |
|---|---|---|---|---|
| R1 | Metric set: five headline columns, four appendix, nothing else — graded by measured separation | §5 | ✅ names | [#123](https://github.com/JStoschek/DEMAE/issues/123) |
| R2 | Pretraining-corpus caveat is symmetric — unavailable to explain a DEMAE loss unless also priced as a discount on a DEMAE win | §2.3, §6.4 | ❌ discursive | [#122](https://github.com/JStoschek/DEMAE/issues/122) |
| R3 | R5 is the headline unconditionally; an R4 win is reported prominently but never becomes C1's number | §0, §2.3 | ❌ discursive | [#122](https://github.com/JStoschek/DEMAE/issues/122) |
| R4 | TerraMind's 20 DEM-tokenizer epochs carry their per-modality citation wherever stated — not the paper's 100 | §2.3, §6.4, §10 | ✅ `20` | [#138](https://github.com/JStoschek/DEMAE/issues/138) |
| R5 | The printed table's shape: which axes pool, blobs headline, the 66-cell C4 rank block | §5, §9 | ✅ axes | [#136](https://github.com/JStoschek/DEMAE/issues/136) |
| R6 | Exactly four arm pairs carry a bootstrap CI mark, named in advance, of 55 computed | §5, §9 | ✅ the four | [#136](https://github.com/JStoschek/DEMAE/issues/136) |
| R7 | A seam gap above **+0.15°** is a seam break | §5 | ✅ `0.15` | [#136](https://github.com/JStoschek/DEMAE/issues/136) |
| R8 | Real-medium+ is unstratified unless the realised histogram clears **100 per band** | §3.3, §5 | ✅ `100` | [#121](https://github.com/JStoschek/DEMAE/issues/121) |
| R9 | The GSD caveat is symmetric, and stated as a **bandwidth** shift rather than a resolution handicap | §2.2, §6.3 | ❌ discursive | [#139](https://github.com/JStoschek/DEMAE/issues/139) |
| R9b | **Refusal:** the FSQ diagnostic carries no threshold — the statistic has no natural scale, and one would imply it decides more than it can | §2.2 | ❌ by design | [#139](https://github.com/JStoschek/DEMAE/issues/139), [#140](https://github.com/JStoschek/DEMAE/issues/140) |
| R10 | C3's low-label sweep is the pre-registered fallback claim; nulls reported with the prominence of wins | §6.7 | ❌ discursive | plan §6 |
| R11 | Bicubic-vs-re-derived ablation row, A5 at 20k, dropped iff #134 returns ≥ 0.99 | **[ADR-0011](0011-terramind-posemb-rederived-at-14x14.md) D4–5** | ✅ `0.99` | [#134](https://github.com/JStoschek/DEMAE/issues/134) |
| R12 | A2 carries no C-claim and is insulated from C1 both ways; the A2→A3 difference is joint, never "what task adaptation bought"; negative result if A2 loses to A1 in a majority of bands | §4.3, §6.8 | ❌ discursive | [#142](https://github.com/JStoschek/DEMAE/issues/142) |

R8 and R11 have **already fired**, and both are the argument for the whole arrangement. R8's floor was
fixed at ~96 projected instances and met a measured 438 — 4.6× its projection — and still produced the
right report, because nobody could move the threshold to the number ([#135](https://github.com/JStoschek/DEMAE/issues/135)).
R11 returned 0.9708 against a bar of 0.99 and the row stayed. Neither outcome was available to argue
with, which is the only property that made either worth registering.

## Consequences

- **The build session inherits a checkable artifact rather than a reading assignment.** It builds
  `task1-prereg-bar.json` to this spec and wires the §6 phrase check; the seven mechanised entries
  cannot drift silently thereafter.
- **A twelfth site is now a defect, not an accident.** Any future ticket adding an evidence-use rule
  adds a register row and one marked site — and the tunability test tells it whether it has one.
- **§6 is load-bearing in a new way.** It was already "goes in the writeup verbatim"; it is now also
  the anchor a committed file validates against, so editing its registered phrases breaks the build
  session's loader. That is the intended coupling, and it is stated here so it is not discovered.
- **Five discursive rules stay unmechanised** *(four until [#142](https://github.com/JStoschek/DEMAE/issues/142)
  added R12)*, R9b irreducibly so. The register makes them countable
  and attributable; it does not make them enforceable, and claiming otherwise would be the error #139
  refused. R12 joins them deliberately rather than by omission: its trigger is an *ordering* against A1, so there
  is no constant for `task1-prereg-bar.json` to hold, which is the same reason its template R10 is discursive.
- **This ADR asserts nothing new.** Every rule in it was already decided by a closed ticket. If it is
  ever found to have changed a rule's content rather than its home, that is a defect in this ADR. *(R12 arrived
  with the 2026-08-12 amendment on the same terms — decided by [#142](https://github.com/JStoschek/DEMAE/issues/142),
  stated in full at §4.3, asserted here.)*

## Alternatives considered

- **An ADR per rule.** Rejected: eleven documents to record one set, and the set is the content. It
  would also strand R9b, which is a refusal and reads as a stub alone.
- **Leave it as prose and complete §6 item 8.** Genuinely competitive and much cheaper — it was the
  strongest alternative, since §6 already collects six of the eleven and already ships. Rejected
  because it leaves the rules with a *collection point* but still no *owner*: §6 would remain one of
  twelve copies with no mechanism making it the authoritative one, which is the state that produced
  every drift instance in [Context](#context).
- **Pull C1–C4 in and make it one pre-registration document.** Rejected under Decision 6 — it would
  duplicate the only unduplicated thing here, and re-open #126 for no gain.
- **Move R11 out of ADR-0011.** Rejected under the header's `Depends on`: the ablation row exists
  *because* of the pos-emb re-derivation and is meaningless apart from it, so moving it would edit
  away ADR-0011's Decision 4–5 — which this repo's supersession convention explicitly forbids
  ("amend the original in place rather than editing the decision text away").
- **Mechanise the discursive rules too** — e.g. a checklist the writeup must tick for R2/R9's
  symmetry. Rejected: symmetry is a property of how a sentence is *used* in argument, and a tick-box
  asserting it would look like enforcement while checking nothing. Same failure mode as a threshold on
  a scale-free statistic (R9b).
