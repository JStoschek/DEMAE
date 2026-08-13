# ADR 0012 — The frozen-feature extractor contract: raw metres in, all twelve blocks out, committed as a file

- **Status:** Accepted
- **Date:** 2026-08-11
- **Amended by:** the build, 2026-08-12 ([issue #156](https://github.com/JStoschek/DEMAE/issues/156)) — Decision 5's "CLS is DEMAE-family only" is narrowed to what its own reason supports; see the note under that decision. And 2026-08-13 ([issue #157](https://github.com/JStoschek/DEMAE/issues/157)) — Decision 6's *unmeasured half* is measured; see [Amendment 1](#amendment-1-2026-08-13--decision-6s-unmeasured-half-is-measured). No decision changes in either.
- **Context ticket:** "Decide what the frozen-feature extractor commits to, now that Task 2 inherits a contract and not a cache" ([issue #137](https://github.com/JStoschek/DEMAE/issues/137)), part of the Task-1 void-fill map ([#117](https://github.com/JStoschek/DEMAE/issues/117))
- **Evidence:** [`p118-terramind-encoder-probe.md`](../research/p118-terramind-encoder-probe.md) §5.3 and §7 (TerraMind's intermediates and the normalization trap); [`src/pretrain/model.py`](../../src/pretrain/model.py) `encode`/`_transform`; [`src/dataset/loader.py`](../../src/dataset/loader.py) (the manifest-stat read); two measurements taken for this ticket, reproducible from [`p137-extractor-contract-checks.md`](../research/p137-extractor-contract-checks.md)
- **Amends:** [`docs/research/task1-void-fill-plan.md`](../research/task1-void-fill-plan.md) §2.1, §3.4, §4.1, §9.4 *(Amended 2026-08-11 — [#129](https://github.com/JStoschek/DEMAE/issues/129): the committed-artifacts section was §3.5 when this ADR was written; [#128](https://github.com/JStoschek/DEMAE/issues/128) deleted the Diff-DEM cross-eval section above it and everything below renumbered. Pointer only — no decision in this ADR changes)*
- **Depends on:** [ADR-0011](0011-terramind-posemb-rederived-at-14x14.md) — the TerraMind arm's `pos_emb` derivation is a declared field of this contract, not a separate convention

## Context

[#121](https://github.com/JStoschek/DEMAE/issues/121) killed the premise §9.4 rested on. Every
Task 1 arm's encoder input is an **A1-filled voided** tile; Task 2 probes **clean** tiles; so Task 1
computes no clean-tile feature at any point, and there was never an artifact for Task 2 to reuse —
with a cache or without one. What genuinely crosses the task boundary is the **extractor**: the code
path from tile to frozen patch tokens. #121 named its fields — normalization, layer indices, dtype —
and fixed none of their values. This map's one declared cross-task constraint ([#117](https://github.com/JStoschek/DEMAE/issues/117))
is that whatever Task 1 settles must be reusable by Task 2 *unchanged*, so the values are the whole
question.

Four facts framed the decision, three of them read out of the code and the corpus rather than argued.

**1. The two encoders disagree about what "the last layer" is.** DEMAE's
[`_transform`](../../src/pretrain/model.py) runs the twelve blocks and then applies `self.norm`, so
`encode()` returns a *normed* tensor. p118 §5.3 measured that TerraMind's `return_intermediates=True`
yields twelve tensors with **no separate post-norm entry**, its last one bit-identical to `encoder(x)`.
`task2-landform-plan.md` §4.2 names indices "3, 6, 9, 12 of 12 **for every encoder**", so under a
naive shared index, "layer 12" would mean a normed tensor on one side and an un-normed one on the other.

**2. DEMAE has no intermediates path at all.** `return_intermediates`, `get_intermediate_layers` and
`hidden_states` appear nowhere in `src/` or `tests/`. The only frozen-feature entry point is `encode`,
which returns `(N, 1 + 196, 768)`.

**3. Recomputing per-tile statistics reproduces the manifest.** Phase 1's loader reads the manifest's
stored `mean`/`std` columns rather than recomputing them. Over 500 real tiles from `data/corpus_500k`
(18 of them containing ocean), recomputing land mean/std from `raw != 0` differs from the stored values
by **max 3.6e-4 m** on the mean (median 8.7e-6) and **max 1.1e-7 relative** on the std — float32
accumulation order, not a definitional gap.

**4. Precision is immaterial by this map's own pre-registered yardstick.** On the R5 release encoder
over 64 corpus tiles / 12,544 patch tokens, fp32 against bf16 autocast gives mean per-token cosine
**0.999968** (min 0.999431, p01 0.999809), relative RMS 0.81%. [#134](https://github.com/JStoschek/DEMAE/issues/134)
fixed ≥ 0.99 as the threshold at which a feature difference is immaterial, and rejected the bicubic
`pos_emb` at 0.9708. Precision sits three orders of magnitude the other side of that line. Measured on
**MPS and on the DEMAE family only** — terratorch is not installed in any live worktree, so the
TerraMind arm's figure does not exist.

## Decision

**1. The extractor takes raw int16 metres and a valid-pixel mask, and applies its own declared
normalization.** It does not accept a pre-normalized tensor. p118 §7 measured the trap this closes:
TerraMind wants `(metres − 670.665) / 951.272`, DEMAE wants per-tile standardization, and the TerraMind
encoder **accepts either silently**, returning a plausible `[1, 768, 14, 14]` both ways. Making the
pretraining loader's output un-passable renders the bug unrepresentable rather than merely tested for.
The mask is defined as **valid = land ∧ ¬void**: Task 1 passes the void-aware mask (§2.1's non-void
stats, which is what stops the leak), Task 2 passes land-only, and the same code path serves both.
TerraMind's normalizer ignores the mask — its affine is global, and the A1 fill happened upstream.

**2. Statistics are always recomputed from the passed mask; never read from the manifest.** One code
path, no branch whose two sides belong to different tasks. Context fact 3 is the price and it is
negligible — but it is pinned as a *checked claim*, a test asserting recomputed-equals-manifest over a
tile sample at the measured bound, so that a future drift in the ETL's land definition surfaces in
Task 1 rather than in Task 2.

**3. The extractor emits all twelve block outputs plus a separate `output` field.** Task 1's §4.1 head
and Task 2's linear probes read `output` — each encoder's own final tensor, normed for DEMAE, identical
to block 12 for TerraMind. Task 2's UperNet reads `intermediates`, selecting its own indices. Building
Task 1's extractor to emit intermediates costs Task 1 nothing it needs and is the whole of what makes
it reusable; a last-layer-only extractor means Task 2 writes a second one and "reusable unchanged"
quietly fails. **The spec carries a field-level note that DEMAE's `intermediates[11]` is *not* its
`output`**, so nobody discovers the final norm by reading tensors. The indices are *not* narrowed to
Task 2's current {3, 6, 9, 12}: §4.2 calls layer choice a known sensitivity, and hard-coding four
indices into Task 1's contract would foreclose Task 2's own sweep.

**4. Task 1 owes the intermediates path a test even though no Task 1 arm reads it.** Shapes, ordering,
count, and the layer-12/`output` relation on both sides. This repo has a named failure for exactly the
shape of an untested seam — *the composition* — and an unexercised path shipped with a contract's
authority is that failure with a document attached.

**5. Canonical layout is `(N, 196, D)` patch tokens, row-major, CLS as an optional side field.** DEMAE
emits a sequence with CLS in front; TerraMind emits `[N, 768, 14, 14]` spatial with no CLS. Sequence is
canonical because it is the form both encoders' patch tokens actually are, and a `(D, 14, 14)` canonical
form would bake an unpacking convention into every consumer. Row-major is not a coin flip on either
side: DEMAE tokenizes through timm's `PatchEmbed` (`flatten(2).transpose(1,2)` over `[B, D, 14, 14]`)
and TerraMind's spatial tensor flattens identically against the same grid its `pos_emb` is built on.
**CLS is DEMAE-family only and never enters a cross-encoder comparison** — TerraMind structurally has
none, so any statistic pooled over it would compare one model's token to the other's absence.

> *(Amended 2026-08-12 — [#156](https://github.com/JStoschek/DEMAE/issues/156).* This sentence was
> written with two encoders in view, and the build had to resolve it against a three-family
> registry: **ImageNet-MAE is a ViT and structurally has a CLS token.** Read against the reason
> this decision gives — TerraMind's *absence* — it sits on the DEMAE side of the line, and
> emitting `None` for it would be a false claim about the architecture rather than a conservative
> one. So the emitted field follows the architecture, declared per encoder as
> `benchmark.arms.Encoder.has_cls`, and the clause that does the work is unchanged and is about
> **comparison**: no statistic pooled over CLS crosses encoders, whichever of them have one.*)

**6. fp32 is the reference tensor; compute dtype is a declared field; storage dtype is outside the
contract.** Context fact 4 is why the field may differ per task rather than being pinned: the
difference is 0.99997 where the map's own threshold is 0.99. A task moving its compute dtype re-checks
it under that threshold. Task 2 casting to fp16 for its own cache is Task 2's call and owes its own
note. "Is fp16 emission part of the contract" is a category error — the contract is about what the
extractor computes, not what a consumer writes to disk. **The unmeasured half is named rather than
inherited:** the TerraMind arm's precision agreement does not exist yet, and the build session measures
it on first run.

**7. The contract is a committed `extractor_spec.json`, not prose.** It joins the contract family whose
defining property is that *the committed file is what later phases re-read* — the selection list, the
split assignment, the val panel, the probe set. The failure that family exists to prevent is precisely
the one available here: Task 2 re-runs "the same extractor" from a description, gets something subtly
different, and both tasks' C1 claims silently stop being about the same features. Like its siblings it
**re-checks itself at load**, refusing rather than warning:

1. **Encoder identity** — declared checkpoint md5 and architecture against the model actually loaded.
   This is where §2.3's R4-vs-R5 `pos_embed` axis-order precondition is enforced rather than remembered.
2. **TerraMind's constants** — the **named** v1 lookup `v1_pretraining_mean/std["tok_dem@224"]`,
   whose value is then checked against `670.665 / 951.272`, inheriting p118's refusal. The two
   halves catch different things and §2.1 needs both: [#138](https://github.com/JStoschek/DEMAE/issues/138)
   found **three** DEM constant pairs shipping across IBM's artifacts (651.663/928.168 in the dataset
   loader, 435.726/560.326 in the v0.1 tokenizer), so the *name* is what selects the pair matching the
   v1 checkpoint this plan benchmarks — hardcoding a value would silently accept whichever pair happened
   to sit at that number. The *value* check is what catches a terratorch upgrade retuning the pair the
   name resolves to. §2.1's "by name rather than by value" is about **selection**, and does not mean the
   value goes unchecked.
3. **Shape and layer count** — twelve intermediates, `(N, 196, D)`, `D` matching the declared width.
4. **The ADR-0011 `pos_emb` derivation** for the TerraMind arm, as a declared field rather than a
   convention living in extraction code.

The spec's own md5 is recorded in the **benchmark's** run record, so a results table says which contract
produced it. This is deliberately *not* a fifth field in Phase 1's data fingerprint, which is a frozen
four-field block.

## Consequences

- **The map's one cross-task constraint becomes checkable.** "Reusable by Task 2 unchanged" stops being
  a sentence in the map's Notes and becomes a file Task 2 loads, with four assertions that fail if it
  is not.
- **Task 1 carries code no Task 1 arm exercises at runtime** — the intermediates path — justified by
  Decision 4's test rather than by intent. This is a real cost and the honest framing is that the test
  is what makes it a feature rather than a plausible-looking function.
- **Task 2's index 12 is not Task 2's linear-probe input.** Under Decision 3, a Task 2 layer sweep that
  includes index 12 reads DEMAE's raw block-12 output while its linear probe reads the normed `output`.
  That is a *declared* asymmetry, not a hidden one, and the alternative — norming every DEMAE
  intermediate — was rejected for giving one encoder a learned affine its opponent has no counterpart
  for, which is what §6's fairness checklist exists to catch.
- **`task2-landform-plan.md` stays wrong on purpose, and now has somewhere to point.** Its §3.3, §4.2
  and §4.4 step 3 claim Task 1 feature caches that never existed. Task 2 is out of this map's scope; its
  map inherits this ADR and decides its own caching, which its 4-layer × 30-epoch read pattern may well
  push the other way from Task 1's single inline forward.
- **One number in this ADR is a DEMAE-only measurement presented beside a two-encoder contract.**
  Decision 6 names it rather than generalizing it. A reader who assumes the 0.99997 covers TerraMind is
  reading something this ADR does not say.

## Alternatives considered

- **Normalization declared per arm on a pre-normalized input.** What §9.4 implied. Rejected: it leaves
  the p118 trap expressible and guarded only by a test, when the failure mode is completely silent —
  the encoder returns a well-shaped, plausible tensor from an input distribution it has never seen.
- **Last layer only, Task 2 writes its own extractor.** Rejected: it is the option under which the
  map's cross-task constraint fails, and it saves Task 1 nothing, since the intermediates are already
  computed and emitting them is a flag.
- **Emit exactly {3, 6, 9, 12}.** Rejected: pins Task 2's current choice into Task 1's contract on a
  question Task 2's own plan calls a known sensitivity.
- **Apply the final norm to every emitted intermediate** (timm / DINOv2 `get_intermediate_layers(norm=True)`).
  Rejected — see Consequences.
- **Un-normed on both sides, no separate `output` field.** Rejected: Task 1's head would read raw
  block 12, which is not what `encode`, §10.3's kNN probe or P102's behavioural check call the encoder
  output. DEMAE's Task 1 features would stop being the features Phase 1 measured.
- **Read the manifest's stored stats for clean tiles.** Rejected: exact numerical continuity with
  Phase 1, at the cost of a branch only one side of which Task 1 ever exercises. Decision 2's checked
  claim buys the continuity without the branch.
- **Pin fp32 as the compute dtype for both tasks.** Rejected on measurement, not on cost — 0.999968
  against a 0.99 bar. It would add an autocast exception inside §4.1's bf16 training step to buy an
  exactness no claim rests on.
- **Plan prose plus a §0 decision-table row, no ADR** (and: folding this into the map's open C1–C4
  pre-registration patch). Rejected: [#126](https://github.com/JStoschek/DEMAE/issues/126)'s reason for
  declining C4 an ADR — "C1–C3 have none" — was about a *claim*. This is an *interface*, cross-task and
  hard to reverse, and ADR-0011 already set this map's precedent for interface calls where we diverge
  from a library's default path.

## Amendments

### Amendment 1 (2026-08-13) — Decision 6's unmeasured half is measured

**Source:** the build of the TerraMind half of this contract,
[#157](https://github.com/JStoschek/DEMAE/issues/157), read out in
[`p157-terramind-precision.md`](../research/p157-terramind-precision.md). **Supersedes:** nothing.
Decision 6 stands exactly as taken — fp32 is the reference, the compute dtype is a per-task
declared field, storage dtype is outside the contract. What it *said* was that the TerraMind
arm's precision agreement did not exist and the build session would measure it on first run. It
did, and this is the number.

**The outcome.** Mean per-token cosine between `torch.autocast(bf16)` and the fp32 reference is
**0.99888** over 60 corpus tiles / 11,760 tokens on MPS, against #134's 0.99 bar. Immaterial, so
A5 and A9 keep the declared `bf16` compute dtype and nothing in the arm table moves.

**And the reason not to have inherited it stands, even though the verdict matched.** Beside the
DEMAE-family figure this ADR's Context fact 4 quotes, every column is the same verdict at a
different magnitude — mean 35× further from 1, 1st percentile 77×, worst token 188×, relative RMS
5.9×. The row that matters: **TerraMind's 1st-percentile token sits at 0.9853, below the bar its
mean clears comfortably**, where DEMAE's worst token of 12,544 was 0.9994. Nothing in scope reads
a token in isolation, so this changes no decision; it is recorded because it is precisely what a
reader assuming the 0.99997 covered TerraMind would have got wrong. Consequence 5 of this ADR
said a reader who assumes that is reading something this ADR does not say. They now have the
other number instead.

**Also settled by that build, and worth writing down where the contract lives:** the encoder
`benchmark.extract` runs holds **90,126,336** parameters against p118 §8's published 90,172,416.
The 46,080 gap is `768 × (16² − 14²)` — ADR-0011's `pos_emb` crop, frozen position, nothing
trainable — and p157 records the arithmetic as a checked field so the two counts are reconciled
rather than left looking like a disagreement.
