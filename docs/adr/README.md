# Architecture Decision Records

One file per decision: `NNNN-short-slug.md`, numbered sequentially. Created lazily
as wayfinder decision tickets resolve (see `../agents/domain.md`). Each ADR states
the context, the decision, and its consequences. When new work contradicts an
existing ADR, surface it rather than silently overriding.

## Index

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-corpus-construction-recipe.md) | Corpus-construction recipe (Phase 0) | Accepted — §4 amended by Gate A3; §3 amended by Phase C2; §4–5 metric basis superseded by [0004](0004-frozen-metric-vector.md); §4–5 strata/quotas superseded by [0005](0005-sampling-design.md) |
| [0002](0002-storage-format-and-manifest.md) | Storage format and manifest | Accepted |
| [0003](0003-dataloader-design.md) | Dataloader design | Accepted — §3's `global` statistic amended by Phase F (tier-specific; within-tier ablation only) |
| [0004](0004-frozen-metric-vector.md) | Frozen metric vector (Phase B / Gate B) | Accepted |
| [0005](0005-sampling-design.md) | Sampling design — stratification, allocation, within-stratum selection (Phase D / Gate E) | Accepted |
| [0006](0006-accept-edm-flm-hollowing-stratum19.md) | Accept the EDM/FLM gate's high-alpine hollowing of stratum 19 | Accepted |
| [0007](0007-accept-degenerate-flatness-gate-flat-strata.md) | Accept the `degenerate` flatness gate's concentration on flat strata 2/5/7/17 | Accepted — Phase-F revisit trigger fired (Gate F `F3`) and **dispositioned by [0008](0008-accept-f3-drainage-shortfall-strata-2-17.md)**; see Amendment 1 |
| [0008](0008-accept-f3-drainage-shortfall-strata-2-17.md) | Accept the `F3` drainage shortfall in strata 2 and 17 — Gate F's FAIL, dispositioned | Accepted — with a Phase-2 downstream revisit trigger |
| [0009](0009-one-shipping-entry-point.md) | One shipping entry point, composing resume and monitoring (Phase 1) | Accepted |
| [0010](0010-frozen-phase1-release-recipe.md) | The frozen Phase-1 release recipe, and the decision rule that adjudicates arms | Accepted |
| [0011](0011-terramind-posemb-rederived-at-14x14.md) | TerraMind's position embedding is re-derived at 14x14, not bicubically resized | Accepted |
| [0012](0012-frozen-feature-extractor-contract.md) | The frozen-feature extractor contract: raw metres in, all twelve blocks out, committed as a file | Accepted |

## Supersession

When later work overrides part of an accepted ADR, amend the original in place rather
than editing the decision text away — a reader should see the original reasoning plus
what changed and why. The convention:

- **Header:** set `**Status:** Accepted — partly superseded` (or fully, as applies), and
  add an `- **Amended by:**` line naming the source (gate/ticket, date, issue link).
- **Stale prose:** leave the original sentence intact; append a short italic marker in
  place — e.g. `*(Superseded YYYY-MM-DD — Amendment N: …)*` — pointing at the amendment.
- **Amendment section:** add `## Amendments` before "Alternatives considered", with an
  `### Amendment N (date) — <title>` entry recording the source, what it supersedes, and
  the new verdict. Number amendments sequentially within the ADR.

If a decision is replaced wholesale, prefer a new ADR that says "supersedes ADR-NNNN" and
mark the old one's status accordingly.
