# Architecture Decision Records

One file per decision: `NNNN-short-slug.md`, numbered sequentially. Created lazily
as wayfinder decision tickets resolve (see `../agents/domain.md`). Each ADR states
the context, the decision, and its consequences. When new work contradicts an
existing ADR, surface it rather than silently overriding.

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
