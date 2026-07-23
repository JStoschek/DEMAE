# E2.4 Gallery review protocol — pre-registered

**Ticket:** [Pin Gate E's pass/fail bar (#47)](https://github.com/JStoschek/DEMAE/issues/47) · **Map:** [Phase E (#46)](https://github.com/JStoschek/DEMAE/issues/46) · **Applied at:** [Gate E review (#54)](https://github.com/JStoschek/DEMAE/issues/54)

This is the one E2 analysis that stays a **human judgement**, not code — you cannot
`assert gallery_looks_right`. So the review is *structured* instead of automated: a fixed
set of questions, decided **before** the gallery ([#53](https://github.com/JStoschek/DEMAE/issues/53))
is rendered, so the reviewer answers a form rather than forming a verdict from scratch. The
machine-readable class, knob binding and fail condition live in
[`e2-gate-bar.json`](e2-gate-bar.json) under `E2.4_gallery_coherence`; this doc is the
procedure that produces those answers.

## What the gallery is actually asking

The gallery is the human backstop that k=20 strata are **terrain-meaningful** — that the
k-means partition tracks real terrain families, not arbitrary cuts through a cloud. The
design constraint that shapes every question below: [#41](https://github.com/JStoschek/DEMAE/issues/41)
found the descriptor space is a **continuum, not clumps**. Adjacent strata *blending* is
therefore *expected*, not a failure.

## The gate: internal coherence

The failure that implicates k is a stratum whose 12 tiles have **nothing in common** — no
shared terrain theme. That means the k-means cell straddles unrelated terrain, which is the
"over/under-split" trigger in the plan §215–228 knob table.

**Per stratum, mark one of:**

| mark | meaning |
|---|---|
| **coherent** | the 12 tiles read as variations on one terrain theme |
| **mixed** | mostly one theme, but 2–3 tiles don't belong |
| **incoherent** | no shared theme; the tiles look unrelated |

**Pre-registered fail condition (a Finding, implicating `k`):**

> **any stratum marked `incoherent`, OR ≥ 3 strata marked `mixed`.**

One `mixed` in a continuum is expected — a stratum sampling a busy boundary region. A
*scatter* of them across the map says the partition isn't tracking terrain.

## The diagnostic prompt: naming (recorded, not gated)

Per stratum, write a short **terrain-family label** — e.g. "dissected uplands", "coastal
plain + estuary", "low-relief arid". This is **not** pass/fail: #41 says the space is a
continuum, so some strata *won't* have a clean folk name, and that's fine. The labels are
the honest artifact — they're what a `mixed` verdict gets justified against, and they seed
the stratum descriptions the E3 notes ([#55](https://github.com/JStoschek/DEMAE/issues/55))
and ADR-0005 ([#56](https://github.com/JStoschek/DEMAE/issues/56)) will want. A stratum you
**can't label at all** is a soft signal feeding the coherence call above, not an
independent gate.

## Explicitly disavowed: distinguishability

Do **not** fail a stratum because it *resembles a neighbour*. "Can you tell stratum A from
stratum B?" is the **wrong** question: in a continuum, neighbouring k-means cells share a
boundary and the tiles near it are genuinely ambiguous — that's what a continuous descriptor
space *is*, not a clustering bug. Gating on pairwise separability would fail a correct k=20
partition. Coherence is an **intra-stratum** property; judge each stratum on its own.

## Procedure rules (fix the review so it can't drift)

1. **Judge each stratum's 12 tiles on their own**, without the adjacent strata shown for
   comparison. This structurally prevents the review from sliding into the disavowed
   distinguishability question.
2. **The 12 tiles are a fixed random sample with a committed seed.** [#53](https://github.com/JStoschek/DEMAE/issues/53)
   commits the seed alongside the gallery so the exact 240 tiles the verdict was rendered
   against are reproducible — otherwise "the gallery looked fine" is unfalsifiable after
   the fact.
3. **Record all 20 marks and all 20 labels**, even on a pass — they're the evidence for
   the verdict and the seed for #55/#56.

## If falsified

The strata are falsified when the terrain partition is **not visually recoverable**. The
fresh map's fallback is plan risk #2 — the published **Iwahashi 40-cluster global product**
as an external stratum label. That is a fresh map, not a repair here: this gate names the
knob (`k`), it does not prescribe the new value.
