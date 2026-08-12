"""Task 1 — the void-filling benchmark (plan `docs/research/task1-void-fill-plan.md`).

Torch-optional in the same way :mod:`dataset` is: the freeze and scoring paths import no
torch, so the committed artifacts and the A1 classical baseline can be built on a box with no
CUDA stack. Only the arms that run an encoder reach for it, and they reach for it lazily.

``python -m benchmark.launch`` is the **only** command that freezes, runs or reports — ADR-0009's
rule carried into Phase 2, for the reason it was written: *a seam that nobody is on both sides
of is a feature that does not exist*. Any per-stage ``main`` that appears during the build is
documented as **not a way to produce a result**, exactly as ``pretrain.train`` is.
"""
