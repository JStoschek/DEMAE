"""DEMAE Phase 1 — MAE pretraining on the shipped 500k corpus.

The package beside :mod:`dataset`, and the division between them is deliberate: `dataset` is
the Phase-0 ETL core and stays lean and torch-free (ADR-0003 keeps even its loader's torch
import lazy), while `pretrain` is where torch and timm are real dependencies. A box that only
builds corpora never installs a CUDA stack; the rented training box installs
``demae[pretrain]``.

What lives here is [`docs/phase1-pretraining-plan.md`](../../docs/phase1-pretraining-plan.md)
made executable. This ticket (#89) lays the three foundations everything else stands on:

- :mod:`pretrain.config` — the frozen :class:`~pretrain.config.RunConfig`, every §2-§8 recipe
  knob at its cited default, round-tripping to a plain dict and hashing to a stable
  ``config_hash``.
- :mod:`pretrain.fingerprint` — the data-fingerprint block every checkpoint and W&B run has to
  carry: ``dataset_version``, the selection-list md5, the split md5, and the git SHA.
- :mod:`pretrain.data` — the split-aware Dataset, and with it plan §15.2's assertion that **no
  test geocell is reachable from a train batch**. It fails at construction, loudly: it is the
  single check standing between DEMAE and silently pretraining on its own Phase-2 evaluation
  regions, a failure that would be invisible in every loss curve.

On top of them, #90 lands the model itself and #92 the objective it trains against:

- :mod:`pretrain.model` — the MAE forward pass. A timm-blocks ViT encoder over the 49 visible
  tokens, a shallow decoder over all 196, and the 75% masking that sits between them, in the
  model rather than the loader exactly as ADR-0003 assumes.
- :mod:`pretrain.loss` — §4's objective over what the model returns: MSE on the masked patches
  only, against per-patch-normalized targets, with the two guards that stop a near-flat DEM
  patch from being scored on its own quantization noise. It also carries the pixel-weight seam
  #96 extends and the per-patch values §4 asks to be logged.

``python -m pretrain`` prints the config + fingerprint header a real run would record.

Note what this module re-exports and what it does not: :mod:`~pretrain.model`,
:mod:`~pretrain.data` and :mod:`~pretrain.loss` are imported from their own modules, because
each pulls in torch at import time and ``python -m pretrain`` — the header, which is pure
config — should keep working on a box that has no CUDA stack at all.
"""

from .config import RunConfig
from .fingerprint import DataFingerprint, collect_fingerprint, fingerprint_for

__all__ = ["RunConfig", "DataFingerprint", "collect_fingerprint", "fingerprint_for"]
