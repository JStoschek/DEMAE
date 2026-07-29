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

``python -m pretrain`` prints the config + fingerprint header a real run would record.
"""

from .config import RunConfig
from .fingerprint import DataFingerprint, collect_fingerprint, fingerprint_for

__all__ = ["RunConfig", "DataFingerprint", "collect_fingerprint", "fingerprint_for"]
