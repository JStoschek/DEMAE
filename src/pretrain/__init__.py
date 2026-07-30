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

On top of them, #90 lands the model itself, #91 the input it sees, and #92 the objective it
trains against:

- :mod:`pretrain.model` — the MAE forward pass. A timm-blocks ViT encoder over the 49 visible
  tokens, a shallow decoder over all 196, and the 75% masking that sits between them, in the
  model rather than the loader exactly as ADR-0003 assumes.
- :mod:`pretrain.augment` — plan §6's augmentation, the full dihedral group and nothing else,
  as a wrapper *around* the read path rather than an edit to it. One draw per sample carries
  both the image and its ``land_mask``, which is what keeps the ocean weighting pointed at
  the pixels it thinks it is weighting.
- :mod:`pretrain.loss` — §4's objective over what the model returns: MSE on the masked patches
  only, against per-patch-normalized targets, with the two guards that stop a near-flat DEM
  patch from being scored on its own quantization noise, and the per-patch values §4 asks to be
  logged. #96 adds the layer under that: ocean and void pixels carry zero weight, each patch
  renormalizes by its valid-pixel count, and a masked patch with under 25% valid terrain leaves
  the loss entirely. Because those weights and their targets arrive by two different routes, the
  module also carries the check that they still describe the same pixels — the one #96 failure
  mode that is invisible in every curve.
- :mod:`pretrain.init` — §8's warm start, the one thing R4 does that no other run does: read an
  externally trained ImageNet MAE encoder in, collapse its RGB patch embed to one elevation
  channel by summing, and leave the decoder fresh. Selected by a config flag, so the scratch
  path has no branch in it. Most of the module is refusal: the state dicts do not match
  key-for-key, so the mapping is written out and asserted total in *both* directions, and every
  layout convention taken from mmselfsup is a checked claim with a source link rather than an
  inference from a shape (#102, #103). That policy earned its keep immediately — their positional
  embedding is the transpose of MAE's.

#93 is where those pieces become one running thing — the tracer bullet, and the first code here
that has a loss curve:

- :mod:`pretrain.optim` — AdamW off the config, and §6's one non-default: bias and norm
  parameters excluded from weight decay. The grouping is computed once, over the real model,
  because getting it wrong trains perfectly well and produces a slightly worse encoder.
- :mod:`pretrain.schedule` — the LR curve: linear warmup from 0, cosine to exactly 0. A pure
  function of (step, config), torch-free, with its boundary values pinned by name. Every wrong
  version of it also trains.
- :mod:`pretrain.train` — the step and the loop around it: bf16 autocast and no GradScaler
  (§7.1), gradient accumulation whose gradient equals the single-batch one, the loader assembled
  from `dataset.loader` unchanged, and a seed that makes two runs of one config the same run.

#95 is what makes that loop a thing a rented box can be pointed at — plan §7's efficiency
levers, and the command that applies them:

- :mod:`pretrain.runtime` — `torch.compile(dynamic=False)` over the full model (§7.3), and
  §7.2's SDPA path *confirmed on the blocks the model actually holds* rather than assumed from
  the config. Both levers share one failure mode: they fail by being silently absent, training
  the identical model more slowly, which no loss curve, gradient norm or checkpoint would ever
  report. So the module's real subject is the checking — including a recompile counter, because
  §7.3's 1.1-1.4x is contingent on the graph being compiled once.
- :mod:`pretrain.throughput` — samples/s, step time, and the **loader wait fraction**, per step
  and per epoch. §7.5 names the dataloader as the plan's #1 risk and sets a measured gate on it;
  these are the numbers that gate is read in. #97 builds the dashboard and the ~10% alarm on
  top of them.

#97 is the scalar layer of plan §10 — what a run *reports*, and what fires when the numbers go
wrong. MAE gives few in-flight signals, so these are the ones that exist:

- :mod:`pretrain.sink` — where every number, histogram and rendered grid goes (§11), and the
  seam #98's reconstruction grids and #99's probe trends write into as well. Payload-typed
  rather than ``dict[str, float]`` precisely because two of its three consumers do not emit
  scalars, and the step axis is the **epoch**, shared. Built against W&B's **offline** mode: no
  account, no network, so the whole logging path is exercisable without waiting on #80's ops
  deliverable, and the online target is one config field away.
- :mod:`pretrain.alarms` — §10.1's three conditions as things that fire rather than sentences in
  a document: collapse toward zero (the broken-`norm_pix`-target symptom), val rising while train
  falls, and a plateau at high loss in the first 20 epochs — plus §10.4's ~10% loader-wait
  threshold. Pure functions over trajectories, so each one is fired on a synthetic curve built
  to trigger it and held quiet on one built not to. Thresholds are fractions of the run's own
  first epoch, because the loss scale is a recipe knob (R3 ablates `norm_pix` off).
- :mod:`pretrain.monitor` — the epoch boundary: the **fixed-mask-seed val pass** that makes
  §10.1's val loss comparable at all, §4's per-patch loss distribution cross-tabbed against
  patch flatness (which is what §4's variance-floor rule is actually adjudicated on, and a mean
  cannot answer), the throughput panel over #95's counters, and the alarms checked on the epoch
  series. ``python -m pretrain.monitor --epochs N`` runs it against a real corpus.

``python -m pretrain`` prints the config + fingerprint header a real run would record;
``python -m pretrain.train --steps N`` runs the bare loop. **``python -m pretrain.launch`` is
how a run is actually started** (:mod:`pretrain.launch`): a config file plus repeatable
``--set group.knob=value`` overrides, the levers confirmed before the first step, the counters
reported throughout, and a ``--dry-run`` that assembles everything and spends nothing.

Note what this module re-exports and what it does not: :mod:`~pretrain.model`,
:mod:`~pretrain.data`, :mod:`~pretrain.augment`, :mod:`~pretrain.loss`, :mod:`~pretrain.init`,
:mod:`~pretrain.optim` and :mod:`~pretrain.train` are imported from their own modules, because
each pulls in torch at import time and ``python -m pretrain`` — the header, which is pure config
— should keep working on a box that has no CUDA stack at all. (:mod:`~pretrain.schedule` needs
no torch either, being arithmetic over four numbers, but it is a piece of the loop rather than a
piece of the vocabulary and stays beside its users.)
"""

from .config import RunConfig
from .fingerprint import DataFingerprint, collect_fingerprint, fingerprint_for

__all__ = ["RunConfig", "DataFingerprint", "collect_fingerprint", "fingerprint_for"]
