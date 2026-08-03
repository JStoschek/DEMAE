"""The run config — every Phase-1 recipe knob, frozen, serializable, hashable (issue #89).

One object holds the whole of the [Phase-1 plan](../../docs/phase1-pretraining-plan.md)
§2-§8 at its cited default, and every run — R0's smoke through R5's release — is *this object
with a knob moved*. Nothing downstream reads a recipe number from anywhere else: the model
(#90) builds from :class:`Architecture`, the loss (#92) from :class:`Loss`, the loop (#93)
from :class:`Optimization` + :class:`Schedule`. That is what makes an ablation a diff.

Structure follows the plan's sections rather than the code's packages, so a reviewer can read
this file beside the plan and check it line by line. Every non-obvious default carries the
section that decided it; where the plan is silent the comment says so and names what filled
the gap, because a number nobody argued is the one that quietly becomes doctrine.

## Three properties the rest of Phase 1 leans on

- **Frozen.** A run's config cannot drift mid-run: a checkpoint's recorded config is the one
  the step after it used. Changing anything means constructing a new object (:meth:`replace`).
- **Round-trips.** ``to_dict`` / ``from_dict`` over plain JSON types, so a config survives a
  checkpoint, a W&B page and a YAML file unchanged, and a partial dict fills the rest from the
  plan — which is exactly what an ablation's config file should look like (the one knob it
  moves, nothing else).
- **Hashes to a stable :attr:`RunConfig.config_hash`.** Plan §15.7 tags every W&B run with it.
  It is a keyed hash over canonical JSON — not :func:`hash`, which ``PYTHONHASHSEED``
  randomizes per process — so two boxes agree. It covers the **recipe** and deliberately not
  the four :data:`ENVIRONMENT` fields — the run's label, and the three paths saying where the
  corpus and the split artifacts are mounted. §11 resumes on a fresh vast.ai instance where
  the corpus sits somewhere else, and a hash that moved with the mount point would read every
  resume as a config mismatch. Note what is *not* environment: ``init.checkpoint`` is hashed,
  because *which* checkpoint a continual arm warm-starts from is the arm. What the run *read*
  is fingerprinted separately and far better, by :mod:`pretrain.fingerprint`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, replace
from typing import Any, Final

from dataset import config as dataset_config

#: Marks a field as *where the run happens* rather than *what it does*: recorded in the
#: config, excluded from :attr:`RunConfig.config_hash`. Eight fields carry it — ``run_id``, the
#: three paths in :class:`Data`, and the four W&B settings in :class:`Monitoring` — and marking
#: a ninth is a decision, not a convenience, because every field marked here is a knob two
#: different recipes could disagree on unseen.
#:
#: The four :class:`Monitoring` ones were added by #97 with the same argument the paths carry,
#: and it is worth stating rather than inheriting: *which* W&B project a run reports to, under
#: which account, in which directory, and whether it reports online or offline are all facts
#: about the box and the ops account (#80), not about the objective. Plan §11's resume on a
#: fresh vast.ai instance and the offline-then-online switch §10 is built against would both
#: read as a recipe change otherwise. What the run *measured* — the fixed val mask seed and
#: every alarm threshold — is hashed, because two runs that alarm at different loss levels are
#: not the same experiment.
_ENVIRONMENT_KEY: Final = "environment"
ENVIRONMENT: Final = {_ENVIRONMENT_KEY: True}

#: Length of the config hash, in hex characters. 16 hex = 64 bits: collision-free at the scale
#: of "every config this project will ever run" by many orders of magnitude, and short enough
#: to read off a W&B tag.
CONFIG_HASH_CHARS: Final = 16


#: §2.3's positional encodings, both fixed 2D sin-cos and differing *only* in which half of the
#: width carries which axis — they are each other's transpose over the patch grid.
#: ``sincos2d`` is MAE's own order (column first) and the default everywhere. ``sincos2d_rowfirst``
#: is mmselfsup's (row first), and it exists because §8's continual arm warm-starts from
#: mmselfsup's checkpoint, whose weights learned that order: `torch.meshgrid` defaults to ``'ij'``
#: where `np.meshgrid` defaults to ``'xy'``, and MAE and mmselfsup each wrote the obvious code in
#: their own library ([P102](../../docs/research/p102-mmselfsup-mae-layout.md) §4). Named here
#: rather than in :mod:`pretrain.model` because this module is the torch-free one and a run's
#: recipe knobs have one home; `pretrain.model.AXIS_ORDERS` maps these onto the constructor's
#: argument and a test pins that the two lists agree.
POS_ENCODINGS: Final = ("sincos2d", "sincos2d_rowfirst")

#: §11's sink modes, and they are W&B's own strings rather than a translation of them:
#: ``offline`` writes a local run directory with no account and no network, ``online`` streams
#: to the project #80 provisions, ``disabled`` makes every log call a no-op. The default is
#: ``offline`` — see :class:`Monitoring`.
WANDB_MODES: Final = ("offline", "online", "disabled")

#: §10.3 — how a tile's ``(1 + n_patches, encoder_dim)`` encoding becomes the one feature vector
#: the kNN probe compares (#99).
#:
#: ``mean`` — the average of the **patch** tokens, CLS excluded — is the default, and the
#: argument is that the CLS token is *never supervised during MAE pretraining*: no term in §4's
#: objective touches it, so what it holds after 400 epochs is whatever attention happened to
#: leave there. MAE's own fine-tuning recipe reads global-average-pooled patch tokens for the
#: same reason. ``cls`` is kept because MAE's *linear* probe uses it and #85 may want both.
#:
#: Named here rather than in :mod:`pretrain.probe` for the reason :data:`POS_ENCODINGS` is: this
#: module is the torch-free one and a run's recipe vocabulary has one home.
PROBE_POOLINGS: Final = ("mean", "cls")


def masked_token_count(n_patches: int, ratio: float) -> int:
    """How many tokens a ratio hides — 147 of 196 at 0.75 (plan §3).

    Lives at module scope because two places need the same answer and must not derive it
    twice: :attr:`RunConfig.masked_tokens`, and :class:`pretrain.model.MaskedAutoencoder`,
    which fixes the count at construction so `torch.compile(dynamic=False)` sees static
    shapes (§7.3). A second copy of ``round(ratio * n)`` is a rounding disagreement waiting
    to happen at some ratio nobody tested.
    """
    return int(round(ratio * n_patches))


class _Frozen:
    """What every frozen piece of the config shares: a re-validating :meth:`replace`."""

    def replace(self, **changes):
        """A copy with ``changes`` applied — re-validated, since ``__post_init__`` reruns."""
        return replace(self, **changes)


def _plain(value: Any) -> Any:
    """JSON-native view of a field: tuples become lists, everything else passes through."""
    return list(value) if isinstance(value, tuple) else value


def _to_dict(obj) -> dict:
    """One group as plain JSON types, field by field."""
    return {f.name: _plain(getattr(obj, f.name)) for f in fields(obj)}


def _from_dict(cls: type, d: dict, *, where: str):
    """Build ``cls`` from a partial plain dict, restoring tuple-typed fields.

    An unknown key raises rather than being ignored: a typo'd knob that silently keeps the
    default is the failure mode this whole object exists to prevent — the run would report the
    recipe it was *asked* for and train the one it defaulted to.
    """
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError(f"unknown {where} knob(s) {unknown}; known: {sorted(known)}")
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


# --- §2 Architecture ----------------------------------------------------------


@dataclass(frozen=True)
class Architecture(_Frozen):
    """The encoder/decoder shape (plan §2). Defaults are the ViT-S/16 iteration backbone."""

    #: §2.1 — ViT-S/16, 22M params, the iteration backbone. SSL4EO-S12 benchmarked this scale
    #: on satellite data; ViT-B/16 (:meth:`RunConfig.vit_base`) is the release run.
    encoder_dim: int = 384
    encoder_depth: int = 12
    encoder_heads: int = 6
    #: §2.2 — shallow and narrow. MAE's own ablation is flat across decoder depth 1-8, and the
    #: remote-sensing lineage (Scale-MAE 3, CROMA 1, SpectralGPT 4, Clay 4) converges lower
    #: still; the width is ~2/3 of the encoder's.
    decoder_depth: int = 4
    decoder_dim: int = 256
    decoder_heads: int = 8
    #: The tile as ADR-0001/0002 ship it: 1 x 224 x 224, patch 16 -> 196 tokens.
    img_px: int = dataset_config.TILE_PX
    patch_px: int = dataset_config.PATCH_PX
    in_chans: int = len(dataset_config.INPUT_CHANNELS)
    #: §2.3 — fixed 2D sin-cos, MAE's default. GSD-aware PE (Scale-MAE) is deferred to Phase 4:
    #: the corpus is single-GSD by construction, so GSDPE has nothing to condition on yet.
    #: One of :data:`POS_ENCODINGS`: the second value is the same encoding with the two axis
    #: halves swapped, and it exists for §8's warm start alone (see :data:`POS_ENCODINGS`).
    pos_encoding: str = "sincos2d"

    def __post_init__(self) -> None:
        if self.pos_encoding not in POS_ENCODINGS:
            raise ValueError(
                f"unknown pos_encoding {self.pos_encoding!r} (§2.3: {list(POS_ENCODINGS)})"
            )
        if self.img_px % self.patch_px:
            raise ValueError(f"patch_px {self.patch_px} does not divide img_px {self.img_px}")
        for what, dim, heads in (
            ("encoder", self.encoder_dim, self.encoder_heads),
            ("decoder", self.decoder_dim, self.decoder_heads),
        ):
            if dim % heads:
                raise ValueError(f"{what}_heads {heads} does not divide {what}_dim {dim}")
        if self.in_chans != 1:
            raise ValueError(f"in_chans {self.in_chans}: Phase 1 is elevation-only (§1)")

    @property
    def n_patches(self) -> int:
        """Tokens per tile — 196 at 224/16."""
        return (self.img_px // self.patch_px) ** 2

    @property
    def patch_dim(self) -> int:
        """Prediction width per patch — 16 x 16 x 1 = 256, one of §1's three
        single-channel changes."""
        return self.patch_px * self.patch_px * self.in_chans


# --- §3 Masking ---------------------------------------------------------------


@dataclass(frozen=True)
class Masking(_Frozen):
    """In-model token masking (plan §3). Lives in the model, never in the loader (ADR-0003)."""

    #: §3 — the MAE optimum (Fig. 5), re-confirmed on RGB satellite imagery by Scale-MAE.
    #: R2 (#83) ablates 0.90, where the smooth-modality evidence points.
    ratio: float = 0.75
    #: §3 — uniform random, per-sample. Random beats block/grid decisively (MAE Table 1f).
    strategy: str = "random"

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio < 1.0:
            raise ValueError(f"mask ratio {self.ratio} must lie in (0, 1)")
        if self.strategy != "random":
            raise ValueError(f"unknown masking strategy {self.strategy!r} (§3 chose 'random')")


# --- §4 Loss ------------------------------------------------------------------


@dataclass(frozen=True)
class Loss(_Frozen):
    """The objective, its two DEM-specific guards, and its ocean weighting (plan §4)."""

    #: §4 — per-patch target normalization ON: on elevation it doubles as local detrending, so
    #: each 480 m patch is judged on its internal relief. R3 (#84) ablates it OFF.
    norm_pix_loss: bool = True
    #: §4 guard 1 — raised from the reference implementation's 1e-6. A near-flat 16x16 patch
    #: (playa, lake, residual ocean) has variance near zero in per-tile-standardized units, and
    #: the corpus's 3 m degenerate screen operates per *tile*, so flat patches survive: at 1e-6
    #: the division amplifies quantization noise up to ~1000x into the loss.
    norm_pix_var_floor: float = 1e-4
    #: §4 guard 2 — target statistics in fp32 whatever the autocast dtype is.
    norm_pix_stats_dtype: str = "fp32"
    #: §4 — masked patches only; computing the loss on all of them costs ~0.5% in MAE's
    #: ablation.
    masked_patches_only: bool = True
    #: §4 — ocean/invalid pixels contribute zero and the patch renormalizes by valid count
    #: (the loader's `land_mask`, ADR-0003 §4). Implemented by #96, and *enforced*: with this set,
    #: :meth:`pretrain.loss.MaeLoss.forward` refuses to score a batch that arrives without
    #: weights, rather than quietly training the unweighted objective under this flag.
    ocean_weighted: bool = True
    #: §4 — a masked patch below this valid fraction leaves the loss entirely, rather than being
    #: weighted down; :func:`pretrain.loss._active_patches` argues the threshold. 0 switches it
    #: off, and an all-ocean patch is excluded either way.
    min_valid_patch_frac: float = 0.25

    def __post_init__(self) -> None:
        if self.norm_pix_var_floor <= 0:
            raise ValueError(f"norm_pix_var_floor {self.norm_pix_var_floor} must be positive")
        if self.norm_pix_stats_dtype != "fp32":
            raise ValueError("§4 requires fp32 norm_pix statistics regardless of autocast")
        if not 0.0 <= self.min_valid_patch_frac <= 1.0:
            raise ValueError(
                f"min_valid_patch_frac {self.min_valid_patch_frac} must lie in [0, 1]"
            )


# --- §5 / §6 Data and augmentation --------------------------------------------


@dataclass(frozen=True)
class Data(_Frozen):
    """What is read and how it is presented (plan §5, and §6's augmentation)."""

    #: Where the corpus is mounted. Environment, not recipe — see the module docstring.
    corpus_dir: str = field(default="data/corpus_500k", metadata=ENVIRONMENT)
    #: The committed split artifacts (#77). Also environment: their *content* is fingerprinted.
    split_assignment_path: str = field(
        default=dataset_config.SPLIT_ASSIGNMENT_PATH, metadata=ENVIRONMENT
    )
    split_blocks_path: str = field(default=dataset_config.SPLIT_BLOCKS_PATH, metadata=ENVIRONMENT)
    #: §10.2's committed 40-tile val panel (#98). Environment for the same reason the two
    #: paths above are: *where* the artifact is mounted is a fact about the box, and its
    #: content is what a run is answerable for. **Its md5 is not yet fingerprinted** — the
    #: panel selects what a human looks at, not what the model is fit on, so a wrong one
    #: mislabels a diagnostic rather than invalidating a checkpoint. :func:`pretrain.panel.
    #: resolve_panel` re-checks it against the manifest at every load instead.
    val_panel_path: str = field(default=dataset_config.VAL_PANEL_PATH, metadata=ENVIRONMENT)
    #: §10.3's committed ~5k-tile kNN probe set (#99). Environment for the same reason, and
    #: unfingerprinted for a *weaker* version of the panel's: a wrong probe set does not
    #: invalidate a checkpoint either, but it does invalidate every cross-arm comparison drawn
    #: off it, which is what #85 adjudicates the release recipe on. So
    #: :func:`pretrain.probe.load_probe_tiles` re-checks it against the manifest at every load
    #: — every tile present, every committed stratum still agreeing, every tile still in `val`.
    probe_set_path: str = field(default=dataset_config.PROBE_SET_PATH, metadata=ENVIRONMENT)
    #: §5 — `per_tile`, the ADR-0003 default. `global` stays the scheduled Phase-3 ablation, and
    #: its failure mechanism is already measured (corpus card §4.8: ~42x more variance between
    #: tiles than within them). Do not re-litigate it in Phase 1.
    #:
    #: **The interaction §5 asks to be recorded in the config, recorded here:** with `per_tile`
    #: input normalization *and* :attr:`Loss.norm_pix_loss` targets, the model neither sees nor
    #: reconstructs absolute altitude anywhere. That is the intended Phase-1 design — terrain
    #: shape only — and the manifest's `mean`/`std` are what invert a reconstruction back to
    #: metres for the §10.2 grids.
    normalization: str = "per_tile"
    #: The split labels this run trains and validates on (plan §9, filled by #77).
    train_split: str = "train"
    val_split: str = "val"
    #: §6 — the full dihedral group (flips + 90 degree rotations). Overhead terrain has no
    #: canonical orientation, and the lineage keeps augmentation geometric-only; intensity
    #: jitter is both harmful (MAE Table 1e) and physically meaningless on standardized elevation.
    dihedral: bool = True
    #: §6 — **a deliberate deviation from the MAE default, recorded as one.** MAE's gain from
    #: RandomResizedCrop over fixed crops is small (84.9 vs 84.7) and rescaling changes the
    #: effective GSD of a modality where scale is physically meaningful; scale augmentation
    #: belongs with GSDPE in Phase 4.
    random_resized_crop: bool = False
    #: Loader plumbing (ADR-0003 §1, which says only "a few"). **Not a cited number** — a
    #: starting point for R0's measured samples/s gate (§7.5, the plan's #1 named risk), which
    #: is where it gets set for real. Nothing downstream should read it as settled.
    num_workers: int = 8

    def __post_init__(self) -> None:
        if self.normalization not in ("per_tile", "global"):
            raise ValueError(f"unknown normalization {self.normalization!r} (ADR-0003 §3)")
        if self.train_split == self.val_split:
            raise ValueError("train and val must read different splits")
        if self.num_workers < 0:
            raise ValueError(f"num_workers {self.num_workers} must be >= 0")


# --- §6 Optimization ----------------------------------------------------------


@dataclass(frozen=True)
class Optimization(_Frozen):
    """Optimizer, batch, and the linear LR scaling rule (plan §6)."""

    optimizer: str = "adamw"
    #: §6 — MAE's appendix values throughout.
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05  # excluded on bias/norm params — #93 owns the grouping
    fused: bool = True  # a free few-% step win
    #: §6 — MAE's default base LR; the actual LR is derived, never typed (:attr:`RunConfig.lr`).
    blr: float = 1.5e-4
    #: §6 — `lr = blr x eff_batch / lr_batch_base` (MAE PRETRAIN.md).
    lr_batch_base: int = 256
    #: §6 — physical batch 1024 fits comfortably in 32 GB at 49 visible tokens; accumulation 2
    #: is the official `--accum_iter` path and lands the effective batch at 2048 -> lr 1.2e-3.
    batch_size: int = 1024
    accum_iter: int = 2
    #: §6 — MAE clips nothing. Set 1.0 only if bf16 instability actually appears.
    grad_clip: float | None = None
    #: §6 — no dropout, no droppath, and **no EMA**: MAE uses none, so do not add one.
    dropout: float = 0.0
    drop_path: float = 0.0
    ema: bool = False

    def __post_init__(self) -> None:
        if self.optimizer != "adamw":
            raise ValueError(f"unknown optimizer {self.optimizer!r} (§6 chose AdamW)")
        if self.batch_size < 1 or self.accum_iter < 1:
            raise ValueError(
                f"batch_size {self.batch_size} and accum_iter {self.accum_iter} must be >= 1"
            )
        if self.blr <= 0 or self.lr_batch_base < 1:
            raise ValueError(f"blr {self.blr} must be positive over a base batch >= 1")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"grad_clip {self.grad_clip} must be positive or None")
        if self.ema:
            raise ValueError("§6 is explicit: MAE uses no weight EMA; do not add one")


# --- §6 Schedule --------------------------------------------------------------


@dataclass(frozen=True)
class Schedule(_Frozen):
    """Epochs, warmup, decay, and the milestone set (plan §6, §14)."""

    #: §6 — 400 epochs x 500k tiles ~= the sample budget of ~156 ImageNet epochs. Extend to 800
    #: only if the kNN probe and val loss are still improving at 400 (§10's decision rule).
    epochs: int = 400
    #: §6 — 5% of the schedule, matching MAE's 40/800; the small-scale reproduction used 20.
    warmup_epochs: int = 20
    decay: str = "cosine"  # §6 — cosine to 0
    #: §14 — kept so Phase 2 can plot downstream performance against pretraining compute.
    #: Milestones past the last epoch are **dropped**, not refused: §12's R2 and R3 read out at
    #: 200 epochs against the same milestone set, and making those runs pass a second override
    #: to say "and keep the checkpoints that actually happen" would be ceremony. A milestone
    #: below 1 is still refused — that is a typo, not a shorter run.
    milestones: tuple[int, ...] = (50, 100, 200, 400)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs {self.epochs} must be >= 1")
        if not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError(f"warmup_epochs {self.warmup_epochs} must lie in [0, {self.epochs}]")
        if self.decay != "cosine":
            raise ValueError(f"unknown decay {self.decay!r} (§6 chose cosine to 0)")
        below = [m for m in self.milestones if m < 1]
        if below:
            raise ValueError(f"milestone(s) {below} are not epochs (1..{self.epochs})")
        reachable = tuple(m for m in self.milestones if m <= self.epochs)
        if reachable != self.milestones:
            object.__setattr__(self, "milestones", reachable)


# --- §7 Runtime ---------------------------------------------------------------


@dataclass(frozen=True)
class Runtime(_Frozen):
    """Precision, kernels, and compilation (plan §7)."""

    #: §7.1 — bf16 autocast, Ada/Blackwell-native. It removes the fp16 NaN failure mode the
    #: official repo documents (issue 65), which is why fp16 is refused rather than offered.
    precision: str = "bf16"
    grad_scaler: bool = False  # §7.1 — bf16 needs none, and one would mask the loss scale
    attention: str = "sdpa"  # §7.2 — `F.scaled_dot_product_attention`, what timm blocks use
    #: §7.3 — expect ~1.1-1.4x. A fixed mask count keeps shapes static and avoids recompiles.
    compile: bool = True
    compile_dynamic: bool = False
    #: §7.6 — deliberately not used: checkpointing is unneeded at S/B scale and costs ~30%;
    #: channels_last is a CNN optimization and irrelevant here.
    grad_checkpointing: bool = False
    channels_last: bool = False
    #: §6 — fixed seed for model init and the DataLoader generator. Two runs from one config
    #: should differ only by nondeterministic kernels.
    seed: int = dataset_config.RANDOM_SEED

    def __post_init__(self) -> None:
        if self.precision not in ("bf16", "fp32"):
            raise ValueError(
                f"precision {self.precision!r}: §7.1 allows bf16 (or fp32 to debug), never fp16"
            )
        if self.grad_scaler:
            raise ValueError("§7.1: bf16 autocast runs without a GradScaler")
        if self.attention not in ("sdpa", "math"):
            raise ValueError(f"unknown attention {self.attention!r} (§7.2 chose sdpa)")
        if self.compile_dynamic and self.compile:
            raise ValueError("§7.3 compiles with dynamic=False; the mask count is fixed")


# --- §8 Initialization --------------------------------------------------------


@dataclass(frozen=True)
class Init(_Frozen):
    """Where the weights start (plan §8)."""

    #: §8 — the main and release runs start from scratch: "elevation-only, from scratch,
    #: reproducible" is part of the project's positioning. `imagenet_mae` is R4's continual
    #: arm (#86/#102), ViT-B only — no ViT-S MAE checkpoint exists at any provenance.
    #:
    #: **This is the flag §8 asks the two arms to differ by.** R4 and R5 run the same code path:
    #: `pretrain.init.initialize_from_config` is a no-op at `scratch` and the warm start at
    #: `imagenet_mae`, so nothing about the scratch runs moves to make the continual one
    #: possible. The checkpoint named is OpenMMLab's Apache-2.0 reproduction, not the official
    #: release — plan §8's 2026-07-29 amendment, evidence in
    #: [P101](../../docs/research/p101-mae-checkpoint-licence.md). The name is kept as
    #: `imagenet_mae` rather than renamed to its publisher: what the mode selects is "an
    #: ImageNet MAE encoder", and :attr:`checkpoint` is what says whose.
    mode: str = "scratch"
    #: Hashed, unlike the other paths in this config: *which* checkpoint an arm warm-starts
    #: from is the arm, so two continual runs from two checkpoints must not tag one W&B run.
    #: It is ``None`` for every scratch run, which is why hashing it costs the resume-on-a-new-
    #: box case nothing.
    checkpoint: str | None = None
    #: §8 — collapse the RGB patch embed to one channel by **summing** the kernels (timm's
    #: `adapt_input_conv` behaviour: sum preserves activation scale, mean shrinks it 3x).
    patch_embed_adapt: str = "sum"

    def __post_init__(self) -> None:
        if self.mode not in ("scratch", "imagenet_mae"):
            raise ValueError(f"unknown init mode {self.mode!r} (§8: scratch | imagenet_mae)")
        if self.patch_embed_adapt not in ("sum", "mean"):
            raise ValueError(
                f"unknown patch_embed_adapt {self.patch_embed_adapt!r} (§8 chose sum)"
            )


# --- §10 Monitoring -----------------------------------------------------------

#: §15.3's **committed** fixed val masking seed, and the reason it is a named constant rather
#: than a number in a default: the checklist item is "the fixed val mask seed is committed", so
#: it wants one value with one home that a reader can point at. Deliberately *not*
#: :attr:`Runtime.seed` — the val mask must not move when a run re-seeds its training stream to
#: get a second sample of the same recipe, or the two runs' val losses stop being comparable,
#: which is the entire property §10.1 asks for. The digits read as "DEAE §10.1".
#:
#: (#97's issue body cites this as checklist item **4**; the plan's item 4 is kill-and-resume
#: (#94) and item 3 is "Fixed val panel … and fixed val mask seed committed". The plan is the
#: source, so §15.3 is what this and every other reference here cites.)
VAL_MASK_SEED: Final = 0xDEAE_10_1


@dataclass(frozen=True)
class Monitoring(_Frozen):
    """What a run reports every epoch, what fires on it, and where it goes (plan §10, §11).

    Two kinds of field, split by the hash. The **measurement** — the val mask seed and every
    alarm threshold — is recipe: two runs that score val under different masks, or that alarm
    at different loss levels, are not the same experiment and must not share a
    :attr:`RunConfig.config_hash`. The **sink** — the four ``wandb_*`` fields — is environment,
    for the reasons :data:`ENVIRONMENT` gives.

    Every threshold below is a judgment call and none of them is cited: plan §10.1 names three
    *conditions* ("sudden collapse toward zero", "val rising while train falls", "a plateau at
    high loss in the first 20 epochs") and no numbers at all. The numbers are here, with the
    argument for each on the field, because a condition with no threshold is a sentence in a
    document rather than something that fires — which is the distinction this whole group
    exists to close.
    """

    #: §10.1, §15.3 — the seed the val pass masks under. See :data:`VAL_MASK_SEED`.
    #:
    #: There is deliberately **no** ``val_every_epochs`` beside it. §10.1 says "train loss + val
    #: loss **every epoch**" without qualification, and a cadence knob would take §4's
    #: per-patch distribution and the val-side collapse alarm off the intervening epochs with
    #: it — a gap in the diagnosis to save one inference pass over ~10k tiles.
    val_mask_seed: int = VAL_MASK_SEED

    # --- §10.2's reconstruction grids (#98)
    #: How many epochs between renders of the fixed val panel. **Plan §10.2's own number**,
    #: unlike every threshold below it: "reconstruction grids every 10 epochs". Recipe rather
    #: than environment, and the same argument the val mask seed carries — two runs whose
    #: grids are rendered at different epochs cannot have their panels compared side by side,
    #: which is the only way a grid is ever read.
    #:
    #: ``0`` switches the grids off, for a unit test or a run with no panel committed. It is
    #: not a knob to reach for otherwise: §15's checklist item 5 gates R1 on the grids
    #: rendering at all, and the drainage shortfall ADR-0008 accepted is expected to show
    #: here first or nowhere.
    recon_every_epochs: int = 10

    # --- §4's per-patch loss distribution
    #: The "top tail" of the per-patch loss distribution that §4's rule adjudicates: *if the
    #: top tail is dominated by near-flat patches, raise the variance floor before concluding
    #: anything about the model.* The top decile is a tail wide enough to be estimated from one
    #: val pass and narrow enough that "dominated by" means something.
    per_patch_tail_quantile: float = 0.90

    # --- §10.1's three alarms
    #: **Collapse toward zero.** Expressed as a fraction of the run's *first* recorded epoch
    #: loss rather than as an absolute number, because the loss scale is a recipe knob: with
    #: `norm_pix` on the target is unit-variance per patch and an untrained model sits near
    #: 1.0, while R3 (#84) ablates it off and scores raw metres-in-sigma. A run 50x below where
    #: it started is not learning terrain; it is the broken-`norm_pix`-target symptom
    #: (transformers issue 29416), where the target measures as identically flat and predicting
    #: zero is exactly right.
    collapse_frac: float = 0.02
    #: The *sudden* half of the same condition. A real MAE curve drops fast in warmup and then
    #: decays for hundreds of epochs; a 10x fall between two consecutive epochs is not a
    #: learning rate finding its feet, it is the target changing underneath the model.
    collapse_drop_factor: float = 10.0
    #: **Val rising while train falls.** How many **epochs** past the val minimum the loss has
    #: to have stayed above it. Ten epochs is 2.5% of the 400-epoch schedule — long enough that
    #: the noise in a ~10k-tile val pass cannot trip it, short enough to leave hundreds of
    #: epochs of GPU time to save. Measured in epochs rather than in val *points* so the number
    #: keeps meaning what this docstring says whatever cadence a run scores val on.
    overfit_patience: int = 10
    #: How far above its own best val loss has to have risen before that patience counts.
    #: Without it a run that has genuinely converged — val flat at its minimum, train still
    #: creeping down — alarms as overfitting, which is the one false positive that would teach
    #: people to ignore this alarm.
    overfit_min_rise: float = 0.02

    #: **A plateau at high loss in the first 20 epochs**, §10.1's LR/normalization-bug signal.
    #: The epoch it is adjudicated at; §6's warmup is 20 epochs, so this reads the curve exactly
    #: where the LR first reaches its peak and a model that is going to learn has started to.
    plateau_by_epoch: int = 20
    #: "High" — still within 10% of where the run started. A model whose LR is wrong or whose
    #: normalization is broken sits at the loss of predicting each patch's own mean and does not
    #: leave; one that is training has taken a visible bite out of that by epoch 20.
    plateau_frac: float = 0.90
    #: "Plateau" — the relative improvement over :attr:`plateau_window` that counts as movement.
    plateau_min_improvement: float = 0.01
    #: How many epochs back "still improving" is measured over. Five is short enough to read
    #: inside §6's 20-epoch warmup and long enough that one noisy epoch does not clear the bar.
    plateau_window: int = 5

    # --- §10.4's throughput dashboard
    #: §7.5 and §10.4's stop-and-fix threshold: fraction of wall-clock the step spends waiting
    #: for the loader. The loader is the plan's #1 named risk — DailyMAE got 5.8x end-to-end
    #: from IO alone — and 10% is the point past which the GPU dollars are buying idle time.
    loader_wait_alarm: float = 0.10

    # --- §10.3's kNN probe (#99)
    #: How often §10.3's frozen-encoder probe runs, in epochs. 20 is the plan's own cadence:
    #: the probe is an inference pass over ~5k tiles plus a kNN, minutes against an epoch's
    #: hours, and representation quality does not move fast enough to be worth watching more
    #: closely. **Epoch 0 is probed too** — that is the untrained baseline every later point is
    #: reported as a delta against, and the first *informative* point is still epoch 20. ``0``
    #: turns the probe off entirely, which is what a unit test and R0's smoke run want.
    probe_every_epochs: int = 20
    #: Neighbours each probe query votes over. See :data:`pretrain.probe.DEFAULT_K`.
    probe_k: int = 20
    #: How a tile's token sequence becomes one feature vector — see
    #: :data:`pretrain.probe.POOLINGS`. Recipe, not environment, and this is the field that
    #: makes it so: two arms pooled differently are not comparable, and the config hash is
    #: where that disagreement has to become visible.
    probe_pooling: str = "mean"

    # --- §11's sink (environment, not recipe)
    #: `offline` writes the whole run to a local directory with no account and no network, and
    #: is the default *and the test target*: it keeps this code off the critical path of #80's
    #: ops provisioning, and R0 (#81) is where a real project first receives a run. `disabled`
    #: drops the sink entirely, for a unit test or a laptop.
    wandb_mode: str = field(default="offline", metadata=ENVIRONMENT)
    #: The W&B project runs report into. #80's ops deliverable; the default is a placeholder
    #: that an offline run never contacts and R0 (#81) is where a real one is first used.
    wandb_project: str = field(default="demae-phase1", metadata=ENVIRONMENT)
    #: The W&B team/user the project lives under. Pinned rather than left at ``None``, which
    #: meant "the API key's current default entity" — an account setting editable from a web
    #: page, not a fact about this project. §10's decision rule reads R0-R5 against each other,
    #: and a default that shifts mid-phase files the later arms under a different entity with
    #: nothing in either run record saying so (#107). ``None`` still means the account default
    #: and is what a fork of this repo wants. Pinning is free because this is ``ENVIRONMENT``:
    #: it moves no :attr:`RunConfig.config_hash`, so R0 — which ran against this entity through
    #: a config file — and every run after it still hash as the same recipe.
    wandb_entity: str | None = field(
        default="jonathan-stoschek-purdue-university", metadata=ENVIRONMENT
    )
    #: Where offline runs are written. ``None`` means the process's working directory, i.e.
    #: `./wandb/`, which is W&B's own default.
    wandb_dir: str | None = field(default=None, metadata=ENVIRONMENT)

    def __post_init__(self) -> None:
        if self.wandb_mode not in WANDB_MODES:
            raise ValueError(f"unknown wandb_mode {self.wandb_mode!r} (one of {WANDB_MODES})")
        if self.recon_every_epochs < 0:
            raise ValueError(
                f"recon_every_epochs {self.recon_every_epochs} is an epoch count; 0 switches "
                f"§10.2's grids off and anything below that is a typo"
            )
        if not 0.0 < self.per_patch_tail_quantile < 1.0:
            raise ValueError(
                f"per_patch_tail_quantile {self.per_patch_tail_quantile} must lie in (0, 1)"
            )
        if not 0.0 < self.collapse_frac < 1.0:
            raise ValueError(f"collapse_frac {self.collapse_frac} must lie in (0, 1)")
        if self.collapse_drop_factor <= 1.0:
            raise ValueError(
                f"collapse_drop_factor {self.collapse_drop_factor} must exceed 1 — below it "
                "every epoch of a healthy run is a 'sudden collapse'"
            )
        if self.overfit_patience < 1:
            raise ValueError(f"overfit_patience {self.overfit_patience} must be >= 1")
        if self.overfit_min_rise < 0:
            raise ValueError(f"overfit_min_rise {self.overfit_min_rise} must be >= 0")
        if self.plateau_by_epoch < 1 or self.plateau_window < 1:
            raise ValueError(
                f"plateau_by_epoch {self.plateau_by_epoch} and plateau_window "
                f"{self.plateau_window} are epoch counts and must be >= 1"
            )
        if not 0.0 < self.plateau_frac <= 1.0:
            raise ValueError(f"plateau_frac {self.plateau_frac} must lie in (0, 1]")
        if self.plateau_min_improvement < 0:
            raise ValueError(
                f"plateau_min_improvement {self.plateau_min_improvement} must be >= 0"
            )
        if not 0.0 < self.loader_wait_alarm < 1.0:
            raise ValueError(f"loader_wait_alarm {self.loader_wait_alarm} must lie in (0, 1)")
        if self.probe_every_epochs < 0:
            raise ValueError(
                f"probe_every_epochs {self.probe_every_epochs} must be >= 0 (0 turns §10.3's "
                "probe off)"
            )
        if self.probe_k < 1:
            raise ValueError(f"probe_k {self.probe_k} must be >= 1")
        if self.probe_pooling not in PROBE_POOLINGS:
            raise ValueError(
                f"unknown probe_pooling {self.probe_pooling!r} (one of {PROBE_POOLINGS})"
            )


# --- The config ---------------------------------------------------------------

_GROUPS: Final = {
    "architecture": Architecture,
    "masking": Masking,
    "loss": Loss,
    "data": Data,
    "optimization": Optimization,
    "schedule": Schedule,
    "runtime": Runtime,
    "init": Init,
    "monitoring": Monitoring,
}


@dataclass(frozen=True)
class RunConfig(_Frozen):
    """One run's whole recipe. Defaults are R1: the ViT-S/16 scratch baseline (plan §12)."""

    #: What this run is called (R0-R5, or an ablation's own name). Environment, not recipe:
    #: two runs of the same recipe under different names share a :attr:`config_hash`.
    run_id: str = field(default="R1", metadata=ENVIRONMENT)
    architecture: Architecture = field(default_factory=Architecture)
    masking: Masking = field(default_factory=Masking)
    loss: Loss = field(default_factory=Loss)
    data: Data = field(default_factory=Data)
    optimization: Optimization = field(default_factory=Optimization)
    schedule: Schedule = field(default_factory=Schedule)
    runtime: Runtime = field(default_factory=Runtime)
    init: Init = field(default_factory=Init)
    monitoring: Monitoring = field(default_factory=Monitoring)

    def __post_init__(self) -> None:
        # The constraints no single group can see. §8's continual arm is ViT-B only, and it
        # needs a checkpoint to continue from.
        if self.init.mode == "imagenet_mae" and self.init.checkpoint is None:
            raise ValueError("init.mode 'imagenet_mae' needs init.checkpoint (§8)")

    # --- presets

    @classmethod
    def vit_base_continual(cls, checkpoint: str, **changes) -> RunConfig:
        """R4 — the ViT-B/16 continual arm, warm-started from an ImageNet MAE (§8, §12).

        :meth:`vit_base` with four knobs moved, and the fourth is the one nobody would think to
        move:

        1. ``init.mode = 'imagenet_mae'`` with ``checkpoint`` — the arm itself.
        2. **150 epochs**, §12's R4 row, against R5's 400.
        3. ``warmup_epochs = 8``, because §6's rule is *5% of the schedule* and inheriting R1's
           20 would make it 13% of a schedule a third the length. The rule is the cited thing;
           the 20 is what the rule produces at 400 epochs.
        4. ``pos_encoding = 'sincos2d_rowfirst'``, because mmselfsup's fixed sin-cos positional
           embedding is the **transpose** of MAE's and these weights learned mmselfsup's
           ([P102](../../docs/research/p102-mmselfsup-mae-layout.md) §4). Nothing about the
           weights would tell you: every name and every shape matches under either ordering.
           :func:`pretrain.init.load_mmselfsup_mae_encoder` refuses a model whose buffer
           disagrees with the checkpoint's own `pos_embed`, so this cannot be forgotten
           silently — but it is set here so the R4 config is right by construction, and so a
           §11 resume rebuilds the same buffer from the same config.

        Everything else is R5's, deliberately: the whole point of the arm is that the *recipe*
        is held fixed and only the initialization moves.
        """
        base = cls.vit_base().replace(
            run_id="R4",
            init=Init(mode="imagenet_mae", checkpoint=checkpoint),
            schedule=Schedule(epochs=150, warmup_epochs=8),
        )
        base = base.replace(
            architecture=base.architecture.replace(pos_encoding="sincos2d_rowfirst")
        )
        return base.replace(**changes) if changes else base

    @classmethod
    def vit_base(cls, **changes) -> RunConfig:
        """The ViT-B/16 release configuration (plan §2.1, §2.2, §6).

        Three deltas from ViT-S and nothing else: the 86M encoder, the 512-wide decoder, and a
        halved physical batch with doubled accumulation — which holds the effective batch at
        2048, so the LR rule lands on the same 1.2e-3 and the recipe stays comparable. Decoder
        heads at 16 are MAE's own decoder default; §2.2 fixes ViT-B's decoder depth and width
        but is silent on its head count, so the reference value fills the gap.
        """
        base = cls(
            run_id="R5",
            architecture=Architecture(
                encoder_dim=768, encoder_depth=12, encoder_heads=12,
                decoder_depth=4, decoder_dim=512, decoder_heads=16,
            ),
            optimization=Optimization(batch_size=512, accum_iter=4),
        )
        return base.replace(**changes) if changes else base

    # --- derived numbers (computed, never stored, so they cannot drift)

    @property
    def effective_batch(self) -> int:
        """§6 — physical batch x gradient accumulation. 1024 x 2 = 2048."""
        return self.optimization.batch_size * self.optimization.accum_iter

    @property
    def lr(self) -> float:
        """§6 — ``blr x eff_batch / 256``. 1.2e-3 at the default effective batch."""
        return self.optimization.blr * self.effective_batch / self.optimization.lr_batch_base

    @property
    def visible_tokens(self) -> int:
        """Tokens the encoder actually sees — 49 of 196 at 75%. The ~4x pretraining discount
        the whole budget rests on (§2.1), and the count `torch.compile` needs static (§3)."""
        return self.architecture.n_patches - self.masked_tokens

    @property
    def masked_tokens(self) -> int:
        """Tokens hidden per sample. Fixed by construction at a fixed ratio, which is what
        keeps shapes static for `torch.compile(dynamic=False)` (§7.3)."""
        return masked_token_count(self.architecture.n_patches, self.masking.ratio)

    # --- serialization

    def to_dict(self) -> dict:
        """The whole config as plain JSON types — nested one level, group by group."""
        out: dict[str, Any] = {"run_id": self.run_id}
        for name in _GROUPS:
            out[name] = _to_dict(getattr(self, name))
        return out

    @classmethod
    def from_dict(cls, d: dict) -> RunConfig:
        """Rebuild from :meth:`to_dict`'s output, or from any subset of it.

        A partial dict is the point: an ablation's config file names the knob it moves and
        inherits the plan for everything else. Unknown groups and unknown knobs both raise —
        silently defaulting a typo is how a run reports one recipe and trains another.
        """
        unknown = sorted(set(d) - set(_GROUPS) - {"run_id"})
        if unknown:
            raise ValueError(f"unknown config group(s) {unknown}; known: {sorted(_GROUPS)}")
        kwargs: dict[str, Any] = {}
        if "run_id" in d:
            kwargs["run_id"] = d["run_id"]
        for name, group_cls in _GROUPS.items():
            if name in d:
                kwargs[name] = _from_dict(group_cls, d[name], where=name)
        return cls(**kwargs)

    # --- the hash

    def recipe_dict(self) -> dict:
        """:meth:`to_dict` minus every :data:`ENVIRONMENT` field — the view the hash covers.

        Driven by the field metadata at both levels, so marking a field environment is the
        *only* way to leave it out of the hash. A second mechanism (a name list here, say)
        would be a second place to forget.
        """
        full = self.to_dict()
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.metadata.get(_ENVIRONMENT_KEY):
                continue
            group = getattr(self, f.name)
            out[f.name] = {
                g.name: full[f.name][g.name]
                for g in fields(group)
                if not g.metadata.get(_ENVIRONMENT_KEY)
            }
        return out

    @property
    def config_hash(self) -> str:
        """A stable fingerprint of the recipe — the W&B tag plan §15.7 asks for.

        Canonical JSON (sorted keys, no whitespace slack) through BLAKE2b, the same hash family
        every tie-break in :mod:`dataset.sampling` uses. Sorted keys make it insensitive to the
        order a dict was built in; JSON — rather than :func:`repr` or :func:`hash` — makes it
        identical in every process and on every box.
        """
        canonical = json.dumps(self.recipe_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(
            canonical.encode("utf-8"), digest_size=CONFIG_HASH_CHARS // 2
        ).hexdigest()
