"""The frozen benchmark run object — every §4 knob, serializable, hashable (#147).

`pretrain.config.RunConfig`'s sibling, and deliberately its sibling rather than its subclass:
one object holds the whole of [the Task 1 plan](../../docs/research/task1-void-fill-plan.md)
§3 and §4 at its cited default, and **an arm is this object with a knob moved**. Structure
follows the plan's sections rather than the code's packages, so a reviewer can read this file
beside the plan and check it line by line.

## The three properties the rest of the benchmark leans on

- **Frozen.** A cell's config cannot drift mid-run: what the results parquet records is what
  the row was computed under. Changing anything means constructing a new object
  (:meth:`replace`).
- **Round-trips.** ``to_dict`` / ``from_dict`` over plain JSON types, and a **partial** dict
  fills the rest from the plan — which is exactly what an arm's config file should look like:
  the one knob it moves, nothing else. An unknown group and an unknown knob both raise, because
  a typo that silently keeps the default is a run that reports the recipe it was *asked* for
  and scores the one it defaulted to.
- **Hashes to a stable :attr:`BenchmarkConfig.config_hash`.** A keyed hash over canonical JSON,
  not :func:`hash` (which ``PYTHONHASHSEED`` randomizes per process), so two boxes agree. It
  covers the **recipe** and deliberately not :class:`Paths` — where the corpus, the artifacts,
  the bank and the encoder checkpoints are mounted. §11's argument carried into Phase 2: a
  rented box mounts them somewhere else, and a hash that moved with the mount point would read
  every relaunch as a config change.

## What is recipe and what is environment, in one sentence each

``arm``, ``size`` and ``seed`` are **recipe**: they are the cell, they are what the results
parquet is keyed on, and two of them differing is two different computations. The whole of
:class:`Paths` is **environment**: *which* encoder an arm reads is the arm — already hashed —
and *where* that checkpoint sits is a fact about the box, whose md5 the resolution line prints
and the run record carries (:mod:`benchmark.artifacts`).

Numbers registered by [ADR-0013](../../docs/adr/0013-task1-pre-registration-register.md) are
**not** here. The +0.15 degree seam threshold, the four bootstrap pairs and the 100-instance
cell floor are read from ``task1-prereg-bar.json``, which refuses to load when its §6 phrase no
longer appears in the plan. A second home for a registered number is how a threshold moves
without anything failing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Final

from dataset import config as dataset_config
from dataset.centroids import K as STRATUM_COUNT

from . import arms

#: Marks a field as *where the run happens* rather than *what it scores*: recorded in the
#: config, excluded from :attr:`BenchmarkConfig.config_hash`. Here it marks one whole group —
#: :class:`Paths` — rather than a scatter of fields, which is the shape this benchmark's
#: environment actually has.
_ENVIRONMENT_KEY: Final = "environment"
ENVIRONMENT: Final = {_ENVIRONMENT_KEY: True}

#: Length of the config hash, in hex characters. 16 hex = 64 bits, matching
#: :data:`pretrain.config.CONFIG_HASH_CHARS` so the two phases' hashes read alike.
CONFIG_HASH_CHARS: Final = 16

#: §3.1's nested label-efficiency subsets, smallest first. Nested rather than independent so
#: C3's sweep measures data volume rather than draw luck.
LABEL_EFFICIENCY_SIZES: Final = ("1k", "5k", "20k")

#: What each size is, in tiles.
SIZE_TILES: Final = {"1k": 1_000, "5k": 5_000, "20k": 20_000}


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

    An unknown key raises rather than being ignored, for the reason the module docstring gives.
    """
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError(f"unknown {where} knob(s) {unknown}; known: {sorted(known)}")
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def _bucket(name: str, span: tuple[int, int]) -> tuple[int, int]:
    """A size bucket, checked: a low-high pair of equivalent diameters in pixels."""
    if len(span) != 2 or span[0] >= span[1] or span[0] < 1:
        raise ValueError(f"{name} bucket {span} is not an ascending (low, high) in pixels")
    return span


# --- §3.1 Tiles and splits ----------------------------------------------------


@dataclass(frozen=True)
class Tiles(_Frozen):
    """The four committed tile draws and the geocells they come from (plan §3.1)."""

    #: §3.1 — 20,000 tiles from *train* geocells, stratified proportionally over the corpus's
    #: k=20 strata under a recorded seed, committed as a `tile_id` list in the
    #: :mod:`pretrain.artifact` format. The 1k and 5k sweeps are nested inside this draw.
    head_train: int = 20_000
    #: §3.1 — early stopping and the head LR pick, from *val* geocells.
    val: int = 2_000
    #: §3.1 — from *test* geocells only, never touched until the final pass. Split containment
    #: is asserted at construction exactly as a Phase-1 Dataset asserts it, so a benchmark
    #: cannot score on ground the encoder pretrained on.
    test: int = 5_000
    #: The corpus's own stratification (`dataset.centroids.K`), quoted rather than retyped.
    strata: int = STRATUM_COUNT
    #: The seed the draws are recorded under. The draw is a *file*, not a reproduction of an
    #: RNG (§3.1) — this is what the file records having been drawn with.
    draw_seed: int = dataset_config.RANDOM_SEED
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    def __post_init__(self) -> None:
        for name in ("head_train", "val", "test"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} {getattr(self, name)} must be >= 1")
        if self.strata < 1:
            raise ValueError(f"strata {self.strata} must be >= 1")
        splits = (self.train_split, self.val_split, self.test_split)
        if len(set(splits)) != 3:
            raise ValueError(f"the three splits must differ; got {splits}")


# --- §3.2 Synthetic void masks ------------------------------------------------


@dataclass(frozen=True)
class Masks(_Frozen):
    """The mask generator's geometry, and the bank it writes (plan §3.2, §3.4).

    Both mask artifacts carry this group, so the determinism contract is one contract rather
    than two: the eval mask list every arm reads and the train mask bank every arm, seed and
    hyperparameter config reads are generated from the same knobs and hash together.
    """

    #: §3.2 — random-walk blobs dilated to target area (the radar-shadow proxy) and axis-aligned
    #: rectangles (the sensor-dropout proxy). The real family is §3.3's transplant and carries
    #: no bucket axis, which is why it is named separately.
    synthetic_families: tuple[str, ...] = ("blob", "rectangle")
    real_family: str = "real"
    #: §3.2's size buckets, as **equivalent diameter** `d = 2·√(area/π)`. The unit is
    #: load-bearing: both families are area-matched through `d`, so a blob and a rectangle in
    #: one bucket cover the same pixel count and the bucket axis means one thing.
    small_px: tuple[int, int] = (16, 48)
    medium_px: tuple[int, int] = (64, 96)
    #: The top is **158**, not the published 160 (#121). Read as a side length the large bucket
    #: would demand 51% of the tile and collide with the 40% cap at 141 px; read correctly as a
    #: diameter the cap permits 159.86 px, so 160 overshot by 0.14. Lowered rather than clamped,
    #: so no draw is ever silently trimmed — see :meth:`max_equivalent_diameter_px`.
    large_px: tuple[int, int] = (128, 158)
    #: §3.2 — the *real* family's bands stay at 128–160 even though the synthetic large bucket
    #: now tops at 158. Those bands describe sizes real voids **have**, measured on SRTM, and
    #: are not draw targets the area cap can constrain; renaming them would misdescribe the
    #: published histogram.
    real_band_top_px: int = 160
    #: §3.2's constraints, checked on every draw rather than assumed: fully interior at ≥ 8 px
    #: from the tile edge, and never more than 40% of the tile.
    edge_margin_px: int = 8
    max_area_frac: float = 0.40
    #: §3.2/§4.1 — K = 50 draws per head-train tile, written once and read by every arm, seed
    #: and config. 50 committed masks per tile is what 50 on-the-fly epochs would have produced,
    #: so per-epoch variation is preserved *exactly* without head training becoming loader-bound.
    bank_draws_per_tile: int = 50
    #: §3.2 — one void per tile per eval instance.
    voids_per_instance: int = 1
    seed: int = dataset_config.RANDOM_SEED

    def __post_init__(self) -> None:
        for name in ("small_px", "medium_px", "large_px"):
            _bucket(name, getattr(self, name))
        if not (self.small_px[1] < self.medium_px[0] and self.medium_px[1] < self.large_px[0]):
            raise ValueError(
                f"the size buckets must be disjoint and ascending; got {self.buckets}"
            )
        if not 0.0 < self.max_area_frac <= 1.0:
            raise ValueError(f"max_area_frac {self.max_area_frac} must lie in (0, 1]")
        if self.edge_margin_px < 0:
            raise ValueError(f"edge_margin_px {self.edge_margin_px} must be >= 0")
        # #121's measurement, as a construction check rather than a comment: a bucket whose top
        # exceeds what the cap permits is one the generator would have to trim, which is the
        # single thing §3.2 says must never happen silently.
        if self.large_px[1] > self.max_equivalent_diameter_px:
            raise ValueError(
                f"large bucket top {self.large_px[1]} px exceeds the "
                f"{self.max_area_frac:.0%} area cap's {self.max_equivalent_diameter_px:.2f} px "
                f"equivalent diameter — a draw there would have to be silently trimmed (§3.2)"
            )
        if self.bank_draws_per_tile < 1 or self.voids_per_instance < 1:
            raise ValueError("bank_draws_per_tile and voids_per_instance must be >= 1")

    @property
    def max_equivalent_diameter_px(self) -> float:
        """The largest `d` the area cap permits on a tile — 159.86 px at 40% of 224² (#121)."""
        area = self.max_area_frac * dataset_config.TILE_PX**2
        return 2.0 * math.sqrt(area / math.pi)

    @property
    def buckets(self) -> tuple[tuple[str, tuple[int, int]], ...]:
        """The three size buckets, named, in ascending order."""
        return (("small", self.small_px), ("medium", self.medium_px), ("large", self.large_px))


# --- §4.1 The shared head -----------------------------------------------------


@dataclass(frozen=True)
class Head(_Frozen):
    """The one head A4, A5 and A6 share (plan §4.1).

    Byte-identical across those arms, differing only in the input channel width the first
    projection absorbs — which is what makes C1 about the representation and nothing else.
    """

    #: §4.1 — LayerNorm on patch tokens, 1x1 conv to 256, then four stages of
    #: [3x3 conv, GroupNorm, GELU, 2x bilinear upsample], the full-resolution void mask
    #: concatenated at each stage's input, and a 1-channel residual output added to the A1 fill.
    proj_dim: int = 256
    stage_dims: tuple[int, ...] = (256, 128, 64, 32)
    #: §4.1 — L1 over L2, robust to the heavy-tailed elevation-error distribution, plus 0.1x an
    #: L1 slope-consistency term on Sobel gradients inside the void to discourage flat-patch
    #: cheating.
    loss: str = "l1"
    slope_weight: float = 0.1
    #: §4.1's optimizer block. **Tuned once on A6 — the neutral encoder — using val only, then
    #: frozen for every arm**; tuning on the arm you hope wins is the classic thumb on the scale.
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.05
    decay: str = "cosine"
    warmup_frac: float = 0.05
    precision: str = "bf16"
    batch_size: int = 128
    epochs: int = 50
    early_stop_metric: str = "val_rmse"
    #: §4.1's seed counts: three per primary cell at 20k, two on the label-efficiency subsets,
    #: and **one** for §2.3's ViT-S appendix cells — an exception rather than a default, and
    #: reported as a single observation without a ±.
    seeds: int = 3
    label_sweep_seeds: int = 2
    appendix_seeds: int = 1

    def __post_init__(self) -> None:
        if self.optimizer != "adamw":
            raise ValueError(f"unknown optimizer {self.optimizer!r} (§4.1 chose AdamW)")
        if self.loss != "l1":
            raise ValueError(f"unknown loss {self.loss!r} (§4.1 chose masked L1 in metres)")
        if self.decay != "cosine":
            raise ValueError(f"unknown decay {self.decay!r} (§4.1 chose cosine to 0)")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError(f"precision {self.precision!r}: §4.1 trains in bf16")
        if self.lr <= 0 or self.weight_decay < 0 or self.slope_weight < 0:
            raise ValueError("lr must be positive; weight_decay and slope_weight >= 0")
        if not 0.0 <= self.warmup_frac < 1.0:
            raise ValueError(f"warmup_frac {self.warmup_frac} must lie in [0, 1)")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be >= 1")
        if not self.stage_dims or self.proj_dim < 1:
            raise ValueError("the head needs a projection width and at least one stage")
        for name in ("seeds", "label_sweep_seeds", "appendix_seeds"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} {getattr(self, name)} must be >= 1")


# --- §4.2 The DEMAE native fine-tune ------------------------------------------


@dataclass(frozen=True)
class Finetune(_Frozen):
    """A3's short adaptation — not a new pretraining (plan §4.2)."""

    #: §4.2 — 20 epochs over the 20k head-train set, reading **the bank's first 20**, so the
    #: void geometry A3 adapts to and the geometry the frozen arms train against are the same
    #: draws rather than merely the same distribution.
    epochs: int = 20
    encoder_lr: float = 1e-4
    layer_decay: float = 0.75
    decoder_lr: float = 1.5e-3
    decay: str = "cosine"
    precision: str = "bf16"
    #: §4.2 — the void-shaped mask plus 25% random patches, so the encoder keeps seeing global
    #: context and the effective mask ratio stays near pretraining's.
    random_patch_frac: float = 0.25
    #: §4.2 — **off**, inverting Phase 1's own default: void filling needs absolute per-patch
    #: scale and norm-pix targets discard it. Eval always re-inverts to metres with non-void
    #: statistics regardless.
    norm_pix_loss: bool = False
    #: §4.2's inference composite: a 16 px cross-fade at patch boundaries inside the void, and
    #: four shifted encodings averaged for near-tile-scale voids, so the patch grid does not
    #: print as a seam the seam-gap column then charges it for.
    crossfade_px: int = 16
    shift_offsets: tuple[int, ...] = (0, 112)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs {self.epochs} must be >= 1")
        if self.encoder_lr <= 0 or self.decoder_lr <= 0:
            raise ValueError("both learning rates must be positive")
        if not 0.0 < self.layer_decay <= 1.0:
            raise ValueError(f"layer_decay {self.layer_decay} must lie in (0, 1]")
        if not 0.0 <= self.random_patch_frac < 1.0:
            raise ValueError(f"random_patch_frac {self.random_patch_frac} must lie in [0, 1)")
        if self.crossfade_px < 0:
            raise ValueError(f"crossfade_px {self.crossfade_px} must be >= 0")
        if any(offset < 0 or offset >= dataset_config.TILE_PX for offset in self.shift_offsets):
            raise ValueError(f"shift_offsets {self.shift_offsets} must lie in [0, tile width)")


# --- §4.4 The from-scratch baseline -------------------------------------------


@dataclass(frozen=True)
class Unet(_Frozen):
    """A7 — the PANGAEA bar, and a hard one (plan §4.4).

    ~17M trainable parameters against every frozen arm's 1.18M, which is why §6 item 4 prints
    total *and* trainable counts: a totals-only column reads A7 as handicapped when it is the
    opposite.
    """

    #: §4.4 — a standard 4-down / 4-up UNet.
    depth: int = 4
    #: §4.4 — the A1-filled tile and the void mask.
    in_channels: int = 2
    #: §4.4/§2.4 — residual over the A1 fill, like every learned arm.
    residual: bool = True

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError(f"depth {self.depth} must be >= 1")
        if self.in_channels != 2:
            raise ValueError("§4.4's input is the A1-filled tile plus the void mask")
        if not self.residual:
            raise ValueError(
                "§2.4: every learned arm predicts a residual over the A1 fill, so a head that "
                "learns nothing scores exactly A1 and the table stays legible"
            )


# --- ADR-0012's extractor contract --------------------------------------------


@dataclass(frozen=True)
class Extract(_Frozen):
    """The frozen-feature contract's configurable half (ADR-0012, plan §9.4).

    The rest of the contract — encoder identity, checkpoint md5, TerraMind's named constants,
    ADR-0011's `pos_emb` derivation — is `extractor_spec.json`'s, which re-checks itself at load
    and refuses rather than warns. What is here is what a *run* may legitimately choose.
    """

    #: ADR-0012 §6 — fp32 is the reference tensor, always.
    reference_dtype: str = "fp32"
    #: ADR-0012 §6 — the compute dtype is a **declared field** and may differ per task, because
    #: the measured difference is cosine 0.999968 against #134's 0.99 materiality bar. A task
    #: moving it re-checks it under that threshold; the TerraMind arm's figure does not exist
    #: yet and is measured on first run.
    compute_dtype: str = "bf16"
    #: ADR-0012 §3 — all twelve block outputs, plus a separately named `output` field, so
    #: DEMAE's post-`LayerNorm` tensor and its `intermediates[11]` are never confused.
    layers: int = 12
    #: ADR-0012 §5 — canonical `(N, 196, D)` patch tokens, row-major.
    tokens: int = dataset_config.TOKENS_PER_TILE
    emit_intermediates: bool = True
    #: ADR-0012 §1 — `valid = land ∧ ¬void`. Task 1 passes the void-aware mask (§2.1's non-void
    #: statistics, which is what stops the leak) and Task 2 passes land-only: one code path.
    valid_mask: str = "land_and_not_void"

    def __post_init__(self) -> None:
        if self.reference_dtype != "fp32":
            raise ValueError("ADR-0012 §6: fp32 is the reference tensor")
        if self.compute_dtype not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"unknown compute_dtype {self.compute_dtype!r}")
        if self.layers != 12 or self.tokens != dataset_config.TOKENS_PER_TILE:
            raise ValueError(
                f"ADR-0012 §3/§5: twelve blocks and {dataset_config.TOKENS_PER_TILE} tokens"
            )
        if not self.emit_intermediates:
            raise ValueError(
                "ADR-0012 §3: the extractor emits intermediates even though no Task 1 arm "
                "reads them — a last-layer-only extractor is the option under which the map's "
                "one cross-task constraint fails and Task 2 writes a second one"
            )
        if self.valid_mask != "land_and_not_void":
            raise ValueError("ADR-0012 §1: valid = land ∧ ¬void, one code path for both tasks")


# --- Where it all is, on this box ---------------------------------------------


@dataclass(frozen=True)
class Paths(_Frozen):
    """Every path this benchmark reads. Environment throughout — see the module docstring.

    Nothing here moves :attr:`BenchmarkConfig.config_hash`. What each file *is* — its md5 — is
    resolved and printed before a run takes a step, and recorded in the run record beside the
    table it produced (:mod:`benchmark.artifacts`).
    """

    corpus_dir: str = "data/corpus_500k"
    split_assignment_path: str = dataset_config.SPLIT_ASSIGNMENT_PATH
    #: Where the committed inputs live: the four tile lists, the eval mask list, the extractor
    #: spec, the pre-registration bar and the dataset card.
    artifacts_dir: str = "artifacts/task1"
    #: §3.4 — 1,000,000 RLE rows at ~740 MB, which **cannot** be committed, on the same rule the
    #: corpus's shards follow. `freeze` generates it, its md5 and generator config hash are
    #: committed in a sidecar, and `run` refuses a bank whose md5 has moved. The determinism
    #: contract is preserved — the mask sequence a run sees is still a committed input, its
    #: *hash* rather than its bytes.
    mask_bank_path: str = "data/task1/train_mask_bank.parquet"
    #: One row per (arm, init, size, seed, family, bucket, band, tile_id, instance). `report` is
    #: a pure function of this file, so re-rendering the table never re-runs an arm.
    results_path: str = "data/task1/results.parquet"
    #: §3.3's fetched SRTMGL1N granules, for the real-void family's shapes.
    srtm_dir: str = "data/srtm_voids"
    #: The Phase-1 release encoders. R5 is §4.2's release checkpoint and R4 its pre-registered
    #: secondary; ViT-S is §2.3's appendix backbone.
    demae_r5_checkpoint: str | None = ".release-work/release-R5-v2/encoder.pt"
    demae_r4_checkpoint: str | None = ".release-work/release-R4-v2/encoder.pt"
    demae_vits_checkpoint: str | None = None
    #: TerraMind's DEM tokenizer and the ImageNet MAE. ``None`` until a box mounts them —
    #: an unmounted checkpoint is *named* at resolution time rather than silently absent, which
    #: is the whole reason these are fields rather than lookups inside a loader.
    terramind_checkpoint: str | None = None
    imagenet_mae_checkpoint: str | None = None
    imagenet_mae_vits_checkpoint: str | None = None


# --- The config ---------------------------------------------------------------

_GROUPS: Final = {
    "tiles": Tiles,
    "masks": Masks,
    "head": Head,
    "finetune": Finetune,
    "unet": Unet,
    "extract": Extract,
    "paths": Paths,
}

#: The top-level scalars. Recipe, all three: they are the cell.
_SCALARS: Final = ("arm", "size", "seed")

#: The groups the committed inputs are a function of — what `freeze` writes depends on these
#: and on nothing else. See :attr:`BenchmarkConfig.artifact_hash`.
_ARTIFACT_GROUPS: Final = ("tiles", "masks")


def _hash(payload: dict) -> str:
    """Canonical JSON through BLAKE2b — identical in every process and on every box.

    Sorted keys make it insensitive to the order a dict was built in; JSON, rather than
    :func:`repr` or :func:`hash`, makes two boxes agree (``PYTHONHASHSEED`` randomizes the
    builtin per process).
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(
        canonical.encode("utf-8"), digest_size=CONFIG_HASH_CHARS // 2
    ).hexdigest()


@dataclass(frozen=True)
class BenchmarkConfig(_Frozen):
    """One cell's whole recipe. Defaults are A4-R5 at 20k, seed 0 — C1's headline cell."""

    #: The registered arm (:mod:`benchmark.arms`). *An arm is this object with this knob moved.*
    arm: str = "A4-R5"
    #: §3.1's label-efficiency size, one of :data:`LABEL_EFFICIENCY_SIZES`.
    size: str = "20k"
    #: §4.1's seed. Recipe rather than environment, following `pretrain.config.Runtime.seed`:
    #: the head init and the bank epoch order both consume it, so two seeds are two different
    #: computations and a results row must say which one it is.
    seed: int = 0
    tiles: Tiles = field(default_factory=Tiles)
    masks: Masks = field(default_factory=Masks)
    head: Head = field(default_factory=Head)
    finetune: Finetune = field(default_factory=Finetune)
    unet: Unet = field(default_factory=Unet)
    extract: Extract = field(default_factory=Extract)
    paths: Paths = field(default_factory=Paths, metadata=ENVIRONMENT)

    def __post_init__(self) -> None:
        arms.get(self.arm)  # refuses, naming the registry
        if self.size not in LABEL_EFFICIENCY_SIZES:
            raise ValueError(
                f"unknown size {self.size!r} (§3.1's nested subsets: "
                f"{list(LABEL_EFFICIENCY_SIZES)})"
            )
        if self.seed < 0:
            raise ValueError(f"seed {self.seed} must be >= 0")
        # The constraint no single group can see: the sweep is nested *inside* one committed
        # draw (§3.1), so a subset bigger than the draw would ask for tiles nobody drew.
        largest = SIZE_TILES[LABEL_EFFICIENCY_SIZES[-1]]
        if self.tiles.head_train < largest:
            raise ValueError(
                f"tiles.head_train {self.tiles.head_train} is smaller than the largest nested "
                f"subset ({largest}); §3.1's sweep is nested inside the one committed draw"
            )

    # --- derived numbers (computed, never stored, so they cannot drift)

    @property
    def train_tiles(self) -> int:
        """How many head-train tiles this cell trains on — the nested subset :attr:`size` names."""
        return SIZE_TILES[self.size]

    @property
    def bank_rows(self) -> int:
        """§3.4's bank size: K draws per head-train tile. 1,000,000 at K = 50 over 20k."""
        return self.masks.bank_draws_per_tile * self.tiles.head_train

    @property
    def synthetic_eval_instances(self) -> int:
        """§3.2 — every eval tile appears once per bucket, for the two synthetic families.

        30,000 at the committed draw. The real family has no bucket axis and is drawn at its
        natural size distribution (§3.2), so its count is a property of the artifact rather
        than a number this config can derive.
        """
        return self.tiles.test * len(self.masks.synthetic_families) * len(self.masks.buckets)

    # --- serialization

    def to_dict(self) -> dict:
        """The whole config as plain JSON types — three scalars, then group by group."""
        out: dict[str, Any] = {name: getattr(self, name) for name in _SCALARS}
        for name in _GROUPS:
            out[name] = _to_dict(getattr(self, name))
        return out

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkConfig:
        """Rebuild from :meth:`to_dict`'s output, or from any subset of it."""
        unknown = sorted(set(d) - set(_GROUPS) - set(_SCALARS))
        if unknown:
            raise ValueError(
                f"unknown config group(s) {unknown}; known: {sorted({*_GROUPS, *_SCALARS})}"
            )
        kwargs: dict[str, Any] = {name: d[name] for name in _SCALARS if name in d}
        for name, group_cls in _GROUPS.items():
            if name in d:
                kwargs[name] = _from_dict(group_cls, d[name], where=name)
        return cls(**kwargs)

    # --- the hash

    def recipe_dict(self) -> dict:
        """:meth:`to_dict` minus every :data:`ENVIRONMENT` field — the view the hash covers.

        Driven by the field metadata rather than by a name list here, so marking a field
        environment is the *only* way to leave it out of the hash. A second mechanism would be
        a second place to forget.
        """
        full = self.to_dict()
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.metadata.get(_ENVIRONMENT_KEY):
                continue
            value = getattr(self, f.name)
            if is_dataclass(value):
                out[f.name] = {
                    g.name: full[f.name][g.name]
                    for g in fields(value)
                    if not g.metadata.get(_ENVIRONMENT_KEY)
                }
            else:
                out[f.name] = full[f.name]
        return out

    @property
    def config_hash(self) -> str:
        """A stable fingerprint of the recipe, printed before a run takes a step.

        Canonical JSON (sorted keys, no whitespace slack) through BLAKE2b — the same
        construction :attr:`pretrain.config.RunConfig.config_hash` uses, so the two phases'
        hashes are read the same way.
        """
        return _hash(self.recipe_dict())

    @property
    def artifact_hash(self) -> str:
        """§3.4's **generator** config hash — what the committed inputs are a function of.

        Narrower than :attr:`config_hash` on purpose. The eval mask list and the train mask
        bank "carry the same generator and the same config hash, so the determinism contract is
        one contract rather than two" (§3.4) — and that hash must **not** move with the arm, the
        size or the seed, because those are what a *cell* varies over inputs every cell shares.
        A `freeze` whose printed hash moved with ``--arm`` would be claiming the artifacts had.
        """
        recipe = self.recipe_dict()
        return _hash({name: recipe[name] for name in _ARTIFACT_GROUPS})


# --- The override surface -----------------------------------------------------


def parse_override(text: str) -> dict:
    """``group.knob=value`` -> the one-key overlay dict it means.

    `pretrain.header.parse_override`'s twin, and the same escape hatch: an arm is
    ``--set arm=A5`` rather than a file. What comes back is shaped like a partial config, so it
    goes through :meth:`BenchmarkConfig.from_dict` and inherits that method's refusal of
    unknown groups and unknown knobs for free.

    **The value is read as JSON, falling back to the string itself** — which is what makes
    ``finetune.norm_pix_loss=false`` the boolean rather than the (truthy) string, keeps
    ``head.epochs=50`` an int, and still leaves ``paths.corpus_dir=data/corpus_500k`` a path.
    """
    key, sep, raw = text.partition("=")
    if not sep:
        raise ValueError(f"override {text!r} is not KNOB=VALUE (e.g. head.epochs=10)")
    path = key.strip().split(".")
    if len(path) > 2:
        raise ValueError(
            f"override {text!r}: a config knob is `group.knob` or a top-level "
            f"{list(_SCALARS)}, and {key.strip()!r} is deeper than that"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw  # a bare string: a path, an arm name, a size
    return {path[0]: value} if len(path) == 1 else {path[0]: {path[1]: value}}


def _overlay(base: dict, overlay: dict) -> dict:
    """A full config dict with ``overlay``'s groups merged in, one level deep.

    An unknown group passes straight through rather than raising here, so
    :meth:`BenchmarkConfig.from_dict` is the single place that names what a config may contain
    and the single message a typo produces.
    """
    merged = dict(base)
    for key, value in overlay.items():
        merged[key] = {**base.get(key, {}), **value} if isinstance(value, dict) else value
    return merged


def build_config(
    *,
    config_path: Path | str | None = None,
    arm: str | None = None,
    size: str | None = None,
    seed: int | None = None,
    corpus: str | None = None,
    artifacts_dir: str | None = None,
    overrides: Sequence[str] | None = None,
) -> BenchmarkConfig:
    """Assemble the run config: the plan's defaults, a partial config file, the cell, ``--set``.

    Order is the point, and it is `pretrain.header.build_config`'s order for the same reasons:

    1. **The plan's defaults** — every §3/§4 knob at its cited value.
    2. **The config file**, naming only the knobs its run moves.
    3. **The cell flags** (``--arm``, ``--size``, ``--seed``) and the two path flags.
    4. **``--set`` overrides**, last, so the command line can always pin a knob whatever the
       file said. It is the general escape hatch, so it wins over the special-case flags: a
       ``--set arm=A6`` beats an ``--arm A7``.
    """
    cfg = BenchmarkConfig()
    if config_path is not None:
        overlay = json.loads(Path(config_path).read_text())
        cfg = BenchmarkConfig.from_dict(_overlay(cfg.to_dict(), overlay))
    cell = {name: value for name, value in (("arm", arm), ("size", size), ("seed", seed))
            if value is not None}
    if cell:
        cfg = cfg.replace(**cell)
    paths = cfg.paths
    if corpus is not None:
        paths = paths.replace(corpus_dir=corpus)
    if artifacts_dir is not None:
        paths = paths.replace(artifacts_dir=artifacts_dir)
    cfg = cfg.replace(paths=paths)
    for override in overrides or ():
        cfg = BenchmarkConfig.from_dict(_overlay(cfg.to_dict(), parse_override(override)))
    return cfg


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """:func:`build_config` off the namespace :func:`add_config_arguments` filled. Use this.

    The two halves of one flag set, written next to each other so they cannot come apart —
    ADR-0009 §4's rule, and the bug it was written for was a *call* that lost one argument of
    six while its parser stayed correct. So there is one call, here, and every entrypoint
    delegates to it rather than retyping the keyword list.
    """
    return build_config(
        config_path=args.config,
        arm=getattr(args, "arm", None),
        size=getattr(args, "size", None),
        seed=getattr(args, "seed", None),
        corpus=args.corpus,
        artifacts_dir=args.artifacts_dir,
        overrides=args.overrides,
    )


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The flags that assemble a :class:`BenchmarkConfig` — :func:`build_config`'s arguments.

    Shared rather than retyped. :func:`config_from_args` is the other half; call the pair.
    """
    parser.add_argument("--config", default=None, help="JSON file of the knobs this run moves")
    parser.add_argument("--corpus", default=None, help="corpus directory (default: the 500k tier)")
    parser.add_argument(
        "--artifacts-dir",
        dest="artifacts_dir",
        default=None,
        help="where the committed Task 1 inputs live (default: artifacts/task1)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="GROUP.KNOB=VALUE",
        help="override one config knob, JSON-valued; repeatable, applied last",
    )
    return parser
