"""Measure whether native 30 m DEMs are off-distribution for TerraMind's DEM tokenizer.

Issue #140, under map #117; the reading of it is `docs/research/p140-gsd-quantizer.md`. Plan
§2.2's third bullet pre-registers a GSD caveat and refuses to leave it asserted; this script is
the number, and #140 requires it **before any arm runs**, so that everything §2.2 says about it
is measured rather than post-hoc.

**What is measured, and why it needs no target.** TerraMind quantizes with FSQ: `quant_proj`
(Conv2d 768->5, 3,845 params) produces a 5-d latent, then a *parameter-free* rounding onto an
`8-8-8-6-5` grid — 15,360 codes (`p118-terramind-encoder-probe.md` §5.2). Two statistics fall
out of that rounding and out of nothing else:

    quantization residual   how far the pre-rounding value sits from the level it is assigned
    code-usage entropy      how much of the 15,360-code book the inputs actually reach

Neither needs a target, a head or any training, which is what makes them comparable across GSDs
in a way metres are not.

**Four arms, all 224 px, all 196 tokens.** The ticket fixes three; the fourth is added here and
flagged as an addition, with its reason.

    native_30m             the benchmark's own regime: 224 px at 30 m, 6.72 km
    upsampled_10m          the same windows bilinearly upsampled 3x to the 10 m grid and
                           centre-cropped to 224 px, 2.24 km. Fixes terrain, varies bandwidth.
    terramesh_10m          real TerraMesh val DEM tiles, 264 px at 10 m centre-cropped to 224 px,
                           2.24 km. Fixes bandwidth, varies geography.
    bandwidth_matched_30m  *added.* The native window with its bandwidth cut ~3x by a bilinear
                           down-up round trip: still 224 px over 6.72 km, but carrying real
                           detail only to ~90 m, so its ratio of source spacing to pixel spacing
                           is TerraMesh's 3:1 at the native geometry.

The fourth arm exists because the ticket's two references leave one hole open between them.
Fixing the pixel count at 224 makes ground extent and GSD the same knob — `upsampled_10m` covers
a ninth of the area `native_30m` does — so any difference between those two arms could be a
bandwidth effect or an extent effect, and neither reference separates them. `bandwidth_matched_30m`
holds the extent and moves only the bandwidth. It costs one extra forward pass and no new data.

**One departure from ADR-0011, measured rather than argued.** Decision 2 replaces terratorch's
bicubic 16->14 `pos_emb` resize with the centre-anchored crop "on every frozen-encoder pass", and
#134 measured the two feature sets 0.9708 apart — material. The arms here drive the *checkpoint's*
own path instead, because "off-distribution for this tokenizer" is a question about the tokenizer
as released. Since that is a departure, every arm is encoded a second time under ADR-0011's
embedding and the record carries `posemb_control`: how far the arm *differences* move between the
two. They should not move at all, the embedding being common to all four arms — but that is what
the second pass is for.

**No threshold and no drop condition.** #140 rules those out by name, deliberately unlike #134:
these statistics have no familiar scale, so any cutoff would be invented to look principled, and
a cutoff would imply the instrument can settle whether the shift *costs* TerraMind anything. It
cannot. The record carries `uniform_residual_rms` as a scale marker — what the residual would be
if the latents ignored the grid entirely — and says in the record itself that no decision hangs
on it.

**terratorch is not a project dependency and must not become one** — see the module docstring of
`scripts/p118_terramind_probe.py`, whose tile reader and normalization this script reuses, and
`scripts/p134_posemb_materiality.py`, whose tile selection it reuses so the two DEMAE arms are
the sample §2.2 already pre-registered. Run it from the same throwaway venv, plus `numcodecs`
for the TerraMesh side (its tiles are zarr-in-zip inside webdataset tars):

    uv venv --python 3.12 /tmp/tt-venv
    VIRTUAL_ENV=/tmp/tt-venv uv pip install -r docs/research/p118-terramind-requirements.txt
    VIRTUAL_ENV=/tmp/tt-venv uv pip install numcodecs
    /tmp/tt-venv/bin/python scripts/p140_gsd_quantizer.py \
        --corpus /path/to/data/corpus_500k --out docs/research/p140-gsd-quantizer.json

Everything above the terratorch import is importable without it, which is what
`tests/test_p140_gsd_quantizer.py` exercises.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import sys
import tarfile
import time
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

_SCRIPTS = Path(__file__).resolve().parent


def _sibling(name: str):
    """Load another `scripts/` module by path — the loader #134 established, same reasons.

    #118 owns the corpus reader and TerraMind's global affine, whose failure mode is silent
    (#118 §7). #134 owns the 300-tile selection rule, and this script reuses it rather than
    drawing its own: the two DEMAE arms here are the same tiles §2.2's ablation row was measured
    on, which makes them paired with each other and comparable with that row.
    """
    spec = importlib.util.spec_from_file_location(f"p140_dep_{name}", _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before it executes: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, so a module loaded purely by path cannot define one.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


p118 = _sibling("p118_terramind_probe")
p134 = _sibling("p134_posemb_materiality")

TERRAMIND_MODEL = p118.TERRAMIND_MODEL

TARGET_PX = 224  # DEMAE's primary regime, and TerraMind's 14x14 token grid at patch 16
TOKENS_PER_TILE = 196
LATENT_DIM = 5

NATIVE_GSD_M = 30.0  # GLO-30, the product both sides of the benchmark use
TERRAMIND_GSD_M = 10.0  # the Sentinel-2 grid TerraMesh resamples that product onto
GSD_RATIO = 3

#: TerraMesh's patch size — 264 px at 10 m ([TMesh] §3.2, and `p138-terramind-corpus.md` §3).
TERRAMESH_TILE_PX = 264

#: 224 is not divisible by 3, so the down-up round trip goes through 75 rather than 74.667.
#: The resulting bandwidth reduction is 224/75 = 2.9867, within 0.5% of TerraMesh's 3:1.
BANDWIDTH_MATCH_PX = 75

#: The tokenizer's own FSQ grid: `codebook_size="8-8-8-6-5", latent_dim=5` (#118 §5.2).
CODEBOOK_LEVELS = (8, 8, 8, 6, 5)
CODEBOOK_SIZE = math.prod(CODEBOOK_LEVELS)
ENTROPY_CEILING_BITS = math.log2(CODEBOOK_SIZE)

#: What the per-token residual would be if the latents fell on the rounding grid at random:
#: each of the 5 dimensions uniform on (-0.5, 0.5), so the L2 has RMS sqrt(5/12) = 0.6455.
#: A scale marker for statistics that have none. **Not a threshold** — see the module docstring.
UNIFORM_RESIDUAL_RMS = math.sqrt(LATENT_DIM / 12)

ARMS = ("native_30m", "upsampled_10m", "bandwidth_matched_30m", "terramesh_10m")
DEMAE_ARMS = ("native_30m", "upsampled_10m", "bandwidth_matched_30m")

ARM_GSD_METRES = {
    "native_30m": NATIVE_GSD_M,
    "upsampled_10m": TERRAMIND_GSD_M,
    "bandwidth_matched_30m": NATIVE_GSD_M,
    "terramesh_10m": TERRAMIND_GSD_M,
}
ARM_GROUND_METRES = {name: TARGET_PX * gsd for name, gsd in ARM_GSD_METRES.items()}

#: TerraMesh's val split for DEM: eight MajorTOM shards and one SSL4EO-S12 shard, which is the
#: split's own source composition. **val, not train** — deliberately. The reference has to be
#: in-distribution by construction (same product, same preprocessing, same grid) without being
#: literally in the tokenizer's training set, or a low residual there could be read as
#: memorization rather than as bandwidth.
TERRAMESH_REPO = "ibm-esa-geospatial/TerraMesh"
TERRAMESH_VAL_DEM_SHARDS = tuple(
    [f"val/DEM/majortom_shard_{i:06d}.tar" for i in range(1, 9)]
    + ["val/DEM/ssl4eos12_shard_000009.tar"]
)

#: Enough of each 388 MB shard to reach its first ~34 members at ~35 KB each, with slack.
#: The whole fetch is ~36 MB against a 16 TB dataset, and only statistics are committed (#128).
SHARD_PREFIX_BYTES = 4 * 1024 * 1024


# --------------------------------------------------------------------------------------
# The four arms
# --------------------------------------------------------------------------------------


def terramind_normalize_t(x: Tensor) -> Tensor:
    """TerraMind's *global* DEM affine on a tensor — #118's `terramind_normalize`, same constants.

    Applied once, after the arms are built. That is safe because the affine is linear and
    bilinear resampling is a convex combination, so the two commute; it is worth doing because
    #118 §7's normalization trap is silent, and one site is easier to keep right than four.
    """
    return (x - p118.TERRAMIND_DEM_MEAN) / p118.TERRAMIND_DEM_STD


def _bilinear(x: Tensor, px: int, antialias: bool = False) -> Tensor:
    """One place for the resampling kwargs, so every arm is resampled the same way.

    Bilinear because that is TerraMesh's own DEM recipe — [TMesh] §3.2 specifies bilinear for
    DEM specifically, against nearest for the Sentinel-2 bands (`p138-terramind-corpus.md` §3).
    An arm that used bicubic would be measuring a resampler this comparison never involves.

    `antialias` is off by default and on for exactly one step, and the asymmetry is the point.
    Upsampling cannot alias, so TerraMesh's step needs nothing. **Downsampling with plain
    bilinear does**: `F.interpolate` point-samples, so detail above the target Nyquist folds back
    in rather than being removed. Measured, before the flag existed: a pixel-Nyquist checkerboard
    came through a 224 -> 75 -> 224 round trip at std 0.129 instead of ~0, which would have given
    `bandwidth_matched_30m` high-frequency content a genuinely coarser product does not have —
    the exact opposite of the arm's purpose.
    """
    return F.interpolate(x, size=(px, px), mode="bilinear", align_corners=False, antialias=antialias)


def centre_crop(x: Tensor, px: int) -> Tensor:
    """The centre `px` of each spatial axis."""
    lo = (x.shape[-1] - px) // 2
    return x[..., lo : lo + px, lo : lo + px]


def upsample_to_10m(x: Tensor) -> Tensor:
    """224 px at 30 m -> the centre 2.24 km of the same window, 224 px at 10 m.

    The ticket's first reference: same geography, three times the bandwidth per pixel. Note what
    "same terrain" can and cannot mean here — fixing the pixel count at 224 makes extent and GSD
    one knob, so this arm sees the centre ninth of the native window's area, not the same area at
    a finer sampling. `bandwidth_matched_30m` is what prices that difference.
    """
    return centre_crop(_bilinear(x, TARGET_PX * GSD_RATIO), TARGET_PX)


def bandwidth_match(x: Tensor) -> Tensor:
    """224 px at 30 m -> 224 px at 30 m carrying real detail only to ~90 m.

    Two steps, and they are not mirror images. Down to 75 px **with anti-aliasing**, which is how
    a coarser elevation product is actually made — TanDEM-X's 30 m posts are a band-limited
    product, not a point-sampling of something finer, so a ~90 m stand-in has to be band-limited
    too. Then back up to 224 px with plain bilinear, which is TerraMesh's own step verbatim
    ([TMesh] §3.2).

    What comes out has the native arm's ground extent and pixel grid, and TerraMesh's 3:1 ratio
    of source post spacing to pixel spacing.
    """
    return _bilinear(_bilinear(x, BANDWIDTH_MATCH_PX, antialias=True), TARGET_PX)


def demae_arms(x: Tensor) -> dict[str, Tensor]:
    """The three arms that come from one DEMAE tile, `[B, 1, 224, 224]` each.

    `native_30m` is returned untouched rather than resampled through an identity, so the control
    arm carries no resampling at all and any difference from it is the other arms' doing.
    """
    return {
        "native_30m": x,
        "upsampled_10m": upsample_to_10m(x),
        "bandwidth_matched_30m": bandwidth_match(x),
    }


def terramesh_arm(x: Tensor) -> Tensor:
    """A 264 px TerraMesh tile -> its centre 224 px. No resampling: it is already at 10 m.

    Not the tokenizer's training crop, which was a random 0.8-1.0 scale crop of the 264 resized
    to 256 (`p138-terramind-corpus.md` §3). A centre crop is used instead because what this arm
    has to match is the *other* 10 m arm's geometry — 224 px over 2.24 km — so that the two
    differ in geography and nothing else.
    """
    return centre_crop(x, TARGET_PX)


# --------------------------------------------------------------------------------------
# The quantizer, and the value it throws away
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Assignment:
    """One batch through the FSQ grid. `residual` and `number` are `[B, T, 5]`, `indices` `[B, T]`."""

    residual: Tensor
    number: Tensor
    indices: Tensor

    def per_token_l2(self) -> Tensor:
        """`[B, T]` — the distance from the pre-rounding value to its assigned level.

        In level units, where every dimension's grid spacing is exactly 1. The alternative is the
        `[0, 1]` code space the decoder reads, where the five dimensions have spacings 1/7, 1/7,
        1/7, 1/5 and 1/4 and a per-dimension distance means five different things. Both are
        reported; this is the primary.
        """
        return self.residual.pow(2).sum(-1).sqrt()

    def per_token_l2_code_space(self, levels: Tensor) -> Tensor:
        """`[B, T]` in the `[0, 1]` units `quantize` actually hands the decoder."""
        return (self.residual / (levels - 1).to(self.residual)).pow(2).sum(-1).sqrt()


def fsq_levels(quantizer) -> Tensor:
    """The `8-8-8-6-5` grid, read off the module rather than off this file."""
    return quantizer._levels.detach().cpu().to(torch.int64)


def fsq_assign(latent: Tensor, levels: Tensor) -> Assignment:
    """`[B, 5, H, W]` -> the pre-rounding value's distance to its level, and the code index.

    This re-derives what `FiniteScalarQuantizer.forward` computes and discards. The module keeps
    `code = round(sigmoid(z) * d) / d` and `indice = sum(round(sigmoid(z) * d) * basis)`; the
    *unrounded* `sigmoid(z) * d` never leaves it, and that is the only thing a residual can be
    measured against. So the derivation is unavoidable — and it is checked, every batch, against
    the two things the module does return (`verify_against_quantizer`).

    Note where the distance lives. `sigmoid` is the reason a small residual is not automatically
    good news: a latent driven far from zero saturates against a terminal level and lands *on*
    the grid for free. `saturated_fraction` is what keeps those two readings apart.
    """
    if latent.shape[1] != LATENT_DIM:
        raise ValueError(f"expected a {LATENT_DIM}-d latent, got {tuple(latent.shape)}")

    batch = latent.shape[0]
    d = (levels - 1).to(latent.device, latent.dtype)
    basis = torch.cumprod(
        torch.tensor([1, *levels.tolist()[:-1]], device=latent.device, dtype=torch.int64), dim=0
    )

    unrounded = torch.sigmoid(latent).permute(0, 2, 3, 1).reshape(batch, -1, LATENT_DIM) * d
    number = unrounded.round()
    return Assignment(
        residual=unrounded - number,
        number=number,
        indices=(number.to(torch.int64) * basis).sum(-1),
    )


def verify_against_quantizer(quantizer, latent: Tensor, assignment: Assignment) -> tuple[bool, bool]:
    """Run the module and check `fsq_assign` agrees about everything the module keeps.

    The control that makes the residual readable. `fsq_assign` reconstructs the quantizer's
    arithmetic from its source, which is a claim about someone else's code — so it is measured
    rather than read, on every batch, and the result goes in the record.
    """
    with torch.no_grad():
        code, _, indices = quantizer(latent)

    levels = fsq_levels(quantizer).to(latent.device)
    batch = latent.shape[0]
    code_matches = bool(
        torch.allclose(
            assignment.number / (levels - 1).to(code.dtype),
            code.permute(0, 2, 3, 1).reshape(batch, -1, LATENT_DIM),
            atol=1e-6,
        )
    )
    index_matches = bool(torch.equal(assignment.indices, indices.reshape(batch, -1).to(torch.int64)))
    return code_matches, index_matches


def saturated_fraction(number: Tensor, levels: Tensor) -> float:
    """Share of `(token, dimension)` assignments sitting in a terminal level, 0 or `L - 1`.

    Reported beside the residual because the two are easy to confuse and point opposite ways. A
    token whose latent saturates `sigmoid` is *pinned* to an end of the grid: its residual goes to
    zero, which looks like a well-represented input and is the opposite of one.
    """
    return float(np.mean(_terminal_flags(number, levels)))


def saturated_fraction_per_dim(number: Tensor, levels: Tensor) -> list[float]:
    """`saturated_fraction`, split by latent dimension — which axis is pinned, if any."""
    return _terminal_flags(number, levels).mean(axis=0).tolist()


def _terminal_flags(number: Tensor, levels: Tensor) -> np.ndarray:
    """`(n_tokens, 5)` float64 flags for "this assignment is at an end of its axis".

    Averaged in NumPy rather than in torch because MPS has no float64, and averaging a boolean
    mask in float32 is exactly the place where a device would quietly disagree with the CPU run.
    The tensor is `[B, 196, 5]`, so the copy costs nothing.
    """
    top = (levels - 1).to(number.device, number.dtype)
    flags = ((number == 0) | (number == top)).detach().cpu().numpy()
    return flags.reshape(-1, LATENT_DIM).astype(np.float64)


def entropy_bits_from_counts(counts: np.ndarray) -> float:
    """Plug-in (maximum-likelihood) entropy of a histogram, in bits."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def bootstrap_entropies(
    per_arm: dict[str, np.ndarray],
    paired_groups: Sequence[Sequence[str]],
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Resample **tiles**, not tokens, and recompute each arm's entropy on every replicate.

    The residual comes with an interval because it is a mean over 300 paired tiles. The entropy
    is a single number over the pooled 58,800 tokens, so without this it would be the only
    statistic in the record with no stated uncertainty — and the comparison would rest on
    eyeballing which gaps look big.

    Tiles, because tokens within a tile are anything but independent: 196 tokens of one mountain
    range share a code distribution, so a token-level resample would report an interval several
    times too narrow. The unit of independent sampling here is the tile.

    `paired_groups` names the arms that must share a replicate's tile draw. The three DEMAE arms
    are three views of the *same* 300 tiles, so pairing them removes terrain from their
    differences exactly as it does for the residual; TerraMesh is a different 300 tiles and is
    resampled on its own.

    **Read the intervals on differences, not on levels.** A draw of n tiles from n holds only
    ~63% distinct tiles, so it sees fewer distinct codes and its plug-in entropy is biased low
    against the point estimate. That offset is small and common to every arm, so it cancels out
    of a difference and does not out of a level — which is the same argument the matched token
    counts rest on, one level up.
    """
    rng = np.random.default_rng(seed)
    out = {name: np.empty(n_boot, dtype=np.float64) for name in per_arm}

    grouped = {name for group in paired_groups for name in group}
    groups = [tuple(group) for group in paired_groups]
    groups += [(name,) for name in per_arm if name not in grouped]

    for replicate in range(n_boot):
        for group in groups:
            n_tiles = per_arm[group[0]].shape[0]
            pick = rng.integers(0, n_tiles, n_tiles)
            for name in group:
                counts = np.bincount(per_arm[name][pick].ravel(), minlength=CODEBOOK_SIZE)
                out[name][replicate] = entropy_bits_from_counts(counts)
    return out


def bootstrap_interval(samples: np.ndarray) -> dict[str, Any]:
    """A percentile interval and a spread from bootstrap replicates. No test, no verdict."""
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"sd": float(samples.std(ddof=1)), "ci95": [float(lo), float(hi)]}


class CodeUsage:
    """Streaming histograms over the 15,360 codes and over each dimension's levels.

    Counts rather than samples, so the estimate is exact for the sample it was given and the
    memory is 15,360 integers whatever the tile count.

    **The estimator's bias is the reason the arms are size-matched.** A plug-in entropy from
    58,800 draws over 15,360 bins is biased low — most bins are unseen, and unseen bins
    contribute nothing where the truth says they contribute something. Miller-Madow is reported
    beside it, but the correction is not what the comparison rests on: every arm draws exactly
    the same number of tokens, so the bias is common to all four and the *differences* between
    them survive it.
    """

    def __init__(self, levels: Sequence[int] = CODEBOOK_LEVELS):
        self.levels = tuple(int(k) for k in levels)
        self.codebook_size = math.prod(self.levels)
        self.n = 0
        self._counts = np.zeros(self.codebook_size, dtype=np.int64)
        self._per_dim = [np.zeros(k, dtype=np.int64) for k in self.levels]
        self._have_per_dim = False

    def update(self, indices: Tensor, numbers: Tensor | None) -> None:
        """`indices` is `[B, T]` of code ids; `numbers` is `[B, T, 5]` of per-dimension levels."""
        flat = np.asarray(indices.detach().cpu().reshape(-1), dtype=np.int64)
        if flat.size and (flat.min() < 0 or flat.max() >= self.codebook_size):
            raise ValueError(f"code index outside [0, {self.codebook_size}): {flat.min()}..{flat.max()}")

        self._counts += np.bincount(flat, minlength=self.codebook_size)
        self.n += int(flat.size)

        if numbers is not None:
            levels = np.asarray(numbers.detach().cpu().reshape(-1, LATENT_DIM), dtype=np.int64)
            for dim, k in enumerate(self.levels):
                self._per_dim[dim] += np.bincount(levels[:, dim], minlength=k)
            self._have_per_dim = True

    @property
    def n_codes_used(self) -> int:
        return int((self._counts > 0).sum())

    def entropy_bits(self) -> float:
        """Plug-in (maximum-likelihood) entropy over the codebook, in bits."""
        return entropy_bits_from_counts(self._counts)

    def entropy_bits_miller_madow(self) -> float:
        """Plug-in plus `(K_used - 1) / (2 n ln 2)`, clamped to the codebook's ceiling.

        Clamped because Miller-Madow is a bias correction and not a bound: on a near-uniform
        sample it overshoots `log2(K)`, and an entropy above the ceiling would read in the
        writeup as a codebook using more than all of itself.
        """
        if self.n == 0:
            return 0.0
        corrected = self.entropy_bits() + (self.n_codes_used - 1) / (2 * self.n * math.log(2))
        return min(corrected, math.log2(self.codebook_size))

    def perplexity(self) -> float:
        """`2 ** entropy` — the effective number of codes in use."""
        return float(2 ** self.entropy_bits())

    def entropy_bits_per_dim(self) -> list[float]:
        """The five marginal entropies, which say *which* axis collapsed when the joint does."""
        if not self._have_per_dim:
            raise ValueError("per-dimension counts were never supplied; pass `numbers` to update()")
        return [entropy_bits_from_counts(counts) for counts in self._per_dim]

    def summary(self) -> dict[str, Any]:
        return {
            "n_codes_used": self.n_codes_used,
            "codebook_size": self.codebook_size,
            "entropy_bits": self.entropy_bits(),
            "entropy_bits_miller_madow": self.entropy_bits_miller_madow(),
            "entropy_ceiling_bits": ENTROPY_CEILING_BITS,
            "perplexity": self.perplexity(),
            "entropy_bits_per_dim": self.entropy_bits_per_dim(),
        }


def paired_difference(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """`b - a` over paired observations: the mean, its spread, and a normal-approximation CI.

    Paired because the DEMAE arms are the *same* 300 tiles. Terrain varies enormously between
    tiles and hardly at all between two views of one tile, so the paired spread is the one that
    describes how reliably the arms differ. No test statistic and no verdict — #140 forbids a
    decision rule, and an interval is a magnitude, not one.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must have the same shape, got {a.shape} and {b.shape}")

    diff = b - a
    sd = float(diff.std(ddof=1))
    se = sd / math.sqrt(diff.size)
    mean = float(diff.mean())
    half = 1.959963984540054 * se
    return {"n": int(diff.size), "mean": mean, "sd": sd, "se": se, "ci95": [mean - half, mean + half]}


def summarize_residual(
    per_token: np.ndarray, per_dim_abs: np.ndarray, code_space: np.ndarray
) -> dict[str, Any]:
    """The distribution, not just its mean — percentiles and the five per-dimension means."""
    p01, p50, p99 = np.percentile(per_token, [1, 50, 99])
    return {
        "units": "fsq_levels",
        "mean": float(per_token.mean()),
        "sd": float(per_token.std()),
        "min": float(per_token.min()),
        "p01": float(p01),
        "median": float(p50),
        "p99": float(p99),
        "max": float(per_token.max()),
        "mean_per_dim": [float(v) for v in per_dim_abs],
        "mean_code_space": float(code_space.mean()),
    }


# --------------------------------------------------------------------------------------
# The TerraMesh side: a few hundred tiles, never the 16 TB
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DemTile:
    """One TerraMesh DEM patch: 264 px of int16 metres, and where on Earth it is."""

    sample_id: str
    lat: float
    lon: float
    metres: np.ndarray


def iter_tar_members(raw: bytes) -> Iterator[tuple[str, bytes]]:
    """Members out of a **prefix** of a webdataset tar, stopping cleanly when the bytes run out.

    The fetch range-reads the first few MB of a 388 MB shard, so the archive always ends
    mid-member. That is the design, not a failure: take the first handful and hang up. Anything
    the stream cannot deliver in full is dropped rather than yielded short, so a truncated
    member never reaches the decoder as a corrupt zip.
    """
    tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r|")
    while True:
        try:
            member = tf.next()
        except (tarfile.TarError, EOFError):
            return
        if member is None:
            return
        if not member.isfile():
            continue
        try:
            handle = tf.extractfile(member)
            payload = handle.read() if handle is not None else b""
        except (tarfile.TarError, EOFError):
            return
        if len(payload) != member.size:
            return
        yield member.name, payload


def decode_dem_zarr_zip(raw: bytes) -> DemTile:
    """One `*.zarr.zip` member -> elevation in metres, plus its centroid and sample id.

    Decoded from the store's own `.zarray` metadata with `numcodecs`, rather than through
    xarray/zarr. Two reasons: the stores are zarr v2 and the resolvable zarr is v3, which needs a
    compatibility flag and a temporary file for a `ZipStore`; and the array is a single
    blosc-zstd chunk, so the whole decode is four reads and a `frombuffer`. Fewer moving parts in
    a throwaway venv, and every step is visible here rather than inside a version-sensitive
    backend.
    """
    import numcodecs

    store = zipfile.ZipFile(io.BytesIO(raw))
    meta = json.loads(store.read("bands/.zarray"))

    shape = tuple(meta["shape"])
    if shape[-2:] != (TERRAMESH_TILE_PX, TERRAMESH_TILE_PX):
        raise ValueError(f"expected a {TERRAMESH_TILE_PX} px TerraMesh patch, got shape {shape}")
    if tuple(meta["chunks"]) != shape:
        raise ValueError(f"expected one chunk per patch, got chunks {meta['chunks']} for shape {shape}")

    payload = numcodecs.get_codec(meta["compressor"]).decode(store.read("bands/0.0.0.0"))
    metres = np.frombuffer(payload, dtype=np.dtype(meta["dtype"])).reshape(shape)[0, 0]

    return DemTile(
        # zarr writes fixed-width unicode as UTF-32-LE; the width lives in `sample/.zarray`, and
        # stripping the padding is equivalent and needs one less read.
        sample_id=store.read("sample/0").decode("utf-32-le").rstrip("\x00"),
        lat=float(np.frombuffer(store.read("center_lat/0"), dtype="<f8")[0]),
        lon=float(np.frombuffer(store.read("center_lon/0"), dtype="<f8")[0]),
        metres=np.ascontiguousarray(metres),
    )


def round_robin(streams: Sequence[Iterator], n: int) -> list:
    """Deal one item from each stream in turn until `n` are collected or all run dry.

    The whole TerraMesh selection rule. The alternative — the first 300 members of one shard —
    would rest on an assumption about how the shard is ordered; dealing across all nine holds
    whatever ordering each shard has and adds the val split's own source mix (eight MajorTOM
    shards, one SSL4EO-S12) rather than picking one of them.
    """
    out: list = []
    alive = list(streams)
    while alive and len(out) < n:
        still_alive = []
        for stream in alive:
            if len(out) >= n:
                still_alive.append(stream)
                continue
            try:
                out.append(next(stream))
            except StopIteration:
                continue
            still_alive.append(stream)
        alive = still_alive
    return out


def fetch_shard_prefix(shard: str, cache: Path, nbytes: int = SHARD_PREFIX_BYTES) -> bytes:
    """The first `nbytes` of one TerraMesh shard, over an HTTP range request, cached on disk.

    A range request rather than `hf_hub_download`: the shard is 388 MB and this needs its first
    4 MB. The cache lives under `data/`, which `.gitignore` excludes — #128's standing rule is
    that this benchmark commits statistics and never tiles, and the share-alike licence on
    TerraMesh is a non-issue for exactly that reason.
    """
    import requests
    from huggingface_hub import hf_hub_url

    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (shard.replace("/", "_") + f".first{nbytes}")
    if path.exists():
        return path.read_bytes()

    url = hf_hub_url(TERRAMESH_REPO, shard, repo_type="dataset")
    response = requests.get(url, headers={"Range": f"bytes=0-{nbytes - 1}"}, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return response.content


def fetch_terramesh_tiles(cache: Path, n_tiles: int) -> tuple[list[DemTile], dict[str, Any]]:
    """`n_tiles` DEM patches dealt round-robin across the nine val shards, plus provenance."""
    from huggingface_hub import HfApi

    prefixes = [fetch_shard_prefix(shard, cache) for shard in TERRAMESH_VAL_DEM_SHARDS]
    streams = [(decode_dem_zarr_zip(payload) for _, payload in iter_tar_members(raw)) for raw in prefixes]
    tiles = round_robin(streams, n_tiles)

    if len(tiles) < n_tiles:
        raise RuntimeError(
            f"only {len(tiles)} TerraMesh tiles in {SHARD_PREFIX_BYTES} B prefixes; raise SHARD_PREFIX_BYTES"
        )

    info = HfApi().dataset_info(TERRAMESH_REPO)
    ids = [tile.sample_id for tile in tiles]
    provenance = {
        "repo_id": TERRAMESH_REPO,
        "revision": info.sha,
        "licence": "cc-by-sa-4.0",
        "gated": bool(info.gated),
        "private": bool(info.private),
        "split": "val",
        "modality": "DEM",
        "n_tiles": len(tiles),
        "n_shards": len(TERRAMESH_VAL_DEM_SHARDS),
        "shards": list(TERRAMESH_VAL_DEM_SHARDS),
        "selection_rule": "round-robin over the first members of each shard's prefix",
        "bytes_fetched": sum(len(raw) for raw in prefixes),
        "tile_px": TERRAMESH_TILE_PX,
        # Statistics and provenance only. #128's rule, and what makes the share-alike licence
        # irrelevant here: no TerraMesh pixel is committed to this repo.
        "tiles_committed": 0,
        "first_sample_id": ids[0],
        "last_sample_id": ids[-1],
        # The whole ordered draw, in 16 characters, so it is re-derivable exactly rather than
        # approximately — without committing 300 ids to describe 300 tiles nobody can see.
        "sample_id_list_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16],
        "lat_range": [min(t.lat for t in tiles), max(t.lat for t in tiles)],
        "lon_range": [min(t.lon for t in tiles), max(t.lon for t in tiles)],
        "elevation_range_metres": [
            float(min(t.metres.min() for t in tiles)),
            float(max(t.metres.max() for t in tiles)),
        ],
    }
    return tiles, provenance


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------


class ArmAccumulator:
    """Everything one arm contributes, streamed: residual quantiles, usage counts, per-tile means."""

    def __init__(self, name: str):
        self.name = name
        self.usage = CodeUsage()
        self._per_token: list[np.ndarray] = []
        self._code_space: list[np.ndarray] = []
        self._per_dim_abs = np.zeros(LATENT_DIM, dtype=np.float64)
        self._saturated = np.zeros(LATENT_DIM, dtype=np.float64)
        self._batches = 0
        self.per_tile_mean: list[float] = []
        #: `(n_tiles, 196)` code ids, kept per tile rather than pooled so the entropy can be
        #: resampled at the tile level. 300 x 196 int16 is 118 KB — the codebook tops out at
        #: 15,359, well inside int16.
        self._per_tile_indices: list[np.ndarray] = []

    def update(self, assignment: Assignment, levels: Tensor) -> None:
        per_token = assignment.per_token_l2().detach().cpu().numpy()
        self._per_token.append(per_token.reshape(-1))
        self._per_tile_indices.append(
            np.asarray(assignment.indices.detach().cpu().numpy(), dtype=np.int16)
        )
        self._code_space.append(assignment.per_token_l2_code_space(levels).detach().cpu().numpy().reshape(-1))
        self.per_tile_mean.extend(per_token.mean(axis=1).tolist())

        self._per_dim_abs += assignment.residual.abs().reshape(-1, LATENT_DIM).mean(0).detach().cpu().numpy()
        self._saturated += np.asarray(saturated_fraction_per_dim(assignment.number, levels))
        self._batches += 1

        self.usage.update(assignment.indices, assignment.number)

    @property
    def per_tile_indices(self) -> np.ndarray:
        """`(n_tiles, 196)` code ids, the unit the entropy bootstrap resamples."""
        return np.concatenate(self._per_tile_indices, axis=0)

    def summary(self) -> dict[str, Any]:
        per_token = np.concatenate(self._per_token)
        per_dim = self._per_dim_abs / self._batches
        saturated = self._saturated / self._batches
        return {
            "gsd_metres": ARM_GSD_METRES[self.name],
            "ground_metres": ARM_GROUND_METRES[self.name],
            "n_tokens": int(per_token.size),
            "residual": summarize_residual(per_token, per_dim, np.concatenate(self._code_space)),
            "saturated_fraction": float(saturated.mean()),
            "saturated_fraction_per_dim": [float(v) for v in saturated],
            "code_usage": self.usage.summary(),
        }


def encode_arm(model, x: Tensor) -> Tensor:
    """`[B, 1, 224, 224]` in TerraMind units -> the `[B, 5, 14, 14]` pre-quantization latent.

    `prepare_input` is in the extraction snippet the selection memo's §3.4 documents, and #118
    §5.1 verified it is the identity for this model (`undo_std=False`, `cls_emb=None`). Called
    anyway rather than skipped, so this drives the documented path instead of a paraphrase of it.
    """
    with torch.no_grad():
        return model.quant_proj(model.encoder(model.prepare_input(x)))


def _demae_batches(corpus: Path, rows: Sequence[dict], batch_size: int) -> Iterator[Tensor]:
    """Tiles out of the shards as `[B, 1, 224, 224]` **metres** — the arms are built before the
    affine, and `terramind_normalize_t` is applied to each arm afterwards."""
    for lo in range(0, len(rows), batch_size):
        chunk = rows[lo : lo + batch_size]
        metres = np.stack(
            [p118.load_tile_metres(corpus, row["shard_id"], row["row_index"]) for row in chunk]
        )
        yield torch.from_numpy(metres)[:, None]


def _terramesh_batches(tiles: Sequence[DemTile], batch_size: int) -> Iterator[Tensor]:
    for lo in range(0, len(tiles), batch_size):
        chunk = tiles[lo : lo + batch_size]
        metres = np.stack([tile.metres.astype(np.float32) for tile in chunk])
        yield torch.from_numpy(metres)[:, None]


def run(
    corpus: Path,
    cache: Path,
    device: str,
    batch_size: int,
    per_stratum: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    started = time.time()

    frame = p134.read_selection_frame(corpus)
    tile_ids = p134.select_tile_ids(frame, per_stratum=per_stratum)
    rows = p134.selection_rows(frame, tile_ids)

    constants = p118.probe_constants()
    if not constants["matches_this_script"]:
        raise AssertionError(
            f"installed terratorch says mean={constants['mean']} std={constants['std']}, "
            f"#118 recorded {p118.TERRAMIND_DEM_MEAN} / {p118.TERRAMIND_DEM_STD}"
        )

    tiles, terramesh = fetch_terramesh_tiles(cache, len(tile_ids))

    model = p118.build_tokenizer().to(device)
    quantizer = model.quantize
    levels = fsq_levels(quantizer)
    if tuple(levels.tolist()) != CODEBOOK_LEVELS:
        raise AssertionError(f"installed tokenizer quantizes onto {levels.tolist()}, not {CODEBOOK_LEVELS}")

    device_levels = levels.to(device)

    # Counted before anything installs a `pos_emb`, or the encoder total would come out 46,080
    # short of #118's 90,172,416 — the difference between a 16x16 embedding and a 14x14 one.
    parameters = {
        "quantizer_parameters": sum(p.numel() for p in quantizer.parameters()),
        "quant_proj_parameters": sum(p.numel() for p in model.quant_proj.parameters()),
        # #140's "encoder-only: 90.2M of the 1.15 GB checkpoint", counted rather than quoted.
        # `build_tokenizer` loads the whole tokenizer (`strict=True`), but nothing below runs the
        # decoder, so this is what the diagnostic actually costs.
        "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "decoder_parameters_unused": sum(p.numel() for p in model.decoder.parameters()),
    }

    def measure(pos_emb: Tensor) -> tuple[dict[str, ArmAccumulator], bool, bool]:
        """Every arm through the encoder once, with `pos_emb` installed."""
        model.encoder.pos_emb = torch.nn.Parameter(pos_emb.to(device), requires_grad=False)
        accumulators = {name: ArmAccumulator(name) for name in ARMS}
        code_ok = index_ok = True

        def consume(name: str, x_metres: Tensor) -> None:
            nonlocal code_ok, index_ok
            latent = encode_arm(model, terramind_normalize_t(x_metres).to(device))
            assignment = fsq_assign(latent, device_levels)
            ok_code, ok_index = verify_against_quantizer(quantizer, latent, assignment)
            code_ok &= ok_code
            index_ok &= ok_index
            accumulators[name].update(assignment, device_levels)

        for batch in _demae_batches(corpus, rows, batch_size):
            for name, arm in demae_arms(batch).items():
                consume(name, arm)
        for batch in _terramesh_batches(tiles, batch_size):
            consume("terramesh_10m", terramesh_arm(batch))
        return accumulators, code_ok, index_ok

    checkpoint_pos_emb = model.encoder.pos_emb.detach().cpu()
    accumulators, code_matches, index_matches = measure(checkpoint_pos_emb)

    # ADR-0011 Decision 2 replaces terratorch's bicubic 16->14 resize with the centre-anchored
    # crop "on every frozen-encoder pass", and #134 measured the two feature sets 0.9708 apart —
    # material. This run drives the *checkpoint's* own path, because "off-distribution for this
    # tokenizer" is a question about the tokenizer as released. That is a departure from the ADR,
    # so it is measured rather than argued: every arm is encoded a second time under ADR-0011's
    # embedding, and the record carries how far the answer moves. It should move by nothing, since
    # the embedding is common to all four arms and only their differences are read — but "should"
    # is what the second pass is for.
    control_accumulators, control_code_ok, control_index_ok = measure(
        p134.posemb_variants(checkpoint_pos_emb)["centre"]
    )
    code_matches &= control_code_ok
    index_matches &= control_index_ok

    import importlib.metadata as md

    arms = {name: acc.summary() for name, acc in accumulators.items()}
    per_tile = {name: np.asarray(acc.per_tile_mean) for name, acc in accumulators.items()}

    replicates = bootstrap_entropies(
        {name: acc.per_tile_indices for name, acc in accumulators.items()},
        paired_groups=[DEMAE_ARMS],
        n_boot=n_boot,
        seed=seed,
    )
    for name, arm in arms.items():
        arm["code_usage"] |= {
            "entropy_bits_bootstrap": bootstrap_interval(replicates[name]),
            "entropy_bits_bootstrap_mean": float(replicates[name].mean()),
        }

    #: What the residual is compared across — the three arms drawn from one tile set, so the
    #: comparison can be paired.
    PAIRED_PAIRS = (
        ("native_30m", "bandwidth_matched_30m"),
        ("native_30m", "upsampled_10m"),
        ("upsampled_10m", "bandwidth_matched_30m"),
    )
    #: What the entropy is compared across — the three above plus the two that reach TerraMesh,
    #: which is a different tile set and therefore resampled independently.
    ENTROPY_PAIRS = PAIRED_PAIRS + (
        ("native_30m", "terramesh_10m"),
        ("upsampled_10m", "terramesh_10m"),
    )

    control_arms = {name: acc.summary() for name, acc in control_accumulators.items()}
    posemb_control = {
        "adr": "0011",
        "note": (
            "ADR-0011 Decision 2 replaces terratorch's bicubic 16->14 pos_emb resize with the "
            "centre-anchored crop on every frozen-encoder pass, and #134 measured the two feature "
            "sets 0.9708 apart. The arms above drive the checkpoint's own path, because "
            "'off-distribution for this tokenizer' is a question about the tokenizer as released. "
            "This is the same run under ADR-0011's embedding, so the departure is measured rather "
            "than argued."
        ),
        "arms": {
            name: {
                "residual_mean": arm["residual"]["mean"],
                "entropy_bits": arm["code_usage"]["entropy_bits"],
                "saturated_fraction": arm["saturated_fraction"],
            }
            for name, arm in control_arms.items()
        },
        "entropy_difference": {
            f"{a}_vs_{b}": control_arms[b]["code_usage"]["entropy_bits"]
            - control_arms[a]["code_usage"]["entropy_bits"]
            for a, b in ENTROPY_PAIRS
        },
        # The only number that decides whether the departure matters: how far the *comparisons*
        # move between the two embeddings. The levels are expected to move; the differences are not.
        "max_abs_shift_in_entropy_difference": max(
            abs(
                (control_arms[b]["code_usage"]["entropy_bits"] - control_arms[a]["code_usage"]["entropy_bits"])
                - (arms[b]["code_usage"]["entropy_bits"] - arms[a]["code_usage"]["entropy_bits"])
            )
            for a, b in ENTROPY_PAIRS
        ),
        "max_abs_shift_in_residual_mean": max(
            abs(control_arms[name]["residual"]["mean"] - arms[name]["residual"]["mean"]) for name in ARMS
        ),
    }

    return {
        "issue": 140,
        "model": TERRAMIND_MODEL,
        "device": device,
        "versions": {
            p: md.version(p) for p in ("terratorch", "diffusers", "torch", "timm", "numpy", "numcodecs")
        },
        "normalization": "terramind_global_z_score",
        "constants": constants,
        "n_tiles_per_arm": len(tile_ids),
        "tokens_per_tile": TOKENS_PER_TILE,
        # The instrument.
        "quantizer": {
            "module": type(quantizer).__name__,
            "levels": levels.tolist(),
            "codebook_size": CODEBOOK_SIZE,
            "parameters": parameters["quantizer_parameters"],
            "quant_proj_parameters": parameters["quant_proj_parameters"],
            "encoder_parameters": parameters["encoder_parameters"],
            "decoder_parameters_unused": parameters["decoder_parameters_unused"],
            "code_matches_module": code_matches,
            "index_matches_module": index_matches,
        },
        "uniform_residual_rms": UNIFORM_RESIDUAL_RMS,
        "uniform_residual_rms_note": (
            "What the per-token residual would be if the latents fell on the rounding grid at "
            "random. A scale marker for a statistic that has none; no decision hangs on it, and "
            "#140 fixes no threshold and no drop condition."
        ),
        "entropy_ceiling_bits": ENTROPY_CEILING_BITS,
        "bandwidth_match_factor": TARGET_PX / BANDWIDTH_MATCH_PX,
        # The samples.
        "demae": {
            "selection_rule": "p134",
            "tiles_per_stratum": per_stratum,
            "n_strata": int(len({row["stratum_id"] for row in rows})),
            "n_source_cogs": len({row["source_cog"] for row in rows}),
            "first_tile_id": tile_ids[0],
            "last_tile_id": tile_ids[-1],
            "lat_range": [min(row["lat"] for row in rows), max(row["lat"] for row in rows)],
            "lon_range": [min(row["lon"] for row in rows), max(row["lon"] for row in rows)],
        },
        "terramesh": terramesh,
        # The answer.
        "arms": arms,
        "bootstrap": {"n_replicates": n_boot, "seed": seed, "resampled": "tiles"},
        # The residual is a mean over paired tiles, so its interval is analytic; the entropy is
        # one number over pooled tokens, so its interval is resampled. Different machinery, same
        # purpose: neither statistic is reported without a spread.
        "paired_residual": {
            f"{a}_vs_{b}": paired_difference(per_tile[a], per_tile[b]) for a, b in PAIRED_PAIRS
        },
        "entropy_difference": {
            f"{a}_vs_{b}": {"mean": float((replicates[b] - replicates[a]).mean())}
            | bootstrap_interval(replicates[b] - replicates[a])
            | {"point": arms[b]["code_usage"]["entropy_bits"] - arms[a]["code_usage"]["entropy_bits"]}
            for a, b in ENTROPY_PAIRS
        },
        "posemb_control": posemb_control,
        # The limit, carried with the number rather than only in the prose around it.
        "reading_rule": (
            "#122's symmetry rule: this result is not available to explain a TerraMind loss "
            "unless it is also priced as a discount on a TerraMind win."
        ),
        "measures": "whether native 30 m inputs are off-distribution for TerraMind's DEM tokenizer",
        "does_not_measure": (
            "what that costs at void filling. Only a retrained head answers that, and plan §2.2 "
            "declines to build one."
        ),
        "limit": (
            "An input-side diagnostic on the tokenizer's own quantizer. It reads on distribution, "
            "never on downstream error, and it fixes no threshold."
        ),
        "runtime_seconds": time.time() - started,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=None, help="path to data/corpus_500k")
    parser.add_argument(
        "--cache", type=Path, default=None, help="where TerraMesh shard prefixes are cached"
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--per-stratum", type=int, default=p134.TILES_PER_STRATUM)
    parser.add_argument("--n-boot", type=int, default=1000, help="tile-level bootstrap replicates")
    parser.add_argument("--seed", type=int, default=140, help="seeds the bootstrap, nothing else")
    parser.add_argument("--out", type=Path, default=None, help="where to write the JSON record")
    args = parser.parse_args(argv)

    corpus = args.corpus or p118.find_corpus()
    record = run(
        corpus=corpus,
        cache=args.cache or corpus.parent / "terramesh_val_dem",
        device=args.device,
        batch_size=args.batch_size,
        per_stratum=args.per_stratum,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
