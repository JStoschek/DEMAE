"""Re-derive TerraMind's DEM-tokenizer position embedding, and measure what terratorch's resize costs.

Issue #133, under map #117; the reading of it is ADR-0011. Plan §2.2's fairness premise is that
both ViTs produce a 14x14 token grid at 224 px. They do — but TerraMind's tokenizer trained at
256 px, so `ViTEncoder.pos_emb` is natively `[1, 768, 16, 16]` and terratorch's `forward` reaches
14x14 with `F.interpolate(..., mode="bicubic", align_corners=False)`. This script quantifies the
gap between that resize and the embedding a 224 px crop actually wants.

**It needs no weights, no GPU and no terratorch.** `build_2d_sincos_posemb` below is a verbatim
port of terratorch 1.2.10's
`terratorch/models/backbones/terramind/tokenizer/models/vit_models.py`, differing only in dtype
(float64, so the bit-identity claim is not a float32 accident) and in using `Tensor.permute`
where the original uses `einops.rearrange`. That is the whole point: TerraMind's embedding is a
closed-form function of the token's *integer* grid index, with `learnable_pos_emb=False`, so it
appears in the checkpoint only because it is wrapped in an `nn.Parameter`. Everything here is
reproducible from the formula.

Four claims, asserted by `tests/test_p133_posemb_geometry.py` against the JSON this writes so
they cannot drift out of ADR-0011's prose silently:

1. `build_2d_sincos_posemb(14, 14)` is **bit-identical** to `pos_emb[:, :, :14, :14]`, because the
   grid index is absolute — so the correct embedding for a crop *is* the crop.
2. Bicubic samples the 16-grid at spacing **1.1429**, not 1.0 — it preserves normalized position
   and therefore stretches the index spacing by 16/14.
3. That costs ~26% relative RMS, and leaves only **20 of 196** tokens ranking their own correct
   embedding nearest — the drift accumulates past a full patch by the far edge.
4. Anchor choice is not a detail: the nine crops differ by up to ~44% relative RMS, and **centre**
   is the nearest honest crop to what terratorch currently does.

    python scripts/p133_posemb_geometry.py --out docs/research/p133-posemb-geometry.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

#: TerraMind's DEM tokenizer: ViT-B width, patch 16, trained at 256 px -> a 16x16 native grid.
EMBED_DIM = 768
NATIVE_GRID = 16
#: DEMAE's primary regime: 224 px at patch 16 (ADR-0001/0002) -> 14x14, and 196 tokens.
TARGET_GRID = 14
N_TOKENS = TARGET_GRID**2
#: ADR-0011's decision: the centre anchor, `pos_emb[:, :, 1:15, 1:15]`.
CENTRE_ANCHOR = (1, 1)


def build_2d_sincos_posemb(h: int, w: int, embed_dim: int = EMBED_DIM, temperature: float = 10000.0) -> Tensor:
    """terratorch's own builder, in float64. Returns ``(1, embed_dim, h, w)``.

    The grid comes from ``torch.arange`` — *absolute* indices, not normalized coordinates. That
    single fact is what makes claim 1 hold: the value at index ``(i, j)`` does not depend on
    ``h`` or ``w``, so a smaller grid is a corner-anchored crop of a larger one.
    """
    grid_w, grid_h = torch.meshgrid(
        torch.arange(w, dtype=torch.float64), torch.arange(h, dtype=torch.float64), indexing="ij"
    )
    if embed_dim % 4:
        raise ValueError(f"embed_dim must be divisible by 4 for a 2D sin-cos embedding, got {embed_dim}")
    pos_dim = embed_dim // 4
    omega = 1.0 / (temperature ** (torch.arange(pos_dim, dtype=torch.float64) / pos_dim))
    out_w = torch.einsum("m,d->md", grid_w.flatten(), omega)
    out_h = torch.einsum("m,d->md", grid_h.flatten(), omega)
    pos_emb = torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None]
    return pos_emb.reshape(1, h, w, embed_dim).permute(0, 3, 1, 2).contiguous()


def as_tokens(pos_emb: Tensor) -> Tensor:
    """``(1, d, h, w)`` -> ``(h*w, d)``, one row per token."""
    return pos_emb.flatten(2).squeeze(0).T


def bicubic_resize(pos_emb: Tensor, grid: int = TARGET_GRID) -> Tensor:
    """Exactly what `ViTEncoder.forward` does on a 224 px pass."""
    return F.interpolate(pos_emb, size=(grid, grid), mode="bicubic", align_corners=False)


def relative_rms(actual: Tensor, reference: Tensor) -> float:
    """RMS of the error, as a fraction of the reference's own RMS."""
    return float((actual - reference).pow(2).mean().sqrt() / reference.pow(2).mean().sqrt())


def identity_retained(actual: Tensor, reference: Tensor) -> int:
    """How many tokens still rank their *own* reference embedding nearest by cosine.

    The headline number, and the one that shows why "0.97 cosine" is not reassuring: adjacent
    positions sit at ~0.986 cosine, so an error of that size is *inside* the gap between
    neighbouring patches, and a token can end up nearer its neighbour's embedding than its own.
    """
    sim = F.cosine_similarity(as_tokens(actual)[:, None, :], as_tokens(reference)[None, :, :], dim=2)
    return int((sim.argmax(1) == torch.arange(reference.shape[-1] ** 2)).sum())


def measure() -> dict:
    native = build_2d_sincos_posemb(NATIVE_GRID, NATIVE_GRID)
    bicubic = bicubic_resize(native)
    top_left = native[:, :, :TARGET_GRID, :TARGET_GRID]
    centre = native[:, :, 1 : 1 + TARGET_GRID, 1 : 1 + TARGET_GRID]

    # Claim 2: align_corners=False maps target index i to (i + 0.5) * 16/14 - 0.5 in the source.
    sample_idx = (torch.arange(TARGET_GRID, dtype=torch.float64) + 0.5) * NATIVE_GRID / TARGET_GRID - 0.5

    distinct = F.cosine_similarity(as_tokens(centre)[:, None, :], as_tokens(centre)[None, :, :], dim=2)
    off_diagonal = distinct[~torch.eye(N_TOKENS, dtype=torch.bool)]

    anchors = {
        f"{r},{c}": {
            "rel_rms_vs_top_left": relative_rms(native[:, :, r : r + TARGET_GRID, c : c + TARGET_GRID], top_left),
            "rel_rms_vs_bicubic": relative_rms(
                bicubic, native[:, :, r : r + TARGET_GRID, c : c + TARGET_GRID]
            ),
            "identity_retained": identity_retained(
                bicubic, native[:, :, r : r + TARGET_GRID, c : c + TARGET_GRID]
            ),
        }
        for r in range(NATIVE_GRID - TARGET_GRID + 1)
        for c in range(NATIVE_GRID - TARGET_GRID + 1)
    }

    return {
        "native_grid": [NATIVE_GRID, NATIVE_GRID],
        "target_grid": [TARGET_GRID, TARGET_GRID],
        "embed_dim": EMBED_DIM,
        # Claim 1 — the crop *is* the re-derivation.
        "rederived_minus_top_left_crop_max_abs": float((build_2d_sincos_posemb(TARGET_GRID, TARGET_GRID) - top_left).abs().max()),
        # Claim 2 — the resize stretches index spacing.
        "bicubic_source_index_first": float(sample_idx[0]),
        "bicubic_source_index_last": float(sample_idx[-1]),
        "bicubic_index_spacing": float(sample_idx[1] - sample_idx[0]),
        "crop_index_spacing": 1.0,
        # Claim 3 — what that costs, against ADR-0011's chosen anchor.
        "bicubic_vs_centre_rel_rms": relative_rms(bicubic, centre),
        "bicubic_vs_top_left_rel_rms": relative_rms(bicubic, top_left),
        "bicubic_vs_top_left_identity_retained": identity_retained(bicubic, top_left),
        "bicubic_vs_centre_identity_retained": identity_retained(bicubic, centre),
        "n_tokens": N_TOKENS,
        # The noise floor that makes claim 3 legible: how far apart *correct* neighbours already are.
        "distinct_token_cosine_min": float(off_diagonal.min()),
        "distinct_token_cosine_max": float(off_diagonal.max()),
        # Claim 4 — the anchor is a decision, not a detail.
        "anchors": anchors,
        "adr_0011_anchor": list(CENTRE_ANCHOR),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the measurements here as JSON")
    args = parser.parse_args()

    result = measure()
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
