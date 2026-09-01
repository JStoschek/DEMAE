"""The MAE model: a ViT encoder, a shallow decoder, and the masking that sits between them (#90).

This is [plan](../../docs/phase1-pretraining-plan.md) §1's "minimal reimplementation on timm
blocks", and it is deliberately a *port of the spec, not of the code*: `facebookresearch/mae`
supplies every number here, but it is CC-BY-NC and frozen on `timm==0.3.2`, which is a poor
base for an open-weights release. What is reused instead is timm's Apache-2.0
:class:`~timm.layers.PatchEmbed` and :class:`~timm.models.vision_transformer.Block` — the
latter already routing through :func:`torch.nn.functional.scaled_dot_product_attention` (§7.2).

## The three single-channel changes, and nothing else (§1)

Elevation is one channel, and the whole delta from stock MAE is: ``in_chans=1`` in the patch
embed, a prediction width of 16 x 16 x 1 = 256 per patch, and — over in the loss (#92) — per-
patch target statistics over 256 values instead of 768. Everything else is the reference
architecture, which is the point: deviations from a proven recipe should be arguable one at a
time, and none of these three is a choice.

## Masking lives here, not in the loader

ADR-0003 assumes it and §3 states it: the model masks, after augmentation, per sample, with a
fresh draw every step. Two properties are load-bearing and neither is visible in a loss curve:

- **The count is fixed at construction**, not recomputed per batch from a ratio. 147 of 196
  tokens, always, so every tensor in this file has a shape that `torch.compile(dynamic=False)`
  can burn in (§7.3, and the #95 precondition). The ratio is still a config knob — R2 (#83)
  ablates 0.90 — but it moves the *constant*, at build time.
- **The draw varies per sample and per step.** A mask that repeated would train perfectly well
  and teach the model 49 fixed positions. :meth:`MaskedAutoencoder.forward` takes an optional
  :class:`torch.Generator` so §10.1's val loss can pin the mask and stay comparable across
  epochs; left ``None`` it draws from the ambient RNG, which is what a training step wants.

## What comes back

:class:`MaeOutput` — the per-patch predictions, the binary mask (1 = hidden, MAE's own
convention), and the restore indices. The loss (#92) needs all three: the mask selects which
patches count, and ``ids_restore`` is what put the predictions back in patch order in the
first place. Nothing here computes a loss; a model that reconstructs and a loss that weights
ocean pixels are separable concerns and #92 owns the second.

## Positional encoding

Fixed 2D sin-cos (§2.3), re-added at the decoder input as well as the encoder's — MAE's
default, and Scale-MAE found the decoder's mattered. Held as **non-persistent buffers**: they
are a pure function of ``img_px``, ``patch_px`` and the widths, so a checkpoint that carried
them could only ever disagree with the config it was loaded under. GSD-aware PE stays Phase 4;
at a single 30 m GSD it has nothing to condition on.

One thing in :func:`sincos2d_pos_embed` looks like a mistake and is not — the halves run
column-then-row, which is MAE's own order and not the obvious one. Its docstring has the whole
argument.

**Corrected 2026-07-29 (#102).** That default used to be justified here by "§8's continual arm
loads weights that learned that ordering", and for the checkpoint §8 now names, *that is false*:
mmselfsup's fixed sin-cos embedding is the row/column transpose of MAE's
([P102](../../docs/research/p102-mmselfsup-mae-layout.md) §4). The ordering is therefore an
:class:`~pretrain.config.Architecture` knob, MAE's remains the default for every scratch run, and
R4 sets the other. The original reasoning — that from scratch the two orders are a harmless
relabelling *and* that a warm start makes them anything but — was right; only the direction was
wrong, and no test could have caught it because the claim was about a file nobody had read.
"""

from __future__ import annotations

from functools import partial
from typing import Final, NamedTuple

import torch
from timm.layers import PatchEmbed
from timm.models.vision_transformer import Block
from torch import Tensor, nn

from .config import Architecture, RunConfig, masked_token_count

#: MAE's Transformer width ratio, unchanged in every model in the lineage. Not a config knob:
#: §2 fixes depth, width and heads and is silent here because nobody has ever moved it.
MLP_RATIO: Final = 4.0

#: MAE's LayerNorm epsilon (its `partial(nn.LayerNorm, eps=1e-6)`), which is *not* torch's
#: 1e-5 default. Small, but it is a recipe number, so it is written down rather than inherited.
NORM_EPS: Final = 1e-6

#: Everything the decoder owns is named ``decoder_*`` — MAE's own convention — with exactly one
#: exception: :attr:`~MaskedAutoencoder.mask_token`, which keeps the reference name because it
#: is the one decoder tensor an outside MAE implementation looks for by key.
DECODER_PREFIX: Final = "decoder_"
_MASK_TOKEN: Final = "mask_token"


def is_decoder_parameter(name: str) -> bool:
    """Whether a parameter name belongs to the decoder half.

    §14 releases **encoder-only** weights and throws the decoder away, so this predicate is
    the definition of "encoder-only" — used by :meth:`MaskedAutoencoder.parameter_counts` here
    and by whatever writes the release artifact (#93). One function rather than a prefix test
    at each call site, because ``mask_token`` is the exception that a prefix test gets wrong,
    quietly, in the direction of shipping a decoder tensor as part of the encoder.
    """
    return name.startswith(DECODER_PREFIX) or name == _MASK_TOKEN


class MaeOutput(NamedTuple):
    """One forward pass. A :class:`~typing.NamedTuple` so it survives `torch.compile`."""

    #: Reconstructed patches, in patch order: ``(N, n_patches, patch_dim)``. Every patch is
    #: predicted; #92 selects the masked ones, because §4 computes the loss on those only.
    pred: Tensor
    #: ``(N, n_patches)``, 1 where the patch was hidden from the encoder and 0 where it was
    #: visible — MAE's convention, and the sign #92's loss mask reads.
    mask: Tensor
    #: ``(N, n_patches)``, the permutation that undoes the shuffle: gathering a
    #: ``[visible..., hidden...]`` sequence by it returns the sequence to patch order.
    ids_restore: Tensor


#: The two legal axis orderings for :func:`sincos2d_pos_embed`, keyed by the value
#: :attr:`Architecture.pos_encoding <pretrain.config.Architecture>` carries. Both put the same
#: functions at the same frequencies on the two axes; they differ only in which half of the
#: width holds which axis, which makes them each other's transpose over the patch grid.
AXIS_ORDERS: Final = {"sincos2d": "col_row", "sincos2d_rowfirst": "row_col"}


def sincos2d_pos_embed(dim: int, grid_px: int, *, axis_order: str = "col_row") -> Tensor:
    """Fixed 2D sin-cos positional embeddings for a CLS token plus a grid (§2.3).

    Returns ``(1, 1 + grid_px**2, dim)``: a leading row of **zeros for the CLS token**, which
    has no position — giving it one is a quiet way to make the pooled feature depend on where
    the grid starts — then one row per patch in the row-major order :meth:`patchify` cuts in.

    Half the width encodes one axis and half the other, each half the usual
    sin/cos-over-geometric-frequencies pair, so ``dim`` must be divisible by 4.

    ## Which axis goes in the first half, and why it is a parameter

    ``axis_order="col_row"`` — the default — puts the **column** in the first half. That is
    MAE's own order, produced by a `np.meshgrid(w, h)` whose default ``'xy'`` indexing makes
    its first argument vary along the columns, a detail its own variable names (`emb_h` built
    from `grid[0]`) actively obscure. `test_the_positional_encoding_matches_maes_own_construction`
    pins it.

    ``axis_order="row_col"`` is the transpose, and it exists because **the checkpoint §8's
    continual arm warm-starts from does not use MAE's order**. mmselfsup's
    `build_2d_sincos_position_embedding` calls `torch.meshgrid`, whose default is ``'ij'`` and
    not ``'xy'``, so its first half — named `out_w` — carries the *row*
    ([P102](../../docs/research/p102-mmselfsup-mae-layout.md) §4, verified numerically).

    From scratch the two orders are a harmless relabelling that no test would surface. Warm
    starting, they are not: the weights learned one of them, and handing them the other makes
    the encoder read every tile as its own transpose. So the order is chosen by config rather
    than fixed here, and :func:`pretrain.init.load_mmselfsup_mae_encoder` asserts the model's
    buffer against the checkpoint's own `pos_embed` tensor before it copies any weight.
    """
    if dim % 4:
        raise ValueError(f"sin-cos 2D needs dim divisible by 4, got {dim}")
    # Read off `AXIS_ORDERS` rather than spelled again: the legal set has one home, so the config
    # vocabulary, the mapping and this check cannot drift into three slightly different lists.
    if axis_order not in set(AXIS_ORDERS.values()):
        raise ValueError(
            f"unknown axis_order {axis_order!r} ({' | '.join(sorted(set(AXIS_ORDERS.values())))})"
        )
    # One quarter of the width per (axis, function) pair: sin and cos of the first axis, then
    # sin and cos of the second.
    omega = 1.0 / 10_000 ** (torch.arange(dim // 4, dtype=torch.float64) / (dim // 4))
    coords = torch.arange(grid_px, dtype=torch.float64)
    row, col = torch.meshgrid(coords, coords, indexing="ij")
    axes = (col, row) if axis_order == "col_row" else (row, col)
    halves = [
        torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)
        for angle in (torch.outer(axis.reshape(-1), omega) for axis in axes)
    ]
    grid = torch.cat(halves, dim=1).to(torch.float32)  # (grid_px**2, dim), row-major
    return torch.cat([torch.zeros(1, dim), grid], dim=0).unsqueeze(0)


def random_masking(
    x: Tensor, n_keep: int, *, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Hide all but ``n_keep`` tokens per sample, uniformly at random (§3).

    Returns the kept tokens ``(N, n_keep, D)``, the binary mask ``(N, L)`` in patch order, and
    the restore permutation. The mask and the gather are two views of one draw, so they cannot
    disagree about which tokens survived.

    The mechanism is MAE's argsort-of-noise, and it is worth naming why rather than sampling
    indices directly: one ``rand`` plus two ``argsort`` calls is a fixed amount of work on a
    fixed shape, per sample, with no rejection loop and no data-dependent control flow —
    which is what keeps the whole model traceable under `torch.compile(dynamic=False)`.

    ``generator`` must live on ``x``'s device. Left ``None`` the draw comes from the ambient
    RNG, which is what a training step wants; §10.1's val pass passes a seeded one so its loss
    is comparable across epochs.
    """
    n, length, dim = x.shape
    if not 0 < n_keep <= length:
        raise ValueError(f"n_keep {n_keep} must lie in (0, {length}]")

    noise = torch.rand(n, length, device=x.device, generator=generator)
    ids_shuffle = torch.argsort(noise, dim=1)  # ascending noise: the first n_keep survive
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :n_keep]
    kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, dim))

    # Build the mask in shuffled order — where "the first n_keep" is a slice — then unshuffle
    # it, so it comes back keyed by patch position like everything the caller sees. fp32
    # regardless of `x`: the mask is bookkeeping, not an activation, and #92 multiplies it
    # against a loss §4 computes in fp32 whatever the autocast dtype is.
    mask = torch.ones(n, length, device=x.device, dtype=torch.float32)
    mask[:, :n_keep] = 0
    return kept, torch.gather(mask, 1, ids_restore), ids_restore


def keep_masking(x: Tensor, keep: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """:func:`random_masking`'s sibling for a keep-set that is **read** rather than drawn.

    Returns the same triple in the same conventions — the kept tokens ``(N, n_keep, D)``, the
    binary mask ``(N, L)`` in patch order with 1 where a token was hidden, and the restore
    permutation — so a caller can hand either function's output to
    :meth:`MaskedAutoencoder.forward_decoder` without knowing which drew the mask.

    Here rather than in the benchmark because it is the model's own question: *this* MAE's
    ``ids_restore`` convention is what a keep-set has to be turned into, and a second
    implementation of the unshuffle living next to the caller is one that can drift from
    :meth:`~MaskedAutoencoder.forward_decoder`'s gather without either one changing.

    §4.3's zero-shot arm is what wants it: the patches a void touches are dropped, so the mask
    is a fact about the void and not a draw. Two consequences follow, and both are refusals
    below rather than surprises later:

    * **The keep count must be uniform across the batch.** One gather has one width, and a
      ragged keep-set would silently truncate every row to the narrowest. A void-driven mask is
      per-tile ragged by nature, so the caller batches by keep count or runs one tile at a time.
    * **A row must keep at least one token.** A void touching every patch leaves the encoder
      nothing to condition on, and what comes back is the decoder's unconditional prior — a
      well-shaped, plausible tile that is not a reconstruction of anything.

    The kept tokens come out in **ascending patch order**, which is the property that makes this
    agree with :func:`random_masking` on the one keep-set they can both produce.
    """
    if keep.dtype != torch.bool:
        raise ValueError(f"keep is a boolean keep-set, not a {keep.dtype} tensor")
    if keep.shape != x.shape[:2]:
        raise ValueError(f"keep {tuple(keep.shape)} does not index tokens {tuple(x.shape[:2])}")

    # Onto `x`'s device before anything is indexed: the keep-set is built from a numpy void mask
    # on the CPU and the model may be on mps or cuda, and a `gather` whose indices live on the
    # other device raises halfway through a pass rather than at its edge.
    keep = keep.to(x.device)
    counts = keep.sum(dim=1)
    n_keep = int(counts[0])
    ragged = (counts != counts[0]).nonzero()
    if ragged.numel():
        other = int(ragged[0, 0])
        raise ValueError(
            f"row 0 keeps {n_keep} tokens and row {other} keeps {int(counts[other])}: one "
            f"gather has one width, so a batch is grouped by keep count (or run one at a time)"
        )
    if n_keep == 0:
        raise ValueError(
            "every token is hidden, so the encoder has no visible token to condition on and "
            "the decoder would return its unconditional prior rather than a reconstruction"
        )

    # `argsort` on the *descending* keep flag puts the kept indices first, and `stable=True` is
    # what holds each half in ascending patch order — which is the ordering `random_masking`
    # produces only by accident of the draw and this function promises.
    ids_shuffle = torch.argsort(~keep, dim=1, stable=True)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :n_keep]
    kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]))
    # fp32 regardless of `x`'s dtype, matching `random_masking`: the mask is bookkeeping, not an
    # activation, and every consumer multiplies it against a loss §4 computes in fp32.
    return kept, (~keep).to(dtype=torch.float32), ids_restore


class MaskedAutoencoder(nn.Module):
    """ViT encoder over the visible tokens, shallow decoder over all of them (§2).

    Build it with :meth:`from_config`: the constructor takes loose numbers so the pieces stay
    testable, but every run in Phase 1 is a :class:`~pretrain.config.RunConfig` and a recipe
    number typed at a call site is a recipe number that can disagree with the one recorded in
    the checkpoint.
    """

    def __init__(
        self,
        arch: Architecture,
        *,
        mask_ratio: float,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        fused_attention: bool = True,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=NORM_EPS)
        #: The §2 shape, held whole rather than unpacked into ten attributes. It is already the
        #: type that owns these numbers *and* validates them — heads dividing dims, the patch
        #: dividing the tile, ``in_chans == 1`` — so taking it means there is no way to build a
        #: model whose shape the config would have refused, and no second copy of 384 anywhere.
        self.arch = arch
        encoder_dim, decoder_dim = arch.encoder_dim, arch.decoder_dim

        # --- encoder (§2.1). `in_chans=1` is the first of §1's three changes.
        self.patch_embed = PatchEmbed(
            img_size=arch.img_px,
            patch_size=arch.patch_px,
            in_chans=arch.in_chans,
            embed_dim=encoder_dim,
        )
        self.n_masked = masked_token_count(self.n_patches, mask_ratio)
        self.n_visible = self.n_patches - self.n_masked
        if not 0 < self.n_visible < self.n_patches:
            raise ValueError(
                f"mask ratio {mask_ratio} hides {self.n_masked} of {self.n_patches} tokens, "
                f"leaving {self.n_visible} for the encoder"
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        #: Which half of the width carries which axis. A config value rather than a constant
        #: because §8's warm start loads weights that learned mmselfsup's ordering, not MAE's
        #: (P102 §4) — and because our positional buffers are *derived from the config*, so a
        #: §11 resume on a fresh box has to be able to rebuild the same one.
        axis_order = AXIS_ORDERS[arch.pos_encoding]
        self.register_buffer(
            "pos_embed",
            sincos2d_pos_embed(encoder_dim, self.grid_px, axis_order=axis_order),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            Block(
                dim=encoder_dim,
                num_heads=arch.encoder_heads,
                mlp_ratio=MLP_RATIO,
                qkv_bias=True,  # ViT/MAE standard; timm's Block defaults it off
                # `proj_drop` only, and no `attn_drop`: MAE's single `drop_rate` never reaches
                # attention. §6 pins both to 0 in pretraining, so this is latent either way —
                # but a knob that quietly does more than the recipe says is how a "reproduction"
                # stops being one.
                proj_drop=dropout,
                drop_path=drop_path,
                norm_layer=norm_layer,
            )
            for _ in range(arch.encoder_depth)
        )
        self.norm = norm_layer(encoder_dim)

        # --- decoder (§2.2): shallow, narrow, and thrown away after pretraining.
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        #: One learned vector standing in for every hidden patch. What the decoder has to work
        #: with at a masked position is this plus that position's PE — which is why §2.2 cares
        #: that the decoder PE is there at all.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.register_buffer(
            "decoder_pos_embed",
            # The same ordering as the encoder's, always: the decoder is fresh so no weight
            # constrains it, but a model whose two halves disagreed about which axis is which
            # would be reconstructing patch (i, j) from position (j, i).
            sincos2d_pos_embed(decoder_dim, self.grid_px, axis_order=axis_order),
            persistent=False,
        )
        self.decoder_blocks = nn.ModuleList(
            # `drop_path` is deliberately encoder-only: stochastic depth regularizes the
            # representation being learned, and the decoder is thrown away. That is MAE's own
            # arrangement, and §6 pins both to 0 in pretraining anyway.
            Block(
                dim=decoder_dim,
                num_heads=arch.decoder_heads,
                mlp_ratio=MLP_RATIO,
                qkv_bias=True,
                proj_drop=dropout,
                norm_layer=norm_layer,
            )
            for _ in range(arch.decoder_depth)
        )
        self.decoder_norm = norm_layer(decoder_dim)
        #: §1's second single-channel change: 16 x 16 x 1 = 256 outputs per patch, not 768.
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim, bias=True)

        if not fused_attention:  # §7.2's `math` escape hatch, for debugging attention itself
            for block in (*self.blocks, *self.decoder_blocks):
                block.attn.fused_attn = False

        self._init_weights()

    # --- construction

    @classmethod
    def from_config(cls, cfg: RunConfig) -> MaskedAutoencoder:
        """The only way a run builds a model: every number below comes off the config.

        Initialization is seeded from :attr:`Runtime.seed <pretrain.config.Runtime>` (§6:
        "fixed seed for model init and the DataLoader generator") inside a
        :func:`torch.random.fork_rng`, so two models from one config are bit-identical
        *without* construction consuming from the ambient stream. That matters more than it
        looks: if it did, adding a decoder block would silently reorder the training corpus.
        """
        opt = cfg.optimization
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.runtime.seed)
            return cls(
                cfg.architecture,
                mask_ratio=cfg.masking.ratio,
                dropout=opt.dropout,
                drop_path=opt.drop_path,
                fused_attention=cfg.runtime.attention == "sdpa",
            )

    def _init_weights(self) -> None:
        """MAE's own initialization: xavier-uniform throughout, 0.02-normal on the two tokens.

        The patch embed is initialized as the linear projection it is — over the flattened
        ``(dim, in_chans * patch_px**2)`` kernel — rather than as a conv, which is what keeps
        the fan-in right at one input channel instead of three.
        """
        w = self.patch_embed.proj.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)
        for token in (self.cls_token, self.mask_token):
            nn.init.normal_(token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # --- shapes, every one of them read off `self.arch` rather than re-derived here

    @property
    def img_px(self) -> int:
        """Tile edge in pixels — 224."""
        return self.arch.img_px

    @property
    def patch_px(self) -> int:
        """Patch edge in pixels — 16."""
        return self.arch.patch_px

    @property
    def in_chans(self) -> int:
        """Input channels — 1, and §1 is why."""
        return self.arch.in_chans

    @property
    def decoder_dim(self) -> int:
        """Decoder width — 256 for ViT-S, 512 for ViT-B (§2.2)."""
        return self.arch.decoder_dim

    @property
    def grid_px(self) -> int:
        """Patches per side — 14 at 224/16."""
        return self.img_px // self.patch_px

    @property
    def n_patches(self) -> int:
        """Tokens per tile — 196."""
        return self.arch.n_patches

    @property
    def patch_dim(self) -> int:
        """Values per patch — 256 at 16 x 16 x 1, the width the decoder predicts."""
        return self.arch.patch_dim

    def parameter_counts(self) -> dict[str, int]:
        """Learnable parameters, split encoder / decoder.

        The split is §14's release boundary — Phase 2 ships encoder-only weights — and it is
        the number §2.1's "22M / 86M" refers to, so it is worth being able to read off rather
        than infer. Buffers (both positional encodings) are excluded because they are not
        learned and not checkpointed.
        """
        decoder = sum(
            p.numel() for n, p in self.named_parameters() if is_decoder_parameter(n)
        )
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": total - decoder, "decoder": decoder, "total": total}

    def _check_tiles(self, imgs: Tensor) -> None:
        """Refuse anything that is not a batch of this model's tiles.

        Checked here rather than left to the first layer that happens to notice: with
        ``in_chans=1`` a 3-channel batch dies inside a conv and a mis-sized one inside timm's
        own assert, and neither message says which model rejected what. A Python-level shape
        check on static shapes costs nothing and is traced out by `torch.compile`.
        """
        expected = (self.in_chans, self.img_px, self.img_px)
        if imgs.ndim != 4 or tuple(imgs.shape[1:]) != expected:
            raise ValueError(
                f"expected tiles of shape (N, {self.in_chans}, {self.img_px}, {self.img_px}), "
                f"got {tuple(imgs.shape)}"
            )

    def patchify(self, imgs: Tensor) -> Tensor:
        """``(N, C, H, W)`` -> ``(N, n_patches, patch_dim)``, row-major over the patch grid.

        The layout the decoder predicts in and #92's targets are cut in, so the two agree by
        construction rather than by convention.
        """
        self._check_tiles(imgs)
        p, c, grid, n = self.patch_px, self.in_chans, self.grid_px, imgs.shape[0]
        x = imgs.reshape(n, c, grid, p, grid, p)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(n, self.n_patches, self.patch_dim)

    def unpatchify(self, patches: Tensor) -> Tensor:
        """The inverse of :meth:`patchify` — what renders a §10.2 reconstruction grid."""
        p, c, grid, n = self.patch_px, self.in_chans, self.grid_px, patches.shape[0]
        x = patches.reshape(n, grid, grid, p, p, c)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(n, c, self.img_px, self.img_px)

    # --- the forward pass

    def embed_tokens(self, imgs: Tensor) -> Tensor:
        """``(N, n_patches, encoder_dim)`` — patches projected and given their position.

        The one place a token acquires a position, shared by the masked and unmasked passes
        below. Two copies of "embed, then add the PE" is how one of them ends up adding
        ``pos_embed`` including the CLS row, and only the reconstructions would show it.
        """
        self._check_tiles(imgs)
        return self.patch_embed(imgs) + self.pos_embed[:, 1:, :]

    def _transform(self, tokens: Tensor) -> Tensor:
        """CLS in front, every block, final norm: ``(N, 1 + tokens, encoder_dim)``."""
        cls_token = (self.cls_token + self.pos_embed[:, :1, :]).expand(tokens.shape[0], -1, -1)
        x = torch.cat([cls_token, tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward_encoder(
        self, imgs: Tensor, *, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed, mask, and encode the survivors: ``(N, 1 + n_visible, encoder_dim)``.

        Masking happens *after* the positional embeddings are added and *before* the blocks
        run, which is the whole efficiency argument: the encoder is 49 tokens wide, not 196,
        and each surviving token still knows where it came from.

        Public so the two halves can be driven apart, but note what it is *not*: it always
        masks. Callers wanting frozen features over the whole tile want :meth:`encode`.
        """
        tokens, mask, ids_restore = random_masking(
            self.embed_tokens(imgs), self.n_visible, generator=generator
        )
        return self._transform(tokens), mask, ids_restore

    def encode_visible(
        self, imgs: Tensor, keep: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """:meth:`forward_encoder` over a keep-set the caller supplies: no draw, no ratio.

        The third of this class's three encoder paths, and the axis they differ on is *which
        tokens the encoder sees*: :meth:`encode` sees all of them, :meth:`forward_encoder` sees
        a random :attr:`n_visible`, and this sees exactly what ``keep`` names. Everything else —
        the embedding, the CLS token, the blocks, the final norm — is the one shared path, which
        is what lets the triple it returns go straight to :meth:`forward_decoder`.

        ``keep`` is ``(N, n_patches)`` boolean, 1 where the encoder should see the patch. §4.3's
        zero-shot inpainting builds it from the patches a void touches; see :func:`keep_masking`
        for the two refusals that come with a mask nobody drew.
        """
        tokens, mask, ids_restore = keep_masking(self.embed_tokens(imgs), keep)
        return self._transform(tokens), mask, ids_restore

    def encode(self, imgs: Tensor) -> Tensor:
        """Every token, nothing hidden: ``(N, 1 + n_patches, encoder_dim)``.

        The frozen-feature view of a tile, and the *only* path here with no RNG in it at all —
        which is what makes it usable as a measurement. #102's behavioural check reads it to
        ask whether a warm-started encoder computes structure or garbage, and §10.3's kNN probe
        wants the same tensor (the probe harness itself is #99's; this is just the pass).
        """
        return self._transform(self.embed_tokens(imgs))

    def forward_decoder(self, latent: Tensor, ids_restore: Tensor) -> Tensor:
        """Refill the hidden positions and reconstruct every patch: ``(N, n_patches, 256)``.

        The mask token is broadcast into the gaps and the sequence unshuffled by
        ``ids_restore``, which is the step that puts each encoded token back under its own
        position. Decoder positional embeddings are added *after* that restore — before it the
        sequence is in shuffled order and the PE would be attached to the wrong patches.
        """
        x = self.decoder_embed(latent)
        fill = self.mask_token.expand(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1)
        rest = torch.cat([x[:, 1:, :], fill], dim=1)  # CLS held out of the shuffle
        rest = torch.gather(
            rest, 1, ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        )
        x = torch.cat([x[:, :1, :], rest], dim=1) + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x = block(x)
        return self.decoder_pred(self.decoder_norm(x))[:, 1:, :]  # drop CLS: it predicts nothing

    def forward(self, imgs: Tensor, *, generator: torch.Generator | None = None) -> MaeOutput:
        """Tiles in, per-patch predictions + the mask that hid them out."""
        latent, mask, ids_restore = self.forward_encoder(imgs, generator=generator)
        return MaeOutput(self.forward_decoder(latent, ids_restore), mask, ids_restore)

    def extra_repr(self) -> str:
        counts = self.parameter_counts()
        return (
            f"patches={self.n_patches}, visible={self.n_visible}, masked={self.n_masked}, "
            f"params={counts['encoder'] / 1e6:.1f}M encoder + "
            f"{counts['decoder'] / 1e6:.1f}M decoder"
        )
