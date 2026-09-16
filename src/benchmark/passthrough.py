"""A9 — the TerraMind DEM tokenizer run as published: encode, FSQ, diffusion decode (#168).

§1's reference row, and the only arm in the table that is not trying to be good at anything.
Every other arm either interpolates (A1), reads a pretrained decoder (A2), or puts something
trainable between an encoder and the output (A3–A7). A9 puts the A1 fill through TerraMind's own
autoencoder and prints what comes back, so the table carries a measurement of **what the 5-d FSQ
bottleneck costs** rather than an argument about it. A row that came back worse than A1 would be
this arm working.

## Why this is a module and not four lines inside a filler

Three of the four things it does are refusals, and all three are silent failures otherwise:

1. **The output raster.** :meth:`DiVAE.decode_quant` sizes its output from the *decoder's* 256 px
   training resolution and takes no hint from the 14x14 latent it is handed, so a 224 px tile
   comes back 256 px and nothing raises ([#163](https://github.com/JStoschek/DEMAE/issues/163)).
   Every §5 metric would then compute against a resampled tile and print. So the raster is read
   off the tile and the decode is *checked* against it — :func:`decode`.
2. **The noise.** The decode is stochastic unless a ``generator`` is passed: #163 measured
   unseeded decodes of one latent differing by 4.8e-3 in TerraMind's normalized units, which is
   4.6 m at its own 951.272 std. A9 trains nothing and scores in a single pass, so there are no
   seeds to average over and unseeded means **unreproducible**, not merely noisy —
   :func:`generator_for`.
3. **The encoder.** The whole tokenizer comes down through terratorch's registry, whose encoder
   half carries the native 16x16 ``pos_emb`` that [ADR-0011](../../docs/adr/
   0011-terramind-position-embedding.md) replaces with the centre-anchored 14x14 crop. #134
   measured those two 0.9708 apart. So the crop is substituted here as well, and the substitution
   is *verified* against the encoder :class:`benchmark.extract.Extractor` builds for A5 — every
   parameter, every buffer, and the class itself (:func:`assert_same_encoder`). #163 checked the
   same thing by comparing two forward passes and got 0.0; this is the stronger form of that
   claim and a cheaper one, because it needs no tiles and no agreement about dtype, and it fires
   at load rather than on the first batch.

The fourth thing is the arm: normalize, encode, decode, return to metres, and write **inside the
void and nowhere else**. That last clause is not this module's manners — :func:`benchmark.metrics.
score` refuses a fill that differs from the tile outside the void, because a fill that moves
untouched ground moves every metric and no column would say so.

## A9 runs at the reference dtype, and refuses a cell that says otherwise

Every other arm that runs an encoder runs it at :attr:`benchmark.config.Extract.compute_dtype`,
which ADR-0012 §6 permits on the strength of a measured agreement against #134's materiality bar.
A9 does not inherit that permission, because what dominates A9 is not the encoder: it is a 196 M
diffusion decoder that §6 does not govern and that nothing in this plan has measured at reduced
precision. The measurement A9's bill is compared against (#163) is an fp32 measurement, and the
pipeline's own scheduler step forces fp32 regardless. So :func:`load` refuses rather than quietly
running fp32 under a config that says bf16 — a run record that names a dtype the pass did not use
is the failure this whole module is shaped around.

## The weights come off the mounted file, not off the hub

``FULL_MODEL_REGISTRY.build(..., pretrained=True)`` would fetch 1.15 GB from Hugging Face and its
``ckpt_path`` branch loads ``strict=False`` and *logs* what did not match. Both are wrong here for
:meth:`benchmark.extract.Extractor._build_terratorch`'s reason: the file this benchmark scores
against is the one whose md5 ``extractor_spec.json`` pins, and a partial load is a warning where
this contract wants a refusal. So the registry builds the *shape*, and the md5-checked file is
loaded into it strictly — 601 tensors, no missing, no unexpected.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np

from . import arms, extract

#: The registry name terratorch builds the DEM tokenizer's shape from. The weights do not come
#: from here — see this module's docstring — so ``pretrained=False`` is not a fallback.
REGISTRY_MODEL: Final = "terramind_v1_tokenizer_dem"

#: The **noise namespace** every pass-through arm draws in, and deliberately a constant rather
#: than the arm's own name (#168).
#:
#: Two things rest on it. A9's committed rows were scored under this literal, so threading the
#: arm name through :func:`generator_for` would re-seed them and they would stop reproducing.
#: And A9-NOTHRESH is registered as A9's *one-knob* ablation: sharing this namespace makes the
#: two arms start their decodes from the same initial noise for the same batch, so what separates
#: their rows is the sampler and not the draw. The spec lookups in :func:`load` take the arm's own
#: name — an arm asserts the encoder *it* declares — which is why this is no longer spelled `ARM`.
PASS_THROUGH_NOISE_NAMESPACE: Final = "A9"

#: The arm this module was written for, kept for the refusals that quote a name where no arm
#: object is in hand.
ARM: Final = "A9"


class PassThroughError(RuntimeError):
    """The tokenizer returned something A9 cannot score, and would have scored silently."""


# --- The two facts a decode has to be pinned to -------------------------------


def batch_key(voids: Sequence[np.ndarray]) -> str:
    """This batch's identity, as the thing its noise is seeded from.

    The **voids**, not the batch's position in the pass. A counter would be one line shorter and
    would make a batch's noise a fact about how far a pass had got, so a run that was resumed, or
    reordered, or sharded would score the same instance against different noise and no column
    would say so. The void is what distinguishes one instance of a tile from its five siblings
    (§3.2 draws one per bucket per family), so a batch of voids names the batch.
    """
    digest = hashlib.blake2b(digest_size=16)
    for void in voids:
        packed = np.packbits(np.asarray(void, dtype=bool), axis=None).tobytes()
        digest.update(len(packed).to_bytes(8, "big"))
        digest.update(packed)
    return digest.hexdigest()


def generator_for(seed: int, key: str) -> Any:
    """A fresh CPU generator for one batch, seeded from the cell and :func:`batch_key`.

    A **CPU** generator whatever the device, because that is what the pipeline draws with:
    ``PipelineCond.__call__`` calls ``torch.randn(shape, generator=generator)`` with no ``device``
    and moves the result afterwards. Passing a CUDA generator there raises; passing none is #163's
    finding 2 — unseeded decodes of one latent differ by 4.8e-3 in TerraMind's normalized units,
    and A9 has no seeds to average over.

    Derived through BLAKE2b rather than by arithmetic, for the reason :func:`benchmark.config.
    _hash` gives: it is identical in every process and on every box, where Python's own ``hash``
    is salted per-interpreter.

    **What this does and does not buy.** It removes the *sampling* nondeterminism, which is the
    one that is 4.6 m wide: given the same cell, the same committed instance list and the same
    :attr:`benchmark.config.Tokenizer.batch_size`, the decode starts from the same noise every
    time. It is not per-instance — the batch's noise is one row-major ``torch.randn`` over the
    whole batch, so an instance's slice of it depends on how many instances share the batch, which
    is why the batch size is recipe and in the config hash (:class:`benchmark.config.Tokenizer`).
    And it does not make a CUDA pass bit-identical, on :attr:`benchmark.config.Runtime.
    micro_batch`'s standing honesty: kernel selection and float summation order are the card's,
    not the seed's. What is claimed here is that two runs of one cell differ by the card and not
    by the draw.
    """
    import torch

    digest = hashlib.blake2b(
        f"{PASS_THROUGH_NOISE_NAMESPACE}:{seed}:{key}".encode(), digest_size=8
    ).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest, "big") % (2**63 - 1))


def unthresholded_scheduler(model) -> Any:
    """The tokenizer's own DDIM sampler with dynamic thresholding off, and nothing else moved.

    Built with ``from_config`` off ``model.noise_scheduler`` so every other term the checkpoint
    recorded — ``zero_terminal_snr``, the beta schedule, ``prediction_type="sample"``, the
    1,000 training timesteps — is carried rather than retyped. Retyping them is how a "one knob"
    ablation quietly becomes several.

    ``clip_sample`` is turned off **with** ``thresholding``, and it is not the knob that matters:
    diffusers applies dynamic thresholding *instead of* the clamp, so with thresholding on,
    ``clip_sample`` is dead. Measured — turning ``clip_sample`` off alone changes the decode by
    nothing at all, to the digit. It is moved so that the sampler this arm runs cannot clamp by
    either mechanism, rather than leaving a live clamp behind the one that was disabled.

    terratorch's own :class:`DDIMScheduler` subclass, not diffusers': the pipeline calls
    ``set_timesteps(timesteps, mode=...)``, which diffusers' signature does not take, and
    ``from_config`` on the wrong class also drops ``zero_terminal_snr`` with a warning that reads
    like a formatting nit.
    """
    from terratorch.models.backbones.terramind.tokenizer.scheduling import DDIMScheduler

    published = model.noise_scheduler
    if not getattr(published.config, "thresholding", False):
        raise PassThroughError(
            f"the tokenizer's sampler already has `thresholding` off, so the arm that exists to "
            f"turn it off has nothing to turn off. terratorch's registry built this model with "
            f"it on at the version this path is verified against; a build that does not is a "
            f"different inference path and the row would not be A9's one-knob ablation"
        )
    return DDIMScheduler.from_config(published.config, thresholding=False, clip_sample=False)


def to_metres(normalized: np.ndarray, *, mean: float, std: float) -> np.ndarray:
    """Invert TerraMind's affine — the exact inverse of :func:`benchmark.extract.normalize`.

    Takes the pair rather than reading it, so the arithmetic is checkable on a box with no
    terratorch on it while the *values* stay :func:`benchmark.extract.terramind_constants`'
    single answer. ADR-0012 Decision 1: the affine is global and has no per-tile term, which is
    why this inverse is two floats where :func:`benchmark.metrics.to_metres` needs a mask — there
    is no leak to avoid here because there is nothing per-tile to leak.
    """
    return np.asarray(normalized, dtype=np.float64) * float(std) + float(mean)


def composite(a1_fill: np.ndarray, decoded_m: np.ndarray, void: np.ndarray) -> np.ndarray:
    """The decode inside the void, the A1 fill everywhere else. §2.4's convention for A9.

    The tokenizer is an autoencoder over the **whole** tile, so its output differs from the input
    everywhere — including the ground no mask asked A9 to predict. Scoring that would be scoring
    pixels the instance does not own, and :func:`benchmark.metrics.score` refuses it outright. So
    the round trip is read where the void is and discarded where it is not, which is the same
    composition :func:`benchmark.head.apply_residual` performs for the learned arms and the same
    one :func:`benchmark.fill.fill` performs for A1.
    """
    a1_fill = np.asarray(a1_fill, dtype=np.float64)
    decoded_m = np.asarray(decoded_m, dtype=np.float64)
    void = np.asarray(void, dtype=bool)
    if a1_fill.shape != decoded_m.shape or a1_fill.shape != void.shape:
        raise ValueError(
            f"the A1 fill {a1_fill.shape}, the decode {decoded_m.shape} and the void "
            f"{void.shape} are one tile"
        )
    if not np.isfinite(decoded_m[void]).all():
        raise PassThroughError(
            f"the tokenizer decode left {int((~np.isfinite(decoded_m[void])).sum())} of "
            f"{int(void.sum())} void pixels non-finite. Every §5 metric is computed inside the "
            f"void, so this would reach the results parquet as a NaN column rather than as a "
            f"refusal"
        )
    out = a1_fill.copy()
    out[void] = decoded_m[void]
    return out


# --- The pass ------------------------------------------------------------------


@dataclass
class PassThrough:
    """The loaded tokenizer, and the one operation A9 asks of it."""

    model: Any
    device: str
    timesteps: int
    seed: int
    mean: float
    std: float
    #: The diffusion sampler the decode runs under, or ``None`` for the tokenizer's own — the
    #: published one, with `thresholding=True`. Set only by an arm declaring
    #: ``sampler_thresholding=False`` (:attr:`benchmark.arms.Arm.sampler_thresholding`).
    #: ``decode_quant`` falls back to ``model.noise_scheduler`` when this is ``None``, so the
    #: published path is the one that passes nothing rather than the one that rebuilds it.
    scheduler: Any = None

    def decode(self, quant: Any, *, tile_px: int, key: str) -> Any:
        """One batch of latents to one batch of rasters, at the tile's own resolution.

        The ``image_size`` is passed and then the result is checked against it. Passing it is
        #163's finding 1; checking it is the part that survives a terratorch upgrade, since the
        argument is advisory as far as this module can tell — what would go wrong silently is the
        raster, so the raster is what is asserted.
        """
        import torch

        with torch.no_grad():
            decoded = self.model.decode_quant(
                quant,
                timesteps=self.timesteps,
                image_size=tile_px,
                verbose=False,
                generator=generator_for(self.seed, key),
                scheduler=self.scheduler,
            )
        height, width = int(decoded.shape[-2]), int(decoded.shape[-1])
        if (height, width) != (tile_px, tile_px):
            raise PassThroughError(
                f"the decode came back {height}x{width} for a {tile_px}x{tile_px} tile. "
                f"`decode_quant` sizes its output from the decoder's own "
                f"{int(self.model.image_size)} px training resolution unless told otherwise "
                f"(#163), and a resampled tile scores every §5 metric without raising"
            )
        return decoded

    def run(self, composed) -> list[np.ndarray]:
        """One batch of composed tiles to one filled tile each, in metres.

        ``composed`` is :class:`benchmark.train.Composed` — the A1 fill, its int16 rounding, the
        void and ADR-0012's valid mask, built by the same :func:`benchmark.train.compose` every
        learned arm's input goes through. Shared on purpose: A5 and A9 read the same encoder, and
        two compositions would make "the same tile" a claim rather than a fact.
        """
        import torch

        tile_px = int(composed.a1_i16.shape[-1])
        x = extract.normalize(composed.a1_i16, composed.valid, normalization="terramind_v1")
        tensor = torch.from_numpy(x).unsqueeze(1).to(self.device)
        with torch.no_grad():
            quant, _, _ = self.model.encode(tensor)
        decoded = self.decode(quant, tile_px=tile_px, key=batch_key(composed.void))
        values = decoded.float().squeeze(1).cpu().numpy()
        return [
            composite(
                composed.a1_fill[i],
                to_metres(values[i], mean=self.mean, std=self.std),
                composed.void[i],
            )
            for i in range(values.shape[0])
        ]


def assert_same_encoder(tokenizer, contract) -> dict:
    """Is the tokenizer's encoder half the encoder :class:`benchmark.extract.Extractor` builds?

    Every parameter and every buffer, by :func:`torch.equal`, plus the class itself — which is
    what makes this an *identity* rather than an agreement. The two objects reach the same weights
    by different routes (terratorch's ``FULL_MODEL_REGISTRY`` here, its ``vit_b_enc`` there), and
    A5 and A9 are supposed to be reading one encoder: if that stops being true, A9's row is about
    a model A5's row is not, and no column in the table would say so.

    #163 established the claim by comparing two forward passes and measuring 0.0. This is the same
    claim without the forward: it needs no tiles, it is unaffected by which dtype either side would
    compute at, and it fires at load instead of on the first batch. The class check is what a
    parameter comparison alone would miss — two modules can hold identical tensors under identical
    names and multiply them in a different order.
    """
    import torch

    if type(tokenizer) is not type(contract):
        raise PassThroughError(
            f"terratorch's registry built a {type(tokenizer).__name__} where the contract builds "
            f"a {type(contract).__name__}. A5 and A9 read one encoder, so two classes here means "
            f"the two builds have diverged and A9's row would be about a different model"
        )
    left = {**dict(tokenizer.state_dict()), **dict(tokenizer.named_buffers())}
    right = {**dict(contract.state_dict()), **dict(contract.named_buffers())}
    if set(left) != set(right):
        differing = sorted(set(left) ^ set(right))
        raise PassThroughError(
            f"the tokenizer's encoder and the contract's differ by {len(differing)} tensor "
            f"name(s), first {differing[0]!r} — the two builds have diverged"
        )
    for name in sorted(left):
        a, b = left[name], right[name]
        if a.shape != b.shape or not torch.equal(a.detach().cpu(), b.detach().cpu()):
            raise PassThroughError(
                f"the tokenizer's encoder and the contract's differ at {name!r}. They are one "
                f"encoder — ADR-0011's crop is substituted into both — so any difference means "
                f"the substitution did not take on one side, or the two terratorch builds have "
                f"diverged. A9's row would then be about a model A5's row is not"
            )
    return {"tensors": len(left), "class": type(tokenizer).__name__, "identical": True}


def load(cfg, arm: arms.Arm, *, device: str | None = None) -> PassThrough:
    """The whole tokenizer, ADR-0011's `pos_emb` in it, off the md5-checked file on this box.

    Every refusal :meth:`benchmark.extract.Extractor._build_terratorch` makes, this makes too —
    the checkpoint's md5, TerraMind's named constants, the native `pos_emb`'s shape and closed
    form — because it reaches them through the same :class:`~benchmark.extract.Extractor`. What it
    adds is the other 196 M parameters: the contract's builder loads the encoder half alone and
    strictly, and A9 is the one arm that needs the decoder. The contract's encoder is then
    compared to this one and dropped (:func:`assert_same_encoder`); it is not kept, because A9
    reads no features through it and an extractor held past its refusals is an extractor something
    later will be tempted to read a figure off.
    """
    import torch
    from torch import nn

    from . import train

    try:
        from terratorch.registry import FULL_MODEL_REGISTRY
    except ImportError as exc:  # pragma: no cover - environment, not behaviour
        raise ImportError(
            f"{arm.name} runs terratorch's DEM tokenizer, which is not on this box: {exc}. It is "
            f"an extra rather than a core dependency — `{extract.TERRAMIND_EXTRA}`, pinned at "
            f"the version this path is verified on."
        ) from None

    entry = arms.ENCODERS[arm.encoder]
    device = train.resolve_device(cfg) if device is None else device
    spec = extract.load_spec(extract.spec_path(cfg))
    if cfg.extract.compute_dtype != spec.reference_dtype:
        raise PassThroughError(
            f"{arm.name} is asked to compute at {cfg.extract.compute_dtype} and runs at "
            f"{spec.reference_dtype}: ADR-0012 §6's permission to move the compute dtype rests on "
            f"a measured agreement for a *frozen encoder*, and what dominates this arm is the "
            f"196 M-parameter diffusion decoder §6 does not govern and nothing in this plan has "
            f"measured below fp32. #163's bill, which this pass is compared against, is an fp32 "
            f"bill. Run the cell with `--set extract.compute_dtype={spec.reference_dtype}` so the "
            f"run record names the dtype the pass used."
        )
    extractor = extract.Extractor(spec, cfg, device=device)
    # Built first, and deliberately: it is the object that applies §7.1's four refusals, so an
    # unmounted checkpoint, a moved md5 or a retuned affine stops the pass here — before 1.15 GB
    # is read a second time, and on a clock that is still cheap.
    contract_encoder = extractor.encoder(arm.name)
    spec_entry = spec.encoder_for(arm.name)
    path = getattr(cfg.paths, entry.checkpoint_field)
    mean, std = extract.terramind_constants()

    model = FULL_MODEL_REGISTRY.build(REGISTRY_MODEL, pretrained=False)
    state, _ = _read_state(path)
    # Strict, where the registry's own `ckpt_path` branch is not. 601 tensors either match this
    # shape or this is not the file `extractor_spec.json` pins, and "loaded with 40 tensors
    # missing" is a log line on a box nobody is watching.
    model.load_state_dict(state, strict=True)

    native_px = spec_entry.pos_emb["native_grid_px"]
    native = model.encoder.pos_emb.detach()
    expected = (1, spec_entry.width, native_px, native_px)
    if tuple(native.shape) != expected:
        raise PassThroughError(
            f"terratorch's registry built `pos_emb` {tuple(native.shape)} against the declared "
            f"{expected}; ADR-0011's crop has no subject at another shape"
        )
    model.encoder.pos_emb = nn.Parameter(
        extract.derive_pos_emb(native, spec_entry.pos_emb).clone(), requires_grad=False
    )
    if int(model.image_size) != native_px * spec.patch_px:
        raise PassThroughError(
            f"the tokenizer's declared image_size {model.image_size} is not "
            f"{native_px}x{spec.patch_px}; p118 §3's 256 px training resolution is "
            f"what the native grid is a fact about"
        )
    # Both sides now carry ADR-0011's crop, so this is the point at which they either are one
    # encoder or are not. After the substitution rather than before it: what A9 runs is the
    # substituted model, and an identity check on the shipped `pos_emb` would prove a claim about
    # a model neither arm uses.
    assert_same_encoder(model.encoder, contract_encoder)
    model = model.eval().to(device)
    return PassThrough(
        model=model,
        device=device,
        timesteps=cfg.tokenizer.timesteps,
        seed=cfg.seed,
        mean=mean,
        std=std,
        # `None` is the published sampler, and the arm that wants it passes nothing rather than
        # rebuilding it — so A9's decode reaches `decode_quant` exactly as it did before #168.
        scheduler=None if arm.sampler_thresholding else unthresholded_scheduler(model),
    )


def _read_state(path):
    """The checkpoint's tensors, through the loader `pretrain.init` already owns."""
    from pretrain.init import read_state_dict

    return read_state_dict(path)


def parameter_count(model) -> dict[str, int]:
    """What A9 actually runs, counted rather than quoted — §6 item 4's pair for this arm.

    The registry's ``params`` for every other encoder is a reviewed literal, and for this one it
    could not be: A9's total is the 90.2 M encoder plus the 196.1 M diffusion decoder, and no
    source in this plan gives that sum. This is where the literal in
    :data:`benchmark.arms.ENCODERS` came from.
    """
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "decoder": sum(p.numel() for p in model.decoder.parameters()),
    }
