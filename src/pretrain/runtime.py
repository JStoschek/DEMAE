"""Plan §7's kernel levers, and the checks that say whether they actually happened (#95).

:class:`~pretrain.config.Runtime` holds §7 as a set of knobs — bf16, SDPA, compile. This
module is the other half: the code that *applies* two of them and then reports what the
running model is really doing. The split matters because §7's levers share one failure mode —
**they all fail silently and in the same direction**. Attention that fell back to the unfused
path trains the identical model, slower. A `torch.compile` that recompiles every step trains
the identical model, slower than eager. Nothing in a loss curve, a gradient norm or a
checkpoint would ever say so; the only evidence is a throughput number nobody has a baseline
for, on a box rented by the hour.

So every function here answers "what is this model about to do", off the modules themselves,
never off the config that asked for it. :func:`verify_attention` is the pair being compared.

## What compiling is contingent on

§7.3 expects ~1.1-1.4x from `torch.compile(dynamic=False)`, and that number holds only if the
graph is compiled once. Two earlier decisions are what make that true, and both were taken
for this reason:

- **The mask count is fixed at construction** (#90), not recomputed per batch, so the encoder
  runs on ``(N, 49, D)`` at every step of every epoch.
- **A short final batch is dropped** (#93's ``drop_last``, and the short accumulation group
  it drops with it), so ``N`` never moves either.

:class:`CompileWatch` is what turns "and therefore it does not recompile" from an argument
into an observation. It reads Dynamo's own frame counter, so it sees a recompile whatever
caused it — a moving shape, a changed Python attribute, a guard nobody predicted.

## What is deliberately not here

§7.6's two non-levers — gradient checkpointing (unneeded at S/B scale, ~30%) and
`channels_last` (a CNN optimization) — are config knobs pinned to ``False`` and no code. So is
the separate `flash-attn` package: at 49-196 tokens SDPA's win is kernel fusion rather than
memory IO (§7.2), and torch's own implementation is the thing timm blocks already call.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from .config import RunConfig


def eager_module(model) -> nn.Module:
    """The real module behind a possibly-compiled one.

    `torch.compile` returns an ``OptimizedModule`` holding the original at ``_orig_mod``. It
    proxies attribute access, so most things work through it — but *most* is the problem: its
    ``state_dict`` keys are prefixed ``_orig_mod.``, which is how a compiled run silently
    writes checkpoints an eager run cannot read. Everything here that wants the model rather
    than the callable goes through this function, and :class:`~pretrain.train.Trainer` keeps
    the eager module as ``self.model`` for the same reason.
    """
    return getattr(model, "_orig_mod", model)


def attention_blocks(model) -> Iterator[nn.Module]:
    """Every timm attention module in the model, encoder and decoder alike.

    Found by duck-typing on the ``fused_attn`` flag rather than by importing timm's
    :class:`~timm.models.vision_transformer.Attention` and testing the type: the flag is the
    thing §7.2 is about, it is public, and a timm refactor that moved the class would turn a
    type test into an empty iterator — which reports "nothing to check" and passes.
    """
    for module in eager_module(model).modules():
        if hasattr(module, "fused_attn"):
            yield module


def attention_path(model) -> str:
    """``"sdpa"``, ``"math"``, or ``"mixed"`` — which path this model's blocks will take.

    ``mixed`` is not a §7.2 option. It is what a model whose blocks disagree gets called, so
    that state is named rather than averaged away into whichever half was inspected first.

    Read off the modules, so it is what will happen rather than what was configured. #90 sets
    the flag from :attr:`Runtime.attention <pretrain.config.Runtime>` at construction and
    timm sets its own default when nobody asks, and those are two different things to be told.
    """
    flags = {bool(block.fused_attn) for block in attention_blocks(model)}
    if not flags:
        raise ValueError(
            "no attention blocks found: this model does not hold timm blocks, and §7.2's "
            "claim that attention routes through F.scaled_dot_product_attention is unchecked"
        )
    if len(flags) > 1:
        return "mixed"
    return "sdpa" if flags.pop() else "math"


def verify_attention(model, cfg: RunConfig) -> str:
    """Assert that what the model will do is what §7.2 was asked for; return the path.

    The one place the two sources of truth are compared. Called at launch (#95's entrypoint),
    because "SDPA confirmed on the attention path" is a pre-flight item and a run that
    quietly took the math path is a throughput measurement of the wrong thing.
    """
    actual = attention_path(model)
    if actual != cfg.runtime.attention:
        raise ValueError(
            f"attention path is {actual!r} but the config asks for "
            f"{cfg.runtime.attention!r} (§7.2). A model built by "
            "`MaskedAutoencoder.from_config` sets the flag from the config, so this means "
            "something moved `attn.fused_attn` afterwards — or timm's own default did"
        )
    return actual


def compile_model(model: nn.Module, cfg: RunConfig) -> nn.Module:
    """§7.3's `torch.compile(dynamic=False)` over the full model — or the model, untouched.

    Returns the callable a step should forward through. At ``runtime.compile = False`` that is
    the model *itself*, identically: eager is the debugging fallback, and a fallback that went
    through a wrapper would be a third thing to debug rather than a way out.

    ``dynamic`` comes from the config, where it is pinned to ``False`` — the config refuses
    ``compile_dynamic`` and ``compile`` together, because the whole argument for compiling
    here is that the shapes are static (see the module docstring). Passing it explicitly
    rather than hardcoding ``False`` keeps torch's own tri-state default (``None``, meaning
    "recompile as dynamic if a shape ever moves") out of the picture: with ``False`` a moving
    shape produces another static graph, which :class:`CompileWatch` can see.

    **One thing compiling does change, and it is not a bug:** inductor generates its own
    randomness rather than drawing from torch's ambient RNG, so a compiled run and an eager run
    from one seed mask *different tokens*. Both are valid draws of §3's uniform mask and neither
    is more correct — but it does mean §6's "two runs of one config differ only by
    nondeterministic kernels" holds within a compile setting, not across one. The parity that
    matters is the arithmetic, and `test_the_compiled_model_trains_the_same_trajectory_as_the_
    eager_one` asserts it with inductor's `fallback_random` on, which is the only way to hold
    the mask draw still long enough to compare the two.
    """
    if not cfg.runtime.compile:
        return model
    return torch.compile(model, dynamic=cfg.runtime.compile_dynamic)


def compiled_frames() -> int:
    """Frames Dynamo has compiled in this process, ever — the counter behind :class:`CompileWatch`.

    ``counters["frames"]["ok"]`` is Dynamo's own tally of successfully compiled frames, and it
    increments on a *re*-compile exactly as it does on the first compile, which is what makes
    a difference of two readings mean "recompiled in between".
    """
    from torch._dynamo.utils import counters

    return int(counters["frames"]["ok"])


class CompileWatch:
    """How many frames were compiled since it was made (or last :meth:`reset`).

    The instrument for §7.3's precondition. Make one *after* the warmup step — where the first
    compile belongs — and anything it counts afterwards is a recompile: a shape that moved, a
    guard that failed, a Python attribute the graph had specialized on. #95's entrypoint warns
    on it at the end of a run and the tests assert it is zero across an epoch boundary.

    Process-wide rather than per-model, because Dynamo's cache is: two models of one class
    share compiled code, and a recompile triggered by either is a recompile this run paid for.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Start counting from now — what a caller does once the warmup step has landed."""
        self._baseline = compiled_frames()

    @property
    def compiled(self) -> int:
        """Frames compiled since :meth:`reset`. Zero is what a warmed-up static loop reports."""
        return compiled_frames() - self._baseline
