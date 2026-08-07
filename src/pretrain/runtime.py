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

## Two paths compile, and only one of them is §7.3's claim

Dynamo's counter is *process-wide*, while the sentence above is about the **training step**:
§7.3 says the fixed mask count keeps *shapes* static, and that is a claim about the shapes a
step runs on. The monitoring path compiles too, and legitimately — §10.1's val pass and
§10.2's grids forward through the same compiled module in ``eval()`` under
:func:`torch.no_grad`, which Dynamo guards on, so they earn their own graphs the first time
they run. That is necessarily the first epoch boundary.

Reading one counter against §7.3's sentence conflates the two, and #109 is what that cost: a
warning that fired on every healthy run, at 18 frames, identically across R1, R2 and R3, and
recommended turning off a lever R0 measured at ~2.4x. So the watch keeps **two** tallies. The
partition is exact rather than estimated, because a boundary and a step never overlap: frames
counted inside :meth:`CompileWatch.monitoring` are the monitoring path's, and everything else
is the training step's. §7.3 is then checkable in the units it is written in.

## What is deliberately not here

§7.6's two non-levers — gradient checkpointing (unneeded at S/B scale, ~30%) and
`channels_last` (a CNN optimization) — are config knobs pinned to ``False`` and no code. So is
the separate `flash-attn` package: at 49-196 tokens SDPA's win is kernel fusion rather than
memory IO (§7.2), and torch's own implementation is the thing timm blocks already call.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

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


def format_peak_memory(cfg: RunConfig, device) -> str | None:
    """Peak GPU memory the run reached, printed beside the batch split that produced it (#115).

    The reason this exists is #113. R0's readout recorded ViT-B at *"17.82 GiB, physical batch
    1024 x 2 accum"* while :meth:`~pretrain.config.RunConfig.vit_base` was in fact running
    512 x 4, so both of its ViT-B figures were 512 measurements wearing a 1024 label. ADR-0010
    §1 then froze the label, and R4 OOM'd on its first step against a batch needing ~31.3 GiB
    of 31.36. No code was wrong anywhere in that chain: a human copied a batch size from the
    plan into a readout while the preset ran a different one.

    So neither number is quotable without the other, and the fix is to stop them being written
    down separately — both are read off the *run*, at the moment it ends, and printed on one
    line in the place a readout writer copies from. A batch size taken from the plan can be
    wrong; one taken from ``cfg.optimization`` at the end of the run it configured cannot.

    ``None`` off CUDA rather than zeros: a laptop ``--dry-run`` or an mps test has no peak worth
    reporting, and a confident ``0.00 GiB`` is exactly the kind of number that ends up cited.
    """
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    o = cfg.optimization
    gib = 1024**3
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / gib
    return (
        f"{torch.cuda.max_memory_allocated() / gib:.2f} GiB allocated / "
        f"{torch.cuda.max_memory_reserved() / gib:.2f} GiB reserved of {total:.2f} GiB"
        f"   at batch {o.batch_size} x {o.accum_iter} accum = {cfg.effective_batch} effective"
    )


def compiled_frames() -> int:
    """Frames Dynamo has compiled in this process, ever — the counter behind :class:`CompileWatch`.

    ``counters["frames"]["ok"]`` is Dynamo's own tally of successfully compiled frames, and it
    increments on a *re*-compile exactly as it does on the first compile, which is what makes
    a difference of two readings mean "recompiled in between".
    """
    from torch._dynamo.utils import counters

    return int(counters["frames"]["ok"])


class CompileWatch:
    """Frames compiled since it was made (or last :meth:`reset`), split by which path paid.

    The instrument for §7.3's precondition. Make one *after* the warmup step — where the first
    compile belongs — and anything it counts afterwards is a recompile: a shape that moved, a
    guard that failed, a Python attribute the graph had specialized on.

    Process-wide rather than per-model, because Dynamo's cache is: two models of one class
    share compiled code, and a recompile triggered by either is a recompile this run paid for.

    **The split is the whole point** (#109; see this module's header). :attr:`train` is what
    §7.3 claims is zero. :attr:`monitoring` is the val pass and the grids, which compile once
    at the first epoch boundary because that is the first time they run — expected, not a
    fault, and the thing the single-counter version misreported for three consecutive runs.
    Both are still recompiles someone pays for, so both are counted; they are just not the
    same claim, and they do not carry the same remedy.
    """

    def __init__(self) -> None:
        self._monitoring = 0
        self.reset()

    def reset(self) -> None:
        """Start counting from now — what a caller does once the warmup step has landed."""
        self._baseline = compiled_frames()
        self._monitoring = 0

    @contextmanager
    def monitoring(self) -> Iterator[None]:
        """Attribute frames compiled in this block to the monitoring path, not the step.

        Exact rather than heuristic: §10's boundary work never runs concurrently with a
        training step, so "compiled while this was open" partitions the process-wide counter
        cleanly. Re-entrant blocks are fine — the inner one attributes to the same bucket, and
        double-counting is impossible because each block only ever moves frames it saw itself.
        """
        before = compiled_frames()
        try:
            yield
        finally:
            self._monitoring += compiled_frames() - before

    @property
    def monitored(self) -> int:
        """Frames the monitoring path compiled. §10.2's ``due(0)`` puts them all at boundary 0."""
        return self._monitoring

    @property
    def compiled(self) -> int:
        """Frames the *training step* compiled since :meth:`reset` — §7.3's claim, in its units.

        Zero is what a warmed-up static loop reports, and now it stays zero through an epoch
        boundary that scores val and renders a grid.
        """
        return compiled_frames() - self._baseline - self._monitoring

    @property
    def total(self) -> int:
        """Both paths — the number the pre-#109 watch reported, kept for the footer's arithmetic."""
        return compiled_frames() - self._baseline
