"""AdamW, and the one thing about it that is not a default: what decays (#93, plan §6).

§6's optimizer row is four values — AdamW, betas (0.9, 0.95), ``fused=True``, weight decay 0.05
— and three of them are a constructor call. The fourth is this module: *"excluded on bias/norm
params"*, the grouping every ViT recipe since the original uses and nobody re-derives.

It is worth a file because of how it fails. Decaying a LayerNorm's gain and every bias trains
perfectly well: no crash, no NaN, no curve that looks wrong — just a slightly worse model,
which is indistinguishable from a slightly unlucky run. The same is true of the reverse. So the
grouping is computed in one place, over the real model, and asserted by name in the tests.

## Two readings of "bias and norm params", and why disagreeing is an error

The reference implementations (timm's ``add_weight_decay``, which MAE calls) test
**dimensionality**: ``param.ndim <= 1 or name.endswith(".bias")``. That is a proxy — it happens
to select exactly the biases and the normalization gains and shifts in a stock ViT, because
those are the only 1-D tensors in one. The direct reading walks module *types* and asks which
parameters belong to a normalization layer.

On this model the two agree exactly, and :func:`no_decay_parameter_names` **checks that they do**
rather than picking one. The case that separates them is real and one config knob away: timm's
:class:`~timm.models.vision_transformer.Block` offers LayerScale, whose ``gamma`` is a 1-D
learnable scale that is neither a bias nor a norm parameter. The reference rule would exempt it
silently; the type walk would decay it silently. §6 turns LayerScale off, so today the question
is moot — which is exactly when it is cheap to refuse to guess. An architecture change that
introduces such a parameter gets a ``ValueError`` naming it, not a coin flip.

The two class tokens are the case people expect to be here and are not: ``cls_token`` and
``mask_token`` are ``(1, 1, D)``, so both readings put them in the **decaying** group. That is
MAE's own behaviour and it is pinned by a test, because "tokens are like biases" is an easy and
wrong intuition to act on.

## fused, and the one place it is allowed to not happen

``fused=True`` is a few-percent step win on the rented box and nothing more — the same AdamW
update, computed in fewer kernels. Where torch has no fused kernel for the parameters' device,
:func:`build_optimizer` falls back to the unfused implementation and **says so**: it is a
performance decision, not a recipe one, and a run that quietly ran 3% slower is a very different
thing from a run that quietly trained a different objective. The recipe knob stays
:attr:`Optimization.fused <pretrain.config.Optimization>`, and a warning is what marks the
difference between what was asked for and what the hardware could do.
"""

from __future__ import annotations

import warnings
from typing import Final

import torch
from torch import nn

from .config import RunConfig

#: The layers whose parameters are "norm params" in §6's sense — which in a ViT is
#: :class:`~torch.nn.LayerNorm` and nothing else. Deliberately *not* padded out with the other
#: normalizations torch offers: a model that grew one would trip the disagreement check in
#: :func:`no_decay_parameter_names` (its gain is 1-D, so the reference rule exempts it while a
#: type walk would not), which names the parameter and asks for a decision. A speculative entry
#: here would answer that question in advance, silently, for an architecture nobody has built.
NORM_LAYERS: Final = (nn.LayerNorm,)


def _reference_no_decay(model: nn.Module) -> set[str]:
    """timm's ``add_weight_decay`` rule, the one MAE calls: 1-D tensors and anything ``.bias``."""
    return {
        name
        for name, param in model.named_parameters()
        if param.requires_grad and (param.ndim <= 1 or name.endswith(".bias"))
    }


def _structural_no_decay(model: nn.Module) -> set[str]:
    """The direct reading: every bias, plus every parameter owned by a normalization layer."""
    names = set()
    for module_name, module in model.named_modules():
        is_norm = isinstance(module, NORM_LAYERS)
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if is_norm or param_name == "bias":
                names.add(f"{module_name}.{param_name}" if module_name else param_name)
    return names


def no_decay_parameter_names(model: nn.Module) -> set[str]:
    """The names §6 excludes from weight decay — and a refusal when that is ambiguous.

    Returns the set on which both readings above agree. Where they disagree the parameter is
    named in a ``ValueError``: see the module docstring for why guessing is the wrong move.

    Frozen parameters are absent from both sets and so from the result; they do not reach an
    optimizer group at all (:func:`parameter_groups`).
    """
    structural, reference = _structural_no_decay(model), _reference_no_decay(model)
    if structural != reference:
        raise ValueError(
            "the two readings of §6's 'weight decay excluded on bias/norm params' disagree. "
            f"Exempt by module type but decayed by the reference rule: {sorted(structural - reference)}. "
            f"Exempt by the reference rule (ndim <= 1 or a .bias) but decayed by module type: "
            f"{sorted(reference - structural)}. A 1-D learnable scale such as timm's LayerScale "
            "gamma is the usual cause. Decide it deliberately — extend NORM_LAYERS, or state the "
            "exception here — rather than letting a coin flip change the recipe: both outcomes "
            "train, and neither shows up in a loss curve"
        )
    return structural


def parameter_groups(model: nn.Module, *, weight_decay: float) -> list[dict]:
    """§6's two AdamW groups: everything at ``weight_decay``, biases and norms at 0.

    Each group carries a ``name`` — ``"decay"`` / ``"no_decay"`` — which torch preserves in
    ``param_groups`` and in the optimizer state dict. It costs nothing and it is what lets a
    test, a checkpoint reader (#94) or a log line (#97) say which group it is looking at
    without counting on the order they were built in.

    Frozen parameters are skipped entirely rather than added with ``lr = 0``: an optimizer
    holding them would allocate their AdamW moments (two tensors per parameter) and carry them
    in every checkpoint. Nothing in Phase 1 freezes anything — this is the §8 warm start's
    neighbour, and the behaviour is pinned so a future frozen-encoder probe cannot silently
    double the optimizer's memory.
    """
    no_decay = no_decay_parameter_names(model)
    groups: dict[str, list[nn.Parameter]] = {"decay": [], "no_decay": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        groups["no_decay" if name in no_decay else "decay"].append(param)
    return [
        {"name": "decay", "params": groups["decay"], "weight_decay": weight_decay},
        {"name": "no_decay", "params": groups["no_decay"], "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: RunConfig) -> torch.optim.AdamW:
    """The only way a run builds an optimizer: §6's row, off the config object.

    The initial LR is :attr:`RunConfig.lr <pretrain.config.RunConfig>` — the linear scaling
    rule's value — but nothing trains at it: :class:`pretrain.schedule.LrSchedule` overwrites
    every group's ``lr`` before each step, starting from 0 at step 0. It is set here so that an
    optimizer built and inspected outside the loop reports the recipe's peak rather than a
    placeholder.
    """
    opt = cfg.optimization  # `optimizer` is AdamW by construction: RunConfig refuses anything else
    groups = parameter_groups(model, weight_decay=opt.weight_decay)
    kwargs = {"lr": cfg.lr, "betas": opt.betas, "weight_decay": opt.weight_decay}
    if not opt.fused:
        return torch.optim.AdamW(groups, **kwargs, fused=False)
    try:
        return torch.optim.AdamW(groups, **kwargs, fused=True)
    except (RuntimeError, ValueError) as exc:
        warnings.warn(
            f"optimization.fused is set but torch has no fused AdamW for these parameters "
            f"({exc}); falling back to the unfused implementation. This is a throughput "
            f"difference only — the same AdamW update in more kernels — but §7's step-time "
            f"budget assumes the fused path, so the box this run is on is worth a look",
            stacklevel=2,
        )
        return torch.optim.AdamW(groups, **kwargs, fused=False)
