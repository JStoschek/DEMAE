"""AdamW and the two parameter groups: what decays, what does not (#93, plan §6).

§6 says weight decay 0.05, "excluded on bias/norm params", and that exclusion is the one part
of the optimizer worth testing hard. It is invisible: decaying a LayerNorm's gain and every
bias trains perfectly well and produces a slightly worse model, which is indistinguishable from
a slightly unlucky run. There is no crash and no curve.

So the tests here are **assertions over the real ViT-S**, not over a toy module: that the
zero-decay group holds exactly the biases and the normalization parameters, that the other
group holds exactly the rest, that every parameter is in one of them, and that the two class
tokens are where MAE's own grouping puts them (in the decaying group — they are 3-D, and the
reference rule is a dimensionality test).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pretrain import optim
from pretrain.config import Optimization, RunConfig
from pretrain.model import MaskedAutoencoder


@pytest.fixture(scope="module")
def vit_s() -> MaskedAutoencoder:
    """The R1 baseline model, built once: init over 24M parameters is not free."""
    return MaskedAutoencoder.from_config(RunConfig())


def _norm_parameter_names(model: nn.Module) -> set[str]:
    """Every parameter owned by a normalization layer, found by module type.

    Deliberately a second, independent derivation of what the module under test computes: it
    walks module *types* where the reference rule tests dimensionality, so agreement between
    them is evidence rather than a restatement.
    """
    names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            names |= {f"{module_name}.{p}" for p, _ in module.named_parameters(recurse=False)}
    return names


# --- The grouping, over the real model ----------------------------------------


def test_the_zero_decay_group_is_exactly_the_biases_and_the_norm_parameters(vit_s):
    no_decay = optim.no_decay_parameter_names(vit_s)
    expected = {n for n, _ in vit_s.named_parameters() if n.endswith(".bias")}
    expected |= _norm_parameter_names(vit_s)
    assert no_decay == expected
    # Not vacuous: the model has both kinds, in quantity.
    assert len(no_decay) > 2 * vit_s.arch.encoder_depth


def test_every_parameter_lands_in_exactly_one_group(vit_s):
    groups = optim.parameter_groups(vit_s, weight_decay=0.05)
    grouped = [p for g in groups for p in g["params"]]
    assert len(grouped) == len(list(vit_s.parameters()))
    assert {id(p) for p in grouped} == {id(p) for p in vit_s.parameters()}


def test_the_decaying_group_gets_the_configs_decay_and_the_other_gets_zero(vit_s):
    by_name = {g["name"]: g for g in optim.parameter_groups(vit_s, weight_decay=0.05)}
    assert by_name["decay"]["weight_decay"] == 0.05
    assert by_name["no_decay"]["weight_decay"] == 0.0


def test_the_parameters_in_each_group_are_the_ones_the_predicate_names(vit_s):
    no_decay = optim.no_decay_parameter_names(vit_s)
    named = {id(p): n for n, p in vit_s.named_parameters()}
    by_name = {g["name"]: g for g in optim.parameter_groups(vit_s, weight_decay=0.05)}
    assert {named[id(p)] for p in by_name["no_decay"]["params"]} == no_decay
    assert not {named[id(p)] for p in by_name["decay"]["params"]} & no_decay


def test_the_class_and_mask_tokens_keep_their_weight_decay(vit_s):
    # MAE's own grouping is a dimensionality test, and both tokens are (1, 1, D) — so they
    # decay. Recorded as a test because it is the one case where "bias and norm params" and
    # the reference rule could plausibly be read apart.
    no_decay = optim.no_decay_parameter_names(vit_s)
    assert "cls_token" not in no_decay and "mask_token" not in no_decay


def test_the_patch_embed_weight_decays_and_its_bias_does_not(vit_s):
    no_decay = optim.no_decay_parameter_names(vit_s)
    assert "patch_embed.proj.weight" not in no_decay
    assert "patch_embed.proj.bias" in no_decay


def test_a_frozen_parameter_reaches_neither_group(vit_s):
    model = MaskedAutoencoder.from_config(RunConfig().replace(run_id="frozen-probe"))
    model.patch_embed.proj.weight.requires_grad_(False)
    grouped = {id(p) for g in optim.parameter_groups(model, weight_decay=0.05) for p in g["params"]}
    assert id(model.patch_embed.proj.weight) not in grouped
    assert id(model.patch_embed.proj.bias) in grouped


# --- The guard on the two definitions -----------------------------------------


class _LayerScaled(nn.Module):
    """A module with a 1-D learnable scale that is neither a bias nor a norm parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.gamma = nn.Parameter(torch.ones(4))


def test_a_one_dimensional_parameter_that_is_neither_bias_nor_norm_is_refused():
    # timm's Block offers LayerScale; §6 turns it off, but nothing in the code says so. The two
    # readings of "excluded on bias/norm params" — by module type and by dimensionality —
    # disagree on such a parameter, and guessing either way silently changes the recipe.
    with pytest.raises(ValueError, match="gamma"):
        optim.no_decay_parameter_names(_LayerScaled())


# --- The optimizer itself -----------------------------------------------------


def test_it_is_adamw_with_maes_betas_and_the_scaling_rules_lr(vit_s):
    cfg = RunConfig()
    opt = optim.build_optimizer(vit_s, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 2
    for group in opt.param_groups:
        assert group["betas"] == (0.9, 0.95)
        assert group["lr"] == pytest.approx(1.2e-3) == pytest.approx(cfg.lr)


def test_the_weight_decay_is_the_configs_and_only_the_decaying_group_carries_it(vit_s):
    cfg = RunConfig().replace(optimization=Optimization(weight_decay=0.3))
    decays = {g["name"]: g["weight_decay"] for g in optim.build_optimizer(vit_s, cfg).param_groups}
    assert decays == {"decay": 0.3, "no_decay": 0.0}


def test_fused_is_requested_when_the_config_asks_for_it(vit_s):
    assert RunConfig().optimization.fused is True
    opt = optim.build_optimizer(vit_s, RunConfig())
    assert all(g["fused"] for g in opt.param_groups)


def test_fused_can_be_turned_off_from_the_config(vit_s):
    cfg = RunConfig().replace(optimization=Optimization(fused=False))
    opt = optim.build_optimizer(vit_s, cfg)
    assert not any(g["fused"] for g in opt.param_groups)


def test_a_device_without_a_fused_kernel_falls_back_loudly(vit_s, monkeypatch):
    """The fallback is a performance decision, so it is announced rather than silent."""
    real = torch.optim.AdamW

    def refuse_fused(*args, **kwargs):
        if kwargs.get("fused"):
            raise RuntimeError("fused=True requires all the params to be CUDA tensors")
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", refuse_fused)
    with pytest.warns(UserWarning, match="fused"):
        opt = optim.build_optimizer(vit_s, RunConfig())
    assert not any(g["fused"] for g in opt.param_groups)


def test_the_optimizer_steps_the_model_it_was_built_for(vit_s):
    # An end-to-end sanity check that the groups are wired to real parameters: one backward,
    # one step, and the weights move.
    model = MaskedAutoencoder.from_config(
        RunConfig().replace(
            architecture=RunConfig().architecture.replace(
                encoder_dim=32, encoder_depth=1, encoder_heads=2,
                decoder_dim=32, decoder_depth=1, decoder_heads=2,
            )
        )
    )
    opt = optim.build_optimizer(model, RunConfig())
    before = model.decoder_pred.weight.detach().clone()
    model(torch.randn(2, 1, 224, 224)).pred.square().mean().backward()
    opt.step()
    assert not torch.allclose(before, model.decoder_pred.weight)
