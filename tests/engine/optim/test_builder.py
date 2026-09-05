"""Tests for builder.py."""

import torch
import torch.nn as nn

from minwm.engine.optim.builder import build_optimizer, trainable_params
from minwm.engine.optim.muon import Muon


class TestTrainableParams:
    def test_all_trainable(self):
        module = nn.Linear(4, 3)
        params = trainable_params(module)
        assert len(params) == len(list(module.parameters()))

    def test_filters_frozen(self):
        module = nn.Linear(4, 3, bias=True)
        module.bias.requires_grad_(False)
        params = trainable_params(module)
        assert len(params) == 1
        assert params[0] is module.weight

    def test_empty_when_all_frozen(self):
        module = nn.Linear(4, 3)
        module.requires_grad_(False)
        assert trainable_params(module) == []

    def test_preserves_module_order(self):
        module = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        params = trainable_params(module)
        expected = list(module.parameters())
        assert params == expected


class TestBuildOptimizer:
    def test_defaults_to_adamw(self):
        module = nn.Linear(4, 3)
        opt = build_optimizer(module, {"lr": 1e-3})
        assert isinstance(opt, torch.optim.AdamW)

    def test_kwargs_wired(self):
        module = nn.Linear(4, 3)
        opt = build_optimizer(module, {"lr": 2e-4, "betas": (0.5, 0.99), "weight_decay": 0.01})
        group = opt.param_groups[0]
        assert group["lr"] == 2e-4
        assert group["betas"] == (0.5, 0.99)
        assert group["weight_decay"] == 0.01

    def test_name_selects_class(self):
        module = nn.Linear(4, 3)
        opt = build_optimizer(module, {"name": "torch.optim:SGD", "lr": 0.1, "momentum": 0.9})
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]["momentum"] == 0.9

    def test_muon_class_splits_matrix_and_vector_params(self):
        module = nn.Linear(4, 3, bias=True)
        opt = build_optimizer(module, {"name": "Muon", "lr": 1e-3, "weight_decay": 0.01})
        assert isinstance(opt, Muon)
        assert opt.state[module.weight]["use_muon"] is True
        assert opt.state[module.bias]["use_muon"] is False
        assert opt.param_groups[0]["weight_decay"] == 0.01

        module(torch.randn(2, 4)).sum().backward()
        weight_before = module.weight.detach().clone()
        bias_before = module.bias.detach().clone()
        opt.step()
        assert not torch.equal(module.weight, weight_before)
        assert not torch.equal(module.bias, bias_before)

    def test_name_key_not_passed_as_kwarg(self):
        module = nn.Linear(4, 3)
        # AdamW has no ``name`` kwarg; build_optimizer must strip it before constructing.
        opt = build_optimizer(module, {"name": "torch.optim:AdamW", "lr": 1e-3})
        assert isinstance(opt, torch.optim.AdamW)

    def test_only_trainable_params_optimized(self):
        module = nn.Linear(4, 3, bias=True)
        module.bias.requires_grad_(False)
        opt = build_optimizer(module, {"lr": 1e-3})
        optimized = [p for g in opt.param_groups for p in g["params"]]
        assert len(optimized) == 1
        assert optimized[0] is module.weight


class TestMuon:
    def test_accepts_flat_tensor_iterable(self):
        module = nn.Linear(4, 3, bias=True)
        opt = Muon(module.parameters(), lr=1e-3)
        assert opt.state[module.weight]["use_muon"] is True
        assert opt.state[module.bias]["use_muon"] is False

    def test_accepts_param_group_dicts(self):
        # torch.optim also accepts a list of param-group dicts; tagging use_muon
        # must read the normalised tensors, not the raw dict items.
        module = nn.Linear(4, 3, bias=True)
        opt = Muon([{"params": list(module.parameters()), "lr": 5e-4}], lr=1e-3)
        assert opt.state[module.weight]["use_muon"] is True
        assert opt.state[module.bias]["use_muon"] is False
        assert opt.param_groups[0]["lr"] == 5e-4

    def test_state_dict_roundtrip(self):
        module = nn.Linear(4, 3, bias=True)
        opt = Muon(module.parameters(), lr=1e-3, weight_decay=0.01)
        module(torch.randn(2, 4)).sum().backward()
        opt.step()
        opt.load_state_dict(opt.state_dict())
        assert "weight_decay" in opt.state_dict()["param_groups"][0]
        # A second step after reload must not raise a missing-key error.
        module(torch.randn(2, 4)).sum().backward()
        opt.step()

    def test_step_updates_conv_weight(self):
        # >2D matrix params go through the reshape/Newton-Schulz path.
        module = nn.Conv2d(2, 3, kernel_size=3)
        opt = Muon(module.parameters(), lr=1e-3)
        assert opt.state[module.weight]["use_muon"] is True
        assert opt.state[module.bias]["use_muon"] is False
        module(torch.randn(1, 2, 8, 8)).sum().backward()
        weight_before = module.weight.detach().clone()
        opt.step()
        assert not torch.equal(module.weight, weight_before)

    def test_adamw_backup_accumulates_state(self):
        module = nn.Linear(4, 3, bias=True)
        opt = Muon(module.parameters(), lr=1e-3)
        module(torch.randn(2, 4)).sum().backward()
        opt.step()
        # Vector params use the internal AdamW path (moment buffers, no muon buf).
        assert "moment1" in opt.state[module.bias]
        assert "moment2" in opt.state[module.bias]
        assert "momentum_buffer" not in opt.state[module.bias]
        assert "momentum_buffer" in opt.state[module.weight]
