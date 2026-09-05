"""Tests for shared train/inference runtime initialization."""

import torch

from minwm.engine.runtime import initialize_runtime


def test_initialize_runtime_local_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    runtime = initialize_runtime(distributed=False)

    assert runtime.device == torch.device("cpu")
    assert runtime.rank == 0
    assert runtime.world_size == 1


def test_initialize_runtime_forwards_parallel_degrees(monkeypatch):
    calls = []

    def _initialize(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "minwm.distributed.maybe_init_distributed_environment_and_model_parallel",
        _initialize,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)

    runtime = initialize_runtime(
        tp_size=2,
        sp_size=4,
        hsdp_shard_size=8,
        distributed=True,
    )

    assert calls == [
        {
            "tp_size": 2,
            "sp_size": 4,
            "data_parallel_shard_size": 8,
        }
    ]
    assert runtime.device == torch.device("cpu")
    assert runtime.rank == 2
    assert runtime.world_size == 8


def test_initialize_runtime_selects_cuda_local_rank(monkeypatch):
    selected_devices = []

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(
        "minwm.distributed.maybe_init_distributed_environment_and_model_parallel",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", selected_devices.append)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)

    runtime = initialize_runtime(distributed=True)

    assert selected_devices == [3]
    assert runtime.device == torch.device("cuda", 3)
    assert runtime.rank == 3
    assert runtime.world_size == 4
