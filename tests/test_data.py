"""Tests for minwm.data (build_dataset/build_dataloader) and minwm.utils.comm."""

import sys
import types

import pytest
import torch
from torch.utils.data import Dataset

from minwm.config import Data
from minwm.data import build_dataloader, build_dataset, cycle
from minwm.data.samplers import ResumableDistributedSampler
from minwm.utils import comm


@pytest.fixture
def fake_dataset_module():
    """A tiny importable Dataset for ``type`` resolution."""
    mod = types.ModuleType("minwm_test_data_fakes")

    class TinyDataset(Dataset):
        def __init__(self, n=6, offset=0):
            self.n = n
            self.offset = offset

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return {"x": torch.tensor([idx + self.offset], dtype=torch.float32)}

    mod.TinyDataset = TinyDataset
    sys.modules["minwm_test_data_fakes"] = mod
    yield mod
    del sys.modules["minwm_test_data_fakes"]


# ---- comm ----------------------------------------------------------------


def test_comm_single_process_defaults():
    assert comm.get_world_size() == 1
    assert comm.get_rank() == 0
    assert comm.is_main_process() is True


def test_comm_dp_falls_back_to_global_when_sp_inactive():
    # No SP / no dist init: DP coords mirror global coords.
    assert comm.get_dp_world_size() == comm.get_world_size()
    assert comm.get_dp_rank() == comm.get_rank()


# ---- build_dataset -------------------------------------------------------


def test_build_dataset(fake_dataset_module):
    ds = build_dataset({"type": "minwm_test_data_fakes:TinyDataset", "n": 4})
    assert isinstance(ds, fake_dataset_module.TinyDataset)
    assert len(ds) == 4


def test_build_dataset_requires_cls():
    with pytest.raises(ValueError, match="type"):
        build_dataset({"n": 4})


# ---- build_dataloader ----------------------------------------------------


def test_build_dataloader_basic(fake_dataset_module):
    loader = build_dataloader(
        Data(
            dataset={"type": "minwm_test_data_fakes:TinyDataset", "n": 6},
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
    )
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0]["x"].shape == (2, 1)


def test_build_dataloader_requires_dataset():
    with pytest.raises(ValueError, match="dataset"):
        build_dataloader(Data(batch_size=2))


def test_build_dataloader_single_process_is_one_replica(fake_dataset_module):
    # One replica runs the same epoch-seeded sampler; no process group needed.
    loader = build_dataloader(Data(dataset={"type": "minwm_test_data_fakes:TinyDataset", "n": 6}))
    assert isinstance(loader.sampler, ResumableDistributedSampler)
    assert loader.sampler.num_replicas == 1


@pytest.mark.parametrize("num_replicas", [1, 2])
def test_resumable_sampler_reshuffles_and_skips(num_replicas):
    sampler = ResumableDistributedSampler(range(10), num_replicas=num_replicas, rank=0)
    sampler.set_epoch(0)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    assert epoch0 != epoch1
    assert len(set(epoch1)) == len(epoch1) == len(sampler)
    sampler.set_epoch(1, start=3)
    assert list(sampler) == epoch1[3:]
    assert len(sampler) == len(epoch1)
    sampler.set_epoch(2)
    assert len(list(sampler)) == len(epoch1)


def test_build_dataloader_drop_last(fake_dataset_module):
    loader = build_dataloader(
        Data(
            dataset={"type": "minwm_test_data_fakes:TinyDataset", "n": 5},
            batch_size=2,
            shuffle=False,
            drop_last=True,
        )
    )
    assert len(list(loader)) == 2  # 5 // 2, last partial batch dropped


def test_cycle_repeats(fake_dataset_module):
    loader = build_dataloader(
        Data(
            dataset={"type": "minwm_test_data_fakes:TinyDataset", "n": 3},
            batch_size=1,
            shuffle=False,
        )
    )
    it = cycle(loader)
    seen = [next(it)["x"].item() for _ in range(7)]
    # 3 distinct samples, cycled: length 7 means it wrapped around twice.
    assert len(seen) == 7
    assert set(seen) == {0.0, 1.0, 2.0}


def test_cycle_reshuffles_each_pass(fake_dataset_module):
    loader = build_dataloader(
        Data(dataset={"type": "minwm_test_data_fakes:TinyDataset", "n": 8}, batch_size=8)
    )
    it = cycle(loader)
    first, second = (next(it)["x"].flatten().tolist() for _ in range(2))
    assert sorted(first) == sorted(second)
    assert first != second
