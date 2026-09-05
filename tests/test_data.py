"""Tests for minwm.data (build_dataset/build_dataloader) and minwm.utils.comm."""

import sys
import types

import pytest
import torch
from torch.utils.data import Dataset

from minwm.config import Data
from minwm.data import build_dataloader, build_dataset, cycle
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


def test_build_dataloader_single_process_has_no_sampler(fake_dataset_module):
    # On one replica, sampler is None so DataLoader owns shuffling.
    loader = build_dataloader(Data(dataset={"type": "minwm_test_data_fakes:TinyDataset", "n": 6}))
    assert loader.sampler is not None  # default RandomSampler
    from torch.utils.data.distributed import DistributedSampler

    assert not isinstance(loader.sampler, DistributedSampler)


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
