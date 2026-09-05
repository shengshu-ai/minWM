"""SP-consistency tests for the data layer.

The contract a sequence-parallel (SP) group depends on: every rank in one SP
group must feed the model the *same* sample for a given step, because Ulysses
splits one sample's sequence across the group. Two layers guard that:

1. **Sampler keying** — the production sampler (torch ``DistributedSampler``,
   built by :func:`minwm.data.build_sampler`) is keyed on ``dp_rank`` /
   ``dp_world_size``. Ranks sharing an SP group share one ``dp_rank``, so they
   must draw an identical index list; distinct DP groups must not. The
   :class:`~minwm.data.samplers.DPSPBatchSampler` (used by parity tooling) is
   checked for the same property directly.
2. **Dataset determinism** — a dataset must be a pure function of the index
   (plus an SP-group-constant seed): the same index yields the same sample on
   every rank, regardless of process/worker. A dataset that touches the global
   RNG in ``__getitem__`` breaks this.
"""

import random

import pytest
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from minwm.data.samplers import DPSPBatchSampler


class _TinyDataset(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return idx


# ---- production path: DistributedSampler keyed on dp coordinates ---------


def _index_list(sampler: DistributedSampler) -> list[int]:
    return list(iter(sampler))


class TestDistributedSamplerKeying:
    """build_sampler returns a DistributedSampler(rank=dp_rank, num_replicas=dp_ws).

    We construct it directly with the dp coordinates an SP group would resolve to
    (the factory just forwards comm.get_dp_rank / get_dp_world_size), so the test
    needs no process group.
    """

    def test_same_dp_rank_gives_identical_indices(self):
        # Two ranks in the SAME SP group share one dp_rank -> identical shard.
        ds = _TinyDataset(64)
        a = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=True, seed=0)
        b = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=True, seed=0)
        assert _index_list(a) == _index_list(b)

    def test_distinct_dp_ranks_get_disjoint_shards(self):
        # Two DISTINCT DP groups (dp_rank 0 vs 1) must not see the same samples.
        ds = _TinyDataset(64)
        g0 = set(_index_list(DistributedSampler(ds, num_replicas=2, rank=0, seed=0)))
        g1 = set(_index_list(DistributedSampler(ds, num_replicas=2, rank=1, seed=0)))
        assert g0.isdisjoint(g1)

    def test_shards_cover_dataset_without_overlap(self):
        ds = _TinyDataset(64)
        g0 = _index_list(DistributedSampler(ds, num_replicas=2, rank=0, seed=0, drop_last=True))
        g1 = _index_list(DistributedSampler(ds, num_replicas=2, rank=1, seed=0, drop_last=True))
        assert len(g0) == len(g1) == 32
        assert set(g0) | set(g1) == set(range(64))


# ---- parity-tooling path: DPSPBatchSampler -------------------------------


class TestDPSPBatchSampler:
    """DPSPBatchSampler must give an SP group identical indices, DP groups not."""

    def _indices(self, global_rank: int, sp_world_size: int, num_dp_groups: int) -> list[int]:
        s = DPSPBatchSampler(
            batch_size=2,
            dataset_size=64,
            num_dp_groups=num_dp_groups,
            sp_world_size=sp_world_size,
            global_rank=global_rank,
            seed=0,
        )
        return [i for batch in s for i in batch]

    def test_same_sp_group_identical(self):
        # world=4, sp=2 -> SP group 0 = {rank0, rank1}; both share dp_group_id 0.
        assert self._indices(0, sp_world_size=2, num_dp_groups=2) == self._indices(
            1, sp_world_size=2, num_dp_groups=2
        )

    def test_distinct_dp_groups_disjoint(self):
        g0 = set(self._indices(0, sp_world_size=2, num_dp_groups=2))
        g1 = set(self._indices(2, sp_world_size=2, num_dp_groups=2))  # SP group 1
        assert g0.isdisjoint(g1)


# ---- dataset determinism: same index -> same sample ----------------------


class _GlobalRngDataset(Dataset):
    """ANTI-PATTERN: draws from the global RNG in __getitem__ (breaks SP)."""

    def __len__(self) -> int:
        return 100

    def __getitem__(self, idx: int) -> float:
        return random.random()


class _SeededDataset(Dataset):
    """SP-safe: randomness is a pure function of (seed, idx)."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def __len__(self) -> int:
        return 100

    def __getitem__(self, idx: int) -> float:
        return random.Random(self.seed + idx).random()


class TestDatasetDeterminism:
    """A dataset must be a pure function of the index for SP safety."""

    def test_global_rng_dataset_is_NOT_deterministic(self):
        # Demonstrates the hazard: two ranks with different global RNG histories
        # get different samples for the same index. This is why __getitem__ must
        # not touch the global RNG.
        ds = _GlobalRngDataset()
        random.seed(1)
        rank0 = ds[5]
        random.seed(2)  # a different process history
        rank1 = ds[5]
        assert rank0 != rank1

    def test_seeded_dataset_is_deterministic_across_rng_state(self):
        # The fix: an index-derived local RNG yields the same sample regardless
        # of the surrounding global RNG state (i.e. regardless of rank/process).
        ds = _SeededDataset(seed=0)
        random.seed(1)
        rank0 = ds[5]
        random.seed(2)
        rank1 = ds[5]
        assert rank0 == rank1


# ---- end-to-end: SP-group seed alignment -> identical noise/timestep -----


class TestSeedAlignmentGivesIdenticalDraws:
    """With seed+dp_rank equal across an SP group, the preprocessor draw matches.

    The trainer seeds the global RNG once with ``base + dp_rank``; within an SP
    group ``dp_rank`` is constant, so two ranks enter the step with identical RNG
    state and FlowNoise must produce bit-identical noise/timestep. This mirrors
    the real per-step draw without spawning a group.
    """

    def test_equal_seed_gives_identical_noise_and_timestep(self):
        from minwm.processors.flow import FlowNoise
        from minwm.sampling.schedulers import FlowMatchingScheduler
        from minwm.utils import set_seed

        base, dp_rank = 1234, 3  # same on every rank of one SP group
        clean = torch.randn(1, 4, 8, 4, 4)

        def draw():
            set_seed(base + dp_rank)
            return FlowNoise(
                uniform_across_frames=True, scheduler=FlowMatchingScheduler(schedule="linear")
            )({"clean_latent": clean.clone()}, torch.device("cpu"))

        rank0, rank1 = draw(), draw()
        assert torch.equal(rank0["noise"], rank1["noise"])
        assert torch.equal(rank0["timestep"], rank1["timestep"])

    def test_distinct_dp_rank_diverges(self):
        from minwm.processors.flow import FlowNoise
        from minwm.sampling.schedulers import FlowMatchingScheduler
        from minwm.utils import set_seed

        clean = torch.randn(1, 4, 8, 4, 4)

        def draw(dp_rank: int):
            set_seed(1234 + dp_rank)
            return FlowNoise(
                uniform_across_frames=True, scheduler=FlowMatchingScheduler(schedule="linear")
            )({"clean_latent": clean.clone()}, torch.device("cpu"))

        assert not torch.equal(draw(0)["noise"], draw(1)["noise"])
