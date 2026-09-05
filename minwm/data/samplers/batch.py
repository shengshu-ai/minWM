"""Batch sampler for data-parallel + sequence-parallel training."""

import torch
from torch.utils.data import Sampler


class DPSPBatchSampler(Sampler[list[int]]):
    """Batch sampler aware of both DP and SP dimensions.

    In a DP+SP setup, ranks are grouped into SP groups (peers processing
    different sequence chunks). This sampler ensures:
      1. Each SP group sees a distinct shard of the dataset (DP partitioning).
      2. Ranks within the same SP group get identical batches (via DP rank).

    The sampler produces epoch-level shuffled indices, shards them by DP group,
    and yields batches sequentially within each shard. Suitable for StatefulDataLoader.

    Args:
        batch_size (int): number of samples per batch.
        dataset_size (int): total dataset size.
        num_dp_groups (int): number of DP groups (world_size // sp_world_size).
        sp_world_size (int): size of each SP group.
        global_rank (int): global rank of this process.
        drop_last (bool): drop incomplete batches at epoch end.
        drop_first_row (bool): exclude index 0 from the dataset (legacy flag).
        seed (int): RNG seed for the random permutation.
    """

    def __init__(
        self,
        batch_size: int,
        dataset_size: int,
        num_dp_groups: int,
        sp_world_size: int,
        global_rank: int,
        drop_last: bool = True,
        drop_first_row: bool = False,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.dataset_size = dataset_size

        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(dataset_size, generator=rng)

        if drop_first_row:
            indices = indices[indices != 0]
            self.dataset_size -= 1

        if drop_last:
            num_batches = self.dataset_size // batch_size
            num_global_batches = num_batches // num_dp_groups
            indices = indices[: num_global_batches * num_dp_groups * batch_size]
        else:
            remainder = self.dataset_size % (num_dp_groups * batch_size)
            if remainder:
                padding = num_dp_groups * batch_size - remainder
                indices = torch.cat([indices, indices[:padding]])

        dp_group_id = global_rank // sp_world_size
        self.indices = indices[dp_group_id::num_dp_groups]

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i : i + self.batch_size].tolist()

    def __len__(self) -> int:
        return len(self.indices) // self.batch_size
