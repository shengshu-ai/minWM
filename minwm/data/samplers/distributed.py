"""SP-aware distributed sampler factory."""

import itertools
from typing import Iterator

from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from minwm.utils import comm


class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler whose next pass can start part-way through an epoch.

    The index order is a pure function of ``(seed, epoch)``, so a resumed run can
    recompute where it stopped and skip the consumed indices without loading
    them. ``len()`` stays the full epoch.
    """

    start: int = 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        """Select the epoch to shuffle for and where the next pass starts.

        Args:
            epoch (int): epoch number; reseeds the shuffle when ``shuffle=True``.
            start (int): number of this replica's indices to skip.
        """
        super().set_epoch(epoch)
        self.start = start

    def __iter__(self) -> Iterator[int]:
        return itertools.islice(super().__iter__(), self.start, None)


def build_sampler(
    dataset: Dataset, shuffle: bool = True, drop_last: bool = True
) -> ResumableDistributedSampler:
    """Return a sampler keyed on DP rank / world size.

    A single replica gets the same sampler with ``num_replicas=1`` (no process
    group needed), so epochs reshuffle and resume the same way on one GPU as on
    many.

    Args:
        dataset (Dataset): the dataset to sample from.
        shuffle (bool): whether to shuffle indices each epoch.
        drop_last (bool): whether to drop the last incomplete batch.

    Returns:
        ResumableDistributedSampler: sampler keyed on DP rank/world_size.
    """
    return ResumableDistributedSampler(
        dataset,
        num_replicas=comm.get_dp_world_size(),
        rank=comm.get_dp_rank(),
        shuffle=shuffle,
        drop_last=drop_last,
    )
