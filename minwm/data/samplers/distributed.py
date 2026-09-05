"""SP-aware distributed sampler factory."""

from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from minwm.utils import comm


def build_sampler(
    dataset: Dataset, shuffle: bool = True, drop_last: bool = True
) -> DistributedSampler | None:
    """Return a DP-aware DistributedSampler, or None on a single replica.

    Returning None lets DataLoader honour shuffle directly, so this works
    without an initialized process group (tests, local debugging).

    Args:
        dataset (Dataset): the dataset to sample from.
        shuffle (bool): whether to shuffle indices each epoch.
        drop_last (bool): whether to drop the last incomplete batch.

    Returns:
        DistributedSampler | None: sampler keyed on DP rank/world_size, or
            None when running on a single replica.
    """
    num_replicas = comm.get_dp_world_size()
    if num_replicas <= 1:
        return None

    return DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=comm.get_dp_rank(),
        shuffle=shuffle,
        drop_last=drop_last,
    )
