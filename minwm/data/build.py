"""Dataset / dataloader factories (detectron2-style).

A ``data`` config is two-level: a nested ``dataset`` sub-config (a ``type``
node built by :func:`minwm.config.build`) plus dataloader-level knobs::

    data:
      dataset:
        type: minwm.data.datasets.lmdb:LatentLMDBDataset
        data_path: /data/latents
        max_pair: 50000
      batch_size: 4
      num_workers: 8
      shuffle: true
      drop_last: true

:func:`build_dataset` resolves just the dataset (handy for inference / eval
where no sampler is needed). :func:`build_dataloader` additionally attaches a
DP-aware distributed sampler and wraps a ``torch.utils.data.DataLoader``.

The sampler is keyed on data-parallel coordinates (:mod:`minwm.utils.comm`), so
ranks sharing a sequence-parallel group receive identical batches. On a single
process it degrades to a plain shuffling loader (no distributed init required).
"""

from typing import Any, Iterator

from torch.utils.data import DataLoader, Dataset

from minwm.config.lazy import build
from minwm.config.schema import Data

from .samplers import build_sampler

__all__ = ["build_dataset", "build_dataloader", "cycle"]


def build_dataset(dataset_cfg: dict) -> Dataset:
    """Instantiate a dataset from a ``type`` config node."""
    if not isinstance(dataset_cfg, dict) or "type" not in dataset_cfg:
        raise ValueError(f"build_dataset expects a dict with a 'type' key, got {dataset_cfg!r}")
    return build(dataset_cfg)


def build_dataloader(data: Data) -> DataLoader:
    """Build a ``DataLoader`` from a typed :class:`~minwm.config.Data` config.

    Reads the open ``dataset`` build node plus the dataloader knobs
    (``batch_size``, ``num_workers``, ``shuffle``, ``drop_last``, ``pin_memory``,
    ``persistent_workers``) and an optional ``collate_fn`` build node off the
    typed config.

    The trainer is responsible for infinite iteration (see :func:`cycle`); this
    returns a plain DataLoader so its ``sampler`` / ``__len__`` stay inspectable.

    Args:
        data (Data): the typed data config; ``data.dataset`` must be a non-empty
            ``type`` node.

    Returns:
        DataLoader: a loader whose sampler is DP-aware under sequence parallelism.

    Raises:
        ValueError: if ``data.dataset`` is empty.
    """
    if not data.dataset:
        raise ValueError("data config must contain a 'dataset' sub-config")

    dataset = build(data.dataset)
    sampler = build_sampler(dataset, data.shuffle, data.drop_last)

    collate_fn = build(data.collate_fn) if data.collate_fn else None

    return DataLoader(
        dataset,
        batch_size=data.batch_size,
        shuffle=data.shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=data.drop_last,
        num_workers=data.num_workers,
        pin_memory=data.pin_memory,
        persistent_workers=data.persistent_workers,
        collate_fn=collate_fn,
    )


def cycle(dataloader: DataLoader) -> Iterator[Any]:
    """Yield batches forever, re-iterating the loader each pass."""
    while True:
        for batch in dataloader:
            yield batch
