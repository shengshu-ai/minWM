"""Tests for minwm.utils.seed: resolve_seed / set_seed.

``resolve_seed`` is the SP-critical primitive: when the configured base seed is
``None`` it must draw a base seed on rank 0 and broadcast it so **every** rank —
across DP groups — agrees on one base. The trainer then adds ``dp_rank`` to that
base, so a draw is identical within an SP group (shared ``dp_rank``) and distinct
across DP groups. The multiprocess test exercises the real broadcast over a gloo
process group; the rest run single-process.
"""

import os
import random
from unittest import mock

import numpy as np
import pytest
import torch

from minwm.utils import resolve_seed, set_seed


class TestResolveSeedSingleProcess:
    """resolve_seed without an initialized process group."""

    def test_explicit_seed_returned_unchanged(self):
        assert resolve_seed(123) == 123

    def test_explicit_zero_is_a_valid_seed_not_a_sentinel(self):
        # 0 is a real seed here (unlike the wan-refactor convention); only None
        # triggers a draw.
        assert resolve_seed(0) == 0

    def test_none_without_dist_draws_in_range(self):
        s = resolve_seed(None)
        assert isinstance(s, int)
        assert 0 <= s < 2**31 - 1

    def test_none_without_dist_is_random_across_calls(self):
        # No process group: each call draws independently off the global RNG.
        seeds = {resolve_seed(None) for _ in range(50)}
        assert len(seeds) > 1


class TestResolveSeedBroadcastPath:
    """resolve_seed when torch.distributed reports an initialized group."""

    def test_none_with_dist_takes_broadcast_branch(self):
        # Mock an initialized CPU group: the rank-0 draw is broadcast in place,
        # so resolve_seed must return whatever broadcast leaves in the tensor —
        # not the locally drawn value.
        def fake_broadcast(tensor, src):
            tensor.fill_(777)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.broadcast", side_effect=fake_broadcast),
        ):
            out = resolve_seed(None, device=torch.device("cpu"))
        assert out == 777

    def test_explicit_seed_skips_broadcast_even_with_dist(self):
        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.broadcast") as bcast,
        ):
            assert resolve_seed(42, device=torch.device("cpu")) == 42
            bcast.assert_not_called()


class TestSetSeed:
    """set_seed makes the three global RNG streams reproducible."""

    def test_seeds_random_numpy_torch_reproducibly(self):
        set_seed(99)
        a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        set_seed(99)
        b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        assert a == b

    def test_distinct_seeds_diverge(self):
        set_seed(1)
        a = torch.rand(4)
        set_seed(2)
        b = torch.rand(4)
        assert not torch.equal(a, b)


# ---- multiprocess broadcast alignment (the real distributed path) --------


def _broadcast_worker(rank: int, world_size: int, return_dict):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29555"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # Every rank passes None; resolve_seed must hand back rank 0's draw on all.
        base = resolve_seed(None, device=torch.device("cpu"))
        return_dict[rank] = base
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_resolve_seed_broadcast_aligns_all_ranks():
    """None on every rank -> all ranks return the SAME base seed (gloo, CPU).

    This is the invariant my change relies on: a job left unseeded still shares
    one base seed across the whole world, so ``base + dp_rank`` stays SP-synced /
    DP-diverse rather than every rank free-running its own random base.
    """
    import torch.multiprocessing as mp

    world_size = 4
    mgr = mp.Manager()
    return_dict = mgr.dict()
    mp.spawn(
        _broadcast_worker,
        args=(world_size, return_dict),
        nprocs=world_size,
        join=True,
    )
    seeds = [return_dict[r] for r in range(world_size)]
    assert len(set(seeds)) == 1, f"ranks disagree on base seed: {seeds}"
