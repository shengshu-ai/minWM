"""Tests for the CPU+CUDA RNG-state tracker (minwm.distributed.rng).

The tracker replaces the fragile "global RNG" idiom: draws wrapped in
``fork()`` advance a dedicated stream in isolation, so an unrelated draw between
two forks can't desync it, and the stream round-trips through a checkpoint. These
run single-process on CPU (fork tracks CPU state too).
"""

import torch

from minwm.distributed.rng import (
    DATA_PARALLEL_RNG_TRACKER_NAME,
    RNGStatesTracker,
    get_rng_states_tracker,
)
from minwm.utils import set_seed


def _fresh_tracker(seed: int = 123) -> RNGStatesTracker:
    t = RNGStatesTracker()
    t.reset()
    t.add(DATA_PARALLEL_RNG_TRACKER_NAME, seed)
    return t


class TestForkStream:
    def test_fork_advances_stream(self):
        t = _fresh_tracker()
        with t.fork():
            a = torch.rand(4)
        with t.fork():
            b = torch.rand(4)
        assert not torch.equal(a, b)

    def test_fork_isolated_from_outer_draws(self):
        """A draw between two forks must not perturb the tracked stream."""
        t = _fresh_tracker()
        with t.fork():
            a = torch.rand(4)
        torch.rand(100)  # noise on the default generator, outside the fork
        with t.fork():
            b = torch.rand(4)

        t2 = _fresh_tracker()
        with t2.fork():
            a2 = torch.rand(4)
        with t2.fork():
            b2 = torch.rand(4)

        assert torch.equal(a, a2)
        assert torch.equal(b, b2)

    def test_fork_restores_outer_state(self):
        """After a fork the default generator is left where the caller had it."""
        torch.manual_seed(7)
        expected = torch.rand(4)
        torch.manual_seed(7)
        t = _fresh_tracker()
        with t.fork():
            torch.rand(50)
        assert torch.equal(torch.rand(4), expected)


class TestReseed:
    def test_reseed_restores_initial_stream(self):
        t = _fresh_tracker()
        with t.fork():
            first = torch.rand(4)
        with t.fork():
            torch.rand(4)
        t.reseed()
        with t.fork():
            again = torch.rand(4)
        assert torch.equal(first, again)


class TestCheckpointRoundtrip:
    def test_state_dict_resumes_stream(self):
        t = _fresh_tracker()
        with t.fork():
            torch.rand(5)  # advance
        sd = t.state_dict()
        assert set(sd) == {
            f"{DATA_PARALLEL_RNG_TRACKER_NAME}-{t.dp_rank}/cpu",
            f"{DATA_PARALLEL_RNG_TRACKER_NAME}-{t.dp_rank}/cuda",
        }

        loaded = _fresh_tracker()
        loaded.load_state_dict(sd)
        with loaded.fork():
            resumed = torch.rand(5)

        ref = _fresh_tracker()
        with ref.fork():
            torch.rand(5)
        with ref.fork():
            expected = torch.rand(5)
        assert torch.equal(resumed, expected)


class TestSeedDpRankOffset:
    """The trainer seeds ``base_seed + dp_rank`` via :func:`minwm.utils.set_seed`."""

    def test_dp_rank_offset_diverges(self):
        set_seed(100 + 0)
        with get_rng_states_tracker().fork():
            a = torch.rand(4)
        set_seed(100 + 1)
        with get_rng_states_tracker().fork():
            b = torch.rand(4)
        assert not torch.equal(a, b)

    def test_same_dp_rank_matches(self):
        set_seed(100 + 2)
        with get_rng_states_tracker().fork():
            a = torch.rand(4)
        set_seed(100 + 2)
        with get_rng_states_tracker().fork():
            b = torch.rand(4)
        assert torch.equal(a, b)
