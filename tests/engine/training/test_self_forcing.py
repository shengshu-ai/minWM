"""Tests for self_forcing.py."""

from minwm.sampling.rollouts import sample_exit_step


class TestSampleExitStep:
    def test_returns_int(self):
        """A single exit step index, shared across all blocks."""
        result = sample_exit_step(5)
        assert isinstance(result, int)

    def test_range(self):
        """Exit step should be in [0, num_denoising_steps)."""
        for _ in range(50):
            assert 0 <= sample_exit_step(4) < 4

    def test_single_step(self):
        """With 1 denoising step, exit must be 0."""
        assert sample_exit_step(1) == 0
