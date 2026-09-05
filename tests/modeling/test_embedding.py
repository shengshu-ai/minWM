"""Golden equivalence tests for the shared sinusoidal embedding op.

Assert the unified :func:`minwm.modeling.common.embedding.sinusoidal_embedding` reproduces
the two families' original implementations: bit-exact for the HY fp32 path
(same ``exp`` formula), tolerance-based for the Wan fp64 path (the original used
``pow`` where the shared op uses the algebraically-equal ``exp``, differing by
~1e-13).
"""

import math

import torch

from minwm.modeling.common.embedding import sinusoidal_embedding


def _old_hy_timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _old_wan_sinusoidal_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half)),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


class TestHYPath:
    def test_bit_exact_even_dim(self):
        t = torch.arange(6, dtype=torch.float32)
        got = sinusoidal_embedding(t, 128, compute_dtype=torch.float32, pad_odd=True)
        want = _old_hy_timestep_embedding(t, 128)
        assert torch.equal(got, want)

    def test_bit_exact_odd_dim_padded(self):
        t = torch.tensor([0.0, 1.5, 999.0])
        got = sinusoidal_embedding(t, 65, compute_dtype=torch.float32, pad_odd=True)
        want = _old_hy_timestep_embedding(t, 65)
        assert got.shape == (3, 65)
        assert torch.equal(got, want)


class TestWanPath:
    def test_matches_old_within_tolerance(self):
        position = torch.arange(16, dtype=torch.float32)
        got = sinusoidal_embedding(position, 64, compute_dtype=torch.float64, pad_odd=False)
        want = _old_wan_sinusoidal_1d(64, position)
        assert got.dtype == torch.float64
        torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-9)

    def test_odd_dim_raises_when_not_padded(self):
        position = torch.arange(4, dtype=torch.float32)
        try:
            sinusoidal_embedding(position, 63, compute_dtype=torch.float64, pad_odd=False)
        except ValueError:
            return
        raise AssertionError("expected ValueError for odd dim with pad_odd=False")
