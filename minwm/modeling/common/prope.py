# MIT License
#
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""PRoPE (projective positional encoding), shared across model families.

Two halves live together here:

- ``prope_qkv`` — the pure, stateless geometry: applies PRoPE to q/k/v and
  returns the transformed tensors plus a closure that applies the matching
  transform to the attention output (no learnable parameters).
- ``add_prope_parameters`` — the Wan-side helper that mutates a model by
  registering the per-block ``prope_o`` output projection (the learnable half).
"""

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn


def prope_qkv(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: torch.Tensor | None,  # (batch, cameras, 3, 3)
    patches_x: int | None = None,  # How many patches wide is each image?
    patches_y: int | None = None,  # How many patches tall is each image?
    image_width: int | None = None,  # Width of the image. Used to normalize intrinsics.
    image_height: int | None = None,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: tuple[torch.Tensor, torch.Tensor] | None = None,
    coeffs_y: tuple[torch.Tensor, torch.Tensor] | None = None,
    mask: torch.Tensor | None = None,
    kv_cache=None,
    is_cache: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Apply PRoPE-style positional encoding to q/k/v.

    Similar to ``torch.nn.functional.scaled_dot_product_attention`` but applies
    PRoPE-style positional encoding. Currently assumes the sequence length equals
    ``cameras * patches_x * patches_y`` and that the ``(seqlen,)`` axis can be
    reshaped into ``(cameras, patches_x, patches_y)``.

    Args:
        q (torch.Tensor): queries ``(batch, num_heads, seqlen, head_dim)``.
        k (torch.Tensor): keys, same shape as ``q``.
        v (torch.Tensor): values, same shape as ``q``.
        viewmats (torch.Tensor): ``camera<-world`` transforms ``(batch, cameras, 4, 4)``.
        Ks (torch.Tensor | None): camera intrinsics ``(batch, cameras, 3, 3)``.
        patches_x (int | None): patches wide per image.
        patches_y (int | None): patches tall per image.
        image_width (int | None): image width, to normalize intrinsics.
        image_height (int | None): image height, to normalize intrinsics.
        coeffs_x (tuple[torch.Tensor, torch.Tensor] | None): optional x coefficients.
        coeffs_y (tuple[torch.Tensor, torch.Tensor] | None): optional y coefficients.
        mask (torch.Tensor | None): optional attention mask (unused here).
        kv_cache: optional KV cache (unused here).
        is_cache (bool): cache flag (unused here).

    Returns:
        tuple: ``(query, key, value, apply_fn_o)`` — the PRoPE-transformed q/k/v
        and a closure applying the matching transform to the attention output.
    """
    # We're going to assume self-attention: all inputs are the same shape.
    batch, num_heads, seqlen, head_dim = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
    # assert seqlen == cameras * patches_x * patches_y

    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns_all_dim(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
    )

    query = apply_fn_q(q)
    key = apply_fn_kv(k)
    value = apply_fn_kv(v)

    return query, key, value, apply_fn_o


def _prepare_apply_fns_all_dim(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: torch.Tensor | None,  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: torch.Tensor | None = None,
    coeffs_y: torch.Tensor | None = None,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding."""
    batch, cameras, _, _ = viewmats.shape

    # Normalize camera intrinsics.
    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0]
        Ks_norm[..., 1, 1] = Ks[..., 1, 1]
        Ks_norm[..., 0, 2] = 0
        Ks_norm[..., 1, 2] = 0
        Ks_norm[..., 2, 2] = 1.0
        Ks_norm = Ks_norm.to(dtype=Ks.dtype)
        del Ks

        # Compute the camera projection matrices we use in PRoPE.
        # - K is an `image<-camera` transform.
        # - viewmats is a `camera<-world` transform.
        # - P = lift(K) @ viewmats is an `image<-world` transform.
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2).to(dtype=viewmats.dtype)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        ).to(dtype=viewmats.dtype)

    else:
        # GTA formula. P is `camera<-world` transform.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), head_dim),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), head_dim),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), head_dim),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)
    batch, num_heads, seqlen, feat_dim = feats.shape
    cameras = matrix.shape[1]
    assert seqlen >= cameras and seqlen % cameras == 0
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: list[tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    out = out.to(dtype=transforms.dtype)
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    out = out.to(dtype=Ks.dtype)
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    out = out.to(dtype=Ks.dtype)
    return out


def add_prope_parameters(model, zero_init: bool = True) -> None:
    """Add learnable PRoPE parameters to each self-attention block.

    Registers per self-attention layer a zero-initialised ``prope_o``
    (``nn.Linear(dim, dim)``) output projection for the PRoPE attention path
    (``output = rope_out + prope_o(prope_out)``).

    Args:
        model: a ``Wan21Model`` or ``CausalWan21Model`` instance.
        zero_init (bool): if True, zero-initialise ``prope_o`` weights and biases.
    """
    try:
        from minwm.modeling.wan21.attention import (
            WanGanCrossAttention,
            WanI2VCrossAttention,
            WanSelfAttention,
            WanT2VCrossAttention,
        )

        cross_attn_types = (WanT2VCrossAttention, WanI2VCrossAttention, WanGanCrossAttention)
    except ImportError:
        WanSelfAttention = None
        cross_attn_types = ()

    try:
        from minwm.modeling.wan21.causal import CausalWanSelfAttention
    except ImportError:
        CausalWanSelfAttention = None

    attn_types = tuple(filter(None, [WanSelfAttention, CausalWanSelfAttention]))

    for _name, module in model.named_modules():
        if isinstance(module, attn_types) and not isinstance(module, cross_attn_types):
            if not hasattr(module, "prope_o"):
                # Get dim from existing projection layer
                dim = module.o.out_features
                prope_o = nn.Linear(dim, dim)

                if zero_init:
                    nn.init.zeros_(prope_o.weight)
                    nn.init.zeros_(prope_o.bias)

                # Move to same device/dtype as existing parameters
                prope_o = prope_o.to(device=module.o.weight.device, dtype=module.o.weight.dtype)

                module.prope_o = prope_o
