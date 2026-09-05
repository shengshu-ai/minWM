# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan
# works and any output and results therefrom are provided "AS IS" without any
# express or implied warranties of any kind including any warranties of title,
# merchantability, noninfringement, course of dealing, usage of trade, or
# fitness for a particular purpose. You are solely responsible for determining
# the appropriateness of using, reproducing, modifying, performing, displaying
# or distributing any of the Tencent Hunyuan works or outputs and assume any and
# all risks associated with your or a third party's use or distribution of any
# of the Tencent Hunyuan works or outputs and your exercise of rights and
# permissions under this agreement.
# See the License for the specific language governing permissions and
# limitations under the License.
"""n-D rotary positional embeddings (real cos/sin form) for the HY15 DiT."""

import torch


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    """Build an n-D meshgrid from start/stop/num specifications.

    Args:
        start (int | tuple): if no extra args, this is ``num``; with one extra
            arg it is the start (stop=args[0], step=1); with two it is the start
            (stop=args[0], num=args[1]). Ints or n-tuples are accepted.
        *args: optional stop and num, see above.
        dim (int): meshgrid dimensionality.

    Returns:
        torch.Tensor: stacked grid, shape ``[dim, ...]``.

    Raises:
        ValueError: if more than two positional ``args`` are given.
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
    grid = torch.stack(grid, dim=0)  # [dim, W, H, D]

    return grid


def reshape_for_broadcast(
    freqs_cis: torch.Tensor | tuple[torch.Tensor],
    x: torch.Tensor,
    head_first=False,
):
    """Reshape the frequency tensor so it broadcasts against ``x``.

    Args:
        freqs_cis (torch.Tensor | tuple): complex freqs, or a ``(cos, sin)`` pair.
        x (torch.Tensor): target tensor to broadcast against.
        head_first (bool): whether the head dim precedes the sequence dim.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: reshaped freqs.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    head_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query/key tensors.

    Args:
        xq (torch.Tensor): query tensor, shape ``[B, S, H, D]``.
        xk (torch.Tensor): key tensor, shape ``[B, S, H, D]``.
        freqs_cis (torch.Tensor | tuple): complex freqs, or a ``(cos, sin)`` pair.
        head_first (bool): whether the head dim precedes the sequence dim.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: rotated ``(xq, xk)``.
    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        # real * cos - imag * sin
        # imag * cos + real * sin
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(
            xq.device
        )  # [S, D//2] --> [1, S, 1, D//2]
        # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
        # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # [B, S, H, D//2]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor: float | list[float] = 1.0,
    interpolation_factor: float | list[float] = 1.0,
):
    """n-D version of precompute_freqs_cis (RoPE for tokens with n-D structure).

    Args:
        rope_dim_list (list[int]): per-axis rope dim; ``sum`` should equal head_dim.
        start (int | tuple | list): see :func:`get_meshgrid_nd`.
        *args: optional stop and num.
        theta (float): frequency scaling factor.
        use_real (bool): if True, return separate ``(cos, sin)`` real tensors.
        theta_rescale_factor (float | list[float]): per-axis theta rescale.
        interpolation_factor (float | list[float]): per-axis interpolation.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: complex freqs, or
        ``(cos, sin)`` when ``use_real`` is True.
    """

    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list))  # [3, W, H, D] / [2, W, H]

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(
        rope_dim_list
    ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(
        rope_dim_list
    ), "len(interpolation_factor) should equal to len(rope_dim_list)"

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )  # 2 x [WHD, rope_dim_list[i]]
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
        sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)  # (WHD, D/2)
        return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.FloatTensor | int,
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Precompute the 1-D rotary frequency tensor.

    Args:
        dim (int): frequency tensor dimension.
        pos (int | torch.FloatTensor): position indices, ``[S]`` or scalar count.
        theta (float): frequency scaling factor.
        use_real (bool): if True, return separate ``(cos, sin)`` real tensors.
        theta_rescale_factor (float): rescale factor for theta.
        interpolation_factor (float): position interpolation factor.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: complex freqs ``[S, D/2]``
        or ``(cos, sin)`` real tensors ``[S, D]``.
    """
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    # rescale rotary embeddings to a longer sequence length without fine-tuning
    # (bloc97); has some connection to NTK literature
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [D/2]
    # assert interpolation_factor == 1.0, f"interpolation_factor: {interpolation_factor}"
    freqs = torch.outer(pos * interpolation_factor, freqs)  # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis
