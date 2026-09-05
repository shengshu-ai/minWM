"""Smoke tests for the moved PRoPE op.

``prope_qkv`` was moved verbatim from ``minwm.ops.prope`` to
``minwm.modeling.common.prope`` (pure import-path change). End-to-end numerics are covered
by ``tests/modeling/test_wan21_prope.py``; here we only assert the op is
importable from its new home, returns the documented 4-tuple, and that the
identity-camera transform is a no-op on q/k/v.
"""

import torch

from minwm.modeling.common.prope import prope_qkv


def test_returns_qkv_and_output_closure():
    b, heads, cams, hd = 1, 2, 1, 8
    seqlen = cams  # 1 patch per camera
    q = torch.randn(b, heads, seqlen, hd)
    k = torch.randn(b, heads, seqlen, hd)
    v = torch.randn(b, heads, seqlen, hd)
    viewmats = torch.eye(4).reshape(1, 1, 4, 4).expand(b, cams, 4, 4).contiguous()

    out = prope_qkv(q, k, v, viewmats=viewmats, Ks=None)
    assert isinstance(out, tuple) and len(out) == 4
    qp, kp, vp, apply_fn_o = out
    assert qp.shape == q.shape and kp.shape == k.shape and vp.shape == v.shape
    assert callable(apply_fn_o)


def test_identity_viewmat_is_noop_without_intrinsics():
    b, heads, cams, hd = 1, 2, 1, 8
    q = torch.randn(b, heads, cams, hd)
    k = torch.randn(b, heads, cams, hd)
    v = torch.randn(b, heads, cams, hd)
    viewmats = torch.eye(4).reshape(1, 1, 4, 4).expand(b, cams, 4, 4).contiguous()

    qp, kp, vp, apply_fn_o = prope_qkv(q, k, v, viewmats=viewmats, Ks=None)
    # identity SE(3) -> block-diagonal identity transform on the head_dim.
    torch.testing.assert_close(qp, q)
    torch.testing.assert_close(kp, k)
    torch.testing.assert_close(apply_fn_o(v), v)
