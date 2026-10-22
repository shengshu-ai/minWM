"""Unit tests for the packed (FlashAttention-varlen) teacher-forcing attention.

The claim to verify is equivalence: attention over the teacher-forcing
``BlockMask`` and attention over the lowered ``(q segment, kv segment)`` table
must compute the same thing, row for row, gradients included. That is a statement
about index arithmetic, so every check here runs on CPU without FlashAttention:

* the lowering's segments are compared against ``_teacher_forcing_mask_mod`` —
  the production mask, evaluated densely;
* the packed math is compared against a dense masked softmax in fp32;
* the packed path end to end is compared against torch's own ``flex_attention``
  on a real ``BlockMask``, output and gradients, which is the strongest check
  available without CUDA.

FlashAttention's kernel is exercised through a stand-in that does the varlen math
in plain torch and records its arguments (shapes / cu_seqlens / no-copy).
"""

import importlib.machinery
import math
import sys
import types

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from minwm.modeling.wan21 import packed_tf as ptf
from minwm.modeling.wan21.causal import _teacher_forcing_mask_mod

# Real stage-1 geometry: 20 frames, frame_seqlen 1560, 4 frames per AR block.
REAL = dict(clean_tokens=20 * 1560, block_tokens=4 * 1560)
# Tiny stand-in with the same structure: 4 frames x 3 tokens = 12 clean tokens,
# 2 AR blocks of 2 frames x 3 tokens = 6 tokens.
TINY = dict(num_frames=4, frame_seqlen=3, num_frame_per_block=2)
TINY_REAL = dict(
    clean_tokens=TINY["num_frames"] * TINY["frame_seqlen"],
    block_tokens=TINY["frame_seqlen"] * TINY["num_frame_per_block"],
)
# Smallest geometry flex_attention itself can run on CPU (tile 4, no padding).
FLEX_TINY = dict(num_frames=4, frame_seqlen=2, num_frame_per_block=1)
FLEX_TILE = 4


def _visible(num_frames, frame_seqlen, num_frame_per_block):
    """The TF mask as a dense ``[L, L]`` bool matrix, from the mask_mod's anchors.

    Anchors (offsets) instead of the model's index tensors: ``block`` is one AR
    block, ``clean`` the clean half; a clean row sees its whole block prefix, a
    noisy row sees the clean prefix plus its own noisy block, and every row sees
    itself (which those ranges already contain outside the padding). Written out
    independently of the production mask_mod on purpose, so that a change in
    either one shows up as a disagreement.
    """
    block = frame_seqlen * num_frame_per_block
    clean = num_frames * frame_seqlen
    total = 2 * clean
    q = torch.arange(total).unsqueeze(1)
    kv = torch.arange(total).unsqueeze(0)
    clean_row = q < clean
    noisy_row = (q >= clean) & (q < total)
    b = (q - clean) // block
    visible = (
        (kv == q)
        | (clean_row & (kv < (q // block + 1) * block))
        | (
            noisy_row
            & ((kv < b * block) | ((kv >= clean + b * block) & (kv < clean + (b + 1) * block)))
        )
    )
    return visible


def _visible_production(num_frames, frame_seqlen, num_frame_per_block):
    """The same dense mask, from the mask_mod the training path actually builds."""
    total = 2 * num_frames * frame_seqlen
    mod = _teacher_forcing_mask_mod(
        num_frames=num_frames,
        frame_seqlen=frame_seqlen,
        num_frame_per_block=num_frame_per_block,
        device=torch.device("cpu"),
    )
    return mod(None, None, torch.arange(total).unsqueeze(1), torch.arange(total).unsqueeze(0))


def _varlen_math(q, k, v, cu_q, cu_k, scale):
    """FlashAttention-varlen semantics in plain torch: softmax within each segment."""
    out = torch.empty_like(q)
    for i in range(cu_q.numel() - 1):
        q0, q1 = int(cu_q[i]), int(cu_q[i + 1])
        k0, k1 = int(cu_k[i]), int(cu_k[i + 1])
        scores = torch.einsum("qhd,khd->qkh", q[q0:q1].float(), k[k0:k1].float()) * scale
        probs = torch.softmax(scores, dim=1).to(q.dtype)
        out[q0:q1] = torch.einsum("qkh,khd->qhd", probs, v[k0:k1])
    return out


def _packed_rows(geom):
    """The rows the packed buffer holds, expanded from the geometry's run table.

    The same expansion ``geom.kv_runs`` describes, spelled as one index so that it
    can also be handed to ``index_select`` as the reference behaviour.
    """
    return torch.cat(
        [
            torch.arange(
                sample * geom.total_tokens + start,
                sample * geom.total_tokens + start + length,
            )
            for sample in range(geom.batch)
            for start, length in geom.kv_runs
        ]
    )


def _packed_reference(q, k, v, geom, scale):
    """The packed math in plain torch, unpacking the segment table by hand."""
    rows = _packed_rows(geom)
    return _varlen_math(q, k[rows], v[rows], geom.cu_q, geom.cu_k, scale)


def _fake_varlen(seen: dict | None = None):
    """Stand-in for ``flash_attn.flash_attn_varlen_func``: real math, records args."""

    def fake(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        if seen is not None:
            seen.update(
                q=q,
                k=k,
                v=v,
                cu_q=cu_seqlens_q,
                cu_k=cu_seqlens_k,
                max_q=max_seqlen_q,
                max_k=max_seqlen_k,
                kwargs=kwargs,
            )
        scale = kwargs.get("softmax_scale") or q.shape[-1] ** -0.5
        return _varlen_math(q, k, v, cu_seqlens_q, cu_seqlens_k, scale)

    return fake


@pytest.fixture
def fake_flash(monkeypatch):
    """Install the flash-attn stand-in; yields the dict the call arguments land in."""
    seen: dict = {}
    module = types.ModuleType("flash_attn")
    # A stub in sys.modules still has to look importable: transformers calls
    # importlib.util.find_spec("flash_attn"), which reads __spec__.
    module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    module.flash_attn_varlen_func = _fake_varlen(seen)
    monkeypatch.setitem(sys.modules, "flash_attn", module)
    return seen


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from the default impl and no cached geometry."""
    monkeypatch.delenv(ptf.ENV_VAR, raising=False)
    ptf._reset_cache()
    yield
    ptf._reset_cache()


class TestGating:
    @pytest.mark.parametrize("value", [None, "", "flex"])
    def test_flex_unless_asked(self, monkeypatch, value):
        if value is not None:
            monkeypatch.setenv(ptf.ENV_VAR, value)
        assert not ptf.packed_tf_enabled()

    @pytest.mark.parametrize("value", ["packed", "PACKED", " packed "])
    def test_packed_enables(self, monkeypatch, value):
        monkeypatch.setenv(ptf.ENV_VAR, value)
        assert ptf.packed_tf_enabled()

    def test_typo_raises(self, monkeypatch):
        monkeypatch.setenv(ptf.ENV_VAR, "packd")
        with pytest.raises(ValueError, match="packd"):
            ptf.packed_tf_enabled()


class TestGeometry:
    def test_real_geometry(self):
        geom = ptf.tf_geometry(batch=1, device=torch.device("cpu"), **REAL)
        assert (geom.num_blocks, geom.block_tokens, geom.clean_tokens) == (5, 6240, 31200)
        assert geom.total_tokens == 62400
        assert geom.max_seqlen_q == 6240
        assert geom.max_seqlen_kv == 31200
        # Both families hold (b+1) blocks of kv: 15 blocks' worth of rows, every
        # segment carrying its own copy of the clean prefix, so 3x the sequence.
        assert geom.kv_tokens == 187200 == 3 * geom.total_tokens
        assert geom.cu_q.dtype == geom.cu_k.dtype == torch.int32
        assert geom.cu_q.tolist() == [6240 * i for i in range(11)]
        assert geom.cu_k.tolist() == [
            0,
            6240,
            18720,
            37440,
            62400,
            93600,
            99840,
            112320,
            131040,
            156000,
            187200,
        ]
        # The rows of the packed buffer, spelled out: clean block b carries the
        # clean prefix up to its own block, noisy block b its clean prefix (which
        # is empty at b=0) plus its own block.
        assert geom.kv_runs == (
            (0, 6240),
            (0, 12480),
            (0, 18720),
            (0, 24960),
            (0, 31200),
            (31200, 6240),
            (0, 6240),
            (37440, 6240),
            (0, 12480),
            (43680, 6240),
            (0, 18720),
            (49920, 6240),
            (0, 24960),
            (56160, 6240),
        )
        assert sum(length for _, length in geom.kv_runs) == geom.kv_tokens

    def test_batch_offsets_are_per_sample(self):
        geom = ptf.tf_geometry(batch=3, device=torch.device("cpu"), **REAL)
        single = ptf.tf_geometry(batch=1, device=torch.device("cpu"), **REAL)
        assert geom.cu_q.numel() == 3 * (2 * 5) + 1
        assert geom.cu_q[-1].item() == 3 * 62400
        assert geom.cu_k[-1].item() == 3 * 187200
        # The run table is per sample: the batch only shifts it by total_tokens,
        # which is what the packing applies (see TestPacking).
        assert geom.kv_runs == single.kv_runs

    def test_cached_and_bounded(self):
        first = ptf.tf_geometry(batch=1, device=torch.device("cpu"), **TINY_REAL)
        assert ptf.tf_geometry(batch=1, device=torch.device("cpu"), **TINY_REAL) is first
        other = ptf.tf_geometry(batch=2, device=torch.device("cpu"), **TINY_REAL)
        assert other is not first  # batch is part of the key
        # The cache holds a bounded number of geometries, not one per shape seen.
        for n in range(2, 2 + ptf._GEOMETRY_CACHE_SIZE):
            ptf.tf_geometry(
                clean_tokens=n * 12, block_tokens=12, batch=1, device=torch.device("cpu")
            )
        assert ptf.tf_geometry(batch=1, device=torch.device("cpu"), **TINY_REAL) is not first

        ptf._reset_cache()
        assert ptf.tf_geometry(batch=1, device=torch.device("cpu"), **TINY_REAL) is not first

    def test_block_must_divide_clean(self):
        with pytest.raises(ValueError, match="divisible"):
            ptf.tf_geometry(batch=1, device=torch.device("cpu"), clean_tokens=10, block_tokens=4)


class TestPacking:
    """The pack op on its own: same rows as the index, gradients summed.

    K/V are the only tensors the lowering moves, so this is where the rows come
    from and where the duplicated rows go back to. Both directions are checked
    against ``index_select`` over :func:`_packed_rows` — the same rows in the same
    order, but computed the general way — so the run table cannot drift from it.
    """

    @pytest.mark.parametrize("batch", [1, 2])
    def test_pack_is_the_same_rows_as_index_select(self, batch):
        torch.manual_seed(0)
        geom = ptf.tf_geometry(batch=batch, device=torch.device("cpu"), **TINY_REAL)
        rows = _packed_rows(geom)
        x = torch.randn(batch, geom.total_tokens, 2, 4)

        got = ptf._pack_kv(x, geom)

        assert got.shape == (geom.kv_tokens, 2, 4)
        assert torch.equal(got, x.reshape(-1, 2, 4).index_select(0, rows))

    def test_backward_sums_the_duplicated_rows(self):
        """dK/dV of a row is the sum over the segments that carry it."""
        torch.manual_seed(0)
        geom = ptf.tf_geometry(batch=2, device=torch.device("cpu"), **TINY_REAL)
        rows = _packed_rows(geom)
        counts = torch.bincount(rows, minlength=geom.batch * geom.total_tokens)
        assert counts.max() > 1  # or there would be nothing to sum

        x = torch.randn(geom.batch, geom.total_tokens, 2, 4, requires_grad=True)
        flat = x.reshape(-1, 2, 4)
        grad_out = torch.randn(geom.kv_tokens, 2, 4)
        (got,) = torch.autograd.grad(ptf._pack_kv(x, geom), x, grad_out)
        (expected,) = torch.autograd.grad(flat.index_select(0, rows), x, grad_out)

        assert torch.equal(got, expected)
        # Not just placement: a row that several segments carry gets each copy's
        # gradient, summed.
        row = int(counts.argmax())
        assert int(counts[row]) > 1
        copies = (rows == row).nonzero().flatten()
        assert torch.equal(got.reshape(-1, 2, 4)[row], grad_out[copies].sum(0))


class TestLoweringMatchesTheMask:
    """Segment table == mask semantics, checked row by row."""

    @pytest.mark.parametrize("batch", [1, 2])
    def test_every_segment_is_its_rows_visible_set(self, batch):
        geom = ptf.tf_geometry(batch=batch, device=torch.device("cpu"), **TINY_REAL)
        visible = _visible_production(**TINY)
        total = visible.shape[0]
        segments = 2 * geom.num_blocks
        packed_rows = _packed_rows(geom)
        for sample in range(batch):
            base = sample * total
            for i in range(segments):
                seg = sample * segments + i
                q0 = int(geom.cu_q[seg]) - base
                q1 = int(geom.cu_q[seg + 1]) - base
                rows = packed_rows[int(geom.cu_k[seg]) : int(geom.cu_k[seg + 1])] - base
                expected = visible[q0].nonzero().flatten()
                # One kv range per segment is exact only because all rows of a
                # group see the same set: assert that, loudly.
                assert torch.equal(visible[q0:q1], visible[q0].expand(q1 - q0, -1))
                assert torch.equal(rows.sort().values, expected)

    @pytest.mark.parametrize("batch", [1, 2])
    def test_packed_math_equals_dense_masked_softmax(self, batch):
        torch.manual_seed(0)
        geom = ptf.tf_geometry(batch=batch, device=torch.device("cpu"), **TINY_REAL)
        total, heads, head_dim = geom.total_tokens, 2, 4
        q = torch.randn(batch * total, heads, head_dim)
        k = torch.randn(batch * total, heads, head_dim)
        v = torch.randn(batch * total, heads, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        packed = _packed_reference(q, k, v, geom, scale)

        visible = _visible_production(**TINY)
        expected = torch.empty_like(q)
        for sample in range(batch):
            lo, hi = sample * total, (sample + 1) * total
            scores = (torch.einsum("qhd,khd->qkh", q[lo:hi], k[lo:hi]) * scale).float()
            scores = scores.masked_fill(~visible.unsqueeze(-1), float("-inf"))
            expected[lo:hi] = torch.einsum(
                "qkh,khd->qhd", torch.softmax(scores, dim=1).to(q.dtype), v[lo:hi]
            )

        torch.testing.assert_close(packed, expected, atol=1e-5, rtol=1e-4)


class TestMaskSemantics:
    """The mask itself: production mask_mod vs an independent transcription."""

    @pytest.mark.parametrize(
        "geometry",
        [
            TINY,
            FLEX_TINY,
            dict(num_frames=5, frame_seqlen=7, num_frame_per_block=5),
            dict(num_frames=8, frame_seqlen=2, num_frame_per_block=3),
        ],
    )
    def test_production_mask_mod_matches_the_transcription(self, geometry):
        torch.testing.assert_close(
            _visible_production(**geometry), _visible(**geometry), rtol=0, atol=0
        )


class TestAgainstFlexAttention:
    """End to end: packed == torch's flex_attention on the same BlockMask.

    Output *and* dQ/dK/dV, because every segment carries its own copy of the rows
    it attends (the clean prefix above all): a token's dK/dV has to end up as the
    sum over all the places it appears in the packed buffer, exactly what the dense
    mask sums.
    """

    def test_packed_equals_flex_with_a_block_mask(self, fake_flash):
        total = 2 * FLEX_TINY["num_frames"] * FLEX_TINY["frame_seqlen"]
        # Tile 4 rather than the production 128 only because the eager CPU
        # fallback of flex_attention needs a small case; the mask under test
        # does not depend on the tile.
        mask_mod = _teacher_forcing_mask_mod(device=torch.device("cpu"), **FLEX_TINY)
        block_mask = create_block_mask(
            mask_mod, None, None, total, total, BLOCK_SIZE=FLEX_TILE, device=torch.device("cpu")
        )
        geom = ptf.tf_geometry(
            batch=1,
            device=torch.device("cpu"),
            clean_tokens=FLEX_TINY["num_frames"] * FLEX_TINY["frame_seqlen"],
            block_tokens=FLEX_TINY["frame_seqlen"] * FLEX_TINY["num_frame_per_block"],
        )
        # Without duplicated rows the gradient check below would be vacuous.
        assert sum(length for _, length in geom.kv_runs) > geom.total_tokens

        heads, head_dim = 2, 8
        torch.manual_seed(0)
        q = torch.randn(1, total, heads, head_dim, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        weight = torch.randn_like(q).float()  # a fixed loss, so dK/dV are non-trivial

        def output_and_grads(run):
            inputs = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
            out = run(*inputs)
            return out, torch.autograd.grad((out.float() * weight).sum(), inputs)

        def _flex(q_, k_, v_):
            return flex_attention(
                q_.transpose(1, 2), k_.transpose(1, 2), v_.transpose(1, 2), block_mask=block_mask
            ).transpose(1, 2)

        ref_out, ref_grads = output_and_grads(_flex)
        got_out, got_grads = output_and_grads(lambda *t: ptf.packed_tf_attention(*t, geom))

        # bf16 is the dtype the kernel takes, so this is bf16 agreement; the very
        # same comparison in fp32 lands at ~4e-7, i.e. the lowering is exact and
        # this tolerance is the dtype's, not the index arithmetic's. Getting the
        # mask wrong the other way (ignoring it) is a difference of ~2.0 in the
        # output and ~1.1 in dK here, which this catches.
        for got, ref in zip((got_out, *got_grads), (ref_out, *ref_grads)):
            torch.testing.assert_close(got.float(), ref.float(), atol=3e-2, rtol=3e-2)
        assert fake_flash["kwargs"]["causal"] is False


class TestFlashAttentionCall:
    """The varlen call itself: shapes, cu_seqlens, and that Q is not copied."""

    def test_call_arguments_and_no_q_copy(self, fake_flash):
        seen = fake_flash
        geom = ptf.tf_geometry(batch=1, device=torch.device("cpu"), **TINY_REAL)
        assert seen == {}  # nothing recorded before the call
        total, heads, head_dim = geom.total_tokens, 2, 4
        q = torch.randn(1, total, heads, head_dim, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out = ptf.packed_tf_attention(q, k, v, geom)

        assert out.shape == q.shape
        assert seen["max_q"] == geom.max_seqlen_q
        assert seen["max_k"] == geom.max_seqlen_kv
        assert seen["kwargs"]["causal"] is False
        assert seen["kwargs"]["dropout_p"] == 0.0
        assert seen["kwargs"]["deterministic"] is False  # flash-attn's default
        assert seen["cu_q"].tolist() == geom.cu_q.tolist()
        assert seen["k"].shape[0] == seen["v"].shape[0] == geom.kv_tokens
        # Zero-copy Q: the pointer FA receives is the model's own Q storage.
        assert seen["q"].data_ptr() == q.data_ptr()
        assert seen["q"].shape == (total, heads, head_dim)

    def test_shape_and_dtype_mismatch_raise(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flash_attn", None)
        geom = ptf.tf_geometry(batch=1, device=torch.device("cpu"), **TINY_REAL)
        good = torch.randn(1, geom.total_tokens, 2, 4, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="cached geometry"):
            ptf.packed_tf_attention(good[:, :6], good[:, :6], good[:, :6], geom)
        with pytest.raises(ValueError, match="fp16/bf16"):
            ptf.packed_tf_attention(good.float(), good.float(), good.float(), geom)
        with pytest.raises(ValueError, match="shapes differ"):
            ptf.packed_tf_attention(good, good[:, : geom.total_tokens - 6], good, geom)

    def test_missing_flash_attn_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flash_attn", None)
        geom = ptf.tf_geometry(batch=1, device=torch.device("cpu"), **TINY_REAL)
        good = torch.randn(1, geom.total_tokens, 2, 4, dtype=torch.bfloat16)

        with pytest.raises(RuntimeError, match="requires flash-attn 2") as exc_info:
            ptf.packed_tf_attention(good, good, good, geom)

        assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


# A 2-layer model of the same shape as the real one, small enough for CPU.
_ARCH = dict(
    model_type="t2v",
    dim=64,
    ffn_dim=128,
    freq_dim=64,
    text_len=8,
    text_dim=64,
    num_heads=4,
    num_layers=2,
    in_dim=16,
    out_dim=16,
)
FRAMES, SIDE = 2, 8  # 2 latent frames of 4x4 tokens = 16 tokens per frame


def _tiny_causal_model(use_prope, dtype, seed=0, num_frame_per_block=1):
    """A tiny CausalWan21Model; the seed makes two builds identical."""
    from minwm.modeling.wan21.causal import CausalWan21Model

    torch.manual_seed(seed)
    model = (
        CausalWan21Model(**_ARCH, use_prope=use_prope, num_frame_per_block=num_frame_per_block)
        .eval()
        .to(dtype)
    )
    # head.head is zero-init, which would make the output (and the comparison)
    # trivially zero; prope_o is zero-init on purpose, so only nudge it too.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("head.head.weight", "prope_o.weight")):
                param.add_(0.1)
    return model


def _tf_inputs(dtype, seed=1, cameras=False):
    """One teacher-forcing batch: ``[clean | noisy]`` at the doubled length."""
    torch.manual_seed(seed)
    frame_seqlen = (SIDE // 2) ** 2
    batch = dict(
        x=[torch.randn(16, FRAMES, SIDE, SIDE, dtype=dtype)],
        t=torch.tensor([[0.3, 0.4]]),
        context=[torch.randn(8, 64, dtype=dtype)],
        seq_len=FRAMES * frame_seqlen,
        clean_x=[torch.randn(16, FRAMES, SIDE, SIDE, dtype=dtype)],
    )
    if cameras:
        batch["viewmats"] = torch.eye(4).view(1, 1, 4, 4).repeat(1, FRAMES, 1, 1)
        batch["Ks"] = torch.eye(3).view(1, 1, 3, 3).repeat(1, FRAMES, 1, 1)
    return batch


class TestTinyModelForward:
    """Forward only, tiny model, bf16 (the kernel's dtype): packed vs flex."""

    def test_teacher_forcing_forward_matches_flex(self, fake_flash, monkeypatch):
        inputs = _tf_inputs(torch.bfloat16)
        with torch.no_grad():
            reference = _tiny_causal_model(False, torch.bfloat16)(**inputs)
        assert reference.float().abs().max() > 0.05  # not a trivially-zero output

        monkeypatch.setenv(ptf.ENV_VAR, "packed")
        with torch.no_grad():
            got = _tiny_causal_model(False, torch.bfloat16)(**inputs)

        assert got.shape == reference.shape
        torch.testing.assert_close(got.float(), reference.float(), atol=1e-2, rtol=1e-2)

    def test_packed_mode_builds_no_block_mask(self, fake_flash, monkeypatch):
        """Packed mode must not pay for a BlockMask it never reads."""
        from minwm.modeling.wan21.causal import CausalWan21Model

        def _boom(*args, **kwargs):
            raise AssertionError("BlockMask built")

        monkeypatch.setattr(CausalWan21Model, "_prepare_teacher_forcing_mask", _boom)
        with torch.no_grad(), pytest.raises(AssertionError, match="BlockMask built"):
            _tiny_causal_model(False, torch.bfloat16)(**_tf_inputs(torch.bfloat16))

        monkeypatch.setenv(ptf.ENV_VAR, "packed")
        with torch.no_grad():
            out = _tiny_causal_model(False, torch.bfloat16)(**_tf_inputs(torch.bfloat16))
        assert out.float().abs().max() > 0.05

    def test_prope_and_plain_attention_share_one_geometry(self, monkeypatch):
        """PRoPE needs the *same* segments (v != v_prope); assert both call sites.

        Forwarded in fp32 with the lowering stubbed out (the PRoPE camera
        projection is fp32-only, so the real packed path cannot run here), which
        is what makes this a wiring test rather than a numerical one — including
        that the attention module received ``num_frame_per_block``.
        """
        from minwm.modeling.wan21 import blocks

        calls = []
        monkeypatch.setattr(
            blocks, "packed_tf_attention", lambda q, k, v, geom: calls.append(geom) or q
        )
        monkeypatch.setenv(ptf.ENV_VAR, "packed")
        with torch.no_grad():
            _tiny_causal_model(True, torch.float32)(**_tf_inputs(torch.float32, cameras=True))

        assert len(calls) == 2 * _ARCH["num_layers"]  # plain + PRoPE per layer
        assert len({id(geom) for geom in calls}) == 1  # one cached geometry
        geom = calls[0]
        assert geom.block_tokens == 1 * (SIDE // 2) ** 2  # num_frame_per_block * frame_seqlen
        assert geom.clean_tokens == FRAMES * (SIDE // 2) ** 2
        assert geom.total_tokens == 2 * geom.clean_tokens
