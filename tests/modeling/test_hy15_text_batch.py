"""Regression tests for HY15 batched text masks."""

import importlib
import sys
import types

import pytest
import torch

from minwm.modeling.hy15.causal import ARHunyuanVideo_1_5_DiffusionTransformer


def _make_tiny_model():
    return ARHunyuanVideo_1_5_DiffusionTransformer(
        patch_size=[1, 2, 2],
        in_channels=4,
        concat_condition=False,
        hidden_size=32,
        heads_num=4,
        mm_double_blocks_depth=1,
        mm_single_blocks_depth=0,
        rope_dim_list=[2, 2, 4],
        text_projection="linear",
        text_states_dim=16,
        text_states_dim_2=16,
        attn_mode="torch",
        use_prope=False,
    ).eval()


def _batch_inputs():
    torch.manual_seed(0)
    batch, frames, channels, height, width = 2, 2, 4, 4, 4
    text_len, text_dim = 5, 16
    hidden_states = torch.randn(batch, channels, frames, height, width)
    timestep = torch.zeros(batch * frames)
    timestep_txt = torch.zeros(batch)
    text_states = torch.randn(batch, text_len, text_dim)
    text_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    return hidden_states, timestep, timestep_txt, text_states, text_mask


def test_flex_block_attn_falls_back_to_dense_for_padded_text_mask(monkeypatch):
    attn = importlib.import_module("minwm.modeling.hy15.layers.attention.attention")
    monkeypatch.setattr(attn, "_is_flash3_available", lambda: False)
    monkeypatch.setattr(attn, "_is_flash2_available", lambda: False)
    monkeypatch.setattr(
        attn,
        "get_infer_state",
        lambda: types.SimpleNamespace(enable_sageattn=True, sage_blocks_range=range(1)),
    )
    fake_sageattention = types.ModuleType("sageattention")

    def fail_sageattn(*args, **kwargs):
        raise AssertionError("padded text masks must not use SageAttention")

    fake_sageattention.sageattn = fail_sageattn
    monkeypatch.setitem(sys.modules, "sageattention", fake_sageattention)
    torch.manual_seed(1)
    batch, img_len, txt_len, heads, dim = 2, 3, 5, 2, 8
    img_q = torch.randn(batch, img_len, heads, dim)
    img_k = torch.randn(batch, img_len, heads, dim)
    img_v = torch.randn(batch, img_len, heads, dim)
    txt_q = torch.randn(batch, txt_len, heads, dim)
    txt_k = torch.randn(batch, txt_len, heads, dim)
    txt_v = torch.randn(batch, txt_len, heads, dim)
    text_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)

    with pytest.warns(UserWarning, match="flash is not available"):
        out = attn.sequence_parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_len,
            img_kv_len=img_len,
            text_mask=text_mask,
            attn_mode="flex-block-attn",
            attn_param={},
            block_idx=0,
        )
    with pytest.warns(UserWarning, match="flash is not available"):
        direct_sage_out = attn.sequence_parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_len,
            img_kv_len=img_len,
            text_mask=text_mask,
            attn_mode="sageattn",
            attn_param={},
            block_idx=0,
        )
    expected = attn.sequence_parallel_attention(
        (img_q, txt_q),
        (img_k, txt_k),
        (img_v, txt_v),
        img_q_len=img_len,
        img_kv_len=img_len,
        text_mask=text_mask,
        attn_mode="torch",
        attn_param={},
    )

    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(direct_sage_out, expected)


def _timestep_for_sample(timestep, sample_idx, frames):
    return timestep[sample_idx * frames : (sample_idx + 1) * frames]


def test_hy_forward_bi_supports_uneven_text_masks_batch_gt_one():
    model = _make_tiny_model()
    hidden_states, timestep, timestep_txt, text_states, text_mask = _batch_inputs()
    frames = hidden_states.shape[2]

    with torch.no_grad():
        out, features = model.forward_bi(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_txt=timestep_txt,
            text_states=text_states,
            text_states_2=None,
            encoder_attention_mask=text_mask,
            extra_kwargs=None,
            mask_type="t2v",
        )

        single_outs = []
        for i in range(hidden_states.shape[0]):
            single_out, single_features = model.forward_bi(
                hidden_states=hidden_states[i : i + 1],
                timestep=_timestep_for_sample(timestep, i, frames),
                timestep_txt=timestep_txt[i : i + 1],
                text_states=text_states[i : i + 1],
                text_states_2=None,
                encoder_attention_mask=text_mask[i : i + 1],
                extra_kwargs=None,
                mask_type="t2v",
            )
            assert single_features is None
            single_outs.append(single_out[0])

    assert features is None
    assert out.shape == hidden_states.shape
    torch.testing.assert_close(out, torch.stack(single_outs), rtol=1e-5, atol=1e-5)


def test_hy_ar_text_cache_masks_padding_tokens_batch_gt_one():
    model = _make_tiny_model()
    hidden_states, timestep, timestep_txt, text_states, text_mask = _batch_inputs()
    frames = hidden_states.shape[2]
    changed_padding = text_states.clone()
    changed_padding[~text_mask] += 1000.0

    with torch.no_grad():
        kv_cache = model.forward_txt(
            timestep_txt=timestep_txt,
            text_states=text_states,
            encoder_attention_mask=text_mask,
            extra_kwargs=None,
            mask_type="t2v",
            cache_txt=True,
        )
        out, _ = model.forward_vision(
            hidden_states=hidden_states,
            timestep=timestep,
            kv_cache=kv_cache,
        )

        single_outs = []
        for i in range(hidden_states.shape[0]):
            single_kv_cache = model.forward_txt(
                timestep_txt=timestep_txt[i : i + 1],
                text_states=text_states[i : i + 1],
                encoder_attention_mask=text_mask[i : i + 1],
                extra_kwargs=None,
                mask_type="t2v",
                cache_txt=True,
            )
            single_out, _ = model.forward_vision(
                hidden_states=hidden_states[i : i + 1],
                timestep=_timestep_for_sample(timestep, i, frames),
                kv_cache=single_kv_cache,
            )
            single_outs.append(single_out[0])

        changed_kv_cache = model.forward_txt(
            timestep_txt=timestep_txt,
            text_states=changed_padding,
            encoder_attention_mask=text_mask,
            extra_kwargs=None,
            mask_type="t2v",
            cache_txt=True,
        )
        changed_out, _ = model.forward_vision(
            hidden_states=hidden_states,
            timestep=timestep,
            kv_cache=changed_kv_cache,
        )

    assert torch.equal(kv_cache[0]["text_mask"], text_mask)
    torch.testing.assert_close(out, torch.stack(single_outs), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out, changed_out, rtol=1e-5, atol=1e-5)


def test_hy_rejects_empty_text_mask_rows():
    model = _make_tiny_model()
    hidden_states, timestep, timestep_txt, text_states, text_mask = _batch_inputs()
    text_mask[1] = False

    with pytest.raises(ValueError, match="at least one valid token"):
        model.forward_bi(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_txt=timestep_txt,
            text_states=text_states,
            text_states_2=None,
            encoder_attention_mask=text_mask,
            extra_kwargs=None,
            mask_type="t2v",
        )


def test_hy_forward_bi_rejects_teacher_forcing_with_padded_batched_text_masks():
    model = _make_tiny_model()
    hidden_states, timestep, timestep_txt, text_states, text_mask = _batch_inputs()

    with pytest.raises(NotImplementedError, match="teacher-forcing forward_bi"):
        model.forward_bi(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_txt=timestep_txt,
            text_states=text_states,
            text_states_2=None,
            encoder_attention_mask=text_mask,
            extra_kwargs=None,
            mask_type="t2v",
            clean_x=torch.randn_like(hidden_states),
            aug_timesteps=timestep,
        )
