"""Tests for the HY DMD self-forcing rollout + recipe wiring (unified pipeline).

The pure-tensor helper tests run anywhere. The end-to-end rollout / train-step
tests build the tiny mock HY transformer and are CUDA-gated (the HY token-refiner
forces flash-attn, which is bf16/CUDA only).

Since the DMD unification the HY model runs through the same model-agnostic
``SelfForcingPipeline.inference_with_trajectory`` and ``ARDMDRecipe`` as Wan;
the model-specific rollout lives in ``HYAdapter``'s ``rollout_*`` hooks.
"""

import pytest
import torch

from minwm.engine.training.recipes.ar_dmd import ARDMDRecipe
from minwm.modeling.hy15.adapter import HYAdapter
from minwm.sampling.rollouts import SelfForcingPipeline

CUDA = torch.cuda.is_available()


class TestHyPrepareCondLatents:
    def test_i2v_layout_first_block(self):
        """i2v first block (f_start=0): image + mask on frame 0, 0 elsewhere; [B,C+1,T,H,W]."""
        B, C, T, H, W = 1, 16, 4, 8, 8
        latents = torch.zeros(B, C, T, H, W)
        image_cond = torch.randn(B, C, 1, H, W)
        out = HYAdapter._rollout_prepare_cond(image_cond, latents, "i2v", 0)
        assert out.shape == (B, C + 1, T, H, W)
        assert torch.equal(out[:, :C, :1], image_cond)
        assert torch.count_nonzero(out[:, :C, 1:]) == 0
        assert torch.all(out[:, C:, :1] == 1.0)
        assert torch.all(out[:, C:, 1:] == 0.0)

    def test_i2v_later_block_all_zero(self):
        """i2v later block (f_start>0): all-zero cond+mask.

        The image lives only on the sequence's global frame 0, so blocks past it
        carry no condition — otherwise each chunk's first frame is re-pulled toward
        the input image, producing chunk-boundary jumps.
        """
        B, C, T, H, W = 1, 16, 4, 8, 8
        latents = torch.zeros(B, C, T, H, W)
        image_cond = torch.randn(B, C, 1, H, W)
        out = HYAdapter._rollout_prepare_cond(image_cond, latents, "i2v", 4)
        assert out.shape == (B, C + 1, T, H, W)
        assert torch.count_nonzero(out) == 0

    def test_t2v_all_zero(self):
        """t2v: no image condition, all-zero condition + mask channels."""
        B, C, T, H, W = 1, 16, 4, 8, 8
        latents = torch.zeros(B, C, T, H, W)
        out = HYAdapter._rollout_prepare_cond(None, latents, "t2v", 0)
        assert out.shape == (B, C + 1, T, H, W)
        assert torch.count_nonzero(out) == 0


def _tiny_model():
    from minwm.modeling.hy15.causal import ARHunyuanVideo_1_5_DiffusionTransformer

    m = ARHunyuanVideo_1_5_DiffusionTransformer(
        patch_size=[1, 2, 2],
        in_channels=16,
        concat_condition=True,
        hidden_size=64,
        heads_num=4,
        mm_double_blocks_depth=2,
        mm_single_blocks_depth=0,
        rope_dim_list=[4, 6, 6],
        text_states_dim=64,
        text_states_dim_2=64,
        use_prope=True,
    )
    return m.to("cuda", torch.bfloat16).train()


def _mock_cond(B, C, H, W, device):
    return dict(
        prompt_embed=torch.zeros(B, 8, 64, device=device, dtype=torch.bfloat16),
        prompt_mask=torch.ones(B, 8, device=device, dtype=torch.bfloat16),
        byt5_text_states=torch.zeros(B, 4, 1472, device=device, dtype=torch.bfloat16),
        byt5_text_mask=torch.ones(B, 4, device=device, dtype=torch.bfloat16),
        vision_states=torch.zeros(B, 1, 1280, device=device, dtype=torch.bfloat16),
        image_cond=torch.zeros(B, C, 1, H, W, device=device, dtype=torch.bfloat16),
        mask_type="i2v",
        context=None,
    )


@pytest.mark.skipif(not CUDA, reason="HY token-refiner forces flash-attn (CUDA/bf16 only)")
class TestHyRolloutEndToEnd:
    def test_rollout_shape_and_determinism(self):
        """The unified rollout runs on HY and is deterministic given the same noise."""
        from minwm.distributed.rng import get_rng_states_tracker
        from minwm.sampling.schedulers import FlowMatchingScheduler

        torch.manual_seed(0)
        get_rng_states_tracker().reset()
        model = _tiny_model()
        B, F, C, H, W = 1, 4, 16, 8, 16
        dev = torch.device("cuda")
        adapter = HYAdapter(task_type="i2v", text_len=8, text_dim=64, dtype="bfloat16")
        adapter.dtype = torch.bfloat16
        pipe = SelfForcingPipeline(
            generator=model,
            scheduler=FlowMatchingScheduler(shift=5.0),
            adapter=adapter,
            denoising_step_list=[750, 500, 250],
            num_frame_per_block=4,
        )
        cond = _mock_cond(B, C, H, W, dev)
        vm = torch.eye(4, device=dev, dtype=torch.bfloat16).expand(B, F, 4, 4).contiguous()
        ks = torch.eye(3, device=dev, dtype=torch.bfloat16).expand(B, F, 3, 3).contiguous()
        noise = torch.randn(B, F, C, H, W, device=dev, dtype=torch.bfloat16)
        tracker = get_rng_states_tracker()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # The rollout draws exit_step / renoise off the tracked stream (which
            # advances across calls and is *not* reset by torch.manual_seed), so
            # reset it before each call to compare two identically-seeded rollouts.
            tracker.reset()
            torch.manual_seed(123)
            out1, _, _ = pipe.inference_with_trajectory(noise, cond, vm, ks)
            tracker.reset()
            torch.manual_seed(123)
            out2, _, _ = pipe.inference_with_trajectory(noise, cond, vm, ks)
        assert out1.shape == (B, F, C, H, W)
        assert torch.equal(out1, out2)


@pytest.mark.skipif(not CUDA, reason="HY token-refiner forces flash-attn (CUDA/bf16 only)")
class TestHyDmdTrainStep:
    def test_generator_and_critic_steps(self):
        """One generator step (step 0) + one critic step (step 1) run end-to-end."""
        from minwm.distributed.rng import get_rng_states_tracker

        torch.manual_seed(0)
        get_rng_states_tracker().reset()
        gen = _tiny_model()
        real_score = _tiny_model().eval().requires_grad_(False)
        fake_score = _tiny_model()

        adapter = HYAdapter(task_type="i2v", text_len=8, text_dim=64, dtype="bfloat16")
        adapter.dtype = torch.bfloat16
        recipe = ARDMDRecipe(
            adapter=adapter,
            critic_schedule="alternating",
            normalize_mode="global",
            denoising_step_list=[750, 500, 250],
            num_frame_per_block=4,
            dfake_gen_update_ratio=2,
            num_train_timesteps=1000,
            timestep_shift=5.0,
            optimizer=dict(lr=1e-4),
            critic_optimizer=dict(lr=1e-4),
        )
        recipe.dtype = torch.bfloat16
        B, F, C, H, W = 1, 4, 16, 8, 16
        dev = torch.device("cuda")
        cond = _mock_cond(B, C, H, W, dev)
        batch = {k: v for k, v in cond.items() if k not in ("mask_type", "context")}
        batch["clean_latent"] = torch.randn(B, F, C, H, W, device=dev, dtype=torch.bfloat16)
        batch["viewmats"] = (
            torch.eye(4, device=dev, dtype=torch.bfloat16).expand(B, F, 4, 4).contiguous()
        )
        batch["Ks"] = torch.eye(3, device=dev, dtype=torch.bfloat16).expand(B, F, 3, 3).contiguous()
        opts = recipe.build_optimizers(gen)
        opts.update(recipe.build_auxiliary_optimizers({"fake_score": fake_score}))
        aux = {"real_score": real_score, "fake_score": fake_score}
        m0 = recipe.train_one_step(gen, batch, opts, 0, auxiliary_models=aux)  # generator
        m1 = recipe.train_one_step(gen, batch, opts, 1, auxiliary_models=aux)  # critic
        assert "generator_loss" in m0 and "critic_loss" in m1
        assert m1["critic_loss"] > 0
