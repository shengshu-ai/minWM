"""CPU smoke tests for the Wan causal inference pipeline + helpers.

These exercise the GPU-independent seams: the ``model.`` prefix-strip loader,
the camera-trajectory parser, and a full block-rollout forward on a tiny
random-weight model (the KV-cache inference path uses SDPA, not FlexAttention,
so it runs on CPU). They assert shapes / finiteness, not numerical parity
(that needs real weights + a GPU).
"""

import numpy as np
import pytest
import torch

from minwm.engine import BaseInferencer
from minwm.engine.checkpoint.formats import strip_wrapper_prefix
from minwm.engine.inference.loop import ARGenerationLoop, BidirectionalGenerationLoop
from minwm.engine.inference.samplers import (
    CMSolver,
    UniPCSolver,
    build_sampler,
)
from minwm.modeling.wan21.adapter import Wan21Adapter
from minwm.modeling.wan21.causal import CausalWan21Model
from minwm.processors import CameraTrajectory, InferenceRuntime, LatentNoise
from minwm.processors.camera import make_camera_tensors, parse_trajectory
from minwm.sampling.schedulers import FlowMatchingScheduler

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


class _StubTextEncoder:
    """Returns one zero context tensor per prompt (shape only; no real weights)."""

    def __init__(self, text_dim: int):
        self.text_dim = text_dim

    def __call__(self, prompts: list[str]) -> list[torch.Tensor]:
        return [torch.zeros(4, self.text_dim) for _ in prompts]


class _StubVAE:
    """Identity-ish decode: maps [C,T,H,W] latents to [3,T,H,W] pixels."""

    def decode(self, zs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [z[:3].clamp(-1, 1) for z in zs]


class _ModuleStubTextEncoder(torch.nn.Module):
    def __init__(self, text_dim: int):
        super().__init__()
        self.text_dim = text_dim

    def forward(self, prompts: list[str]) -> list[torch.Tensor]:
        return [torch.zeros(4, self.text_dim) for _ in prompts]


class _ZeroFlowAdapter:
    wants_inference_autocast = False

    def __init__(self, text_encoder=None, negative_prompt=None):
        self.calls = 0
        self.forward_calls = 0
        self.refresh_calls = 0
        self.init_use_prope = []
        self.forward_viewmats_seen = []
        self.text_encoder = text_encoder
        self.negative_prompt = negative_prompt

    def attach_text_encoder(self, text_encoder):
        if self.text_encoder is None:
            self.text_encoder = text_encoder

    def set_negative_prompt(self, negative_prompt):
        self.negative_prompt = negative_prompt

    def encode_text(self, prompts, batch_size, device):
        if prompts is None:
            prompts = [""] * batch_size
        return [c.to(device) for c in self.text_encoder(prompts)]

    def conditioning(self, batch, batch_size, device):
        return {"context": self.encode_text(batch.get("prompts"), batch_size, device)}

    def null_conditioning(self, batch, batch_size, device):
        negatives = [self.negative_prompt or ""] * batch_size
        return {"context": self.encode_text(negatives, batch_size, device)}

    def denoise(self, model, *, noisy, timestep, context, clean=None, viewmats=None, Ks=None):
        self.calls += 1
        assert timestep.shape == noisy.shape[:2]
        return torch.zeros_like(noisy)

    def decode_latents(self, vae, latents):
        zs = [latents[i].permute(1, 0, 2, 3) for i in range(latents.shape[0])]
        return torch.stack([d.permute(1, 0, 2, 3) for d in vae.decode(zs)])

    # Rollout hooks — the ARGenerationLoop drives every family through these.
    def rollout_init_cache(self, model, *, use_prope_cache=True, **kwargs):
        self.init_use_prope.append(use_prope_cache)
        return {"tag": "cache"}

    def rollout_forward(self, model, *, noisy_block, timestep, viewmats=None, **kwargs):
        self.forward_calls += 1
        self.forward_viewmats_seen.append(viewmats)
        assert timestep.shape == noisy_block.shape[:2]
        return torch.zeros_like(noisy_block)

    def rollout_refresh_cache(self, model, *, clean_block, **kwargs):
        self.refresh_calls += 1


class _TinyCachedCausal(torch.nn.Module):
    use_prope = False
    local_attn_size = -1
    num_heads = 2
    dim = 8
    text_len = 4
    patch_size = (1, 2, 2)

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity()])


def _build_pipeline(use_prope: bool = False, num_frame_per_block: int = 1):
    gen = CausalWan21Model(
        **_ARCH, use_prope=use_prope, num_frame_per_block=num_frame_per_block
    ).eval()
    scheduler = FlowMatchingScheduler(num_train_timesteps=1000, shift=5.0)
    adapter = Wan21Adapter(
        causal=True,
        text_len=_ARCH["text_len"],
        text_dim=_ARCH["text_dim"],
        dtype="float32",
        text_encoder=_ModuleStubTextEncoder(_ARCH["text_dim"]),
    )
    sampler = CMSolver(scheduler, [1000, 750, 500, 250], warp_denoising_step=True)
    return ARGenerationLoop(
        generator=gen,
        vae=_StubVAE(),
        adapter=adapter,
        sampler=sampler,
        num_frame_per_block=num_frame_per_block,
        context_noise=0,
    )


class TestTrajectoryParser:
    def test_w19_shape_and_identity_first(self):
        vm = parse_trajectory("w*19")
        assert vm.shape == (20, 4, 4)
        assert np.allclose(vm[0], np.eye(4))

    def test_chained_segments(self):
        vm = parse_trajectory("w*10,d*9")
        assert vm.shape == (20, 4, 4)

    def test_make_camera_tensors_shapes(self):
        vm, ks = make_camera_tensors("w*19")
        assert tuple(vm.shape) == (1, 20, 4, 4)
        assert tuple(ks.shape) == (1, 20, 3, 3)

    def test_default_intrinsics_are_half(self):
        _, ks = make_camera_tensors("w*1")
        assert torch.allclose(ks[0, 0], torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1.0]]))

    def test_bad_segment_raises(self):
        with pytest.raises(ValueError):
            parse_trajectory("forward*5")


class TestPrefixStripLoader:
    def test_model_prefix_strip_matches_keys(self):
        # Wrapped checkpoints may carry a `model.` prefix; stripping it must land
        # exactly on the native CausalWan21Model keys.
        gen = CausalWan21Model(**_ARCH, use_prope=True)
        native_keys = set(gen.state_dict().keys())
        wrapped = {f"model.{k}": v for k, v in gen.state_dict().items()}
        stripped = {
            k[len("model.") :] if k.startswith("model.") else k: v for k, v in wrapped.items()
        }
        assert set(stripped.keys()) == native_keys
        missing, unexpected = gen.load_state_dict(stripped, strict=True)
        assert missing == [] and unexpected == []

    def test_fsdp_prefix_also_stripped(self):
        gen = CausalWan21Model(**_ARCH)
        wrapped = {f"model._fsdp_wrapped_module.{k}": v for k, v in gen.state_dict().items()}

        def _strip(k: str) -> str:
            for p in ("model._fsdp_wrapped_module.", "model."):
                if k.startswith(p):
                    return k[len(p) :]
            return k

        stripped = {_strip(k): v for k, v in wrapped.items()}
        assert set(stripped.keys()) == set(gen.state_dict().keys())

    def test_stacked_wrapper_prefixes_are_stripped(self):
        assert strip_wrapper_prefix("model._fsdp_wrapped_module._orig_mod.blocks.0.weight") == (
            "blocks.0.weight"
        )
        assert strip_wrapper_prefix("_orig_mod.model.blocks.0.weight") == "blocks.0.weight"


class TestInferencerConfigGuards:
    def test_build_batch_scopes_inference_mode(self):
        class _GradCheckingPreprocessor:
            def __init__(self):
                self.grad_enabled = None

            def __call__(self, batch, runtime):
                self.grad_enabled = torch.is_grad_enabled()
                return batch

        pre = _GradCheckingPreprocessor()
        inferencer = BaseInferencer.__new__(BaseInferencer)
        inferencer.runtime = None
        inferencer.input_preprocessors = [pre]

        with torch.enable_grad():
            batch = inferencer.build_batch("prompt", trajectory="w*1")
            assert torch.is_grad_enabled()

        assert pre.grad_enabled is False
        assert batch == {"prompts": ["prompt"], "trajectory": "w*1"}

    def test_flow_unipc_sampler_config_is_built(self):
        sampler = build_sampler(
            inference_cfg={
                "num_inference_steps": 50,
                "sampler": {"solver": "UniPCSolver", "num_train_timesteps": 1000, "shift": 5.0},
            }
        )

        assert sampler.num_inference_steps == 50
        assert sampler.num_train_timesteps == 1000
        assert sampler.shift == 5.0

    def test_sampler_solver_accepts_import_path(self):
        # An out-of-tree sampler can be named by a 'module.path:Name' import path.
        sampler = build_sampler(
            inference_cfg={
                "sampler": {
                    "solver": "minwm.engine.inference.samplers:CMSolver",
                    "denoising_step_list": [1000, 500],
                }
            }
        )
        assert isinstance(sampler, CMSolver)
        assert sampler.denoising_steps.numel() == 2

    def test_unknown_sampler_solver_raises(self):
        with pytest.raises(NotImplementedError, match="not_a_sampler"):
            build_sampler(inference_cfg={"sampler": {"solver": "not_a_sampler"}})

    def test_few_step_sampler_config_is_built(self):
        # few_step sampler is named by its class directly, no alias.
        sampler = build_sampler(
            inference_cfg={
                "sampler": {
                    "solver": "CMSolver",
                    "denoising_step_list": [1000, 750, 500, 250],
                    "shift": 5.0,
                    "sigma_min": 0.0,
                }
            }
        )
        assert isinstance(sampler, CMSolver)
        assert sampler.denoising_steps.numel() == 4

    def _build_loop_via_inferencer(self, cfg, inference_cfg):
        inferencer = BaseInferencer.__new__(BaseInferencer)
        inferencer.cfg = cfg
        inferencer.inference_cfg = inference_cfg
        inferencer.device = torch.device("cpu")
        inferencer.dtype = torch.float32
        inferencer._build_loop()
        return inferencer

    def test_bidirectional_pipeline_is_selected(self, monkeypatch):
        cfg = {
            "model": {"type": "model"},
            "text_encoder": {"type": "text"},
            "adapter": {"type": "adapter"},
            "recipe": {
                "num_frame_per_block": 99,
                "adapter": {"type": "training-only-adapter"},
            },
        }
        inference_cfg = {
            "loop": "BidirectionalGenerationLoop",
            "num_inference_steps": 50,
            "sampler": {"solver": "UniPCSolver", "num_train_timesteps": 1000, "shift": 5.0},
        }

        def _build_stub(cfg):
            if cfg["type"] == "text":
                return _ModuleStubTextEncoder(_ARCH["text_dim"])
            if cfg["type"] == "adapter":
                return _ZeroFlowAdapter()
            raise AssertionError(cfg)

        monkeypatch.setattr(
            "minwm.engine.inferencer.build_model",
            lambda cfg: torch.nn.Linear(1, 1),
        )
        monkeypatch.setattr("minwm.engine.inferencer.build", _build_stub)

        inferencer = self._build_loop_via_inferencer(cfg, inference_cfg)

        assert isinstance(inferencer.loop, BidirectionalGenerationLoop)

    def test_causal_ar_pipeline_is_selected(self, monkeypatch):
        cfg = {
            "model": {"type": "model"},
            "text_encoder": {"type": "text"},
            "adapter": {"type": "adapter"},
            "recipe": {"num_frame_per_block": 99},
        }
        inference_cfg = {
            "loop": "ARGenerationLoop",
            "num_inference_steps": 2,
            "sampler": {"solver": "UniPCSolver", "num_train_timesteps": 1000, "shift": 5.0},
            "num_frame_per_block": 1,
        }

        def _build_stub(cfg):
            if cfg["type"] == "text":
                return _ModuleStubTextEncoder(_ARCH["text_dim"])
            if cfg["type"] == "adapter":
                return _ZeroFlowAdapter()
            raise AssertionError(cfg)

        monkeypatch.setattr(
            "minwm.engine.inferencer.build_model",
            lambda cfg: _TinyCachedCausal(),
        )
        monkeypatch.setattr("minwm.engine.inferencer.build", _build_stub)

        inferencer = self._build_loop_via_inferencer(cfg, inference_cfg)

        assert isinstance(inferencer.loop, ARGenerationLoop)
        assert inferencer.loop.num_frame_per_block == 1

    def test_default_preprocessors_build_noise_and_camera(self):
        runtime = InferenceRuntime(
            device=torch.device("cpu"),
            dtype=torch.float32,
            inference_cfg={"num_frames": 2, "latent_shape": (4, 3, 5)},
        )
        inferencer = BaseInferencer.__new__(BaseInferencer)
        inferencer.runtime = runtime
        inferencer.input_preprocessors = [LatentNoise(), CameraTrajectory()]

        batch = inferencer.build_batch("camera prompt", trajectory="w*1")

        assert batch["prompts"] == ["camera prompt"]
        assert batch["noise"].shape == (1, 2, 4, 3, 5)
        assert batch["viewmats"].shape == (1, 2, 4, 4)
        assert batch["Ks"].shape == (1, 2, 3, 3)


class TestPipelineForward:
    def test_flow_unipc_timesteps_match_reference(self):
        sampler = UniPCSolver(num_inference_steps=5, shift=5.0)
        scheduler = sampler.new_scheduler(torch.device("cpu"))

        assert scheduler.timesteps.tolist() == [999, 952, 882, 768, 555]

    def test_flow_unipc_50_step_sigmas_match_reference(self):
        sampler = UniPCSolver(num_inference_steps=50, shift=5.0)
        scheduler = sampler.new_scheduler(torch.device("cpu"))

        assert scheduler.timesteps[:5].tolist() == [999, 995, 991, 987, 982]
        assert scheduler.timesteps[-5:].tolist() == [356, 302, 241, 172, 92]
        assert scheduler.sigmas[[0, 15, 19, 45, 47, 50]].view(torch.int32).tolist() == [
            1065349858,
            1064024630,
            1063516924,
            1052162556,
            1048021704,
            0,
        ]

    def test_bidirectional_diffusion_shapes_and_cfg_calls(self):
        torch.manual_seed(0)
        sampler = UniPCSolver(num_inference_steps=2, shift=5.0)
        adapter = _ZeroFlowAdapter(
            text_encoder=_ModuleStubTextEncoder(_ARCH["text_dim"]),
            negative_prompt="bad quality",
        )
        pipe = BidirectionalGenerationLoop(
            generator=torch.nn.Linear(1, 1),
            vae=_StubVAE(),
            sampler=sampler,
            adapter=adapter,
            guidance_scale=2.0,
        )
        noise = torch.randn(1, 2, 16, 8, 8)

        output = pipe.generate({"noise": noise, "prompts": ["a test prompt"]})

        assert output["latents"].shape == noise.shape
        assert output["video"].shape == (1, 2, 3, 8, 8)
        assert adapter.calls == sampler.num_inference_steps * 2
        assert torch.isfinite(output["video"]).all()

    def test_causal_ar_shapes_and_cfg_calls(self):
        torch.manual_seed(0)
        sampler = UniPCSolver(num_inference_steps=2, shift=5.0)
        adapter = _ZeroFlowAdapter(
            text_encoder=_ModuleStubTextEncoder(_ARCH["text_dim"]),
            negative_prompt="bad quality",
        )
        pipe = ARGenerationLoop(
            generator=_TinyCachedCausal(),
            vae=_StubVAE(),
            adapter=adapter,
            sampler=sampler,
            guidance_scale=2.0,
            num_frame_per_block=1,
        )
        noise = torch.randn(1, 2, 16, 4, 4)

        output = pipe.generate({"noise": noise, "prompts": ["a test prompt"]})

        num_blocks = noise.shape[1]
        assert output["latents"].shape == noise.shape
        assert output["video"].shape == (1, 2, 3, 4, 4)
        # CFG: cond + uncond forward per step, per block.
        assert adapter.forward_calls == num_blocks * sampler.num_inference_steps * 2
        # cond + uncond cache refresh per block.
        assert adapter.refresh_calls == num_blocks * 2
        assert torch.isfinite(output["video"]).all()

    def test_causal_ar_prope_cache_requires_camera_inputs(self):
        torch.manual_seed(0)
        sampler = UniPCSolver(num_inference_steps=1, shift=5.0)
        adapter = _ZeroFlowAdapter(
            text_encoder=_ModuleStubTextEncoder(_ARCH["text_dim"]),
            negative_prompt="bad quality",
        )
        generator = _TinyCachedCausal()
        generator.use_prope = True
        pipe = ARGenerationLoop(
            generator=generator,
            vae=None,
            adapter=adapter,
            sampler=sampler,
            guidance_scale=2.0,
            num_frame_per_block=1,
        )
        noise = torch.randn(1, 1, 16, 4, 4)

        pipe.generate({"noise": noise, "prompts": ["no camera"]})
        # No camera -> the pipeline asks the adapter NOT to allocate a PRoPE cache.
        assert adapter.init_use_prope
        assert all(flag is False for flag in adapter.init_use_prope)

        adapter.init_use_prope.clear()
        adapter.forward_viewmats_seen.clear()
        vm, ks = make_camera_tensors("w*0")
        pipe.generate({"noise": noise, "prompts": ["camera"], "viewmats": vm, "Ks": ks})

        assert adapter.init_use_prope
        assert all(flag is True for flag in adapter.init_use_prope)
        assert all(v is not None for v in adapter.forward_viewmats_seen)

    def test_t2v_rollout_shapes(self):
        torch.manual_seed(0)
        pipe = _build_pipeline(use_prope=False, num_frame_per_block=1)
        noise = torch.randn(1, 3, 16, 8, 8)
        output = pipe.generate({"noise": noise, "prompts": ["a test prompt"]})
        video, latents = output["video"], output["latents"]
        assert latents.shape == (1, 3, 16, 8, 8)
        assert video.shape[:2] == (1, 3)
        assert torch.isfinite(video).all()
        assert ((video >= 0) & (video <= 1)).all()

    def test_prope_rollout_with_camera(self):
        torch.manual_seed(0)
        pipe = _build_pipeline(use_prope=True, num_frame_per_block=1)
        noise = torch.randn(1, 4, 16, 8, 8)
        vm, ks = make_camera_tensors("w*3")  # 4 frames
        output = pipe.generate(
            {"noise": noise, "prompts": ["camera move"], "viewmats": vm, "Ks": ks}
        )
        video, latents = output["video"], output["latents"]
        assert latents.shape == (1, 4, 16, 8, 8)
        assert torch.isfinite(video).all()

    def test_block_size_multiple(self):
        torch.manual_seed(0)
        pipe = _build_pipeline(use_prope=False, num_frame_per_block=2)
        noise = torch.randn(1, 4, 16, 8, 8)
        video = pipe.generate({"noise": noise, "prompts": ["blocked"]})["video"]
        assert torch.isfinite(video).all()
