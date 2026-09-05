"""Wan diffusion model wrapper for ODE sampling with PRoPE camera conditioning.

Builds the causal Wan transformer from config.json with use_prope=True baked in,
then loads SFT checkpoint weights via load_state_dict. Provides text encoding and
flow/x0 prediction for the 48-step ODE solver.
"""

import os
import types

import torch
from torch import nn
from wan_utils.scheduler import FlowMatchScheduler, SchedulerInterface

from minwm.modeling.wan21 import (
    CausalWan21Model,
    GanAttentionBlock,
    RegisterTokens,
    Wan21Model,
)
from minwm.modeling.wan21.t5 import umt5_xxl
from minwm.modeling.wan21.tokenizers import HuggingfaceTokenizer


def _wan_base(model_root: str | None, model_name: str) -> str:
    """Resolve the Wan base-model directory that ships config.json + T5 + VAE.

    Args:
        model_root (str | None): parent dir holding the model folder. When
            ``None`` it falls back to ``$MINWM_WAN_BASE`` and then to
            ``"wan_models"``.
        model_name (str): model folder name (e.g. ``Wan2.1-T2V-1.3B``).

    Returns:
        str: ``<root>/<model_name>`` — defaults to ``wan_models/Wan2.1-T2V-1.3B``,
        the path the ODE tools' documented symlink already points at, so leaving
        the environment unset reproduces the previous hard-coded behaviour.
    """
    root = model_root or os.environ.get("MINWM_WAN_BASE", "wan_models")
    return os.path.join(root, model_name)


class Wan21TextEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "Wan2.1-T2V-1.3B", model_root: str | None = None) -> None:
        super().__init__()

        base = _wan_base(model_root, model_name)

        self.text_encoder = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            .eval()
            .requires_grad_(False)
        )
        self.text_encoder.load_state_dict(
            torch.load(
                os.path.join(base, "models_t5_umt5-xxl-enc-bf16.pth"),
                map_location="cpu",
                weights_only=False,
            )
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(base, "google/umt5-xxl/"), seq_len=512, clean="whitespace"
        )

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: list[str]) -> dict:
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {"prompt_embeds": context}


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=8.0,
        is_causal=False,
        local_attn_size=-1,
        sink_size=0,
        use_camera=False,
        model_root: str | None = None,
    ):
        super().__init__()

        # trailing sep matches the previous f"wan_models/{model_name}/" exactly
        base = os.path.join(_wan_base(model_root, model_name), "")

        if is_causal:
            # Build architecture from config.json only; the real weights are
            # loaded from the SFT checkpoint (--generator_ckpt) via
            # load_state_dict afterwards, so no diffusion_pytorch_model.* is
            # needed in wan_models/ (it ships only config.json + T5 + VAE).
            # use_prope must be baked in at construction so the per-block
            # prope_o projections exist and match the SFT checkpoint keys.
            cfg = CausalWan21Model.load_config(base)
            self.model = CausalWan21Model.from_config(
                cfg, local_attn_size=local_attn_size, sink_size=sink_size, use_prope=use_camera
            )
        else:
            self.model = Wan21Model.from_pretrained(base)
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal
        self.use_camera = use_camera

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = None  # dynamically computed from input shape
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class),
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
        )

        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int | None = None,
        classify_mode: bool | None = False,  # DF
        concat_time_embeddings: bool | None = False,  # DF
        clean_x: torch.Tensor | None = None,  # TF
        # for TF clean GT, if also noisy and needs denoising, aug_t is its timestep
        aug_t: torch.Tensor | None = None,
        cache_start: int | None = None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        prope_kv_cache: list[dict] | None = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        # Dynamically compute seq_len from input: [B, F, C, H, W]
        # After patch_embedding (2x2 spatial), tokens = F * (H/2) * (W/2)
        _, F, _, H, W = noisy_image_or_video.shape
        seq_len = F * (H // 2) * (W // 2)

        logits = None

        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                viewmats=viewmats,
                Ks=Ks,
                prope_kv_cache=prope_kv_cache,
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),  # => [B, C, F, H, W]
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),  # => [B, C, F, H, W]
                    aug_t=aug_t,
                    viewmats=viewmats,
                    Ks=Ks,
                ).permute(0, 2, 1, 3, 4)
            else:
                # diffusion forcing or bidirectional
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                        viewmats=viewmats,
                        Ks=Ks,
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=seq_len,
                        viewmats=viewmats,
                        Ks=Ks,
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler
        )
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
