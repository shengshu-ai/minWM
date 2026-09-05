"""Self-forcing backward-simulation inference pipeline.

One model-agnostic autoregressive denoising loop drives every model family. The
pipeline owns the block/step structure, the ``x0`` conversion, the exit-step
truncated-BPTT gradient gate, and the re-noise to the next timestep; the
:class:`~minwm.modeling.adapter.ModelAdapter` owns everything model-specific via
three rollout hooks (``rollout_init_cache`` / ``rollout_forward`` /
``rollout_refresh_cache``): the KV-cache shape, the cached AR forward, and the
per-block cache-refresh.

Latents are ``[B, F, C, H, W]``; the adapter hooks convert to each model's own
layout internally.
"""

import torch
import torch.distributed as dist

from minwm.distributed import get_rng_states_tracker

from .schedulers import FlowMatchingScheduler

__all__ = [
    "SelfForcingPipeline",
    "sample_exit_step",
]


class SelfForcingPipeline:
    """Multi-step KV-cached denoising loop for DMD backward simulation.

    The loop is identical across model families; the model-specific pieces are
    delegated to the adapter's rollout hooks. Wan and HY differ only in those
    hooks (cache topology, forward call, cache-refresh pass).

    Args:
        generator (nn.Module): the trainable generator being rolled out.
        scheduler (FlowMatchingScheduler): shifted flow-matching noise scheduler.
        adapter (ModelAdapter): model call-convention bridge.
        denoising_step_list (list[int]): timestep values most-noisy -> least-noisy.
        num_frame_per_block (int): frames per causal temporal block.
        context_noise (int): timestep used by the adapter's cache-refresh pass.
        use_prope_cache (bool): whether adapters should allocate PRoPE KV cache.
    """

    def __init__(
        self,
        generator,
        scheduler: FlowMatchingScheduler,
        adapter,
        denoising_step_list: list[int] | None = None,
        num_frame_per_block: int = 1,
        context_noise: int = 0,
        use_prope_cache: bool = False,
    ):
        self.generator = generator
        self.scheduler = scheduler
        self.adapter = adapter
        self.denoising_step_list = [t for t in (denoising_step_list or []) if t != 0]
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.use_prope_cache = use_prope_cache

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        cond: dict,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        initial_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, int]:
        """Autoregressive denoising loop with exit-step truncated backprop."""

        B, F, _C, H, W = noise.shape
        device = noise.device
        dtype = noise.dtype
        patch_h, patch_w = self.generator.patch_size[1], self.generator.patch_size[2]
        frame_seqlen = (H // patch_h) * (W // patch_w)
        seq_len = F * frame_seqlen

        prefix_frames = 0 if initial_latent is None else initial_latent.shape[1]
        if prefix_frames >= F:
            raise ValueError("initial_latent must leave at least one frame to generate")
        if (F - prefix_frames) % self.num_frame_per_block != 0:
            raise ValueError(
                "generated frames after initial_latent must be divisible by "
                f"num_frame_per_block={self.num_frame_per_block}"
            )
        num_blocks = (F - prefix_frames) // self.num_frame_per_block
        num_steps = len(self.denoising_step_list)

        with get_rng_states_tracker().fork():
            exit_step_idx = sample_exit_step(num_denoising_steps=num_steps, device=device)
        if dist.is_available() and dist.is_initialized():
            exit_tensor = torch.tensor(exit_step_idx, device=device)
            dist.broadcast(exit_tensor, src=0)
            exit_step_idx = int(exit_tensor.item())

        cache = self.adapter.rollout_init_cache(
            self.generator,
            batch_size=B,
            num_frames=F,
            frame_seqlen=frame_seqlen,
            seq_len=seq_len,
            dtype=dtype,
            device=device,
            cond=cond,
            use_prope_cache=self.use_prope_cache,
        )

        output = torch.zeros_like(noise)

        if initial_latent is not None:
            initial_latent = initial_latent.to(device=device, dtype=dtype)
            output[:, :prefix_frames] = initial_latent
            prefix_meta = {
                "block_idx": -1,
                "f_start": 0,
                "f_end": prefix_frames,
                "frame_seqlen": frame_seqlen,
                "seq_len": seq_len,
                "current_start": 0,
                "cache_start": 0,
                "context_noise": self.context_noise,
            }
            self.adapter.rollout_refresh_cache(
                self.generator,
                clean_block=initial_latent.detach(),
                cache=cache,
                cond=cond,
                meta=prefix_meta,
                viewmats=viewmats[:, :prefix_frames] if viewmats is not None else None,
                Ks=Ks[:, :prefix_frames] if Ks is not None else None,
                action=action[:, :prefix_frames] if action is not None else None,
            )

        for block_idx in range(num_blocks):
            f_start = prefix_frames + block_idx * self.num_frame_per_block
            f_end = f_start + self.num_frame_per_block
            n_f = self.num_frame_per_block

            noisy_block = noise[:, f_start:f_end]
            vm_chunk = viewmats[:, f_start:f_end] if viewmats is not None else None
            ks_chunk = Ks[:, f_start:f_end] if Ks is not None else None
            act_chunk = action[:, f_start:f_end] if action is not None else None
            meta = {
                "block_idx": block_idx,
                "f_start": f_start,
                "f_end": f_end,
                "frame_seqlen": frame_seqlen,
                "seq_len": seq_len,
                "current_start": f_start * frame_seqlen,
                "context_noise": self.context_noise,
            }
            denoised_block = None
            for step_idx, timestep_val in enumerate(self.denoising_step_list):
                t = torch.full((B, n_f), timestep_val, device=device, dtype=dtype)

                is_exit = step_idx == exit_step_idx
                with torch.set_grad_enabled(is_exit and torch.is_grad_enabled()):
                    flow_pred = self.adapter.rollout_forward(
                        self.generator,
                        noisy_block=noisy_block,
                        timestep=t,
                        cache=cache,
                        cond=cond,
                        meta=meta,
                        viewmats=vm_chunk,
                        Ks=ks_chunk,
                        action=act_chunk,
                    )

                denoised_block = self.scheduler.predict_x0(noisy_block, flow_pred, t)

                if is_exit:
                    break

                next_ts = self.denoising_step_list[step_idx + 1]
                next_t = torch.full((B * n_f,), next_ts, device=device, dtype=dtype)
                flat = denoised_block.flatten(0, 1)
                with get_rng_states_tracker().fork():
                    renoise = torch.randn_like(flat)
                noisy_block = self.scheduler.add_noise(flat, renoise, next_t).unflatten(0, (B, n_f))

            output[:, f_start:f_end] = denoised_block

            self.adapter.rollout_refresh_cache(
                self.generator,
                clean_block=denoised_block,
                cache=cache,
                cond=cond,
                meta=meta,
                viewmats=vm_chunk,
                Ks=ks_chunk,
                action=act_chunk,
            )

        # ``from_ts`` / ``to_ts`` bound the score-timestep sampling window when the
        # recipe opts in via ``use_rollout_{min,max}_timestep``. They must be in the
        # *pre-shift* index domain ``num_train_timesteps - argmin_index`` (legacy
        # ``denoised_timestep_{from,to}``, self_forcing_training.py:239-245), because
        # ``_sample_timestep`` re-applies ``score_timestep_shift`` to the drawn value.
        # Returning the already-shifted ``timesteps[argmin]`` value here would double-
        # shift (and mis-place the sampling band) the moment either flag is enabled.
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        num_train = self.scheduler.num_train_timesteps
        exit_ts = self.denoising_step_list[exit_step_idx]
        t_tensor = torch.tensor([exit_ts], dtype=dtype, device=device)
        from_ts = num_train - int((self.scheduler.timesteps - t_tensor).abs().argmin().item())
        if exit_step_idx == num_steps - 1:
            to_ts = 0
        else:
            next_exit_ts = self.denoising_step_list[exit_step_idx + 1]
            t_next = torch.tensor([next_exit_ts], dtype=dtype, device=device)
            to_ts = num_train - int((self.scheduler.timesteps - t_next).abs().argmin().item())

        return output, from_ts, to_ts


def sample_exit_step(num_denoising_steps: int, device: torch.device | None = None) -> int:
    """Sample the shared exit step index for self-forcing truncated denoising."""

    return int(torch.randint(0, num_denoising_steps, (1,), device=device).item())
