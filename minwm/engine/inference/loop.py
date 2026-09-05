"""Generation loops (denoising strategies) built around plain ``dict`` batches."""

from contextlib import AbstractContextManager, nullcontext

import torch
import torch.nn as nn
from torch import Tensor

from .samplers import CMSolver, EulerSolver, UniPCSolver


def _inference_autocast(
    adapter, dtype: torch.dtype, device: torch.device
) -> AbstractContextManager:
    """The half-precision autocast context an adapter's ``denoise`` runs under.

    Families whose text / time embedders build fp32 tensors internally and train
    under half autocast (HY) set :attr:`ModelAdapter.wants_inference_autocast` so
    inference matches; models that keep an explicit fp32 path (Wan) get a
    ``nullcontext``. Shared by both generation loops so neither special-cases a
    family.

    Args:
        adapter: the model adapter (read for ``wants_inference_autocast``).
        dtype (torch.dtype): the working dtype.
        device (torch.device): the compute device.

    Returns:
        AbstractContextManager: a ``torch.autocast`` context or ``nullcontext``.
    """
    if (
        adapter.wants_inference_autocast
        and dtype in (torch.float16, torch.bfloat16)
        and device.type == "cuda"
    ):
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


class BidirectionalGenerationLoop:
    """Full-clip bidirectional diffusion loop with CFG sampling."""

    def __init__(
        self,
        generator: nn.Module,
        vae,
        sampler: UniPCSolver,
        adapter,
        guidance_scale: float = 1.0,
    ) -> None:
        self.generator = generator
        self.vae = vae
        self.sampler = sampler
        self.adapter = adapter
        self.guidance_scale = guidance_scale

    @classmethod
    def from_config(
        cls, gen: dict, cfg: dict, *, generator, vae, adapter, sampler
    ) -> "BidirectionalGenerationLoop":
        """Build from the inference config with caller-injected components.

        Args:
            gen (dict): the ``inference`` config node.
            cfg (dict): the full config (unused here; kept for a uniform signature).
            generator: the built DiT.
            vae: the built video VAE (or None).
            adapter: the built model adapter.
            sampler: the built full-diffusion sampler.

        Returns:
            BidirectionalGenerationLoop: the assembled loop.

        Raises:
            KeyError: if no adapter is configured.
        """
        if adapter is None:
            raise KeyError("BidirectionalGenerationLoop requires an adapter config")
        return cls(
            generator=generator,
            vae=vae,
            sampler=sampler,
            adapter=adapter,
            guidance_scale=gen.get("guidance_scale", 1.0),
        )

    @torch.no_grad()
    def generate(self, batch: dict) -> dict:
        """Generate an entire latent video at once, without causal KV caches."""

        noise: Tensor = batch["noise"]
        device = noise.device
        dtype = noise.dtype
        batch_size, num_frames = noise.shape[:2]

        # The adapter owns conditioning assembly for both families: Wan encodes the
        # batch prompts (and its negative prompt) into a ``context`` list, HY pulls
        # the pre-encoded multi-modal stream out of the batch. The ``forward``
        # closure splats whichever dict it gets into ``adapter.denoise``.
        shared = {
            "clean": batch.get("clean"),
            "viewmats": batch.get("viewmats"),
            "Ks": batch.get("Ks"),
        }
        cond = self.adapter.conditioning(batch, batch_size, device)
        # CFG needs a second, unconditional forward per step; at guidance_scale <= 1
        # the uncond term cancels (flow == f_cond), so skip that forward entirely.
        use_cfg = self.guidance_scale > 1.0
        uncond = self.adapter.null_conditioning(batch, batch_size, device) if use_cfg else None

        def forward(noisy: Tensor, timestep: Tensor) -> Tensor:
            flow = self.adapter.denoise(
                self.generator, noisy=noisy, timestep=timestep, **shared, **cond
            )
            if not use_cfg:
                return flow
            f_uncond = self.adapter.denoise(
                self.generator, noisy=noisy, timestep=timestep, **shared, **uncond
            )
            return f_uncond + self.guidance_scale * (flow - f_uncond)

        with _inference_autocast(self.adapter, dtype, device):
            latents = self.sampler.step(
                forward=forward,
                noise=noise,
                batch_size=batch_size,
                num_frames=num_frames,
                device=device,
                fixed_prefix=batch.get("initial_latent"),
            )

        out: dict = {"latents": latents}
        if self.vae is not None:
            video = self.adapter.decode_latents(self.vae, latents)
            out["video"] = (video * 0.5 + 0.5).clamp(0, 1)
        return out


class ARGenerationLoop:
    """KV-cached autoregressive chunk-by-chunk generation for every model family.

    One deployment loop drives Wan and HunyuanVideo, few-step and teacher-forcing
    alike — the deployment counterpart of the training
    :class:`~minwm.sampling.rollouts.SelfForcingPipeline`. The loop owns
    the outer block loop, the dual-cache CFG wiring, and the per-block cache
    refresh; the model-specific pieces (KV-cache shape, cached AR forward,
    cache-refresh pass) are delegated to the adapter's ``rollout_*`` hooks, and
    the per-block denoise math is delegated to the sampler's ``step``:

    - :class:`~minwm.engine.inference.samplers.CMSolver` — Wan DMD/CD/ODE
      (predict_x0 + renoise, CFG off).
    - :class:`~minwm.engine.inference.samplers.UniPCSolver` — Wan teacher-forcing
      (UniPC, CFG on).
    - :class:`~minwm.engine.inference.samplers.EulerSolver` — HunyuanVideo
      few-step *and* teacher-forcing (Euler walk; the two differ only in step
      count / guidance scale).

    CFG runs a second, independent KV cache built from the unconditional
    conditioning; it is enabled when ``guidance_scale > 1``.

    Args:
        generator (nn.Module): the AR transformer.
        vae: the video VAE (or None to skip decode).
        adapter: the model call-convention bridge exposing the ``rollout_*`` hooks.
        sampler: the per-block denoise strategy (exposes ``step``).
        guidance_scale (float): CFG scale; ``<= 1`` disables the uncond branch.
        num_frame_per_block (int): latent frames per causal block.
        context_noise (int): timestep for the per-block cache-refresh pass.
        offload_before_decode (bool): move the generator to CPU before the VAE
            decode (HunyuanVideo's temporal-only VAE needs the card to itself).
    """

    def __init__(
        self,
        generator: nn.Module,
        vae,
        adapter,
        sampler: CMSolver | EulerSolver | UniPCSolver,
        guidance_scale: float = 1.0,
        num_frame_per_block: int = 1,
        context_noise: int = 0,
        offload_before_decode: bool = False,
    ) -> None:
        self.generator = generator
        self.vae = vae
        self.adapter = adapter
        self.sampler = sampler
        self.guidance_scale = guidance_scale
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.offload_before_decode = offload_before_decode

    @classmethod
    def from_config(
        cls, gen: dict, cfg: dict, *, generator, vae, adapter, sampler
    ) -> "ARGenerationLoop":
        """Build from the inference config with caller-injected components.

        Args:
            gen (dict): the ``inference`` config node.
            cfg (dict): the full config (read for the model's ``num_frame_per_block``).
            generator: the built AR DiT.
            vae: the built video VAE (or None).
            adapter: the built model adapter.
            sampler: the built per-block sampler.

        Returns:
            ARGenerationLoop: the assembled loop.

        Raises:
            KeyError: if no adapter is configured.
        """
        if adapter is None:
            raise KeyError("ARGenerationLoop requires an adapter config")
        return cls(
            generator=generator,
            vae=vae,
            adapter=adapter,
            sampler=sampler,
            guidance_scale=gen.get("guidance_scale", 1.0),
            # AR chunk size is a property of the causal model, so it is the single
            # source of truth on the model config (both Wan and HY carry it there).
            num_frame_per_block=cfg.get("model", {}).get("num_frame_per_block", 1),
            context_noise=gen.get("context_noise", 0),
            offload_before_decode=gen.get("offload_before_decode", False),
        )

    @torch.no_grad()
    def generate(self, batch: dict) -> dict:
        """Autoregressively roll out the latent video block by block.

        Expected keys: ``noise`` ``[B,F,C,H,W]``, the family's conditioning
        (Wan ``prompts`` or the HY pre-encoded stream), and optional ``viewmats``
        / ``Ks`` / ``action``.
        """

        noise: Tensor = batch["noise"]
        device = noise.device
        dtype = noise.dtype
        batch_size, num_frames = noise.shape[:2]
        initial_latent = batch.get("initial_latent")
        prefix_frames = 0 if initial_latent is None else initial_latent.shape[1]
        if prefix_frames >= num_frames:
            raise ValueError("initial_latent must leave at least one frame to generate")
        if (num_frames - prefix_frames) % self.num_frame_per_block != 0:
            raise ValueError(
                "generated frames after initial_latent must be divisible by "
                f"num_frame_per_block={self.num_frame_per_block}"
            )
        height, width = noise.shape[3:]

        patch_h, patch_w = self.generator.patch_size[1], self.generator.patch_size[2]
        frame_seqlen = (height // patch_h) * (width // patch_w)
        seq_len = num_frames * frame_seqlen

        num_blocks = (num_frames - prefix_frames) // self.num_frame_per_block

        cond = self.adapter.conditioning(batch, batch_size, device)
        use_cfg = self.guidance_scale > 1.0
        uncond = self.adapter.null_conditioning(batch, batch_size, device) if use_cfg else None

        viewmats = batch.get("viewmats")
        Ks = batch.get("Ks")
        action = batch.get("action")
        # Wan sizes its self-attn / PRoPE KV buffers; PRoPE is allocated only when
        # the model uses it *and* camera is present. HY ignores both flags (its
        # vision KV grows by concat).
        use_prope_cache = viewmats is not None

        def _init_cache(cond_stream: dict) -> dict:
            return self.adapter.rollout_init_cache(
                self.generator,
                batch_size=batch_size,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                cond=cond_stream,
                use_prope_cache=use_prope_cache,
                use_local_window=True,
            )

        with _inference_autocast(self.adapter, dtype, device):
            # CFG runs two independent KV caches; ``branches`` pairs each
            # conditioning stream with its cache so the block loop drives both by
            # iteration instead of a mirrored cond/uncond code path. The cond
            # branch is first, so ``forward`` reads ``flows[0]`` as conditional.
            branches = [(cond, _init_cache(cond))]
            if use_cfg:
                branches.append((uncond, _init_cache(uncond)))

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
                vm_prefix = viewmats[:, :prefix_frames] if viewmats is not None else None
                ks_prefix = Ks[:, :prefix_frames] if Ks is not None else None
                act_prefix = action[:, :prefix_frames] if action is not None else None
                for cond_stream, cache in branches:
                    self.adapter.rollout_refresh_cache(
                        self.generator,
                        clean_block=initial_latent,
                        cache=cache,
                        cond=cond_stream,
                        meta=prefix_meta,
                        viewmats=vm_prefix,
                        Ks=ks_prefix,
                        action=act_prefix,
                    )
            for block_idx in range(num_blocks):
                f_start = prefix_frames + block_idx * self.num_frame_per_block
                f_end = f_start + self.num_frame_per_block
                frame_slice = slice(f_start, f_end)
                vm_chunk = viewmats[:, frame_slice] if viewmats is not None else None
                ks_chunk = Ks[:, frame_slice] if Ks is not None else None
                act_chunk = action[:, frame_slice] if action is not None else None
                meta = {
                    "block_idx": block_idx,
                    "f_start": f_start,
                    "f_end": f_end,
                    "frame_seqlen": frame_seqlen,
                    "seq_len": seq_len,
                    "current_start": f_start * frame_seqlen,
                    "context_noise": self.context_noise,
                }

                def forward(noisy_block: Tensor, timestep: Tensor) -> Tensor:
                    flows = [
                        self.adapter.rollout_forward(
                            self.generator,
                            noisy_block=noisy_block,
                            timestep=timestep,
                            cache=cache,
                            cond=cond_stream,
                            meta=meta,
                            viewmats=vm_chunk,
                            Ks=ks_chunk,
                            action=act_chunk,
                        )
                        for cond_stream, cache in branches
                    ]
                    if not use_cfg:
                        return flows[0]
                    flow, flow_uncond = flows
                    return flow_uncond + self.guidance_scale * (flow - flow_uncond)

                clean = self.sampler.step(
                    forward=forward,
                    noise=noise[:, frame_slice],
                    batch_size=batch_size,
                    num_frames=self.num_frame_per_block,
                    device=device,
                )

                output[:, frame_slice] = clean
                for cond_stream, cache in branches:
                    self.adapter.rollout_refresh_cache(
                        self.generator,
                        clean_block=clean,
                        cache=cache,
                        cond=cond_stream,
                        meta=meta,
                        viewmats=vm_chunk,
                        Ks=ks_chunk,
                        action=act_chunk,
                    )

        # Free the (possibly large, fully-grown) KV caches before the VAE decode:
        # decode's temporal conv3d wants a big contiguous allocation and the
        # rollout caches are dead weight past this point. Drop the reference
        # (rather than ``del``) so the ``forward`` closure's captured name stays
        # bound.
        branches = None

        out: dict = {"latents": output}
        if self.vae is not None:
            # HY's VAE decode runs a temporal conv3d over the full window and only
            # tiles spatially, so a busy card can OOM with the generator resident.
            # Offload it to CPU for the decode (the reference does the same), then
            # restore it for the next prompt.
            offload = self.offload_before_decode and device.type == "cuda"
            if offload:
                self.generator.to("cpu")
                torch.cuda.empty_cache()
            video = self.adapter.decode_latents(self.vae, output)
            if offload:
                self.generator.to(device)
            out["video"] = (video * 0.5 + 0.5).clamp(0, 1)
        return out
