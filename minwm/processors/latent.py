"""Latent and latent-noise input preparation for inference."""

import numpy as np
import torch

from .base import BatchPreprocessor, InferenceInputPreprocessor, InferenceRuntime


class LatentNoise(InferenceInputPreprocessor):
    """Create or place the ``noise`` tensor used as the generation seed."""

    def __init__(self, key: str = "noise") -> None:
        self.key = key

    def _latent_shape(self, inference_cfg: dict) -> tuple[int, int, int, int]:
        channels, height, width = inference_cfg["latent_shape"]
        return (
            int(inference_cfg["num_frames"]),
            int(channels),
            int(height),
            int(width),
        )

    def __call__(self, batch: dict, runtime: InferenceRuntime) -> dict:
        if self.key in batch and batch[self.key] is not None:
            batch[self.key] = batch[self.key].to(device=runtime.device, dtype=runtime.dtype)
            return batch

        prompts = batch["prompts"]
        num_frames, channels, height, width = self._latent_shape(runtime.inference_cfg)
        batch[self.key] = torch.randn(
            [len(prompts), num_frames, channels, height, width],
            device=runtime.device,
            dtype=runtime.dtype,
        )
        return batch


class InitialLatent(InferenceInputPreprocessor):
    """VAE-encode input images into a clean latent prefix.

    The generation pipeline consumes only ``initial_latent``; image loading and
    VAE calling stay in this input preprocessor. The supported VAE contract is
    either Wan's ``encode(list[C,T,H,W]) -> list[C,T,h,w]`` or a diffusers-style
    ``encode(B,C,T,H,W)`` result exposing ``latent_dist``.
    """

    def __init__(self, image_key: str = "image", output_key: str = "initial_latent") -> None:
        self.image_key = image_key
        self.output_key = output_key

    @staticmethod
    def _target_size(runtime: InferenceRuntime) -> tuple[int, int]:
        cfg = runtime.inference_cfg
        _, latent_h, latent_w = cfg["latent_shape"]
        return (
            int(cfg.get("target_height", int(latent_h) * 16)),
            int(cfg.get("target_width", int(latent_w) * 16)),
        )

    @staticmethod
    def _load(path: str, height: int, width: int) -> torch.Tensor:
        from PIL import Image

        image = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
        return (torch.from_numpy(array).permute(2, 0, 1) / 255.0 - 0.5) * 2.0

    @torch.no_grad()
    def __call__(self, batch: dict, runtime: InferenceRuntime) -> dict:
        source = batch.get(self.image_key)
        if source is None:
            return batch

        batch_size = len(batch["prompts"])
        height, width = self._target_size(runtime)
        if isinstance(source, str):
            images = torch.stack([self._load(source, height, width)] * batch_size)
        elif isinstance(source, (list, tuple)) and source and isinstance(source[0], str):
            if len(source) != batch_size:
                raise ValueError(f"got {len(source)} images for batch size {batch_size}")
            images = torch.stack([self._load(path, height, width) for path in source])
        elif isinstance(source, torch.Tensor):
            images = source
            if images.ndim == 3:
                images = images.unsqueeze(0)
            if images.ndim == 5:
                if images.shape[2] != 1:
                    raise ValueError("InitialLatent accepts one image per sample")
                images = images.squeeze(2)
            if images.ndim != 4:
                raise ValueError("image tensor must have shape [B,C,H,W] or [B,C,1,H,W]")
        else:
            raise TypeError("image must be a path, list of paths, or tensor")

        vae = runtime.components["vae"]
        images = images.to(device=runtime.device, dtype=runtime.dtype)
        videos = [image.unsqueeze(1) for image in images]
        try:
            encoded = vae.encode(videos)
        except (TypeError, AttributeError):
            encoded = vae.encode(torch.stack(videos))

        if isinstance(encoded, (list, tuple)):
            latent = torch.stack(list(encoded))
        else:
            latent_dist = getattr(encoded, "latent_dist", None)
            latent = latent_dist.mode() if latent_dist is not None else encoded
            scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", 1.0)
            latent = latent * scaling_factor
        if latent.ndim != 5:
            raise ValueError(f"VAE encode must return [B,C,T,H,W], got {tuple(latent.shape)}")
        batch[self.output_key] = latent.permute(0, 2, 1, 3, 4).to(runtime.dtype)
        return batch


class LatentToDevice(BatchPreprocessor):
    """Move + cast latent and camera tensors onto the working device/dtype.

    Args:
        dtype (torch.dtype, optional): working dtype for the moved tensors. Left
            ``None`` when built from config; the recipe injects its runtime dtype
            in :meth:`~minwm.engine.recipe_base.Recipe.build_preprocessors`.
        keys (tuple[str, ...]): batch keys to move if present. Defaults cover
            every recipe's latent + camera tensors.
    """

    def __init__(
        self,
        dtype: torch.dtype | None = None,
        keys: tuple[str, ...] = ("clean_latent", "ode_latent", "viewmats", "Ks"),
    ) -> None:
        self.dtype = dtype
        self.keys = keys

    def __call__(self, batch: dict, device: torch.device) -> dict:
        for key in self.keys:
            if key in batch and batch[key] is not None:
                batch[key] = batch[key].to(device=device, dtype=self.dtype)
        return batch
