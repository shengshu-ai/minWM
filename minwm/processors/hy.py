"""HunyuanVideo i2v conditioning preprocessor for inference.

Encodes the *positive* conditioning stream for one generation batch:
    - ``image_cond``: VAE-encoded first frame latent ``[B, C, 1, h, w]``
    - ``vision_states``: SigLIP features of the first frame
    - ``prompt_embed`` / ``prompt_mask``: LLM text-encoder output
    - ``byt5_text_states`` / ``byt5_text_mask``: ByT5 glyph stream (all-zeros when
      the caption has no quoted glyph text — the official default)

The CFG *negative* stream is not built here: it is supplied by the adapter from a
pre-encoded ``.pt`` (:meth:`~minwm.modeling.hy15.adapter.HYAdapter._load_neg`).
"""

import re

import numpy as np
import torch

from .base import InferenceInputPreprocessor, InferenceRuntime

__all__ = ["HYConditioning"]

_BYT5_EMBED_DIM = 1472
_BYT5_MAX_LENGTH = 256


def _load_image(path: str, target_height: int, target_width: int) -> torch.Tensor:
    """Load and resize one RGB image to ``[3, H, W]`` in ``[-1, 1]``.

    Args:
        path (str): image file path.
        target_height (int): output pixel height.
        target_width (int): output pixel width.

    Returns:
        Tensor: the image as ``[3, H, W]`` float in ``[-1, 1]``.
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img = img.resize((target_width, target_height), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor / 255.0 - 0.5) * 2.0


class HYConditioning(InferenceInputPreprocessor):
    """Encode image + caption into the HY positive conditioning stream.

    Built like any other preprocessor from an ``input_preprocessors`` spec (no
    constructor args). Its model components — ``vae`` / ``text_encoder`` /
    ``vision_encoder`` — are recipe-built and read from
    :attr:`InferenceRuntime.components` at call time, not stored at construction.
    """

    @staticmethod
    def _target_size(vae, cfg: dict) -> tuple[int, int]:
        ffactor = getattr(vae, "ffactor_spatial", 16)
        _, latent_h, latent_w = cfg["latent_shape"]
        height = cfg.get("target_height", int(latent_h) * ffactor)
        width = cfg.get("target_width", int(latent_w) * ffactor)
        return int(height), int(width)

    @torch.no_grad()
    def __call__(self, batch: dict, runtime: InferenceRuntime) -> dict:
        """Populate the positive HY conditioning fields on ``batch``.

        Args:
            batch (dict): must carry ``prompts`` (captions) and ``image`` (a path
                or list of paths, one per prompt).
            runtime (InferenceRuntime): device / dtype / inference config, plus the
                ``vae`` / ``text_encoder`` / ``vision_encoder`` in
                :attr:`~InferenceRuntime.components`.

        Returns:
            dict: ``batch`` with ``image_cond`` / ``vision_states`` /
            ``prompt_embed`` / ``prompt_mask`` / ``byt5_text_states`` /
            ``byt5_text_mask`` set.

        Raises:
            KeyError: if ``batch`` has no ``image`` entry, or the runtime is
                missing a required component.
        """
        device = runtime.device
        dtype = runtime.dtype
        vae = runtime.components["vae"]
        text_encoder = runtime.components["text_encoder"]
        vision_encoder = runtime.components["vision_encoder"]
        prompts = batch["prompts"]
        batch_size = len(prompts)

        image_paths = batch.get("image")
        if image_paths is None:
            raise KeyError("HY i2v inference requires batch['image'] (path or list)")
        if isinstance(image_paths, str):
            image_paths = [image_paths] * batch_size

        target_h, target_w = self._target_size(vae, runtime.inference_cfg)
        images = torch.stack([_load_image(p, target_h, target_w) for p in image_paths]).to(
            device=device
        )

        vae_dtype = next(vae.parameters()).dtype
        latent_dist = vae.encode(images.unsqueeze(2).to(vae_dtype)).latent_dist
        image_cond = latent_dist.mode() * vae.config.scaling_factor
        batch["image_cond"] = image_cond.to(dtype)

        vision_states = []
        for img in images:
            img_np = ((img + 1.0) * 127.5).clamp(0, 255).cpu().numpy()
            img_np = img_np.transpose(1, 2, 0).astype(np.uint8)
            vision_states.append(vision_encoder.encode_images(img_np).last_hidden_state)
        batch["vision_states"] = torch.cat(vision_states, dim=0).to(device=device, dtype=dtype)

        text_len = text_encoder.max_length
        embeds, masks = [], []
        for prompt in prompts:
            tokens = text_encoder.text2tokens(prompt, data_type="video", max_length=text_len)
            out = text_encoder.encode(tokens, data_type="video", device=device)
            embeds.append(out.hidden_state.to(dtype=dtype))
            masks.append(out.attention_mask)
        batch["prompt_embed"] = torch.cat(embeds, dim=0).to(device)
        if masks[0] is not None:
            batch["prompt_mask"] = torch.cat(masks, dim=0).to(device)

        batch["byt5_text_states"] = self._byt5(prompts, device, dtype)
        batch["byt5_text_mask"] = torch.zeros(
            batch_size, _BYT5_MAX_LENGTH, device=device, dtype=torch.int64
        )
        return batch

    def _byt5(self, prompts: list[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the ByT5 glyph stream — all-zeros unless a caption quotes glyph text."""
        if any(re.search(r'"[^"]+"', p) for p in prompts):
            raise NotImplementedError(
                "captions contain quoted glyph text; the ByT5 glyph encoder is not "
                "wired into inference (positive glyph defaults to zeros)"
            )
        return torch.zeros(
            len(prompts), _BYT5_MAX_LENGTH, _BYT5_EMBED_DIM, device=device, dtype=dtype
        )
