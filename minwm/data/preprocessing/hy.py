"""Shared HunyuanVideo 1.5 latent-extraction logic for data preprocessing.

Wraps the HY15 VAE + SigLIP vision encoder + LLM text encoder + ByT5 glyph
encoder into a single :class:`VideoLatentExtractor`. The ``tools/data/hy/``
scripts own only argparse + the distributed loop; the encode core lives here.

Two frame-selection modes share one encode path:

* sequential (``camera_indices=None``) — load the first ``max_frames`` frames;
* camera-aligned (``camera_indices`` given) — sample frames aligned to the VAE's
  4x temporal downsampling via :func:`sample_frame_indices`, and additionally
  return the chosen ``frame_indices``.

``decord`` (preferred) / ``cv2`` are imported lazily inside :func:`load_video`.
"""

import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from minwm.modeling.hy15 import AutoencoderKLConv3D
from minwm.modeling.hy15.encoders import (
    PROMPT_TEMPLATE,
    MultilingualPromptFormat,
    TextEncoder,
    VisionEncoder,
    load_glyph_byT5_v2,
)

__all__ = ["VideoLatentExtractor", "load_video", "sample_frame_indices"]

_BYT5_EMBED_DIM = 1472


def sample_frame_indices(camera_indices: list[int], max_frames: int = 77) -> list[int] | None:
    """Sample ``max_frames`` frame indices aligned to VAE 4x temporal downsampling.

    Selects ``n_select = (max_frames - 1) // 4 + 1`` camera frames from the middle
    of ``camera_indices``, then fills three uniformly-spaced frames between each
    adjacent pair so every group of four frames encodes to one latent whose last
    frame is a camera frame. Total length is ``1 + (n_select - 1) * 4 == max_frames``.

    Args:
        camera_indices (list[int]): available camera frame indices (deduped/sorted
            internally).
        max_frames (int): target number of video frame indices to return.

    Returns:
        list[int] | None: ``max_frames`` integer frame indices, or ``None`` if
        there are fewer than ``n_select`` camera frames.
    """
    cam = sorted(set(int(x) for x in camera_indices))
    n_cam = len(cam)
    n_select = (max_frames - 1) // 4 + 1

    if n_cam < n_select:
        return None

    start = (n_cam - n_select) // 2
    selected = cam[start : start + n_select]

    result = [selected[0]]
    for i in range(1, len(selected)):
        a, b = selected[i - 1], selected[i]
        for j in range(1, 4):
            result.append(int(round(a + j * (b - a) / 4)))
        result.append(b)
    return result


def load_video(
    video_path: str,
    target_height: int,
    target_width: int,
    frame_indices: list[int] | None = None,
    max_frames: int | None = None,
) -> torch.Tensor:
    """Load and resize video frames, returning a ``[C, F, H, W]`` tensor in ``[-1, 1]``.

    Reads with ``decord`` when available, else ``cv2``. When ``frame_indices`` is
    given those exact frames are read (out-of-range indices dropped); otherwise the
    first ``max_frames`` frames are read (all frames if ``max_frames`` is ``None``).

    Args:
        video_path (str): path to the video file.
        target_height (int): output frame height.
        target_width (int): output frame width.
        frame_indices (list[int] | None): explicit frame indices to read.
        max_frames (int | None): cap on sequential frames when ``frame_indices``
            is ``None``.

    Returns:
        torch.Tensor: ``[3, F, H, W]`` float tensor normalized to ``[-1, 1]``.

    Raises:
        ValueError: if no frames could be read from ``video_path``.
    """
    try:
        import decord

        decord.bridge.set_bridge("torch")
        use_decord = True
    except ImportError:
        import cv2

        use_decord = False

    if use_decord:
        vr = decord.VideoReader(video_path)
        if frame_indices is None:
            n = len(vr) if max_frames is None else min(max_frames, len(vr))
            frame_indices = list(range(n))
        else:
            frame_indices = [i for i in frame_indices if i < len(vr)]
        frames = vr.get_batch(frame_indices)
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()
        video_tensor = torch.from_numpy(frames).float().permute(3, 0, 1, 2)
    else:
        cap = cv2.VideoCapture(video_path)
        frames = []
        if frame_indices is None:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if max_frames and len(frames) >= max_frames:
                    break
        else:
            for fi in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError(f"No frames loaded from {video_path}")
        video_tensor = torch.from_numpy(np.stack(frames)).float().permute(3, 0, 1, 2)

    video_tensor = F.interpolate(
        video_tensor.unsqueeze(0),
        size=(video_tensor.shape[1], target_height, target_width),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)

    video_tensor = (video_tensor / 255.0 - 0.5) * 2.0
    return video_tensor


class VideoLatentExtractor:
    """Extract HY15 latents + conditioning from videos and captions.

    Loads the VAE, SigLIP vision encoder, LLM text encoder, and ByT5 glyph
    encoder once, then :meth:`extract` encodes one (video, caption) pair into the
    ``.pt`` payload consumed by the HY datasets.

    Args:
        checkpoint_path (str): HunyuanVideo-1.5 checkpoint root.
        device (int | str | torch.device): device for the encoders.
        target_size (tuple[int, int]): ``(height, width)`` to resize frames to.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device,
        target_size: tuple[int, int] = (480, 832),
    ):
        self.device = device
        self.target_size = target_size
        self._load_models(checkpoint_path)

    def _load_models(self, checkpoint_path: str) -> None:
        self.vae = (
            AutoencoderKLConv3D.from_pretrained(
                os.path.join(checkpoint_path, "vae"), torch_dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )

        self.vision_encoder = VisionEncoder(
            vision_encoder_type="siglip",
            vision_encoder_precision="fp16",
            vision_encoder_path=os.path.join(checkpoint_path, "vision_encoder/siglip"),
            processor_type=None,
            processor_path=None,
            output_key=None,
            logger=None,
            device=self.device,
        )

        self.text_encoder = TextEncoder(
            text_encoder_type="llm",
            tokenizer_type="llm",
            text_encoder_path=os.path.join(checkpoint_path, "text_encoder/llm"),
            max_length=1000,
            text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=None,
            device=self.device,
        )
        self.text_len = self.text_encoder.max_length

        load_from = os.path.join(checkpoint_path, "text_encoder")
        glyph_root = os.path.join(load_from, "Glyph-SDXL-v2")
        byt5_args = dict(
            byT5_google_path=os.path.join(load_from, "byt5-small"),
            byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
            multilingual_prompt_format_color_path=os.path.join(glyph_root, "assets/color_idx.json"),
            multilingual_prompt_format_font_path=os.path.join(
                glyph_root, "assets/multilingual_10-lang_idx.json"
            ),
            byt5_max_length=256,
        )
        byt5_kwargs = load_glyph_byT5_v2(
            byt5_args,
            device=f"cuda:{self.device}" if isinstance(self.device, int) else str(self.device),
        )
        self.prompt_format = MultilingualPromptFormat(
            font_path=byt5_args["multilingual_prompt_format_font_path"],
            color_path=byt5_args["multilingual_prompt_format_color_path"],
        )
        self.byt5_model = byt5_kwargs["byt5_model"]
        self.byt5_tokenizer = byt5_kwargs["byt5_tokenizer"]
        self.byt5_max_length = byt5_kwargs["byt5_max_length"]

    def _process_byt5_prompt(self, prompt_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode quoted glyph text via ByT5; zeros when the caption has none."""
        byt5_embeddings = torch.zeros(
            (1, self.byt5_max_length, _BYT5_EMBED_DIM), device=self.device
        )
        byt5_mask = torch.zeros((1, self.byt5_max_length), device=self.device, dtype=torch.int64)

        pattern = r'\"(.*?)\"|"(.*?)"'
        matches = re.findall(pattern, prompt_text)
        glyph_texts = [m[0] or m[1] for m in matches]
        glyph_texts = list(dict.fromkeys(glyph_texts)) if len(glyph_texts) > 1 else glyph_texts

        if glyph_texts:
            text_styles = [{"color": None, "font-family": None} for _ in glyph_texts]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)
            inputs = self.byt5_tokenizer(
                formatted_text,
                padding="max_length",
                max_length=self.byt5_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_ids = inputs.input_ids.to(self.device)
            text_mask = inputs.attention_mask.to(self.device)
            byt5_embeddings = self.byt5_model(text_ids, attention_mask=text_mask.float())[0]
            byt5_mask = text_mask

        return byt5_embeddings, byt5_mask

    @torch.no_grad()
    def extract(
        self,
        video_path: str,
        caption: str,
        camera_indices: list[int] | None = None,
        max_frames: int | None = None,
    ) -> dict | None:
        """Encode one (video, caption) pair into the HY ``.pt`` payload.

        Args:
            video_path (str): path to the source video.
            caption (str): text caption (drives both LLM and ByT5 branches).
            camera_indices (list[int] | None): when given, frames are sampled via
                :func:`sample_frame_indices` and the chosen ``frame_indices`` are
                returned; when ``None``, the first ``max_frames`` frames are used.
            max_frames (int | None): frame budget. Required (e.g. 77) when
                ``camera_indices`` is given.

        Returns:
            dict | None: payload with ``latent``, ``image_cond``, ``prompt_embeds``,
            ``prompt_mask``, ``vision_states``, ``byt5_text_states``,
            ``byt5_text_mask`` (plus ``frame_indices`` in camera mode); ``None`` if
            camera mode lacks enough camera frames.
        """
        if camera_indices is not None:
            frame_indices = sample_frame_indices(camera_indices, max_frames)
            if frame_indices is None:
                return None
        else:
            frame_indices = None

        video_tensor = load_video(
            video_path,
            self.target_size[0],
            self.target_size[1],
            frame_indices=frame_indices,
            max_frames=max_frames,
        ).to(self.device)

        video_input = video_tensor.unsqueeze(0)
        video_latents = self.vae.encode(video_input).latent_dist.mode()
        video_latents = video_latents * self.vae.config.scaling_factor

        image_cond = video_latents[:, :, 0:1, :, :]

        first_frame = video_tensor[:, 0, :, :]
        first_frame_uint8 = ((first_frame + 1) * 127.5).clamp(0, 255)
        first_frame_np = first_frame_uint8.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        vision_states = self.vision_encoder.encode_images(first_frame_np)
        vision_states = vision_states.last_hidden_state.to(device=self.device, dtype=torch.bfloat16)

        text_inputs = self.text_encoder.text2tokens(
            caption, data_type="video", max_length=self.text_len
        )
        prompt_outputs = self.text_encoder.encode(
            text_inputs, data_type="video", device=self.device
        )
        prompt_embeds = prompt_outputs.hidden_state.to(
            dtype=self.text_encoder.dtype, device=self.device
        )
        attention_mask = (
            prompt_outputs.attention_mask.to(self.device)
            if prompt_outputs.attention_mask is not None
            else None
        )

        byt5_embeddings, byt5_masks = self._process_byt5_prompt(caption)

        out = {
            "latent": video_latents.to(torch.bfloat16),
            "image_cond": image_cond.to(torch.bfloat16),
            "prompt_embeds": prompt_embeds,
            "prompt_mask": attention_mask,
            "vision_states": vision_states,
            "byt5_text_states": byt5_embeddings,
            "byt5_text_mask": byt5_masks,
        }
        if frame_indices is not None:
            out["frame_indices"] = frame_indices
        return out
