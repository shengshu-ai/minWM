# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan
# works and any output and results therefrom are provided "AS IS" without any
# express or implied warranties of any kind including any warranties of title,
# merchantability, noninfringement, course of dealing, usage of trade, or
# fitness for a particular purpose. You are solely responsible for determining
# the appropriateness of using, reproducing, modifying, performing, displaying
# or distributing any of the Tencent Hunyuan works or outputs and assume any and
# all risks associated with your or a third party's use or distribution of any
# of the Tencent Hunyuan works or outputs and your exercise of rights and
# permissions under this agreement.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SigLIP vision encoder (image features from latents) for the HY15 pipeline."""

from dataclasses import dataclass

import torch
import torch.nn as nn

PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

VISION_ENCODER_PATH = {}


def use_default(value, default):
    """Return ``value`` if not None, else ``default``."""
    return value if value is not None else default


def load_vision_encoder(
    vision_encoder_type,
    vision_encoder_precision=None,
    vision_encoder_path=None,
    logger=None,
    device=None,
):
    """Load a SigLIP vision encoder from a HuggingFace path.

    Args:
        vision_encoder_type (str): only ``"siglip"`` is supported.
        vision_encoder_precision (str, optional): dtype key.
        vision_encoder_path (str, optional): HF path; falls back to ``VISION_ENCODER_PATH``.
        logger: unused; kept for signature parity.
        device: optional target device.

    Returns:
        tuple: ``(vision_encoder, vision_encoder_path)``.

    Raises:
        ValueError: if the encoder type is unsupported.
    """
    from transformers import SiglipVisionModel

    if vision_encoder_path is None:
        vision_encoder_path = VISION_ENCODER_PATH[vision_encoder_type]

    if vision_encoder_type == "siglip":
        vision_encoder = SiglipVisionModel.from_pretrained(
            vision_encoder_path, subfolder="image_encoder"
        )
    else:
        raise ValueError(f"Unsupported vision encoder type: {vision_encoder_type}")

    if vision_encoder_precision is not None:
        vision_encoder = vision_encoder.to(dtype=PRECISION_TO_TYPE[vision_encoder_precision])

    vision_encoder.requires_grad_(False)

    if device is not None:
        vision_encoder = vision_encoder.to(device)

    return vision_encoder, vision_encoder_path


def load_image_processor(processor_type, processor_path=None, logger=None):
    """Load a SigLIP image processor from a HuggingFace path.

    Args:
        processor_type (str): only ``"siglip"`` is supported.
        processor_path (str, optional): HF path; falls back to ``VISION_ENCODER_PATH``.
        logger: unused; kept for signature parity.

    Returns:
        tuple: ``(processor, processor_path)``.

    Raises:
        ValueError: if the processor type is unsupported.
    """
    from transformers import SiglipImageProcessor

    if processor_path is None:
        processor_path = VISION_ENCODER_PATH[processor_type]

    if processor_type == "siglip":
        processor = SiglipImageProcessor.from_pretrained(
            processor_path, subfolder="feature_extractor"
        )
    else:
        raise ValueError(f"Unsupported processor type: {processor_type}")

    return processor, processor_path


@dataclass
class VisionEncoderModelOutput:
    """Vision-encoder output bundle.

    Args:
        last_hidden_state (torch.FloatTensor): last-layer hidden states ``[B, L, C]``.
        pooler_output (torch.FloatTensor, optional): pooled output ``[B, C]``.
        hidden_states (tuple[torch.FloatTensor, ...], optional): per-layer states.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


class VisionEncoder(nn.Module):
    """SigLIP-backed vision encoder.

    Args:
        vision_encoder_type (str): encoder type (must contain ``"siglip"``).
        vision_encoder_precision (str, optional): dtype key.
        vision_encoder_path (str, optional): HF model path.
        processor_type (str, optional): processor key; defaults to encoder type.
        processor_path (str, optional): HF processor path; defaults to encoder path.
        output_key (str, optional): model output attribute; defaults to
            ``"last_hidden_state"``.
        logger: unused; kept for signature parity.
        device: optional target device.
    """

    def __init__(
        self,
        vision_encoder_type: str,
        vision_encoder_precision: str | None = None,
        vision_encoder_path: str | None = None,
        processor_type: str | None = None,
        processor_path: str | None = None,
        output_key: str | None = None,
        logger=None,
        device=None,
    ):
        super().__init__()
        self.vision_encoder_type = vision_encoder_type
        self.precision = vision_encoder_precision
        self.model_path = vision_encoder_path
        self.processor_type = processor_type if processor_type is not None else vision_encoder_type
        self.processor_path = processor_path if processor_path is not None else vision_encoder_path
        self.logger = logger

        if "siglip" in vision_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        else:
            raise ValueError(f"Unsupported vision encoder type: {vision_encoder_type}")

        self.model, self.model_path = load_vision_encoder(
            vision_encoder_type=self.vision_encoder_type,
            vision_encoder_precision=self.precision,
            vision_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
        )
        self.dtype = self.model.dtype
        self.device = self.model.device

        self.processor, self.processor_path = load_image_processor(
            processor_type=self.processor_type,
            processor_path=self.processor_path,
            logger=self.logger,
        )

    def __repr__(self):
        return f"{self.vision_encoder_type} ({self.precision} - {self.model_path})"

    def encode_latents_to_images(self, latents, vae, reorg_token=False):
        """Decode latents to uint8 RGB images via the VAE.

        Args:
            latents: input latents (4D image or 5D video; first frame used).
            vae: VAE model used to decode latents.
            reorg_token (bool): unused; kept for signature parity.

        Returns:
            numpy.ndarray: decoded images ``[B, H, W, 3]`` uint8.
        """
        import numpy as np

        first_image_latents = latents[:, :, 0, ...] if len(latents.shape) == 5 else latents
        first_image_latents = 1 / vae.config.scaling_factor * first_image_latents
        first_image = vae.decode(first_image_latents.unsqueeze(2).to(vae.dtype), return_dict=False)[
            0
        ].cpu()
        first_image = first_image[:, :, 0, :, :]
        first_image = (first_image / 2 + 0.5).clamp(0, 1)
        first_image = (first_image * 255.0).clamp(0, 255.0)
        first_image = first_image.to(torch.uint8).numpy()
        first_image = first_image.transpose(0, 2, 3, 1)

        assert isinstance(first_image, np.ndarray)
        assert first_image.ndim == 4 and first_image.shape[3] == 3
        assert first_image.dtype == np.uint8

        return first_image

    def encode_images(self, images):
        """Encode images to vision features.

        Args:
            images: numpy array (preprocessed internally) or a preprocessed batch.

        Returns:
            VisionEncoderModelOutput: encoder output.
        """
        import numpy as np

        if isinstance(images, np.ndarray):
            preprocessed = self.processor.preprocess(images=images, return_tensors="pt").to(
                device=self.model.device, dtype=self.model.dtype
            )
        else:
            preprocessed = images

        outputs = self.model(**preprocessed)

        return VisionEncoderModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=(outputs.pooler_output if hasattr(outputs, "pooler_output") else None),
            hidden_states=(outputs.hidden_states if hasattr(outputs, "hidden_states") else None),
        )

    def encode_latents(self, latents, vae, reorg_token=False):
        """Decode latents to images then encode them.

        Args:
            latents: input latent tensors.
            vae: VAE used to decode latents to images.
            reorg_token (bool): unused; kept for signature parity.

        Returns:
            torch.FloatTensor: encoded image features ``[B, L, C]``.
        """
        images = self.encode_latents_to_images(latents, vae, reorg_token)
        outputs = self.encode_images(images)
        return outputs.last_hidden_state

    def forward(self, images):
        """Encode images directly.

        Args:
            images: input images.

        Returns:
            VisionEncoderModelOutput: encoder output.
        """
        return self.encode_images(images)
