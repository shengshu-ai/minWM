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
"""LLM text encoder (tokenizer + HF backbone) for the HY15 pipeline."""

import os
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn

C_SCALE = 1_000_000_000_000_000

PROMPT_TEMPLATE_ENCODE_IMAGE_JSON = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Describe the image by detailing the following"
        " aspects:         1. The main content and theme of the image.         2. The color,"
        " shape, size, texture, quantity, text, and spatial relationships of the objects."
        "         3. The background environment, light, style and atmosphere.",
    },
    {"role": "user", "content": "{}"},
]

PROMPT_TEMPLATE_ENCODE_VIDEO_JSON = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Describe the video by detailing the following"
        " aspects:         1. The main content and theme of the video.         2. The color,"
        " shape, size, texture, quantity, text, and spatial relationships of the objects."
        "         3. Actions, events, behaviors temporal relationships, physical movement"
        " changes of the objects.         4. background environment, light, style and"
        " atmosphere.         5. camera angles, movements, and transitions used in the video.",
    },
    {"role": "user", "content": "{}"},
]

PROMPT_TEMPLATE = {
    "li-dit-encode-image-json": {
        "template": PROMPT_TEMPLATE_ENCODE_IMAGE_JSON,
        "crop": -1,
    },
    "li-dit-encode-video-json": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO_JSON,
        "crop_start": -1,
    },
}

MODEL_BASE = os.getenv("MODEL_BASE", "")
TEXT_ENCODER_PATH = {}
TOKENIZER_PATH = {}

PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def use_default(value, default):
    """Return ``value`` if not None, else ``default``."""
    return value if value is not None else default


def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    logger=None,
    device=None,
):
    """Load an LLM text-encoder backbone from a HuggingFace path.

    Args:
        text_encoder_type (str): encoder type key; only ``"llm"`` is supported.
        text_encoder_precision (str, optional): one of ``"fp32"``/``"fp16"``/``"bf16"``.
        text_encoder_path (str, optional): HF path; falls back to ``TEXT_ENCODER_PATH``.
        logger: unused; kept for signature parity.
        device: optional target device.

    Returns:
        tuple: ``(text_encoder, text_encoder_path)``.

    Raises:
        ValueError: if ``text_encoder_path`` is None and the type is unknown.
    """
    from transformers import AutoModel

    if text_encoder_path is None:
        if text_encoder_type not in TEXT_ENCODER_PATH:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
        text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]

    text_encoder = AutoModel.from_pretrained(text_encoder_path, low_cpu_mem_usage=True)

    if hasattr(text_encoder, "language_model"):
        lm = text_encoder.language_model
        # Newer transformers (~5.10.1) loads Qwen2_5_VLModel whose language submodule
        # expects keys without the ``model.`` prefix, but HunyuanVideo checkpoints use
        # ``model.*`` keys. When from_pretrained fails to map them, language_model params
        # are randomly initialized instead of loaded. Detect by comparing a probe weight
        # from the checkpoint against the model's current value, and reload with the
        # prefix stripped when they differ.
        import glob

        from safetensors import safe_open
        from safetensors.torch import load_file

        shard_files = sorted(glob.glob(os.path.join(text_encoder_path, "model-*.safetensors")))
        probe_ckpt_key = "model.layers.0.self_attn.q_proj.weight"
        probe_model_key = "layers.0.self_attn.q_proj.weight"
        needs_remap = False
        for sf in shard_files:
            with safe_open(sf, framework="pt") as f:
                if probe_ckpt_key in f.keys():
                    ckpt_tensor = f.get_tensor(probe_ckpt_key)
                    model_tensor = lm.state_dict()[probe_model_key]
                    needs_remap = not torch.equal(ckpt_tensor.to(model_tensor.dtype), model_tensor)
                    break
        if needs_remap and shard_files:
            state_dict = {}
            for sf in shard_files:
                state_dict.update(load_file(sf, device="cpu"))
            remapped = {
                k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")
            }
            missing, unexpected = lm.load_state_dict(remapped, strict=False)
            print(
                "[text_encoder] from_pretrained did not correctly load language_model "
                "weights (key prefix mismatch). Reloaded with remapping model.* -> * "
                f"(matched={len(remapped) - len(unexpected)}, missing={len(missing)}, "
                f"unexpected={len(unexpected)})"
            )
        else:
            print("[text_encoder] language_model weights already loaded correctly.")
        text_encoder = lm
    text_encoder.final_layer_norm = text_encoder.norm

    if text_encoder_precision is not None:
        text_encoder = text_encoder.to(dtype=PRECISION_TO_TYPE[text_encoder_precision])

    text_encoder.requires_grad_(False)

    if device is not None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path


def load_tokenizer(tokenizer_type, tokenizer_path=None, padding_side="right", logger=None):
    """Load a tokenizer from a HuggingFace path.

    Args:
        tokenizer_type (str): tokenizer key; falls back to ``TOKENIZER_PATH``.
        tokenizer_path (str, optional): HF path.
        padding_side (str): tokenizer padding side.
        logger: unused; kept for signature parity.

    Returns:
        tuple: ``(tokenizer, tokenizer_path, processor)`` where processor is None.

    Raises:
        ValueError: if ``tokenizer_path`` is None and the type is unknown.
    """
    from transformers import AutoTokenizer

    processor = None
    if tokenizer_path is None:
        if tokenizer_type not in TOKENIZER_PATH:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        tokenizer_path = TOKENIZER_PATH[tokenizer_type]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side=padding_side)

    return tokenizer, tokenizer_path, processor


@dataclass
class TextEncoderModelOutput:
    """Text-encoder output bundle.

    Args:
        hidden_state (torch.FloatTensor): last-layer hidden states ``[B, L, C]``.
        attention_mask (torch.LongTensor, optional): padding mask ``[B, L]``.
        hidden_states_list (tuple[torch.FloatTensor, ...], optional): per-layer states.
        text_outputs (list, optional): decoded texts.
        image_features (list, optional): optional image features.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: torch.LongTensor | None = None
    hidden_states_list: tuple[torch.FloatTensor, ...] | None = None
    text_outputs: list | None = None
    image_features: list | None = None


class TextEncoder(nn.Module):
    """LLM-backed text encoder with optional prompt-template and attention mask.

    Args:
        text_encoder_type (str): encoder type (only ``"llm"`` supported).
        max_length (int): maximum token sequence length.
        text_encoder_precision (str, optional): dtype key.
        text_encoder_path (str, optional): HF model path.
        tokenizer_type (str, optional): tokenizer key; defaults to encoder type.
        tokenizer_path (str, optional): HF tokenizer path; defaults to encoder path.
        output_key (str, optional): model output attribute; defaults to
            ``"last_hidden_state"``.
        use_attention_mask (bool): pass attention mask to the model.
        prompt_template (dict, optional): image prompt template dict.
        prompt_template_video (dict, optional): video prompt template dict.
        hidden_state_skip_layer (int, optional): layer to read hidden states from.
        apply_final_norm (bool): apply final layer norm to intermediate states.
        reproduce (bool): deterministic sampling flag.
        logger: unused; kept for signature parity.
        device: optional target device.
    """

    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_precision: str | None = None,
        text_encoder_path: str | None = None,
        tokenizer_type: str | None = None,
        tokenizer_path: str | None = None,
        output_key: str | None = None,
        use_attention_mask: bool = True,
        prompt_template: dict | None = None,
        prompt_template_video: dict | None = None,
        hidden_state_skip_layer: int | None = None,
        apply_final_norm: bool = False,
        reproduce: bool = False,
        logger=None,
        device=None,
    ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = tokenizer_type if tokenizer_type is not None else text_encoder_type
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else text_encoder_path
        self.use_attention_mask = use_attention_mask
        if prompt_template_video is not None:
            assert (
                use_attention_mask is True
            ), "Attention mask is True required when training videos."
        self.prompt_template = prompt_template
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.reproduce = reproduce
        self.logger = logger

        self.use_template = self.prompt_template is not None
        if self.use_template:
            assert isinstance(self.prompt_template, dict) and "template" in self.prompt_template, (
                "`prompt_template` must be a dictionary with a key 'template', "
                f"got {self.prompt_template}"
            )
            assert "{}" in str(self.prompt_template["template"]), (
                "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template['template']}"
            )

        self.use_video_template = self.prompt_template_video is not None
        if self.use_video_template:
            if self.prompt_template_video is not None:
                assert (
                    isinstance(self.prompt_template_video, dict)
                    and "template" in self.prompt_template_video
                ), (
                    "`prompt_template_video` must be a with a key 'template', "
                    f"got {self.prompt_template_video}"
                )
            assert "{}" in str(self.prompt_template_video["template"]), (
                "`prompt_template_video['template']` must contain a placeholder `{}` for the "
                f"input text, got {self.prompt_template_video['template']}"
            )

        if text_encoder_type != "llm":
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
        self.output_key = output_key or "last_hidden_state"

        self.model, self.model_path = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_precision=self.precision,
            text_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
        )
        self.tokenizer, self.tokenizer_path, self.processor = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side="right",
            logger=self.logger,
        )

        if self.use_template and self.prompt_template is not None:
            self.text2tokens("a photo of a cat", data_type="image")
        if self.use_video_template and self.prompt_template_video is not None:
            self.text2tokens("a photo of a cat", data_type="video")

    @property
    def dtype(self):
        """Model dtype."""
        return self.model.dtype

    @property
    def device(self):
        """Model device."""
        return self.model.device

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """Apply text to a prompt template.

        Args:
            text (str): input text.
            template (str or list): template string or chat conversation list.
            prevent_empty_text (bool): replace empty text with a space.

        Returns:
            str or list: text-filled template.

        Raises:
            TypeError: if template type is unsupported.
        """
        if isinstance(template, str):
            return template.format(text)
        elif isinstance(template, list):
            template_copy = deepcopy(template)
            for item in template_copy:
                if isinstance(item, dict) and "content" in item:
                    item["content"] = item["content"].format(
                        text if text else (" " if prevent_empty_text else "")
                    )
            return template_copy
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def calculate_crop_start(self, tokenized_input):
        """Calculate the token position where prompt content begins.

        Args:
            tokenized_input: tokenizer output containing ``input_ids``.

        Returns:
            int: crop-start index.
        """
        input_ids = tokenized_input["input_ids"][0].tolist()
        marker = "<|im_start|>user\n"
        marker_tokens = self.tokenizer(marker, add_special_tokens=False)["input_ids"]
        for i in range(len(input_ids) - len(marker_tokens) + 1):
            if input_ids[i : i + len(marker_tokens)] == marker_tokens:
                return i + len(marker_tokens)
        if hasattr(self.tokenizer, "special_tokens_map"):
            for token_name, token_value in self.tokenizer.special_tokens_map.items():
                if "user" in token_name.lower():
                    user_token_id = self.tokenizer.convert_tokens_to_ids(token_value)
                    if user_token_id in input_ids:
                        return input_ids.index(user_token_id) + 1
        return 0

    def text2tokens(self, text, data_type="image", max_length=300):
        """Tokenize input text, applying the configured prompt template.

        Args:
            text (str or list): input text.
            data_type (str): ``"image"`` or ``"video"``.
            max_length (int): max token count.

        Returns:
            dict: tokenizer output.
        """
        tokenize_input_type = "str"
        if self.use_template or self.use_video_template:
            if data_type == "image":
                prompt_template = self.prompt_template["template"]
                crop_start = self.prompt_template.get("crop_start", -1)
            elif data_type == "video":
                prompt_template = self.prompt_template_video["template"]
                crop_start = self.prompt_template_video.get("crop_start", -1)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            if isinstance(text, (list, tuple)):
                text = [self.apply_text_to_template(one_text, prompt_template) for one_text in text]
                if isinstance(text[0], list):
                    tokenize_input_type = "list"
            elif isinstance(text, str):
                text = self.apply_text_to_template(text, prompt_template)
                if isinstance(text, list):
                    tokenize_input_type = "list"
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")

            if crop_start == -1:
                temp_kwargs = dict(
                    truncation=True, max_length=256, padding="max_length", return_tensors="pt"
                )
                if tokenize_input_type == "str":
                    temp_tokenized = self.tokenizer(
                        text,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_attention_mask=True,
                        **temp_kwargs,
                    )
                elif tokenize_input_type == "list":
                    temp_tokenized = self.tokenizer.apply_chat_template(
                        text,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        **temp_kwargs,
                    )
                crop_start = self.calculate_crop_start(temp_tokenized)
                if data_type == "image":
                    self.prompt_template["crop_start"] = crop_start
                else:
                    self.prompt_template_video["crop_start"] = crop_start
        else:
            crop_start = 0

        kwargs = dict(
            truncation=True,
            max_length=max_length + (crop_start if crop_start > 0 else 0),
            padding="max_length",
            return_tensors="pt",
        )
        if tokenize_input_type == "str":
            tokenized_output = self.tokenizer(
                text,
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=True,
                **kwargs,
            )
        elif tokenize_input_type == "list":
            tokenized_output = self.tokenizer.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")
        return tokenized_output

    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=None,
        hidden_state_skip_layer=None,
        return_texts=False,
        data_type="image",
        device=None,
        is_uncond=False,
    ):
        """Encode a pre-tokenized batch.

        Args:
            batch_encoding (dict): tokenizer output with ``input_ids`` and
                ``attention_mask``.
            use_attention_mask (bool, optional): override ``self.use_attention_mask``.
            output_hidden_states (bool): return per-layer hidden states.
            do_sample (bool, optional): sampling flag for decoder-only LLMs.
            hidden_state_skip_layer (int, optional): override layer skip.
            return_texts (bool): unused; kept for signature parity.
            data_type (str): ``"image"`` or ``"video"``.
            device: override target device.
            is_uncond (bool): unused; kept for signature parity.

        Returns:
            TextEncoderModelOutput: encoder output with hidden state and mask.
        """
        device = self.model.device if device is None else device
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(hidden_state_skip_layer, self.hidden_state_skip_layer)
        do_sample = use_default(do_sample, not self.reproduce)

        attention_mask = batch_encoding["attention_mask"].to(device) if use_attention_mask else None
        outputs = self.model(
            input_ids=batch_encoding["input_ids"].to(device),
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states or hidden_state_skip_layer is not None,
        )
        if hidden_state_skip_layer is not None:
            last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
            if hidden_state_skip_layer > 0 and self.apply_final_norm:
                last_hidden_state = self.model.final_layer_norm(last_hidden_state)
        else:
            last_hidden_state = outputs[self.output_key]

        if self.use_template:
            if data_type == "image":
                crop_start = self.prompt_template.get("crop_start", 0)
            elif data_type == "video":
                crop_start = self.prompt_template_video.get("crop_start", 0)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            if crop_start > 0:
                last_hidden_state = last_hidden_state[:, crop_start:]
                attention_mask = attention_mask[:, crop_start:] if use_attention_mask else None

        if output_hidden_states:
            return TextEncoderModelOutput(last_hidden_state, attention_mask, outputs.hidden_states)
        return TextEncoderModelOutput(last_hidden_state, attention_mask)

    def forward(
        self,
        text,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=False,
        hidden_state_skip_layer=None,
        return_texts=False,
    ):
        """Tokenize and encode text.

        Args:
            text (str or list): input text.
            use_attention_mask (bool, optional): override instance flag.
            output_hidden_states (bool): return per-layer states.
            do_sample (bool): sampling flag.
            hidden_state_skip_layer (int, optional): layer skip.
            return_texts (bool): unused.

        Returns:
            TextEncoderModelOutput: encoder output.
        """
        batch_encoding = self.text2tokens(text, max_length=self.max_length)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            do_sample=do_sample,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
        )
