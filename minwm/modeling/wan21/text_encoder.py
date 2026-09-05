"""Wan umt5-xxl text encoder wrapper.

Wraps the vendored umt5-xxl encoder (:mod:`minwm.modeling.wan21.t5`) plus its
HuggingFace tokenizer into a frozen module that turns prompt strings into the
per-sample context list the Wan DiT consumes (one ``[L_i, dim]`` tensor per
prompt, padding rows zeroed). Checkpoint + tokenizer paths are explicit
constructor args (no hard-coded repo layout), so a config points them at the
downloaded Wan2.1-T2V base.

Built lazily by the config system and handed to :class:`Wan21Adapter`, so the
recipe stays text-encoder-agnostic: with an encoder the adapter encodes real
prompts, without one it falls back to zeros for mock smoke tests.
"""

import torch
from torch import Tensor

from minwm.utils.dtype import resolve_dtype

from .t5 import umt5_xxl
from .tokenizers import HuggingfaceTokenizer


class Wan21TextEncoder(torch.nn.Module):
    """Frozen umt5-xxl encoder producing per-prompt context embeddings.

    Args:
        checkpoint_path (str): path to ``models_t5_umt5-xxl-enc-bf16.pth``.
        tokenizer_path (str): path to the ``google/umt5-xxl`` tokenizer directory.
        text_len (int): max token length (padding target). Defaults to 512.
        dtype (str | torch.dtype): encoder weight dtype. Defaults to ``"bfloat16"``,
            matching the shipped ``-bf16`` checkpoint and the legacy Wan encoder;
            a config passes the string form.
    """

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        text_len: int = 512,
        dtype: str | torch.dtype = "bfloat16",
    ) -> None:
        super().__init__()
        self.text_len = text_len
        torch_dtype = resolve_dtype(dtype)

        self.text_encoder = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=torch_dtype,
                device=torch.device("cpu"),
            )
            .eval()
            .requires_grad_(False)
        )
        self.text_encoder.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        )
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean="whitespace"
        )

    @property
    def device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    @torch.no_grad()
    def forward(self, prompts: list[str]) -> list[Tensor]:
        """Encode prompts into per-sample variable-length context tensors.

        Args:
            prompts (list[str]): one prompt per sample.

        Returns:
            list[Tensor]: per-prompt context, each ``[L_i, dim]`` with padding rows
            dropped (the unpadded length per prompt).
        """
        ids, mask = self.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]
