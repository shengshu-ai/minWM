"""Wan21Adapter: call-convention bridge for the Wan DiT family.

The bidirectional ``Wan21Model`` and causal ``CausalWan21Model`` consume a per-sample
list of ``[C, F, H, W]`` tensors, a context list of ``[L, C]`` embeddings, and an
explicit ``seq_len`` token count; they return ``[B, C, F, H, W]``. This adapter
performs the ``[B,F,C,H,W]`` <-> list/channel-first conversion and computes
``seq_len`` from the model's patch size.
"""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..adapter import ModelAdapter

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


class Wan21Adapter(ModelAdapter):
    """Adapter for the Wan DiT bidirectional and causal model variants."""

    def __init__(
        self,
        causal: bool = False,
        text_len: int = 512,
        text_dim: int = 4096,
        dtype: str = "bfloat16",
        text_encoder: nn.Module | None = None,
        negative_prompt: str | None = None,
    ) -> None:
        super().__init__(text_len=text_len, text_dim=text_dim, dtype=dtype)
        self.causal = causal
        self.text_encoder = text_encoder
        self.negative_prompt = negative_prompt
        # The encoder is config-built on CPU; move it to the training/inference
        # device on first use and cache that device.
        self._text_encoder_device: torch.device | None = None

    def attach_text_encoder(self, text_encoder: nn.Module) -> None:
        """Store the recipe-built text encoder (only if not already config-built)."""
        if self.text_encoder is None:
            self.text_encoder = text_encoder

    def set_negative_prompt(self, negative_prompt: str) -> None:
        """Store the CFG negative-prompt text for :meth:`null_conditioning`."""
        self.negative_prompt = negative_prompt

    def null_conditioning(
        self, batch: dict, batch_size: int, device: torch.device
    ) -> dict[str, Any]:
        """CFG uncond: encode the configured negative prompt, else zero text.

        Inference sets :attr:`negative_prompt` to the real negative-prompt string
        (owned by the adapter, not the pipeline), so the uncond branch encodes it
        just like the cond branch encodes the batch prompts. When it is ``None``
        (the training default) this falls back to the base zero-text context, so
        the DMD / CD uncond stream is unchanged.

        Args:
            batch (dict): the (preprocessed) batch.
            batch_size (int): batch size ``B``.
            device (torch.device): target device.

        Returns:
            dict[str, Any]: kwargs splatted into :meth:`denoise` for the
            unconditional forward.
        """
        if self.negative_prompt is None:
            return super().null_conditioning(batch, batch_size, device)
        negatives = [self.negative_prompt] * batch_size
        return {"context": self.encode_text(negatives, batch_size, device)}

    def encode_text(
        self, prompts: list | None, batch_size: int, device: torch.device
    ) -> list[Tensor]:
        if self.text_encoder is not None and prompts is not None:
            prompts = list(prompts)
            if len(prompts) != batch_size:
                raise ValueError(
                    f"encode_text got {len(prompts)} prompts for batch_size {batch_size}"
                )
            if self._text_encoder_device != device:
                self.text_encoder.to(device)
                self._text_encoder_device = device
            return [c.to(device=device, dtype=self.dtype) for c in self.text_encoder(prompts)]
        return [
            torch.zeros(self.text_len, self.text_dim, device=device, dtype=self.dtype)
            for _ in range(batch_size)
        ]

    def denoise(
        self,
        model: nn.Module,
        *,
        noisy: Tensor,
        timestep: Tensor,
        context: Any,
        clean: Tensor | None = None,
        aug_t: Tensor | None = None,
        viewmats: Tensor | None = None,
        Ks: Tensor | None = None,
    ) -> Tensor:
        B, F, C, H, W = noisy.shape
        seq_len = F * (H // model.patch_size[1]) * (W // model.patch_size[2])
        t = timestep.float()
        if not self.causal:
            t = t[:, 0]

        # FSDP may only cast inputs at sharded module boundaries. Match the
        # stored param dtype for replicated frozen score nets as well.
        p_dtype = next(model.parameters()).dtype
        noisy = noisy.to(p_dtype)
        if clean is not None:
            clean = clean.to(p_dtype)
        context = [c.to(p_dtype) for c in context]

        kwargs: dict[str, Any] = {
            "x": [noisy[i].permute(1, 0, 2, 3) for i in range(B)],
            "t": t,
            "context": context,
            "seq_len": seq_len,
            "viewmats": viewmats,
            "Ks": Ks,
        }
        if clean is not None:
            kwargs["clean_x"] = [clean[i].permute(1, 0, 2, 3) for i in range(B)]
        if aug_t is not None:
            kwargs["aug_t"] = aug_t
        return model(**kwargs).permute(0, 2, 1, 3, 4)

    # ------------------------------------------------------------------
    # Self-forcing rollout hooks (DMD Stage 3)
    # ------------------------------------------------------------------

    def _sp_num_heads(self, num_heads: int) -> int:
        try:
            from minwm.distributed.parallel_dims import get_parallel_state

            ps = get_parallel_state()
        except Exception:
            return num_heads

        if not ps.sp_enabled:
            return num_heads
        if num_heads % ps.sp != 0:
            raise ValueError(f"Wan SP requires num_heads={num_heads} to be divisible by sp={ps.sp}")
        return num_heads // ps.sp

    def _self_kv_cache(
        self,
        model: nn.Module,
        *,
        batch_size: int,
        kv_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[dict[str, Tensor]]:
        return [
            {
                "k": torch.zeros(
                    batch_size, kv_size, num_heads, head_dim, dtype=dtype, device=device
                ),
                "v": torch.zeros(
                    batch_size, kv_size, num_heads, head_dim, dtype=dtype, device=device
                ),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
            for _ in range(len(model.blocks))
        ]

    def rollout_init_cache(
        self,
        model: nn.Module,
        *,
        batch_size: int,
        num_frames: int,
        frame_seqlen: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        cond: dict[str, Any],
        use_prope_cache: bool = True,
        use_local_window: bool = False,
    ) -> dict[str, Any]:
        """Per-block self-attn, cross-attn, and optional PRoPE KV cache.

        Args:
            use_local_window (bool): if True and the model has a finite
                ``local_attn_size``, size the self-attn / PRoPE KV buffers to the
                local window (``local_attn_size * frame_seqlen``) instead of the
                full clip. Deployment inference sets this; the DMD self-forcing
                training loop leaves it False (full-window), so training cache
                shape is unchanged.
        """

        num_heads = self._sp_num_heads(model.num_heads)
        head_dim = model.dim // model.num_heads
        local = getattr(model, "local_attn_size", -1)
        if use_local_window and local != -1:
            kv_size = local * frame_seqlen
        else:
            kv_size = num_frames * frame_seqlen
        kv_cache = self._self_kv_cache(
            model,
            batch_size=batch_size,
            kv_size=kv_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        prope_kv_cache = (
            self._self_kv_cache(
                model,
                batch_size=batch_size,
                kv_size=kv_size,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )
            if getattr(model, "use_prope", False) and use_prope_cache
            else None
        )

        ca_heads = model.num_heads
        ca_head_dim = model.dim // ca_heads
        crossattn_cache = [
            {
                "k": torch.zeros(
                    batch_size,
                    model.text_len,
                    ca_heads,
                    ca_head_dim,
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    batch_size,
                    model.text_len,
                    ca_heads,
                    ca_head_dim,
                    dtype=dtype,
                    device=device,
                ),
                "is_init": False,
            }
            for _ in range(len(model.blocks))
        ]
        return {
            "kv_cache": kv_cache,
            "crossattn_cache": crossattn_cache,
            "prope_kv_cache": prope_kv_cache,
        }

    def rollout_forward(
        self,
        model: nn.Module,
        *,
        noisy_block: Tensor,
        timestep: Tensor,
        cache: dict[str, Any],
        cond: dict[str, Any],
        meta: dict[str, Any],
        viewmats: Tensor | None = None,
        Ks: Tensor | None = None,
        action: Tensor | None = None,
    ) -> Tensor:
        """KV-cached AR forward for one block via ``CausalWan21Model`` inference."""

        B = noisy_block.shape[0]
        context = cond["context"]
        p_dtype = next(model.parameters()).dtype
        noisy_block = noisy_block.to(p_dtype)
        context = [c.to(p_dtype) for c in context]
        out = model(
            x=[noisy_block[i].permute(1, 0, 2, 3) for i in range(B)],
            t=timestep.float(),
            context=context,
            seq_len=meta["seq_len"],
            kv_cache=cache["kv_cache"],
            crossattn_cache=cache["crossattn_cache"],
            current_start=meta["current_start"],
            cache_start=meta.get("cache_start", meta["current_start"]),
            viewmats=viewmats,
            Ks=Ks,
            prope_kv_cache=cache.get("prope_kv_cache"),
        )
        return out.permute(0, 2, 1, 3, 4)

    def rollout_refresh_cache(
        self,
        model: nn.Module,
        *,
        clean_block: Tensor,
        cache: dict[str, Any],
        cond: dict[str, Any],
        meta: dict[str, Any],
        viewmats: Tensor | None = None,
        Ks: Tensor | None = None,
        action: Tensor | None = None,
    ) -> None:
        """No-grad refresh pass writing this block's clean KV in place."""

        B, n_f = clean_block.shape[:2]
        context = cond["context"]
        p_dtype = next(model.parameters()).dtype
        clean_block = clean_block.to(p_dtype)
        context = [c.to(p_dtype) for c in context]
        t_zero = torch.zeros(B, n_f, device=clean_block.device, dtype=clean_block.dtype)
        with torch.no_grad():
            model(
                x=[clean_block[i].detach().permute(1, 0, 2, 3) for i in range(B)],
                t=t_zero.float(),
                context=context,
                seq_len=meta["seq_len"],
                kv_cache=cache["kv_cache"],
                crossattn_cache=cache["crossattn_cache"],
                current_start=meta["current_start"],
                cache_start=meta.get("cache_start", meta["current_start"]),
                viewmats=viewmats,
                Ks=Ks,
                prope_kv_cache=cache.get("prope_kv_cache"),
            )
