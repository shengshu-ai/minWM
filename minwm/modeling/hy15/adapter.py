"""HYAdapter: call-convention bridge for the HunyuanVideo 1.5 AR transformer.

The model's ``forward_bi`` path consumes a batched ``hidden_states [B, C, F, H, W]``,
a video/text timestep pair, dual text streams plus an attention mask, and returns an
``(img, None)`` tuple.
"""

import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..adapter import ModelAdapter


class HYAdapter(ModelAdapter):
    """Adapter for the HunyuanVideo 1.5 AR transformer (``forward_bi``).

    Runs inference under half-precision autocast (see
    :attr:`~minwm.modeling.adapter.ModelAdapter.wants_inference_autocast`): the
    HY text / time embedders build fp32 tensors internally and the training
    forward runs under half autocast, so the fp32-promoted text-refiner input
    must hit the bf16 linear under autocast at inference too.

    Args:
        task_type (str): ``"i2v"`` (image-conditioned) or ``"t2v"`` (zero image
            condition). Defaults to ``"i2v"``.
        neg_prompt_path (str | None): path to the negative-prompt embedding .pt for
            the CFG uncond branch. Defaults to ``None`` (no negative prompt).
        neg_byt5_path (str | None): path to the negative ByT5 embedding .pt for the
            CFG uncond branch. Defaults to ``None``.
        **kwargs: forwarded to :class:`~minwm.modeling.adapter.ModelAdapter`.
    """

    wants_inference_autocast = True

    def __init__(
        self,
        task_type: str = "i2v",
        neg_prompt_path: str | None = None,
        neg_byt5_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.task_type = task_type
        self.neg_prompt_path = neg_prompt_path or os.environ.get("NEG_PROMPT_PT")
        self.neg_byt5_path = neg_byt5_path or os.environ.get("NEG_BYT5_PT")
        self._neg: dict[str, Tensor] | None = None

    def encode_text(
        self, prompts: list | None, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Zero-text stand-in (tests only); real runs use the batch's ``prompt_embed``."""
        text_states = torch.zeros(
            batch_size, self.text_len, self.text_dim, device=device, dtype=self.dtype
        )
        attn_mask = torch.ones(batch_size, self.text_len, device=device, dtype=torch.bool)
        return (text_states, attn_mask)

    def conditioning(self, batch: dict, batch_size: int, device: torch.device) -> dict[str, Any]:
        """The HY multi-modal conditioning stream (text + glyph + vision + image).

        Returns the real pre-encoded conditioning from the batch — the recipe does
        not assemble it. When ``prompt_embed`` is absent (mock smoke tests) it
        falls back to the zero-text ``context`` so :meth:`denoise` still runs.

        Args:
            batch (dict): the batch (carries ``prompt_embed`` / ``prompt_mask`` /
                ``byt5_*`` / ``vision_states`` / ``image_cond`` / ``action``).
            batch_size (int): batch size ``B``.
            device (torch.device): target device.

        Returns:
            dict[str, Any]: the conditional :meth:`denoise` kwargs.
        """
        if batch.get("prompt_embed") is None:
            return {"context": self.encode_text(batch.get("prompts"), batch_size, device)}
        return {
            "context": None,
            "prompt_embed": batch.get("prompt_embed"),
            "prompt_mask": batch.get("prompt_mask"),
            "byt5_text_states": batch.get("byt5_text_states"),
            "byt5_text_mask": batch.get("byt5_text_mask"),
            "vision_states": batch.get("vision_states"),
            "image_cond": batch.get("image_cond"),
            "action": batch.get("action"),
        }

    def null_conditioning(
        self, batch: dict, batch_size: int, device: torch.device
    ) -> dict[str, Any]:
        """CFG uncond: the **negative-prompt** text stream, same vision / image.

        Mirrors the refactor's ``_build_uncond_kwargs`` — only the text / glyph
        streams swap to the negative prompt; ``vision_states`` / ``image_cond`` /
        ``action`` are unchanged (the camera/image condition is not part of CFG).
        Falls back to zero text when no negative prompt is configured or when the
        batch has no real text (mock).

        Args:
            batch (dict): the batch.
            batch_size (int): batch size ``B``.
            device (torch.device): target device.

        Returns:
            dict[str, Any]: the unconditional :meth:`denoise` kwargs.
        """
        if batch.get("prompt_embed") is None:
            return {"context": self.encode_text(None, batch_size, device)}
        neg = self._load_neg(device)
        if neg is None:
            text = {
                "prompt_embed": torch.zeros_like(batch["prompt_embed"]),
                "prompt_mask": batch["prompt_mask"],
                "byt5_text_states": torch.zeros_like(batch["byt5_text_states"]),
                "byt5_text_mask": batch["byt5_text_mask"],
            }
        else:
            text = {
                "prompt_embed": neg["prompt_embed"].unsqueeze(0).expand(batch_size, -1, -1),
                "prompt_mask": neg["prompt_mask"].unsqueeze(0).expand(batch_size, -1),
                "byt5_text_states": neg["byt5_text_states"].unsqueeze(0).expand(batch_size, -1, -1),
                "byt5_text_mask": neg["byt5_text_mask"].unsqueeze(0).expand(batch_size, -1),
            }
        return {
            "context": None,
            **text,
            "vision_states": batch.get("vision_states"),
            "image_cond": batch.get("image_cond"),
            "action": batch.get("action"),
        }

    def _load_neg(self, device: torch.device) -> dict[str, Tensor] | None:
        """Load + cache the negative-prompt embeddings for the CFG uncond branch."""
        if self._neg is not None or not (self.neg_prompt_path and self.neg_byt5_path):
            return self._neg
        neg = torch.load(self.neg_prompt_path, map_location=device, weights_only=True)
        byt5 = torch.load(self.neg_byt5_path, map_location=device, weights_only=True)
        self._neg = {
            "prompt_embed": neg["negative_prompt_embeds"][0].to(dtype=self.dtype),
            "prompt_mask": neg["negative_prompt_mask"][0].bool(),
            "byt5_text_states": byt5["byt5_text_states"][0].to(dtype=self.dtype),
            "byt5_text_mask": byt5["byt5_text_mask"][0].bool(),
        }
        return self._neg

    def _concat_condition(self, hidden: Tensor, image_cond: Tensor | None) -> Tensor:
        # Append condition latent + task mask. i2v puts the image latent on frame 0;
        # t2v leaves both zero. -> [B, 2*C + 1, F, H, W].
        b, c, f, h, w = hidden.shape
        cond_latent = torch.zeros(b, c, f, h, w, device=hidden.device, dtype=hidden.dtype)
        mask = torch.zeros(b, 1, f, h, w, device=hidden.device, dtype=hidden.dtype)
        if self.task_type == "i2v" and image_cond is not None:
            cond_latent[:, :, :1] = image_cond.to(device=hidden.device, dtype=hidden.dtype)
            mask[:, :, :1] = 1.0
        return torch.cat([hidden, cond_latent, mask], dim=1)

    def denoise(
        self,
        model: nn.Module,
        *,
        noisy: Tensor,
        timestep: Tensor,
        context: Any = None,
        clean: Tensor | None = None,
        aug_t: Tensor | None = None,
        viewmats: Tensor | None = None,
        Ks: Tensor | None = None,
        prompt_embed: Tensor | None = None,
        prompt_mask: Tensor | None = None,
        byt5_text_states: Tensor | None = None,
        byt5_text_mask: Tensor | None = None,
        vision_states: Tensor | None = None,
        image_cond: Tensor | None = None,
        action: Tensor | None = None,
    ) -> Tensor:
        """Run ``forward_bi`` on ``[B,F,C,H,W]`` latents, return a flow prediction.

        Threads the TI2V conditioning: ``prompt_embed`` / ``prompt_mask`` (real text,
        else the zero-text ``context`` fallback for tests), ``byt5_*`` (glyph stream),
        and ``vision_states`` / ``image_cond`` (the conditioning image, ``i2v``).

        Args:
            noisy (Tensor): noisy latents ``[B, F, C, H, W]``.
            timestep (Tensor): per-frame timesteps ``[B, F]``.
            context (Any): zero-text ``(text_states, mask)`` fallback for tests.

        Returns:
            Tensor: flow prediction ``[B, F, C, H, W]``.
        """
        B, F = noisy.shape[:2]
        # Compute dtype must follow the MODEL, not ``noisy``. The DMD score path
        # builds ``noisy = (1-σ)·x + σ·ε`` with an fp32 σ (train_sigmas is fp32),
        # which promotes ``noisy`` to fp32; deriving dtype from it would feed the
        # bf16 transformer fp32 hidden_states / timestep / text and diverge from the
        # refactor (which explicitly casts every model input to bf16). Use the
        # model's own parameter dtype as the source of truth.
        device = noisy.device
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = noisy.dtype

        def to_dev(t: Tensor | None, *, cast: bool = False) -> Tensor | None:
            # Conditioning fields arrive on CPU (only latent/camera go through
            # LatentToDevice); move them here so non-FSDP debug runs work too.
            if t is None:
                return None
            return t.to(device=device, dtype=dtype) if cast else t.to(device=device)

        concat = getattr(getattr(model, "config", None), "concat_condition", True)
        hidden = noisy.permute(0, 2, 1, 3, 4)
        clean_x = clean.permute(0, 2, 1, 3, 4) if clean is not None else None
        if concat:
            hidden = self._concat_condition(hidden, image_cond)
            clean_x = self._concat_condition(clean_x, image_cond) if clean_x is not None else None
        # Cast the assembled latents to the model dtype (refactor does
        # ``latents_concat.to(bfloat16)``); ``noisy`` may be fp32 from an fp32 σ.
        hidden = hidden.to(dtype)
        clean_x = clean_x.to(dtype) if clean_x is not None else None

        if prompt_embed is not None:
            text_states = to_dev(prompt_embed, cast=True)
            attn_mask = to_dev(prompt_mask)
        else:
            text_states, attn_mask = context

        # Refactor-faithful: the transformer receives the timestep in the model
        # dtype (bf16), NOT fp32. This matters when t is not bf16-exact (e.g. the
        # DMD renoise draws a uniform t like 808): an fp32 t would round to a
        # different time-embedding than the refactor's `timestep.to(bfloat16)`.
        t_vid = timestep.reshape(-1).to(dtype)
        t_txt = torch.zeros(B, device=device, dtype=dtype)
        if aug_t is None:
            aug_t = torch.zeros(B * F, device=device, dtype=dtype)
        else:
            aug_t = aug_t.reshape(-1)

        extra_kwargs = None
        if byt5_text_states is not None:
            extra_kwargs = {
                "byt5_text_states": to_dev(byt5_text_states, cast=True),
                "byt5_text_mask": to_dev(byt5_text_mask),
            }

        img, _ = model(
            bi_inference=True,
            hidden_states=hidden,
            timestep=t_vid,
            timestep_txt=t_txt,
            text_states=text_states,
            text_states_2=None,
            encoder_attention_mask=attn_mask,
            clean_x=clean_x,
            aug_timesteps=aug_t,
            viewmats=to_dev(viewmats, cast=True),
            Ks=to_dev(Ks, cast=True),
            vision_states=to_dev(vision_states, cast=True),
            action=to_dev(action),
            extra_kwargs=extra_kwargs,
            mask_type=self.task_type,
        )
        return img.permute(0, 2, 1, 3, 4)  # [B,C,F,H,W] -> [B,F,C,H,W]

    def decode_latents(self, vae, latents: Tensor) -> Tensor:
        """Decode ``[B,F,C,H,W]`` latents to ``[B,F,3,H,W]`` via the HY 3D VAE.

        The ``AutoencoderKLConv3D`` decodes a batched ``[B,C,T,H,W]`` tensor
        (unscaled by ``config.scaling_factor``) and returns a ``DecoderOutput``.

        Args:
            vae: the HY ``AutoencoderKLConv3D``.
            latents (Tensor): latent video ``[B, F, C, H, W]``.

        Returns:
            Tensor: decoded pixels ``[B, F, 3, H, W]``.
        """
        vae_dtype = next(vae.parameters()).dtype
        z = (latents.permute(0, 2, 1, 3, 4) / vae.config.scaling_factor).to(vae_dtype)
        # Decode under bf16 autocast (the reference does the same): the 3D conv
        # activations over the full temporal window are large, and this VAE only
        # tiles spatially, so fp32 decode of a 20-frame latent OOMs a busy card.
        autocast: AbstractContextManager = nullcontext()
        if vae_dtype in (torch.float16, torch.bfloat16) and latents.device.type == "cuda":
            autocast = torch.autocast("cuda", dtype=vae_dtype)
        with autocast:
            px = vae.decode(z).sample
        return px.permute(0, 2, 1, 3, 4).float()

    # ------------------------------------------------------------------
    # Self-forcing rollout hooks (DMD Stage 3)
    #
    # Ports the HY cm-rollout's per-block forward_vision + context-rerun into the
    # model-agnostic SelfForcingPipeline loop. The pipeline owns the outer
    # block/step loop, x0 conversion, exit-step truncation and re-noise; these
    # hooks own the HY-specific text-KV precache, i2v condition channels, cached
    # ar_vision forward, and the growing vision-KV concatenation.
    # ------------------------------------------------------------------

    @staticmethod
    def _rollout_prepare_cond(
        image_cond: Tensor | None, latents: Tensor, mask_type: str, f_start: int
    ) -> Tensor:
        """Build i2v condition channels ``[B, C+1, n_f, H, W]`` for one block.

        The i2v condition is defined over the WHOLE sequence with the image latent
        and task-mask on the **global** frame 0 only (every later frame is zero),
        then sliced per block — exactly the refactor's ``_prepare_cond_latents``
        (``image_cond.repeat(...)[:, :, 1:] = 0``; ``mask[0] = 1``) followed by
        ``cond_latents[:, :, start:end]``. So the image / mask land on this block
        only when it contains global frame 0 (``f_start == 0``); all later blocks
        get zero condition — otherwise every chunk's first frame would be re-pulled
        toward the input image, producing chunk-boundary jumps.

        Args:
            image_cond (Tensor): conditioning image latent ``[B, C, 1, H, W]`` or None.
            latents (Tensor): block latents ``[B, C, n_f, H, W]`` (shape reference).
            mask_type (str): ``"i2v"`` or ``"t2v"``.
            f_start (int): this block's global start frame index.

        Returns:
            Tensor: condition channels ``[B, C+1, n_f, H, W]`` in ``latents.dtype``.
        """
        b, c, t, h, w = latents.shape
        cond = torch.zeros_like(latents)
        mask = torch.zeros(b, 1, t, h, w, device=latents.device, dtype=latents.dtype)
        if mask_type == "i2v" and image_cond is not None and f_start == 0:
            ic = image_cond.to(device=latents.device, dtype=latents.dtype)
            cond[:, :, :1] = ic[:, :, :1]
            mask[:, :, :1] = 1.0
        return torch.cat([cond, mask], dim=1).to(latents.dtype)

    def _rollout_empty_kv(self, model: nn.Module) -> list[dict]:
        """Empty per-double-block HY KV cache (all four streams None)."""
        return [
            {"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None}
            for _ in range(len(model.double_blocks))
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
        """Populate the text KV cache once + stash the i2v condition channels.

        ``use_local_window`` is ignored: HY's vision KV grows by concatenation, it
        has no pre-allocated fixed-window buffer to size down.
        """
        mask_type = cond.get("mask_type", self.task_type)
        extra_kwargs = {
            "byt5_text_states": cond.get("byt5_text_states"),
            "byt5_text_mask": cond.get("byt5_text_mask"),
        }
        timestep_txt = torch.tensor([0], device=device, dtype=dtype)
        with torch.no_grad():
            kv_cache = model(
                bi_inference=False,
                ar_txt_inference=True,
                ar_vision_inference=False,
                timestep_txt=timestep_txt,
                text_states=cond["prompt_embed"],
                encoder_attention_mask=cond["prompt_mask"],
                vision_states=cond["vision_states"],
                mask_type=mask_type,
                extra_kwargs=extra_kwargs,
                kv_cache=self._rollout_empty_kv(model),
                cache_txt=True,
            )
        return {"kv_cache": kv_cache, "mask_type": mask_type, "image_cond": cond.get("image_cond")}

    def _rollout_cond_block(
        self, cache: dict[str, Any], latents_block: Tensor, f_start: int
    ) -> Tensor:
        """The i2v condition channels for one temporal block ``[B, C+1, n_f, H, W]``."""
        return self._rollout_prepare_cond(
            cache.get("image_cond"), latents_block, cache["mask_type"], f_start
        )

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
        """Cached ``ar_vision`` forward for one block, returning flow ``[B,n_f,C,H,W]``."""
        dtype = noisy_block.dtype
        latents = noisy_block.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,n_f,H,W]
        cond_block = self._rollout_cond_block(cache, latents, meta["f_start"])
        hidden = torch.cat([latents, cond_block], dim=1)
        # HY vision forward wants a per-frame timestep of length n_f.
        t_block = timestep[0].reshape(-1).to(dtype) if timestep.dim() > 1 else timestep.to(dtype)
        img, _ = model(
            bi_inference=False,
            ar_txt_inference=False,
            ar_vision_inference=True,
            hidden_states=hidden,
            timestep=t_block,
            timestep_r=None,
            mask_type=cache["mask_type"],
            return_dict=False,
            kv_cache=cache["kv_cache"],
            cache_vision=False,
            rope_temporal_size=meta["f_end"],
            start_rope_start_idx=meta["f_start"],
            viewmats=viewmats,
            Ks=Ks,
            action=action,
        )
        return img.permute(0, 2, 1, 3, 4)  # [B,C,n_f,H,W] -> [B,n_f,C,H,W]

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
        """Context-rerun: append this block's vision KV onto the growing cache."""
        n_f = clean_block.shape[1]
        device, dtype = clean_block.device, clean_block.dtype
        latents = clean_block.detach().permute(0, 2, 1, 3, 4).contiguous()  # [B,C,n_f,H,W]
        cond_block = self._rollout_cond_block(cache, latents, meta["f_start"])
        hidden = torch.cat([latents, cond_block], dim=1)
        context_noise = meta.get("context_noise", 0)
        context_timestep = torch.full((n_f,), context_noise, device=device, dtype=dtype)
        kv_cache = cache["kv_cache"]
        with torch.no_grad():
            new_kv = model(
                bi_inference=False,
                ar_txt_inference=False,
                ar_vision_inference=True,
                hidden_states=hidden,
                timestep=context_timestep,
                timestep_r=None,
                mask_type=cache["mask_type"],
                return_dict=False,
                kv_cache=kv_cache,
                cache_vision=True,
                rope_temporal_size=meta["f_end"],
                start_rope_start_idx=meta["f_start"],
                viewmats=viewmats,
                Ks=Ks,
                action=action,
            )
        for i in range(len(kv_cache)):
            if kv_cache[i]["k_vision"] is None:
                kv_cache[i]["k_vision"] = new_kv[i]["k_vision"]
                kv_cache[i]["v_vision"] = new_kv[i]["v_vision"]
            else:
                kv_cache[i]["k_vision"] = torch.cat(
                    [kv_cache[i]["k_vision"], new_kv[i]["k_vision"]], dim=2
                )
                kv_cache[i]["v_vision"] = torch.cat(
                    [kv_cache[i]["v_vision"], new_kv[i]["v_vision"]], dim=2
                )
