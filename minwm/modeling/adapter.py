"""ModelAdapter: bridge a recipe's tensor space to a model family's call convention.

A :class:`~minwm.engine.recipe_base.Recipe` works entirely in ``[B, F, C, H, W]`` latent
space and thinks in terms of "run the denoiser, get a flow prediction back". But each
model family speaks a different dialect: the Wan DiT wants a per-sample list of
``[C, F, H, W]`` tensors, a ``context`` list, an explicit ``seq_len``, and returns
``[B, C, F, H, W]``; the HY15 transformer wants a batched ``[B, C, F, H, W]``
``hidden_states``, dual text streams, an attention mask, and returns an ``(img, None)``
tuple.

A :class:`ModelAdapter` owns exactly that translation, so the recipe never writes the
list-comprehension / permute / ``seq_len`` boilerplate inline. Swapping models is then a
one-line ``adapter=...`` change, and a recipe's per-step math stays model-agnostic. The
adapter lives in the modeling layer because it is a model wrapper: it encodes how a
specific model family is called, not how a training stage works.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class ModelAdapter(ABC):
    """Translate between recipe ``[B,F,C,H,W]`` space and a model's call convention.

    The adapter owns everything model-facing: the text dims/working dtype the
    model is fed (:attr:`text_len`, :attr:`text_dim`, :attr:`dtype`) and the two
    seams a recipe uses to touch the model — :meth:`encode_text` (prompts -> the
    opaque ``context`` the model consumes) and :meth:`denoise` (noisy latents ->
    flow prediction, hiding the model's input/output layout). The recipe forwards
    ``context`` between them without inspecting it, and reads :attr:`dtype` off
    the adapter when it needs the model's working precision.

    Args:
        text_len (int): padded text sequence length the model expects.
        text_dim (int): text embedding dimension the model expects.
        dtype (str): working dtype, ``"bfloat16"`` or ``"float32"``.

    Attributes:
        wants_inference_autocast (bool): whether the full-diffusion pipelines
            should run :meth:`denoise` under half-precision ``torch.autocast``.
            Models whose text / time embedders build fp32 tensors internally and
            train under half autocast (HY) set this ``True`` so inference matches;
            models that cast explicitly and keep an fp32 path (Wan) leave it
            ``False``. Class-level so the pipeline never special-cases a family.
    """

    wants_inference_autocast: bool = False

    def __init__(self, text_len: int = 512, text_dim: int = 4096, dtype: str = "bfloat16") -> None:
        self.text_len = text_len
        self.text_dim = text_dim
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    @abstractmethod
    def encode_text(self, prompts: list | None, batch_size: int, device: torch.device) -> Any:
        """Encode prompts into the model-specific ``context`` object.

        Threads the adapter's own :attr:`text_len` / :attr:`text_dim` /
        :attr:`dtype` — the recipe does not pass them in.

        Args:
            prompts (list, optional): text prompts (currently unused; a real text
                encoder is not yet wired in, so zeros are returned).
            batch_size (int): batch size ``B``.
            device (torch.device): target device.

        Returns:
            Any: the opaque context consumed by :meth:`denoise` (a ``list[Tensor]`` for
            Wan, a ``(text_states, mask)`` tuple for HY).
        """
        raise NotImplementedError

    def conditioning(self, batch: dict, batch_size: int, device: torch.device) -> dict[str, Any]:
        """The conditional model-input kwargs to splat into :meth:`denoise`.

        This is the CFG *conditional* branch — the model-facing conditioning the
        recipe should not have to assemble itself. The default reproduces the
        single-stream text convention (``{"context": encode_text(prompts)}``);
        multi-modal models (HY) override it to return their full conditioning
        stream (text + glyph + vision + image). Camera / action signals that are
        *shared* across the CFG split are also returned here.

        Args:
            batch (dict): the (preprocessed) batch.
            batch_size (int): batch size ``B``.
            device (torch.device): target device.

        Returns:
            dict[str, Any]: kwargs splatted into :meth:`denoise` for the
            conditional forward.
        """
        return {"context": self.encode_text(batch.get("prompts"), batch_size, device)}

    def null_conditioning(
        self, batch: dict, batch_size: int, device: torch.device
    ) -> dict[str, Any]:
        """The CFG *unconditional* model-input kwargs to splat into :meth:`denoise`.

        Mirrors :meth:`conditioning` for the unconditional branch. The default is
        the zero / empty text context (``encode_text(None)``); models whose
        "unconditional" is a real negative prompt (HY) override this. Shared
        camera / action signals are returned identically to :meth:`conditioning`.

        Args:
            batch (dict): the (preprocessed) batch.
            batch_size (int): batch size ``B``.
            device (torch.device): target device.

        Returns:
            dict[str, Any]: kwargs splatted into :meth:`denoise` for the
            unconditional forward.
        """
        return {"context": self.encode_text(None, batch_size, device)}

    # ------------------------------------------------------------------
    # Recipe-driven setup (uniform across families; no ``hasattr`` probing)
    # ------------------------------------------------------------------

    def attach_text_encoder(self, text_encoder: nn.Module) -> None:
        """Attach the recipe-built text encoder.

        Default is a no-op: families whose conditioning is pre-encoded (HY reads
        ``prompt_embed`` straight off the batch) do not need it. Families that
        encode prompts at inference (Wan) override this to store the encoder.

        Args:
            text_encoder (nn.Module): the recipe-built text encoder.
        """

    def set_negative_prompt(self, negative_prompt: str) -> None:
        """Set the CFG negative-prompt text.

        Default is a no-op: families whose unconditional stream is not a text
        prompt (HY loads a pre-encoded negative ``.pt``) ignore it. Text-CFG
        families (Wan) override this to store the string.

        Args:
            negative_prompt (str): the negative-prompt text for the CFG uncond branch.
        """

    def decode_latents(self, vae: Any, latents: Tensor) -> Tensor:
        """Decode ``[B,F,C,H,W]`` latents to ``[B,F,3,H,W]`` pixels.

        Default is the Wan convention: the VAE decodes a per-sample list of
        ``[C,F,H,W]`` tensors. Families whose VAE takes a batched
        ``[B,C,T,H,W]`` tensor (HY) override this.

        Args:
            vae (Any): the video VAE.
            latents (Tensor): latent video ``[B, F, C, H, W]``.

        Returns:
            Tensor: decoded pixels ``[B, F, 3, H, W]``.
        """
        zs = [latents[i].permute(1, 0, 2, 3) for i in range(latents.shape[0])]
        decoded = vae.decode(zs)
        return torch.stack([d.permute(1, 0, 2, 3) for d in decoded])

    @abstractmethod
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
        """Run the denoiser on ``[B,F,C,H,W]`` latents, return a flow prediction.

        The recipe calls this generically as ``denoise(model, context=..., **inputs)``
        where ``inputs`` is the subset of the preprocessed batch matching these
        keyword names; the adapter reshapes ``timestep`` to its model family's
        convention (e.g. ``[B]`` for the bidirectional Wan DiT, ``[B*F]`` for HY).

        Args:
            model (nn.Module): the denoiser to call.
            noisy (Tensor): noisy latents ``[B, F, C, H, W]``.
            timestep (Tensor): per-frame timesteps ``[B, F]``.
            context (Any): the opaque context from :meth:`encode_text`.
            clean (Tensor, optional): clean teacher-forcing context ``[B, F, C, H, W]``.
            aug_t (Tensor, optional): timesteps for the clean half (noise augmentation).
            viewmats (Tensor, optional): camera extrinsics ``[B, F, 4, 4]``.
            Ks (Tensor, optional): camera intrinsics ``[B, F, 3, 3]``.

        Returns:
            Tensor: flow (velocity) prediction ``[B, F, C, H, W]``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Self-forcing rollout hooks (DMD Stage 3)
    #
    # These three hooks let a single model-agnostic ``SelfForcingPipeline`` loop
    # drive either model family: the pipeline owns the outer block/step loop, the
    # x0 conversion, and the re-noise; the adapter owns the KV-cache shape, the
    # cached AR forward, and the per-block cache-refresh — i.e. everything that
    # depends on how a specific transformer's attention / cache is called.
    # ------------------------------------------------------------------

    @abstractmethod
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
        """Build the per-rollout cache state for the self-forcing loop.

        Returns an opaque dict the pipeline threads back into
        :meth:`rollout_forward` / :meth:`rollout_refresh_cache` without
        inspecting it. Its contents are model-specific (Wan: a per-block
        self-attn KV buffer plus a cross-attn cache; HY: a per-block
        ``k_txt/v_txt/k_vision/v_vision`` cache pre-populated with the text KV).

        Args:
            model (nn.Module): the generator being rolled out.
            batch_size (int): batch size ``B``.
            num_frames (int): total latent frames ``F``.
            frame_seqlen (int): tokens per frame after patch embedding.
            seq_len (int): total token count ``F * frame_seqlen``.
            dtype (torch.dtype): cache dtype.
            device (torch.device): cache device.
            cond (dict): the conditional stream from :meth:`conditioning`.
            use_prope_cache (bool): whether camera-control adapters should
                allocate their separate PRoPE KV cache.
            use_local_window (bool): whether fixed-buffer caches should be sized
                to the model's local-attention window instead of the full clip.
                Deployment inference sets this; the training self-forcing loop
                leaves it False. Adapters whose cache grows by concatenation (HY)
                ignore it.

        Returns:
            dict[str, Any]: the opaque cache state for this rollout.
        """
        raise NotImplementedError

    @abstractmethod
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
        """Cached autoregressive forward for one temporal block, returning flow.

        Args:
            model (nn.Module): the generator.
            noisy_block (Tensor): the block's noisy latents ``[B, n_f, C, H, W]``.
            timestep (Tensor): per-frame timesteps for the block ``[B, n_f]``.
            cache (dict): the opaque cache state from :meth:`rollout_init_cache`.
            cond (dict): the conditional stream from :meth:`conditioning`.
            meta (dict): block metadata (``f_start`` / ``f_end`` / ``block_idx`` /
                ``frame_seqlen`` / ``seq_len``) the adapter needs for RoPE offsets.
            viewmats (Tensor, optional): block camera extrinsics ``[B, n_f, 4, 4]``.
            Ks (Tensor, optional): block camera intrinsics ``[B, n_f, 3, 3]``.
            action (Tensor, optional): block discrete-action conditioning ``[B, n_f]``.

        Returns:
            Tensor: flow (velocity) prediction for the block ``[B, n_f, C, H, W]``.
        """
        raise NotImplementedError

    @abstractmethod
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
        """No-grad cache-refresh pass appending this block's KV for later blocks.

        Runs a clean forward over the block's denoised latents so that subsequent
        blocks attend to it. Wan re-runs at ``t=0`` (its cache buffer is written
        in place by the model); HY runs a context-rerun at ``context_noise`` and
        concatenates the returned vision KV onto ``cache``.

        Args:
            model (nn.Module): the generator.
            clean_block (Tensor): the block's denoised latents ``[B, n_f, C, H, W]``.
            cache (dict): the opaque cache state (mutated in place).
            cond (dict): the conditional stream from :meth:`conditioning`.
            meta (dict): block metadata (see :meth:`rollout_forward`).
            viewmats (Tensor, optional): block camera extrinsics ``[B, n_f, 4, 4]``.
            Ks (Tensor, optional): block camera intrinsics ``[B, n_f, 3, 3]``.
            action (Tensor, optional): block discrete-action conditioning ``[B, n_f]``.
        """
        raise NotImplementedError
