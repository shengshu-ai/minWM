"""Flow-matching preprocessors: noise/timestep sampling + clean-context aug.

The noise/timestep draw must be identical within an SP group (Ulysses splits one
sample across the group) but differ across data-parallel replicas. Both hold
because the draws run inside ``get_rng_states_tracker().fork()`` (see
:mod:`minwm.distributed.rng`): the ``data-parallel-rng`` stream is seeded ``seed +
dp_rank`` at dist-init — constant within an SP group, distinct between groups — and
``fork()`` advances that stream in isolation, so an unrelated RNG-consuming kernel
(dropout, etc.) between two draws can't perturb it and a rank on a divergent code
path can't silently desync the shared stream.
"""

import torch

from minwm.sampling.schedulers import FlowMatchingScheduler

from .base import BatchPreprocessor


class FlowNoise(BatchPreprocessor):
    """Sample timesteps + noise and build the flow-matching training tensors.

    Reads ``clean_latent`` ``[B, F, C, H, W]`` and writes ``noise``, ``noisy``,
    ``timestep`` ``[B, F]``, ``target``, and ``weight`` ``[B, F, 1, 1, 1]`` (or
    ``None``). The shared ``noise`` is left on the batch so a downstream
    :class:`CleanContextNoiseAug` can re-noise the clean context with it.

    Noise / timestep are drawn inside the RNG-tracker ``fork()`` (seeded ``seed +
    dp_rank`` at dist-init), so they are identical within an SP group and distinct
    across DP groups — see the module docstring.

    Args:
        uniform_across_frames (bool): if True (bidirectional SFT) all frames in a
            sample share one timestep; if False (AR / diffusion forcing) each
            frame is sampled independently unless ``num_frame_per_block > 1``.
        num_frame_per_block (int): when greater than one, reuse the first sampled
            timestep within every temporal block. This matches block-wise causal
            training while preserving the default per-frame behavior.
        num_prefix_frames (int): leading fixed-prefix frames excluded from block
            grouping. Their timestep may be overwritten by a downstream fixed-
            prefix preprocessor.
        weighting_scheme (str, optional): timestep-sampling density for HY runs
            (e.g. ``"logit_normal"``); ``None`` (default) is the uniform Wan path.
        logit_mean (float): logit-normal mean.
        logit_std (float): logit-normal std.
        mode_scale (float): mode-weighting scale (``weighting_scheme="mode"``).
        apply_weight (bool): if True write the per-timestep loss weight; if False
            write ``weight=None`` for an unweighted (masked) mean.
        scheduler (FlowMatchingScheduler, optional): the recipe's shared scheduler,
            injected in :meth:`~minwm.engine.recipe_base.Recipe.build_preprocessors`.
    """

    def __init__(
        self,
        uniform_across_frames: bool,
        num_frame_per_block: int = 1,
        num_prefix_frames: int = 0,
        weighting_scheme: str | None = None,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 0.0,
        apply_weight: bool = True,
        scheduler: FlowMatchingScheduler | None = None,
    ) -> None:
        if num_frame_per_block < 1:
            raise ValueError("num_frame_per_block must be positive")
        if num_prefix_frames < 0:
            raise ValueError("num_prefix_frames must be non-negative")
        self.uniform_across_frames = uniform_across_frames
        self.num_frame_per_block = num_frame_per_block
        self.num_prefix_frames = num_prefix_frames
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale
        self.apply_weight = apply_weight
        self.scheduler = scheduler

    def __call__(self, batch: dict, device: torch.device) -> dict:
        from minwm.distributed import get_rng_states_tracker

        clean = batch["clean_latent"]
        B, F = clean.shape[:2]
        with get_rng_states_tracker().fork():
            noise = torch.randn(clean.shape, dtype=clean.dtype).to(device)
            timestep = self.scheduler.sample_timesteps(
                B,
                F,
                device,
                uniform_across_frames=self.uniform_across_frames,
                weighting_scheme=self.weighting_scheme,
                logit_mean=self.logit_mean,
                logit_std=self.logit_std,
                mode_scale=self.mode_scale,
            )
            if not self.uniform_across_frames and self.num_frame_per_block > 1:
                suffix_frames = F - self.num_prefix_frames
                if suffix_frames <= 0 or suffix_frames % self.num_frame_per_block != 0:
                    raise ValueError(
                        f"num_frames - num_prefix_frames={suffix_frames} must be "
                        f"positive and divisible by num_frame_per_block="
                        f"{self.num_frame_per_block}"
                    )
                suffix = timestep[:, self.num_prefix_frames :].reshape(
                    B, -1, self.num_frame_per_block
                )
                suffix = suffix[:, :, :1].expand_as(suffix).reshape(B, suffix_frames)
                timestep = torch.cat([timestep[:, : self.num_prefix_frames], suffix], dim=1)
        noisy = self.scheduler.add_noise(
            clean.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (B, F))
        if self.apply_weight:
            weight = (
                self.scheduler.training_weight(timestep.flatten(0, 1))
                .unflatten(0, (B, F))
                .reshape(B, F, 1, 1, 1)
            )
        else:
            weight = None
        batch["noise"] = noise
        batch["noisy"] = noisy
        batch["timestep"] = timestep
        batch["target"] = self.scheduler.training_target(clean, noise)
        batch["weight"] = weight
        return batch


class FirstFrameConditioning(BatchPreprocessor):
    """Turn the leading latent frames into fixed, loss-free conditioning.

    This preprocessor is protocol-level rather than model-specific: it runs after
    :class:`FlowNoise`, restores the clean prefix in ``noisy``, assigns timestep
    zero to that prefix, and writes a broadcastable ``loss_mask``. Models only
    receive the already prepared tensors.

    Args:
        num_prefix_frames (int): number of clean leading frames to preserve.
        initial_latent_key (str): optional batch key containing an explicit
            prefix. When absent, the prefix is read from ``clean_latent``.
    """

    def __init__(
        self,
        num_prefix_frames: int = 1,
        initial_latent_key: str = "initial_latent",
        source_key: str = "clean_latent",
    ) -> None:
        if num_prefix_frames < 1:
            raise ValueError("num_prefix_frames must be positive")
        self.num_prefix_frames = num_prefix_frames
        self.initial_latent_key = initial_latent_key
        self.source_key = source_key

    def __call__(self, batch: dict, device: torch.device) -> dict:
        clean = batch[self.source_key]
        prefix_frames = self.num_prefix_frames
        if clean.shape[1] <= prefix_frames:
            raise ValueError(
                "fixed-prefix training requires at least one predicted frame; "
                f"got {clean.shape[1]} total and {prefix_frames} prefix frames"
            )

        prefix = batch.get(self.initial_latent_key)
        if prefix is None:
            prefix = clean[:, :prefix_frames]
        if prefix.shape != clean[:, :prefix_frames].shape:
            raise ValueError(
                f"{self.initial_latent_key} must have shape "
                f"{tuple(clean[:, :prefix_frames].shape)}, got {tuple(prefix.shape)}"
            )

        batch[self.initial_latent_key] = prefix.to(device=device, dtype=clean.dtype)
        if "noisy" in batch:
            noisy = batch["noisy"]
            batch["noisy"] = noisy.clone()
            batch["noisy"][:, :prefix_frames] = prefix.to(device=noisy.device, dtype=noisy.dtype)
        if "timestep" in batch:
            batch["timestep"] = batch["timestep"].clone()
            batch["timestep"][:, :prefix_frames] = 0
        loss_mask = torch.ones(
            clean.shape[0], clean.shape[1], 1, 1, 1, device=device, dtype=torch.bool
        )
        loss_mask[:, :prefix_frames] = False
        batch["loss_mask"] = loss_mask
        return batch


class CleanContextNoiseAug(BatchPreprocessor):
    """Build the teacher-forcing clean context, optionally noise-augmented.

    Writes ``clean`` ``[B, F, C, H, W]`` and ``aug_t`` (``[B, F]`` or
    ``None``). With ``max_timestep == 0`` the clean latent passes through
    untouched and ``aug_t`` is ``None``. Otherwise the clean context is noised to
    a random timestep in ``[max_timestep, num_train_timesteps)`` using the
    ``noise`` left on the batch by :class:`FlowNoise`. The augmentation timestep
    is drawn inside the RNG-tracker ``fork()`` (see the module docstring).

    Args:
        max_timestep (int): noise-augmentation lower bound; ``0`` disables it.
        scheduler (FlowMatchingScheduler, optional): the recipe's shared scheduler,
            injected in :meth:`~minwm.engine.recipe_base.Recipe.build_preprocessors`.
    """

    def __init__(
        self,
        max_timestep: int,
        scheduler: FlowMatchingScheduler | None = None,
    ) -> None:
        self.max_timestep = max_timestep
        self.scheduler = scheduler

    def __call__(self, batch: dict, device: torch.device) -> dict:
        clean = batch["clean_latent"]
        if self.max_timestep <= 0:
            batch["clean"] = clean
            batch["aug_t"] = None
            return batch

        B, F = clean.shape[:2]
        noise = batch["noise"]
        n = self.scheduler.timesteps.shape[0]
        hi = int(self.max_timestep / self.scheduler.num_train_timesteps * n)
        from minwm.distributed import get_rng_states_tracker

        with get_rng_states_tracker().fork():
            aug_idx = torch.randint(hi, n, (B, F)).to(device)
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        aug_t = self.scheduler.timesteps[aug_idx]
        batch["clean"] = self.scheduler.add_noise(
            clean.flatten(0, 1), noise.flatten(0, 1), aug_t.flatten(0, 1)
        ).unflatten(0, (B, F))
        batch["aug_t"] = aug_t
        return batch
