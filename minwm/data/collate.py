"""Collate adapters that bridge dataset output to the recipe batch contract.

The minwm Stage-2 recipes were written against a *mock* batch contract —
``clean_latent`` ``[B, F, C, H, W]`` (or ``ode_latent`` ``[B, N, F, C, H, W]``)
plus ``viewmats`` / ``Ks`` and a list of string ``prompts``. The real HY camera
datasets (:mod:`minwm.data.datasets.camera_pt`) instead emit per-sample
``latent`` / ``ode_trajectory`` in ``[C, F, ...]`` axis order, pre-encoded text
embeddings, and bookkeeping fields like ``video_path`` (a ``str``).

:class:`HYCollator` reconciles the two without touching either side: it stacks
tensor fields, passes list/str/int fields through as Python lists, renames the
latent fields, and transposes the channel/frame axes so the existing recipes
consume real data unchanged. HY-only signals (``action``, ``byt5_text_states``,
``image_cond`` …) are stacked and kept in the batch even though the current
recipes ignore them, so nothing is lost ahead of wiring the real HY forward.
"""

import torch
from torch import Tensor

__all__ = ["HYCollator"]

# Per-sample latent fields are stored channel-first ``[C, F, ...]`` but the
# recipes expect frame-first ``[F, C, ...]``. Each entry maps a source key to
# its ``(new_name, axis_a, axis_b)`` rename + channel/frame swap.
_LATENT_RENAME = {
    "latent": ("clean_latent", 0, 1),  # [C, F, H, W] -> [F, C, H, W]
    "ode_trajectory": ("ode_latent", 1, 2),  # [N, C, F, H, W] -> [N, F, C, H, W]
}


class HYCollator:
    """Collate HY camera samples into the recipe batch contract.

    Stacks tensor fields along a new batch dim, leaves non-tensor fields as
    Python lists, and renames + axis-swaps the latent fields (see
    :data:`_LATENT_RENAME`) so the channel/frame order matches what the
    recipes expect.

    Args:
        rename_latents (bool): apply the latent rename + channel/frame swap.
            Set ``False`` to keep raw dataset keys/axes (e.g. for debugging).
    """

    def __init__(self, rename_latents: bool = True):
        self.rename_latents = rename_latents

    def __call__(self, batch: list[dict]) -> dict:
        out: dict = {}
        for key in batch[0]:
            values = [sample[key] for sample in batch]
            if not isinstance(values[0], Tensor):
                out[key] = values
                continue
            if self.rename_latents and key in _LATENT_RENAME:
                new_key, axis_a, axis_b = _LATENT_RENAME[key]
                values = [v.transpose(axis_a, axis_b) for v in values]
                out[new_key] = torch.stack(values)
            else:
                out[key] = torch.stack(values)
        return out
