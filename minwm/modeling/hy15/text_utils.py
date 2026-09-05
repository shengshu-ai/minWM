"""Text-token helpers for HY15 transformer paths."""

import torch


def pack_text_tokens(
    txt: torch.Tensor, text_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pack text only for single-sample batches; keep B>1 mask-aligned."""
    text_mask = text_mask.bool().to(txt.device)
    if not text_mask.any(dim=1).all():
        raise ValueError("HY15 text_mask must contain at least one valid token per sample.")
    if txt.shape[0] == 1:
        return txt[text_mask].unsqueeze(0), None
    return txt, text_mask
