"""CPU tests for the FSDP2 layer-selection + activation-checkpointing helpers.

These exercise the selection logic that decides *which* modules get sharded and
checkpointed, without invoking ``fully_shard`` (which needs a process group).
Both backbone predicate styles are covered: Wan's ``parts[1].isdigit()`` (which
also matches a block's descendants) and HY's last-part-digit (block only).
"""

import pytest
from torch import nn

from minwm.distributed.fsdp import _apply_activation_checkpointing, _iter_fsdp_targets


def _wan_pred(name: str, module: nn.Module) -> bool:
    """Wan-style: matches ``blocks.<i>`` *and* every descendant under it."""
    parts = name.split(".")
    return len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit()


def _hy_pred(name: str, module: nn.Module) -> bool:
    """HY-style: matches only the indexed block module itself."""
    return "blocks" in name and name.split(".")[-1].isdigit()


class _Block(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.ffn = nn.Linear(dim, dim)


class _ToyModel(nn.Module):
    def __init__(self, n=3, dim=4):
        super().__init__()
        self.blocks = nn.ModuleList(_Block(dim) for _ in range(n))
        self.head = nn.Linear(dim, dim)


def _is_ckpt(module: nn.Module) -> bool:
    return type(module).__name__ == "CheckpointWrapper"


@pytest.mark.parametrize("pred", [_wan_pred, _hy_pred])
def test_iter_targets_selects_top_most_blocks_only(pred):
    model = _ToyModel(n=3)
    names = [name for name, _ in _iter_fsdp_targets(model, [pred])]
    # Exactly the 3 top-level blocks — never their attn/ffn descendants, even
    # for the Wan predicate that would also match `blocks.0.attn`.
    assert names == ["blocks.0", "blocks.1", "blocks.2"]


@pytest.mark.parametrize("pred", [_wan_pred, _hy_pred])
def test_activation_checkpointing_wraps_each_block_once(pred):
    model = _ToyModel(n=3)
    names = [name for name, _ in _iter_fsdp_targets(model, [pred])]
    _apply_activation_checkpointing(model, names)

    # One wrapper per block, and no nesting: the wrapped inner module is a plain
    # _Block, not another CheckpointWrapper.
    assert all(_is_ckpt(b) for b in model.blocks)
    assert sum(_is_ckpt(m) for m in model.modules()) == 3
    for wrapped in model.blocks:
        assert not _is_ckpt(wrapped._checkpoint_wrapped_module)


@pytest.mark.parametrize("pred", [_wan_pred, _hy_pred])
def test_iter_targets_after_ac_selects_block_wrappers_once(pred):
    """After AC, the shard pass re-selects each block's wrapper exactly once.

    This is what lets ``shard_model`` apply ``fully_shard`` *outside* the
    checkpoint wrapper (FSDP(Checkpoint(block))) without descending into the
    renamed ``_checkpoint_wrapped_module`` internals.
    """
    model = _ToyModel(n=3)
    names = [name for name, _ in _iter_fsdp_targets(model, [pred])]
    _apply_activation_checkpointing(model, names)

    targets = list(_iter_fsdp_targets(model, [pred]))
    assert [name for name, _ in targets] == ["blocks.0", "blocks.1", "blocks.2"]
    assert all(_is_ckpt(module) for _, module in targets)
