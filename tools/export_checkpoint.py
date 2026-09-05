"""Export a minWM DCP training checkpoint to a single-file model weight.

FSDP2 training saves checkpoints in PyTorch Distributed Checkpoint (DCP)
format — one shard per rank, unreadable by plain ``torch.load``. This tool
consolidates the shards into a single file that the inference engine can consume
directly.

Three output formats are supported:

* ``pt``  — a plain ``{"model": state_dict}`` ``.pt`` file loaded with
  ``torch.load``; the format ``load_weights`` already handles natively (used by
  Wan configs via ``inference.checkpoint``).
* ``safetensors`` — a ``HuggingFace`` safetensors file; safer (no pickle),
  memory-mappable, and compatible with the broader ecosystem.
* ``diffusers`` — a ``config.json`` + ``diffusion_pytorch_model.safetensors``
  directory, loadable via ``model=dict(_from_pretrained=<dir>)`` (the path HY
  configs use). Requires ``--config`` to supply the model architecture. This is
  the format a promote step should produce: a bare ``model.pt`` cannot be
  loaded by ``ModelMixin.from_pretrained``.

The tool strips common FSDP / compile wrapper prefixes (via
:func:`minwm.engine.checkpoint.formats.strip_wrapper_prefix`) so the exported
weights are keyed exactly as the bare model expects
(``blocks.0.self_attn.q.weight``, not ``model._fsdp_wrapped_module.blocks.0...``).

Usage (single GPU, no torchrun needed)::

    # --output omitted → writes ./checkpoint_1000.pt next to nothing (cwd)
    python tools/export_checkpoint.py \\
        --checkpoint logs/wan21/action2v/sft/checkpoint_1000

    python tools/export_checkpoint.py \\
        --checkpoint logs/wan21/action2v/sft/checkpoint_1000 \\
        --output model.safetensors \\
        --format safetensors

    # diffusers dir for a from_pretrained inference config (needs --config)
    python tools/export_checkpoint.py \\
        --checkpoint outputs/hy_action2v_stage0_bi_sft/checkpoint_3000 \\
        --output ./ckpts/HY15/Action2V/stage0_bi_sft \\
        --format diffusers \\
        --config configs/hy/action2v/train/stage0_bi_sft.py
"""

import argparse
import os
import sys

import torch

from minwm.engine.checkpoint.checkpointer import dcp_reader
from minwm.engine.checkpoint.formats import strip_wrapper_prefix
from minwm.engine.checkpoint.storage import Storage, is_remote, join


def _available_app_keys(checkpoint_dir: str, storage: Storage) -> list[str]:
    """Top-level entries under ``app`` in a DCP checkpoint's metadata.

    Read straight from the metadata (no tensor data), used only to build a
    helpful error when the requested ``model_key`` is absent — the keyed load in
    :func:`_load_dcp` returns an empty dict for a missing key, so the real entry
    names must come from here.
    """
    reader = dcp_reader(checkpoint_dir, storage)
    md = reader.read_metadata()
    seen = []
    for flat in md.state_dict_metadata:
        # flattened keys look like "app.model.blocks.0.weight" or
        # "app.opt/main.state..." — recover the entry right under "app".
        parts = flat.split(".")
        if len(parts) >= 2 and parts[0] == "app" and parts[1] not in seen:
            seen.append(parts[1])
    return seen


def _load_dcp(
    checkpoint_dir: str, model_key: str = "model", storage: Storage | None = None
) -> dict[str, torch.Tensor]:
    """Consolidate a DCP checkpoint directory into a flat model state dict.

    Reads only the requested ``model_key`` subtree (local or remote) in a single
    process — no ``torchrun`` — via torch's ``_load_state_dict_from_keys``, which
    pulls just the entries under the given metadata prefix and so skips optimizer
    / RNG state (~2/3 of a training checkpoint's bytes). The trainer saves
    ``{"app": {"step", "model", "aux/<name>", "opt/<name>", "rng"}}`` (see
    ``minwm.engine.trainer._TrainerAppState``); ``model_key`` selects the primary
    model or an auxiliary model such as ``aux/generator_ema``, and wrapper
    prefixes are stripped from every key.

    Args:
        checkpoint_dir (str): DCP checkpoint directory path or ``s3://`` /
            ``oss://`` URL.
        model_key (str): model entry inside the ``app`` state to export.
        storage (Storage | None): storage backend for remote reads; defaults to
            a local/remote auto-dispatching :class:`Storage`.

    Returns:
        dict[str, torch.Tensor]: model weights keyed as the bare model expects.

    Raises:
        RuntimeError: if the checkpoint has no ``model_key`` entry, or the entry
            is not a state dict.
    """
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys

    storage = storage or Storage()
    print(f"consolidating DCP shards from {checkpoint_dir} (key={model_key!r}) …", flush=True)
    # Read only the model subtree; the trainer nests it under "app", so the
    # metadata prefix is "app.<model_key>". Returns a nested {"app": {...}}.
    state = _load_state_dict_from_keys(
        {f"app.{model_key}"}, storage_reader=dcp_reader(checkpoint_dir, storage)
    )
    if "app" in state:
        state = state["app"]
    weights = state.get(model_key)
    if not weights:
        # Keyed load yields an empty result for a missing key, so list the real
        # entries from metadata rather than the (empty) loaded state.
        available = _available_app_keys(checkpoint_dir, storage)
        raise RuntimeError(f"checkpoint has no {model_key!r} entry; available entries: {available}")
    if not isinstance(weights, dict):
        raise RuntimeError(f"{model_key!r} entry is not a state dict but {type(weights).__name__}")
    return {strip_wrapper_prefix(k): v for k, v in weights.items()}


def _ensure_parent(output: str) -> None:
    """Create the parent directory of a local output path if it doesn't exist.

    Skips remote (``s3://`` / ``oss://``) paths, where object PUTs need no parent
    directory. Lets the export helpers be called with a fresh destination (e.g.
    ``auto_dump`` submitting to the cluster) without a separate mkdir step.
    """
    if is_remote(output):
        return
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)


def export_pt(weights: dict[str, torch.Tensor], output: str) -> None:
    """Save weights as a ``{"model": weights}`` .pt file."""
    _ensure_parent(output)
    torch.save({"model": weights}, output)
    size_mb = os.path.getsize(output) / 1024**2
    print(f"saved .pt  → {output}  ({size_mb:.0f} MB)")


def export_safetensors(weights: dict[str, torch.Tensor], output: str) -> None:
    """Save weights as a safetensors file."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("safetensors not installed. Run: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    _ensure_parent(output)
    # safetensors requires contiguous tensors.
    weights = {k: v.contiguous() for k, v in weights.items()}
    save_file(weights, output)
    size_mb = os.path.getsize(output) / 1024**2
    print(f"saved .safetensors → {output}  ({size_mb:.0f} MB)")


def export_diffusers(weights: dict[str, torch.Tensor], output_dir: str, config_file: str) -> None:
    """Save weights as a diffusers directory (``config.json`` + safetensors).

    Builds the model architecture from a config's ``model`` node (a diffusers
    ``ModelMixin`` — the Wan or HY transformer), overlays the consolidated
    trained weights, and calls ``save_pretrained`` so the result loads via
    ``model=dict(_from_pretrained=<dir>)`` — the same path inference uses. This
    is the format the promote step should produce; a bare ``model.pt`` cannot be
    loaded by ``ModelMixin.from_pretrained``.

    Args:
        weights (dict[str, Tensor]): consolidated model weights.
        output_dir (str): directory to write ``config.json`` + safetensors into.
        config_file (str): config whose ``model`` node defines the architecture.

    Raises:
        RuntimeError: if the config has no ``model`` node, or the built model is
            not a diffusers ``save_pretrained``-able module.
    """
    from minwm.config import load
    from minwm.modeling.build import build_model

    cfg = load(config_file)
    if "model" not in cfg:
        raise RuntimeError(f"config {config_file} has no 'model' node to define the architecture")
    print(f"building model architecture from {config_file} …")
    model = build_model(dict(cfg["model"]))
    if not hasattr(model, "save_pretrained"):
        raise RuntimeError(
            f"{type(model).__name__} has no 'save_pretrained'; diffusers format needs a ModelMixin"
        )
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        print(f"  note: {len(missing)} missing / {len(unexpected)} unexpected keys vs architecture")
    model.save_pretrained(output_dir, safe_serialization=True)
    print(
        f"saved diffusers dir → {output_dir}  (config.json + diffusion_pytorch_model.safetensors)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a minWM DCP checkpoint to .pt or .safetensors."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the DCP checkpoint directory (e.g. logs/.../checkpoint_1000).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output file path (e.g. model.pt or model.safetensors). Defaults to "
            "the checkpoint directory name + format extension in the cwd "
            "(checkpoint_1000 → ./checkpoint_1000.pt)."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["pt", "safetensors", "diffusers"],
        default=None,
        help=(
            "Output format. Inferred from --output extension when both are given "
            "(*.safetensors → safetensors, else pt); defaults to pt otherwise. "
            "'diffusers' writes a config.json + safetensors directory loadable via "
            "model=dict(_from_pretrained=<dir>); requires --config."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Config file defining the model architecture (its 'model' node). "
            "Required for --format diffusers."
        ),
    )
    parser.add_argument(
        "--model-key",
        default="model",
        help=(
            "DCP model entry to export. Use 'aux/generator_ema' for the Wan21 "
            "DMD EMA generator; defaults to the primary 'model'."
        ),
    )
    return parser.parse_args()


def _resolve_output(checkpoint: str, output: str | None, fmt: str) -> str:
    """Pick the output path, defaulting to the checkpoint dir name + extension."""
    if output is not None:
        return output
    stem = os.path.basename(os.path.normpath(checkpoint))
    ext = "safetensors" if fmt == "safetensors" else "pt"
    return f"{stem}.{ext}"


def main() -> None:
    args = parse_args()

    if args.format is not None:
        fmt = args.format
    elif args.output is not None:
        fmt = "safetensors" if args.output.endswith(".safetensors") else "pt"
    else:
        fmt = "pt"

    storage = Storage()
    if is_remote(args.checkpoint):
        if not storage.exists(join(args.checkpoint, ".metadata")):
            print(f"not a DCP checkpoint (no .metadata): {args.checkpoint}", file=sys.stderr)
            sys.exit(1)
    elif not os.path.isdir(args.checkpoint):
        print(f"checkpoint directory not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    if fmt == "diffusers" and args.config is None:
        print("--format diffusers requires --config", file=sys.stderr)
        sys.exit(1)

    weights = _load_dcp(args.checkpoint, model_key=args.model_key, storage=storage)
    print(f"loaded {len(weights)} parameter tensors")

    if fmt == "diffusers":
        stem = os.path.basename(os.path.normpath(args.checkpoint))
        output = args.output or stem
        export_diffusers(weights, output, args.config)
    elif fmt == "safetensors":
        export_safetensors(weights, _resolve_output(args.checkpoint, args.output, fmt))
    else:
        export_pt(weights, _resolve_output(args.checkpoint, args.output, fmt))


if __name__ == "__main__":
    main()
