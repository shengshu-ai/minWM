"""Wan Action-T2V SFT — bidirectional camera (PRoPE) supervised fine-tuning.

The entry point of the Wan Action-T2V pipeline. Mirrors
``configs/hy/action2v/train/stage0_bi_sft.py`` on the Wan backbone:
a bidirectional ``Wan21Model`` loaded from the Wan2.1-T2V-1.3B base (dims read from
its ``config.json``), with ``use_prope=True`` so the camera stream (viewmats/Ks)
modulates self-attention. Flow-matching SFT over the pre-encoded camera LMDB,
real umt5 text conditioning.

Precision is handled by the framework: the recipe's ``adapter.dtype="bfloat16"``
drives ``Recipe.autocast()`` on a single GPU and FSDP mixed precision
(``training.fsdp_param_dtype``) on multi-GPU — no manual model cast here.

Override ``_BASE`` / ``_LMDB`` (or pass on the CLI) to point at your paths:

    torchrun --nproc_per_node=8 tools/train_mwm.py \
        --config-file configs/wan21/action2v/train/stage0_bi_sft.py \
        --output-dir logs/wan21/sft training.sp_size=4
"""

_base_ = "../../../_base_/default.py"

# Wan2.1-T2V-1.3B base: diffusers-format dir (config.json + safetensors) plus the
# umt5 encoder checkpoint + tokenizer used for text conditioning. Defaults to the
# README model-download target; symlink here if your copy lives elsewhere.
_BASE = "./ckpts/Wan2.1-T2V-1.3B"
# Pre-encoded camera latent LMDB (latents/prompts/intrinsics/poses). Defaults to
# the README data-prep output; symlink here if your LMDB lives elsewhere
# (``ln -s /abs/path/to/data ./dataset/Wan21/Action2V/data``).
_LMDB = "./dataset/Wan21/Action2V/data"

training = dict(
    max_steps=10_000,
    log_interval=10,
    ckpt_interval=1_000,
    output_dir="logs/wan21/stage0_bi_sft",
    # Recompute block activations in backward so the full 1.3B model + full-res
    # 20-frame sequence fits one GPU without FSDP. Orthogonal to autocast precision.
    activation_checkpointing=True,
)

# Bidirectional Wan21Model from the base checkpoint; dims come from the base
# config.json, we only switch on the camera (PRoPE) stream. from_pretrained loads
# the base weights non-strictly, leaving the new zero-init prope_o a no-op until
# trained. low_cpu_mem_usage=False so use_prope's prope_o params (absent from the
# base checkpoint) are materialized instead of left on the meta device.
model = dict(
    type="Wan21Model",
    _from_pretrained=_BASE,
    use_prope=True,
    low_cpu_mem_usage=False,
)

recipe = dict(
    type="BiSFTRecipe",
    optimizer=dict(lr=2e-6, betas=(0.0, 0.999), weight_decay=0.01),
    timestep_shift=5.0,
    # Match the legacy Wan stage0 noise schedule (WanDiffusionWrapper overrode the
    # bare-class defaults to these), so the new framework's noisy/timestep inputs
    # replicate the old framework's exactly. Verified max|Δ|=0 vs legacy.
    sigma_min=0.0,
    extra_one_step=True,
    adapter=dict(
        type="Wan21Adapter",
        text_len=512,
        text_dim=4096,
        dtype="bfloat16",
        text_encoder=dict(
            type="Wan21TextEncoder",
            checkpoint_path=f"{_BASE}/models_t5_umt5-xxl-enc-bf16.pth",
            tokenizer_path=f"{_BASE}/google/umt5-xxl/",
            text_len=512,
            dtype="bfloat16",
        ),
    ),
    loss=dict(type="FlowMatchingLoss"),
    batch_preprocessors=[
        dict(type="LatentToDevice"),
        dict(type="FlowNoise", uniform_across_frames=True),
    ],
)

data = dict(
    dataset=dict(
        type="CameraLatentLMDBDataset",
        data_path=_LMDB,
    ),
    batch_size=1,
    num_workers=4,
    shuffle=True,
    drop_last=True,
)
