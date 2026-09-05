"""Wan Action-T2V TF — causal teacher-forcing AR diffusion on real data.

Second stage: convert the bidirectional SFT camera model into a causal
``CausalWan21Model`` and train
AR diffusion with teacher forcing on the same encoded camera LMDB. PRoPE camera
conditioning is kept (``use_prope=True``).

Differences from ``sft.py``:
- causal ``CausalWan21Model`` with ``num_frame_per_block=4`` and ``local_attn_size=20``;
- ``Wan21Adapter(causal=True)`` so the per-frame ``[B, F]`` timestep is passed through;
- block-wise timesteps (four frames share one noise level) plus a
  teacher-forced clean context (``CleanContextNoiseAug``);
- weights start from the promoted SFT checkpoint (``checkpoint.pretrained``),
  so prepare the ``_SFT`` artifact before starting this stage.

Precision is handled by the framework (``adapter.dtype="bfloat16"`` ->
``Recipe.autocast()`` single-GPU / FSDP mixed precision multi-GPU).

    torchrun --nproc_per_node=8 tools/train_mwm.py \
        --config-file configs/wan21/action2v/train/stage1_ar_tf.py \
        --output-dir logs/wan21/tf training.sp_size=4 \
        checkpoint.pretrained=/path/to/sft/model.pt
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
# Promoted SFT (bidirectional camera) checkpoint to start AR training from.
_SFT = "./ckpts/Wan21/Action2V/stage0_bi_sft/model.pt"

training = dict(
    max_steps=10_000,
    log_interval=10,
    ckpt_interval=1_000,
    output_dir="logs/wan21/stage1_ar_tf",
    # Recompute block activations in backward so the full 1.3B causal model + the
    # teacher-forcing-doubled sequence fits one GPU without FSDP.
    activation_checkpointing=True,
)

# Initialize AR training from the promoted SFT checkpoint.
checkpoint = dict(pretrained=_SFT)

# Causal Wan21Model from the base checkpoint; dims come from the base config.json.
# use_prope keeps the camera stream; num_frame_per_block=4 is the AR chunk size,
# local_attn_size=20 the temporal window. from_pretrained loads the base weights
# non-strictly (new zero-init prope_o is a no-op until trained);
# low_cpu_mem_usage=False materializes prope_o instead of leaving it on meta.
model = dict(
    type="CausalWan21Model",
    _from_pretrained=_BASE,
    use_prope=True,
    num_frame_per_block=4,
    local_attn_size=20,
    low_cpu_mem_usage=False,
)

recipe = dict(
    type="ARTFRecipe",
    optimizer=dict(lr=2e-6, betas=(0.0, 0.999), weight_decay=0.01),
    timestep_shift=5.0,
    # Match the legacy Wan noise schedule (verified max|Δ|=0 vs legacy).
    sigma_min=0.0,
    extra_one_step=True,
    adapter=dict(
        type="Wan21Adapter",
        causal=True,  # pass the per-frame [B, F] timestep through to the causal model
        text_len=512,
        text_dim=4096,
        dtype="bfloat16",
        text_encoder=dict(
            type="Wan21TextEncoder",
            checkpoint_path=f"{_BASE}/models_t5_umt5-xxl-enc-bf16.pth",
            tokenizer_path=f"{_BASE}/google/umt5-xxl/",
            text_len=512,
        ),
    ),
    loss=dict(type="FlowMatchingLoss"),
    batch_preprocessors=[
        dict(type="LatentToDevice"),
        # AR diffusion forcing: frames in one causal block share a timestep.
        dict(type="FlowNoise", uniform_across_frames=False, num_frame_per_block=4),
        # Teacher forcing: clean context, un-noised (max_timestep=0 -> aug_t=None).
        dict(type="CleanContextNoiseAug", max_timestep=0),
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
