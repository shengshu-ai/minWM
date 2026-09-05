"""Wan Action-T2V ODE-distill — causal ODE distillation initialization.

Regress the causal student against a pre-solved ODE trajectory (Causal Forcing
init).
Same causal ``CausalWan21Model`` + PRoPE camera stream as ``tf.py``, but the training
signal is ODE regression in ``x0`` space instead of flow matching.

Differences from ``tf.py``:
- ``ODERegressionLoss`` (regress ``x0`` against the trajectory's near-clean
  target) instead of ``FlowMatchingLoss``;
- the ODE trajectory dataset (``CameraODERegressionLMDBDataset``) emits a
  pre-solved ``ode_latent`` ``[N, F, C, H, W]`` (N=6: 4 denoising points +
  48-step target + clean) rather than a single clean latent;
- a single ``ODETrajectorySample`` preprocessor samples one of the 4 denoising
  points and labels it with the WARPED timestep (``warp_denoising_step=True``) —
  the trajectory was solved on the shifted flow schedule, so the raw step values
  ``[1000, 750, 500, 250]`` must be mapped to their schedule timesteps;
- optimizer ``betas=(0.9, 0.999)``, ``weight_decay=0.01`` (matching the reference
  ODE stage), not ``tf.py``'s ``(0.0, 0.999)`` / ``0.0``;
- weights start from the promoted TF (AR teacher-forcing) checkpoint.

``sigma_min=0.0`` / ``extra_one_step=True`` and a 1000-entry timestep table
(``num_inference_steps`` left unset) reproduce the legacy Wan schedule the warp
indexes into.

    torchrun --nproc_per_node=8 tools/train_mwm.py \
        --config-file configs/wan21/action2v/train/stage2_ar_ode.py \
        --output-dir logs/wan21/ode_distill training.sp_size=4 \
        checkpoint.pretrained=/path/to/tf/model.pt
"""

_base_ = "../../../_base_/default.py"

# Wan2.1-T2V-1.3B base: diffusers-format dir (config.json + safetensors) plus the
# umt5 encoder checkpoint + tokenizer used for text conditioning. Defaults to the
# README model-download target; symlink here if your copy lives elsewhere.
_BASE = "./ckpts/Wan2.1-T2V-1.3B"
# Pre-solved ODE-trajectory LMDB (ode_latent/prompts/viewmats/Ks). The solver
# emits per-clip ``.pt`` files (keys: prompt, latents [1,6,F,C,H,W], viewmats, Ks);
# build them into a single LMDB with the legacy ``wan_utils/build_ode_prope_lmdb.py``
# (latents row shape S=6,F,C,H,W). Defaults to the README data-prep output; symlink
# here if your LMDB lives elsewhere
# (``ln -s /abs/path/to/ode_lmdb ./dataset/Wan21/Action2V/ode_lmdb``).
# Verified end-to-end on a 16-clip subset LMDB (loss decreases, no OOM).
_LMDB = "./dataset/Wan21/Action2V/ode_lmdb"
# Promoted TF (AR teacher-forcing) checkpoint to initialize from.
_TF = "./ckpts/Wan21/Action2V/stage1_ar_tf/model.pt"

# Raw ODE denoising steps (most-noisy -> least-noisy). Warped to schedule
# timesteps by ODETrajectorySample; their count (4) bounds the sampled step idx.
_DENOISING_STEPS = [1000, 750, 500, 250]

training = dict(
    max_steps=10_000,
    log_interval=10,
    ckpt_interval=1_000,
    output_dir="logs/wan21/stage2_ar_ode",
    # Recompute block activations in backward so the full 1.3B causal model fits
    # one GPU without FSDP.
    activation_checkpointing=True,
)

# Initialize from the promoted TF checkpoint (weights only; strict load keeps
# the PRoPE params). The legacy TF ckpt wraps its weights under ``generator``
# with ``model.``-prefixed keys; the trainer's loader unwraps that. Override the
# path on the CLI if your promoted ckpt lives elsewhere:
#     checkpoint.pretrained=/path/to/tf/model.pt
checkpoint = dict(pretrained=_TF)

# Causal Wan21Model from the base checkpoint; dims come from the base config.json.
# Same knobs as tf.py (use_prope camera stream, AR chunk size 4, temporal
# window 20). from_pretrained loads base weights non-strictly; low_cpu_mem_usage
# materializes prope_o instead of leaving it on meta.
model = dict(
    type="CausalWan21Model",
    _from_pretrained=_BASE,
    use_prope=True,
    num_frame_per_block=4,
    local_attn_size=20,
    low_cpu_mem_usage=False,
)

recipe = dict(
    type="ARODERecipe",
    optimizer=dict(lr=2e-6, betas=(0.9, 0.999), weight_decay=0.01),
    timestep_shift=5.0,
    # Match the legacy Wan noise schedule (1000-entry table for the warp).
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
    loss=dict(type="ODERegressionLoss"),
    batch_preprocessors=[
        dict(type="LatentToDevice"),
        # ODE regression: sample one denoising point, label with the warped
        # schedule timestep (the recipe injects its scheduler for the warp).
        dict(
            type="ODETrajectorySample",
            denoising_step_list=_DENOISING_STEPS,
            warp_denoising_step=True,
        ),
    ],
)

data = dict(
    dataset=dict(
        type="CameraODERegressionLMDBDataset",
        data_path=_LMDB,
    ),
    batch_size=1,
    num_workers=4,
    shuffle=True,
    drop_last=True,
)
