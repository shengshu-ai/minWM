"""Wan Action-T2V CM — causal consistency distillation on real data.

Distill the causal student into a few-step consistency model
(Causal Forcing++). A frozen
teacher does a CFG Euler step ``t -> t_next``; the student predicts ``x0`` at ``t``
and an EMA network predicts ``x0`` at ``t_next``; the loss is the MSE between the
two ``x0`` predictions.

Three model copies (all start from the promoted TF AR teacher-forcing
checkpoint, matching the legacy ``camera_naive_cd``):
- student (trainable) -> prediction at ``t``;
- ema (frozen EMA of student) -> target at ``t_next``;
- teacher (frozen) -> CFG Euler step ``t -> t_next``.

Teacher + EMA are trainer-owned ``auxiliary_models`` (frozen, FSDP-sharded
alongside the student); the recipe seeds them from the student via ``copy_params``
on the first step, so loading the TF checkpoint into the student seeds all
three identically (the legacy code loads the same ckpt into generator/teacher/ema).

Same causal ``CausalWan21Model`` + PRoPE camera stream as tf/ode_distill; differs
from ``ode_distill.py`` in the training signal (consistency distillation, not ODE
regression) and uses the clean-latent camera LMDB (no pre-solved trajectory).

    torchrun --nproc_per_node=8 tools/train_mwm.py \
        --config-file configs/wan21/action2v/train/stage2_ar_cd.py \
        --output-dir logs/wan21/cm training.sp_size=4 \
        checkpoint.pretrained=/path/to/tf/model.pt
"""

from minwm.modeling.wan21.adapter import DEFAULT_NEGATIVE_PROMPT as _NEGATIVE_PROMPT

_base_ = "../../../_base_/default.py"

# Wan2.1-T2V-1.3B base: diffusers-format dir (config.json + safetensors) plus the
# umt5 encoder checkpoint + tokenizer used for text conditioning. Defaults to the
# README model-download target; symlink here if your copy lives elsewhere.
_BASE = "./ckpts/Wan2.1-T2V-1.3B"
# Pre-encoded camera latent LMDB (latents/prompts/intrinsics/poses). Defaults to
# the README data-prep output; symlink here if your LMDB lives elsewhere
# (``ln -s /abs/path/to/data ./dataset/Wan21/Action2V/data``).
_LMDB = "./dataset/Wan21/Action2V/data"
# Promoted TF (AR teacher-forcing) checkpoint; seeds student/teacher/ema.
_TF = "./ckpts/Wan21/Action2V/stage1_ar_tf/model.pt"

training = dict(
    max_steps=10_000,
    log_interval=10,
    ckpt_interval=1_000,
    output_dir="logs/wan21/stage2_ar_cd",
    # Recompute block activations in backward; the recipe holds three 1.3B causal
    # copies (student + teacher + ema), so memory is tight.
    activation_checkpointing=True,
)

# Seed student/teacher/ema from the promoted TF checkpoint.
checkpoint = dict(pretrained=_TF)

# Causal Wan21Model from the base checkpoint; same knobs as tf.py / ode_distill.py.
_model = dict(
    type="CausalWan21Model",
    _from_pretrained=_BASE,
    use_prope=True,
    num_frame_per_block=4,
    local_attn_size=20,
    low_cpu_mem_usage=False,
)

model = dict(**_model)

# Frozen teacher + EMA, built and FSDP-sharded by the trainer alongside the
# student and seeded from it on the first step (see ARCDRecipe).
# shard=True so they match the student's DTensor layout (copy_params / EMA copy
# between them); same arch as the student, their base weights are overwritten by
# the student's via copy_params, so what matters here is only the architecture.
auxiliary_models = dict(
    teacher=dict(trainable=False, shard=True, **_model),
    ema=dict(trainable=False, shard=True, **_model),
)

recipe = dict(
    type="ARCDRecipe",
    optimizer=dict(lr=2e-6, betas=(0.0, 0.999), weight_decay=0.01),
    timestep_shift=5.0,
    # 50-entry discrete CD schedule; t/t_next pairs are drawn from it. sigma_min /
    # extra_one_step match the legacy Wan schedule the pairs come from.
    discrete_cd_n=50,
    sigma_min=0.0,
    extra_one_step=True,
    guidance_scale=3.0,
    ema_decay=0.99,
    adapter=dict(
        type="Wan21Adapter",
        causal=True,  # pass the per-frame [B, F] timestep through to the causal model
        text_len=512,
        text_dim=4096,
        dtype="bfloat16",
        negative_prompt=_NEGATIVE_PROMPT,
        text_encoder=dict(
            type="Wan21TextEncoder",
            checkpoint_path=f"{_BASE}/models_t5_umt5-xxl-enc-bf16.pth",
            tokenizer_path=f"{_BASE}/google/umt5-xxl/",
            text_len=512,
        ),
    ),
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
