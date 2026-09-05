"""Wan Action-T2V DMD — causal DMD distillation on real data.

Distill the causal student into a few-step generator with
Distribution Matching Distillation and self-rollout (the legacy
``causal_forcing_dmd_camera`` build). Four model copies,
two optimizers, alternating per step:

- generator (``model``, trainable, causal) -> backward-simulated via the
  self-forcing pipeline (no dataset clean latents drive it; only the shape is
  used);
- real_score (frozen teacher) -> fixed CFG-guided score;
- fake_score (trainable critic) -> learns the generator's output distribution.
- generator_ema (frozen EMA) -> promoted inference weights after step 200.

The generator updates once every ``dfake_gen_update_ratio`` steps (matching the
real/fake score x0 distributions, KL gradient); the critic updates every step
(flow-matching denoising loss on the generator's no-grad output). real_score is
built frozen+sharded and fake_score trainable+sharded by ``BaseTrainer`` from
``auxiliary_models``.

Seeds (legacy ``causal_forcing_dmd_camera.yaml``): the generator starts from the
promoted ODE-distill (causal ODE) checkpoint, the two score nets from the SFT
(bidirectional) checkpoint. ``denoising_step_list=[1000, 750, 500, 250]`` with
``warp_denoising_step=True`` (the raw steps are mapped onto the shifted schedule
the rollout re-noises against, as in ODE-distill). ``guidance_scale=3.0``,
``dfake_gen_update_ratio=5``, generator ``lr=2e-6`` / critic ``lr=4e-7``, both
``betas=(0.0, 0.999)``.

    torchrun --nproc_per_node=8 tools/train_mwm.py \
        --config-file configs/wan21/action2v/train/stage3_ar_dmd.py \
        --output-dir logs/wan21/dmd training.sp_size=4 \
        checkpoint.pretrained=/path/to/ode_distill/model.pt
"""

from minwm.modeling.wan21.adapter import DEFAULT_NEGATIVE_PROMPT as _NEGATIVE_PROMPT

_base_ = "../../../_base_/default.py"

# Wan2.1-T2V-1.3B base: diffusers-format dir (config.json + safetensors) plus the
# umt5 encoder checkpoint + tokenizer used for text conditioning. Defaults to the
# README model-download target; symlink here if your copy lives elsewhere.
_BASE = "./ckpts/Wan2.1-T2V-1.3B"
# Pre-encoded camera latent LMDB (latents/prompts/intrinsics/poses); DMD only uses
# the clean-latent SHAPE (the generator is backward-simulated from noise), plus the
# prompts + camera for conditioning. Defaults to the README data-prep output;
# symlink here if your LMDB lives elsewhere
# (``ln -s /abs/path/to/data ./dataset/Wan21/Action2V/data``).
_LMDB = "./dataset/Wan21/Action2V/data"
# Promoted ODE-distill (causal ODE) checkpoint; seeds the generator.
_ODE = "./ckpts/Wan21/Action2V/stage2_ar_ode/model.pt"
# SFT (bidirectional) checkpoint; seeds real_score + fake_score.
_SFT = "./ckpts/Wan21/Action2V/stage0_bi_sft/model.pt"

# Raw DMD denoising steps (most-noisy -> least-noisy). Warped onto the shifted
# schedule by the recipe (warp_denoising_step) before the self-rollout uses them.
_DENOISING_STEPS = [1000, 750, 500, 250]

training = dict(
    max_steps=10_000,
    log_interval=10,
    ckpt_interval=1_000,
    output_dir="logs/wan21/stage3_ar_dmd",
    # Recompute block activations in backward; the recipe holds four 1.3B copies
    # (causal generator/EMA + bidirectional score nets) and simulates a full
    # clip per step, so memory is tight.
    #
    # Never run this config with ``nproc_per_node=1``: besides crashing
    # activation_checkpointing, a single process is guaranteed to OOM.
    activation_checkpointing=True,
)

# Seed the causal generator/EMA from ODE distillation and both bidirectional
# score networks from SFT.
checkpoint = dict(
    pretrained=_ODE,
    auxiliary_pretrained={
        "real_score": _SFT,
        "fake_score": _SFT,
        "generator_ema": _ODE,
    },
)

# Causal generator; bidirectional score networks match the SFT checkpoint.
_generator_model = dict(
    type="CausalWan21Model",
    _from_pretrained=_BASE,
    use_prope=True,
    num_frame_per_block=4,
    local_attn_size=20,
    low_cpu_mem_usage=False,
)
_score_model = dict(
    type="Wan21Model",
    _from_pretrained=_BASE,
    use_prope=True,
    low_cpu_mem_usage=False,
)

model = dict(**_generator_model)

# real_score (frozen teacher) + fake_score (trainable critic), built by the
# trainer from these entries. Both sharded: real_score runs no-grad teacher
# forcing (no KV-cache state), so sharding it just all-gathers params for the
# forward then frees them — needed to fit three 1.3B nets + dense full-clip
# activations under FSDP. fake_score: trainable -> sharded, optimized by the
# critic optimizer. Both seed from the SFT checkpoint when one is supplied.
auxiliary_models = dict(
    real_score=dict(trainable=False, shard=True, **_score_model),
    fake_score=dict(trainable=True, **_score_model),
    generator_ema=dict(trainable=False, shard=True, **_generator_model),
)

recipe = dict(
    type="ARDMDRecipe",
    optimizer=dict(lr=2e-6, betas=(0.0, 0.999), weight_decay=0.01),
    critic_optimizer=dict(lr=4e-7, betas=(0.0, 0.999), weight_decay=0.01),
    timestep_shift=5.0,
    # Match the legacy Wan noise schedule so the framework's noisy/timestep inputs
    # replicate the reference exactly (same as sft/tf/ode_distill/cm). The DMD
    # renoise schedule MUST match the score nets' training schedule: real_score/
    # fake_score seed from phase1 (sigma_min=0.0 / extra_one_step=True, max|Δ|=0
    # vs legacy), so the noise level DMD adds at a timestep matches what those nets
    # learned. Without this the recipe falls back to 0.003/1.002 defaults and the
    # low-noise band diverges 2-3x.
    sigma_min=0.0,
    extra_one_step=True,
    denoising_step_list=_DENOISING_STEPS,
    warp_denoising_step=True,
    num_frame_per_block=4,
    # Enable PRoPE KV-cache in the DMD self-rollout. NOTE: this differs from
    # origin/dev-meta's legacy SelfForcingPipeline, which never allocated
    # prope_kv_cache (the PRoPE residual was gated off during self-rollout).
    # The new default is False to match dev-meta; this config explicitly opts in.
    self_rollout_use_prope_cache=True,
    dfake_gen_update_ratio=5,
    guidance_scale=3.0,
    use_rollout_min_timestep=False,
    use_rollout_max_timestep=False,
    score_timestep_shift=True,
    score_timestep_clip=(20.0, 980.0),
    score_timestep_discrete=True,
    cfg_base="cond",
    generator_ema_decay=0.99,
    generator_ema_start_step=200,
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
    score_adapter=dict(type="Wan21Adapter", causal=False, dtype="bfloat16"),
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
