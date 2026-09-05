"""Wan Action2V teacher-forcing AR-diffusion inference config."""

from minwm.modeling.wan21.adapter import DEFAULT_NEGATIVE_PROMPT as _NEGATIVE_PROMPT

_base_ = "model_ar.py"

inference = dict(
    loop="ARGenerationLoop",
    benchmark="assets/example_t2v.json",
    checkpoint="./ckpts/Wan21/Action2V/stage1_ar_tf/model.pt",
    output_dir="./outputs/wan_action2v_stage1_ar_tf",
    guidance_scale=3.0,
    prefer_ema=True,
    dtype="bfloat16",
    seed=0,
    sp_size=1,
    num_inference_steps=50,
    sampler=dict(
        solver="UniPCSolver",
        num_train_timesteps=1000,
        shift=5.0,
    ),
    negative_prompt=_NEGATIVE_PROMPT,
    context_noise=0,
    num_frames=20,
    latent_shape=(16, 60, 104),
    fps=16,
)
