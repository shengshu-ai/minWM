"""Wan Action2V causal consistency inference config."""

_base_ = "model_ar.py"

inference = dict(
    loop="ARGenerationLoop",
    benchmark="assets/example_t2v.json",
    checkpoint="./ckpts/Wan21/Action2V/stage2_ar_cd/model.pt",
    output_dir="./outputs/wan_action2v_stage2_ar_cd",
    guidance_scale=1.0,
    prefer_ema=True,
    dtype="bfloat16",
    seed=0,
    sp_size=1,
    sampler=dict(
        solver="CMSolver",
        denoising_step_list=[1000, 750, 500, 250],
        shift=5.0,
        sigma_min=0.0,
        num_train_timesteps=1000,
    ),
    context_noise=0,
    num_frames=20,
    latent_shape=(16, 60, 104),
    fps=16,
)
