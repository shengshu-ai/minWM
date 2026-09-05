"""HY Action2V causal-consistency few-step AR-rollout inference (4-step, CFG off)."""

_base_ = "model_ar.py"

inference = dict(
    loop="ARGenerationLoop",
    benchmark="assets/example.json",
    checkpoint="./ckpts/HY15/Action2V/stage2_ar_cd",
    output_dir="outputs/hy_action2v_stage2_ar_cd",
    guidance_scale=1.0,
    dtype="bfloat16",
    seed=42,
    sp_size=1,
    vae_tiling=False,
    sampler=dict(solver="EulerSolver", num_inference_steps=4, shift=5.0),
    context_noise=0,
    num_frames=20,
    latent_shape=(32, 30, 52),
    fps=16,
    input_preprocessors=[
        dict(type="HYConditioning"),
        dict(type="LatentNoise"),
        dict(type="CameraTrajectory"),
    ],
)
