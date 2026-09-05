"""HY Action2V teacher-forcing AR-diffusion inference (50-step, CFG 6.0).

The undistilled causal AR model: same chunk-by-chunk KV-cached rollout as the
few-step configs, but run for the full diffusion step count with CFG — the AR
analogue of the bidirectional SFT baseline.
"""

_base_ = "model_ar.py"

inference = dict(
    loop="ARGenerationLoop",
    benchmark="assets/example.json",
    checkpoint="./ckpts/HY15/Action2V/stage1_ar_tf",
    output_dir="outputs/hy_action2v_stage1_ar_tf",
    guidance_scale=6.0,
    dtype="bfloat16",
    seed=42,
    sp_size=1,
    vae_tiling=False,
    sampler=dict(solver="EulerSolver", num_inference_steps=50, shift=5.0),
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
