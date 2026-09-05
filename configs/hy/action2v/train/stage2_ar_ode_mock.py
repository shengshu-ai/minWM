"""Stage 2a HY causal ODE regression — mock dry-run config.

Tiny ARHunyuanVideo_1_5 transformer + MockCameraLatentDataset (ODE trajectory
mode, camera on) for GPU data-flow smoke testing. Text is mocked as zeros.

``batch_size=1``: the HY teacher-forcing path varlen-packs text into a single
batch-1 sequence, so it is batch-1 by design (B>=2 fails on GPU too). The
forward+backward step uses FlexAttention and only runs on CUDA.

Run:
    python tools/train_mwm.py \
        --config-file configs/hy/action2v/train/stage2_ar_ode_mock.py \
        --output-dir outputs/smoke_hy_stage2_ar_ode
"""

_base_ = ["../../../_base_/default.py", "../../_model_mock.py"]

# patch_size [1,2,2]; latent dims (num_frames, channels, height, width)
_F, _C, _H, _W = 4, 16, 8, 16
# ode trajectory length = len(denoising_step_list) + 1 (last entry = clean)
_DENOISING_STEPS = [1000, 750, 500, 250]
_ODE_STEPS = len(_DENOISING_STEPS) + 1

training = dict(
    max_steps=3,
    log_interval=1,
    ckpt_interval=0,
    output_dir="outputs/smoke_hy_stage2_ar_ode",
)

recipe = dict(
    type="ARODERecipe",
    adapter=dict(
        type="HYAdapter",
        text_len=8,
        text_dim=64,
        dtype="float32",
    ),
    optimizer=dict(lr=1e-4),
    loss=dict(type="ODERegressionLoss"),
    batch_preprocessors=[
        dict(type="LatentToDevice"),
        dict(
            type="ODETrajectorySample",
            denoising_step_list=_DENOISING_STEPS,
        ),
    ],
)

data = dict(
    dataset=dict(
        type="MockCameraLatentDataset",
        length=8,
        num_frames=_F,
        channels=_C,
        height=_H,
        width=_W,
        with_camera=True,
        ode_steps=_ODE_STEPS,
    ),
    batch_size=1,
    num_workers=0,
    shuffle=False,
    drop_last=True,
)
