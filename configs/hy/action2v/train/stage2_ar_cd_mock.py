"""Stage 2b HY causal consistency distillation — mock dry-run config.

Tiny ARHunyuanVideo_1_5 transformer + MockCameraLatentDataset (single clean
latent, camera on) for GPU data-flow smoke testing. Teacher + EMA are lazily
deep-copied from the student on the first step (no real Stage 1/2a weights
needed). Text is mocked as zeros.

``batch_size=1``: the HY teacher-forcing path varlen-packs text into a single
batch-1 sequence, so it is batch-1 by design (B>=2 fails on GPU too). The
forward+backward step uses FlexAttention and only runs on CUDA.

Run:
    python tools/train_mwm.py \
        --config-file configs/hy/action2v/train/stage2_ar_cd_mock.py \
        --output-dir outputs/smoke_hy_stage2_ar_cd
"""

_base_ = ["../../../_base_/default.py", "../../_model_mock.py"]

# patch_size [1,2,2]; latent dims (num_frames, channels, height, width)
_F, _C, _H, _W = 4, 16, 8, 16

training = dict(
    max_steps=3,
    log_interval=1,
    ckpt_interval=0,
    output_dir="outputs/smoke_hy_stage2_ar_cd",
)

recipe = dict(
    type="ARCDRecipe",
    adapter=dict(
        type="HYAdapter",
        text_len=8,
        text_dim=64,
        dtype="float32",
    ),
    optimizer=dict(lr=1e-4),
    discrete_cd_n=8,
    guidance_scale=5.0,
    ema_decay=0.95,
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
        ode_steps=0,
    ),
    batch_size=1,
    num_workers=0,
    shuffle=False,
    drop_last=True,
)
