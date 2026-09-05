"""Stage 1 HY Action2V AR diffusion (teacher forcing) — mock dry-run config.

Tiny ARHunyuanVideo_1_5 transformer + MockCameraLatentDataset (camera ON) for
GPU data-flow smoke testing. ProPE camera conditioning is enabled
(``use_prope=True``) and the dataset emits viewmats/Ks (``with_camera=True``).
Per-frame independent timesteps plus teacher-forced clean context
(``CleanContextNoiseAug``). Text is mocked as zeros.

``batch_size=1``: the HY teacher-forcing path varlen-packs text into a single
batch-1 sequence, so it is batch-1 by design (B>=2 fails on GPU too). The step
runs on CUDA only and in ``bfloat16``: the text token-refiner uses flash-attn
(no fp32 kernel), so the recipe autocasts the forward to the adapter's working
dtype. Under FSDP the trainer overrides that dtype to ``fsdp_param_dtype``.

Run:
    python tools/train_mwm.py \
        --config-file configs/hy/action2v/train/stage1_ar_tf_mock.py \
        --output-dir outputs/smoke_hy_stage1_ar_tf
"""

_base_ = ["../../../_base_/default.py", "../../_model_mock.py"]

# patch_size [1,2,2]; latent dims (num_frames, channels, height, width)
_F, _C, _H, _W = 4, 16, 8, 16

training = dict(
    max_steps=3,
    log_interval=1,
    ckpt_interval=0,
    output_dir="outputs/smoke_hy_stage1_ar_tf",
)

recipe = dict(
    type="ARTFRecipe",
    adapter=dict(
        type="HYAdapter",
        text_len=8,
        text_dim=64,
        dtype="bfloat16",
    ),
    optimizer=dict(lr=1e-4),
    loss=dict(type="FlowMatchingLoss"),
    batch_preprocessors=[
        dict(type="LatentToDevice"),
        dict(type="FlowNoise", uniform_across_frames=False),
        dict(type="CleanContextNoiseAug", max_timestep=0),
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
    ),
    batch_size=1,
    num_workers=0,
    shuffle=False,
    drop_last=True,
)
