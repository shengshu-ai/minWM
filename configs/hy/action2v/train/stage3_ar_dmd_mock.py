"""Stage 3 HY Action2V causal DMD distillation — mock dry-run config.

Tiny generator + real_score + fake_score ARHunyuanVideo_1_5 transformers with
MockCameraLatentDataset (camera ON) for GPU data-flow smoke testing.
real_score/fake_score are built by BaseTrainer from ``auxiliary_models``
(per-entry ``trainable`` flag); real_score is frozen, fake_score is the
trainable critic. ProPE camera conditioning is enabled
(``use_prope=True``) and the dataset emits viewmats/Ks (``with_camera=True``).
No real Stage 1/2 weights are needed for a smoke test. Text is mocked as zeros.

``batch_size=1``: the HY teacher-forcing path varlen-packs text into a single
batch-1 sequence, so it is batch-1 by design. The forward+backward step uses
FlexAttention and only runs on CUDA.

Run:
    python tools/train_mwm.py \
        --config-file configs/hy/action2v/train/stage3_ar_dmd_mock.py \
        --output-dir outputs/smoke_hy_stage3_ar_dmd
"""

_base_ = "../../../_base_/default.py"

# patch_size [1,2,2]; latent dims (num_frames, channels, height, width)
_F, _C, _H, _W = 4, 16, 8, 16

_model = dict(
    type="ARHunyuanVideo_1_5_DiffusionTransformer",
    patch_size=[1, 2, 2],
    in_channels=_C,
    concat_condition=False,
    hidden_size=64,
    heads_num=4,
    mm_double_blocks_depth=2,
    mm_single_blocks_depth=0,
    rope_dim_list=[4, 6, 6],
    text_states_dim=64,
    text_states_dim_2=64,
    use_prope=True,
)

training = dict(
    max_steps=4,
    log_interval=1,
    ckpt_interval=0,
    output_dir="outputs/smoke_hy_stage3_ar_dmd",
)

model = dict(**_model)

auxiliary_models = dict(
    real_score=dict(trainable=False, **_model),
    fake_score=dict(trainable=True, **_model),
)

recipe = dict(
    type="ARDMDRecipe",
    adapter=dict(
        type="HYAdapter",
        text_len=8,
        text_dim=64,
        dtype="float32",
    ),
    optimizer=dict(lr=1e-4),
    critic_optimizer=dict(lr=1e-4),
    dfake_gen_update_ratio=2,
    guidance_scale=5.0,
    denoising_step_list=[750, 500, 250],
    num_frame_per_block=_F,  # single block (must divide _F)
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
