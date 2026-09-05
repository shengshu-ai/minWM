"""HY Action2V Stage 1 — causal teacher-forcing AR diffusion on real data.

The real-data counterpart of ``stage1_mock.py`` and the teacher-forcing sibling
of ``sft.py``. Same 480p_i2v transformer, ``ARTFRecipe``, Muon optimizer, PRoPE
camera conditioning and ``CameraPluckerDataset`` as SFT; the only additions that
turn the bidirectional SFT step into causal AR teacher forcing are:

- a :class:`~minwm.processors.flow.CleanContextNoiseAug` with
  ``max_timestep=0`` appended to the preprocessor chain. It writes ``clean`` (the
  un-noised clean context) and ``aug_t=None``, which ``HYAdapter.denoise`` passes
  to ``forward_bi`` as ``clean_x`` / ``aug_timesteps``. ``clean_x is not None``
  switches the model onto the teacher-forcing path (clean+noisy concat, the
  block-causal ``flex_tf`` mask). This mirrors the refactor's
  ``ar_camera_training_entry.py`` with ``causal=True`` +
  ``noise_augmentation_max_timestep=0`` (clean = latents, aug_t = zeros).

This config mirrors the refactor stage-1 run
(``scripts/training/hy15_camera/run_ar_hunyuan_mem_multinode.sh``):
``train_time_shift=3.0``, ``window_frames=20``, ``logit_normal`` sampling,
``training.seed=42``, Muon ``lr=1e-5`` (lower than SFT's 2e-5), grad clip 1.0 /
skip>=10.

Note the timestep is sampled **uniform across frames** (``uniform_across_frames=
True``) — the refactor's HY path repeats one sampled timestep over all T frames
(``indices.repeat_interleave(latent_t)``), unlike Wan's per-frame diffusion
forcing. ``stage1_mock.py`` uses ``uniform_across_frames=False`` for data-flow
smoke testing only; the real run matches the refactor here.

Discrete-action conditioning (``use_discrete_action=True`` in the refactor) is
not wired yet — the i2v base ckpt has no ``action_in`` and minWM does not build a
zero-init one, so ``action`` is dropped. The zero-init action embedder is a no-op
at init, so this does not change the forward output; it is a follow-up.

Run (8 GPUs, FSDP):
    torchrun --nproc_per_node=8 tools/train_mwm.py \\
        --config-file configs/hy/action2v/train/stage1_ar_tf.py \\
        --output-dir outputs/hy-action2v-stage1_ar_tf training.max_steps=100
"""

_base_ = "../../../_base_/default.py"

# Relative to the project root (where torchrun is launched), resolved via symlinks
# laid out per configs/hy/action2v/README.md.
_PRETRAINED = "./ckpts/HunyuanVideo-1.5/transformer/480p_i2v"
_INDEX = "./dataset/HY15/Action2V/train_index.json"
_NEG_PROMPT = "./dataset/others/HY/Action2V/hunyuan_neg_prompt.pt"
_NEG_BYT5 = "./dataset/others/HY/Action2V/hunyuan_neg_byt5_prompt.pt"

training = dict(
    fsdp_param_dtype="bfloat16",
    activation_checkpointing=True,
    max_steps=200_000,
    log_interval=10,
    ckpt_interval=1_000,
    output_dir="outputs/hy-action2v-stage1_ar_tf",
    seed=42,
)

model = dict(
    type="ARHunyuanVideo_1_5_DiffusionTransformer",
    _from_pretrained=_PRETRAINED,
    use_prope=True,
    low_cpu_mem_usage=False,
    torch_dtype="bfloat16",  # fp32 weights are ~64 GB/rank, OOM before FSDP shards
)

recipe = dict(
    type="ARTFRecipe",
    # HY flow: linear σ table, shift on the sampled index (not Wan's σ-table warp).
    schedule="linear",
    timestep_shift=3.0,
    # Clip the global gradient norm to 1.0; step skipping is disabled by default.
    max_grad_norm=1.0,
    adapter=dict(
        type="HYAdapter",
        task_type="i2v",
        text_len=1000,
        text_dim=3584,
        dtype="bfloat16",
    ),
    # Muon (>=2D params) + AdamW backup (1D/scalar). Stage 1 uses lr=1e-5
    # (SFT used 2e-5) per the refactor stage-1 run.
    optimizer=dict(
        name="Muon",
        lr=1e-5,
        weight_decay=1e-4,
        momentum=0.95,
        adamw_betas=(0.9, 0.999),
    ),
    loss=dict(type="FlowMatchingLoss"),
    batch_preprocessors=[
        dict(type="LatentToDevice"),
        dict(
            type="FlowNoise",
            uniform_across_frames=True,
            weighting_scheme="logit_normal",
            logit_mean=0.0,
            logit_std=1.0,
            # unweighted (masked) mean, not Wan's bell weight.
            apply_weight=False,
        ),
        # Teacher forcing: clean context, un-noised (max_timestep=0 -> aug_t=None).
        dict(type="CleanContextNoiseAug", max_timestep=0),
    ],
)

data = dict(
    dataset=dict(
        type="CameraPluckerDataset",
        json_path=_INDEX,
        window_frames=20,
        cfg_rate=0.0,
        task_type="i2v",
        neg_prompt_path=_NEG_PROMPT,
        neg_byt5_path=_NEG_BYT5,
    ),
    collate_fn=dict(type="HYCollator"),
    batch_size=1,
    num_workers=2,
    shuffle=True,
    drop_last=True,
)
