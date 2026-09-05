"""HY Action2V Stage 2b — causal consistency distillation on real data.

Phase-2 Stage 2(b): distill the causal student into a few-step consistency model.
A frozen teacher does a CFG Euler step ``t -> t_next``; the student predicts ``x0``
at ``t`` and an EMA network predicts ``x0`` at ``t_next``; the loss is the MSE
between the two ``x0`` predictions (consistency).

Three model copies (all seeded from the same 480p_i2v base on the first step,
matching the refactor ``ar_causal_cd_entry`` which loads the same checkpoint into
student + teacher and initialises the EMA from the student):

- student (trainable) -> prediction at ``t``;
- ema (frozen EMA of student) -> target at ``t_next``;
- teacher (frozen) -> CFG Euler step ``t -> t_next``.

Teacher + EMA are trainer-owned ``auxiliary_models`` (frozen, FSDP-sharded
alongside the student); :class:`ARCDRecipe` seeds them from
the student via ``copy_params`` on the first step, so all three start identical.

Same 480p_i2v transformer + PRoPE camera stream + ``CameraPluckerDataset`` as
``stage1.py`` (teacher forcing is on: ``clean`` = the clean latent, ``aug_t`` =
zeros). Differs from Stage 1 in the training signal — consistency distillation,
not flow matching — and in the discrete CD schedule the ``(t, t_next)`` pairs are
drawn from.

Mirrors the refactor CD run
(``scripts/training/hy15_camera/run_ar_causal_cd.sh`` +
``trainer/pipelines_camera/ar_causal_cd_entry.py``): ``cd_num_steps=50``,
``ode_shift=5.0``, ``cfg-scale=5.0``, Muon ``lr=1e-5`` ``weight_decay=1e-4``, grad clip 1.0 /
skip>=10, ``window_frames=20``, ``ema_decay=0.999``.

Discrete-action conditioning (``use_discrete_action=True`` in the refactor) is not
wired yet — the i2v base ckpt has no ``action_in`` and minWM does not build a
zero-init one, so ``action`` is dropped (the zero-init embedder is a no-op at init,
so this does not change the forward output; follow-up, same as ``stage1.py``).

Run (8 GPUs, FSDP):
    torchrun --nproc_per_node=8 tools/train_mwm.py \\
        --config-file configs/hy/action2v/train/stage2_ar_cd.py \\
        --output-dir outputs/hy-action2v-stage2_ar_cd training.max_steps=100
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
    # The recipe holds three 8.55B copies (student + teacher + EMA), so recompute
    # block activations in backward to fit the forward+backward in memory.
    activation_checkpointing=True,
    max_steps=200_000,
    log_interval=10,
    ckpt_interval=500,
    output_dir="outputs/hy-action2v-stage2_ar_cd",
    # Global RNG seed (seeded seed+dp_rank at dist-init) for the dataloader and
    # the CD noise + step-index draws (SP-synced, DP-diverse, like FlowNoise).
    # 3208 matches the refactor CD run's seed.
    seed=3208,
)

# Causal AR transformer from the 480p_i2v base; same knobs as Stage 1.
_model = dict(
    type="ARHunyuanVideo_1_5_DiffusionTransformer",
    _from_pretrained=_PRETRAINED,
    use_prope=True,
    low_cpu_mem_usage=False,
    torch_dtype="bfloat16",  # fp32 weights are ~64 GB/rank, OOM before FSDP shards
)

model = dict(**_model)

# Frozen teacher + EMA, built and FSDP-sharded by the trainer alongside the
# student and seeded from it on the first step (see
# ARCDRecipe._lazy_init_aux). shard=True so they match the
# student's DTensor layout for copy_params / EMA copy.
auxiliary_models = dict(
    teacher=dict(trainable=False, shard=True, **_model),
    ema=dict(trainable=False, shard=True, **_model),
)

recipe = dict(
    type="ARCDRecipe",
    # Discrete CD schedule (FlowMatchingScheduler): sigma_min=0 / extra_one_step
    # reproduce the HY discrete σ table (linspace(1,0,N+1) warped by timestep_shift),
    # the same shifted table ARGenerationLoop walks at inference.
    discrete_cd_n=50,
    timestep_shift=5.0,
    sigma_min=0.0,
    extra_one_step=True,
    guidance_scale=5.0,
    ema_decay=0.999,
    # Clip the global gradient norm to 1.0; step skipping is disabled by default.
    max_grad_norm=1.0,
    adapter=dict(
        type="HYAdapter",
        task_type="i2v",
        text_len=1000,
        text_dim=3584,
        dtype="bfloat16",
        # CFG uncond uses the negative-prompt embeddings (not zeros).
        neg_prompt_path=_NEG_PROMPT,
        neg_byt5_path=_NEG_BYT5,
    ),
    # Muon (>=2D params) + AdamW backup (1D/scalar); lr=1e-5 weight_decay=1e-4
    # per the refactor CD run.
    optimizer=dict(
        name="Muon",
        lr=1e-5,
        weight_decay=1e-4,
        momentum=0.95,
        adamw_betas=(0.9, 0.999),
    ),
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
