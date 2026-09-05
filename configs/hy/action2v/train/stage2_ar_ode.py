"""HY Action2V Stage 2a — causal ODE distillation initialization on real data.

The real-data counterpart of ``stage2a_mock.py``: instead of the tiny model +
``MockCameraLatentDataset``, this trains the full ``ARHunyuanVideo_1_5`` causal
transformer against the pre-solved ODE trajectories in
``./dataset/HY15/Action2V_ode/`` (the HY sibling of Wan's ``ode_distill.py``).

Differences from ``stage1.py`` (teacher-forcing AR diffusion):

- ``ODERegressionLoss`` (regress ``x0`` against the trajectory's near-clean
  target) instead of ``FlowMatchingLoss``;
- ``CausalODEDataset`` emits a pre-solved ``ode_trajectory`` ``[N, C, F, H, W]``
  (N=6: 4 denoising points + 48-step target + clean) which ``HYCollator``
  renames+transposes to ``ode_latent`` ``[B, N, F, C, H, W]``;
- a single ``ODETrajectorySample`` preprocessor samples one of the 4 denoising
  points and labels it with the WARPED timestep (``warp_denoising_step=True``).
  The trajectory was solved on the shifted flow schedule (``ODE_SHIFT=5.0`` in
  ``tools/data/hy/get_causal_ode_data_prope.py``), so the scheduler must be
  ``schedule="shifted"`` with ``timestep_shift=5.0`` / ``sigma_min=0.0`` /
  ``extra_one_step=True`` — this reproduces that exact σ-table, so the warp
  recovers the σ each snapshot was solved at (``linear`` leaves the shift
  un-baked and mislabels 3 of the 4 steps). This matches Wan's ``ode_distill.py``.

Weights initialize from the promoted Stage-1 (AR teacher-forcing) checkpoint
``./ckpts/HY15/Action2V/stage1_ar_tf`` — a diffusers directory of the same
``ARHunyuanVideo_1_5_DiffusionTransformer`` class, so ``_from_pretrained`` loads
it directly (no separate ``checkpoint.pretrained`` needed).

Run (8 GPUs, FSDP):
    torchrun --nproc_per_node=8 tools/train_mwm.py \\
        --config-file configs/hy/action2v/train/stage2_ar_ode.py \\
        --output-dir outputs/hy-action2v-stage2_ar_ode training.max_steps=100
"""

_base_ = "../../../_base_/default.py"

# Relative to the project root (where torchrun is launched), resolved via symlinks
# laid out per configs/hy/action2v/README.md.
_PRETRAINED = "./ckpts/HY15/Action2V/stage1_ar_tf"
_INDEX = "./dataset/HY15/Action2V_ode/train_index.json"
_NEG_PROMPT = "./dataset/others/HY/Action2V/hunyuan_neg_prompt.pt"
_NEG_BYT5 = "./dataset/others/HY/Action2V/hunyuan_neg_byt5_prompt.pt"

# Raw ODE denoising steps (most-noisy -> least-noisy). Warped to schedule
# timesteps by ODETrajectorySample; their count (4) bounds the sampled step idx,
# matching the trajectory's 6 snapshots (4 noisy + 48-step target + clean).
_DENOISING_STEPS = [1000, 750, 500, 250]

training = dict(
    fsdp_param_dtype="bfloat16",
    activation_checkpointing=True,
    max_steps=10_000,
    log_interval=10,
    ckpt_interval=1_000,
    output_dir="outputs/hy-action2v-stage2_ar_ode",
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
    type="ARODERecipe",
    # The warp maps each raw step back to the σ the trajectory was actually
    # solved at, so scheduler.timesteps MUST carry the ODE solver's shift baked
    # in. Only schedule="shifted" bakes it (linear applies shift at sampling,
    # leaving timesteps unshifted → the warp would recover the wrong σ for 3/4
    # steps). shifted + sigma_min=0.0 + extra_one_step=True reproduces exactly
    # build_ode_sigmas(shift=5.0) from tools/data/hy/get_causal_ode_data_prope.py
    # (verified bit-identical), matching Wan's ode_distill.py.
    schedule="shifted",
    timestep_shift=5.0,
    sigma_min=0.0,
    extra_one_step=True,
    max_grad_norm=1.0,
    adapter=dict(
        type="HYAdapter",
        task_type="i2v",
        text_len=1000,
        text_dim=3584,
        dtype="bfloat16",
    ),
    optimizer=dict(
        name="Muon",
        lr=1e-5,
        weight_decay=1e-4,
        momentum=0.95,
        adamw_betas=(0.9, 0.999),
    ),
    loss=dict(type="ODERegressionLoss"),
    batch_preprocessors=[
        dict(type="LatentToDevice"),
        # ODE regression: sample one denoising point, label with the warped
        # schedule timestep (the recipe injects its scheduler for the warp).
        dict(
            type="ODETrajectorySample",
            denoising_step_list=_DENOISING_STEPS,
            warp_denoising_step=True,
        ),
    ],
)

data = dict(
    dataset=dict(
        type="CausalODEDataset",
        json_path=_INDEX,
        window_frames=20,
        task_type="i2v",
        neg_prompt_path=_NEG_PROMPT,
        neg_byt5_path=_NEG_BYT5,
    ),
    collate_fn=dict(type="HYCollator"),
    batch_size=1,  # HY teacher-forcing varlen-packs text into a batch-1 sequence
    num_workers=2,
    shuffle=True,
    drop_last=True,
)
