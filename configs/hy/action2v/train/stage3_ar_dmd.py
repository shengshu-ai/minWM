"""HY Action2V Stage 3 — causal Distribution Matching Distillation (DMD).

Phase-2 final stage: distill the causal few-step generator via self-forcing DMD
with the cm solver. No real video supervision — the generator self-rolls a fake
video from pure noise (AR cm rollout, truncated BPTT at one step ``k_i``), then
its output distribution is matched to a frozen real-score teacher's CFG-guided
distribution while a trainable fake-score critic learns the generator's output.

Three model copies (all the 480p_i2v base on the first step — a valid parity
baseline since one shared model plays all three roles at step 1; the refactor's
production run instead loads a Phase-1 bidirectional teacher into real/fake score):

- generator (trainable, AR/causal) -> self-forced fake video;
- real_score (frozen, bidirectional) -> CFG-guided real distribution;
- fake_score (trainable critic, bidirectional) -> learns the generator's dist.

Mirrors the refactor DMD run
(``scripts/training/hy15_camera/run_ar_hunyuan_dmd.sh`` +
``trainer/pipelines_camera/ar_hunyuan_dmd_distill_entry.py``): ``solver=cm``,
``dmd_denoising_steps=[1000,750,500,250]``, ``num_frame_per_block=4``, CFG=5.0
(negative-prompt uncond), Muon
(generator ``lr=1e-5`` / critic ``lr=8e-6``, ``weight_decay=0.01``), grad clip 1.0,
``dfake_gen_update_ratio=5``, ``seed=1000``.

Discrete-action conditioning (``use_discrete_action=True`` in the refactor) is
dropped here (the i2v base ckpt has no ``action_in``; zero-init is a no-op at
init — same follow-up as ``stage1.py`` / ``stage2b.py``).

Run (8 GPUs, FSDP):
    torchrun --nproc_per_node=8 tools/train_mwm.py \\
        --config-file configs/hy/action2v/train/stage3_ar_dmd.py \\
        --output-dir outputs/hy-action2v-stage3_ar_dmd training.max_steps=100
"""

_base_ = "../../../_base_/default.py"

_PRETRAINED = "./ckpts/HunyuanVideo-1.5/transformer/480p_i2v"
_INDEX = "./dataset/HY15/Action2V/train_index.json"
_NEG_PROMPT = "./dataset/others/HY/Action2V/hunyuan_neg_prompt.pt"
_NEG_BYT5 = "./dataset/others/HY/Action2V/hunyuan_neg_byt5_prompt.pt"

training = dict(
    fsdp_param_dtype="bfloat16",
    # generator + real_score + fake_score 8.55B copies — recompute block
    # activations in backward to fit the rollout forward+backward in memory.
    activation_checkpointing=True,
    max_steps=200_000,
    log_interval=10,
    ckpt_interval=500,
    output_dir="outputs/hy-action2v-stage3_ar_dmd",
    # 1000 matches the refactor DMD run's --seed; the trainer seeds the tracked
    # noise stream with training.seed + dp_rank.
    seed=1000,
)

_model = dict(
    type="ARHunyuanVideo_1_5_DiffusionTransformer",
    _from_pretrained=_PRETRAINED,
    use_prope=True,
    low_cpu_mem_usage=False,
    torch_dtype="bfloat16",
)

model = dict(**_model)

# real_score frozen; fake_score the trainable critic.
# All FSDP-sharded alongside the generator (shard=True) so DTensor layouts match.
auxiliary_models = dict(
    real_score=dict(trainable=False, shard=True, **_model),
    fake_score=dict(trainable=True, shard=True, **_model),
)

recipe = dict(
    type="ARDMDRecipe",
    critic_schedule="alternating",
    denoising_step_list=[1000, 750, 500, 250],
    warp_denoising_step=True,
    num_frame_per_block=4,
    dfake_gen_update_ratio=5,
    guidance_scale=5.0,
    # min/max timestep ratios 0.0/1.0 -> clamp [0, 1000].
    min_timestep=0,
    max_timestep=1000,
    timestep_shift=5.0,
    num_train_timesteps=1000,
    # Match the score nets' training schedule (the legacy HY discrete scheduler
    # ran to sigma≈0.001, i.e. effectively sigma_min=0); the unified
    # FlowMatchingScheduler must set 0.0/True to reproduce that band, else it falls
    # back to the 0.003/1.002 defaults and the renoise noise level diverges from
    # what real_score/fake_score learned.
    sigma_min=0.0,
    extra_one_step=True,
    normalize_mode="global",
    context_noise=0,
    # Grad clip 1.0, always steps (refactor clips at 1.0; no skip rule here).
    max_grad_norm=1.0,
    adapter=dict(
        type="HYAdapter",
        task_type="i2v",
        text_len=1000,
        text_dim=3584,
        dtype="bfloat16",
        neg_prompt_path=_NEG_PROMPT,
        neg_byt5_path=_NEG_BYT5,
    ),
    # Generator Muon: lr=1e-5, weight_decay=0.01, betas=(0.0, 0.999).
    optimizer=dict(
        name="Muon",
        lr=1e-5,
        weight_decay=0.01,
        momentum=0.95,
        adamw_betas=(0.0, 0.999),
    ),
    # Critic (fake_score) Muon: lr=8e-6, weight_decay=0.01, betas=(0.0, 0.999).
    critic_optimizer=dict(
        name="Muon",
        lr=8e-6,
        weight_decay=0.01,
        momentum=0.95,
        adamw_betas=(0.0, 0.999),
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
