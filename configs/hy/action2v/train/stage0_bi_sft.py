"""Minimal example config: HY Action2V autoregressive SFT.

Mirrors ``configs/wan21/action2v/train/stage0_bi_sft.py`` but targets the HunyuanVideo 1.5
autoregressive transformer (ProPE camera conditioning enabled).

Run:
    python tools/train_mwm.py --config-file configs/hy/action2v/train/stage0_bi_sft.py \\
        --output-dir outputs/debug training.max_steps=100
"""

_base_ = "../../../_base_/default.py"

# Defaults match the README download / data-prep targets; symlink any of these
# into ``./ckpts`` / ``./dataset`` if your copies live elsewhere
# (e.g. ``ln -s /abs/path/to/train_index.json ./dataset/HY15/Action2V/train_index.json``).
_PRETRAINED = "./ckpts/HunyuanVideo-1.5/transformer/480p_i2v"
_INDEX = "./dataset/HY15/Action2V/train_index.json"
_NEG_PROMPT = "./dataset/others/HY/Action2V/hunyuan_neg_prompt.pt"
_NEG_BYT5 = "./dataset/others/HY/Action2V/hunyuan_neg_byt5_prompt.pt"

training = dict(
    fsdp_param_dtype="bfloat16",
    activation_checkpointing=True,
    max_steps=100_000,
    log_interval=10,
    ckpt_interval=1_000,
    output_dir="outputs/hy-action2v-stage0_bi_sft",
    # Global RNG seed (seeded seed+dp_rank at dist-init): SP-synced, DP-diverse
    # noise/timestep off the global stream.
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
    type="BiSFTRecipe",
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
    # Muon (>=2D params) + AdamW backup (1D/scalar).
    optimizer=dict(
        name="Muon",
        lr=2e-5,
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
