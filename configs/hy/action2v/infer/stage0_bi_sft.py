"""HY Action2V bidirectional i2v inference (50-step CFG).

Runs the Phase-1 bidirectional SFT checkpoint (before distillation). Image +
caption + camera trajectory in, video out. Self-contained: the bidirectional
loop runs the full clip in one shot, so this config carries its own
model / encoder / VAE / adapter defs rather than sharing the causal
``model_ar.py`` base.

Bidirectional inference reuses the same ARHunyuanVideo transformer class as the
AR stages: HYAdapter.denoise always passes viewmats/Ks/action, which only
``causal.py``'s ``forward_bi`` accepts.

Quick start (single GPU)::

    python tools/infer_mwm.py --config-file configs/hy/action2v/infer/stage0_bi_sft.py
"""

from minwm.modeling.hy15.encoders import PROMPT_TEMPLATE

_CKPT = "./ckpts/HunyuanVideo-1.5"
_NEG_PROMPT = "./dataset/others/HY/Action2V/hunyuan_neg_prompt.pt"
_NEG_BYT5 = "./dataset/others/HY/Action2V/hunyuan_neg_byt5_prompt.pt"

model = dict(
    type="ARHunyuanVideo_1_5_DiffusionTransformer",
    _from_pretrained=f"{_CKPT}/transformer/480p_i2v",
    use_prope=True,
    torch_dtype="bfloat16",
    low_cpu_mem_usage=False,
)

text_encoder = dict(
    type="TextEncoder",
    text_encoder_type="llm",
    tokenizer_type="llm",
    text_encoder_path=f"{_CKPT}/text_encoder/llm",
    max_length=1000,
    text_encoder_precision="fp16",
    prompt_template=PROMPT_TEMPLATE["li-dit-encode-video-json"],
    prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
    hidden_state_skip_layer=2,
    apply_final_norm=False,
)

vision_encoder = dict(
    type="VisionEncoder",
    vision_encoder_type="siglip",
    vision_encoder_precision="fp16",
    vision_encoder_path=f"{_CKPT}/vision_encoder/siglip",
)

vae = dict(
    type="AutoencoderKLConv3D",
    _from_pretrained=f"{_CKPT}/vae",
    torch_dtype="bfloat16",
)

adapter = dict(
    type="HYAdapter",
    task_type="i2v",
    text_len=1000,
    text_dim=3584,
    dtype="bfloat16",
    neg_prompt_path=_NEG_PROMPT,
    neg_byt5_path=_NEG_BYT5,
)

inference = dict(
    loop="BidirectionalGenerationLoop",
    benchmark="assets/example.json",
    checkpoint="./ckpts/HY15/Action2V/stage0_bi_sft",
    output_dir="outputs/hy_action2v_stage0_bi_sft",
    guidance_scale=6.0,
    dtype="bfloat16",
    seed=42,
    sp_size=1,
    # HY's VAE 3D-conv decode OOMs when tiling is on for the full clip; the AR
    # stages disable it too. Without this the bidirectional decode allocates ~32
    # GiB in one shot and dies on an 80 GiB card.
    vae_tiling=False,
    sampler=dict(solver="UniPCSolver", num_inference_steps=50, shift=5.0),
    num_frames=20,
    latent_shape=(32, 30, 52),
    fps=16,
    input_preprocessors=[
        dict(type="HYConditioning"),
        dict(type="LatentNoise"),
        dict(type="CameraTrajectory"),
    ],
)
