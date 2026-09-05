"""HY Action2V inference base: shared model / encoders / VAE / adapter.

Component definitions shared by every stage config in this directory. The same
``ARHunyuanVideo`` transformer and encoders drive both the causal AR-rollout
stages (``stage1_ar_tf`` / ``stage2_ar_cd`` / ``stage2_ar_ode`` /
``stage3_ar_dmd``) and the bidirectional ``stage0_bi_sft`` — they differ only in
the finetuned checkpoint and the ``inference`` block each stage inlines. Stage
configs override ``model._from_pretrained`` with their own checkpoint dir.
"""

from minwm.modeling.hy15.encoders import PROMPT_TEMPLATE

_CKPT = "./ckpts/HunyuanVideo-1.5"
_NEG_PROMPT = "./dataset/others/HY/Action2V/hunyuan_neg_prompt.pt"
_NEG_BYT5 = "./dataset/others/HY/Action2V/hunyuan_neg_byt5_prompt.pt"

model = dict(
    type="ARHunyuanVideo_1_5_DiffusionTransformer",
    _from_pretrained=f"{_CKPT}/transformer/480p_i2v",
    use_prope=True,
    num_frame_per_block=4,
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
