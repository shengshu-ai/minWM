"""Wan Action2V bidirectional inference config."""

from minwm.modeling.wan21.adapter import DEFAULT_NEGATIVE_PROMPT as _NEGATIVE_PROMPT

_BASE = "./ckpts/Wan2.1-T2V-1.3B"

model = dict(
    type="Wan21Model",
    _from_pretrained=_BASE,
    use_prope=True,
    low_cpu_mem_usage=False,
)

text_encoder = dict(
    type="Wan21TextEncoder",
    checkpoint_path=f"{_BASE}/models_t5_umt5-xxl-enc-bf16.pth",
    tokenizer_path=f"{_BASE}/google/umt5-xxl/",
    text_len=512,
)

vae = dict(
    type="Wan21VAE",
    vae_pth=f"{_BASE}/Wan2.1_VAE.pth",
)

adapter = dict(
    type="Wan21Adapter",
    causal=False,
    text_len=512,
    text_dim=4096,
    dtype="bfloat16",
)

inference = dict(
    loop="BidirectionalGenerationLoop",
    benchmark="assets/example_t2v.json",
    checkpoint="./ckpts/Wan21/Action2V/stage0_bi_sft/model.pt",
    output_dir="./outputs/wan_action2v_stage0_bi_sft",
    guidance_scale=8.0,
    dtype="bfloat16",
    seed=0,
    sp_size=1,
    sampler=dict(solver="UniPCSolver", num_train_timesteps=1000, shift=5.0, num_inference_steps=50),
    negative_prompt=_NEGATIVE_PROMPT,
    num_frames=20,
    latent_shape=(16, 60, 104),
    fps=16,
)
