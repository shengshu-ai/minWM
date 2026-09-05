"""Wan Action2V causal inference base config."""

_BASE = "./ckpts/Wan2.1-T2V-1.3B"

model = dict(
    type="CausalWan21Model",
    _from_pretrained=_BASE,
    use_prope=True,
    num_frame_per_block=4,
    local_attn_size=20,
    low_cpu_mem_usage=False,
)

text_encoder = dict(
    type="Wan21TextEncoder",
    checkpoint_path=f"{_BASE}/models_t5_umt5-xxl-enc-bf16.pth",
    tokenizer_path=f"{_BASE}/google/umt5-xxl/",
    text_len=512,
)

adapter = dict(
    type="Wan21Adapter",
    causal=True,
    text_len=512,
    text_dim=4096,
    dtype="bfloat16",
)

vae = dict(
    type="Wan21VAE",
    vae_pth=f"{_BASE}/Wan2.1_VAE.pth",
)
