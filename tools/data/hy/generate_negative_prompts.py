"""Generate negative-prompt embeddings for HunyuanVideo classifier-free guidance.

Writes the empty-prompt LLM + ByT5 embeddings the training/dataloader pipeline
loads as the CFG negative prompt.

Usage:
    python tools/data/hy/generate_negative_prompts.py \
        --hunyuan_checkpoint_path /path/to/HunyuanVideo-1.5 \
        --output_dir ./dataset/others/HY/TI2V
"""

import argparse
import os

import torch

from minwm.modeling.hy15.encoders import PROMPT_TEMPLATE, TextEncoder, load_glyph_byT5_v2

_BYT5_EMBED_DIM = 1472


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hunyuan_checkpoint_path", type=str, required=True, help="Path to HunyuanVideo checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory (same as preencoded data)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading text encoders...")

    text_encoder = TextEncoder(
        text_encoder_type="llm",
        tokenizer_type="llm",
        text_encoder_path=os.path.join(args.hunyuan_checkpoint_path, "text_encoder/llm"),
        max_length=1000,
        text_encoder_precision="fp16",
        prompt_template=PROMPT_TEMPLATE["li-dit-encode-video-json"],
        prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
        hidden_state_skip_layer=2,
        apply_final_norm=False,
        reproduce=False,
        logger=None,
        device=device,
    )

    load_from = os.path.join(args.hunyuan_checkpoint_path, "text_encoder")
    glyph_root = os.path.join(load_from, "Glyph-SDXL-v2")
    byt5_args = dict(
        byT5_google_path=os.path.join(load_from, "byt5-small"),
        byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
        multilingual_prompt_format_color_path=os.path.join(glyph_root, "assets/color_idx.json"),
        multilingual_prompt_format_font_path=os.path.join(
            glyph_root, "assets/multilingual_10-lang_idx.json"
        ),
        byt5_max_length=256,
    )
    byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=str(device))
    byt5_max_length = byt5_kwargs["byt5_max_length"]

    print("Generating negative prompts...")

    negative_prompt = ""
    text_inputs = text_encoder.text2tokens(negative_prompt, data_type="video", max_length=1000)
    prompt_outputs = text_encoder.encode(text_inputs, data_type="video", device=device)
    negative_prompt_embeds = prompt_outputs.hidden_state.to(dtype=text_encoder.dtype, device=device)
    negative_prompt_mask = (
        prompt_outputs.attention_mask.to(device)
        if prompt_outputs.attention_mask is not None
        else None
    )

    byt5_embeddings = torch.zeros((1, byt5_max_length, _BYT5_EMBED_DIM), device=device)
    byt5_mask = torch.zeros((1, byt5_max_length), device=device, dtype=torch.int64)

    os.makedirs(args.output_dir, exist_ok=True)

    neg_prompt_path = os.path.join(args.output_dir, "hunyuan_neg_prompt.pt")
    torch.save(
        {
            "negative_prompt_embeds": negative_prompt_embeds.cpu(),
            "negative_prompt_mask": negative_prompt_mask.cpu(),
        },
        neg_prompt_path,
    )
    print(f"Saved: {neg_prompt_path}")

    neg_byt5_path = os.path.join(args.output_dir, "hunyuan_neg_byt5_prompt.pt")
    torch.save(
        {
            "byt5_text_states": byt5_embeddings.cpu(),
            "byt5_text_mask": byt5_mask.cpu(),
        },
        neg_byt5_path,
    )
    print(f"Saved: {neg_byt5_path}")

    neg_prompt_compat_path = os.path.join(args.output_dir, "negative_prompt.pt")
    torch.save(
        {
            "negative_prompt_embeds": negative_prompt_embeds.cpu(),
            "negative_prompt_mask": negative_prompt_mask.cpu(),
        },
        neg_prompt_compat_path,
    )
    print(f"Saved: {neg_prompt_compat_path}")

    print("\nNegative prompt files generated successfully!")


if __name__ == "__main__":
    main()
