# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan
# works and any output and results therefrom are provided "AS IS" without any
# express or implied warranties of any kind including any warranties of title,
# merchantability, noninfringement, course of dealing, usage of trade, or
# fitness for a particular purpose. You are solely responsible for determining
# the appropriateness of using, reproducing, modifying, performing, displaying
# or distributing any of the Tencent Hunyuan works or outputs and assume any and
# all risks associated with your or a third party's use or distribution of any
# of the Tencent Hunyuan works or outputs and your exercise of rights and
# permissions under this agreement.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ByT5 glyph-embedding projection head and loaders for the HY15 DiT."""

import json

import torch
import torch.nn as nn


def load_glyph_byT5_v2(args, device):
    """Load the ByT5 tokenizer and encoder for glyph encoding.

    Args:
        args (dict): config with ByT5 paths and settings.
        device (str or torch.device): target device.

    Returns:
        dict: ``{"byt5_tokenizer", "byt5_model", "byt5_max_length"}``.
    """
    byt5_tokenizer, byt5_model, byt5_max_length = create_byt5(args, device)
    byt5_model = byt5_model.to(device=device)
    return {
        "byt5_tokenizer": byt5_tokenizer,
        "byt5_model": byt5_model,
        "byt5_max_length": byt5_max_length,
    }


def create_byt5(args, device):
    """Create the ByT5 tokenizer and encoder, loading custom weights if given.

    Args:
        args (dict): config with ByT5 paths and settings.
        device (str or torch.device): target device.

    Returns:
        tuple: ``(byt5_tokenizer, byt5_model, byt5_max_length)``.
    """
    byt5_max_length = args["byt5_max_length"]
    byt5_config = dict(
        byt5_name=args["byT5_google_path"],
        special_token=True,
        color_special_token=True,
        font_special_token=True,
        color_ann_path=args["multilingual_prompt_format_color_path"],
        font_ann_path=args["multilingual_prompt_format_font_path"],
        multilingual=True,
    )
    huggingface_cache_dir = None
    byt5_model, byt5_tokenizer = load_byt5_and_byt5_tokenizer(
        **byt5_config,
        huggingface_cache_dir=huggingface_cache_dir,
        device=device,
    )

    if args["byT5_ckpt_path"] is not None:
        byt5_state_dict = torch.load(args["byT5_ckpt_path"], map_location=device)
        if "state_dict" in byt5_state_dict:
            sd = byt5_state_dict["state_dict"]
            newsd = {}
            for k, v in sd.items():
                if k.startswith("module.text_tower.encoder."):
                    newsd[k[len("module.text_tower.encoder.") :]] = v
            byt5_state_dict = newsd
        byt5_model.load_state_dict(byt5_state_dict)
    byt5_model.requires_grad_(False)
    return byt5_tokenizer, byt5_model, byt5_max_length


def add_special_token(
    tokenizer,
    text_encoder,
    add_color,
    add_font,
    color_ann_path,
    font_ann_path,
    multilingual=False,
):
    """Add color/font special tokens to the tokenizer and encoder.

    Args:
        tokenizer: HuggingFace tokenizer.
        text_encoder: HuggingFace T5 encoder.
        add_color (bool): add color tokens.
        add_font (bool): add font tokens.
        color_ann_path (str): path to the color annotation JSON.
        font_ann_path (str): path to the font annotation JSON.
        multilingual (bool): use multilingual font tokens.
    """
    with open(font_ann_path, "r") as f:
        idx_font_dict = json.load(f)
    with open(color_ann_path, "r") as f:
        idx_color_dict = json.load(f)

    if multilingual:
        font_token = [
            f"<{font_code[:2]}-font-{idx_font_dict[font_code]}>" for font_code in idx_font_dict
        ]
    else:
        font_token = [f"<font-{i}>" for i in range(len(idx_font_dict))]
    color_token = [f"<color-{i}>" for i in range(len(idx_color_dict))]
    additional_special_tokens = []
    if add_color:
        additional_special_tokens += color_token
    if add_font:
        additional_special_tokens += font_token

    tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
    text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)


def load_byt5_and_byt5_tokenizer(
    byt5_name="google/byt5-small",
    special_token=False,
    color_special_token=False,
    font_special_token=False,
    color_ann_path="assets/color_idx.json",
    font_ann_path="assets/font_idx_512.json",
    huggingface_cache_dir=None,
    multilingual=False,
    device=None,
):
    """Load a ByT5 encoder and tokenizer, adding special tokens if requested.

    Args:
        byt5_name (str): model name or path.
        special_token (bool): add special tokens.
        color_special_token (bool): add color tokens.
        font_special_token (bool): add font tokens.
        color_ann_path (str): path to the color annotation JSON.
        font_ann_path (str): path to the font annotation JSON.
        huggingface_cache_dir (str, optional): HuggingFace cache directory.
        multilingual (bool): use multilingual font tokens.
        device (str or torch.device, optional): target device.

    Returns:
        tuple: ``(byt5_text_encoder, byt5_tokenizer)``.
    """
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    byt5_tokenizer = AutoTokenizer.from_pretrained(byt5_name, cache_dir=huggingface_cache_dir)
    byt5_text_encoder = T5ForConditionalGeneration.from_pretrained(
        byt5_name, cache_dir=huggingface_cache_dir
    ).get_encoder()

    device = torch.device(device)
    byt5_text_encoder = byt5_text_encoder.to(device)

    if special_token:
        add_special_token(
            byt5_tokenizer,
            byt5_text_encoder,
            add_color=color_special_token,
            add_font=font_special_token,
            color_ann_path=color_ann_path,
            font_ann_path=font_ann_path,
            multilingual=multilingual,
        )
    return byt5_text_encoder, byt5_tokenizer


class ByT5Mapper(nn.Module):
    """Projects ByT5 glyph embeddings into the model embedding space.

    Args:
        in_dim (int): input embedding dimension.
        out_dim (int): intermediate output dimension; must equal ``in_dim`` when
            ``use_residual`` is True.
        hidden_dim (int): hidden dimension of the first projection.
        out_dim1 (int): final output dimension.
        use_residual (bool): add the input as a residual to the output.
    """

    def __init__(self, in_dim, out_dim, hidden_dim, out_dim1, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim1)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x2 = self.act_fn(x)
        x2 = self.fc3(x2)
        if self.use_residual:
            x2 = x2 + residual
        return x2
