"""Conditioning encoders for the HY15 pipeline: text (Qwen/T5), vision (SigLIP), ByT5 glyph.

HuggingFace backends are loaded lazily inside the loaders/constructors, so
importing this subpackage never forces an optional dependency or a weight load.
"""

from .byt5_mapper import (
    ByT5Mapper,
    add_special_token,
    create_byt5,
    load_byt5_and_byt5_tokenizer,
    load_glyph_byT5_v2,
)
from .format_prompt import MultilingualPromptFormat
from .text_encoder import (
    PROMPT_TEMPLATE,
    TextEncoder,
    TextEncoderModelOutput,
    load_text_encoder,
    load_tokenizer,
)
from .vision_encoder import (
    VisionEncoder,
    VisionEncoderModelOutput,
    load_image_processor,
    load_vision_encoder,
)

__all__ = [
    "ByT5Mapper",
    "add_special_token",
    "create_byt5",
    "load_byt5_and_byt5_tokenizer",
    "load_glyph_byT5_v2",
    "MultilingualPromptFormat",
    "PROMPT_TEMPLATE",
    "TextEncoder",
    "TextEncoderModelOutput",
    "load_text_encoder",
    "load_tokenizer",
    "VisionEncoder",
    "VisionEncoderModelOutput",
    "load_image_processor",
    "load_vision_encoder",
]
