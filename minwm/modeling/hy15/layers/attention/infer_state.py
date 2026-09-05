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
"""Process-global inference state for the HY15 DiT.

Holds runtime toggles (SageAttention, torch.compile, fp8 GEMM, VAE parallel)
consulted by the attention dispatch. During training the state is never
initialized, so :func:`get_infer_state` returns ``None`` and all inference-only
fast paths stay disabled.
"""

from dataclasses import dataclass, field


@dataclass
class InferState:
    """Runtime inference toggles for the HY15 DiT.

    Args:
        enable_sageattn (bool): whether to use SageAttention.
        sage_blocks_range (range | None): block indices that use SageAttention.
        enable_torch_compile (bool): whether to use torch.compile.
        use_fp8_gemm (bool): whether to use fp8 GEMM.
        quant_type (str): fp8 quantization type.
        include_patterns (list[str]): module-name patterns included in fp8 GEMM.
        use_vae_parallel (bool): whether to use VAE parallelism.
    """

    enable_sageattn: bool = False
    sage_blocks_range: range | None = None
    enable_torch_compile: bool = False

    # fp8 gemm related
    use_fp8_gemm: bool = False
    quant_type: str = "fp8-per-block"
    include_patterns: list[str] = field(default_factory=lambda: ["double_blocks"])

    # vae related
    use_vae_parallel: bool = False


__infer_state = None


def parse_range(value: str) -> list[int]:
    """Parse a ``"start-end"`` range or ``"a,b,c"`` list into a list of ints.

    Args:
        value (str): either a hyphen range (inclusive) or a comma-separated list.

    Returns:
        list[int]: the expanded integer indices.
    """
    if "-" in value:
        start, end = map(int, value.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(x) for x in value.split(",")]


def initialize_infer_state(args) -> InferState:
    """Build and store the process-global :class:`InferState` from CLI args.

    Args:
        args: namespace exposing ``sage_blocks_range``, ``use_sageattn``,
            ``enable_torch_compile``, ``use_fp8_gemm``, ``quant_type``,
            ``include_patterns``, and ``use_vae_parallel``.

    Returns:
        InferState: the initialized, process-global inference state.
    """
    global __infer_state
    sage_blocks_range = parse_range(args.sage_blocks_range)
    use_sageattn = getattr(args, "use_sageattn", False)

    include_patterns = getattr(args, "include_patterns", "double_blocks")
    if isinstance(include_patterns, str):
        include_patterns = [p.strip() for p in include_patterns.split(",") if p.strip()]

    __infer_state = InferState(
        enable_sageattn=use_sageattn,
        sage_blocks_range=sage_blocks_range,
        enable_torch_compile=args.enable_torch_compile,
        use_fp8_gemm=args.use_fp8_gemm,
        quant_type=args.quant_type,
        include_patterns=include_patterns,
        use_vae_parallel=args.use_vae_parallel,
    )
    return __infer_state


def get_infer_state() -> InferState | None:
    """Return the process-global inference state, or ``None`` if uninitialized."""
    return __infer_state
