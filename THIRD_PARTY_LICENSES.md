# Third-Party Licenses

minWM is released under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
It also contains, derives from, or adapts third-party works that carry their own
license terms. Those terms govern the corresponding files regardless of minWM's
own license. Full license texts are in the [`licenses/`](licenses/) directory.

## Bundled / derived source code

| Component | Location in this repo | Upstream | License | Full text |
|---|---|---|---|---|
| HunyuanVideo 1.5 | `minwm/modeling/hy15/` | [Tencent-Hunyuan/HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | Tencent Hunyuan Community License Agreement (THCL) | [licenses/TENCENT_HUNYUAN_COMMUNITY_LICENSE.txt](licenses/TENCENT_HUNYUAN_COMMUNITY_LICENSE.txt) |
| Wan 2.1 | `minwm/modeling/wan21/` | [Wan-AI/Wan](https://github.com/Wan-AI/Wan) | Apache-2.0 | [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt) |
| vLLM | `minwm/distributed/sp/` | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 | [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt) |
| NVIDIA Megatron-LM | `minwm/distributed/sp/parallel_state.py` (adapted via vLLM) | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | See per-file SPDX / copyright headers | — |

Per-file provenance headers are retained where present: THCL headers on the
`minwm/modeling/hy15/` files, and `SPDX-License-Identifier: Apache-2.0` plus
`Adapted from …` attribution on the vLLM/Megatron-derived files under
`minwm/distributed/sp/`.

### Tencent Hunyuan Community License — key obligations

The THCL is **not** an OSI-approved open-source license. Any use of the
`minwm/modeling/hy15/` code and of derived weights (fine-tuned, distilled, or
DMD-student) must comply with it, notably:

- **Territory** — the license applies worldwide **except** the European Union,
  the United Kingdom, and South Korea.
- **Model Derivatives** — fine-tuning, distillation, and pattern-transfer models
  are "Model Derivatives" and remain subject to the agreement.
- **Pass-through** — downstream agreements must carry the THCL use restrictions,
  and a copy of the agreement must accompany distributions.
- **Output restriction** — Tencent Hunyuan outputs may not be used to improve any
  other AI model (other than Tencent Hunyuan or its Model Derivatives).
- **Commercial scale** — very large deployments may require a separate license
  from Tencent.

## Training data

The example videos in the minWM dataset are generated with HunyuanVideo
(HY-WorldPlay); their use is subject to the upstream model's license.

## Acknowledged references

Algorithms and reference implementations that informed minWM but whose code is
not bundled here:
[Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) and
[FastVideo](https://github.com/hao-ai-lab/FastVideo). See the README
Acknowledgements for the full list.
