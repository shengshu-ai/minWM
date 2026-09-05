# minWM

> *A full-stack framework and tutorial for newcomers, rather than a specific model.*

**minWM** is a full-stack open-source framework that walks you end-to-end
through turning a bidirectional T2V foundation model into an action-conditioned
video world model — with example data, runnable scripts, Claude skills capturing
hands-on experience, and onboarding knowledge for newcomers.

This site is the developer documentation for the `minwm` package. For the
project overview, demo, and citation, see the
[README on GitHub](https://github.com/shengshu-ai/minWM).

## Where to start

- [Getting Started](getting-started.md) — install the framework and run the demos.
- [Architecture](architecture.md) — the layered design: how the engine, recipes,
  modeling, and data layers fit together.
- [API Reference](api/index.md) — auto-generated reference for every `minwm`
  subpackage, rendered from in-code docstrings.

## What minWM covers

The complete **data → training → inference** pipeline is open-sourced; every
stage exposes input/output checkpoints so you can stop, swap, or fork anywhere.

- **Data** — construct training-ready datasets paired with camera poses, and the
  pipeline that turns them into latents.
- **Training** — FSDP + sequence parallelism, single-/multi-node training, and the
  full distillation pipeline from a bidirectional diffusion model to a 4-step AR
  student.
- **Inference** — 4-step DMD inference for HY Action2V / HY TI2V / Wan Action2V,
  multi-GPU sequence parallelism, camera-trajectory control.

### Multi-backbone support

| Backbone             | Architecture          | Params | Training       | Inference    |
| -------------------- | --------------------- | ------ | -------------- | ------------ |
| **Wan 2.1**          | Cross-attention + DiT | 1.3 B  | all 4 stages   | 4-step DMD   |
| **HunyuanVideo 1.5** | MMDiT                 | 8 B    | all 4 stages   | 4-step DMD   |

Both lines share the same trainer / loss / dataset abstractions, so adding a
third backbone is structurally a wrapper-and-config exercise.
