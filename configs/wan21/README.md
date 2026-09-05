# minWM Training — Wan 2.1 Backbone

Model line built on Wan 2.1:

- **Wan Action2V** — camera-controlled action-to-video (PRoPE camera conditioning). Full pipeline in [`configs/wan21/action2v/`](action2v/README.md).

Each line splits into two phases:
- **Phase 1 Bidirectional SFT** — bidirectional multi-step base.
- **Phase 2 Causal Forcing** — distillation to causal few-step.

Phase 2 has 4 stages: Stage 1 Teacher Forcing AR Diffusion, Stage 2(a) Causal ODE Distillation Initialization (Causal Forcing), Stage 2(b) Causal Consistency Distillation (Causal Forcing++), Stage 3 Asymmetric DMD with Self Rollout.

> HunyuanVideo-backbone training lives in [`configs/hy/`](../hy/README.md). Quick Start / inference commands live in the main [README](../../README.md).
