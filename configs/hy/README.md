# minWM Training — HunyuanVideo 1.5 Backbone

Model line built on HunyuanVideo 1.5:

- **HY Action2V** — camera-controlled action-to-video (ProPE camera conditioning). Full pipeline in [`configs/hy/action2v/`](action2v/README.md).

Each line splits into two phases:

- **Phase 1 Bidirectional SFT** — bidirectional multi-step base.
- **Phase 2 Causal Forcing** — distillation to causal few-step.

Phase 2 has 4 stages: Stage 1 Teacher Forcing AR Diffusion, Stage 2(a) Causal ODE Distillation Initialization (Causal Forcing), Stage 2(b) Causal Consistency Distillation (Causal Forcing++), Stage 3 Asymmetric DMD with Self Rollout.

> Wan-backbone training lives in [`configs/wan21/`](../wan21/README.md). Quick Start / inference commands live in the main [README](../../README.md).

