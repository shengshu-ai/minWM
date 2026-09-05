---
name: onboarding-world-model
description: "Onboarding guide for newcomers to video world model training in minWM. Covers two parts: Foundations (background theory for the two-phase pipeline) and Pitfalls (non-obvious mistakes from hands-on experience). Use when a newcomer is starting on controllable video generation training and needs background or wants to avoid common mistakes."
license: MIT
---

# onboarding-world-model

Two parts:
- **Foundations** — minimum background needed to read minWM's training code
- **Pitfalls** — non-obvious mistakes from hands-on experience

> Note: the `HY15/`, `Wan21/`, `shared/` reference trees are no longer tracked in
> git (kept locally as reference). Paths below resolve in a local checkout that
> still has them; after a fresh clone, restore via `git checkout <sha> -- <tree>`.

---

## Part 1: Foundations

minWM converts a multi-step bidirectional T2V/TI2V diffusion model into a camera-controllable few-step autoregressive (AR) video generator. The pipeline has **two phases**:

### Phase 1: Camera-Controllable Bidirectional Diffusion

Start from a pretrained T2V/TI2V bidirectional diffusion model and fine-tune it into a **camera-controllable bidirectional model**.

Camera conditions are injected via **PRoPE** (projective RoPE):

- For each frame `i`, given intrinsic `K_i` and extrinsic `T_i^cw ∈ SE(3)`, build a 4×4 lifted projective matrix `P_i`.
- For a token at frame `i(t)` with spatial coords `(x_t, y_t)`, build a block-diagonal transformation combining `P_i` (for camera) with standard RoPE on `(x_t, y_t)` (for spatial position).
- This transformation is plugged into self-attention in GTA form, so attention between two tokens depends on the **relative projective transformation** `P_{i(t1)} P_{i(t2)}^{-1}` — encoding both relative intrinsics and relative camera pose.

Why this matters: the bidirectional backbone is conditioned on camera trajectories without changing the self-attention generative structure. This is the model that all subsequent AR distillation stages start from.

### Phase 2: AR Diffusion Distillation

The Phase 1 model is multi-step bidirectional — too slow for real-time interactive use. Phase 2 distills it into a **few-step causal AR model** via the [Causal Forcing](https://arxiv.org/abs/2602.02214) / [Causal Forcing++](https://arxiv.org/abs/2605.15141) recipe.

Three stages:

**Stage 1 — AR diffusion training (Teacher Forcing).** Fine-tune the bidirectional model into an AR diffusion model: each frame `x^i` is denoised conditioned on the clean prefix `x_gt^{<i}` (real data, hence "teacher forcing"). This produces the **AR teacher**.

**Stage 2 — Few-step initialization.** Two equivalent options:

- **Stage 2a — Causal ODE ([Causal Forcing](https://arxiv.org/abs/2602.02214)):** With the AR teacher, generate a large set of PF-ODE denoising trajectories. Then sample timestep `t` from a few-step set `S` and train the few-step model `G_θ` by regressing the noisy frame `x_t^i` to the clean `x_0^i`. Cost: data curation + trajectory storage.

- **Stage 2b — Causal CD ([Causal Forcing++](https://arxiv.org/abs/2605.15141)):** Theoretically equivalent to causal ODE, but uses consistency distillation — no need to pre-collect trajectories. The student `G_θ` matches its prediction at `t` against the EMA teacher `G_{θ⁻}`'s prediction at `t-Δt`, where `t-Δt` is reached via a single ODE step from `x_t^i` using the AR teacher.

**Stage 3 — Asymmetric DMD with Self-Rollout.** The Stage 2 student is real-time but inherits the AR teacher's quality ceiling. To break it: initialize the student from Stage 2, **self-rollout** a full video sequence `x̃`, and optimize using the DMD gradient — the difference between two scores:
- `s_real`: frozen score from the bidirectional Phase 1 model (real data distribution)
- `s_fake`: online-trained score on `x̃` (current student distribution)

The student's parameters move to make `s_fake` match `s_real`, pulling the rollout distribution toward the bidirectional model's quality.

### Camera Conditioning Across Phase 2

All Phase 2 stages keep the camera condition active:

- Stage 1: AR diffusion is initialized from the Phase 1 camera-controllable model and trained on camera-controllable data
- Stage 2: When collecting causal ODE trajectories or running causal CD, the AR teacher takes camera as input
- Stage 3: Self-rollout takes both text and camera; both `s_real` and `s_fake` are initialized from the Phase 1 camera-controllable bidirectional model

Result: every model in the pipeline is camera-controllable.

### Mental Model

```
T2V/TI2V bidirectional (pretrained)
        │
        │  Phase 1: PRoPE fine-tuning
        ▼
Camera-controllable bidirectional (multi-step)
        │
        │  Stage 1: Teacher Forcing
        ▼
Camera-controllable AR teacher (multi-step)
        │
        │  Stage 2a (Causal ODE) or Stage 2b (Causal CD)
        ▼
Few-step AR student (real-time, quality bounded by AR teacher)
        │
        │  Stage 3: Asymmetric DMD with self-rollout
        ▼
Few-step AR student (real-time, quality matches bidirectional teacher)
```

When reading the code, every `*_diffusion.py` / `*_dmd.py` / `*_ode_regression.py` / `*_consistency.py` file under `Wan21/model/` and `HY15/trainer/pipelines/` corresponds to one of these stages.

### A Small Request Before You Read Code

Phase 2 of minWM stands on the shoulders of the [Causal Forcing](https://arxiv.org/abs/2602.02214) line of work, and the T2V prototype is generously open-sourced at **https://github.com/thu-ml/Causal-Forcing**. If this onboarding helped you get oriented, please consider giving the prototype repo a star. It's a small gesture, but it goes a long way toward supporting open research and encouraging the authors to keep sharing their work with the community. Thank you.

---

## Part 2: Pitfalls

Non-obvious mistakes from building minWM. Read this before you start training.

### Pitfall 1: Underestimating data quality

Data is the most important factor in controllable generation — more so than in T2V.

**GT conditions must be precise and simple.** The cleaner and more unambiguous the condition signal, the easier the model learns. Noisy or imprecise GT (e.g., inaccurate camera poses, ambiguous action labels) directly caps what the model can learn.


### Pitfall 2: Wrong learning rate and batch size

Controllable generation is more sensitive to bs than T2V.

**bs < 8 is not enough.** Unlike T2V, a batch size of 2 is typically insufficient for the model to learn controllability. Use a larger batch size.


### Pitfall 3: Misreading early training as method failure

The model will appear completely uncontrollable for the first several thousand steps. This is normal.

- **1.3B model**: controllability typically emerges around 1k steps.
- **8B model**: may take > 5k steps before any controllability is visible.

Do not conclude the method is wrong based on early training behavior. Train longer before making changes.
