---
name: integrate-new-backbone
description: "Step-by-step recipe for plugging a new video DiT backbone into minWM, grounded in the HunyuanVideo (HY15) and Wan 2.1 reference integrations. Use when a user wants to add a new backbone to the framework."
license: MIT
---

# integrate-new-backbone

> Reference implementations: HY15 (MMDiT, 8B) and Wan21 (Cross-attention DiT, 1.3B).
> Whichever path you choose, it usually helps to follow the reference whose architecture is closer to your new backbone.

> Note: the `HY15/`, `Wan21/`, `shared/` reference trees are no longer tracked in
> git (kept locally as reference). Paths below resolve in a local checkout that
> still has them; after a fresh clone, restore via `git checkout <sha> -- <tree>`.

---

## Framework Layout

```
minWM/
├── HY15/                  # HunyuanVideo backbone
│   ├── hyvideo/           # model implementation (models/, pipelines/, prope/, schedulers/)
│   └── trainer/           # trainer (configs/, pipelines/, training/)
├── Wan21/                 # Wan 2.1 backbone
│   ├── model/             # model implementation (base.py + per-stage model files)
│   ├── pipeline/          # inference / training pipelines
│   └── wan_utils/         # wrapper, loss, dataset, scheduler, etc.
└── shared/                # shared across backbones (SP comms, config bases, algorithms)
```

Each backbone is a **self-contained** top-level directory; the two backbones do not import from each other, and their run-scripts set their own `PYTHONPATH` (only the `minwm` package is pip-installed). A new backbone follows the same pattern: create a new top-level directory (e.g. `NewBB/`).

---

## Step 1: Wrapper Layer — wrap the new model behind a unified interface

Wan21 does this in `wan_utils/wan_wrapper.py`, exposing three wrappers:

- `WanDiffusionWrapper`: wraps the DiT, handles forward and the causal/bidirectional switch
- `Wan21TextEncoder`: text encoder
- `Wan21VAEWrapper`: VAE encode/decode

**Suggested approach**: write the equivalent three wrappers for the new backbone, ideally keeping the interface aligned with:

- `DiffusionWrapper.forward(latents, timestep, encoder_hidden_states, **kwargs) -> noise_pred`
- `VAEWrapper.encode_to_latent(pixel) -> latent`
- `VAEWrapper.decode_to_pixel(latent) -> pixel`

The HY15 counterparts live in `hyvideo/models/` and `trainer/models/` — useful as a second reference.

---

## Step 2: BaseModel — hook into the training abstraction

Wan21 training models inherit from `Wan21/model/base.py::BaseModel`. The core is `_initialize_models()`, which instantiates:

```python
self.generator    # trainable DiT (causal)
self.real_score   # frozen teacher (bidirectional)
self.fake_score   # trainable fake score
self.text_encoder
self.vae
self.scheduler
```

**Suggested approach**: create `NewBB/model/base.py` with a similar structure, replacing `WanDiffusionWrapper` etc. with your own wrappers.

The per-stage model files (`camera_dmd.py`, `dmd.py`, `ode_regression.py`, ...) inherit `BaseModel` and implement their own `forward()`. The new backbone follows the same pattern.

---

## Step 3: Pipeline — inference and training flow

Each backbone has its own pipeline directory:

- `Wan21/pipeline/`: `SelfForcingTrainingPipeline`, `BidirectionalTrainingPipeline`
- `HY15/trainer/pipelines/`: split per stage (`ar_hunyuan_training_pipeline.py`, `ar_causal_cd_pipeline.py`, ...)

Pipelines are responsible for: given noise + condition, run the denoising steps and return a trajectory.

**Suggested approach**:

1. Start by getting the bidirectional (Phase 1) pipeline working — the simplest implementation in `Wan21/pipeline/` is a friendly place to begin.
2. Then progressively integrate AR diffusion TF → causal ODE → causal CD → DMD stages.

Key interface: `inference_with_trajectory(noise, clean_image_or_video, **conditional_dict)`, returning `(pred, timestep_from, timestep_to)`.

---

## Step 4: ProPE — condition injection

Camera control is injected via ProPE, implemented in:

- `Wan21/prope/` (Wan implementation)
- `HY15/hyvideo/prope/` (HY implementation)

ProPE turns a pose string (e.g. `"a*4,w*8,s*7"`) into positional encodings injected into the DiT's RoPE.

**Suggested approach**: locate the new backbone's RoPE implementation and follow the HY15 or Wan21 ProPE wiring pattern, passing the pose embedding through `DiffusionWrapper.forward()`.

---

## Step 5: SP (Sequence Parallel) adaptation

Shared SP communication primitives live in `shared/sp/`. SP tends to be the trickiest part of integrating a new backbone, so it's worth walking through every item below carefully (a fuller checklist lives in the `debug-world-model` skill):

- FSDP `process_group` uses DP group, not world group
- seed offset uses `dp_rank`, not `global_rank`
- no random number calls inside any `if rank == 0:` block
- KV cache lives in the head-parallel domain; head count = `H // sp_size`
- strip SP padding before attention, restore after attention

---

## Step 6: Training entry point and configs

- Wan21 entry: `wan_train.py` + `wan_utils/configs/`
- HY15 entry: `trainer/training/` + `trainer/configs/`

For a new backbone, the Wan21 entry style is often the easier starting point thanks to its simplicity. You might create `NewBB/new_bb_train.py` and use `ModelConfig` from `shared/configs/base.py` as the config base class.

---

## Recommended Integration Order

1. Get the wrapper layer running on a single GPU for bidirectional inference
2. Hook in `BaseModel`, get Phase 1 SFT training running
3. Integrate pipelines stage by stage (TF → ODE / CD → DMD)
4. Add ProPE condition injection
5. Turn on SP, validate every item in the checklist

Each phase has corresponding HY15 / Wan21 reference files. When you run into a concrete issue, a useful habit is to ask "how does HY do this? how does Wan do this?" and let the two references guide your decision.
