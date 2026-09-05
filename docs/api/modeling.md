# Modeling

HY15 and Wan model definitions.

::: minwm.modeling

## Model adapters

The adapter layer bridges a recipe's `[B, F, C, H, W]` tensor space to each
backbone's call convention and owns text encoding (prompts → the `context` the
model consumes). Adapters are config-built and injected into a recipe, so a
stage stays backbone-agnostic. They are imported from their submodules (not
re-exported at `minwm.modeling`) and documented here from their defining module.

::: minwm.modeling.adapter

::: minwm.modeling.wan21.adapter

## Wan text encoder

The Wan line conditions on real prompts via a frozen umt5-xxl encoder, built by
the config system and injected into the [`Wan21Adapter`][minwm.modeling.wan21.adapter.Wan21Adapter]
(falling back to zero embeddings when absent, for mock smoke tests).

::: minwm.modeling.wan21.text_encoder
