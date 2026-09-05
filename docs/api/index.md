# API Reference

Auto-generated from the Google-style docstrings in the `minwm` package. Each
page below documents one subpackage; private members (leading underscore) are
hidden.

| Subpackage | What lives here |
|---|---|
| [config](config.md) | Lazy config loading + `module.path:ClassName` instantiation. |
| [data](data.md) | Dataset classes + dataloader factories. |
| [distributed](distributed.md) | Collectives, sequence parallelism, FSDP helpers. |
| [engine](engine.md) | `BaseTrainer`, `BaseInferencer`, and the training `Recipe` contract. |
| [modeling](modeling.md) | Backbone DiTs, model-family layers, adapters, and shared modeling code. |
| [processors](processors.md) | Composable `_cls`-buildable batch/input preprocessing stages. |
| [recipes](recipes.md) | Per-stage training recipes (flow matching, DMD) — `minwm.engine.training`. |
| [sampling](sampling.md) | Model-agnostic noise-schedule / flow-matching primitives. |
| [optim](optim.md) | Optimizers (Muon), EMA, and optimizer builders — `minwm.engine.optim`. |
| [utils](utils.md) | Logging, distributed comm, misc helpers. |
