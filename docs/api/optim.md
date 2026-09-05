# Optim

Optimization utilities for minWM training: the **Muon** optimizer (vendored from
[Keller Jordan's reference implementation](https://github.com/KellerJordan/Muon/blob/master/muon.py)),
plus first-party EMA helpers and optimizer builders.

!!! note "Hand-written, not auto-generated"
    Unlike the rest of the API reference, this page is written by hand.
    `minwm.engine.optim.muon` is vendored upstream code kept verbatim (see the exception
    in `.flake8` and `CLAUDE.md`), so its docstrings don't follow the project's
    Google style. Auto-API is reserved for first-party `minwm` code that conforms
    to that style; vendored modules are documented in prose so the strict doc build
    stays a meaningful gate on real docstrings.

## `Muon` — MomentUm Orthogonalized by Newton-Schulz

`minwm.engine.optim.muon.Muon` is a hybrid optimizer. Every parameter of rank ≥ 2 is
updated by Muon — momentum SGD whose update is orthogonalized by a quintic
Newton-Schulz iteration — while everything else (1-D tensors, embeddings, heads)
falls back to an internal AdamW.

```python
Muon(
    lr=1e-3,
    wd=0.1,
    muon_params=None,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
    adamw_params=None,
    adamw_betas=(0.95, 0.95),
    adamw_eps=1e-8,
)
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `lr` | `1e-3` | Learning rate; Muon updates have spectral norm ≈ `lr` (≈0.02 is a good Muon default). |
| `wd` | `0.1` | Weight decay, applied to both the Muon and AdamW groups. |
| `muon_params` | `None` | Parameters optimized by Muon. Must be rank ≥ 2. |
| `momentum` | `0.95` | Momentum of the internal SGD. |
| `nesterov` | `True` | Use Nesterov-style momentum in the internal SGD (recommended). |
| `ns_steps` | `5` | Newton-Schulz iterations per step (≈6 is plenty). |
| `adamw_params` | `None` | Extra parameters optimized by the internal AdamW. |
| `adamw_betas` | `(0.95, 0.95)` | Betas for the internal AdamW. |
| `adamw_eps` | `1e-8` | Epsilon for the internal AdamW. |

`step(closure=None)` performs one optimization step, returning the loss when a
closure is supplied.

## `get_muon_optimizer` — convenience constructor

```python
get_muon_optimizer(
    model,
    lr=1e-3,
    weight_decay=0.1,
    momentum=0.95,
    adamw_betas=(0.95, 0.95),
    adamw_eps=1e-8,
)
```

Splits `model.named_parameters()` by rank — tensors of rank ≥ 2 go to the Muon
group, the rest to the AdamW group — and returns a configured `Muon` instance.
This is the entry point most training code uses.

## `zeropower_via_newtonschulz5` — orthogonalization kernel

`zeropower_via_newtonschulz5(G, steps=5)` computes the zeroth matrix power
(orthogonalization) of `G` via a quintic Newton-Schulz iteration whose
coefficients maximize the slope at zero. It is `DTensor`-aware: a sharded input
is gathered to a full tensor, processed, then redistributed. This kernel backs
each Muon update and is not usually called directly.

## Optimizer builders

::: minwm.engine.optim.builder

## EMA helpers

::: minwm.engine.optim.ema
