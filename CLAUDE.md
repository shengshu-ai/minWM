# minWM Engine — Code Style & Conventions

Project-level coding standards for `minwm/`. These bind both human contributors
and code agents. Reference style: detectron2 / PyTorch source. Scope: the
`minwm/` package and `tools/` + `tests/`. Legacy trees (`HY15/`, `Wan21/`,
`shared/`) are exempt until migrated.

---

## Type annotations

Use **built-in generics** (PEP 585), never the capitalized `typing` aliases.

- `list[int]`, `dict[str, Tensor]`, `tuple[int, ...]` — NOT `List`, `Dict`, `Tuple`.
- Use **PEP 604 unions**: `X | None` for optionals, `A | B` for unions — NOT
  `Optional[X]` or `Union[A, B]`. Don't import `Optional` or `Union` from `typing`.
- Do **not** add `from __future__ import annotations`. We target Python 3.10+
  where built-in generics and `X | None` unions evaluate fine at runtime, and we
  have no forward-ref or annotation-introspection need that would justify it.
  (Vendored trees that carry it from upstream — e.g. `minwm/distributed/sp/` —
  are left as-is.)

```python
import torch
from torch import Tensor


def encode(frames: list[Tensor], mask: Tensor | None = None) -> dict[str, Tensor]:
    ...
```

Do **not** import `List`, `Dict`, `Tuple`, `Set`, `Type`, `Optional`, `Union`
from `typing`. `Any`, `Iterator`, `Iterable`, `Callable` are fine.

---

## Docstrings — Google style

Public functions, classes, and methods **must** have a Google-style docstring.
Private helpers (`_name`), one-liners, `@property`, and dunder methods
(`__len__`, `__getitem__`) may omit it.

Sections: `Args:`, `Returns:`, `Raises:` (only those that apply). Include the
**type in the docstring too** — `name (type): description` — even though the
signature is annotated. This keeps rendered docs self-contained.

```python
def build_dataloader(data_cfg: dict) -> DataLoader:
    """Build a DataLoader from a two-level ``data`` config.

    Args:
        data_cfg (dict): config with a nested ``dataset`` node plus dataloader
            knobs (``batch_size``, ``num_workers``, ``shuffle``, ``drop_last``).

    Returns:
        DataLoader: a loader whose sampler is DP-aware under sequence parallelism.

    Raises:
        ValueError: if ``data_cfg`` has no ``dataset`` sub-config.
    """
    ...
```

One-line docstrings for simple public functions are acceptable:

```python
def is_main_process() -> bool:
    """True on global rank 0 (logs / writes checkpoints)."""
    return get_rank() == 0
```

Module docstring: every module begins with a one-line summary, optionally
followed by a short paragraph and a usage example.

---

## Imports

isort + black enforced (profile black, line length 100). Order: future →
stdlib → third-party → first-party (`minwm`) → local (`.`). Absolute imports
for `minwm.*`; relative imports within the same subpackage — including
`__init__.py` importing its own siblings (e.g. `from .collective import …`
inside `minwm/distributed/__init__.py`).

Heavy / optional deps (`lmdb`, `scipy`, `PIL`, CUDA-only modules) must not be
imported at package top level. Import them lazily inside the function/class that
needs them, or behind a PEP 562 `__getattr__`, so importing a package never
forces an optional dependency.

---

## Formatting

- **black**, line length **100**. Always run `black` before lint checks — black
  is the source of truth for formatting (e.g. blank lines around early returns).
  Don't hand-format; let black decide.
- **flake8** clean (config in `.flake8`).
- 4-space indent, no tabs. UTF-8. Newline at EOF, no trailing whitespace.

---

## Naming

- `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE`
  constants, `_leading_underscore` for non-public.
- Config-buildable classes are referenced by `"module.path:ClassName"` strings
  (the `_cls` key). Keep public class names importable from their subpackage.

---

## Comments

Default to **no comment**. Add one only when the *why* is non-obvious: a hidden
constraint, a subtle invariant, a workaround. Don't restate what the code says.
Never leave commented-out code or `# removed ...` markers — delete it.

---

## What not to do

- No `typing.List` / `Dict` / `Tuple` / `Optional` / `Union` — use PEP 585/604
  builtins (`list`, `dict`, `X | None`, `A | B`) instead.
- No wildcard imports (`from x import *`) outside vendored code.
- No top-level heavy imports in `__init__.py`.
- No docstring on trivial/private helpers just to fill space.
- No backwards-compat shims, dead code, or speculative abstractions.
