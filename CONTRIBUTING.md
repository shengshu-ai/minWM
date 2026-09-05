# Contributing to minWM

Thanks for your interest in improving minWM. This guide covers how to set up a
dev environment, the coding standards we enforce, and how to get a change merged.

## Ways to contribute

- **Report bugs / request features** — open a GitHub issue with enough detail to
  reproduce (command, config, stack trace, environment).
- **Submit code** — fork, branch, and open a pull request against the default
  branch. Small, focused PRs review faster than large mixed ones.
- **Improve docs** — the training guides, API docstrings, and this repo's docs
  site (`mkdocs`) all welcome fixes.

## Development setup

Follow [INSTALL.md](INSTALL.md): create the environment, install the pinned
runtime deps, then install the package editable so `import minwm` resolves:

```bash
pip install -r requirements/base.txt
pip install -e .
```

## Coding standards

Project coding standards live in [CLAUDE.md](CLAUDE.md) and bind both human and
agent contributors. The essentials for `minwm/`, `tools/`, and `tests/`:

- **Formatting** — `black`, line length **100** (the source of truth; run it
  before pushing). 4-space indent, newline at EOF, no trailing whitespace.
- **Imports** — `isort` (profile black), and no top-level heavy/optional imports
  in `__init__.py` (import CUDA-only / optional deps lazily).
- **Lint** — `flake8` must be clean (config in `.flake8`).
- **Type hints** — built-in generics (PEP 585: `list[int]`, `dict[str, Tensor]`)
  and PEP 604 unions (`X | None`, `A | B`). Do **not** import `List`/`Dict`/
  `Optional`/`Union` from `typing`, and do **not** add
  `from __future__ import annotations`.
- **Docstrings** — Google style on public functions, classes, and methods, with
  the type included in the docstring too. Trivial/private helpers may omit it.
- **Comments** — only where the *why* is non-obvious. No commented-out code.

Legacy trees are exempt until migrated; new code goes under `minwm/`.

## Running checks locally

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs these on every
push/PR to `main` and `dev-meta`; run them before opening a PR:

```bash
black --check minwm tests
isort --check-only minwm tests
flake8 minwm tests
pytest tests/ -q
```

The docs site is validated with a strict build
([.github/workflows/docs.yml](.github/workflows/docs.yml)):

```bash
mkdocs build --strict
```

## Commit and PR conventions

- Write [Conventional Commits](https://www.conventionalcommits.org/) messages —
  `type(scope): summary` (e.g. `fix(engine): ...`, `docs: ...`), matching the
  existing history.
- Keep a PR to one logical change; explain the *why* in the description.
- Make sure the lint, test, and docs checks above pass.

## License of contributions

By contributing you agree that your contributions to the minWM framework are
licensed under [Apache-2.0](LICENSE). Note that files under
`minwm/modeling/hy15/` are governed by the Tencent Hunyuan Community License
(see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)); changes there must
comply with it.
