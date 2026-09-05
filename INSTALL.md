## Requirements

- **Linux** with an NVIDIA GPU. CPU-only works for linting and the unit tests,
but not for training or inference.
- **Python 3.12** recommended (matches CI; the project supports 3.10+). 3.12
  also has a prebuilt flash-attn wheel for torch 2.9 (see below), so you can
  skip the source build.
- **PyTorch 2.9.1 + torchvision 0.24.1**, built for **CUDA 12.8**. The exact
pinned wheels (including the `nvidia-*-cu12==12.8.`* runtime) live in
`[requirements/base.txt](requirements/base.txt)`.
- An NVIDIA **driver new enough for CUDA 12.8**. Older drivers make PyTorch
silently fall back to CPU — see
[Common Installation Issues](#common-installation-issues).
- **flash-attn** — installed separately (see below). Building it needs **ninja**
(pinned in `requirements/base.txt`) and a CUDA toolchain.
- `git`, `conda` (or `mamba`/`venv`), and a recent `pip`.

Requirements are organized by mode under `[requirements/](requirements/)`:


| File                      | Install                                  | Contents                                                                 |
| ------------------------- | ---------------------------------------- | ------------------------------------------------------------------------ |
| `requirements/base.txt`   | `pip install -r requirements/base.txt`   | Full frozen training + inference env.                                    |
| `requirements/dev.txt`    | `pip install -r requirements/dev.txt`    | `base` + lint/test tooling (black, isort, flake8, pytest), pinned to CI. |
| `requirements/deploy.txt` | `pip install -r requirements/deploy.txt` | Slim inference-only subset (no training/data/export deps).               |


## Installation

```bash
# 1. Clone
git clone <repo-url> minWM-engine
cd minWM-engine

# 2. Create + activate the env
conda create -n minwm python=3.12 -y
conda activate minwm

# 3. Install pinned dependencies (torch, torchvision, CUDA 12.8 wheels, …)
pip install -r requirements/base.txt

# 4. Install flash-attn (separate: needs torch already present, hence
#    --no-build-isolation; the build also needs ninja, which step 3 installed)
pip install flash-attn --no-build-isolation

# 5. Install minwm editable (registers `import minwm` in the env; source edits
#    are picked up live — no PYTHONPATH needed)
pip install -e .
```

> **Editable install vs. PYTHONPATH.** `pip install -e .` puts `minwm` on the
> env's import path permanently, so `import minwm` works in any shell, under
> `torchrun`, and under cluster launchers — no `export PYTHONPATH` to remember
> or to pass through. Only the `minwm` package is installed; the legacy trees
> (`HY15/`, `Wan21/`, `shared/`) are not packaged, and their run-scripts set
> their own PYTHONPATH (see the cluster note in
> [Common Installation Issues](#common-installation-issues)).

## Remote Checkpoint Storage (optional)

Set `checkpoint.remote_root` to an object-store prefix and the run's
checkpoints route there instead of the local disk. `training.output_dir` stays a
scheme-free local run name — it always holds the run's logs / TensorBoard /
`wandb` dir (those only work on a local filesystem) — and the checkpoints land
at `<remote_root>/<output_dir>/ckpts`:

```python
training = dict(output_dir="wan21-action2v/stage0_bi_sft")   # local logs live here
checkpoint = dict(remote_root="s3://<bucket>/<prefix>")  # ckpts route here
```

The backend is decided from the `remote_root` URL scheme: a bare path (or
`file://`) — or leaving `remote_root` unset — keeps checkpoints local under
`<output_dir>/ckpts`; a remote scheme routes through
[fsspec](https://filesystem-spec.readthedocs.io) to the matching backend.
`fsspec` ships **no** cloud backend on its own, so install the one for the scheme
you use:

```bash
pip install s3fs      # for  s3://  (Amazon S3 and S3-compatible stores)
pip install ossfs     # for  oss://  (Aliyun OSS)
```

> **Baidu BOS and other S3-compatible stores** are reached through the `s3://`
> scheme (install `s3fs`), **not** a `bos://` scheme — fsspec has no `bos`
> backend. Point `checkpoint.remote_root` at the store with `s3://<bucket>/<prefix>`
> and set the custom
> endpoint. minWM auto-detects endpoint and addressing style from, in precedence
> order, the `AWS_ENDPOINT_URL_S3` / `AWS_ENDPOINT_URL` / `AWS_S3_ADDRESSING_STYLE`
> env vars, then a nested `s3 =` block under your profile in `~/.aws/credentials`
> (or `~/.aws/config`). Access key / secret are read by botocore from the same
> profile — never hardcoded, never printed. For example, a `~/.aws/credentials`
> profile for BOS Beijing:
>
> ```ini
> [default]
> aws_access_key_id = <your-key>
> aws_secret_access_key = <your-secret>
> s3 =
>     endpoint_url = http://s3.bj.bcebos.com
>     addressing_style = virtual
> ```
>
> To smoke-test the full save → resume stack (single-file, sync DCP, async DCP)
> against a real bucket, run `python tools/checkpoint_s3_e2e.py --base
> s3://<bucket>/<prefix>` (needs live credentials and a direct connection — do
> not set `HTTPS_PROXY` for object-store traffic).

## Verify Installation

From the repo root, with the env activated and `minwm` installed editable:

```bash
# Torch sees CUDA (prints True on a GPU box with a new-enough driver)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# flash-attn imported its compiled CUDA extension
python -c "import flash_attn; print(flash_attn.__version__)"

# The framework package resolves (installed editable via `pip install -e .`)
python -c "import minwm; print('minwm OK')"

# The training entry point parses (prints usage, then exits)
python tools/train_mwm.py --help
```

Expected: `2.9.1+... True`, a flash-attn version, `minwm OK`, and the
`train_mwm.py` usage banner. If any step fails, jump to
[Common Installation Issues](#common-installation-issues).

## Developer Setup

Contributing to `minwm/` (or `tests/`) means matching what CI enforces. CI runs
two jobs on **Python 3.12**: a **lint** job and a **CPU-only test** job. You can
reproduce both locally without a GPU.

**Lint** — black is the source of truth for formatting; run it first, then
isort and flake8. If you already have the full env, the tools come with
`requirements/dev.txt`:

```bash
pip install -r requirements/dev.txt   # base env + black/isort/flake8/pytest

black --check minwm tests     # drop --check to auto-format
isort --check-only minwm tests
flake8 minwm tests
```

Config lives in `[pyproject.toml](pyproject.toml)` (black/isort, line length
**100**) and `[.flake8](.flake8)`. Legacy trees (`HY15/`, `Wan21/`, `shared/`)
are excluded until migrated.

**Tests (no GPU needed)** — the suite uses only basic tensor/nn APIs, so you can
skip the full env entirely and install a CPU build of torch plus a handful of
deps (this is exactly what CI does); GPU-only cases are gated behind
`@cuda_only` and skip on CPU:

```bash
# CPU-only torch (the +cpu wheel lives on the PyTorch index, not PyPI)
pip install "torch==2.6.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
pip install "numpy>=1.26" "termcolor==3.3.0" "omegaconf==2.3.0" \
    "pytest>=8.0" "einops==0.8.2" "diffusers==0.35.0"
pip install -e . --no-deps   # register minwm without pulling any deps

pytest tests/ -q --tb=short
```

> You don't need the full env (or flash-attn) just to lint and run unit tests —
> the CPU torch + the handful of deps above is what CI uses.

## Common Installation Issues

`ModuleNotFoundError: No module named 'minwm'`

`minwm` isn't installed in the active env — you skipped `pip install -e .`, or
you're in a different env than the one you installed into. Install it editable
from the repo root:

```bash
pip install -e .          # or: pip install -e . --no-deps
```

`python tools/train_mwm.py` puts `tools/` on `sys.path[0]` — **not** the repo
root — and the entry script deliberately does not patch `sys.path`, so it relies
on `minwm` being installed in the env (editable). Confirm with
`python -c "import minwm; print(minwm.__file__)"` — it should point into your
source tree.



`torch.cuda.is_available()` returns `False` / training silently runs on CPU

Usually the NVIDIA **driver is too old** for the pinned CUDA 12.8 wheels. Torch
then disables CUDA and falls back to CPU instead of erroring. Look for:

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
```

Fix: upgrade the driver to one that supports CUDA 12.8, or run on a host/image
whose driver is new enough. Confirm with `nvidia-smi` (CUDA Version field) before
launching a long job — a silent CPU fallback wastes an entire run.



`flash-attn` fails to build / install

flash-attn compiles a CUDA extension at install time, so:

- Install **after** torch (it reads the installed torch to pick the build), and
pass `--no-build-isolation` so it sees that torch.
- **ninja** must be present (`requirements/base.txt` pins it) — without it the
build is extremely slow or fails.
- A CUDA toolchain compatible with torch's CUDA (12.8) must be available.

```bash
pip install ninja
pip install flash-attn --no-build-isolation
```

On **Python 3.12** you can skip the source build entirely: a prebuilt wheel for
torch 2.9 / CUDA 12 / cp312 exists upstream and installs in seconds. (There is
no equivalent cp311 wheel for torch 2.9, which is why 3.12 is recommended.)

```bash
pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
```



`import minwm` works locally but fails under `torchrun` or a cluster launcher

This is rare once `minwm` is installed editable (`pip install -e .`) — the
package lives in the env's `site-packages`, so any process using that env's
interpreter finds it, including `torchrun` and wrapped cluster jobs. If it still
fails, the launcher is using a **different env/interpreter** than the one you
installed into. Make sure the job activates the same conda env, then re-run
`pip install -e .` in it if unsure.

Note this only concerns the `minwm` package. The legacy `HY15/`/`Wan21/`
run-scripts are not affected: they self-set their own `PYTHONPATH` (e.g.
`export PYTHONPATH="$PROJECT_ROOT/HY15:$PROJECT_ROOT/shared:$PYTHONPATH"`) and
need no manual export.

