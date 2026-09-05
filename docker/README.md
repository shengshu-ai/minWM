# Docker images for minWM-engine

Two images, both of which build **only the environment** (the pinned
torch/CUDA stack + flash-attn). The `minwm` package itself is **not** baked in —
mount the repo at runtime so source edits stay live and the image stays
decoupled from the checkout.

| File             | Image                  | Contents                                                  |
| ---------------- | ---------------------- | --------------------------------------------------------- |
| `Dockerfile`     | runtime                | `requirements/base.txt` (full train + inference) + flash-attn |
| `Dockerfile.dev` | dev (on top of runtime)| runtime + `requirements/dev.txt` (black, isort, flake8, pytest) |

The matching `*.dockerignore` files trim the build context (repo root) down to
just `requirements/`.

## Build

Build **from the repo root** — the build context must include `requirements/`,
so only the `-f` path points into this directory:

```bash
# runtime
docker build -f docker/Dockerfile -t minwm-engine:cu130 .

# dev (builds on the runtime tag, so build runtime first)
docker build -f docker/Dockerfile.dev -t minwm-engine:cu130-dev .
```

Point the dev image at a different base with
`--build-arg BASE_IMAGE=<registry>/minwm-engine:cu130`.

## Run

The repo is **not** in the image. Mount it at `/workspace` (the image sets
`WORKDIR=/workspace` and `PYTHONPATH=/workspace`, so `import minwm` resolves with
no editable install):

```bash
docker run --rm -it --gpus all -v "$PWD:/workspace" minwm-engine:cu130
```

### Sanity check

```bash
docker run --rm --gpus all -v "$PWD:/workspace" minwm-engine:cu130 \
  python -c "import torch, flash_attn, minwm; \
print(torch.__version__, torch.cuda.is_available(), flash_attn.__version__)"
```

Expect `2.9.1+... True ...` on a GPU host with a driver new enough for CUDA 12.8.

### Lint + tests (dev image)

```bash
docker run --rm -v "$PWD:/workspace" minwm-engine:cu130-dev \
  bash -lc "black --check minwm tests && isort --check-only minwm tests && \
flake8 minwm tests && pytest tests/ -q --tb=short"
```

## On a kubectl job / pod

The venv lives at `/opt/venv` and is on `PATH` via `ENV`, which `kubectl exec`
inherits — so once you exec in, `python`/`pip` already resolve to the ready env
with no `activate` step. Mount the repo at `/workspace` in the pod spec (the same
path the image expects) and `import minwm` works out of the box.

## Notes / design choices

- **Base image:** `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04`. Ubuntu 24.04
  ships **Python 3.12**, matching the version minWM is pinned to. The torch /
  flash-attn wheels target **CUDA 12.8** and bundle their own CUDA runtime, so
  the base image's CUDA 13.0 userspace is not what torch links against — only the
  **host NVIDIA driver** needs to be new enough for CUDA 12.8. The `devel` image
  is kept so `nvcc`/headers exist if a wheel ever needs to compile an extension.
- **flash-attn:** prebuilt `cp312 / torch2.9 / cu12` wheel (v2.8.3) from the URL
  in [`INSTALL.md`](../INSTALL.md) — no source build, no `nvcc` needed.
- **`.dockerignore` placement:** the per-Dockerfile `<name>.dockerignore` files
  sit next to each Dockerfile (a BuildKit convention; enabled by default in
  modern Docker) and take precedence over any root `.dockerignore`.

See [`../INSTALL.md`](../INSTALL.md) for the full (non-Docker) install and the
common-issues guide.
