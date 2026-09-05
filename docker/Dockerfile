# minWM-engine runtime image.
#
# Builds *only* the environment minWM needs (the pinned torch/CUDA stack +
# flash-attn). The `minwm` package itself is intentionally NOT installed here —
# mount the repo at runtime (see PYTHONPATH below) so source edits are live and
# the image stays decoupled from the checkout.
#
# Build from the repo root (the build context must include requirements/):
#   docker build -f docker/Dockerfile -t minwm-engine:cu130 .
#
# Base: latest CUDA 13.0 + cuDNN devel on Ubuntu 24.04. Ubuntu 24.04 ships
# Python 3.12, which matches the version minWM is pinned to (INSTALL.md). Note
# the wheels in requirements/base.txt (torch 2.9.1, flash-attn) target CUDA 12.8
# and bundle their own CUDA runtime, so the base image's CUDA 13.0 userspace is
# not what torch actually links against — the host NVIDIA driver only needs to
# be new enough for CUDA 12.8. The devel image is kept so nvcc/headers are
# available if a wheel ever needs to compile a CUDA extension.

FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: Python 3.12 + venv, a build toolchain (ninja for any native
# wheel build), git, the media/IO shared libs that opencv / decord / av load at
# runtime, and a minimal interactive shell + editor (zsh, vim) so an exec'd pod
# is usable out of the box (heavier dev tools live in Dockerfile.dev).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        build-essential ninja-build \
        git wget curl ca-certificates \
        ffmpeg libgl1 libglib2.0-0 \
        zsh vim \
    && rm -rf /var/lib/apt/lists/*

# Self-contained venv on PATH: every shell in the pod (kubectl exec, the job
# command, torchrun) gets the right interpreter with no `activate` step. Docker
# ENV is inherited by `kubectl exec`, so `python`/`pip` just resolve here.
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Belt-and-suspenders for shells that re-init PATH (login shells, `su -s`, zsh):
# Docker's ENV PATH covers `kubectl exec`, but login/zsh shells can reset it, so
# also export the venv from the system shell init files.
# - /etc/profile.d/venv.sh: sourced by bash/zsh login shells
# - /etc/zsh/zshenv: sourced by zsh unconditionally (even non-login/-interactive)
RUN echo 'export PATH="/opt/venv/bin:$PATH"' > /etc/profile.d/venv.sh && \
    chmod +x /etc/profile.d/venv.sh && \
    mkdir -p /etc/zsh && \
    echo 'export PATH="/opt/venv/bin:$PATH"' > /etc/zsh/zshenv

RUN pip install --upgrade pip setuptools wheel

# Pinned training + inference stack (torch 2.9.1, torchvision 0.24.1, the
# CUDA 12.8 runtime wheels, etc). Only the requirements files are copied in —
# the repo is not — so a code change does not bust this layer's cache.
COPY requirements/ /tmp/requirements/
RUN pip install -r /tmp/requirements/base.txt \
    && rm -rf /tmp/requirements

# flash-attn: prebuilt cp312 wheel for torch 2.9 / CUDA 12 (the exact URL from
# INSTALL.md). Skips the slow source build and needs no nvcc.
RUN pip install --no-build-isolation \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# Make `import minwm` resolve when the repo is mounted at /workspace, without an
# editable install. tools/train_mwm.py puts tools/ (not the repo root) on
# sys.path, so it relies on minwm being importable — this PYTHONPATH supplies it.
WORKDIR /workspace
ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]
