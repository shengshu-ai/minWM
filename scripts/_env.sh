# Shared environment setup for minWM launchers, sourced by scripts/dump.sh and
# scripts/infer.sh. Two jobs, both of which the cluster launch tool does NOT cover:
#
#   1. PYTHONPATH — put the repo root on it so `import minwm` resolves the code
#      in this checkout. The cluster image ships third-party deps but NOT minwm
#      itself, so this is required on the cluster too, not just locally.
#   2. Interpreter resolution for LOCAL dev, which never goes through the launch tool.
#
# Third-party runtime deps (torch, flash-attn, diffusers, ...) are the cluster
# image's job: on the cluster, `conda_bin_path: /opt/venv/bin` puts the image's
# venv on PATH, so bare `python` already resolves there. The resolution order
# below matters mainly off-cluster:
#
#   1. $MINWM_PYTHON            — explicit override (any absolute path)
#   2. $CONDA_PREFIX/bin/python — an already-activated conda env
#   3. /opt/venv/bin/python     — cluster image venv; absolute-path belt-and-
#                                 suspenders for shells that reset PATH
#   4. python                   — last-resort (local dev with deps on PATH)
#
# For LOCAL dev, build your own env per INSTALL.md (`conda create -n minwm ...;
# pip install -r requirements/base.txt; pip install -e .`), then either
# `conda activate minwm` (case 2) or point $MINWM_PYTHON at it (case 1).

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Only append the existing PYTHONPATH when set, so an unset one doesn't leave a
# trailing ':' — Python reads a trailing/empty entry as the cwd and would put it
# on sys.path.
export PYTHONPATH="${_repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

# The cluster image's baked-in venv (docker/Dockerfile builds it at /opt/venv).
_MINWM_ENV_PYTHON="/opt/venv/bin/python"

resolve_python() {
    if [ -n "${MINWM_PYTHON:-}" ] && [ -x "${MINWM_PYTHON}" ]; then
        echo "${MINWM_PYTHON}"; return
    fi
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        echo "${CONDA_PREFIX}/bin/python"; return
    fi
    if [ -x "${_MINWM_ENV_PYTHON}" ]; then
        echo "${_MINWM_ENV_PYTHON}"; return
    fi
    echo "python"
}

PYTHON="$(resolve_python)"
export PYTHON
