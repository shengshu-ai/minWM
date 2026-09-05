"""Run-output path resolution: where a run's logs and checkpoints live.

One seam decides the two output roots of a training run from ``training.output_dir``
plus ``checkpoint.remote_root``:

* **local run dir** — logs, TensorBoard, and the ``wandb/`` run dir. These only
  work on a local filesystem, so ``output_dir`` is always a local relative path
  and doubles as the local run dir; ``remote_root`` never touches it.
* **checkpoint dir** — the ``ckpts/`` tree plus its ``latest.txt`` pointer. Local
  by default (``<output_dir>/ckpts``); when ``checkpoint.remote_root`` is set the
  checkpoints route to ``<remote_root>/<output_dir>/ckpts`` on the object store
  the prefix's URL scheme names, while logs stay local.

Keeping ``output_dir`` a scheme-free run name is what lets both roots share it
without a remote URL leaking into a local-only writer (TensorBoard / ``log.txt``).
"""

from .checkpoint.storage import join

CKPT_SUBDIR = "ckpts"
LOGS_SUBDIR = "logs"


def local_run_dir(output_dir: str | None) -> str | None:
    """Local base for a run's logs / TensorBoard / ``wandb`` dir.

    Args:
        output_dir (str | None): the run's local ``training.output_dir``.

    Returns:
        str | None: ``output_dir`` itself (it is always local), or ``None`` when
        unset — the caller then disables file logging / metric writers.
    """
    return output_dir or None


def log_dir(output_dir: str | None) -> str | None:
    """The ``logs/`` subdirectory of the local run dir for the framework ``log.txt``.

    Args:
        output_dir (str | None): the run's local ``training.output_dir``.

    Returns:
        str | None: ``<output_dir>/logs``, or ``None`` when ``output_dir`` is unset.
    """
    base = local_run_dir(output_dir)
    return join(base, LOGS_SUBDIR) if base else None


def ckpt_dir(output_dir: str | None, remote_root: str | None = None) -> str:
    """The ``ckpts/`` directory holding checkpoints and the ``latest.txt`` pointer.

    Args:
        output_dir (str | None): the run's local ``training.output_dir``.
        remote_root (str | None): optional object-store prefix (``s3://`` /
            ``oss://``); when set, checkpoints route to
            ``<remote_root>/<output_dir>/ckpts`` instead of the local tree.

    Returns:
        str: ``<remote_root>/<output_dir>/ckpts`` when ``remote_root`` is set,
        else ``<output_dir>/ckpts``; ``""`` when ``output_dir`` is unset, which
        disables saving (dry runs / pytest).
    """
    if not output_dir:
        return ""
    if remote_root:
        return join(remote_root, output_dir, CKPT_SUBDIR)
    return join(output_dir, CKPT_SUBDIR)
