"""Checkpointer: one owner for saving and loading minWM checkpoints.

Detectron2-shaped (``save`` / ``load`` / ``resume_or_load`` / ``has_checkpoint``
/ ``tag_last_checkpoint``), adapted to minWM's two realities:

* **format duality** — under FSDP the state is a sharded
  :mod:`torch.distributed.checkpoint` (DCP) directory; single-process runs write
  one ``.pt`` file. The ``sharded`` flag selects the path.
* **storage duality** — the destination is decided by the ``save_dir`` URL
  scheme (local vs ``s3://`` / ``oss://``), resolved once by the injected
  :class:`~minwm.engine.checkpoint.storage.Storage`.

The Checkpointer operates on a mapping of **Stateful** checkpointables. The
trainer bundles model + auxiliary models + optimizers + RNG + step into a single
``app`` Stateful (the DCP-native shape), so the usual call is
``Checkpointer(save_dir, checkpointables={"app": app_state}, sharded=...)``.

Initializing a *fresh* model from a weight file (the pretrained / inference
path) is a separate, module-targeted operation — :func:`load_model_weights` —
because it loads only model weights (optionally broadcast into a sharded module)
and never touches optimizer / step / RNG state.
"""

import os
from typing import Any, Protocol, runtime_checkable

from minwm.utils import comm
from minwm.utils.logger import init_logger

from . import formats
from .storage import Storage, is_remote, join

logger = init_logger(__name__)

POINTER = "latest.txt"


def dcp_reader(path: str, storage: Storage):
    """DCP ``StorageReader`` for ``path`` — fsspec-backed when remote, else local.

    The single seam for "how do I read a DCP directory from here", shared by the
    :class:`Checkpointer` (resume / load) and the offline export tools
    (``tools/export_checkpoint.py``), so both dispatch local vs ``s3://`` /
    ``oss://`` the same way instead of each open-coding it.

    Args:
        path (str): DCP directory path or ``s3://`` / ``oss://`` URL.
        storage (Storage): backend supplying fsspec options for remote reads.

    Returns:
        A ``StorageReader``: ``FsspecReader`` for remote paths, else
        ``FileSystemReader``.
    """
    if is_remote(path):
        from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader

        return FsspecReader(path, **storage.options_for(path))
    from torch.distributed.checkpoint import FileSystemReader

    return FileSystemReader(path)


@runtime_checkable
class Stateful(Protocol):
    """Anything with ``state_dict`` / ``load_state_dict`` (duck-typed)."""

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


class Checkpointer:
    """Save/load a bundle of Stateful checkpointables to local or remote storage.

    Args:
        save_dir (str): directory (or ``s3://`` / ``oss://`` URL) that holds the
            checkpoints and the ``latest.txt`` resume pointer. Empty disables
            saving (dry runs / pytest).
        checkpointables (dict[str, Stateful] | None): named Stateful objects to
            persist and restore (e.g. ``{"app": app_state}``).
        sharded (bool): write/read a sharded DCP directory (FSDP) instead of a
            single ``.pt`` file.
        storage (Storage | None): storage backend; defaults to a local/remote
            auto-dispatching :class:`Storage` with no extra options.
        save_to_disk (bool | None): whether this process writes the single-file
            checkpoint; defaults to "main process only". Ignored for the DCP path,
            where every rank participates.
        async_save (bool): offload the DCP write to a background thread
            (:func:`torch.distributed.checkpoint.async_save`). ``save`` returns
            once state is staged to CPU; the write finishes — and the resume
            pointer is bumped — on the next ``save`` or on
            :meth:`wait_for_saves`. Only one save is in flight at a time. No
            effect on the single-file path.
        allow_partial_load (bool): when loading, allow a checkpoint that is
            missing keys present in the current checkpointables. ``False``
            (default) raises on any missing key so a stale checkpoint is never
            silently ignored. Set ``True`` only when intentionally resuming from
            an older checkpoint whose set of keys differs from the current config
            (maps to ``checkpoint.allow_partial_resume`` in the trainer config).
            For the DCP path this is forwarded to the
            :class:`~torch.distributed.checkpoint.DefaultLoadPlanner`; for the
            single-file path it suppresses the :exc:`KeyError` on a missing key.
    """

    def __init__(
        self,
        save_dir: str,
        *,
        checkpointables: dict[str, Any] | None = None,
        sharded: bool = False,
        storage: Storage | None = None,
        save_to_disk: bool | None = None,
        async_save: bool = False,
        allow_partial_load: bool = False,
    ) -> None:
        self.save_dir = str(save_dir) if save_dir else ""
        self.checkpointables: dict[str, Any] = dict(checkpointables or {})
        self.sharded = sharded
        self.storage = storage or Storage()
        self.save_to_disk = comm.is_main_process() if save_to_disk is None else save_to_disk
        self.async_save = async_save
        self.allow_partial_load = allow_partial_load
        self._pending: tuple[Any, str] | None = None

    # ---- discovery / pointer --------------------------------------------

    @property
    def _pointer_path(self) -> str:
        return join(self.save_dir, POINTER)

    def has_checkpoint(self) -> bool:
        """True if a ``latest.txt`` resume pointer exists under ``save_dir``."""
        return bool(self.save_dir) and self.storage.exists(self._pointer_path)

    def get_checkpoint_file(self) -> str:
        """Return the full path of the checkpoint ``latest.txt`` points at."""
        target = self.storage.read_text(self._pointer_path).strip()
        return join(self.save_dir, target)

    def tag_last_checkpoint(self, target_name: str) -> None:
        """Point ``latest.txt`` at ``target_name`` (rank-0 atomic write).

        Written last, after the checkpoint bytes are durable, so the pointer
        never names a half-written checkpoint.
        """
        if not comm.is_main_process():
            return
        self.storage.write_text(self._pointer_path, target_name)

    # ---- save ------------------------------------------------------------

    def save(self, name: str) -> str | None:
        """Persist all checkpointables under ``name`` and bump ``latest.txt``.

        Sharded runs write a DCP directory ``name/`` (every rank participates);
        otherwise the main process writes a single ``name.pt`` file. An empty
        ``save_dir`` is a no-op with a warning.

        Args:
            name (str): checkpoint base name (e.g. ``"checkpoint_1000"``).

        Returns:
            str | None: the on-disk target name (``name`` for DCP, ``name.pt``
            for single-file), or ``None`` when nothing was written.
        """
        if not self.save_dir:
            logger.warning("save_checkpoint: save_dir is empty; skipping save")
            return None
        self.storage.makedirs(self.save_dir)

        if self.sharded:
            target = name
            path = join(self.save_dir, target)
            if self.async_save:
                self._flush_pending()  # serialize: at most one save in flight
                self._pending = (self._dcp_async_save(path), target)
                logger.info("async save started %s", path)
                return target
            self._dcp_save(path)
        else:
            if not self.save_to_disk:
                return None
            target = f"{name}.pt"
            self._single_file_save(join(self.save_dir, target))

        self.tag_last_checkpoint(target)
        logger.info("saved checkpoint %s", join(self.save_dir, target))
        return target

    def wait_for_saves(self) -> None:
        """Block until any in-flight async save is durable and its pointer set.

        Call before process exit (and before reading the resume pointer) so an
        async DCP write is never left unfinished.
        """
        self._flush_pending()

    def _flush_pending(self) -> None:
        if self._pending is None:
            return
        future, target = self._pending
        self._pending = None
        # Catch any write failure *before* the barrier so that every rank
        # participates in the all_reduce_flag call below — if one rank's
        # future.result() raised and it skipped synchronization, the remaining
        # ranks would block in comm.synchronize() forever (a deadlock that is
        # very hard to diagnose).  CheckpointException inherits BaseException,
        # so we catch BaseException here rather than Exception.
        write_exc: BaseException | None = None
        try:
            future.result()  # this rank's shards are durable
        except BaseException as exc:  # noqa: BLE001
            write_exc = exc
        any_failed = comm.all_reduce_flag(write_exc is not None)
        if any_failed:
            if write_exc is not None:
                raise RuntimeError(
                    f"async DCP write for {join(self.save_dir, target)!r} failed "
                    f"on rank {comm.get_rank()}"
                ) from write_exc
            # Another rank failed; raise here too so all ranks exit cleanly.
            raise RuntimeError(
                f"async DCP write for {join(self.save_dir, target)!r} failed on "
                f"a peer rank; see that rank's log for the root cause"
            )
        self.tag_last_checkpoint(target)
        logger.info("async save finalized %s", join(self.save_dir, target))

    def _dcp_save(self, path: str) -> None:
        import torch.distributed.checkpoint as dcp

        dcp.save(self.checkpointables, storage_writer=self._dcp_writer(path))

    def _dcp_async_save(self, path: str):
        import torch.distributed.checkpoint as dcp

        return dcp.async_save(self.checkpointables, storage_writer=self._dcp_writer(path))

    def _single_file_save(self, path: str) -> None:
        import torch

        payload = {name: obj.state_dict() for name, obj in self.checkpointables.items()}
        if is_remote(path):
            # A single object PUT commits atomically — write straight to path.
            with self.storage.open(path, "wb") as f:
                torch.save(payload, f)
            return
        tmp = f"{path}.tmp"
        with self.storage.open(tmp, "wb") as f:
            torch.save(payload, f)
        os.replace(tmp, path)

    # ---- load ------------------------------------------------------------

    def load(self, path: str, *, checkpointables: list[str] | None = None) -> None:
        """Restore checkpointables from the checkpoint at ``path``.

        The format is sniffed from ``path`` (DCP directory vs ``.pt`` file) and
        loaded through the matching backend. Only the named ``checkpointables``
        are restored (default: all).

        A checkpoint that is missing a registered checkpointable fails loudly
        (rather than silently leaving it at its freshly-built state) unless
        :attr:`allow_partial_load` is set — the same behavior on both the DCP and
        single-file paths. For DCP the check is enforced by the load planner
        (``allow_partial_load``); for single-file it is enforced here, because a
        missing top-level key would otherwise be a no-op.

        Args:
            path (str): checkpoint directory or file path/URL.
            checkpointables (list[str] | None): subset of names to restore;
                ``None`` restores every registered checkpointable.

        Raises:
            ValueError: if ``path`` is neither a DCP directory nor a ``.pt`` file.
            KeyError: if a single-file checkpoint lacks a registered
                checkpointable and :attr:`allow_partial_load` is ``False``.
        """
        if not path:
            logger.warning("load: empty path; nothing to restore")
            return
        self._flush_pending()
        which = (
            self.checkpointables
            if checkpointables is None
            else {name: self.checkpointables[name] for name in checkpointables}
        )
        fmt = formats.detect_format(path, self.storage)
        if fmt == formats.DCP_DIR:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

            dcp.load(
                which,
                storage_reader=self._dcp_reader(path),
                planner=DefaultLoadPlanner(allow_partial_load=self.allow_partial_load),
            )
        elif fmt == formats.TORCH:
            import torch

            local = self.storage.get_local_path(path)
            state = torch.load(local, map_location="cpu", weights_only=False)
            for name, obj in which.items():
                if name not in state:
                    if self.allow_partial_load:
                        logger.warning("checkpoint %s has no '%s'; leaving it as built", path, name)
                        continue
                    raise KeyError(
                        f"checkpoint {path!r} has no '{name}' but it is a registered "
                        f"checkpointable. Refusing to silently skip it (that would leave "
                        f"it at its freshly-built state). Set allow_partial_load=True to "
                        f"opt into a partial load."
                    )
                obj.load_state_dict(state[name])
        else:
            raise ValueError(f"cannot load checkpoint {path!r}: unsupported format {fmt!r}")
        logger.info("loaded checkpoint %s", path)

    def resume_or_load(self, path: str | None = None, *, resume: bool = True) -> str | None:
        """Resume from the latest checkpoint if available, else load ``path``.

        With ``resume`` set and a ``latest.txt`` present, restores the latest
        checkpoint (full state) and returns its path. Otherwise, if ``path`` is
        given, loads it and returns it. Returns ``None`` when there is nothing to
        restore — the caller then starts fresh (and may separately initialize
        model weights via :func:`load_model_weights`).

        Args:
            path (str | None): explicit checkpoint to load when not resuming.
            resume (bool): prefer the auto-discovered latest checkpoint.

        Returns:
            str | None: the path restored from, or ``None``.
        """
        self._flush_pending()
        if resume and self.has_checkpoint():
            target = self.get_checkpoint_file()
            self.load(target)
            return target
        if path:
            self.load(path)
            return path
        return None

    # ---- DCP storage plumbing -------------------------------------------

    def _dcp_writer(self, path: str):
        """DCP ``StorageWriter`` for ``path`` (fsspec-backed when remote)."""
        if is_remote(path):
            from torch.distributed.checkpoint._fsspec_filesystem import FsspecWriter

            # Object-store streams have no OS file descriptor, so DCP's post-write
            # ``os.fsync(stream.fileno())`` raises ``io.UnsupportedOperation`` on
            # torch builds that don't guard it. A PUT is durable on close, so
            # disable fsync for remote writers (also avoids a needless os.sync()).
            return FsspecWriter(path, sync_files=False, **self.storage.options_for(path))
        from torch.distributed.checkpoint import FileSystemWriter

        return FileSystemWriter(path)

    def _dcp_reader(self, path: str):
        """DCP ``StorageReader`` for ``path`` (fsspec-backed when remote)."""
        return dcp_reader(path, self.storage)


def load_model_weights(
    module: Any,
    path: str,
    *,
    storage: Storage | None = None,
    sharded: bool = False,
    select: Any = None,
    key: str = "auto",
    prefer_ema: bool = False,
    strict: bool = True,
) -> None:
    """Load model weights from a file or Diffusers directory into ``module``.

    The weights-only initialization path shared by training (``pretrained`` /
    ``auxiliary_pretrained``) and inference. Reads and normalizes weights via
    :func:`~minwm.engine.checkpoint.formats.read_model_state`, then loads them: under
    ``sharded`` a full state dict is broadcast from rank 0 and redistributed into
    the sharded module; otherwise a plain ``load_state_dict``.

    Args:
        module (nn.Module): destination module (primary model or an auxiliary).
        path (str): checkpoint file or Diffusers model directory path/URL.
        storage (Storage | None): storage backend (defaults to auto-dispatch).
        sharded (bool): redistribute a full state dict into an FSDP module.
        select (Callable[[dict], dict] | None): weight-selection policy; ``None``
            uses the general ``select_state_dict`` with ``key`` / ``prefer_ema``.
        key (str): explicit wrapper key or ``"auto"`` (default selector only).
        prefer_ema (bool): prefer ``generator_ema`` in auto mode (default selector).
        strict (bool): strict key matching (non-sharded path only).
    """
    storage = storage or Storage()
    weights = formats.read_model_state(path, storage, select=select, key=key, prefer_ema=prefer_ema)
    if sharded:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        set_model_state_dict(
            module,
            weights,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
    else:
        module.load_state_dict(weights, strict=strict)
