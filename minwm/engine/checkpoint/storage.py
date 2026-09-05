"""Storage backend for checkpoints: local filesystem or remote object store.

Where a checkpoint lives is decided entirely by its ``save_dir`` URL scheme: a
bare path (or ``file://``) is local; ``s3://`` / ``oss://`` (and other fsspec
schemes) route to the matching fsspec backend (``s3fs`` / ``ossfs``). This is
the single seam that resolves "where does this checkpoint live" — the
Checkpointer and the DCP layer above it stay backend-agnostic.

The backend is reached through fsspec, so the whole surface is a thin wrapper
around :func:`fsspec.core.url_to_fs` plus a few checkpoint-shaped helpers
(atomic pointer write, download-to-local for ``torch.load``). ``fsspec`` is the
only hard dependency; the concrete remote backends (``s3fs`` for ``s3://``,
``ossfs`` for ``oss://``) are imported lazily by fsspec the first time a remote
URL is used, so importing this module never forces an optional dependency.

Example::

    storage = Storage(client_kwargs={"endpoint_url": "http://s3.bj.bcebos.com"})
    storage.makedirs("s3://bucket/run/ckpts")
    storage.write_text("s3://bucket/run/ckpts/latest.txt", "checkpoint_1000")
    local = storage.get_local_path("s3://bucket/run/ckpts/checkpoint_1000.pt")
"""

import configparser
import hashlib
import os
import tempfile
from pathlib import Path
from typing import IO, Any

_LOCAL_PROTOCOLS = ("file", "local")
_S3_PROTOCOLS = ("s3", "s3a")


def _read_aws_s3_block(profile: str) -> dict[str, str]:
    """Read a profile's nested ``s3 =`` block from ``~/.aws/{config,credentials}``.

    botocore stores a custom ``endpoint_url`` / ``addressing_style`` in an ``s3``
    sub-block but does **not** apply it to the S3 client automatically, so we
    parse it out ourselves. Access key / secret are intentionally ignored here —
    botocore reads those for the selected profile on its own.

    Args:
        profile (str): AWS profile name (``"default"`` etc.).

    Returns:
        dict[str, str]: keys found in the ``s3`` block (e.g. ``endpoint_url``,
        ``addressing_style``); empty when absent.
    """
    out: dict[str, str] = {}
    files = (
        ("config", "default" if profile == "default" else f"profile {profile}"),
        ("credentials", profile),
    )
    for fname, section in files:
        path = Path.home() / ".aws" / fname
        if not path.exists():
            continue
        parser = configparser.ConfigParser()
        try:
            parser.read(path)
        except configparser.Error:
            continue
        if section not in parser:
            continue
        raw = parser[section].get("s3")
        if not raw:
            continue
        for line in raw.splitlines():
            name, sep, value = line.partition("=")
            if sep:
                out[name.strip()] = value.strip()
    return out


def detect_s3_options(profile: str | None = None) -> dict[str, Any]:
    """Discover S3 client options (endpoint / addressing) from env and ``~/.aws``.

    Resolves the bits botocore won't apply on its own — a custom ``endpoint_url``
    and ``addressing_style`` (e.g. for Baidu BOS) — from, in precedence order,
    the ``AWS_ENDPOINT_URL_S3`` / ``AWS_ENDPOINT_URL`` /
    ``AWS_S3_ADDRESSING_STYLE`` env vars, then the profile's ``s3`` block in
    ``~/.aws/config`` / ``~/.aws/credentials``. **No credentials are embedded**:
    the returned ``profile`` tells fsspec/botocore which ``~/.aws/credentials``
    entry to read the access key / secret from.

    Args:
        profile (str | None): AWS profile; defaults to ``$AWS_PROFILE`` or
            ``"default"``.

    Returns:
        dict[str, Any]: fsspec ``storage_options`` (``profile`` plus, when found,
        ``client_kwargs``/``config_kwargs``). Never contains secrets.
    """
    profile = profile or os.environ.get("AWS_PROFILE") or "default"
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    addressing = os.environ.get("AWS_S3_ADDRESSING_STYLE")
    if endpoint is None or addressing is None:
        block = _read_aws_s3_block(profile)
        endpoint = endpoint or block.get("endpoint_url")
        addressing = addressing or block.get("addressing_style")
    options: dict[str, Any] = {"profile": profile}
    if endpoint:
        options["client_kwargs"] = {"endpoint_url": endpoint}
    if addressing:
        options["config_kwargs"] = {"s3": {"addressing_style": addressing}}
    return options


def _protocol(path: str | os.PathLike) -> str:
    """Return the fsspec protocol/scheme of ``path`` (``"file"`` for bare paths)."""
    from fsspec.utils import get_protocol

    return get_protocol(str(path))


def is_remote(path: str | os.PathLike) -> bool:
    """True when ``path`` targets a remote object store (a non-local scheme).

    Args:
        path (str | os.PathLike): a filesystem path or a scheme URL.

    Returns:
        bool: ``True`` for ``s3://`` / ``oss://`` / ``memory://`` and similar,
        ``False`` for bare local paths and ``file://`` URLs.
    """
    return _protocol(path) not in _LOCAL_PROTOCOLS


def join(base: str, *parts: str) -> str:
    """Join URL / path segments with ``/`` (object-store and POSIX safe).

    Object stores and POSIX paths both use ``/`` separators, so a plain
    ``rstrip`` + join keeps a scheme prefix (``s3://bucket``) intact where
    ``os.path.join`` would not.

    Args:
        base (str): the leading path or URL (scheme prefix preserved).
        *parts (str): trailing segments to append.

    Returns:
        str: the joined path.
    """
    out = base.rstrip("/")
    for part in parts:
        out = out + "/" + str(part).strip("/")
    return out


class Storage:
    """A filesystem-or-object-store handle bound to one set of backend options.

    The concrete backend is resolved per call from each path's URL scheme via
    fsspec. ``storage_options`` (e.g. ``client_kwargs={"endpoint_url": ...}``,
    ``profile``, ``config_kwargs``) are applied **only** to remote paths, so the
    same handle transparently serves local paths (which take no options) and
    remote URLs alike.

    Args:
        profile (str | None): AWS profile for S3 auto-detection; defaults to
            ``$AWS_PROFILE`` / ``"default"``.
        auto_detect (bool): for ``s3://`` paths, auto-resolve endpoint /
            addressing / profile from env and ``~/.aws`` (see
            :func:`detect_s3_options`) when not given explicitly. Nothing is read
            from disk until an S3 path is actually used.
        **storage_options (Any): fsspec backend options forwarded to remote
            filesystems (ignored for local paths); take precedence over any
            auto-detected values.
    """

    def __init__(
        self,
        *,
        profile: str | None = None,
        auto_detect: bool = True,
        **storage_options: Any,
    ) -> None:
        self.profile = profile
        self.auto_detect = auto_detect
        self.storage_options = storage_options
        self._detected: dict[str, Any] | None = None

    def options_for(self, path: str | os.PathLike) -> dict[str, Any]:
        """Resolve the fsspec ``storage_options`` to use for ``path``.

        Local paths take none; ``s3://`` paths get auto-detected endpoint /
        credentials-profile (overlaid by any explicit ``storage_options``) when
        ``auto_detect`` is on; other remote schemes get ``storage_options`` as-is.
        """
        proto = _protocol(path)
        if proto in _LOCAL_PROTOCOLS:
            return {}
        if proto in _S3_PROTOCOLS and self.auto_detect:
            if self._detected is None:
                self._detected = detect_s3_options(self.profile)
            return {**self._detected, **self.storage_options}
        return self.storage_options

    def _fs_path(self, path: str | os.PathLike):
        """Return the ``(fsspec filesystem, inner path)`` pair for ``path``."""
        import fsspec

        return fsspec.core.url_to_fs(str(path), **self.options_for(path))

    def validate_access(self, path: str | os.PathLike) -> None:
        """Probe ``path`` so a missing endpoint / credential fails fast and clear.

        Args:
            path (str | os.PathLike): a bucket/prefix or URL to reach.

        Raises:
            RuntimeError: if the store cannot be reached or authenticated,
                annotated with the resolved profile / endpoint (never secrets).
        """
        try:
            self.exists(path)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            opts = self.options_for(path)
            endpoint = opts.get("client_kwargs", {}).get("endpoint_url", "<default>")
            raise RuntimeError(
                f"cannot access {str(path)!r} (profile={opts.get('profile', '<none>')}, "
                f"endpoint={endpoint}): {exc}. Check that ~/.aws/credentials has the "
                "profile with a nested 's3' endpoint_url block, or set AWS_ENDPOINT_URL."
            ) from exc

    def exists(self, path: str | os.PathLike) -> bool:
        """True if ``path`` exists on its backend."""
        fs, inner = self._fs_path(path)
        return bool(fs.exists(inner))

    def makedirs(self, path: str | os.PathLike) -> None:
        """Create ``path`` and any missing parents (no-op if it exists).

        Object stores have no real directories; the fsspec backend treats this
        as a prefix and most implementations make it a cheap no-op.
        """
        fs, inner = self._fs_path(path)
        fs.makedirs(inner, exist_ok=True)

    def listdir(self, path: str | os.PathLike) -> list[str]:
        """List the base names directly under ``path`` (non-recursive)."""
        fs, inner = self._fs_path(path)
        return [entry.rstrip("/").rsplit("/", 1)[-1] for entry in fs.ls(inner, detail=False)]

    def remove(self, path: str | os.PathLike, *, recursive: bool = False) -> None:
        """Delete ``path`` (a file, or a prefix/dir when ``recursive``)."""
        fs, inner = self._fs_path(path)
        fs.rm(inner, recursive=recursive)

    def open(self, path: str | os.PathLike, mode: str = "rb") -> IO[Any]:
        """Open ``path`` for reading/writing, returning a file-like object.

        Args:
            path (str | os.PathLike): the file to open.
            mode (str): a standard file mode (e.g. ``"rb"``, ``"w"``).

        Returns:
            IO[Any]: an open file-like object (use as a context manager).
        """
        fs, inner = self._fs_path(path)
        return fs.open(inner, mode)

    def write_text(self, path: str | os.PathLike, text: str) -> None:
        """Write ``text`` to ``path`` as atomically as the backend allows.

        Local writes go to a ``.tmp`` sibling swapped in with :func:`os.replace`
        (atomic rename). On object stores a single PUT is itself atomic, so the
        object is written directly. Either way a reader never observes a
        half-written pointer — the property the resume pointer relies on.

        Args:
            path (str | os.PathLike): destination file.
            text (str): contents to write (UTF-8).
        """
        fs, inner = self._fs_path(path)
        if is_remote(path):
            with fs.open(inner, "w") as f:
                f.write(text)
            return
        tmp = f"{inner}.tmp"
        with fs.open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, inner)

    def read_text(self, path: str | os.PathLike) -> str:
        """Read and return the UTF-8 contents of ``path``."""
        fs, inner = self._fs_path(path)
        with fs.open(inner, "r") as f:
            return f.read()

    def get_local_path(self, path: str | os.PathLike, *, cache_dir: str | None = None) -> str:
        """Return a local filesystem path for ``path``, downloading if remote.

        Local paths are returned unchanged. A remote file is downloaded into a
        cache keyed by the full URL and reused on subsequent calls. The cache is
        keyed by URL only, not by content, so a hit is reused *without*
        re-checking the remote — this relies on the caller's URLs being
        effectively immutable. That holds for minWM checkpoints (each step writes
        a fresh ``checkpoint_<step>`` name and never rewrites an existing one),
        but it means overwriting a file at the same URL will not be picked up
        until the cache entry is removed. Use this to feed a remote single-file
        checkpoint to ``torch.load``, which needs a seekable local file.

        Args:
            path (str | os.PathLike): the file to localize.
            cache_dir (str | None): directory for downloaded copies; defaults to
                ``$TMPDIR/minwm-ckpt-cache``.

        Returns:
            str: a local path pointing at the file's contents.
        """
        if not is_remote(path):
            return str(path)
        cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "minwm-ckpt-cache")
        os.makedirs(cache_dir, exist_ok=True)
        digest = hashlib.sha1(str(path).encode()).hexdigest()[:16]
        base = str(path).rstrip("/").rsplit("/", 1)[-1]
        local = os.path.join(cache_dir, f"{digest}_{base}")
        if not os.path.exists(local):
            fs, inner = self._fs_path(path)
            tmp = f"{local}.partial"
            fs.get(inner, tmp)
            os.replace(tmp, local)
        return local


def get_storage(**storage_options: Any) -> Storage:
    """Build a :class:`Storage` handle with the given backend options."""
    return Storage(**storage_options)
