"""Tests for minwm.engine.checkpoint.storage.

The core operations are exercised backend-agnostically over two roots: a local
temp dir and an fsspec ``memory://`` filesystem (a built-in backend, no extra
install), which stands in for a remote object store — same code path as
``s3://`` / ``oss://``. A real object-store round-trip runs only when
``MINWM_TEST_S3_URL`` is set, so CI stays credential-free.
"""

import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minwm.engine.checkpoint.storage import Storage, detect_s3_options, is_remote, join


def test_is_remote_classification():
    assert is_remote("s3://bucket/key")
    assert is_remote("oss://bucket/key")
    assert is_remote("memory://run/ckpts")
    assert not is_remote("/tmp/run/ckpts")
    assert not is_remote("relative/path")
    assert not is_remote("file:///tmp/run")


def test_join_preserves_scheme():
    assert join("s3://bucket/run", "ckpts", "latest.txt") == "s3://bucket/run/ckpts/latest.txt"
    assert join("s3://bucket/run/", "/ckpts/") == "s3://bucket/run/ckpts"
    assert join("/tmp/run", "ckpts") == "/tmp/run/ckpts"


# --- S3 credential / endpoint auto-detection (hermetic) --------------------


def test_detect_s3_options_from_env(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "myprof")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://s3.example.com")
    monkeypatch.setenv("AWS_S3_ADDRESSING_STYLE", "path")
    opts = detect_s3_options()
    assert opts["profile"] == "myprof"
    assert opts["client_kwargs"] == {"endpoint_url": "http://s3.example.com"}
    assert opts["config_kwargs"] == {"s3": {"addressing_style": "path"}}


def test_detect_s3_options_env_endpoint_url_s3_wins(monkeypatch):
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://s3-specific.example.com")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://generic.example.com")
    opts = detect_s3_options()
    assert opts["profile"] == "default"
    assert opts["client_kwargs"]["endpoint_url"] == "http://s3-specific.example.com"


def test_options_for_scheme_dispatch(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://s3.example.com")
    st = Storage()
    # Local and non-S3 remote schemes never trigger S3 detection.
    assert st.options_for("/tmp/run") == {}
    assert st.options_for("memory://run") == {}
    # S3 paths auto-resolve the endpoint from the environment.
    s3_opts = st.options_for("s3://bucket/key")
    assert s3_opts["client_kwargs"]["endpoint_url"] == "http://s3.example.com"


def test_explicit_options_override_autodetect(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://auto.example.com")
    st = Storage(client_kwargs={"endpoint_url": "http://explicit.example.com"})
    assert (
        st.options_for("s3://b/k")["client_kwargs"]["endpoint_url"] == "http://explicit.example.com"
    )


def test_auto_detect_disabled(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://auto.example.com")
    assert Storage(auto_detect=False).options_for("s3://b/k") == {}


@pytest.fixture(params=["local", "memory"])
def root(request, tmp_path):
    """Yield a fresh (Storage, base_url) pair for each backend under test."""
    if request.param == "local":
        yield Storage(), str(tmp_path / "run")
    else:
        base = f"memory://mwm-test-{uuid.uuid4().hex[:8]}"
        st = Storage()
        yield st, base
        try:
            st.remove(base, recursive=True)
        except FileNotFoundError:
            pass


def test_makedirs_and_exists(root):
    st, base = root
    ckpts = join(base, "ckpts")
    assert not st.exists(ckpts)
    st.makedirs(ckpts)
    assert st.exists(ckpts)


def test_write_text_is_atomic_and_reads_back(root):
    st, base = root
    st.makedirs(base)
    ptr = join(base, "latest.txt")
    st.write_text(ptr, "checkpoint_1000")
    assert st.read_text(ptr) == "checkpoint_1000"
    # Overwrite must fully replace, never leave a half state or a .tmp sibling.
    st.write_text(ptr, "checkpoint_2000")
    assert st.read_text(ptr) == "checkpoint_2000"
    assert not st.exists(join(base, "latest.txt.tmp"))


def test_open_binary_roundtrip(root):
    st, base = root
    st.makedirs(base)
    blob = join(base, "blob.bin")
    payload = b"\x00\x01minWM\xff"
    with st.open(blob, "wb") as f:
        f.write(payload)
    with st.open(blob, "rb") as f:
        assert f.read() == payload


def test_listdir_and_remove(root):
    st, base = root
    st.makedirs(base)
    st.write_text(join(base, "a.txt"), "a")
    st.write_text(join(base, "b.txt"), "b")
    assert set(st.listdir(base)) == {"a.txt", "b.txt"}
    st.remove(join(base, "a.txt"))
    assert set(st.listdir(base)) == {"b.txt"}


def test_get_local_path_localizes_remote(root, tmp_path):
    st, base = root
    st.makedirs(base)
    blob = join(base, "weights.pt")
    with st.open(blob, "wb") as f:
        f.write(b"payload-bytes")

    cache = str(tmp_path / "cache")
    local = st.get_local_path(blob, cache_dir=cache)

    assert os.path.isfile(local)
    with open(local, "rb") as f:
        assert f.read() == b"payload-bytes"

    if is_remote(blob):
        # Remote is copied into the cache dir; local paths return unchanged.
        assert local.startswith(cache)
        assert st.get_local_path(blob, cache_dir=cache) == local  # cache hit
    else:
        assert local == blob


# --- real object-store smoke (opt-in) --------------------------------------
#
# Credentials + endpoint are auto-detected from ~/.aws (see detect_s3_options),
# so only the bucket/prefix is needed:
#   MINWM_TEST_S3_URL=s3://<bucket>/<prefix>/checkpoints \
#   pytest tests/engine/checkpoint/test_storage.py -k real_object_store -s
@pytest.mark.skipif(
    not os.environ.get("MINWM_TEST_S3_URL"),
    reason="set MINWM_TEST_S3_URL to run the real object-store smoke test",
)
def test_real_object_store_roundtrip():
    pytest.importorskip("s3fs")
    base = os.environ["MINWM_TEST_S3_URL"].rstrip("/")
    st = Storage()
    st.validate_access(base)

    root_url = join(base, f"_mwm_storage_test_{uuid.uuid4().hex[:8]}")
    try:
        st.makedirs(root_url)
        ptr = join(root_url, "latest.txt")
        st.write_text(ptr, "checkpoint_42")
        assert st.read_text(ptr) == "checkpoint_42"
        assert st.exists(ptr)

        blob = join(root_url, "weights.bin")
        with st.open(blob, "wb") as f:
            f.write(b"hello-object-store")
        local = st.get_local_path(blob)
        with open(local, "rb") as f:
            assert f.read() == b"hello-object-store"

        assert "latest.txt" in st.listdir(root_url)
    finally:
        st.remove(root_url, recursive=True)
