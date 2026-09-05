"""Tests for minwm.engine.paths — run-output path resolution."""

import pytest

from minwm.engine.paths import ckpt_dir, local_run_dir, log_dir


@pytest.mark.parametrize(
    "output_dir, expected",
    [
        ("outputs/exp", "outputs/exp"),
        ("logs/wan21/stage0", "logs/wan21/stage0"),
        ("", None),
        (None, None),
    ],
)
def test_local_run_dir(output_dir, expected):
    assert local_run_dir(output_dir) == expected


@pytest.mark.parametrize(
    "output_dir, expected",
    [
        ("outputs/exp", "outputs/exp/logs"),
        ("", None),
        (None, None),
    ],
)
def test_log_dir(output_dir, expected):
    assert log_dir(output_dir) == expected


@pytest.mark.parametrize(
    "output_dir, remote_root, expected",
    [
        # local: no remote_root -> under output_dir
        ("outputs/exp", None, "outputs/exp/ckpts"),
        # remote_root set -> prefixed onto the scheme-free run name
        ("outputs/exp", "s3://bucket/runs", "s3://bucket/runs/outputs/exp/ckpts"),
        ("wan21-t2v/stage0", "oss://b/p", "oss://b/p/wan21-t2v/stage0/ckpts"),
        # trailing slash on the prefix stays clean (no s3:/ junk, no double slash)
        ("exp", "s3://bucket/", "s3://bucket/exp/ckpts"),
        # no output_dir -> save-disabled, regardless of remote_root
        ("", "s3://bucket/runs", ""),
        (None, None, ""),
    ],
)
def test_ckpt_dir(output_dir, remote_root, expected):
    assert ckpt_dir(output_dir, remote_root) == expected


def test_remote_root_never_leaks_into_logs():
    """A remote checkpoint prefix must not affect the local log/run dir."""
    output_dir = "outputs/exp"
    assert local_run_dir(output_dir) == output_dir
    assert log_dir(output_dir) == "outputs/exp/logs"
    assert ckpt_dir(output_dir, "s3://bucket/runs").startswith("s3://")
