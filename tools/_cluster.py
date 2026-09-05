"""Shared cluster-submission helpers for the auto_dump / auto_sample tools."""

import hashlib
import os
import re
import sys

MAX_JOB_NAME_LEN = 30

# Name of the cluster job-submission binary. This repo does not ship one: the
# ``<launch-tool>`` default is a placeholder for *your* cluster's submit command
# (Slurm ``sbatch``, a k8s wrapper, an in-house tool, ...). The submit invocation
# built below — ``<launch-tool> submit -j <name> -n <nodes> [-c ...] [-q ...]
# --no-log -- <script> <args>`` — is likewise an example of one such tool's CLI;
# adapt it (or set ``MINWM_LAUNCH_TOOL``) to whatever your scheduler expects.
# Use ``--local`` to skip cluster submission entirely.
LAUNCH_TOOL_PLACEHOLDER = "<launch-tool>"
LAUNCH_TOOL = os.environ.get("MINWM_LAUNCH_TOOL", LAUNCH_TOOL_PLACEHOLDER)


def build_submit_argv(
    job_name: str, nodes: int, cluster: str | None, queue: str | None
) -> list[str]:
    """Build the cluster-submit ``argv`` prefix (before ``-- <script> ...``).

    Bails out with an actionable message if the launch tool is still the
    ``<launch-tool>`` placeholder, so the tools never silently ``exec`` a
    command that does not exist. Set ``MINWM_LAUNCH_TOOL`` to your cluster's
    submit binary, edit this helper to match its CLI, or pass ``--local``.

    Args:
        job_name (str): cluster job name.
        nodes (int): node count for the job.
        cluster (str | None): cluster target, appended as ``-c`` when set.
        queue (str | None): queue name, appended as ``-q`` when set.

    Returns:
        list[str]: argv from the tool name through ``--no-log`` (caller appends
        ``["--", script, *args]``).

    Raises:
        SystemExit: if the launch tool is unset (still the placeholder).
    """
    if LAUNCH_TOOL == LAUNCH_TOOL_PLACEHOLDER:
        print(
            f"[cluster] {LAUNCH_TOOL_PLACEHOLDER} is a placeholder for your cluster's "
            "job-submission tool. Set MINWM_LAUNCH_TOOL to its binary (and adapt the "
            "submit CLI in tools/_cluster.py to match), or pass --local to run without "
            "a cluster.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    submit = [LAUNCH_TOOL, "submit", "-j", job_name, "-n", str(nodes)]
    if cluster:
        submit += ["-c", cluster]
    if queue:
        submit += ["-q", queue]
    submit += ["--no-log"]
    return submit


def build_safe_job_name(exp_name: str, step: int, kind: str) -> str:
    """Build an ``<=30``-char cluster-safe job name, hashing when it's too long.

    Produces ``<exp>-<kind>-<step>`` lowercased with non-alphanumerics collapsed
    to single dashes. If that exceeds :data:`MAX_JOB_NAME_LEN`, the exp name is
    truncated and a short hash is inserted so the name stays unique and ends with
    the ``<kind>-<step>`` suffix.

    Args:
        exp_name (str): experiment / run name (basename of the output dir).
        step (int): checkpoint step.
        kind (str): job kind, e.g. ``"dump"`` or ``"sample"``.

    Returns:
        str: a sanitized job name no longer than :data:`MAX_JOB_NAME_LEN`.
    """
    raw = f"{exp_name}-{kind}-{step}".lower()
    safe = re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]", "-", raw)).strip("-") or kind
    if len(safe) <= MAX_JOB_NAME_LEN:
        return safe
    suffix = f"{kind}-{step}"
    name_hash = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:6]
    budget = MAX_JOB_NAME_LEN - len(suffix) - len(name_hash) - 2
    safe_exp = re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]", "-", exp_name.lower())).strip("-")
    return f"{safe_exp[:budget].rstrip('-')}-{name_hash}-{suffix}"
