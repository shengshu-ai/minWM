"""End-to-end Checkpointer round-trip against a real S3-compatible object store.

A standalone smoke test (deliberately *not* part of the pytest suite — it needs
real credentials and network) that exercises the full checkpoint stack over
``s3://``: single-file, synchronous DCP, and asynchronous DCP save → resume.

Credentials and endpoint are auto-detected from ``~/.aws`` via
:func:`minwm.engine.checkpoint.storage.detect_s3_options` — nothing is hardcoded and no
secret is printed. Object-store traffic goes direct (do not set ``HTTPS_PROXY``).

Usage::

    python tools/checkpoint_s3_e2e.py --base s3://<bucket>/<prefix>/checkpoints
    # or rely on $MINWM_TEST_S3_URL:
    MINWM_TEST_S3_URL=s3://<bucket>/<prefix>/checkpoints python tools/checkpoint_s3_e2e.py
"""

import argparse
import os
import sys
import uuid

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

from minwm.engine.checkpoint import Checkpointer
from minwm.engine.checkpoint.storage import Storage, join


class _Bundle:
    """Minimal AppState-shaped Stateful: a model plus a training step."""

    def __init__(self, model: nn.Module, step: int = 0) -> None:
        self.model = model
        self.step = step

    def state_dict(self) -> dict:
        return {"model": get_model_state_dict(self.model), "step": torch.tensor(self.step)}

    def load_state_dict(self, state_dict: dict) -> None:
        set_model_state_dict(self.model, state_dict["model"])
        self.step = int(state_dict["step"].item())


def _model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))


def _params_equal(a: nn.Module, b: nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _roundtrip(save_dir: str, storage: Storage, *, sharded: bool, async_save: bool) -> None:
    src = _model(1)
    step = 4242
    src_ckpt = Checkpointer(
        save_dir,
        checkpointables={"app": _Bundle(src, step=step)},
        sharded=sharded,
        storage=storage,
        async_save=async_save,
    )
    target = src_ckpt.save("checkpoint_4242")
    src_ckpt.wait_for_saves()

    dst_bundle = _Bundle(_model(2), step=0)
    dst_ckpt = Checkpointer(
        save_dir, checkpointables={"app": dst_bundle}, sharded=sharded, storage=storage
    )
    restored = dst_ckpt.resume_or_load(resume=True)

    assert restored == join(save_dir, target), f"pointer mismatch: {restored}"
    assert dst_bundle.step == step, f"step mismatch: {dst_bundle.step} != {step}"
    assert _params_equal(src, dst_bundle.model), "weights differ after round-trip"


def run(base: str) -> int:
    storage = Storage()
    opts = storage.options_for(base)
    endpoint = opts.get("client_kwargs", {}).get("endpoint_url", "<default>")
    print(f"base       = {base}")
    print(f"profile    = {opts.get('profile', '<none>')}")
    print(f"endpoint   = {endpoint}")
    print(f"addressing = {opts.get('config_kwargs', {}).get('s3', {}).get('addressing_style')}")

    print("\nvalidating access …")
    storage.validate_access(base)
    print("  ok")

    cases = [
        ("single-file", dict(sharded=False, async_save=False)),
        ("sync DCP", dict(sharded=True, async_save=False)),
        ("async DCP", dict(sharded=True, async_save=True)),
    ]
    failures = 0
    for label, kwargs in cases:
        run_dir = join(base, f"_mwm_ckpt_e2e_{uuid.uuid4().hex[:8]}")
        print(f"\n[{label}] {run_dir}")
        try:
            _roundtrip(run_dir, storage, **kwargs)
            print(f"  PASS ({label})")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"  FAIL ({label}): {type(exc).__name__}: {exc}")
        finally:
            try:
                storage.remove(run_dir, recursive=True)
                print("  cleaned up")
            except Exception as exc:  # noqa: BLE001
                print(f"  cleanup warning: {exc}")

    print("\n" + ("ALL PASSED" if failures == 0 else f"{failures} CASE(S) FAILED"))
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=os.environ.get("MINWM_TEST_S3_URL"),
        help="Base s3:// prefix to test under (or set MINWM_TEST_S3_URL).",
    )
    args = parser.parse_args()
    if not args.base:
        print("provide --base s3://... or set MINWM_TEST_S3_URL", file=sys.stderr)
        sys.exit(2)
    sys.exit(run(args.base.rstrip("/")))


if __name__ == "__main__":
    main()
