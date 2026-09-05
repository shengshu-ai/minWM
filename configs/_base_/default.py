"""Shared training scaffolding inherited by every per-line stage config.

Holds only the rarely-changing, family-agnostic defaults. Leaf configs add
``_base_ = "../../_base_/default.py"`` and override stage-specific ``training``
keys (``max_steps``, ``output_dir``, ``log_interval``, ``ckpt_interval``) plus
their own ``model`` / ``recipe`` / ``data`` nodes.
"""

training = dict(
    sp_size=1,
    tp_size=1,
    eval_interval=0,
)

profile = dict(
    enabled=False,
    start_iter=0,
    end_iter=1,
    rank=0,
)

checkpoint = dict(
    resume=False,
)
