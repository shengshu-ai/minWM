"""Recipe: the per-stage training logic, decoupled from the trainer loop.

A :class:`BaseTrainer` owns the boilerplate (distributed init, dataloader,
checkpointing, the step loop). A :class:`Recipe` owns what differs between
training stages: which optimizers exist, and what one optimization step does
(forward / loss / backward / step).

Recipe mapping:
    bidirectional SFT       -> BiSFTRecipe
    AR teacher forcing      -> ARTFRecipe
    AR ODE regression       -> ARODERecipe
    AR consistency          -> ARCDRecipe
    AR DMD                  -> ARDMDRecipe

The first three recipes share the internal flow-matching implementation; public
configs reference the concrete recipe modules under ``minwm.engine.training.recipes``::

    recipe:
      type: minwm.engine.training.recipes.bi_sft:BiSFTRecipe
      loss:
        type: minwm.engine.training.losses:FlowMatchingLoss
"""

from abc import ABC
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import nn

from minwm.engine.optim.builder import build_optimizer
from minwm.utils.logger import init_logger

from .events import get_event_storage, has_event_storage, record_time

logger = init_logger(__name__)

DEFAULT_OPTIMIZER_CFG = {"lr": 2e-6, "betas": (0.0, 0.999), "weight_decay": 0.0}


def _grad_norm_value(grad_norm: torch.Tensor) -> float:
    if hasattr(grad_norm, "full_tensor"):
        grad_norm = grad_norm.full_tensor()
    return float(grad_norm)


def _reduce_device() -> torch.device:
    """Device for the skip-consensus collectives (NCCL needs CUDA tensors)."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _local_shard_grad_norm(params: list) -> float:
    total = 0.0
    for p in params:
        g = p.grad
        if g is None:
            continue
        if hasattr(g, "to_local"):
            g = g.to_local()
        total += g.detach().float().pow(2).sum().item()
    return total**0.5


def _summarize_value(value: Any) -> str:
    """Render one batch value compactly for a skipped-step dump.

    Tensors collapse to ``shape/dtype/device/min/max/mean/nan/inf`` (and inline
    the full values when tiny, e.g. a per-sample ``timestep``); everything else
    (prompts, ids, scalars) prints in full. Stats are computed without copying
    the whole tensor to host.

    Args:
        value (Any): one value from the batch dict.

    Returns:
        str: a single-line summary.
    """
    if not isinstance(value, torch.Tensor):
        return repr(value)

    t = value.detach()
    if hasattr(t, "to_local"):  # DTensor shard — summarize the local piece
        t = t.to_local()

    head = f"Tensor(shape={list(t.shape)} dtype={t.dtype} dev={t.device}"
    if t.numel() == 0:
        return head + " empty)"
    if t.numel() <= 8:
        return head + f" values={t.flatten().tolist()})"

    f = t.float()
    nan = int(torch.isnan(f).sum().item())
    inf = int(torch.isinf(f).sum().item())
    finite = f[torch.isfinite(f)]
    if finite.numel() == 0:
        return head + f" all-nonfinite nan={nan} inf={inf})"
    return (
        f"{head} min={finite.min().item():.4g} max={finite.max().item():.4g} "
        f"mean={finite.mean().item():.4g} nan={nan} inf={inf})"
    )


def _format_batch(batch: Any) -> str:
    """Format a batch dict as one ``key: summary`` line per entry.

    Args:
        batch (Any): the batch handed to the step; only ``dict`` is summarized
            field-by-field, anything else falls back to a single summary line.

    Returns:
        str: a multi-line block (no trailing newline).
    """
    if not isinstance(batch, dict):
        return "  " + _summarize_value(batch)
    return "\n".join(f"  {k}: {_summarize_value(v)}" for k, v in batch.items())


def _default_adapter():
    """Construct the default Wan model adapter (lazy import to avoid an import cycle)."""
    from minwm.modeling.wan21.adapter import Wan21Adapter

    return Wan21Adapter()


class Recipe(ABC):
    """Per-stage training logic.

    Single-optimizer stages only need to implement :meth:`train_one_step` and
    pass an ``optimizer`` config to :meth:`__init__`; the inherited
    :meth:`build_optimizers` wraps the model in a single ``"main"`` optimizer.
    Multi-optimizer stages (e.g. DMD's generator + critic) override
    :meth:`build_optimizers` / :meth:`build_auxiliary_optimizers` to return
    multiple named optimizers and alternate between them.

    All model-facing config (text dims, working dtype) lives on the
    :attr:`adapter`; the recipe reads the model's working precision back through
    the :attr:`dtype` property rather than owning a copy.
    """

    def __init__(
        self,
        optimizer: dict | None = None,
        adapter: Any | None = None,
        max_grad_norm: float = 10.0,
        skip_grad_norm: float | None = None,
        batch_preprocessors: list | None = None,
    ) -> None:
        """Store optimizer config + the shared model adapter and step knobs.

        Args:
            optimizer (dict, optional): optimizer config for :func:`build_optimizer`
                — optional ``name`` (``"module:Class"``, default ``torch.optim:AdamW``)
                plus constructor kwargs (``lr``, ``betas``, ``weight_decay``). Defaults
                to AdamW with ``lr=2e-6, betas=(0.0, 0.999), weight_decay=0.0``.
            adapter (ModelAdapter, optional): bridges recipe ``[B,F,C,H,W]`` space to a
                model family's call convention and owns the model-facing text dims /
                working dtype. Defaults to :class:`~minwm.modeling.wan21.adapter.Wan21Adapter`.
            max_grad_norm (float): gradient-clipping norm used by
                :meth:`optimizer_step` — grads with a larger norm are scaled down
                to it (norms below it pass through untouched).
            skip_grad_norm (float, optional): if set, a step whose global grad norm
                exceeds it is **skipped** entirely (no clip, no optimizer step) on
                every rank in unison. Must be ``>= max_grad_norm``. ``None`` (default)
                disables skipping — the plain clip-only behaviour.
            batch_preprocessors (list, optional): the recipe's preprocessor chain
                as a list of ``type``-dicts, stored raw here and instantiated
                lazily on the first :meth:`preprocess` call (once the subclass
                scheduler exists).

        Raises:
            ValueError: if ``skip_grad_norm`` is set and is below ``max_grad_norm``
                (the skip threshold must sit above the scale threshold).
        """
        if skip_grad_norm is not None and skip_grad_norm < max_grad_norm:
            raise ValueError(
                f"skip_grad_norm ({skip_grad_norm}) must be >= max_grad_norm "
                f"({max_grad_norm}): the skip threshold sits above the scale threshold."
            )
        self.batch_preprocessors_cfg = batch_preprocessors
        self.optimizer_cfg = optimizer or dict(DEFAULT_OPTIMIZER_CFG)
        self.adapter = adapter if adapter is not None else _default_adapter()
        self.max_grad_norm = max_grad_norm
        self.skip_grad_norm = skip_grad_norm
        self.batch_preprocessors: list | None = None

    @property
    def dtype(self) -> torch.dtype:
        """The model's working dtype, owned by the :attr:`adapter`."""
        return self.adapter.dtype

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        """Override the working dtype (the trainer matches it to the FSDP compute
        dtype, so a config's static ``adapter.dtype`` does not have to know whether
        the run is sharded)."""
        self.adapter.dtype = value

    def autocast(self) -> AbstractContextManager:
        """Autocast context for the forward when the working dtype is half.

        Mixed-precision (FSDP ``bfloat16``) runs cast the sharded params to
        :attr:`dtype`, but the flow pipeline (the scheduler's fp32 sigma promotes
        ``noisy`` / ``clean`` back to fp32) and some backbones' internals (HY's
        time / text embedders build fp32 tensors) are not dtype-clean. Autocast
        bridges them at every op so non-dtype-clean backbones run sharded without
        scattering manual casts. fp32 runs (single-process / pytest) get a no-op,
        leaving those paths unchanged.

        Returns:
            AbstractContextManager: a CUDA autocast context in half precision, or
            a no-op context otherwise.
        """
        if self.dtype in (torch.float16, torch.bfloat16) and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=self.dtype)
        return nullcontext()

    def build_preprocessors(self) -> list:
        """Build the batch-preprocessor chain from the config-supplied spec.

        The chain is built from ``batch_preprocessors`` (a list of ``type``-dicts
        passed to :meth:`__init__`) via :func:`minwm.config.lazy.build`. A static
        config cannot reference the recipe's runtime ``scheduler`` / ``dtype``, so
        any built preprocessor that left those attributes ``None`` has them
        injected here. Called lazily by :meth:`preprocess`, after the subclass
        ``__init__`` has created ``self.scheduler``, so the injection sees it.

        Returns:
            list: the preprocessor chain to assign to ``self.batch_preprocessors``.

        Raises:
            ValueError: if no ``batch_preprocessors`` were supplied in config.
        """
        if not self.batch_preprocessors_cfg:
            raise ValueError(
                f"{type(self).__name__} requires a 'batch_preprocessors' config "
                "(a list of type-dicts); none was supplied."
            )

        from minwm.config.lazy import build

        chain = []
        for entry in self.batch_preprocessors_cfg:
            preprocessor = build(entry)
            if getattr(preprocessor, "scheduler", "unset") is None:
                preprocessor.scheduler = getattr(self, "scheduler", None)
            if getattr(preprocessor, "dtype", "unset") is None:
                preprocessor.dtype = self.dtype
            chain.append(preprocessor)
        return chain

    def preprocess(self, batch: dict, device: torch.device) -> dict:
        """Run the batch through this recipe's preprocessor chain.

        The chain is built lazily on the first call (via
        :meth:`build_preprocessors`), so the scheduler the subclass creates after
        ``super().__init__()`` is already present when the preprocessors are
        instantiated and injected.

        Args:
            batch (dict): one raw dataloader batch.
            device (torch.device): target device for moved/created tensors.

        Returns:
            dict: the model-ready batch after every preprocessor has run.
        """
        if self.batch_preprocessors is None:
            self.batch_preprocessors = self.build_preprocessors()
        for preprocessor in self.batch_preprocessors:
            batch = preprocessor(batch, device)
        return batch

    def optimizer_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        params: Any,
        backward_name: str = "backward",
        batch: Any | None = None,
    ) -> bool:
        """Zero-grad, timed backward, grad-clip-or-skip, and step.

        With :attr:`skip_grad_norm` set, a step is governed by two thresholds:
        a global grad norm ``> skip_grad_norm`` **skips** the whole step (no clip,
        no ``optimizer.step()``); a norm in ``(max_grad_norm, skip_grad_norm]`` is
        scaled to ``max_grad_norm``; a norm ``<= max_grad_norm`` passes through.
        The skip decision is reduced across ranks so every rank skips in unison
        (FSDP requires it — a partial skip would desync the shards).

        Args:
            loss (Tensor): scalar loss to backpropagate.
            optimizer (torch.optim.Optimizer): optimizer to step.
            params (Any): parameters to clip (an iterable of tensors).
            backward_name (str): event name for the timed backward. DMD passes
                ``"generator_backward"`` / ``"critic_backward"`` to separate its
                two optimizers; defaults to ``"backward"``.
            batch (Any, optional): the step's batch. When set and the step is
                skipped, each rank logs a per-field summary of it next to the
                skip report, to pin down which samples drove the grad spike.

        Returns:
            bool: True if the optimizer stepped, False if the step was skipped.
        """
        params = list(params)
        optimizer.zero_grad()
        with record_time(backward_name):
            loss.backward()

        # Measure first (no clipping) so the skip decision sees the true norm.
        # clip_grad_norm_(inf) normally leaves grads untouched, but on a non-finite
        # norm the coefficient is inf/inf = nan and the grads get nan'd. That only
        # happens on the skip branch below, where the grads are discarded by the
        # next zero_grad() — it never reaches the optimizer.
        if self.skip_grad_norm is not None:
            total_norm = _grad_norm_value(torch.nn.utils.clip_grad_norm_(params, float("inf")))
            if self._skip_step_consensus(total_norm):
                self._report_grad_skip(total_norm, params, batch)
                if has_event_storage():
                    get_event_storage().put_scalar("grad_norm", total_norm)
                    get_event_storage().put_scalar("grad_skipped", 1.0)
                return False
            if total_norm > self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        else:
            total_norm = _grad_norm_value(
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            )

        if has_event_storage():
            get_event_storage().put_scalar("grad_norm", total_norm)
            if self.skip_grad_norm is not None:
                get_event_storage().put_scalar("grad_skipped", 0.0)
        optimizer.step()
        return True

    def _skip_step_consensus(self, local_norm: float) -> bool:
        """True if any rank's grad norm exceeds :attr:`skip_grad_norm`.

        Under FSDP the norm is already global (identical on every rank), but the
        consensus all-reduce also covers the no-shard / mixed cases and guarantees
        bit-identical skip decisions even if a rank's norm is NaN/inf.

        Args:
            local_norm (float): this rank's observed global grad norm.

        Returns:
            bool: whether to skip the step (agreed across all ranks).
        """
        over = not (local_norm <= self.skip_grad_norm)  # True also for NaN
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return over
        flag = torch.tensor([1.0 if over else 0.0], device=_reduce_device())
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        return bool(flag.item() > 0.0)

    def _report_grad_skip(self, local_norm: float, params: list, batch: Any | None = None) -> None:
        """Log the skip, naming the rank whose local grads drove it.

        Args:
            local_norm (float): this rank's observed global grad norm.
            params (list): the clipped parameters (for a local-shard norm probe).
            batch (Any, optional): the skipped step's batch. When set, only the
                culprit rank (largest local-shard grad) logs its per-field
                summary, so the dump is exactly the batch that caused the skip.
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            logger.warning(
                "grad-norm %.4g > skip_grad_norm %.4g — skipping optimizer step.",
                local_norm,
                self.skip_grad_norm,
            )
            if batch is not None:
                logger.warning("skipped-step batch:\n%s", _format_batch(batch))
            return

        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        # Probe the local shard's own grad norm so the report names the culprit
        # rank, not just the (replicated) global norm.
        local_shard = _local_shard_grad_norm(params)
        buf = torch.tensor([local_shard], device=_reduce_device())
        gathered = [torch.zeros_like(buf) for _ in range(world)]
        torch.distributed.all_gather(gathered, buf)
        norms = [g.item() for g in gathered]
        worst = max(range(world), key=lambda r: norms[r])
        if rank == 0:
            logger.warning(
                "grad-norm %.4g > skip_grad_norm %.4g — skipping step on ALL %d ranks. "
                "Largest local-shard grad on rank %d (%.4g); per-rank=%s",
                local_norm,
                self.skip_grad_norm,
                world,
                worst,
                norms[worst],
                [round(x, 3) for x in norms],
            )
        # Only the culprit rank dumps its batch — that's the data that drove the
        # grad spike. Other ranks' batches are noise for this purpose.
        if batch is not None and rank == worst:
            logger.warning("[rank %d] skip-triggering batch:\n%s", rank, _format_batch(batch))

    def build_optimizers(self, model: nn.Module) -> dict[str, torch.optim.Optimizer]:
        """Return named optimizers. Single-optimizer default: ``{"main": opt}``.

        Args:
            model (nn.Module): the trainer-owned primary model.

        Returns:
            dict[str, torch.optim.Optimizer]: ``{"main": optimizer}`` over the
            model's trainable parameters, built from ``self.optimizer_cfg``.
        """
        return {"main": build_optimizer(model, self.optimizer_cfg)}

    def build_auxiliary_optimizers(
        self, auxiliary_models: dict[str, nn.Module]
    ) -> dict[str, torch.optim.Optimizer]:
        """Optimizers for trainer-owned auxiliary models. Default: none.

        Multi-model stages (e.g. DMD's trainable critic ``fake_score``) override
        this to add their optimizers, which the trainer merges into the
        optimizer dict alongside :meth:`build_optimizers`.

        Args:
            auxiliary_models (dict[str, nn.Module]): trainer-owned auxiliary nets.

        Returns:
            dict[str, torch.optim.Optimizer]: named optimizers (possibly empty).
        """
        return {}

    def train_one_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizers: dict[str, torch.optim.Optimizer],
        step: int,
        auxiliary_models: dict[str, nn.Module] | None = None,
    ) -> dict[str, float]:
        """Run one optimization step — the trainer's per-step entry point.

        This is the one method the trainer calls per step. The generic
        single-model flow lives in the concrete recipes under
        :mod:`minwm.engine.training.recipes`
        (preprocess → encode_text → adapter.denoise → loss → optimizer_step);
        multi-optimizer / multi-forward stages (consistency distillation, DMD)
        implement their own orchestration.

        Args:
            model (nn.Module): the trainer-owned primary model.
            batch (Any): one dataloader batch.
            optimizers (dict[str, torch.optim.Optimizer]): named optimizers from
                :meth:`build_optimizers` merged with
                :meth:`build_auxiliary_optimizers`.
            step (int): current global step.
            auxiliary_models (dict[str, nn.Module], optional): trainer-owned
                auxiliary nets.

        Returns:
            dict[str, float]: scalar metrics to log (must include ``"loss"``).

        Raises:
            NotImplementedError: if a subclass does not implement it.
        """
        raise NotImplementedError
