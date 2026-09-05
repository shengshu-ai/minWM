"""BaseTrainer: stage-agnostic training loop.

Owns the boilerplate shared by every stage:
  - build model + recipe from config (via ``type`` lazy import)
  - build dataloader
  - the step loop, logging, periodic checkpoint / eval

What differs per stage lives in the :class:`Recipe` (optimizers + one
step). The trainer never hardcodes flow-matching vs DMD vs ODE.

Subclasses typically only override hooks: :meth:`build_model`,
:meth:`build_dataloader`, :meth:`save_checkpoint`, :meth:`load_checkpoint`,
or :meth:`evaluate`. The default :meth:`build_model` / :meth:`build_recipe`
resolve the corresponding ``type`` config node.
"""

import os
import time
from contextlib import nullcontext
from typing import Any, Iterator

import torch
from torch import nn

from minwm.config import MWMConfig
from minwm.config.lazy import build as _build_cfg
from minwm.config.schema import Data
from minwm.data import build_dataloader as _build_dataloader_cfg
from minwm.modeling import build_model as _build_model_cfg
from minwm.utils import comm
from minwm.utils.logger import init_logger

from .checkpoint import Checkpointer, load_model_weights
from .checkpoint.formats import select_training_pretrained_state as _extract_pretrained_state
from .checkpoint.storage import join
from .events import EventStorage
from .monitor import MetricsProcessor
from .paths import ckpt_dir, local_run_dir
from .recipe_base import Recipe
from .runtime import initialize_runtime

logger = init_logger(__name__)


class _TrainerAppState:
    """:class:`~torch.distributed.checkpoint.stateful.Stateful` view of a trainer.

    Bridges the trainer's model / auxiliary models / optimizers / step to a
    single DCP application-state object. The DTensor-aware ``get_*_state_dict``
    helpers produce sharded state on save and ``set_*_state_dict`` reconstruct
    (and lazily allocate optimizer) state on load, which is what makes resuming
    FSDP optimizer momentum correct. Satisfies the runtime-checkable ``Stateful``
    protocol by duck typing (``state_dict`` / ``load_state_dict``), so the base
    trainer module imports without pulling in distributed checkpointing.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        self.trainer = trainer

    def state_dict(self) -> dict[str, Any]:
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        t = self.trainer
        sd: dict[str, Any] = {"step": torch.tensor(t.step)}
        sd["model"] = get_model_state_dict(t.model)
        for name, net in t.auxiliary_models.items():
            sd[f"aux/{name}"] = get_model_state_dict(net)
        for name, opt in t.optimizers.items():
            sd[f"opt/{name}"] = get_optimizer_state_dict(t._module_for_optimizer(opt), opt)
        from minwm.distributed.rng import get_rng_states_tracker

        sd["rng"] = get_rng_states_tracker().state_dict()
        # Persist the recipe's EMA-seeded flag so a resumed run does NOT re-seed
        # (i.e. overwrite) the restored generator-EMA with the live generator's
        # weights. The EMA aux model itself is saved under ``aux/`` above; without
        # this flag the recipe defaults it to False on resume and the first update
        # would clobber the smoothed EMA. Only recipes that own the flag emit it.
        seeded = getattr(self.trainer.recipe, "_generator_ema_seeded", None)
        if seeded is not None:
            sd["recipe/generator_ema_seeded"] = torch.tensor(bool(seeded))
        return sd

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        from torch.distributed.checkpoint.state_dict import (
            set_model_state_dict,
            set_optimizer_state_dict,
        )

        t = self.trainer
        allow_partial = t.cfg.checkpoint.allow_partial_resume
        set_model_state_dict(t.model, sd["model"])
        for name, net in t.auxiliary_models.items():
            if f"aux/{name}" not in sd:
                if allow_partial:
                    logger.warning("resume checkpoint has no 'aux/%s'; leaving it as built", name)
                    continue
                raise KeyError(
                    f"resume checkpoint has no 'aux/{name}' but the config declares "
                    f"auxiliary model '{name}'. Resuming would silently leave it at its "
                    f"freshly-built weights. Set checkpoint.allow_partial_resume=True to "
                    f"opt into partial resume."
                )
            set_model_state_dict(net, sd[f"aux/{name}"])
        for name, opt in t.optimizers.items():
            if f"opt/{name}" not in sd:
                if allow_partial:
                    logger.warning(
                        "resume checkpoint has no 'opt/%s'; leaving it uninitialized", name
                    )
                    continue
                raise KeyError(
                    f"resume checkpoint has no 'opt/{name}' but the config declares "
                    f"optimizer '{name}'. Set checkpoint.allow_partial_resume=True to "
                    f"opt into partial resume."
                )
            set_optimizer_state_dict(
                t._module_for_optimizer(opt), opt, optim_state_dict=sd[f"opt/{name}"]
            )
        if "rng" in sd and not t.cfg.training.no_load_rng:
            from minwm.distributed.rng import get_rng_states_tracker

            get_rng_states_tracker().load_state_dict(sd["rng"])
        # Restore the EMA-seeded flag (see state_dict). If the recipe owns the flag
        # but the checkpoint predates this fix, fall back to True whenever the EMA
        # aux weights were restored: those weights are the seed, so re-seeding must
        # not clobber them. Only a from-scratch run (no aux restore) leaves it False.
        if hasattr(t.recipe, "_generator_ema_seeded"):
            if "recipe/generator_ema_seeded" in sd:
                t.recipe._generator_ema_seeded = bool(sd["recipe/generator_ema_seeded"].item())
            elif "aux/generator_ema" in sd:
                t.recipe._generator_ema_seeded = True
        t.step = int(sd["step"].item())


class BaseTrainer:
    def __init__(self, cfg: MWMConfig | dict) -> None:
        """Build model / recipe / optimizers / dataloader from a typed config.

        Args:
            cfg (MWMConfig | dict): the typed umbrella config, or a raw mapping
                (py/yaml dict) which is parsed via :meth:`MWMConfig.from_dict`.
                Accepting both keeps the public ``trainer_cls(cfg)`` form and
                lets callers pass dict literals unchanged.
        """
        self.cfg = cfg if isinstance(cfg, MWMConfig) else MWMConfig.from_dict(cfg)
        self.step = 0

        self._fsdp = False
        self._checkpointer: Checkpointer | None = None

        self.setup_distributed()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model: nn.Module = self.build_model(self.cfg.model)
        self.auxiliary_models = self.build_auxiliary_models(self.cfg.auxiliary_models)
        if self.cfg.training.activation_offload and not self._fsdp:
            logger.warning("activation offload requires FSDP2; no model was sharded")
        self.recipe: Recipe = self.build_recipe(self.cfg.recipe)
        if self._fsdp:
            from minwm.distributed.fsdp import resolve_dtype

            self.recipe.dtype = resolve_dtype(self.cfg.training.fsdp_param_dtype)
        self.optimizers = self.recipe.build_optimizers(self.model)
        self.optimizers.update(self.recipe.build_auxiliary_optimizers(self.auxiliary_models))
        self.dataloader = self.build_dataloader(self.cfg.data)
        self._data_iter: Iterator[Any] | None = None

        self.storage = EventStorage(self.step)
        run_dir = local_run_dir(self.cfg.training.output_dir)
        self.metrics_processor = MetricsProcessor(self.cfg.monitor, run_dir or "")

    # ---- hooks (override as needed) --------------------------------------

    def setup_distributed(self) -> None:
        """Initialize the distributed/SP environment under torchrun.

        torchrun always sets ``RANK`` in the environment; a bare ``python`` run
        (single-process debugging, pytest) does not. Skip init in that case so
        the trainer is usable off-GPU instead of forcing an nccl/cuda init.

        Reads ``self.cfg.training.sp_size`` / ``tp_size`` (both default 1).

        After init, seeds the global RNG with ``training.seed + dp_rank`` so the
        per-step noise/timestep draws (off the global stream) are identical within
        an SP group and distinct across DP groups (see :func:`minwm.utils.set_seed`).
        When ``training.seed`` is ``None`` a random base seed is drawn on rank 0 and
        broadcast to every rank first (see :func:`minwm.utils.resolve_seed`).
        """
        from minwm.utils import resolve_seed, set_seed

        launched = "RANK" in os.environ
        initialize_runtime(
            tp_size=self.cfg.training.tp_size,
            sp_size=self.cfg.training.sp_size,
            hsdp_shard_size=self.cfg.training.hsdp_shard_size,
            distributed=launched,
        )
        if not launched:
            base_seed = resolve_seed(self.cfg.training.seed)
            logger.info("global RNG seed: %d", base_seed)
            set_seed(base_seed)
            return
        from minwm.distributed import get_parallel_state

        training = self.cfg.training
        base_seed = resolve_seed(training.seed)
        logger.info("base RNG seed: %d (+ dp_rank)", base_seed)
        set_seed(base_seed + get_parallel_state().dp_rank)

    def build_model(self, model_cfg: dict) -> nn.Module:
        """Build the primary model on the trainer device, sharded if FSDP is on.

        Building and FSDP sharding happen together (rather than in a later pass)
        so a freshly materialized full model is sharded before the next one is
        built — this keeps peak memory bounded for multi-model stages like DMD.

        Args:
            model_cfg (dict): the ``type`` config node for the primary model.

        Returns:
            nn.Module: the model on ``self.device``, FSDP-sharded when enabled.
        """
        model = _build_model_cfg(model_cfg).to(self.device)
        model.train()
        return self._maybe_shard(model)

    @staticmethod
    def _enable_native_activation_checkpointing(model: nn.Module) -> None:
        """Turn on backbone-native activation checkpointing for a non-FSDP model.

        Prefers diffusers' ``gradient_checkpointing_enable()``; falls back to the
        conventional ``gradient_checkpointing`` model flag.

        Args:
            model (nn.Module): the freshly built model to configure in place.
        """
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif hasattr(model, "gradient_checkpointing"):
            model.gradient_checkpointing = True

    def _should_fsdp(self) -> bool:
        """True iff FSDP2 sharding should be applied this run.

        FSDP2 is the only multi-process data-parallel path (there is no DDP), so
        it is always applied once there is an initialized model-parallel state
        with a >1 process world. Single-process / pytest runs (no ``RANK``, so no
        model-parallel init and a world size of 1) return False and keep models
        as plain unsharded modules.
        """
        from minwm.distributed import model_parallel_is_initialized

        return comm.get_world_size() > 1 and model_parallel_is_initialized()

    def _maybe_shard(
        self,
        module: nn.Module,
        *,
        shard: bool = True,
        activation_memory: bool = True,
    ) -> nn.Module:
        """Shard ``module`` in place with FSDP2 if eligible, else return as-is.

        Sharding is applied immediately after a module is built (by
        :meth:`build_model` / :meth:`build_auxiliary_models`) so its parameters
        are sharded before the next module materializes. Whether to shard is an
        explicit decision (the ``shard`` arg), **decoupled from trainability**:

        - The trainable model is always sharded (gradients to reduce).
        - A frozen auxiliary net is sharded only when asked. ``fully_shard``
          supports ``requires_grad=False`` params, so a frozen net *can* shard —
          and a frozen teacher/EMA **must** match the (sharded) student's
          ``DTensor`` layout for :func:`~minwm.engine.optim.ema.copy_params` /
          ``update_ema_params`` to copy between them. But sharding a frozen net
          can also perturb other nets' backward under multi-forward recipes
          (e.g. DMD's reused real-score), so it stays opt-in per config entry.

        Args:
            module (nn.Module): the freshly built module, already on device.
            shard (bool): request sharding. Sharding still only happens when the
                run is multi-process and the module defines
                ``_fsdp_shard_conditions``.
            activation_memory (bool): allow activation checkpointing/offload on
                this module. Frozen auxiliary models pass ``False`` since their
                no-grad forwards save no activations for backward.

        Returns:
            nn.Module: ``module`` (sharded in place when eligible). Sets
            ``self._fsdp`` once any module is sharded, to drive DCP checkpointing.
        """
        training = self.cfg.training
        activation_checkpointing = training.activation_checkpointing and activation_memory
        activation_offload = training.activation_offload and activation_memory
        if not shard or not self._should_fsdp():
            if activation_checkpointing:
                self._enable_native_activation_checkpointing(module)
            return module
        if not getattr(module, "_fsdp_shard_conditions", None):
            if activation_checkpointing:
                self._enable_native_activation_checkpointing(module)
            logger.warning(
                "%s defines no '_fsdp_shard_conditions'; leaving it unsharded.",
                type(module).__name__,
            )
            return module
        from minwm.distributed import get_fsdp_mesh
        from minwm.distributed.fsdp import build_mp_policy, shard_model

        shard_model(
            module,
            mesh=get_fsdp_mesh(),
            mp_policy=build_mp_policy(training.fsdp_param_dtype, training.fsdp_reduce_dtype),
            reshard_after_forward=training.fsdp_reshard_after_forward,
            cpu_offload=training.fsdp_cpu_offload,
            activation_checkpointing=activation_checkpointing,
            activation_offload=activation_offload,
        )
        self._fsdp = True
        return module

    def build_auxiliary_models(self, aux_cfg: dict) -> dict[str, nn.Module]:
        """Build trainer-owned auxiliary nets (e.g. DMD score nets, CD teacher/EMA).

        Each entry is built via the lazy config builder, moved to the trainer
        device, set trainable or frozen per a per-entry ``trainable`` flag
        (default ``True``; ``False`` -> ``requires_grad_(False).eval()``),
        activation-checkpointed when ``training.activation_checkpointing`` is set
        (the trainable critic needs it as much as the primary model — DMD holds
        three 1.3B nets), and FSDP-sharded per a per-entry ``shard`` flag
        (default = ``trainable``).
        Sharding is decoupled from trainability so a frozen net can still be
        sharded to match the student's layout (CD teacher/EMA, for
        ``copy_params``) while another frozen net stays replicated (DMD
        real-score, whose sharding perturbs the multi-forward backward). Empty by
        default — single-model stages have no ``auxiliary_models`` node.

        Args:
            aux_cfg (dict): mapping of name -> model config node. Each node may
                carry ``trainable`` (bool, default True; False ->
                ``requires_grad_(False).eval()``) and ``shard`` (bool, default =
                ``trainable``; set True on a frozen net to FSDP-shard it anyway).

        Returns:
            dict[str, nn.Module]: named auxiliary models on the trainer device,
            FSDP-sharded per their ``shard`` flag when multi-process.
        """
        models: dict[str, nn.Module] = {}
        for name, node in aux_cfg.items():
            node = dict(node)
            trainable = node.pop("trainable", True)
            shard = node.pop("shard", trainable)
            net = _build_model_cfg(node).to(self.device)
            if trainable:
                net.requires_grad_(True)
                net.train()
            else:
                net.requires_grad_(False)
                net.eval()
            models[name] = self._maybe_shard(
                net,
                shard=shard,
                activation_memory=trainable,
            )
        return models

    def build_recipe(self, recipe_cfg: dict) -> Recipe:
        recipe = _build_cfg(recipe_cfg)
        if not isinstance(recipe, Recipe):
            raise TypeError(f"recipe.type must build a Recipe, got {type(recipe)}")
        return recipe

    def build_dataloader(self, data: Data) -> Any:
        """Build the training dataloader, or ``None`` when no dataset is configured.

        Args:
            data (Data): the typed ``data`` config; an empty ``data.dataset``
                (no ``type`` node) means "no loader" (e.g. pytest / dry runs).

        Returns:
            Any: a ``DataLoader``, or ``None`` when ``data.dataset`` is empty.
        """
        if not data.dataset:
            return None
        return _build_dataloader_cfg(data)

    def _module_for_optimizer(self, optimizer: torch.optim.Optimizer) -> nn.Module:
        """Find the module whose parameters ``optimizer`` steps.

        Needed by DCP's optimizer state-dict helpers, which key optimizer state
        on a module's fully-qualified parameter names. Matches by parameter
        identity across the primary model and auxiliary nets, falling back to the
        primary model.

        Args:
            optimizer (torch.optim.Optimizer): the optimizer to locate.

        Returns:
            nn.Module: the owning module (default: ``self.model``).
        """
        param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        for module in (self.model, *self.auxiliary_models.values()):
            if any(id(p) in param_ids for p in module.parameters()):
                return module
        return self.model

    def _get_checkpointer(self) -> Checkpointer:
        """Build (once) the :class:`Checkpointer` bound to this trainer's app state.

        Deferred until first use so it captures ``self._fsdp`` (set during model
        build) and picks the sharded (DCP) vs single-file path accordingly. The
        checkpoint dir comes from :func:`~minwm.engine.paths.ckpt_dir`
        (``<output_dir>/ckpts`` locally, ``<remote_root>/<output_dir>/ckpts`` when
        ``checkpoint.remote_root`` is set); an empty ``output_dir`` yields a
        save-disabled Checkpointer so dry runs / pytest don't crash.
        """
        if self._checkpointer is None:
            save_dir = ckpt_dir(self.cfg.training.output_dir, self.cfg.checkpoint.remote_root)
            self._checkpointer = Checkpointer(
                save_dir,
                checkpointables={"app": _TrainerAppState(self)},
                sharded=self._fsdp,
                async_save=self.cfg.checkpoint.async_save,
                allow_partial_load=self.cfg.checkpoint.allow_partial_resume,
            )
        return self._checkpointer

    def save_checkpoint(self) -> None:
        """Save model + optimizers + auxiliary models + step, then bump ``latest.txt``.

        Under FSDP the state is written as a sharded
        :mod:`torch.distributed.checkpoint` (DCP) directory
        ``ckpts/checkpoint_{step}/`` (every rank participates); otherwise a single
        rank-0 ``ckpts/checkpoint_{step}.pt``. Checkpoints and ``latest.txt`` live
        under the ``ckpts/`` dir from :func:`~minwm.engine.paths.ckpt_dir`,
        separate from logs / ``wandb/``. A missing ``output_dir`` is a no-op with a
        warning so dry runs / pytest don't crash. Delegates to :class:`Checkpointer`,
        which also handles the async (``checkpoint.async_save``) DCP path.
        """
        self._get_checkpointer().save(f"checkpoint_{self.step}")

    def load_checkpoint(self) -> None:
        """Restore from a checkpoint per ``checkpoint`` config (resume > pretrained).

        ``resume=True`` restores model + optimizers + auxiliary models + step from
        the latest checkpoint under ``output_dir/ckpts``; a missing pointer raises
        ``FileNotFoundError`` rather than silently falling back to a fresh init or
        ``pretrained`` (which would train from the wrong weights). A resume
        checkpoint that lacks a config-declared optimizer / auxiliary model also
        fails loudly unless ``checkpoint.allow_partial_resume`` is set (wired to
        the Checkpointer's ``allow_partial_load``): on the DCP path the DCP load
        planner rejects the missing keys, on the single-file path
        :meth:`Checkpointer.load` rejects a missing top-level ``app`` and
        :meth:`_TrainerAppState.load_state_dict` rejects a missing per-key
        optimizer / aux entry — so both paths fail loud on a stale checkpoint.
        Otherwise ``pretrained`` (if set) loads **model
        weights only** (optimizer / aux state is not populated), and
        ``auxiliary_pretrained`` (if set) seeds named auxiliary models from their
        own weight files — the two are independent, so a stage may seed only the
        model, only some aux nets, or both. Under FSDP all paths use DCP /
        DTensor-aware loading so a full (unsharded) weight file is redistributed
        into the sharded module.
        """
        ckpt = self.cfg.checkpoint
        if ckpt.resume:
            if self._get_checkpointer().resume_or_load(resume=True) is None:
                save_dir = ckpt_dir(self.cfg.training.output_dir, self.cfg.checkpoint.remote_root)
                pointer = join(save_dir, "latest.txt") if save_dir else "<no output_dir>"
                raise FileNotFoundError(
                    f"checkpoint.resume=True but no resume pointer was found at "
                    f"{pointer!r}. Refusing to silently fall back to a fresh init or "
                    f"`checkpoint.pretrained`: doing so would train from the wrong weights "
                    f"(e.g. a CD stage seeding student/teacher/ema from the base model "
                    f"instead of the promoted TF checkpoint). Set checkpoint.resume=False "
                    f"to start from `pretrained`, or point training.output_dir "
                    f"(+ checkpoint.remote_root) at a run that has a latest.txt."
                )
            return
        if ckpt.pretrained or ckpt.auxiliary_pretrained:
            if ckpt.pretrained:
                self._load_pretrained_into(self.model, ckpt.pretrained)
                logger.info("loaded pretrained weights from %s", ckpt.pretrained)
            for name, path in (ckpt.auxiliary_pretrained or {}).items():
                if name not in self.auxiliary_models:
                    raise KeyError(
                        f"auxiliary_pretrained names '{name}' but the only auxiliary "
                        f"models are {sorted(self.auxiliary_models)}"
                    )
                self._load_pretrained_into(self.auxiliary_models[name], path)
                logger.info("loaded pretrained weights for auxiliary '%s' from %s", name, path)

    def _load_pretrained_into(self, module: nn.Module, path: str) -> None:
        """Load a full (unsharded) weight file into ``module`` (weights only).

        Redistributes a full state dict into ``module`` — DCP / DTensor-aware
        under FSDP (broadcast from rank 0), plain ``load_state_dict`` otherwise.
        Accepts minwm-native ``{"model": ...}`` wrappers, legacy DMD
        ``{"generator"|"generator_ema": ...}`` checkpoints (with ``model.`` /
        FSDP-prefixed keys), and raw state dicts — see
        :func:`_extract_pretrained_state`.

        Args:
            module (nn.Module): destination module (primary model or an auxiliary).
            path (str): path to the full weight file.
        """
        load_model_weights(module, path, sharded=self._fsdp, select=_extract_pretrained_state)

    def evaluate(self) -> dict[str, float]:
        return {}

    def close(self) -> None:
        """Flush and release the metrics processor's writers (idempotent)."""
        self.metrics_processor.close()

    # ---- properties ------------------------------------------------------

    @property
    def is_main_process(self) -> bool:
        return comm.is_main_process()

    # ---- loop ------------------------------------------------------------

    def _next_batch(self) -> Any:
        if self._data_iter is None:
            self._data_iter = iter(self.dataloader)
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            return next(self._data_iter)

    def train(self) -> None:
        """Run the optimization loop: per-step recipe call + periodic log / eval / ckpt.

        Resume (``load_checkpoint``) runs first so ``self.step`` and the storage
        iteration reflect the loaded step. The :class:`EventStorage` is pushed
        **once** around the whole loop (so a mid-loop exception still pops it via
        ``__exit__``), and writers are always flushed in ``finally`` even on crash.
        The final checkpoint saves inside ``try`` so a failing run does not persist
        a half-trained state.
        """
        self.load_checkpoint()
        nvtx_ctx = None
        self.storage.nvtx_enabled = False
        self.storage.iter = self.step
        self.metrics_processor.start_window(self.step)
        training = self.cfg.training
        with self.storage:
            try:
                while self.step < training.max_steps:
                    self.storage.iter = self.step
                    profile = self.cfg.profile
                    profile_rank = profile.rank in (-1, comm.get_rank())
                    if (
                        profile.enabled
                        and profile_rank
                        and torch.cuda.is_available()
                        and self.step == profile.start_iter
                    ):
                        torch.cuda.cudart().cudaProfilerStart()
                        self.storage.nvtx_enabled = True
                        nvtx_ctx = torch.autograd.profiler.emit_nvtx(record_shapes=True)
                        nvtx_ctx.__enter__()
                    if (
                        nvtx_ctx is not None
                        and profile.enabled
                        and profile_rank
                        and self.step == profile.end_iter
                    ):
                        torch.cuda.cudart().cudaProfilerStop()
                        nvtx_ctx.__exit__(None, None, None)
                        self.storage.nvtx_enabled = False
                        nvtx_ctx = None
                    t0 = time.perf_counter()
                    batch = self._next_batch()
                    self.storage.put_scalar("time/data(ms)", 1000 * (time.perf_counter() - t0))
                    nvtx_range = (
                        torch.cuda.nvtx.range("train_step")
                        if self.storage.nvtx_enabled
                        else nullcontext()
                    )
                    with nvtx_range:
                        metrics = self.recipe.train_one_step(
                            self.model,
                            batch,
                            self.optimizers,
                            self.step,
                            auxiliary_models=self.auxiliary_models,
                        )
                    self.storage.put_scalars(**metrics)
                    self.step += 1

                    if training.log_interval and self.step % training.log_interval == 0:
                        self.metrics_processor.log(self.step, self.storage.latest())
                    if training.eval_interval and self.step % training.eval_interval == 0:
                        eval_metrics = self.evaluate()
                        if eval_metrics:
                            self.metrics_processor.log(self.step, eval_metrics)
                    if training.ckpt_interval and self.step % training.ckpt_interval == 0:
                        self.save_checkpoint()

                self.save_checkpoint()
            finally:
                if self._checkpointer is not None:
                    self._checkpointer.wait_for_saves()
                if nvtx_ctx is not None:
                    torch.cuda.cudart().cudaProfilerStop()
                    nvtx_ctx.__exit__(None, None, None)
                    self.storage.nvtx_enabled = False
                self.close()
