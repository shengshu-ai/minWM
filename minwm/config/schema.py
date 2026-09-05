"""Typed umbrella config for the trainer: one ``MWMConfig`` owning every section.

The trainer takes a single ``cfg: MWMConfig`` and reads ``cfg.training.max_steps``,
``cfg.checkpoint.resume``, etc. — no more ``training.get("max_steps", 0)`` on a raw
dict. The split is by **extensibility**:

  - Fixed-schema sections (``training``, ``checkpoint``, ``profile``, the dataloader knobs in
    ``data``) are **frozen dataclasses** with field docstrings, so an IDE gives
    jump-to-def + hover docs and an unknown key (a typo) is a hard error.
  - Open/polymorphic sections (``model``, ``recipe``, ``auxiliary_models`` and the
    ``dataset`` / ``collate_fn`` build-graph nodes) stay :class:`DictConfig` —
    attribute-access (``cfg.model.dim``) and fed straight to
    :func:`minwm.config.build`, since their schema is per-class and can't be
    statically enumerated.

Parse a loaded config (py/yaml dict) with :meth:`MWMConfig.from_dict`.
"""

from dataclasses import dataclass, field, fields
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _empty_node() -> DictConfig:
    """Empty ``DictConfig`` default for open sections (``DictConfig()`` needs an arg)."""
    return OmegaConf.create({})


@dataclass(frozen=True)
class Training:
    """Training-loop knobs (the loop's stop condition + cadence + parallelism)."""

    max_steps: int = 0
    """Total optimization steps to run (the loop's stop condition)."""
    log_interval: int = 100
    """Steps between metric logs; 0 disables logging."""
    ckpt_interval: int = 0
    """Steps between checkpoint saves; 0 disables periodic saves."""
    eval_interval: int = 0
    """Steps between eval passes; 0 disables eval."""
    output_dir: str = ""
    """Local run directory (a scheme-free relative path). Holds the run's logs /
    TensorBoard / ``wandb`` dir, and — unless ``checkpoint.remote_root`` is set —
    the ``ckpts/`` tree too. Empty disables file logging and checkpoint saving."""
    sp_size: int = 1
    """Sequence-parallel degree (read by setup_distributed)."""
    tp_size: int = 1
    """Tensor-parallel degree."""
    seed: int | None = None
    """Base RNG seed. When ``None`` (the default) the trainer draws a random base
    seed on rank 0 and broadcasts it to every rank, so a job is reproducible from
    its logged seed without the user pinning one; set an int to fix it. At
    dist-init the trainer seeds ``random`` / ``numpy`` / ``torch`` (CPU+CUDA)
    globally with ``seed + dp_rank`` — identical within an SP group (its ranks
    share one ``dp_rank``, so the per-step noise/timestep draws off the global RNG
    match, as required since SP splits one sample across the group) and distinct
    across DP groups (noise diversity). The registered ``data-parallel-rng``
    tracker stream (drawn via ``get_rng_states_tracker().fork()`` in training
    recipes and preprocessors) is saved with the checkpoint and restored on
    resume unless ``no_load_rng`` is set."""
    no_load_rng: bool = False
    """Skip restoring the RNG-tracker state from the checkpoint on resume. When
    ``False`` (default) the noise stream resumes bit-exactly, so a resumed run
    reproduces the trajectory of an uninterrupted one; set ``True`` to reseed the
    stream fresh from ``seed + dp_rank`` instead."""
    fsdp_param_dtype: str = "bfloat16"
    """Compute dtype for sharded params under FSDP mixed precision."""
    fsdp_reduce_dtype: str = "float32"
    """Dtype for the FSDP gradient reduce-scatter."""
    fsdp_reshard_after_forward: bool = True
    """Reshard params after forward (True=FULL_SHARD, False=SHARD_GRAD_OP)."""
    fsdp_cpu_offload: bool = False
    """Offload FSDP params / grads / optimizer state to CPU."""
    activation_checkpointing: bool = False
    """Recompute transformer layers during backward to reduce activation memory.
    The trainer selects the backbone implementation without FSDP, the FSDP-layer
    wrapper under FSDP, or the offloader-compatible wrapper when activation
    offload is also enabled."""
    activation_offload: bool = False
    """Under FSDP2, asynchronously offload each transformer layer's saved
    activations to pinned CPU memory and prefetch them during backward. D2H/H2D
    copies overlap adjacent-layer compute. When activation checkpointing is also
    enabled, the offloader installs a compatible per-layer checkpoint wrapper;
    mutable KV-cache forwards use pure offload to preserve recomputation
    correctness. A no-op without FSDP2."""
    hsdp_shard_size: int = 0
    """HSDP shard degree; 0 shards over the full DP dimension (pure FSDP)."""


@dataclass(frozen=True)
class Checkpoint:
    """Checkpoint / resume knobs."""

    resume: bool = False
    """Resume from the latest checkpoint under the run's ``ckpts/`` dir."""
    remote_root: str | None = None
    """Object-store prefix for checkpoints (``s3://bucket/runs`` / ``oss://...``).
    When set, checkpoints and ``latest.txt`` route to
    ``<remote_root>/<training.output_dir>/ckpts`` on the backend the URL scheme
    names; when ``None`` (default) they stay local under ``output_dir/ckpts``.
    Logs / TensorBoard / ``wandb`` always stay local (they can't target a URL)."""
    pretrained: str | None = None
    """Optional path to pretrained weights to initialize from."""
    auxiliary_pretrained: DictConfig | None = None
    """Optional ``{aux_name: weight_path}`` map to seed named auxiliary models
    (e.g. DMD ``real_score`` / ``fake_score`` from an SFT checkpoint). Loaded
    only in the ``pretrained`` (fresh-init) path, not on ``resume``."""
    allow_partial_resume: bool = False
    """Permit a resume checkpoint that is missing state the current config
    declares (an optimizer or auxiliary model). ``False`` (default) raises so a
    checkpoint that would silently leave an aux net / optimizer uninitialized
    fails loudly; set ``True`` only to intentionally resume an older checkpoint
    whose set of optimizers / auxiliary models differs from the current config."""
    async_save: bool = False
    """Offload the sharded (DCP) checkpoint write to a background thread. ``save``
    returns once state is staged to CPU; the write — and the resume-pointer bump —
    completes on the next save or at loop end. No effect on single-file saves."""


@dataclass(frozen=True)
class Profile:
    """Nsight Systems / operator-NVTX capture window."""

    enabled: bool = False
    """Enable CUDA profiler API and operator-level NVTX annotations."""
    start_iter: int = 0
    """First training iteration included in the capture."""
    end_iter: int = 1
    """Iteration at which capture stops; the end is exclusive."""
    rank: int = 0
    """Global rank to capture; ``-1`` captures every rank."""

    def __post_init__(self) -> None:
        if self.start_iter < 0:
            raise ValueError("profile.start_iter must be non-negative")
        if self.end_iter <= self.start_iter:
            raise ValueError("profile.end_iter must be greater than profile.start_iter")
        if self.rank < -1:
            raise ValueError("profile.rank must be -1 or a non-negative global rank")


@dataclass(frozen=True)
class Data:
    """Dataloader knobs plus the open ``dataset`` / ``collate_fn`` build nodes."""

    dataset: DictConfig = field(default_factory=_empty_node)
    """Polymorphic ``type`` node built into a Dataset (open schema)."""
    collate_fn: DictConfig | None = None
    """Optional polymorphic ``type`` node for a collate fn (open schema)."""
    batch_size: int = 1
    """Per-replica batch size."""
    num_workers: int = 0
    """DataLoader worker processes."""
    shuffle: bool = True
    """Shuffle the dataset each epoch."""
    drop_last: bool = True
    """Drop the final partial batch."""
    pin_memory: bool = False
    """Pin host memory for faster H2D copies."""
    persistent_workers: bool = False
    """Keep worker processes alive between epochs."""


@dataclass(frozen=True)
class Inference:
    """Inference knobs: the data source, loop/sampler selection, and output.

    Fixed-schema like :class:`Training` — a key not listed here is a typo and
    raises at parse time, instead of silently falling through a ``.get()`` default
    (e.g. a misspelled ``guidance_scale`` used to disable CFG in silence). The
    open ``sampler`` / ``input_preprocessors`` nodes stay untyped build-graph
    specs. The chunk size ``num_frame_per_block`` is deliberately absent: it is a
    property of the causal model and lives only on the ``model`` node.
    """

    benchmark: str = ""
    """Path to the benchmark JSON (the data source). Resolved CWD-independently."""
    limit: int | None = None
    """Keep only the first ``limit`` benchmark items (smoke runs); ``None`` = all."""
    loop: str = "ARGenerationLoop"
    """Generation-loop class name (resolved by ``_resolve_loop_cls``)."""
    checkpoint: str | None = None
    """Path to the model weights to load (``None`` runs the base weights)."""
    checkpoint_key: str = "auto"
    """Which state-dict entry to load (``auto`` tries ema/generator/model)."""
    prefer_ema: bool = False
    """Prefer an EMA weight key when ``checkpoint_key='auto'``."""
    strict: bool = True
    """Require the checkpoint keys to match the model exactly."""
    output_dir: str = "outputs/infer"
    """Directory for ``{id}.mp4`` + ``{id}.latents.pt`` + ``manifest.json``."""
    guidance_scale: float = 1.0
    """CFG scale; ``<= 1`` disables the unconditional branch."""
    negative_prompt: str = ""
    """Negative prompt for text-CFG families (Wan)."""
    dtype: str = "bfloat16"
    """Compute dtype for the model / VAE."""
    seed: int | None = 0
    """Base seed; per-sample seed is ``seed + index``. ``None`` draws on rank 0."""
    sp_size: int = 1
    """Sequence-parallel degree; ``dp_size = world_size // sp_size`` is derived."""
    num_frames: int = 20
    """Latent frames to generate."""
    latent_shape: tuple = ()
    """Per-frame latent shape ``(C, H, W)`` for the noise sampler."""
    fps: int = 16
    """Frames per second for the written mp4."""
    context_noise: int = 0
    """Timestep for the AR per-block cache-refresh pass."""
    num_inference_steps: int | None = None
    """Diffusion steps (few-step samplers ignore it, using their step list)."""
    num_train_timesteps: int | None = None
    """Scheduler training-timestep count (sampler fallback)."""
    timestep_shift: float | None = None
    """Flow-matching schedule shift (sampler fallback)."""
    vae_tiling: bool = True
    """Enable VAE tiling; disable for HY's 3D-conv decode to avoid OOM."""
    offload_before_decode: bool = False
    """Move the generator to CPU before the VAE decode (HY temporal VAE)."""
    sampler: DictConfig = field(default_factory=_empty_node)
    """Open ``type`` node built into a sampler (open schema)."""
    input_preprocessors: tuple = ()
    """Ordered open ``type`` specs for the input-preprocessor chain."""


@dataclass(frozen=True)
class Monitor:
    """Metric fan-out knobs: which sinks to log to and where."""

    backends: tuple[str, ...] = ()
    """Metric sinks to fan out to: any of ``"tensorboard"``, ``"wandb"``. Empty -> STDOUT only."""
    wandb_project: str | None = None
    """WandB project name (else the ``WANDB_PROJECT`` env var, else ``"minwm"``)."""
    wandb_run_name: str | None = None
    """WandB run name (else the ``WANDB_RUN_NAME`` env var)."""

    def __post_init__(self) -> None:
        # YAML / py configs spell a list (``["tensorboard"]``); freeze it to a tuple.
        if not isinstance(self.backends, tuple):
            object.__setattr__(self, "backends", tuple(self.backends))


@dataclass(frozen=True)
class MWMConfig:
    """Umbrella config: typed fixed sections + open build-graph sections."""

    training: Training = field(default_factory=Training)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    profile: Profile = field(default_factory=Profile)
    data: Data = field(default_factory=Data)
    monitor: Monitor = field(default_factory=Monitor)
    inference: Inference = field(default_factory=Inference)
    model: DictConfig = field(default_factory=_empty_node)
    recipe: DictConfig = field(default_factory=_empty_node)
    auxiliary_models: DictConfig = field(default_factory=_empty_node)

    @classmethod
    def from_dict(cls, cfg: dict | DictConfig) -> "MWMConfig":
        """Parse a loaded config into a typed ``MWMConfig``.

        Fixed sections (``training`` / ``checkpoint`` / ``profile`` / ``data`` knobs) are
        splatted into their dataclasses **strictly** — an unknown key raises so
        typos like ``max_step`` fail loudly instead of silently defaulting. Open
        sections (``model`` / ``recipe`` / ``auxiliary_models`` and ``data``'s
        ``dataset`` / ``collate_fn``) become :class:`DictConfig`. Extra top-level
        keys the loader leaks (helper vars, a ``trainer`` selector) are ignored.

        Args:
            cfg (dict | DictConfig): the loaded config mapping.

        Returns:
            MWMConfig: the typed config.

        Raises:
            ValueError: if a fixed section carries a key not in its dataclass.
        """
        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        return cls(
            training=_strict(Training, cfg.get("training", {})),
            checkpoint=_strict(Checkpoint, cfg.get("checkpoint", {})),
            profile=_strict(Profile, cfg.get("profile", {})),
            data=_parse_data(cfg.get("data", {})),
            monitor=_strict(Monitor, cfg.get("monitor", {})),
            inference=parse_inference(cfg.get("inference", {})),
            model=OmegaConf.create(cfg.get("model", {})),
            recipe=OmegaConf.create(cfg.get("recipe", {})),
            auxiliary_models=OmegaConf.create(cfg.get("auxiliary_models", {})),
        )


def _strict(dc: type, node: dict) -> Any:
    """Build a fixed-schema dataclass from ``node``, rejecting unknown keys.

    Args:
        dc (type): the target dataclass.
        node (dict): the config sub-node.

    Returns:
        Any: an instance of ``dc``.

    Raises:
        ValueError: if ``node`` has a key that is not a field of ``dc``.
    """
    valid = {f.name for f in fields(dc)}
    unknown = set(node) - valid
    if unknown:
        keys = ", ".join(sorted(valid))
        bad = ", ".join(repr(k) for k in sorted(unknown))
        raise ValueError(f"unknown key(s) {bad} in {dc.__name__}; valid keys: {keys}")
    return dc(**node)


def parse_inference(node: dict) -> Inference:
    """Parse the ``inference`` node: strict knobs, open ``sampler`` node.

    Rejects unknown keys (so a typo like ``guidance_scal`` fails loudly instead
    of silently disabling CFG), while keeping the polymorphic ``sampler`` and
    ``input_preprocessors`` build-graph nodes untyped.

    Args:
        node (dict): the ``inference`` config sub-node.

    Returns:
        Inference: the typed inference config.

    Raises:
        ValueError: if ``node`` has a key that is not a field of :class:`Inference`.
    """
    node = dict(node)
    valid = {f.name for f in fields(Inference)}
    unknown = set(node) - valid
    if unknown:
        keys = ", ".join(sorted(valid))
        bad = ", ".join(repr(k) for k in sorted(unknown))
        raise ValueError(f"unknown key(s) {bad} in Inference; valid keys: {keys}")

    sampler = OmegaConf.create(node.pop("sampler", {}))
    preprocessors = tuple(node.pop("input_preprocessors", ()) or ())
    latent_shape = node.pop("latent_shape", ())
    return Inference(
        sampler=sampler,
        input_preprocessors=preprocessors,
        latent_shape=tuple(latent_shape) if latent_shape else (),
        **node,
    )


def _parse_data(node: dict) -> Data:
    """Parse the ``data`` node: open ``dataset`` / ``collate_fn``, strict knobs.

    Args:
        node (dict): the ``data`` config sub-node.

    Returns:
        Data: the typed data config.

    Raises:
        ValueError: if ``node`` has a key that is not a field of :class:`Data`.
    """
    node = dict(node)
    valid = {f.name for f in fields(Data)}
    unknown = set(node) - valid
    if unknown:
        keys = ", ".join(sorted(valid))
        bad = ", ".join(repr(k) for k in sorted(unknown))
        raise ValueError(f"unknown key(s) {bad} in Data; valid keys: {keys}")

    dataset = OmegaConf.create(node.pop("dataset", {}))
    collate = node.pop("collate_fn", None)
    return Data(
        dataset=dataset,
        collate_fn=OmegaConf.create(collate) if collate is not None else None,
        **node,
    )
