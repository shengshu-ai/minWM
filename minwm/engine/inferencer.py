"""Inferencer runtime: config build, checkpoint load, seeding and output writing."""

import json
import os
import re

import torch

from minwm.config import build, locate, parse_inference
from minwm.modeling.build import build_model
from minwm.processors import (
    CameraTrajectory,
    InferenceInputPreprocessor,
    InferenceRuntime,
    LatentNoise,
)
from minwm.utils.dtype import resolve_dtype
from minwm.utils.seed import resolve_seed, set_seed

from .checkpoint import load_model_weights
from .inference.samplers import build_sampler
from .runtime import initialize_runtime

# Repo root (parent of the ``minwm`` package), used to resolve a relative
# ``inference.benchmark`` path when the process is launched from another CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sanitize_filename(text: str, limit: int = 100) -> str:
    text = re.sub(r"[^A-Za-z0-9._ -]+", "_", text).strip().replace(" ", "_")
    return (text or "sample")[:limit]


def _resolve_benchmark_path(path: str) -> str:
    """Resolve a benchmark path so it works regardless of the launch CWD.

    A relative path is tried as-is first (relative to the CWD), then against the
    repo root — so ``inference.benchmark="./benchmark/..."`` in the shipped
    configs resolves whether the process starts at the repo root or elsewhere.

    Args:
        path (str): the configured benchmark path (absolute or relative).

    Returns:
        str: an existing path if one of the candidates resolves, else ``path``
        unchanged (so the caller's ``open`` raises a clear FileNotFoundError).
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    rooted = os.path.join(_REPO_ROOT, path)
    return rooted if os.path.exists(rooted) else path


def read_benchmark(path: str, limit: int | None = None) -> list[dict]:
    """Read a benchmark JSON into per-sample dicts, one output named by its ``id``.

    A single JSON schema serves both modes: an ``image`` makes the item i2v
    (encoded by ``HYConditioning``); omitting it makes the item t2v (Wan's
    default preprocessors ignore ``image``). Every item must carry an ``id`` —
    the output video is named ``{id}.mp4`` (see :meth:`BaseInferencer.run`).

    Args:
        path (str): benchmark JSON holding a list of ``{id, caption, trajectory}``
            items, each optionally with ``image`` (present -> i2v, absent -> t2v).
            Resolved CWD-independently via :func:`_resolve_benchmark_path`.
        limit (int | None): keep only the first ``limit`` items (for smoke runs);
            ``None`` (default) keeps them all.

    Returns:
        list[dict]: one ``{"id", "prompt", "trajectory"[, "image"]}`` dict per
        item, with any ``image`` resolved relative to the JSON file's directory.

    Raises:
        ValueError: if any item is missing ``id``.
    """
    resolved = _resolve_benchmark_path(path)
    with open(resolved, encoding="utf-8") as f:
        items = json.load(f)
    if limit is not None:
        items = items[:limit]
    root = os.path.dirname(os.path.abspath(resolved))
    samples = []
    for idx, item in enumerate(items):
        if "id" not in item:
            raise ValueError(f"benchmark item {idx} is missing required 'id' field")
        sample = {
            "id": item["id"],
            "prompt": item["caption"],
            "trajectory": item.get("trajectory"),
        }
        image = item.get("image")
        if image is not None:
            if not os.path.isabs(image):
                image = os.path.join(root, image)
            sample["image"] = image
        samples.append(sample)
    return samples


class BaseInferencer:
    """Inference engine parallel to :class:`BaseTrainer`."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        if "inference" not in cfg:
            raise KeyError("BaseInferencer requires cfg['inference']")
        # Validate the inference node against the typed schema so a misspelled
        # knob (e.g. ``guidance_scal``) fails loudly here instead of silently
        # falling through a ``.get()`` default. The raw dict is still used for
        # ``.get()`` access below (open ``sampler`` / ``input_preprocessors``).
        parse_inference(cfg["inference"])
        self.inference_cfg = dict(cfg["inference"])

        self.dtype = resolve_dtype(self.inference_cfg.get("dtype", "bfloat16"))
        self.sp_size = int(self.inference_cfg.get("sp_size", 1))
        launched_world_size = int(os.environ.get("WORLD_SIZE") or 1)
        runtime = initialize_runtime(
            sp_size=self.sp_size,
            distributed=launched_world_size > 1 or self.sp_size > 1,
        )
        self.device = runtime.device
        self.rank = runtime.rank
        self.world_size = runtime.world_size

        # world_size = sp_size * dp_size. SP groups are consecutive rank chunks
        # ([0..sp-1], [sp..2sp-1], ...), so the number of SP groups equals the DP
        # degree: ranks in one SP group cooperate on a single sample's sequence,
        # while distinct SP groups process distinct samples for throughput.
        if self.world_size % self.sp_size != 0:
            raise ValueError(
                f"world_size={self.world_size} is not divisible by sp_size={self.sp_size}; "
                "launch --nproc_per_node as a multiple of inference.sp_size"
            )
        self.dp_size = self.world_size // self.sp_size
        self.sp_group_index = self.rank // self.sp_size
        # The first rank of each SP group owns that group's output writes.
        self.is_writer = self.rank % self.sp_size == 0

        # Resolve after distributed init so an unset (None) seed is drawn on rank 0
        # and broadcast. Base seed covers build-time RNG; run() re-seeds per sample.
        self.seed = resolve_seed(self.inference_cfg.get("seed", 0))
        set_seed(self.seed)

        self._build_loop()

    def _build_loop(self) -> None:
        """Load the shared model stack once and build the configured generation loop.

        The heavy components (DiT, text encoder, VAE, adapter, preprocessors) are
        shared across loop families; only the denoising strategy differs. They are
        loaded here and handed to the loop resolved from ``inference.loop``, setting
        :attr:`loop`, :attr:`runtime` and :attr:`input_preprocessors`.
        """
        cfg, gen = self.cfg, self.inference_cfg
        device, dtype = self.device, self.dtype

        loop_cls = _resolve_loop_cls(gen.get("loop", "ARGenerationLoop"))

        model = build_model(cfg["model"])
        checkpoint = gen.get("checkpoint")
        if checkpoint:
            load_model_weights(
                model,
                checkpoint,
                key=gen.get("checkpoint_key", "auto"),
                prefer_ema=bool(gen.get("prefer_ema", False)),
                strict=bool(gen.get("strict", True)),
            )
        model = model.to(device=device, dtype=dtype).eval()

        text_encoder = self.build_text_encoder()
        vae = self.build_vae()
        adapter = self.build_adapter()
        if adapter is not None:
            # Uniform setup: text-CFG families (Wan) store these; families whose
            # conditioning is pre-encoded / whose uncond is a loaded ``.pt`` (HY)
            # no-op. No ``hasattr`` family probing.
            adapter.attach_text_encoder(text_encoder)
            adapter.set_negative_prompt(gen.get("negative_prompt", ""))

        self.loop = loop_cls.from_config(
            gen,
            cfg,
            generator=model,
            vae=vae,
            adapter=adapter,
            sampler=build_sampler(inference_cfg=gen),
        )

        # Built model objects a preprocessor may need at call time (e.g.
        # HYConditioning's image/text/vision encode) ride on the runtime, so
        # preprocessors stay uniform config-built specs with no family branch.
        components: dict = {"vae": vae, "text_encoder": text_encoder}
        if "vision_encoder" in cfg:
            components["vision_encoder"] = build(cfg["vision_encoder"]).to(device).eval()
        self.runtime = InferenceRuntime(
            device=device, dtype=dtype, inference_cfg=gen, components=components
        )
        self.input_preprocessors = build_input_preprocessors(gen)

    def build_text_encoder(self):
        """Build the text encoder onto the inference device (overridable hook).

        Returns:
            nn.Module: the eval-mode text encoder.

        Raises:
            KeyError: if the config has no top-level ``text_encoder`` node.
        """
        text_encoder_cfg = self.cfg.get("text_encoder")
        if text_encoder_cfg is None:
            raise KeyError("inference config needs a top-level text_encoder node")
        return build(text_encoder_cfg).to(self.device).eval()

    def build_vae(self):
        """Build the VAE (or ``None`` when the config omits it) — overridable hook.

        Returns:
            nn.Module | None: the eval-mode VAE, tiling toggled by
            ``inference.vae_tiling``, or ``None`` if no ``vae`` node is configured.
        """
        cfg = self.cfg
        if "vae" not in cfg:
            return None
        vae_cfg = dict(cfg["vae"])
        if "_from_pretrained" in vae_cfg:
            vae = build_model(vae_cfg).to(self.device).eval()
            if self.inference_cfg.get("vae_tiling", True) and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            return vae
        vae_cls = locate(vae_cfg.pop("type"))
        return vae_cls(device=self.device, dtype=self.dtype, **vae_cfg)

    def build_adapter(self):
        """Build the model-call adapter (or ``None``) — overridable hook.

        Returns:
            The built adapter, or ``None`` if no ``adapter`` node is configured.
        """
        adapter_cfg = self.cfg.get("adapter")
        if adapter_cfg is None:
            return None
        adapter_cfg = dict(adapter_cfg)
        adapter_cfg.pop("text_encoder", None)
        return build(adapter_cfg)

    @torch.inference_mode()
    def build_batch(
        self,
        raw: str | dict,
        trajectory: str | list[str] | None = None,
    ) -> dict:
        """Build a single-sample generation batch by running the preprocessor chain.

        Args:
            raw (str | dict): a prompt string or a per-sample dict (needs
                ``prompt``/``prompts``; may carry ``image``, ``trajectory``).
            trajectory (str | list[str] | None): optional camera trajectory
                spec(s) merged into the batch for the CameraTrajectory preprocessor.

        Returns:
            dict: the assembled batch consumed by ``loop.generate``.
        """
        batch = _normalize_generation_input(raw)
        if trajectory is not None:
            batch["trajectory"] = trajectory
        for preprocessor in self.input_preprocessors:
            batch = preprocessor(batch, self.runtime)
        return batch

    def build_benchmark(self) -> list[dict]:
        """Load the benchmark named by the config into per-sample dicts.

        The data source lives in the engine (parallel to the trainer's
        ``build_dataloader(cfg.data)``): the CLI only names the config. Reads
        ``inference.benchmark`` (required) and the optional ``inference.limit``.

        Returns:
            list[dict]: per-sample dicts (see :func:`read_benchmark`).

        Raises:
            KeyError: if ``inference.benchmark`` is unset.
        """
        benchmark = self.inference_cfg.get("benchmark")
        if not benchmark:
            raise KeyError("config requires 'inference.benchmark' (path to the benchmark JSON)")
        limit = self.inference_cfg.get("limit")
        return read_benchmark(benchmark, limit=None if limit is None else int(limit))

    @torch.inference_mode()
    def run(self, samples: list[dict] | None = None) -> None:
        """Generate and write one ``{id}.mp4`` per input sample.

        Samples are sharded across the SP groups (``dp_size`` independent
        workers): ranks in one SP group take the same shard and cooperate on each
        sample's sequence, while that group's first rank writes its own outputs.
        Each sample is seeded by its global index (``seed + index``) so its video
        is identical regardless of ``sp_size`` / ``dp_size`` or shard order.

        Args:
            samples (list[dict] | None): per-sample dicts, each with a required
                ``id`` (names the output ``{id}.mp4``) and a ``prompt``, plus
                optional ``image`` (present -> i2v) and ``trajectory``. When
                ``None`` (default) they are loaded from ``inference.benchmark``
                via :meth:`build_benchmark`.
        """

        if samples is None:
            samples = self.build_benchmark()

        out_dir = self.inference_cfg.get("output_dir", "outputs/infer")
        writer = ResultWriter(out_dir, fps=int(self.inference_cfg.get("fps", 16)))
        if self.is_writer:
            writer.ensure_dir()

        # Stride the samples by SP group so the dp_size groups partition the work
        # with no overlap; the global index is preserved for naming and seeding.
        shard = list(enumerate(samples))[self.sp_group_index :: self.dp_size]
        manifest = []
        for local_pos, (idx, sample) in enumerate(shard):
            sample_id = sample["id"]
            # Per-sample seed decouples a sample's noise from processing order, so
            # sharding across DP workers cannot change its output.
            sample_seed = self.seed + idx
            set_seed(sample_seed)
            # ``id`` is output-naming metadata, not a generation input: keep it off
            # the batch the pipeline sees. ``trajectory`` rides on the dict and is
            # consumed by the CameraTrajectory preprocessor.
            batch_input = {k: v for k, v in sample.items() if k != "id"}
            batch = self.build_batch(batch_input)
            result = self.loop.generate(batch)
            if self.is_writer:
                # ``result`` carries both {'video', 'latents'}; the writer keeps
                # the latents next to the mp4 for latent-space downstream use.
                out_path = writer.write(_sanitize_filename(str(sample_id)), result)
                print(
                    f"[dp{self.sp_group_index} {local_pos + 1}/{len(shard)}] wrote {out_path}",
                    flush=True,
                )
                manifest.append(
                    {
                        "index": idx,
                        "id": sample_id,
                        "prompt": sample.get("prompt"),
                        "trajectory": sample.get("trajectory"),
                        "seed": sample_seed,
                        "video": out_path,
                    }
                )

        manifest = self._gather_manifest(manifest)
        if self.rank == 0:
            self._write_manifest(manifest)

    def _gather_manifest(self, local_manifest: list[dict]) -> list[dict]:
        """Collect per-writer manifest items onto global rank 0, ordered by index.

        Args:
            local_manifest (list[dict]): this rank's manifest items (empty on
                non-writer ranks).

        Returns:
            list[dict]: on rank 0, all items gathered across ranks sorted by
            ``index``; on other ranks the input is returned unchanged.
        """

        if self.world_size == 1:
            return local_manifest

        import torch.distributed as dist

        if not dist.is_initialized():
            return local_manifest
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, local_manifest)
        items = [item for part in gathered if part for item in part]
        items.sort(key=lambda entry: entry["index"])
        return items

    def _write_manifest(self, manifest: list[dict]) -> None:
        out_dir = self.inference_cfg.get("output_dir", "outputs/infer")
        path = os.path.join(out_dir, "manifest.json")
        payload = {
            "inference": self.inference_cfg,
            "items": manifest,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


class ResultWriter:
    """Persist a loop's ``generate`` result (video + latents) under a directory.

    ``loop.generate`` returns ``{'latents', 'video'?}`` — a video only when a VAE
    is attached. The writer decodes the mp4 when a video is present and always
    keeps the latents next to it (as ``{name}.latents.pt``), since latent-space
    downstream use (policy eval, rollout re-feed, latent metrics) is a first-class
    world-model consumer that the old ``write_video``-only path discarded.
    """

    def __init__(self, out_dir: str, *, fps: int = 16) -> None:
        self.out_dir = out_dir
        self.fps = fps

    def ensure_dir(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)

    def write(self, name: str, result: dict) -> str:
        """Write one sample's outputs and return the mp4 (or latents) path.

        Args:
            name (str): sanitized base filename (no extension).
            result (dict): a ``loop.generate`` result with ``latents`` and an
                optional ``video`` ``[B,F,C,H,W]`` tensor in ``[0,1]``.

        Returns:
            str: the mp4 path when a video was written, else the latents path.
        """
        latents = result.get("latents")
        latents_path = os.path.join(self.out_dir, f"{name}.latents.pt")
        if latents is not None:
            torch.save(latents.detach().cpu(), latents_path)
        video = result.get("video")
        if video is None:
            return latents_path
        out_path = os.path.join(self.out_dir, f"{name}.mp4")
        self._save_video(video[0], out_path)
        return out_path

    def _save_video(self, video: torch.Tensor, path: str) -> None:
        """Save one ``[F,C,H,W]`` video tensor in ``[0,1]``."""

        from torchvision.io import write_video

        frames = (video * 255.0).clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8).cpu()
        write_video(path, frames, fps=self.fps)


def build_input_preprocessors(inference_cfg: dict) -> list[InferenceInputPreprocessor]:
    """Build the configured inference input-preprocessor chain."""

    specs = inference_cfg.get("input_preprocessors")
    if specs:
        return [build(spec) for spec in specs]
    return [LatentNoise(), CameraTrajectory()]


def _resolve_loop_cls(name: str) -> type:
    """Resolve ``inference.loop`` to a generation-loop class.

    Accepts a class name exported by :mod:`minwm.engine.inference.loop` (e.g.
    ``"ARGenerationLoop"``) or an explicit ``"module.path:Name"`` import path
    for out-of-tree loops. This mirrors the ``type`` -> :func:`locate` convention
    used for model / vae / adapter nodes, so loop selection is config-driven with
    no ``if/else``.

    Args:
        name (str): the configured loop class name or import path.

    Returns:
        type: the resolved loop class (exposing ``generate`` and ``from_config``).

    Raises:
        NotImplementedError: if the name is neither a known loop class nor a
            resolvable import path.
    """
    if ":" in name:
        return locate(name)
    from .inference import loop as loop_mod

    cls = getattr(loop_mod, name, None)
    if cls is None:
        raise NotImplementedError(
            f"inference.loop={name!r} is not wired; name a loop class in "
            "minwm.engine.inference.loop or use a 'module.path:Name' import path"
        )
    return cls


def _normalize_generation_input(raw: str | dict) -> dict:
    if isinstance(raw, str):
        return {"prompts": [raw]}

    batch = dict(raw)
    if "prompts" not in batch:
        if "prompt" not in batch:
            raise KeyError("generation input requires 'prompt' or 'prompts'")
        batch["prompts"] = [batch.pop("prompt")]
    elif isinstance(batch["prompts"], str):
        batch["prompts"] = [batch["prompts"]]
    else:
        batch["prompts"] = list(batch["prompts"])
    return batch
