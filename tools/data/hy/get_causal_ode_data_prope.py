"""Generate causal ODE-regression latents for HunyuanVideo 1.5 (PRoPE).

minwm-native rewrite of the legacy ``ar_ode_sampling_entry.py`` pipeline. For each
SFT-encoded ``.pt`` sample it runs a 48-step CFG flow-matching ODE solve on the
AR transformer (teacher-forced on the clean latent, PRoPE camera conditioning),
then stores a 6-snapshot trajectory in the same layout the DMD stage consumes.

The model is built straight from the Stage-1 checkpoint directory
(``config.json`` + ``diffusion_pytorch_model.safetensors``); ``use_prope`` is
forced on because the Action2V checkpoint carries per-block ProPE projections
that the bare config omits. Attention silently falls back ``flash -> torch`` on
GPUs without a flash kernel, so the same script runs on B200 and A-series cards.

Usage::

    torchrun --nproc_per_node=8 tools/data/hy/get_causal_ode_data_prope.py \\
        --generator_ckpt ckpts/HY15/Action2V/stage1_ar_tf \\
        --rawdata_path dataset/HY15/Action2V/latents \\
        --output_folder dataset/HY15/Action2V/ode_latents \\
        --neg_prompt dataset/others/HY/Action2V/hunyuan_neg_prompt.pt \\
        --neg_byt5 dataset/others/HY/Action2V/hunyuan_neg_byt5_prompt.pt \\
        --guidance_scale 5.0
"""

import argparse
import glob
import math
import os

import torch
import torch.distributed as dist
from tqdm import tqdm

from minwm.data.datasets.geometry import build_viewmats_and_Ks
from minwm.modeling.hy15.causal import ARHunyuanVideo_1_5_DiffusionTransformer

ODE_SHIFT = 5.0
ODE_STEPS = 48
NUM_TRAIN_TIMESTEPS = 1000
SNAPSHOT_INDICES = [0, 12, 24, 36, -2, -1]


def launch_distributed_job() -> tuple[int, int, int]:
    """Init the NCCL process group from torchrun env vars.

    Returns:
        tuple[int, int, int]: ``(global_rank, local_rank, world_size)``.
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, local_rank, world_size


def build_ode_sigmas(shift: float, steps: int, device: torch.device) -> torch.Tensor:
    """Build the shifted flow-matching sigma table (length ``steps + 1``).

    Mirrors ``FlowMatchDiscreteScheduler`` with ``reverse=True``: a linear
    ``1 -> 0`` ramp pushed through the SD3 time-shift. The final entry is exactly
    0 so the last Euler step lands on the clean sample.

    Args:
        shift (float): SD3 time-shift strength.
        steps (int): number of inference steps.
        device (torch.device): target device for the table.

    Returns:
        Tensor: sigma schedule, shape ``[steps + 1]``.
    """
    sigmas = torch.linspace(1, 0, steps + 1, device=device)
    return (shift * sigmas) / (1 + (shift - 1) * sigmas)


def get_task_mask(latent_target_length: int, device: torch.device) -> torch.Tensor:
    """i2v task mask: 1.0 on the first latent frame, 0.0 elsewhere."""
    mask = torch.zeros(latent_target_length, device=device)
    mask[0] = 1.0
    return mask


def prepare_cond_latents(
    image_cond: torch.Tensor, latents: torch.Tensor, task_mask: torch.Tensor
) -> torch.Tensor:
    """Build the i2v conditioning channels appended to the noisy/clean input.

    The image latent is broadcast across time with only frame 0 kept (the rest
    zeroed), then a per-frame binary mask channel (1 on conditioned frames) is
    concatenated, yielding 33 extra channels for a 32-channel latent.

    Args:
        image_cond (Tensor): first-frame condition ``[B, C, 1, H, W]``.
        latents (Tensor): reference latents ``[B, C, T, H, W]`` (for shape/T).
        task_mask (Tensor): per-frame 0/1 mask ``[T]``.

    Returns:
        Tensor: conditioning latents ``[B, C + 1, T, H, W]``.
    """
    B, C, T, H, W = latents.shape
    latents_concat = image_cond.repeat(1, 1, T, 1, 1)
    latents_concat[:, :, 1:, :, :] = 0.0

    mask_concat = torch.zeros(B, 1, T, H, W, device=latents.device, dtype=latents.dtype)
    cond_frames = torch.nonzero(task_mask).squeeze(1)
    mask_concat[:, :, cond_frames] = 1.0

    return torch.cat([latents_concat, mask_concat], dim=1)


def init_model(ckpt_dir: str, device: torch.device) -> ARHunyuanVideo_1_5_DiffusionTransformer:
    """Build the AR transformer from a Stage-1 checkpoint dir with ProPE on.

    Args:
        ckpt_dir (str): directory holding ``config.json`` +
            ``diffusion_pytorch_model.safetensors``.
        device (torch.device): target device.

    Returns:
        ARHunyuanVideo_1_5_DiffusionTransformer: eval-mode bf16 model.
    """
    model = ARHunyuanVideo_1_5_DiffusionTransformer.from_pretrained(ckpt_dir, use_prope=True)
    return model.to(device).to(torch.bfloat16).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator_ckpt",
        type=str,
        required=True,
        help="Stage-1 checkpoint dir (config.json + safetensors).",
    )
    parser.add_argument(
        "--rawdata_path", type=str, required=True, help="Directory of SFT-encoded .pt samples."
    )
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument(
        "--neg_prompt",
        type=str,
        required=True,
        help=".pt with negative_prompt_embeds / negative_prompt_mask.",
    )
    parser.add_argument(
        "--neg_byt5",
        type=str,
        required=True,
        help=".pt with byt5_text_states / byt5_text_mask (negative).",
    )
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    args = parser.parse_args()

    global_rank, local_rank, world_size = launch_distributed_job()
    device = torch.device(f"cuda:{local_rank}")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = init_model(args.generator_ckpt, device)
    sigmas = build_ode_sigmas(ODE_SHIFT, ODE_STEPS, device)
    timesteps = (sigmas[:-1] * NUM_TRAIN_TIMESTEPS).to(torch.float32)

    neg = torch.load(args.neg_prompt, map_location="cpu", weights_only=False)
    neg_byt5 = torch.load(args.neg_byt5, map_location="cpu", weights_only=False)
    neg_text_states = neg["negative_prompt_embeds"].to(device, torch.bfloat16)
    neg_text_mask = neg["negative_prompt_mask"].to(device)
    neg_extra = {
        "byt5_text_states": neg_byt5["byt5_text_states"].to(device, torch.bfloat16),
        "byt5_text_mask": neg_byt5["byt5_text_mask"].to(device),
    }

    pt_files = sorted(glob.glob(os.path.join(args.rawdata_path, "*.pt")))
    if global_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    dist.barrier()

    total_steps = int(math.ceil(len(pt_files) / world_size))
    for index in tqdm(range(total_steps), disable=(global_rank != 0)):
        file_index = index * world_size + global_rank
        if file_index >= len(pt_files):
            continue

        sample = torch.load(pt_files[file_index], map_location="cpu", weights_only=False)

        clean_latent = sample["latent"].to(device, torch.bfloat16)  # [1, 32, T, H, W]
        image_cond = sample["image_cond"].to(device, torch.bfloat16)  # [1, 32, 1, H, W]
        text_states = sample["prompt_embeds"].to(device, torch.bfloat16)  # [1, L, 3584]
        prompt_mask = sample["prompt_mask"].to(device)  # [1, L]
        vision_states = sample["vision_states"].to(device, torch.bfloat16)
        pos_extra = {
            "byt5_text_states": sample["byt5_text_states"].to(device, torch.bfloat16),
            "byt5_text_mask": sample["byt5_text_mask"].to(device),
        }

        B, C, T, H, W = clean_latent.shape

        viewmats_np, Ks_np = build_viewmats_and_Ks(
            sample["intrinsics"].numpy(), sample["poses"].numpy()
        )
        viewmats = torch.from_numpy(viewmats_np).to(device).unsqueeze(0)  # float32
        Ks = torch.from_numpy(Ks_np).to(device).unsqueeze(0)  # float32

        task_mask = get_task_mask(T, device)
        cond_input = prepare_cond_latents(image_cond, clean_latent, task_mask)
        clean_hidden = torch.cat([clean_latent, cond_input], dim=1)

        x = torch.randn_like(clean_latent)
        aug_timesteps = torch.zeros(B * T, device=device, dtype=torch.bfloat16)
        timestep_txt = torch.zeros(B, device=device, dtype=torch.bfloat16)

        trajectory = []
        for step_id, t in enumerate(timesteps):
            timesteps_in = t.expand(B * T).to(device, torch.bfloat16)
            trajectory.append(x)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cond_pred = model(
                    bi_inference=True,
                    hidden_states=torch.cat([x, cond_input], dim=1),
                    timestep=timesteps_in,
                    timestep_txt=timestep_txt,
                    text_states=text_states,
                    text_states_2=None,
                    encoder_attention_mask=prompt_mask,
                    vision_states=vision_states,
                    mask_type="i2v",
                    extra_kwargs=pos_extra,
                    clean_x=clean_hidden,
                    aug_timesteps=aug_timesteps,
                    viewmats=viewmats,
                    Ks=Ks,
                )[0]

                if args.guidance_scale > 1:
                    uncond_pred = model(
                        bi_inference=True,
                        hidden_states=torch.cat([x, cond_input], dim=1),
                        timestep=timesteps_in,
                        timestep_txt=timestep_txt,
                        text_states=neg_text_states,
                        text_states_2=None,
                        encoder_attention_mask=neg_text_mask,
                        vision_states=vision_states,
                        mask_type="i2v",
                        extra_kwargs=neg_extra,
                        clean_x=clean_hidden,
                        aug_timesteps=aug_timesteps,
                        viewmats=viewmats,
                        Ks=Ks,
                    )[0]
                    pred = uncond_pred + args.guidance_scale * (cond_pred - uncond_pred)
                else:
                    pred = cond_pred

            dt = sigmas[step_id + 1] - sigmas[step_id]
            x = x.to(torch.float32) + pred.to(torch.float32) * dt
            x = x.to(torch.bfloat16)

        trajectory.append(x)
        trajectory.append(clean_latent)
        trajectory = torch.stack(trajectory, dim=1)  # [B, ODE_STEPS+2, C, T, H, W]
        noisy_inputs = trajectory[:, SNAPSHOT_INDICES]  # [B, 6, C, T, H, W]

        latents_dict = {
            "latent": clean_latent.cpu().detach(),
            "prompt_embeds": text_states.cpu().detach(),
            "prompt_mask": prompt_mask.cpu().detach(),
            "image_cond": image_cond.cpu().detach(),
            "vision_states": vision_states.cpu().detach(),
            "byt5_text_states": pos_extra["byt5_text_states"].cpu().detach(),
            "byt5_text_mask": pos_extra["byt5_text_mask"].cpu().detach(),
            "ode_trajectory": noisy_inputs.cpu().detach(),
            "viewmats": viewmats.cpu().detach(),
            "Ks": Ks.cpu().detach(),
        }

        out_name = os.path.basename(pt_files[file_index])
        torch.save(latents_dict, os.path.join(args.output_folder, out_name))

    dist.barrier()


if __name__ == "__main__":
    main()
