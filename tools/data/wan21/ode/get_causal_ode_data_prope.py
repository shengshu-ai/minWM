import argparse
import math
import os
import sys

import torch
import torch.distributed as dist
from tqdm import tqdm

# Add parent directory to path for wan_utils imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from wan_utils.scheduler import FlowMatchScheduler
from wan_utils.wan_wrapper import WanDiffusionWrapper, Wan21TextEncoder

from minwm.data.datasets import CameraLatentLMDBDataset
from minwm.distributed import maybe_init_distributed_environment_and_model_parallel


def init_model(device):
    model = WanDiffusionWrapper(is_causal=True, use_camera=True).to(device).to(torch.float32)
    model.model.num_frame_per_block = 4  # PRoPE: 4 frames per block
    encoder = Wan21TextEncoder().to(device).to(torch.float32)

    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=48, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    sample_neg_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )

    unconditional_dict = encoder(text_prompts=[sample_neg_prompt])

    return model, encoder, scheduler, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--rawdata_path", type=str)
    parser.add_argument("--generator_ckpt", type=str)
    parser.add_argument("--guidance_scale", type=float, default=6.0)

    args = parser.parse_args()

    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    global_rank = dist.get_rank()

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, encoder, scheduler, unconditional_dict = init_model(device=device)
    state_dict = torch.load(args.generator_ckpt, map_location="cpu")

    # Support three checkpoint layouts produced by export_checkpoint.py:
    #   {"generator": ...}  – exported with --model-key generator
    #   {"model": ...}      – exported with --model-key model (TF/SFT default)
    #   bare state_dict     – raw parameter dict with no wrapping key
    if "generator" in state_dict:
        gen_sd = state_dict["generator"]
    elif "model" in state_dict:
        gen_sd = state_dict["model"]
    else:
        gen_sd = state_dict
    fixed = {}
    for k, v in gen_sd.items():
        if k.startswith("model._fsdp_wrapped_module."):
            k = k.replace("model._fsdp_wrapped_module.", "", 1)
        if k.startswith("model."):
            k = k.replace("model.", "", 1)
        fixed[k] = v
    state_dict = fixed
    model.model.load_state_dict(state_dict, strict=True)

    dataset = CameraLatentLMDBDataset(args.rawdata_path)

    if global_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    dist.barrier()  # Wait for directory creation

    total_steps = int(math.ceil(len(dataset) / dist.get_world_size()))
    for index in tqdm(
        range(total_steps),
        disable=(global_rank != 0),
    ):
        prompt_index = index * dist.get_world_size() + global_rank
        if prompt_index >= len(dataset):
            continue

        sample = dataset[prompt_index]
        prompt = sample["prompts"]

        clean_latent = sample["clean_latent"].to(device).unsqueeze(0)  # (1, F, C, H, W)
        viewmats = sample["viewmats"].to(device).unsqueeze(0)  # (1, F, 4, 4)
        Ks = sample["Ks"].to(device).unsqueeze(0)  # (1, F, 3, 3)

        num_latent_frames = clean_latent.shape[1]

        conditional_dict = encoder(text_prompts=prompt)

        latents = torch.randn(
            [1, num_latent_frames, 16, 60, 104], dtype=torch.float32, device=device
        )

        noisy_input = []

        for progress_id, t in enumerate(tqdm(scheduler.timesteps, disable=(global_rank != 0))):
            timestep = t * torch.ones([1, num_latent_frames], device=device, dtype=torch.float32)
            noisy_input.append(latents)
            f_cond, x0_pred_cond = model(
                latents, conditional_dict, timestep, clean_x=clean_latent, viewmats=viewmats, Ks=Ks
            )

            f_uncond, x0_pred_uncond = model(
                latents,
                unconditional_dict,
                timestep,
                clean_x=clean_latent,
                viewmats=viewmats,
                Ks=Ks,
            )

            flow_pred = f_uncond + args.guidance_scale * (f_cond - f_uncond)

            latents = scheduler.step(
                flow_pred.flatten(0, 1), timestep.flatten(0, 1), latents.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        noisy_input.append(latents)
        noisy_input.append(clean_latent)

        noisy_inputs = torch.stack(noisy_input, dim=1)

        noisy_inputs = noisy_inputs[:, [0, 12, 24, 36, -2, -1]]

        stored_data = noisy_inputs

        torch.save(
            {
                "prompt": prompt,
                "latents": stored_data.cpu().detach(),  # (1, 6, F, C, H, W)
                "viewmats": viewmats.cpu().detach(),  # (1, F, 4, 4)
                "Ks": Ks.cpu().detach(),  # (1, F, 3, 3)
            },
            os.path.join(args.output_folder, f"{prompt_index:05d}.pt"),
        )

    dist.barrier()


if __name__ == "__main__":
    main()
