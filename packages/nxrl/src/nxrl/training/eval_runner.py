"""Eval-GIF rollout.

Drives a frozen world model with a policy + receding goal, decodes each
predicted latent through the VAE, and stitches the result into an animated
GIF. Negative ``start_frame`` zero-pads the prefix so the seed lands at
the end of the context window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from nxwm.inference.flow_matching import FlowMatchingSampler
from PIL import Image

from nxrl.serve.server import PolicyServer

LATENT_SCALE = 0.18215


def _decode_latent(latent: torch.Tensor, vae) -> Image.Image:
    z = latent.unsqueeze(0) if latent.dim() == 3 else latent
    decoded = vae.decode((z / LATENT_SCALE).half()).sample
    img = ((decoded + 1.0) / 2.0).clamp(0, 1)
    img_np = img.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    return Image.fromarray((img_np * 255).astype(np.uint8))


@torch.no_grad()
def generate_eval_gif(
    *,
    policy_server: PolicyServer,
    world_model: torch.nn.Module,
    vae,
    seed_episode: Path,
    start_frame: int,
    frames: int,
    goal_offset: int,
    flow_steps: int,
    cfg_scale: float,
    output_path: Path,
    device: torch.device,
) -> int:
    data = np.load(seed_episode, mmap_mode="r")
    latents = torch.from_numpy(data["latents"][:].astype(np.float32))
    actions_data = torch.from_numpy(data["actions"][:].astype(np.float32))

    bc_seq_len = policy_server.sequence_length
    total_frames = len(latents)

    pad_frames = 0
    if start_frame < 0:
        pad_frames = -start_frame
        start_frame = 0

    real_context_len = bc_seq_len - pad_frames
    min_needed = real_context_len + frames + goal_offset
    if start_frame + min_needed > total_frames:
        start_frame = max(0, total_frames - min_needed)

    available_rollout = total_frames - start_frame - real_context_len
    if available_rollout < frames:
        frames = max(1, available_rollout)

    real_latents = latents[start_frame : start_frame + real_context_len]
    real_actions = actions_data[start_frame : start_frame + real_context_len]
    if pad_frames > 0:
        lat_shape = latents.shape[1:]
        act_shape = actions_data.shape[1:]
        pad_lats = torch.zeros(pad_frames, *lat_shape)
        pad_acts = torch.zeros(pad_frames, *act_shape)
        context_latents = torch.cat([pad_lats, real_latents], dim=0)
        context_actions = torch.cat([pad_acts, real_actions], dim=0)
    else:
        context_latents = real_latents
        context_actions = real_actions

    policy_obs = context_latents.unsqueeze(0).to(device)
    wm_seq_len = world_model.config.seq_len  # type: ignore
    wm_initial_latents = context_latents[-wm_seq_len:].to(device)  # type: ignore
    wm_initial_actions = context_actions[-wm_seq_len:].to(device)  # type: ignore

    initial_goal_idx = start_frame + real_context_len + goal_offset
    initial_goal = latents[min(initial_goal_idx, total_frames - 1)].to(device)
    state = world_model.init_rollout_state(wm_initial_latents, wm_initial_actions, initial_goal)  # type: ignore
    sampler = FlowMatchingSampler()

    gif_frames: list[Image.Image] = []
    for i in range(min(5, wm_seq_len)):  # type: ignore
        gif_frames.append(_decode_latent(wm_initial_latents[-(i + 1)].to(device), vae))
    gif_frames.reverse()

    for step in range(frames):
        action_np = policy_server.predict(policy_obs.squeeze(0).cpu().numpy())
        action = torch.from_numpy(action_np).to(device)

        goal_idx = start_frame + real_context_len + step + goal_offset
        if goal_idx < total_frames:
            new_goal = latents[goal_idx].to(device)
            state = world_model.update_goal(state, new_goal)  # type: ignore

        new_state, predicted = world_model.step_rollout(  # type: ignore
            state, action, sampler=sampler, flow_steps=flow_steps, cfg_scale=cfg_scale
        )
        state = new_state

        gif_frames.append(_decode_latent(predicted, vae))
        policy_obs = torch.cat([policy_obs[:, 1:], predicted.unsqueeze(0).unsqueeze(0)], dim=1)

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=33,
        loop=0,
    )
    return len(gif_frames)
