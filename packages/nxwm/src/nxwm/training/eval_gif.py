"""Eval-rollout GIF generation for training-time visualization.

Drives ``model.step_rollout`` + ``FlowMatchingSampler`` for N frames seeded
from a fixed val sample, decodes each predicted latent via the VAE, and
stitches an animated GIF. Two seeding modes:

  - ``repeat_last``: pulls one val batch, uses its first sample's
    observations/actions as context, holds ``goals[0, 0]`` constant for
    every step. Cheap and deterministic across epochs since the val
    dataloader is unshuffled — same scene every time, easy to eyeball
    drift.

  - ``from_mmap``: bypasses the dataloader and reads the val dataset's
    mmap'd ``.npz`` directly so we have access to the full episode. Uses
    a receding goal (fresh from ``goal_offset`` frames ahead each step)
    and the dataset's recorded actions, matching what ``nxwm rollout``
    produces. Slightly more code, slightly more I/O, but apples-to-apples
    with the offline rollout subcommand.

The trainer wires this in after ``val_epoch`` when
``config.eval_gif_every_n_epochs > 0``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nxwm.inference.flow_matching import FlowMatchingSampler
from nxwm.inference.vae import LATENT_SCALE


def _decode_latent(z: torch.Tensor, vae) -> Image.Image:
    """Latent (C, H, W) or (1, C, H, W) -> RGB PIL image.

    Casts the input to whatever dtype the VAE's parameters are stored in,
    so this works whether the launcher loaded the VAE in fp32 (lpips
    convention) or fp16 (inference convention).
    """
    if z.dim() == 3:
        z = z.unsqueeze(0)
    target_dtype = next(vae.parameters()).dtype
    decoded = vae.decode((z / LATENT_SCALE).to(target_dtype)).sample
    img = ((decoded + 1.0) / 2.0).clamp(0, 1)
    arr = img.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def _save_gif(frames: list[Image.Image], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=33,
        loop=0,
    )
    return output_path


@torch.no_grad()
def _from_repeat_last(
    model: torch.nn.Module,
    vae,
    val_loader,
    device: torch.device,
    output_path: Path,
    *,
    frames: int,
    flow_steps: int,
    cfg_scale: float,
) -> Path:
    batch = next(iter(val_loader))
    obs = batch["observations"][0].to(device)
    actions = batch["actions"][0].to(device)
    goal = batch["goals"][0, 0].to(device)
    future_actions = batch["future_actions"][0].to(device)

    wm_seq_len = model.config.seq_len  # type: ignore
    initial_latents = obs[-wm_seq_len:]  # type: ignore
    initial_actions = actions[-wm_seq_len:]  # type: ignore
    state = model.init_rollout_state(initial_latents, initial_actions, goal)  # type: ignore
    sampler = FlowMatchingSampler()

    gif_frames: list[Image.Image] = []
    for i in range(min(5, wm_seq_len)):  # type: ignore
        gif_frames.append(_decode_latent(initial_latents[-(i + 1)], vae))
    gif_frames.reverse()

    k = future_actions.shape[0]
    for step in range(frames):
        action = future_actions[min(step, k - 1)]
        state, predicted = model.step_rollout(  # type: ignore
            state, action, sampler=sampler, flow_steps=flow_steps, cfg_scale=cfg_scale
        )
        gif_frames.append(_decode_latent(predicted, vae))

    return _save_gif(gif_frames, output_path)


@torch.no_grad()
def _from_mmap(
    model: torch.nn.Module,
    vae,
    val_loader,
    device: torch.device,
    output_path: Path,
    *,
    frames: int,
    flow_steps: int,
    cfg_scale: float,
    episode_idx: int,
    start_frame: int,
) -> Path:
    ds = val_loader.dataset
    if not hasattr(ds, "mapped_data") or episode_idx not in ds.mapped_data:
        # Dataset doesn't expose mmap; fall back to the cheap path.
        return _from_repeat_last(
            model,
            vae,
            val_loader,
            device,
            output_path,
            frames=frames,
            flow_steps=flow_steps,
            cfg_scale=cfg_scale,
        )

    data = ds.mapped_data[episode_idx]
    latents = torch.from_numpy(np.asarray(data["latents"])).float()
    actions = torch.from_numpy(np.asarray(data["actions"])).float()
    total = len(latents)
    wm_seq_len = model.config.seq_len  # type: ignore
    goal_offset = int(getattr(ds, "goal_offset", 30))

    # Clamp start so the rollout fits within the episode.
    start = max(0, min(start_frame, total - wm_seq_len - frames - goal_offset))  # type: ignore
    initial_latents = latents[start : start + wm_seq_len].to(device)  # type: ignore
    initial_actions = actions[start : start + wm_seq_len].to(device)  # type: ignore
    initial_goal_idx = start + wm_seq_len + goal_offset  # type: ignore
    initial_goal = latents[min(initial_goal_idx, total - 1)].to(device)

    state = model.init_rollout_state(initial_latents, initial_actions, initial_goal)  # type: ignore
    sampler = FlowMatchingSampler()

    gif_frames: list[Image.Image] = []
    for i in range(min(5, wm_seq_len)):  # type: ignore
        gif_frames.append(_decode_latent(initial_latents[-(i + 1)], vae))
    gif_frames.reverse()

    for step in range(frames):
        goal_idx = start + wm_seq_len + step + goal_offset  # type: ignore
        if goal_idx < total:
            state = model.update_goal(state, latents[goal_idx].to(device))  # type: ignore

        act_idx = start + wm_seq_len + step  # type: ignore
        action = (
            actions[act_idx].to(device)
            if act_idx < total
            else torch.zeros(actions.shape[1], device=device)
        )

        state, predicted = model.step_rollout(  # type: ignore
            state, action, sampler=sampler, flow_steps=flow_steps, cfg_scale=cfg_scale
        )
        gif_frames.append(_decode_latent(predicted, vae))

    return _save_gif(gif_frames, output_path)


@torch.no_grad()
def _from_repeat_last_fa_dit(
    model: torch.nn.Module,
    vae,
    val_loader,
    device: torch.device,
    output_path: Path,
    *,
    frames: int,
    flow_steps: int,
) -> Path:
    """fa_dit-shaped eval-gif: pull one val batch, hold its anchor constant,
    drive ``step_rollout`` with the recorded ``future_actions`` for ``frames``
    steps. ``cfg_scale`` is unused here — fa_dit's ``step_rollout`` doesn't
    combine conditional/unconditional samples (no CFG mixing built into the
    rectified-flow loop).
    """
    batch = next(iter(val_loader))
    past_latents = batch["observations"][0].to(device)
    past_poses = batch["actions"][0].to(device)
    future_latents = batch["future_latents"][0].to(device)
    future_poses = batch["future_poses"][0].to(device)
    future_actions = batch["future_actions"][0].to(device)
    target_pose = future_actions[0]

    state = model.init_rollout_state(  # type: ignore[attr-defined]
        past_latents, past_poses, future_latents, future_poses, target_pose
    )

    gif_frames: list[Image.Image] = []
    for i in range(min(5, past_latents.shape[0])):
        gif_frames.append(_decode_latent(past_latents[-(i + 1)], vae))
    gif_frames.reverse()

    k = future_actions.shape[0]
    for step in range(frames):
        action = future_actions[min(step, k - 1)]
        state, predicted, _plan = model.step_rollout(  # type: ignore[attr-defined]
            state, action, flow_steps=flow_steps
        )
        gif_frames.append(_decode_latent(predicted, vae))

    return _save_gif(gif_frames, output_path)


@torch.no_grad()
def _from_mmap_fa_dit(
    model: torch.nn.Module,
    vae,
    val_loader,
    device: torch.device,
    output_path: Path,
    *,
    frames: int,
    flow_steps: int,
    episode_idx: int,
    start_frame: int,
) -> Path:
    """fa_dit-shaped eval-gif over the val mmap. Same scene-fitting strategy
    as :func:`_from_mmap` for dit_v1: pull a contiguous window straight out
    of the on-disk episode so the rollout matches what ``nxwm rollout`` /
    ``nxwm serve`` would produce.
    """
    ds = val_loader.dataset
    if not hasattr(ds, "mapped_data") or episode_idx not in ds.mapped_data:
        return _from_repeat_last_fa_dit(
            model, vae, val_loader, device, output_path,
            frames=frames, flow_steps=flow_steps,
        )

    data = ds.mapped_data[episode_idx]
    latents = torch.from_numpy(np.asarray(data["latents"])).float()
    actions = torch.from_numpy(np.asarray(data["actions"])).float()
    total = len(latents)
    cfg = model.config  # type: ignore[attr-defined]
    T_past = cfg.past_seq_len
    F_a = cfg.future_anchor_len
    goal_offset = int(getattr(ds, "goal_offset", 30))

    # Clamp start so the past window, rollout, and anchor all fit in-episode.
    needed = T_past + frames + max(goal_offset + F_a, 1)
    start = max(0, min(start_frame, total - needed))

    past_latents = latents[start : start + T_past].to(device)
    past_poses = actions[start : start + T_past].to(device)
    anchor_start = start + T_past + goal_offset
    future_latents = latents[anchor_start : anchor_start + F_a].to(device)
    future_poses = actions[anchor_start : anchor_start + F_a].to(device)
    target_pose = actions[start + T_past].to(device)

    state = model.init_rollout_state(  # type: ignore[attr-defined]
        past_latents, past_poses, future_latents, future_poses, target_pose
    )

    gif_frames: list[Image.Image] = []
    for i in range(min(5, T_past)):
        gif_frames.append(_decode_latent(past_latents[-(i + 1)], vae))
    gif_frames.reverse()

    for step in range(frames):
        act_idx = start + T_past + step
        action = (
            actions[act_idx].to(device)
            if act_idx < total
            else torch.zeros(actions.shape[1], device=device)
        )
        state, predicted, _plan = model.step_rollout(  # type: ignore[attr-defined]
            state, action, flow_steps=flow_steps
        )
        gif_frames.append(_decode_latent(predicted, vae))

    return _save_gif(gif_frames, output_path)


def _is_fa_dit(model: torch.nn.Module) -> bool:
    """Identify fa_dit by config shape rather than isinstance to keep this
    module free of the fa_dit import (kept lazy where possible).
    """
    cfg = getattr(model, "config", None)
    return cfg is not None and getattr(cfg, "future_anchor_len", 0) > 0


def generate_eval_gif(
    *,
    model: torch.nn.Module,
    vae,
    val_loader,
    device: torch.device,
    output_path: Path,
    frames: int = 90,
    flow_steps: int = 5,
    cfg_scale: float = 1.0,
    mode: str = "repeat_last",
    episode_idx: int = 0,
    start_frame: int = 0,
) -> Path:
    is_fa_dit = _is_fa_dit(model)
    if mode == "from_mmap":
        if is_fa_dit:
            return _from_mmap_fa_dit(
                model, vae, val_loader, device, output_path,
                frames=frames, flow_steps=flow_steps,
                episode_idx=episode_idx, start_frame=start_frame,
            )
        return _from_mmap(
            model,
            vae,
            val_loader,
            device,
            output_path,
            frames=frames,
            flow_steps=flow_steps,
            cfg_scale=cfg_scale,
            episode_idx=episode_idx,
            start_frame=start_frame,
        )
    if mode != "repeat_last":
        raise ValueError(f"unknown eval_gif_mode {mode!r} (expected repeat_last|from_mmap)")
    if is_fa_dit:
        return _from_repeat_last_fa_dit(
            model, vae, val_loader, device, output_path,
            frames=frames, flow_steps=flow_steps,
        )
    return _from_repeat_last(
        model,
        vae,
        val_loader,
        device,
        output_path,
        frames=frames,
        flow_steps=flow_steps,
        cfg_scale=cfg_scale,
    )
