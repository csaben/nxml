"""GIF export + wandb video logging for training visualizations.

Replaces MIRA's ffmpeg mp4 pipeline (``training/visualization.py``) with the PIL-GIF +
``wandb.Video`` pattern nxwm uses (``nxwm/training/eval_gif.py``) — no system ffmpeg needed.
``video_to_uint8`` and ``draw_text_on_first_frame`` are ports of MIRA's helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor


def video_to_uint8(video: Tensor) -> Tensor:
    """Convert a floating-point video in [0, 1] to uint8 in [0, 255] (no-op if already uint8)."""
    if video.dtype != torch.uint8:
        if not torch.is_floating_point(video):
            raise ValueError(f"Expected uint8 or floating point video tensor, got dtype {video.dtype}")

        # Cast to float32 first so that rounding in reduced-precision dtypes (bfloat16 / float16)
        # does not push values outside [0, 255].
        video = video.float()
        video = torch.clamp(video * 255.0, 0, 255).to(torch.uint8)
    return video


def draw_text_on_first_frame(video: Tensor, texts: list[str]) -> Tensor:
    """Draw text labels on the first frame of each batch item.

    Args:
        video: (B, T, C, H, W) uint8 tensor.
        texts: list of length B with text to draw on each item's first frame.
    """
    video = video.clone()
    for i, text in enumerate(texts):
        frame = video[i, 0]  # (C, H, W)
        img = Image.fromarray(rearrange(frame.cpu().numpy(), "c h w -> h w c"))
        draw = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        padding = 4
        draw.rectangle([4, 4, 4 + text_w + 2 * padding, 4 + text_h + 2 * padding], fill=(0, 0, 0, 180))
        draw.text((4 + padding, 4 + padding), text, fill="white", font=font)
        video[i, 0] = torch.from_numpy(rearrange(np.array(img)[..., :3], "h w c -> c h w"))
    return video


def add_prediction_border(video: Tensor, n_context_frames: int, width: int = 3) -> Tensor:
    """Draw a coloured border on the predicted (non-context) frames of a rollout.

    Args:
        video: (B, T, C, H, W) uint8.
        n_context_frames: frames [0, n) are context (no border); the rest get an orange border.
    """
    video = video.clone()
    color = torch.tensor([255, 140, 0], dtype=torch.uint8, device=video.device).view(1, 1, 3, 1, 1)
    pred = video[:, n_context_frames:]
    pred[..., :width, :] = color
    pred[..., -width:, :] = color
    pred[..., :, :width] = color
    pred[..., :, -width:] = color
    return video


def save_gif(video: Tensor, path: str | Path, fps: float) -> Path:
    """Write a ``(T, C, H, W)`` uint8 video tensor as an animated GIF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        Image.fromarray(rearrange(frame.cpu().numpy(), "c h w -> h w c")) for frame in video
    ]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
    )
    return path


def log_wandb_video(wandb_run, key: str, path: str | Path, fps: float, step: int) -> None:
    """Log a saved GIF to wandb (no-op when ``wandb_run`` is None)."""
    if wandb_run is None:
        return
    import wandb

    wandb_run.log({key: wandb.Video(str(path), format="gif")}, step=step)
