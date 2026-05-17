"""Visual sanity-check helper: decode raw episode latents to a GIF."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


@torch.no_grad()
def save_episode_gif(
    npz_path: Path,
    *,
    start_frame: int = 0,
    num_frames: int = 60,
    output_path: str = "episode_sanity_check.gif",
    vae: AutoencoderKL | None = None,
    device: str = "cuda",
    fps: int = 10,
) -> None:
    from PIL import Image

    if vae is None:
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

        vae_path = os.getenv("vae_path", "stabilityai/sd-vae-ft-mse")
        vae = AutoencoderKL.from_pretrained(vae_path).to(device).half()  # type: ignore[no-untyped-call]
        vae.eval()

    data = np.load(npz_path, mmap_mode="r")
    total_frames = data["latents"].shape[0]
    end_frame = min(start_frame + num_frames, total_frames)

    frames = []
    for i in range(start_frame, end_frame):
        latent = torch.from_numpy(data["latents"][i : i + 1]).to(device).half()
        latent = latent / 0.18215
        decoded = vae.decode(latent).sample  # type: ignore[arg-type, attr-defined]
        img = (decoded + 1.0) / 2.0
        img = img.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        frames.append(Image.fromarray(img))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )
