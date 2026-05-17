from __future__ import annotations

import warnings
from typing import Final

import cv2
import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

LATENT_SCALE: Final[float] = 0.18215


def load_vae(
    path: str = "stabilityai/sd-vae-ft-mse", device: torch.device | str = "cuda"
) -> AutoencoderKL:
    # diffusers still calls hf_hub_download(local_dir_use_symlinks=...) which
    # huggingface_hub deprecated. Suppress just that one warning here — it's
    # noise we can't fix until diffusers ships a patched release.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*local_dir_use_symlinks.*",
            category=UserWarning,
        )
        vae = AutoencoderKL.from_pretrained(path).to(device).half()
    vae.eval()
    return vae


@torch.no_grad()
def encode_jpeg(jpeg_bytes: bytes, vae: AutoencoderKL, device: torch.device | str) -> torch.Tensor:
    """JPEG → scaled latent (C, H, W). Verbatim from old _vae_encode."""
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 128))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    latent = vae.encode(tensor.unsqueeze(0).half().to(device)).latent_dist.mode()  # type: ignore[union-attr]
    return latent.squeeze(0).float() * LATENT_SCALE


@torch.no_grad()
def decode_to_jpeg(latent: torch.Tensor, vae: AutoencoderKL, jpeg_quality: int = 85) -> bytes:
    """Scaled latent → JPEG bytes."""
    _, jpeg = decode_to_jpeg_and_bgr(latent, vae, jpeg_quality=jpeg_quality)
    return jpeg


@torch.no_grad()
def decode_to_jpeg_and_bgr(
    latent: torch.Tensor, vae: AutoencoderKL, jpeg_quality: int = 85
) -> tuple[np.ndarray, bytes]:
    """Scaled latent → (BGR uint8 HxWx3, JPEG bytes). One VAE decode.

    Used by the WM serve loop when a detector wants pixels for its CV pass —
    the JPEG is what the browser renders, the BGR is what cv2 ingests.
    """
    latent_for_decode = latent / LATENT_SCALE
    decoded = vae.decode(latent_for_decode.half()).sample  # type: ignore[arg-type, attr-defined]
    img = (decoded + 1.0) / 2.0
    img = img.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, jpeg_bytes = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return img_bgr, jpeg_bytes.tobytes()


@torch.no_grad()
def encode_image(
    img: np.ndarray, vae: AutoencoderKL, device: torch.device | str
) -> torch.Tensor:
    """RGB uint8 (H, W, 3) image → scaled latent (C, H/8, W/8).

    Resizes to the canonical (256, 128) input shape used during training.
    """
    if img.dtype != np.uint8:
        raise ValueError(f"encode_image expects uint8, got {img.dtype}")
    img = cv2.resize(img, (256, 128))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    latent = vae.encode(tensor.unsqueeze(0).half().to(device)).latent_dist.mode()  # type: ignore[union-attr]
    return latent.squeeze(0).float() * LATENT_SCALE


@torch.no_grad()
def decode_to_image(latent: torch.Tensor, vae: AutoencoderKL) -> np.ndarray:
    """Scaled latent → RGB uint8 (H, W, 3) image."""
    latent_for_decode = latent / LATENT_SCALE
    if latent_for_decode.dim() == 3:
        latent_for_decode = latent_for_decode.unsqueeze(0)
    decoded = vae.decode(latent_for_decode.half()).sample  # type: ignore[arg-type, attr-defined]
    img = (decoded + 1.0) / 2.0
    img = img.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)
