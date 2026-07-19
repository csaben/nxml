"""Shared fixtures for the nxwm-mira codec tests.

The RAE encoder needs the DINOv3 backbone, loaded through ``torch.hub``: that requires
network access (or a populated hub cache) plus a few transitive deps. Tests that build the
full codec go through :func:`build_codec_or_skip`, which skips gracefully when the backbone
can't be constructed (offline / no hub cache / missing deps). Weights are never required
(``require_dino_weights=False`` → random frozen backbone), so the gated DINOv3 download is
not a test dependency.
"""

from __future__ import annotations

import pytest
import torch

from nxwm_mira.codec import (
    RAEEncoderConfig,
    StridedConvBottleneckConfig,
    VideoCodec,
    VideoCodecConfig,
    ViTDecoderConfig,
)
from nxwm_mira.data.batch import VideoBatch
from nxwm_mira.ml import ImageConfig

TINY_VIDEO = ImageConfig(height=64, width=64, channels=3, timesteps=4, fps=15)


def tiny_raev2_config(*, aspect_mode: str = "pad") -> VideoCodecConfig:
    """A tiny RAEv2-tdown config (64x64, 4 frames, shallow decoder) for fast CPU runs.

    Keeps the release architecture family (DINOv3 backbone, layer aggregation, td=2 strided
    bottleneck, ViT decoder) but with the vitb16 backbone and every shape shrunk so a full
    forward+backward is cheap enough for a unit test.
    """
    encoder = RAEEncoderConfig(
        latent_dim=8,
        rae_model="dinov3_vitb16",
        aggregation_layers=[2, 5, 8, 11],
        aspect_mode=aspect_mode,
        bottleneck=StridedConvBottleneckConfig(stride=2, temporal_stride=2, noise_tau=0.0),
        compile_dino=False,
        video=TINY_VIDEO,
    )
    decoder = ViTDecoderConfig(
        video=TINY_VIDEO,
        latent_dim=8,
        activation_checkpointing=False,
        bottleneck=StridedConvBottleneckConfig(stride=2),
        vit_width=64,
        vit_depth=2,
        vit_num_heads=4,
        mlp_dim_multiplier=2,
        qk_norm="layernorm",
        patch_size=16,
        patch_size_t=2,
    )
    return VideoCodecConfig(encoder=encoder, decoder=decoder)


def build_codec_or_skip(
    config: VideoCodecConfig | None = None, *, require_dino_weights: bool = False
) -> VideoCodec:
    """Build a codec, skipping the test if the DINOv3 backbone can't be loaded."""
    config = config or tiny_raev2_config()
    try:
        return VideoCodec(config, require_dino_weights=require_dino_weights).eval()
    except Exception as exc:  # noqa: BLE001 -- any backbone-load failure should skip, not fail
        pytest.skip(f"DINOv3 backbone unavailable, skipping: {type(exc).__name__}: {exc}")


def random_video(
    batch: int = 1, frames: int = 4, height: int = 64, width: int = 64
) -> torch.Tensor:
    """A ``(B, T, 3, H, W)`` uint8 video, as produced by the data loader."""
    return torch.randint(0, 256, (batch, frames, 3, height, width), dtype=torch.uint8)


def tiny_batch(batch: int = 1) -> VideoBatch:
    return VideoBatch(video=random_video(batch=batch))
