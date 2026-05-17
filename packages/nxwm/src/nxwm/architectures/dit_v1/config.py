from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiTV1Config:
    """Hyperparameters for the dit_v1 world model architecture."""

    embed_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    patch_size: int = 2
    seq_len: int = 10
    latent_channels: int = 4
    latent_height: int = 16
    latent_width: int = 32
    action_dims: int = 26
