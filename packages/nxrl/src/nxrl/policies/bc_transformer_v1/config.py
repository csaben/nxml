from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BCTransformerV1Config:
    """Hyperparameters for the bc_transformer_v1 architecture.

    Latent input is fixed at sd-vae-ft-mse @ 128x256 (4x16x32) — i.e. 2048
    flattened per frame. If the world model ever changes VAE/resolution,
    add a new architecture rather than retrofitting these numbers.
    """

    sequence_length: int = 300
    hidden_size: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.3
    latent_channels: int = 4
    latent_height: int = 16
    latent_width: int = 32
