from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BCLstmV1Config:
    """Hyperparameters for the bc_lstm_v1 architecture."""

    sequence_length: int = 10
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    latent_channels: int = 4
    latent_height: int = 16
    latent_width: int = 32
