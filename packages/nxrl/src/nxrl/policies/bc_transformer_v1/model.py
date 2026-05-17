"""bc_transformer_v1: causal transformer BC policy.

Causal-masked transformer over a sequence of VAE latents. Last-token
features feed two heads: ``stick_head`` (4 floats, tanh) and
``button_head`` (22 raw logits — apply sigmoid + threshold at inference).
The 26-dim action vector layout matches ``nx_packets``.
"""

from __future__ import annotations

import torch
from torch import nn

from nxrl.core.registry import policy_registry
from nxrl.policies.bc_transformer_v1.config import BCTransformerV1Config


@policy_registry.register("bc_transformer_v1", config_cls=BCTransformerV1Config)
class BCTransformerV1(nn.Module):
    def __init__(self, config: BCTransformerV1Config) -> None:
        super().__init__()
        self.config = config
        self.sequence_length = config.sequence_length
        self.latent_dim = config.latent_channels * config.latent_height * config.latent_width

        self.projection = nn.Sequential(
            nn.Linear(self.latent_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, config.sequence_length, config.hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(config.sequence_length)
        self.register_buffer("causal_mask", causal_mask)

        self.trunk = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.stick_head = nn.Linear(256, 4)
        self.button_head = nn.Linear(256, 22)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _c, _h, _w = x.shape
        x = x.view(b, n, -1)
        x = self.projection(x)
        x = x + self.pos_emb[:, :n, :]
        x = self.transformer(x, mask=self.causal_mask[:n, :n])
        h = x[:, -1, :]
        features = self.trunk(h)
        sticks = torch.tanh(self.stick_head(features))
        button_logits = self.button_head(features)
        return torch.cat([sticks, button_logits], dim=1)
