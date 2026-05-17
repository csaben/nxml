"""bc_lstm_v1: LSTM BC policy.

LSTM over VAE-latent frames; final hidden state feeds the same trunk-head
shape as bc_transformer_v1 so callers can swap the two without touching
output handling.
"""

from __future__ import annotations

import torch
from torch import nn

from nxrl.core.registry import policy_registry
from nxrl.policies.bc_lstm_v1.config import BCLstmV1Config


@policy_registry.register("bc_lstm_v1", config_cls=BCLstmV1Config)
class BCLstmV1(nn.Module):
    def __init__(self, config: BCLstmV1Config) -> None:
        super().__init__()
        self.config = config
        self.sequence_length = config.sequence_length
        self.latent_dim = config.latent_channels * config.latent_height * config.latent_width

        self.projection = nn.Sequential(
            nn.Linear(self.latent_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout,
        )
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
        _, (hn, _) = self.lstm(x)
        h = hn[-1]
        features = self.trunk(h)
        sticks = torch.tanh(self.stick_head(features))
        button_logits = self.button_head(features)
        return torch.cat([sticks, button_logits], dim=1)
