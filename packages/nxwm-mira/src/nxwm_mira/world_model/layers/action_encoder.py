"""The action encoder: embeds Switch controller actions into per-latent-frame tokens.

Structural port of MIRA's keyboard+mouse encoder with the analog/discrete split remapped:
their mouse-delta MLP becomes a stick-axes MLP (4 continuous dims, already in [-1, 1] so no
symlog needed), their per-key embeddings become per-button embeddings (22 binary buttons),
and the mouse-sensitivity / per-player-subset machinery is dropped (single-player, one
controller type). Temporal pooling to the latent frame rate, whole-row dropout tokens, the
joint MLP and the prepended initial-action token are unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from nxwm_mira.data.actions import SwitchActions
from nxwm_mira.ml.init import init_weights


class ActionEncoder(torch.nn.Module):
    def __init__(
        self,
        action_dim: int,
        n_stick_axes: int,
        dim: int,
        temporal_downsampling: int,
        dropout_prob: float = 0.0,
        learned_temporal_pool: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_stick_axes = n_stick_axes
        self.n_buttons = action_dim - n_stick_axes
        self.temporal_downsampling = temporal_downsampling
        self.dropout_prob = dropout_prob

        stick_dim = dim // 2
        button_dim = dim - stick_dim
        self.stick_mlp = nn.Linear(n_stick_axes, stick_dim)

        # One learned embedding per button, sized to the closest power of 2 that fits.
        button_split_dim = 2 ** math.floor(math.log2(button_dim / self.n_buttons))
        button_remaining_dim = button_dim - self.n_buttons * button_split_dim
        if button_remaining_dim > 0:
            self.register_buffer(
                "button_zero_vector",
                torch.zeros((1, 1, button_remaining_dim)),
                persistent=False,
            )
        else:
            self.button_zero_vector = None

        self.button_embedding_dict = nn.ModuleDict()
        for k in range(self.n_buttons):
            self.button_embedding_dict[str(k)] = nn.Embedding(2, button_split_dim)

        self.button_mlp = nn.Linear(button_dim, button_dim)

        self.learned_temporal_pool = learned_temporal_pool
        if learned_temporal_pool:
            self.stick_temporal_pool = nn.Linear(temporal_downsampling * stick_dim, stick_dim)
            self.button_temporal_pool = nn.Linear(temporal_downsampling * button_dim, button_dim)

        self.joint_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        self.stick_dropout_token, self.button_dropout_token = None, None
        if dropout_prob > 0:
            self.stick_dropout_token = nn.Parameter(0.02 * torch.randn(1, 1, stick_dim))
            self.button_dropout_token = nn.Parameter(0.02 * torch.randn(1, 1, button_dim))

        # initial action token
        self.initial_action_token = nn.Parameter(0.02 * torch.randn(1, 1, dim))

        self.apply(init_weights)

    def forward(self, actions: SwitchActions | Tensor, drop_mask: Tensor | None = None) -> Tensor:
        """Encode ``(B, T, action_dim)`` actions to ``(B, T/td + 1, dim)`` conditioning tokens.

        ``drop_mask`` (inference-only, ``(B,)`` bool) replaces that row's whole action stream
        with the learned dropout tokens — the CFG "unconditional" branch.
        """
        raw = actions.actions if isinstance(actions, SwitchActions) else actions
        batch_size, n_actions, _ = raw.shape
        device = raw.device

        sticks = raw[..., : self.n_stick_axes].float()
        buttons = (raw[..., self.n_stick_axes :] > 0.5).long()

        stick_embed = self.stick_mlp(sticks)

        button_embed_list = []
        for k in range(self.n_buttons):
            button_embed_list.append(self.button_embedding_dict[str(k)](buttons[:, :, k]))
        if self.button_zero_vector is not None:
            button_embed_list.append(self.button_zero_vector.expand(batch_size, n_actions, -1))
        button_embed = torch.cat(button_embed_list, dim=-1)
        button_embed = self.button_mlp(button_embed)

        # Temporally downsample to the latent frame rate.
        stick_embed = stick_embed.unflatten(dim=1, sizes=(-1, self.temporal_downsampling))
        button_embed = button_embed.unflatten(dim=1, sizes=(-1, self.temporal_downsampling))
        if self.learned_temporal_pool:
            stick_embed = self.stick_temporal_pool(stick_embed.flatten(2))
            button_embed = self.button_temporal_pool(button_embed.flatten(2))
        else:
            stick_embed = stick_embed.mean(dim=2)
            button_embed = button_embed.mean(dim=2)

        if self.training and self.dropout_prob > 0:
            assert self.stick_dropout_token is not None and self.button_dropout_token is not None
            drop_sticks = (torch.rand((batch_size,), device=device) < self.dropout_prob).view(-1, 1, 1)
            stick_embed = torch.where(
                drop_sticks, self.stick_dropout_token.to(stick_embed.dtype), stick_embed
            )
            drop_buttons = (torch.rand((batch_size,), device=device) < self.dropout_prob).view(-1, 1, 1)
            button_embed = torch.where(
                drop_buttons, self.button_dropout_token.to(button_embed.dtype), button_embed
            )
        elif drop_mask is not None and self.stick_dropout_token is not None:
            assert self.button_dropout_token is not None
            mask = drop_mask.view(-1, 1, 1)
            stick_embed = torch.where(mask, self.stick_dropout_token.to(stick_embed.dtype), stick_embed)
            button_embed = torch.where(
                mask, self.button_dropout_token.to(button_embed.dtype), button_embed
            )

        actions_embed = torch.cat((stick_embed, button_embed), dim=-1)
        actions_embed = self.joint_mlp(actions_embed)

        # append initial action token
        initial_action_token = self.initial_action_token.expand(batch_size, -1, -1)
        actions_embed = torch.cat([initial_action_token, actions_embed], dim=1)
        return actions_embed
