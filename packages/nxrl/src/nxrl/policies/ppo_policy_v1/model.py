"""ppo_policy_v1: BC base policy + value head, for PPO.

Wraps any registered BC base policy with:
  - a value head branching off the 256-dim trunk
  - a learnable per-stick-axis ``log_std`` (Gaussian noise for exploration)
  - a non-trainable ``button_bias`` buffer (per-button logit offset, applied
    only at sampling/eval — not in ``forward()`` so the BC anchor sees the
    true underlying logits)

The base policy is built via ``policy_registry`` so this wrapper is
architecture-agnostic — any BC policy that exposes ``projection``,
``trunk``, ``stick_head``, ``button_head`` works.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Bernoulli, Independent, Normal

from nxrl.core.registry import policy_registry
from nxrl.policies.ppo_policy_v1.config import PPOPolicyV1Config


@policy_registry.register("ppo_policy_v1", config_cls=PPOPolicyV1Config)
class PPOPolicyV1(nn.Module):
    def __init__(self, config: PPOPolicyV1Config) -> None:
        super().__init__()
        self.config = config

        base_cls = policy_registry.get(config.base_policy_name)
        base_cfg_cls = policy_registry.get_config(config.base_policy_name)
        base_cfg = base_cfg_cls(**config.base_policy_config)
        base = base_cls(base_cfg)

        self.base_policy_name = config.base_policy_name
        self.base_config = base_cfg
        self.sequence_length = base.sequence_length

        # Steal layers from the base policy
        self.projection = base.projection
        self.trunk = base.trunk
        self.stick_head = base.stick_head
        self.button_head = base.button_head

        self.is_transformer = hasattr(base, "transformer")
        if self.is_transformer:
            self.transformer = base.transformer
            self.pos_emb = base.pos_emb
            self.register_buffer("causal_mask", base.causal_mask, persistent=False)
        else:
            self.lstm = base.lstm

        # New: value head and exploration parameters.
        import math

        self.value_head = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.full((4,), math.log(config.action_std_init)))
        self.register_buffer("button_bias", torch.zeros(22))

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _c, _h, _w = x.shape
        x = x.view(b, n, -1)
        x = self.projection(x)
        if self.is_transformer:
            x = x + self.pos_emb[:, :n, :]
            x = self.transformer(x, mask=self.causal_mask[:n, :n])
            h = x[:, -1, :]
        else:
            _, (hn, _) = self.lstm(x)
            h = hn[-1]
        return self.trunk(h)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(action_out, value)`` where ``action_out`` is ``(B, 26)``
        (sticks tanh'd, buttons raw logits — BC anchor sees this) and ``value``
        is ``(B,)``.
        """
        features = self._get_features(obs)
        sticks = torch.tanh(self.stick_head(features))
        button_logits = self.button_head(features)
        action_out = torch.cat([sticks, button_logits], dim=1)
        value = self.value_head(features).squeeze(-1)
        return action_out, value

    def get_action_and_value(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_out, value = self.forward(obs)
        stick_mean = action_out[:, :4]
        button_logits = action_out[:, 4:] + self.button_bias

        stick_std = self.log_std.exp().expand_as(stick_mean)
        stick_dist = Independent(Normal(stick_mean, stick_std), 1)
        stick_sample = stick_dist.rsample()
        stick_action = stick_sample.clamp(-1.0, 1.0)

        button_dist = Independent(Bernoulli(logits=button_logits), 1)
        button_action = button_dist.sample()

        log_prob = stick_dist.log_prob(stick_sample) + button_dist.log_prob(button_action)
        entropy = stick_dist.entropy() + button_dist.entropy()

        action = torch.cat([stick_action, button_action], dim=1)
        return action, log_prob, entropy, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_out, value = self.forward(obs)
        stick_mean = action_out[:, :4]
        button_logits = action_out[:, 4:] + self.button_bias

        stick_std = self.log_std.exp().expand_as(stick_mean)
        stick_dist = Independent(Normal(stick_mean, stick_std), 1)
        button_dist = Independent(Bernoulli(logits=button_logits), 1)

        log_prob = stick_dist.log_prob(actions[:, :4]) + button_dist.log_prob(actions[:, 4:])
        entropy = stick_dist.entropy() + button_dist.entropy()
        return log_prob, entropy, value

    def load_bc_state_dict(self, bc_state_dict: dict) -> None:
        """Copy weights from a BC checkpoint into the base layers.

        Expected keys are the BC architecture's own (``projection.*``,
        ``trunk.*``, ``stick_head.*``, ``button_head.*``, plus ``transformer.*``
        + ``pos_emb`` + ``causal_mask`` for transformer / ``lstm.*`` for LSTM).
        Value head, log_std, and button_bias are left at their freshly-initialized
        values.
        """
        own_state = self.state_dict()
        unexpected: list[str] = []
        for k, v in bc_state_dict.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k].copy_(v)
            else:
                unexpected.append(k)
        if unexpected:
            print(f"[ppo_policy_v1] {len(unexpected)} BC key(s) ignored (not found in PPO policy or shape mismatch)")
