from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PPOPolicyV1Config:
    """Hyperparameters for ppo_policy_v1 — wraps a registered BC policy.

    The BC base is identified by ``base_policy_name`` (e.g. ``bc_transformer_v1``)
    and built via the same ``policy_registry`` as standalone BC training.
    ``base_policy_config`` is forwarded to the base policy's config dataclass.
    BC weights are loaded separately by the launcher (from a BC checkpoint
    when starting fresh PPO, or as part of the full PPO state_dict on resume).
    """

    base_policy_name: str = "bc_transformer_v1"
    base_policy_config: dict[str, Any] = field(default_factory=dict)
    action_std_init: float = 0.3
