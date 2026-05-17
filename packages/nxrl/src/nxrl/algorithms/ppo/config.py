from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PPOAlgorithmConfig:
    """Hyperparameters for PPO."""

    lr: float = 7.5e-5
    gamma: float = 0.995
    gae_lambda: float = 0.97
    clip_epsilon: float = 0.1
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    bc_reg_coef: float = 1.0
    bc_reg_right_stick_coef: float | None = None  # falls back to bc_reg_coef when None
    max_grad_norm: float = 0.5
    ppo_epochs: int = 2
    minibatch_size: int = 64
    rollouts_per_update: int = 10
    total_updates: int = 1000
    flow_steps: int = 5
    cfg_scale: float = 1.0
    button_logit_bias_indices: list[int] = field(default_factory=list)
    button_logit_bias_value: float = 0.0
    project_name: str = "nxrl-ppo"


@dataclass
class SeedSpec:
    """One seed scene: which episode .npz to load and at which frame the
    agent starts generating. ``start_frame`` may be negative — in that case
    the rollout zero-pads the missing prefix so the seed lands at the end of
    the context window — supports seeds near episode start.
    """

    npz_path: str
    start_frame: int = 0


@dataclass
class RolloutSpec:
    """Per-rollout knobs (length, goal offset). Reward + termination come from
    a separately-configured callable; we keep those out of this dataclass to
    avoid coupling PPO to a specific reward stack.
    """

    frames: int = 150
    goal_offset: int = 30
