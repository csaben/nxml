from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BCAlgorithmConfig:
    """Hyperparameters for supervised BC training."""

    lr: float = 1.0e-4
    weight_decay: float = 0.01
    button_loss_weight: float = 1.0
    action_weight: float = 1.0
    stick_deadzone: float = 0.1  # frames where |stick| > this are "active"
    grad_clip: float = 1.0
    epochs: int = 100
    project_name: str = "nxrl-bc"
