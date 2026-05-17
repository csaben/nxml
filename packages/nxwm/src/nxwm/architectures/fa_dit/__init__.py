"""Future-Anchored Diffusion Transformer (fa_dit).

Implementation of comma.ai's "Learning to Drive from a World Model"
(arXiv:2504.19077v1). See ``model.py`` for the architectural notes."""

from .config import FaDiTConfig
from .model import (
    FaDiTBlock,
    FaDiTWorldModel,
    PlanHead,
    mhp_laplace_nll,
    rectified_flow_loss,
)
from .rollout import FaDiTRolloutState

__all__ = [
    "FaDiTBlock",
    "FaDiTConfig",
    "FaDiTRolloutState",
    "FaDiTWorldModel",
    "PlanHead",
    "mhp_laplace_nll",
    "rectified_flow_loss",
]
