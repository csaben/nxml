"""nxwm_mira.world_model: the action-conditioned latent diffusion world model.

Port of MIRA's single-player world model with Switch-controller action conditioning
(the multiplayer wrapper is not ported). Public API:

    SwitchActionConfig / SwitchActions — the 26-dim action stream (re-exported from data.actions)
    LatentWorldModelConfig — architecture/training config of the world model
    WorldModelInferenceConfig — sampling knobs for the autoregressive rollout
    LatentWorldModel — the frozen-codec + diffusion-transformer world model
    InferenceOutputs — the outputs of an autoregressive rollout
    DiffusionTransformer — the action-conditioned flow-matching transformer over codec latents
    ActionEncoder — embeds Switch controller actions into per-latent-frame conditioning tokens
    build_inference_schedule — the tau integration grid for the denoiser
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nxwm_mira.data.actions import SwitchActionConfig, SwitchActions

from .config import LatentWorldModelConfig, WorldModelInferenceConfig
from .diffusion_transformer import DiffusionTransformer
from .layers.action_encoder import ActionEncoder
from .schedule import build_inference_schedule

if TYPE_CHECKING:
    from .latent_world_model import InferenceOutputs, LatentWorldModel

__all__ = [
    "ActionEncoder",
    "DiffusionTransformer",
    "InferenceOutputs",
    "LatentWorldModel",
    "LatentWorldModelConfig",
    "SwitchActionConfig",
    "SwitchActions",
    "WorldModelInferenceConfig",
    "build_inference_schedule",
]

# LatentWorldModel pulls in the codec (heavier import chain); resolve lazily (PEP 562).
_LAZY = {"LatentWorldModel", "InferenceOutputs"}


def __getattr__(name: str):
    if name in _LAZY:
        from . import latent_world_model

        return getattr(latent_world_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
