from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class WorldModel(Protocol):
    """Architecture-agnostic interface for world models."""

    @property
    def latent_shape(self) -> tuple[int, int, int]: ...

    @property
    def context_length(self) -> int: ...

    @property
    def action_dims(self) -> int: ...

    def init_rollout_state(
        self,
        initial_latents: torch.Tensor,  # (T, C, H, W) or (B, T, C, H, W)
        initial_actions: torch.Tensor,  # (T, A) or (B, T, A)
        goal_latent: torch.Tensor,  # (C, H, W) or (B, C, H, W)
    ) -> Any: ...

    def step_rollout(
        self,
        state: Any,
        action: torch.Tensor,  # (A,) or (B, A)
        **inference_kwargs: Any,  # arch-specific (flow_steps, cfg_scale, etc.)
    ) -> tuple[Any, torch.Tensor]: ...  # (new_state, predicted_latent)

    def update_goal(self, state: Any, goal_latent: torch.Tensor) -> Any: ...
