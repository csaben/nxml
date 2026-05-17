"""Rollout state for fa_dit.

Differs from dit_v1's state in two ways:
  1) Holds both PAST context and FUTURE ANCHORING latents/poses (the paper's
     "non-causal" sequence with future frames prepended to the past).
  2) Holds a per-frame noise level (``tau``) buffer so the model can apply
     the paper's noise-level augmentation during training (some context
     frames noisy, future-anchor frames always clean). At inference the
     trainer/sampler sets all context taus to 0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class FaDiTRolloutState:
    """Immutable rollout state for the fa_dit world model.

    All tensors are unbatched (no leading batch dim) and live on the model's
    device. Shapes:

    Past context (most recent past_seq_len frames):
      - ``past_latents``: (past_seq_len, C, H, W)
      - ``past_poses``:   (past_seq_len, P) — pose (or action) for each past frame
      - ``past_taus``:    (past_seq_len,) — noise levels for the past context
                          frames (0 at inference; nonzero during noise-aug training)

    Future anchoring (future_anchor_len frames sampled from the future):
      - ``future_latents``: (future_anchor_len, C, H, W)
      - ``future_poses``:   (future_anchor_len, P)
      Future-anchor frames are ALWAYS clean (no noise applied) per the paper's
      noise-level augmentation prescription, so no ``future_taus`` field.

    Target slot (the frame being predicted):
      - ``target_pose``: (P,) — pose/action paired with the target frame
    """

    past_latents: torch.Tensor
    past_poses: torch.Tensor
    past_taus: torch.Tensor
    future_latents: torch.Tensor
    future_poses: torch.Tensor
    target_pose: torch.Tensor

    def with_future(
        self, future_latents: torch.Tensor, future_poses: torch.Tensor
    ) -> FaDiTRolloutState:
        """Return a copy with refreshed future-anchor context."""
        return replace(
            self, future_latents=future_latents, future_poses=future_poses
        )

    def with_target_pose(self, target_pose: torch.Tensor) -> FaDiTRolloutState:
        """Return a copy with updated target pose (commanded by the policy)."""
        return replace(self, target_pose=target_pose)
