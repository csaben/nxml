"""Config for Future-Anchored Diffusion Transformer (fa_dit).

Port of the architecture from comma.ai's
"Learning to Drive from a World Model" (arXiv:2504.19077v1).

The paper's "pose" conditioning vector (6-DOF vehicle state) generalizes
to any per-frame conditioning vector — for nxwm games this is the 26-dim
action vector. The pose/action terminology used interchangeably below.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaDiTConfig:
    """Hyperparameters for the fa_dit world model.

    Three sizes from the paper (matching gpt-2 variants):
      - gpt:        embed_dim=768,  depth=12, num_heads=12  (~250M params)
      - gpt-medium: embed_dim=1024, depth=24, num_heads=16  (~500M params)
      - gpt-large:  embed_dim=1280, depth=36, num_heads=20  (~1B params)

    Defaults below correspond roughly to the small (gpt) configuration but
    scaled down for nxwm's 16×32 latent resolution vs the paper's larger
    driving frames.
    """

    # Transformer backbone
    embed_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: int = 4

    # Patching (spatial only — temporal patch is 1, i.e. one token-set per frame)
    patch_size: int = 2

    # Context structure
    past_seq_len: int = 10
    """Number of past frames in the context window (T in the paper)."""

    future_anchor_len: int = 5
    """Number of future-anchor frames prepended to the sequence (f_e − f_s)."""

    # Latent shape (VAE compressed image)
    latent_channels: int = 4
    latent_height: int = 16
    latent_width: int = 32

    # Per-frame conditioning vector (paper: 6-DOF pose; nxwm: action)
    pose_dims: int = 26

    # Plan head (trajectory output)
    plan_horizon: int = 50
    """Number of future steps the plan head predicts (paper uses 10s @ 5Hz = 50)."""

    plan_features: int = 26
    """Per-step features of the predicted trajectory. For nxwm, action_dims;
    for driving, this would be (x, y, z, vx, vy, vz, ax, ay, az, roll,
    pitch, yaw, roll_rate, pitch_rate, yaw_rate) = ~15."""

    plan_num_hypotheses: int = 5
    """K in Multi-Hypothesis Planning loss; paper uses 5."""

    plan_depth: int = 3
    """Number of residual feed-forward blocks in the plan head."""

    # Sampling
    flow_steps: int = 15
    """Default Euler discretization steps for rectified-flow sampling."""

    # ------------------------------------------------------------------
    # Convenience accessors / derived shapes
    # ------------------------------------------------------------------

    @property
    def tokens_per_frame(self) -> int:
        return (self.latent_height // self.patch_size) * (
            self.latent_width // self.patch_size
        )

    @property
    def total_seq_len(self) -> int:
        """Past + future-anchor + 1 (target) — the full transformer sequence
        in frame units. Multiply by tokens_per_frame for the token count."""
        return self.past_seq_len + self.future_anchor_len + 1
