"""Future-Anchored Diffusion Transformer (fa_dit).

Implementation of the architecture from comma.ai's
"Learning to Drive from a World Model" (arXiv:2504.19077v1, April 2025).

Key differences from ``dit_v1``:

  1. **Future anchoring**: a block of future frames (sampled at horizon F =
     (f_s, f_e), f_s > T) is *prepended* to the sequence as additional
     non-causal context. The block-wise causal mask is unchanged — past
     context can attend to future-anchor frames because they appear earlier
     in the sequence. This gives the model "recovery pressure": rollouts
     converge to the goal state at F regardless of where they start.

  2. **Per-frame AdaLN conditioning**: the conditioning vector (sum of pose
     embed + diffusion-noise embed + world-timestep embed) varies along
     the time dimension. The paper modifies AdaLN to use per-frame
     (shift, scale, gate) instead of one global vector.

  3. **Per-frame noise levels (τ)**: in the paper's noise-augmentation
     training scheme some past frames carry noise τ > 0 (Logit-Normal(0,
     0.25)) and the future-anchor frames are always clean (τ = 0). The
     diffusion loss is only computed on the target frame T. This makes the
     model robust to autoregressive drift at inference.

  4. **Plan head**: a stack of residual feed-forward blocks reading the
     pooled target-frame representation. Outputs K hypothesis trajectories
     for Multi-Hypothesis Planning (MHP) loss. Each hypothesis emits a
     mean and log-scale per timestep × feature (Laplace prior, hetero-
     scedastic NLL).

  5. **Block-wise frame-causal attention mask**: bidirectional within a
     frame, causal across the sequence dimension. Enables kv-caching at
     inference. (Same masking strategy as dit_v1, but the sequence layout
     is [future_anchor | past_context | target] instead of just
     [past_context | target].)

The forward signature returns BOTH the predicted velocity for the target
frame AND the trajectory hypotheses, so a single forward pass computes
both terms of the paper's joint loss ``L = L_RF + α · L_T``.

Trainer responsibilities (out of scope for this module):
  - Sample noise levels τ ~ Logit-Normal per the paper's scheme
  - Apply rectified-flow noising: ``o_τ = τ ε + (1 − τ) o``
  - Compute the joint loss and back-prop
  - Sequential Euler sampling at inference (15 steps; see ``flow_steps``)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nxwm.core.registry import architecture_registry

from .config import FaDiTConfig
from .rollout import FaDiTRolloutState


# ============================================================================
# Per-frame AdaLN block
# ============================================================================


class FaDiTBlock(nn.Module):
    """DiT block with per-frame (time-varying) AdaLN conditioning.

    Args at forward time:
      - ``x``: (B, S, D) flattened token sequence; S = total_frames × tokens_per_frame
      - ``c``: (B, total_frames, D) per-frame conditioning vectors
      - ``tokens_per_frame``: int, used to broadcast c across spatial tokens
      - ``mask``: (S, S) block-causal attention mask
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
        )
        # Per-frame conditioning → 6 vectors (shift_msa, scale_msa, gate_msa,
        # shift_mlp, scale_mlp, gate_mlp). 6×hidden_size out from one Linear.
        # Zero-init the AdaLN output (standard DiT trick — makes the block
        # behave as an identity transformer at init, letting the model learn
        # to use AdaLN gradually instead of fighting random modulation).
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        tokens_per_frame: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # c: (B, F, D) → modulation params (B, F, 6D) → (B, F, 6, D) split into 6 of (B, F, D)
        mod = self.adaLN_modulation(c)  # (B, F, 6D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Broadcast per-frame conditioning across spatial tokens within each frame:
        # (B, F, D) → (B, F * tokens_per_frame, D) by repeat_interleave on frame axis.
        def _expand(p: torch.Tensor) -> torch.Tensor:
            return p.repeat_interleave(tokens_per_frame, dim=1)

        shift_msa = _expand(shift_msa)
        scale_msa = _expand(scale_msa)
        gate_msa = _expand(gate_msa)
        shift_mlp = _expand(shift_mlp)
        scale_mlp = _expand(scale_mlp)
        gate_mlp = _expand(gate_mlp)

        # Self-attention with frame-causal mask + per-token modulation
        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + gate_msa * attn_out

        # MLP with per-token modulation
        h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(h)
        return x


# ============================================================================
# Plan head — residual FF blocks → K hypothesis trajectories
# ============================================================================


class _ResidualFFBlock(nn.Module):
    """One residual FF block used in the plan head."""

    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class PlanHead(nn.Module):
    """Predicts K trajectory hypotheses given a pooled context vector.

    Each hypothesis is a sequence of (mean, log_scale) pairs for a
    heteroscedastic Laplace likelihood — output shape is
    ``(B, K, horizon, 2 * features)`` where the first ``features`` entries
    along the last axis are the means and the next ``features`` are the
    log-scales (log_b in Laplace(μ, b)).
    """

    def __init__(
        self,
        embed_dim: int,
        horizon: int,
        features: int,
        num_hypotheses: int,
        depth: int,
    ):
        super().__init__()
        self.horizon = horizon
        self.features = features
        self.num_hypotheses = num_hypotheses

        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [_ResidualFFBlock(embed_dim) for _ in range(depth)]
        )
        # Per-hypothesis projection to (horizon × 2*features). Stack the
        # output dim so a single Linear emits all K hypotheses at once.
        self.hyp_proj = nn.Linear(embed_dim, num_hypotheses * horizon * 2 * features)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """ctx: (B, D) pooled context → (B, K, H, 2F)."""
        h = self.input_proj(ctx)
        for blk in self.blocks:
            h = blk(h)
        out = self.hyp_proj(h)
        return out.view(
            -1, self.num_hypotheses, self.horizon, 2 * self.features
        )


def mhp_laplace_nll(
    plan_pred: torch.Tensor,
    target: torch.Tensor,
    log_scale_clamp: tuple[float, float] = (-7.0, 7.0),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multi-Hypothesis Planning loss with Laplace heteroscedastic NLL.

    Paper §4.2.2:
      - K=5 hypotheses, each trained with heteroscedastic NLL (Laplace prior).
      - Winner-takes-all: only the closest hypothesis (in L1 distance) gets
        the NLL gradient at each sample. The classifier choosing the winner
        is implicit in argmin.

    Args:
      plan_pred: (B, K, H, 2F) — first F are means μ, next F are log-scales log b.
      target:    (B, H, F)     — ground-truth trajectory.

    Returns: scalar loss (mean over batch).
    """
    B, K, H, twoF = plan_pred.shape
    F_dim = twoF // 2
    mu = plan_pred[..., :F_dim]              # (B, K, H, F)
    log_b = plan_pred[..., F_dim:]
    log_b = log_b.clamp(*log_scale_clamp)
    b = log_b.exp() + eps                    # (B, K, H, F)

    tgt = target.unsqueeze(1)                # (B, 1, H, F) → broadcasts over K
    abs_err = (mu - tgt).abs()

    # Per-hypothesis L1 distance → winner selection
    l1_per_hyp = abs_err.sum(dim=(2, 3))     # (B, K)
    winner = l1_per_hyp.argmin(dim=1)        # (B,)

    # Heteroscedastic Laplace NLL per element:
    #   -log p(x | μ, b) = |x − μ| / b + log(2b)
    nll = abs_err / b + math.log(2.0) + log_b  # (B, K, H, F)

    # Gather only the winning hypothesis's NLL per sample
    winner_idx = winner.view(B, 1, 1, 1).expand(-1, 1, H, F_dim)
    winner_nll = nll.gather(1, winner_idx).squeeze(1)  # (B, H, F)
    return winner_nll.mean()


# ============================================================================
# FaDiT World Model
# ============================================================================


@architecture_registry.register("fa_dit", config_cls=FaDiTConfig)
class FaDiTWorldModel(nn.Module):
    """Future-Anchored Diffusion Transformer.

    Sequence layout (frames, ordered along the transformer's sequence dim):

        [ future_anchor (F_a) | past_context (T) | target (1) ]

    Each frame contributes ``tokens_per_frame`` spatial tokens. The block-
    causal mask allows token *i* to attend to all earlier frames in this
    layout, so past-context queries naturally see future-anchor context.
    The target frame is the only one being predicted (rectified-flow
    velocity); all other frames provide conditioning.
    """

    def __init__(self, config: FaDiTConfig):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # Embeddings
        # ------------------------------------------------------------------
        self.patch_embed = nn.Conv2d(
            config.latent_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.pose_encoder = nn.Linear(config.pose_dims, config.embed_dim)
        self.noise_embed = nn.Sequential(
            nn.Linear(1, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )
        # World-timestep embedding (relative position of each frame in the
        # sequence). Future-anchor frames carry a different range than past
        # context; we encode them as integer positions 0..total_seq_len-1.
        self.world_step_embed = nn.Embedding(config.total_seq_len, config.embed_dim)

        # ------------------------------------------------------------------
        # Positional encodings (within-frame spatial + per-frame temporal)
        # ------------------------------------------------------------------
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.tokens_per_frame, config.embed_dim)
        )
        self.temp_embed = nn.Parameter(
            torch.zeros(1, config.total_seq_len, config.embed_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.temp_embed, std=0.02)

        # ------------------------------------------------------------------
        # Transformer trunk
        # ------------------------------------------------------------------
        self.blocks = nn.ModuleList(
            [
                FaDiTBlock(
                    hidden_size=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                )
                for _ in range(config.depth)
            ]
        )

        # ------------------------------------------------------------------
        # Diffusion (velocity) head — applied only to target-frame tokens
        # ------------------------------------------------------------------
        self.final_norm = nn.LayerNorm(config.embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.embed_dim, 2 * config.embed_dim),
        )
        self.final_proj = nn.Linear(
            config.embed_dim,
            config.patch_size * config.patch_size * config.latent_channels,
        )
        # Zero-init the final-block modulation + projection so the model's
        # initial velocity prediction is exactly zero — strong starting
        # point for rectified-flow training (predicting 0 ≈ MSE of ε - x_0).
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

        # ------------------------------------------------------------------
        # Plan head
        # ------------------------------------------------------------------
        self.plan_head = PlanHead(
            embed_dim=config.embed_dim,
            horizon=config.plan_horizon,
            features=config.plan_features,
            num_hypotheses=config.plan_num_hypotheses,
            depth=config.plan_depth,
        )

        # Cached attention mask
        self.register_buffer(
            "_causal_mask",
            self._build_block_causal_mask(
                config.total_seq_len, config.tokens_per_frame
            ),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Convenience accessors (WorldModel protocol parity)
    # ------------------------------------------------------------------

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (
            self.config.latent_channels,
            self.config.latent_height,
            self.config.latent_width,
        )

    @property
    def seq_len(self) -> int:
        """Past-context length (matches dit_v1's ``seq_len`` semantics — the
        number of past frames the model conditions on, NOT the full
        transformer sequence). The full sequence length is
        ``total_seq_len`` on the config."""
        return self.config.past_seq_len

    @property
    def context_length(self) -> int:
        return self.config.past_seq_len

    @property
    def action_dims(self) -> int:
        return self.config.pose_dims

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_block_causal_mask(
        num_frames: int, tokens_per_frame: int
    ) -> torch.Tensor:
        """Frame-causal (bidirectional within frame, causal across frames)
        mask of shape (num_frames × tokens_per_frame,) squared, with
        ``-inf`` at masked positions and 0 elsewhere."""
        frame_mask = torch.triu(
            torch.full((num_frames, num_frames), float("-inf")), diagonal=1
        )
        # Expand each frame-frame entry to a tokens_per_frame × tokens_per_frame block
        mask = frame_mask.repeat_interleave(tokens_per_frame, dim=0).repeat_interleave(
            tokens_per_frame, dim=1
        )
        return mask

    # ------------------------------------------------------------------
    # Forward (training + inference shared)
    # ------------------------------------------------------------------

    def forward(
        self,
        target_xt: torch.Tensor,        # (B, C, H, W) — noised target latent
        past_latents: torch.Tensor,     # (B, T_past, C, H, W)
        past_poses: torch.Tensor,       # (B, T_past, P)
        future_latents: torch.Tensor,   # (B, F_a, C, H, W)
        future_poses: torch.Tensor,     # (B, F_a, P)
        target_pose: torch.Tensor,      # (B, P)
        taus: torch.Tensor,             # (B, total_seq_len) per-frame noise τ
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full transformer once.

        Returns:
          v_pred: (B, C, H, W) predicted rectified-flow velocity for the target.
          plan:   (B, K, H_plan, 2*F_plan) trajectory hypotheses for MHP loss.
        """
        B = target_xt.shape[0]
        C, H, W = self.latent_shape
        L = self.config.tokens_per_frame
        F_a = self.config.future_anchor_len
        T = self.config.past_seq_len

        # ---- Patchify all frames into token sequences ----
        future_tokens = self._patchify(future_latents)          # (B, F_a, L, D)
        past_tokens = self._patchify(past_latents)              # (B, T, L, D)
        target_tokens = self._patchify(target_xt.unsqueeze(1))  # (B, 1, L, D)

        # Concat along frame axis: [future_anchor | past | target]
        x_frames = torch.cat([future_tokens, past_tokens, target_tokens], dim=1)
        # (B, total_seq_len, L, D) — add spatial + temporal positional encodings
        x_frames = (
            x_frames
            + self.pos_embed.unsqueeze(1)         # (1, 1, L, D) broadcast over frames
            + self.temp_embed.unsqueeze(2)        # (1, total, 1, D) broadcast over tokens
        )
        # Flatten to (B, total * L, D)
        x = x_frames.view(B, -1, self.config.embed_dim)

        # ---- Per-frame conditioning c_t = pose_embed + noise_embed + world_step_embed ----
        all_poses = torch.cat(
            [future_poses, past_poses, target_pose.unsqueeze(1)], dim=1
        )                                                       # (B, total, P)
        pose_c = self.pose_encoder(all_poses)                   # (B, total, D)
        noise_c = self.noise_embed(taus.unsqueeze(-1))          # (B, total, D)
        world_steps = torch.arange(self.config.total_seq_len, device=x.device)
        world_c = self.world_step_embed(world_steps).unsqueeze(0).expand(B, -1, -1)
        c = pose_c + noise_c + world_c                          # (B, total, D)

        # ---- Transformer trunk ----
        mask = self._causal_mask
        for block in self.blocks:
            x = block(x, c, tokens_per_frame=L, mask=mask)

        # ---- Diffusion velocity head — only target-frame tokens ----
        target_out = x[:, -L:]                                  # (B, L, D)
        target_c = c[:, -1]                                     # (B, D)
        shift, scale = self.final_modulation(target_c).chunk(2, dim=-1)
        target_out = self.final_norm(target_out) * (
            1 + scale.unsqueeze(1)
        ) + shift.unsqueeze(1)
        v_pred = self.final_proj(target_out)                    # (B, L, P²·C)
        # Proper unpatchify: token (i * Wp + j) holds the (P, P, C) values for
        # patch (i, j) in the Hp × Wp patch grid. Reassemble by interleaving
        # the patch interior with the patch grid.
        P = self.config.patch_size
        C_lat = self.config.latent_channels
        Hp = self.config.latent_height // P
        Wp = self.config.latent_width // P
        v_pred = v_pred.reshape(B, Hp, Wp, P, P, C_lat)          # (B, Hp, Wp, P, P, C)
        v_pred = v_pred.permute(0, 5, 1, 3, 2, 4).contiguous()   # (B, C, Hp, P, Wp, P)
        v_pred = v_pred.reshape(B, C_lat, Hp * P, Wp * P)        # (B, C, H, W)

        # ---- Plan head — pooled over target-frame tokens ----
        plan_ctx = target_out.mean(dim=1)                       # (B, D)
        plan = self.plan_head(plan_ctx)                         # (B, K, H_plan, 2F_plan)

        return v_pred, plan

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """(B, F, C, H, W) → (B, F, L, D) tokens via Conv2d patch embed."""
        B, F_, C, H, W = latents.shape
        patches = self.patch_embed(latents.view(B * F_, C, H, W))
        # (B*F, D, H/p, W/p) → (B*F, L, D) → (B, F, L, D)
        return patches.flatten(2).transpose(1, 2).view(B, F_, -1, self.config.embed_dim)

    # ------------------------------------------------------------------
    # Rollout API (parallels dit_v1.WorldModel; FA-specific state type)
    # ------------------------------------------------------------------

    def init_rollout_state(
        self,
        past_latents: torch.Tensor,         # (T, C, H, W)
        past_poses: torch.Tensor,           # (T, P)
        future_latents: torch.Tensor,       # (F_a, C, H, W)
        future_poses: torch.Tensor,         # (F_a, P)
        target_pose: torch.Tensor,          # (P,)
        past_taus: torch.Tensor | None = None,
    ) -> FaDiTRolloutState:
        """Build a rollout state from a real (T, …) past seed and a future
        anchor sampled from horizon F. ``past_taus`` defaults to zeros
        (clean context — the standard inference setting); set non-zero
        during noise-augmentation training per the paper's prescription.
        """
        T = self.config.past_seq_len
        F_a = self.config.future_anchor_len
        if past_latents.shape[0] != T:
            raise ValueError(
                f"past_latents must have {T} frames, got {past_latents.shape[0]}"
            )
        if future_latents.shape[0] != F_a:
            raise ValueError(
                f"future_latents must have {F_a} frames, got {future_latents.shape[0]}"
            )
        if past_taus is None:
            past_taus = torch.zeros(T, device=past_latents.device)
        return FaDiTRolloutState(
            past_latents=past_latents,
            past_poses=past_poses,
            past_taus=past_taus,
            future_latents=future_latents,
            future_poses=future_poses,
            target_pose=target_pose,
        )

    @torch.no_grad()
    def step_rollout(
        self,
        state: FaDiTRolloutState,
        target_pose: torch.Tensor,        # (P,) pose commanded by policy/plan
        *,
        flow_steps: int | None = None,
    ) -> tuple[FaDiTRolloutState, torch.Tensor, torch.Tensor]:
        """Sample the next latent + plan via Euler-step rectified flow.

        Mirrors the paper's §4.3 sequential-sampling loop:
          - All context latents are clean (τ=0).
          - Target slot starts as pure noise xt ~ N(0, I) at τ=1.
          - For 15 Euler steps with Δτ = 1/flow_steps:
              xt ← xt + Δτ · v(xt, …, τ + Δτ)
            (paper writes ``Δτ · w(x̃_τ, p, τ + Δτ)``; same form.)
          - Returns the final clean target latent and shifts the past window.

        Returns:
          new_state: state with past shifted by one timestep (oldest dropped,
                     just-predicted latent appended) and same future anchor.
          predicted: (C, H, W) the just-sampled target latent.
          plan:      (K, H_plan, 2*F_plan) trajectory hypotheses at this step.
        """
        n_steps = flow_steps if flow_steps is not None else self.config.flow_steps
        dt = 1.0 / n_steps
        device = state.past_latents.device

        # Batchify (B=1) and ensure context taus are 0.
        past_latents = state.past_latents.unsqueeze(0)
        past_poses = state.past_poses.unsqueeze(0)
        future_latents = state.future_latents.unsqueeze(0)
        future_poses = state.future_poses.unsqueeze(0)
        target_pose_b = target_pose.unsqueeze(0)
        # All context taus = 0 at inference; only the target is noised.
        taus = torch.zeros(1, self.config.total_seq_len, device=device)

        # Initial noise
        xt = torch.randn(1, *self.latent_shape, device=device)
        last_plan = None
        for i in range(n_steps):
            # Training: x_t = τ·eps + (1-τ)·x_clean, so τ=1 is noise, τ=0 is clean.
            # Inference starts at xt~N(0,I) (τ=1) and integrates DOWN to τ=0.
            tau = 1.0 - i / n_steps
            taus[:, -1] = tau
            v, last_plan = self.forward(
                target_xt=xt,
                past_latents=past_latents,
                past_poses=past_poses,
                future_latents=future_latents,
                future_poses=future_poses,
                target_pose=target_pose_b,
                taus=taus,
            )
            xt = xt + dt * v

        predicted = xt.squeeze(0)
        # Slide the past window by one.
        new_past_latents = torch.cat(
            [state.past_latents[1:], predicted.unsqueeze(0)], dim=0
        )
        new_past_poses = torch.cat(
            [state.past_poses[1:], target_pose.unsqueeze(0)], dim=0
        )
        new_state = FaDiTRolloutState(
            past_latents=new_past_latents,
            past_poses=new_past_poses,
            past_taus=state.past_taus,  # stays zeros at inference
            future_latents=state.future_latents,
            future_poses=state.future_poses,
            target_pose=target_pose,
        )
        return new_state, predicted, last_plan.squeeze(0)

    def update_future(
        self,
        state: FaDiTRolloutState,
        future_latents: torch.Tensor,
        future_poses: torch.Tensor,
    ) -> FaDiTRolloutState:
        """Replace the future-anchoring window (e.g. for a receding goal)."""
        return state.with_future(future_latents, future_poses)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, vae) -> torch.Tensor:
        """Pass-through to VAE helpers — same convention as dit_v1."""
        from nxwm.inference.vae import LATENT_SCALE

        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
        return vae.decode((latent / LATENT_SCALE).half()).sample


# ============================================================================
# Loss helpers — exported for the trainer
# ============================================================================


def rectified_flow_loss(
    v_pred: torch.Tensor, x_clean: torch.Tensor, eps: torch.Tensor
) -> torch.Tensor:
    """Paper §4.2.2 eq. 5: ``L_RF = ‖w(o_τ, p, τ) − (o − ε)‖²``.

    Here ``v_pred`` is the model's velocity prediction and the target is
    ``x_clean − eps`` (the rectified-flow ground-truth velocity for the
    sampled noise ε and clean latent o).
    """
    return F.mse_loss(v_pred, x_clean - eps)
