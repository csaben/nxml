import torch
import torch.nn as nn

from nxwm.core.registry import architecture_registry

from .config import DiTV1Config
from .rollout import DiTRolloutState


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        # Goal cross-attention for spatial goal conditioning
        self.goal_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.goal_cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, x, c, goal_tokens, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)

        # Self-Attention with Block-Causal Mask
        res = self.norm1(x)
        res = res * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(res, res, res, attn_mask=mask, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # Goal Cross-Attention (no mask - attend to all goal patches)
        res = self.goal_norm(x)
        cross_out, _ = self.goal_cross_attn(res, goal_tokens, goal_tokens)
        x = x + cross_out

        # MLP
        res = self.norm2(x)
        res = res * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(res)
        return x


@architecture_registry.register("dit_v1", config_cls=DiTV1Config)
class DiTWorldModel(nn.Module):
    def __init__(self, config: DiTV1Config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim
        self.seq_len = config.seq_len  # History length
        self._latent_channels = config.latent_channels
        self._latent_height = config.latent_height
        self._latent_width = config.latent_width
        self._action_dims = config.action_dims
        self.tokens_per_frame = (config.latent_height // config.patch_size) * (
            config.latent_width // config.patch_size
        )

        # 1. Embeddings
        self.patch_embed = nn.Conv2d(
            config.latent_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.action_encoder = nn.Linear(config.action_dims, config.embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        # 2. Positional Encodings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.tokens_per_frame, config.embed_dim))
        self.temp_embed = nn.Parameter(torch.zeros(1, config.seq_len + 1, config.embed_dim))

        # 3. Transformer Blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(config.embed_dim, config.num_heads) for _ in range(config.depth)]
        )

        # 4. Final Projection to Latent Velocity
        self.final_layer = nn.Linear(
            config.embed_dim, config.patch_size * config.patch_size * config.latent_channels
        )

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self._latent_channels, self._latent_height, self._latent_width)

    @property
    def context_length(self) -> int:
        return self.seq_len

    @property
    def action_dims(self) -> int:
        return self._action_dims

    def get_block_causal_mask(self, num_frames, patches_per_frame, device):
        """Block-causal mask: bidirectional within frames, causal across time.

        Args:
            num_frames: Number of frames (T+1 for history + target)
            patches_per_frame: Number of patches per frame (L)
            device: torch device

        Returns:
            Mask of shape (num_frames * patches_per_frame, num_frames * patches_per_frame)
        """
        # Create frame-level causal mask
        frame_mask = torch.triu(
            torch.full((num_frames, num_frames), float("-inf"), device=device),
            diagonal=1,
        )
        # Expand to patch level: each frame-frame relationship applies to all L×L pairs
        mask = frame_mask.repeat_interleave(patches_per_frame, dim=0)
        mask = mask.repeat_interleave(patches_per_frame, dim=1)
        return mask

    def forward(self, xt, t, cond_obs, cond_actions, cond_goal):
        B, T, C, H, W = cond_obs.shape
        L = self.tokens_per_frame

        # Patchify History (B, T, L, D) and Target (B, 1, L, D)
        history = self.patch_embed(cond_obs.view(-1, C, H, W)).flatten(2).transpose(1, 2)
        history = history.view(B, T, L, self.embed_dim)

        target = self.patch_embed(xt).flatten(2).transpose(1, 2).view(B, 1, L, self.embed_dim)

        # Construct Sequence: (B, T+1, L, D)
        x = torch.cat([history, target], dim=1)

        # Apply Positional Encodings
        x = x + self.pos_embed.unsqueeze(1) + self.temp_embed.unsqueeze(2)
        x = x.view(B, -1, self.embed_dim)  # Flatten to (B, (T+1)*L, D)

        # Action Sequence Injection: inject all actions into the sequence
        action_tokens = self.action_encoder(cond_actions)  # (B, T, D)
        # Repeat last action for target frame (it caused this transition)
        last_action = action_tokens[:, -1:, :]  # (B, 1, D)
        action_tokens = torch.cat([action_tokens, last_action], dim=1)  # (B, T+1, D)
        # Expand to patch level and add to sequence
        action_sequence = action_tokens.repeat_interleave(L, dim=1)  # (B, (T+1)*L, D)
        x = x + action_sequence

        # Goal tokens for cross-attention (with spatial positional encoding)
        goal_tokens = self.patch_embed(cond_goal).flatten(2).transpose(1, 2)  # (B, L, D)
        goal_tokens = goal_tokens + self.pos_embed.squeeze(1)  # Add spatial pos embed

        # Conditioning Vector c = Time + Action (last) + Goal (global)
        # Keep goal in adaLN for global context alongside cross-attention
        goal_c = self.patch_embed(cond_goal).flatten(2).mean(dim=-1)  # (B, D)
        c = self.time_embed(t.unsqueeze(-1)) + self.action_encoder(cond_actions[:, -1]) + goal_c

        # Block-Causal Masking (bidirectional within frame, causal across time)
        mask = self.get_block_causal_mask(T + 1, L, x.device)

        # Transformer Blocks with goal cross-attention
        for block in self.blocks:
            x = block(x, c, goal_tokens, mask=mask)

        # We only predict the velocity for the last frame (the target)
        target_out = x[:, -L:]
        v_pred = self.final_layer(target_out)  # (B, L, P*P*C)

        # Reshape to (B, C_lat, H_lat, W_lat)
        v_pred = v_pred.transpose(1, 2).reshape(
            B, self._latent_channels, self._latent_height, self._latent_width
        )
        return v_pred

    # ------------------------------------------------------------------
    # Rollout API (WorldModel protocol)
    # ------------------------------------------------------------------

    def init_rollout_state(
        self,
        initial_latents: torch.Tensor,
        initial_actions: torch.Tensor,
        goal_latent: torch.Tensor,
    ) -> DiTRolloutState:
        """Build a rollout state from a real (T,...) seed.

        Tensors are stored unbatched. Caller is responsible for moving them to
        the model's device beforehand.
        """
        if initial_latents.shape[0] != self.seq_len:
            raise ValueError(
                f"initial_latents must have {self.seq_len} frames, got {initial_latents.shape[0]}"
            )
        if initial_actions.shape[0] != self.seq_len:
            raise ValueError(
                f"initial_actions must have {self.seq_len} frames, got {initial_actions.shape[0]}"
            )
        return DiTRolloutState(
            latent_history=initial_latents,
            action_history=initial_actions,
            goal_latent=goal_latent,
        )

    @torch.no_grad()
    def step_rollout(
        self,
        state: DiTRolloutState,
        action: torch.Tensor,
        *,
        sampler,
        flow_steps: int = 5,
        cfg_scale: float = 1.0,
    ) -> tuple[DiTRolloutState, torch.Tensor]:
        """Predict the next latent and roll the history forward.

        Action alignment: snapshot the *current* obs/actions BEFORE appending the
        incoming action. The incoming action belongs to the transition we are
        about to predict; we append it after the prediction so it becomes paired
        with the new latent for the next step.
        """
        obs = state.latent_history.unsqueeze(0)
        actions = state.action_history.unsqueeze(0)
        goal = state.goal_latent.unsqueeze(0)

        predicted = sampler.sample(
            model=self,
            obs=obs,
            actions=actions,
            goal=goal,
            flow_steps=flow_steps,
            cfg_scale=cfg_scale,
        ).squeeze(0)

        new_latent_history = torch.cat(
            [state.latent_history[1:], predicted.unsqueeze(0)], dim=0
        )
        new_action_history = torch.cat(
            [state.action_history[1:], action.unsqueeze(0)], dim=0
        )
        new_state = DiTRolloutState(
            latent_history=new_latent_history,
            action_history=new_action_history,
            goal_latent=state.goal_latent,
        )
        return new_state, predicted

    def update_goal(self, state: DiTRolloutState, goal_latent: torch.Tensor) -> DiTRolloutState:
        return state.with_goal(goal_latent)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, vae) -> torch.Tensor:
        """Pass-through to vae helpers; exposed for env adapter convenience."""
        from nxwm.inference.vae import LATENT_SCALE

        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
        return vae.decode((latent / LATENT_SCALE).half()).sample
