"""Rollout buffer + collection for PPO.

Drives an ``nxwm`` ``WorldModel`` via ``init_rollout_state`` / ``step_rollout``
+ ``FlowMatchingSampler``. Reward is computed by a caller-supplied callable
``reward_fn(action, predicted_latent, info) -> (reward, components_dict)``.

This stays game-agnostic — the 13-component pokemon_za reward stack is a
plug-in. Generic rewards (action_entropy, frame_difference, stagnation)
live in ``nxwm.env.rewards.generic`` and can be wired in via config.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from nxwm.inference.flow_matching import FlowMatchingSampler

from nxrl.algorithms.ppo.config import PPOAlgorithmConfig, RolloutSpec, SeedSpec
from nxrl.policies.ppo_policy_v1 import PPOPolicyV1

RewardFn = Callable[[torch.Tensor, torch.Tensor, dict], tuple[float, dict]]  # pyright: ignore[reportMissingTypeArgument]


@dataclass
class RolloutBuffer:
    observations: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    reward_components: list[dict] = field(default_factory=list)  # pyright: ignore[reportMissingTypeArgument]
    terminal: bool = False
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_gae(self, gamma: float, gae_lambda: float, last_value: float) -> None:
        n = len(self.rewards)
        advantages = torch.zeros(n)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value - self.values[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae
        self.advantages = advantages
        self.returns = advantages + torch.tensor(self.values)

    def flatten(self) -> dict[str, torch.Tensor]:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("compute_gae() must be called before flatten()")
        return {
            "observations": torch.stack(self.observations),
            "actions": torch.stack(self.actions),
            "log_probs": torch.tensor(self.log_probs),
            "advantages": self.advantages,
            "returns": self.returns,
        }


def merge_buffers(buffers: list[RolloutBuffer]) -> dict[str, torch.Tensor]:
    """Concat per-rollout flattened buffers (kept on CPU; PPO moves minibatches
    to GPU). Caller normalizes advantages — single-GPU normalizes immediately
    after merging; DDP defers until after the cross-rank all-gather so the
    normalization is computed over the global batch, not per-rank-local
    distributions (which would bias the gradient).
    """
    flats = [b.flatten() for b in buffers]
    merged: dict[str, torch.Tensor] = {}
    for key in flats[0]:
        merged[key] = torch.cat([f[key] for f in flats], dim=0)
    return merged


def normalize_advantages(merged: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Standard PPO advantage normalization (zero-mean, unit-variance). In-place
    on the ``advantages`` key. Returns the same dict for chaining."""
    adv = merged["advantages"]
    merged["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)
    return merged


def load_seed_episode(seed: SeedSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Load latents + actions for one seed scene from a chunked latent .npz."""
    data = np.load(Path(seed.npz_path), mmap_mode="r")
    latents = torch.from_numpy(data["latents"][:].astype(np.float32))
    actions = torch.from_numpy(data["actions"][:].astype(np.float32))
    return latents, actions


@torch.no_grad()
def collect_rollout(
    policy: PPOPolicyV1,
    world_model: torch.nn.Module,
    sampler: FlowMatchingSampler,
    latents: torch.Tensor,
    actions_data: torch.Tensor,
    *,
    config: PPOAlgorithmConfig,
    rollout: RolloutSpec,
    reward_fn: RewardFn,
    device: torch.device,
    start_frame: int = 0,
    predicted_latents_out: list[torch.Tensor] | None = None,
) -> RolloutBuffer:
    """Roll out one trajectory. Policy drives the world model; reward_fn is
    called every step with ``(action, predicted_latent, info)``.

    Negative ``start_frame`` zero-pads: ``start_frame=-K`` means the seed
    frame should be the K-th from the end of the context window; the
    missing prefix is zero-padded.

    If ``predicted_latents_out`` is provided, each step's predicted latent is
    appended to it (CPU tensors). Used by the offline debug runner to decode
    frames after the rollout completes; training leaves it at ``None``.
    """
    bc_seq_len = policy.sequence_length
    n_frames = rollout.frames
    flow_steps = config.flow_steps
    cfg_scale = config.cfg_scale
    goal_offset = rollout.goal_offset

    total_frames = len(latents)

    pad_frames = 0
    if start_frame < 0:
        pad_frames = -start_frame
        start_frame = 0

    min_needed = bc_seq_len - pad_frames + n_frames + goal_offset
    if start_frame + min_needed > total_frames:
        start_frame = max(0, total_frames - min_needed)

    available_rollout = total_frames - start_frame - (bc_seq_len - pad_frames)
    if available_rollout < n_frames:
        n_frames = max(1, available_rollout)

    real_context_len = bc_seq_len - pad_frames
    real_latents = latents[start_frame : start_frame + real_context_len]
    real_actions = actions_data[start_frame : start_frame + real_context_len]
    if pad_frames > 0:
        lat_shape = latents.shape[1:]
        act_shape = actions_data.shape[1:]
        pad_lats = torch.zeros(pad_frames, *lat_shape)
        pad_acts = torch.zeros(pad_frames, *act_shape)
        context_latents = torch.cat([pad_lats, real_latents], dim=0)
        context_actions = torch.cat([pad_acts, real_actions], dim=0)
    else:
        context_latents = real_latents
        context_actions = real_actions

    policy_obs = context_latents.unsqueeze(0).to(device)

    # World-model rollout state — uses the WM's own seq_len, which may differ
    # from the policy's bc_seq_len. The WM only needs its last `wm_seq_len`
    # frames of history; the policy gets the full bc_seq_len window.
    wm_seq_len = world_model.config.seq_len  # type: ignore
    wm_initial_latents = context_latents[-wm_seq_len:].to(device)  # type: ignore
    wm_initial_actions = context_actions[-wm_seq_len:].to(device)  # type: ignore

    # Initial goal — receding goal advances each step.
    initial_goal_idx = start_frame + real_context_len + goal_offset
    initial_goal = latents[min(initial_goal_idx, total_frames - 1)].to(device)
    state = world_model.init_rollout_state(  # type: ignore
        wm_initial_latents, wm_initial_actions, initial_goal
    )

    buf = RolloutBuffer()
    terminal = False

    for step in range(n_frames):
        buf.observations.append(policy_obs.squeeze(0).cpu())

        action, log_prob, _entropy, value = policy.get_action_and_value(policy_obs)

        # Receding goal: advance one frame per step.
        goal_idx = start_frame + real_context_len + step + goal_offset
        if goal_idx < total_frames:
            new_goal = latents[goal_idx].to(device)
            state = world_model.update_goal(state, new_goal)  # type: ignore

        new_state, predicted = world_model.step_rollout(  # type: ignore
            state,
            action.squeeze(0),
            sampler=sampler,
            flow_steps=flow_steps,
            cfg_scale=cfg_scale,
        )
        state = new_state
        if predicted_latents_out is not None:
            predicted_latents_out.append(predicted.detach().cpu())

        info = {
            "step": step,
            "n_frames": n_frames,
            # Pokémon-ZA's reward (and any future game-specific reward) needs
            # access to the seed episode's latents to extract start/goal frames
            # at config-specified indices. Game-agnostic rewards just ignore
            # this key.
            "seed_latents": latents,
        }
        reward, components = reward_fn(action, predicted.unsqueeze(0), info)

        buf.actions.append(action.squeeze(0).cpu())
        buf.rewards.append(float(reward))
        buf.reward_components.append(components)
        buf.values.append(value.item())
        buf.log_probs.append(log_prob.item())

        # Slide the policy window with the predicted latent.
        policy_obs = torch.cat([policy_obs[:, 1:], predicted.unsqueeze(0).unsqueeze(0)], dim=1)

        if components.get("__terminal__"):
            terminal = True
            break

    if terminal:
        last_value_f = 0.0
    else:
        _, last_value = policy.forward(policy_obs)
        last_value_f = last_value.item()
    buf.terminal = terminal
    buf.compute_gae(config.gamma, config.gae_lambda, last_value_f)
    return buf
