"""PPO trainer.

Clipped surrogate + value loss + entropy bonus + BC anchor against a frozen
reference policy. Registered as ``"ppo"`` in ``algorithm_registry``.

Constructor surface differs from ``BCTrainer`` (no train_loader/val_loader;
the launcher hands in pre-built rollout dependencies). The launcher
dispatches on ``algorithm.name`` to choose the right orchestration shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nxml_core.checkpoint import save_checkpoint
from nxwm.inference.flow_matching import FlowMatchingSampler

from nxrl.algorithms.ppo.config import PPOAlgorithmConfig, RolloutSpec, SeedSpec
from nxrl.algorithms.ppo.rollout import (
    RewardFn,
    collect_rollout,
    load_seed_episode,
    merge_buffers,
    normalize_advantages,
)
from nxrl.core.registry import algorithm_registry
from nxrl.policies.ppo_policy_v1 import PPOPolicyV1


@algorithm_registry.register("ppo", config_cls=PPOAlgorithmConfig)
class PPOTrainer:
    def __init__(
        self,
        *,
        policy: nn.Module,
        frozen_policy: nn.Module,
        world_model: nn.Module,
        config: PPOAlgorithmConfig,
        rollout_spec: RolloutSpec,
        seeds: list[SeedSpec],
        reward_fn: RewardFn,
        device: torch.device,
        policy_name: str,
        policy_config,
        is_main_node: bool,
        run_dir: Path,
        eval_mkv_config: dict | None = None,
        policy_module: PPOPolicyV1 | None = None,
        rank: int = 0,
        world_size: int = 1,
        local_rollouts_per_update: int | None = None,
    ) -> None:
        # `policy` may be a DDP-wrapped PPOPolicyV1 (when world_size > 1) or
        # bare. `policy_module` is always the inner PPOPolicyV1 — used for
        # custom-attribute / custom-method access (e.g. `sequence_length`,
        # `get_action_and_value`, `evaluate_actions`) which DDP does not
        # auto-delegate. Forward calls during the gradient pass still go
        # through `self.policy(...)` so DDP's backward sync fires.
        self.policy = policy
        self.policy_module = policy_module if policy_module is not None else policy
        self.frozen_policy = frozen_policy
        self.world_model = world_model
        # DDP coordinates: rank/world_size are 0/1 in single-GPU mode (back-compat).
        # local_rollouts_per_update defaults to the full global count (= single-GPU
        # behavior). The launcher computes it as rollouts_per_update // world_size
        # when running DDP — see _ppo_main_worker in launcher.py.
        self.rank = rank
        self.world_size = world_size
        self.local_rollouts_per_update = (
            local_rollouts_per_update
            if local_rollouts_per_update is not None
            else config.rollouts_per_update
        )
        self.config = config
        self.rollout_spec = rollout_spec
        self.seeds = list(seeds)
        self.reward_fn = reward_fn
        self.device = device
        self.policy_name = policy_name
        self.policy_config = policy_config
        self.is_main_node = is_main_node
        self.run_dir = run_dir
        # eval_mkv_config: { every_n_updates: int, seed_indices: list[int], fps: int }
        # every_n_updates=0 disables the hook. Eval only runs on main node and
        # only when the reward fn exposes a VAE (pokemon_za:make_reward_fn does).
        self.eval_mkv_config = eval_mkv_config or {}

        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, eps=1e-5)
        self.sampler = FlowMatchingSampler()

        # Apply per-button logit bias from config (e.g. encourage attack-button presses).
        # Goes through policy_module: `button_bias` is on the inner module, and
        # `register_buffer` makes it visible via DDP's __getattr__ fallback only
        # for `get_buffer(...)` (not `.button_bias` direct access on the wrapper).
        if config.button_logit_bias_indices:
            with torch.no_grad():
                bias = self.policy_module.get_buffer("button_bias").clone()
                for idx in config.button_logit_bias_indices:
                    bias[idx] = config.button_logit_bias_value
                self.policy_module.button_bias.copy_(bias)  # type: ignore[arg-type]

        self._wandb_run = None
        if self.is_main_node:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._init_wandb()

        self._update = 0
        self._loaded_seed_data: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # Best-checkpoint tracking: rolling 5-update mean of rollout/mean_reward,
        # save best.pt whenever the smoothed value sets a new high.
        # Smoothing window=5 dampens single-update noise without lagging too far
        # behind the actual training signal.
        self._reward_window: list[float] = []
        self._best_smoothed_reward = float("-inf")
        self._best_update = 0

    def _init_wandb(self) -> None:
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError:
            print("(wandb not installed; skipping logging)")
            return
        from dataclasses import asdict, is_dataclass

        cfg_dict = {
            "policy": {
                "name": self.policy_name,
                "config": asdict(self.policy_config) if is_dataclass(self.policy_config) and not isinstance(self.policy_config, type) else dict(self.policy_config),  # type: ignore[arg-type]
            },
            "algorithm": asdict(self.config),
            "rollout": asdict(self.rollout_spec),
            "n_seeds": len(self.seeds),
            "run_dir": str(self.run_dir),
        }
        self._wandb_run = wandb.init(project=self.config.project_name, config=cfg_dict)

    def _wandb_log(self, payload: dict) -> None:
        if self._wandb_run is None:
            return
        import wandb  # type: ignore[import-not-found]

        wandb.log(payload)

    def _seed_data(self, seed: SeedSpec) -> tuple[torch.Tensor, torch.Tensor]:
        # Cache loaded latents+actions across updates so we don't re-mmap.
        key = seed.npz_path
        if key not in self._loaded_seed_data:
            self._loaded_seed_data[key] = load_seed_episode(seed)
        return self._loaded_seed_data[key]

    def collect_update_buffers(self):
        self.policy.eval()
        buffers = []
        rewards = []
        terminal_count = 0
        ep_lengths = []
        # SeedSpec.start_frame is "where the agent starts generating" (per its
        # docstring). collect_rollout's start_frame is "where the context
        # window starts" — convert by backing up one bc_seq_len.
        bc_seq_len = self.policy_module.sequence_length
        # Rank-offset seed selection: rank 0 takes the first chunk of seeds,
        # rank 1 takes the next chunk, etc. Distinct seeds per rank so DDP
        # all-gather (step 5) yields the same global rollout set as single-GPU
        # (where one rank would do all 10 sequentially). Single-GPU is
        # bit-equivalent because rank=0, local=global, offset is 0.
        seed_start = self.rank * self.local_rollouts_per_update
        for i in range(self.local_rollouts_per_update):
            seed = self.seeds[(seed_start + i) % len(self.seeds)]
            latents, actions_data = self._seed_data(seed)
            # Rollout uses policy_module (no_grad context — no need to route
            # through DDP since no gradients flow; also `get_action_and_value`
            # is a custom method that DDP doesn't delegate).
            buf = collect_rollout(
                self.policy_module,
                self.world_model,
                self.sampler,
                latents,
                actions_data,
                config=self.config,
                rollout=self.rollout_spec,
                reward_fn=self.reward_fn,
                device=self.device,
                start_frame=seed.start_frame - bc_seq_len,
            )
            buffers.append(buf)
            rewards.extend(buf.rewards)
            terminal_count += int(buf.terminal)
            ep_lengths.append(len(buf))
        merged = merge_buffers(buffers)

        # DDP: gather buffers + stats from all ranks so every rank's _ppo_update
        # sees the full global batch. Single-GPU skips both. Advantage
        # normalization happens AFTER the gather so it's computed over the
        # global distribution, not per-rank-local (which would bias gradients).
        if self.world_size > 1:
            merged = self._all_gather_merged(merged)
            total_reward, total_count, total_terminal, total_ep_sum, total_ep_n = (
                self._all_reduce_stats(
                    sum(rewards), len(rewards), terminal_count,
                    sum(ep_lengths), len(ep_lengths),
                )
            )
        else:
            total_reward = float(sum(rewards))
            total_count = len(rewards)
            total_terminal = terminal_count
            total_ep_sum = sum(ep_lengths)
            total_ep_n = len(ep_lengths)

        normalize_advantages(merged)

        stats = {
            "rollout/mean_reward": float(total_reward / max(total_count, 1)),
            "rollout/terminal_frac": float(total_terminal / max(total_ep_n, 1)),
            "rollout/mean_episode_len": float(total_ep_sum / max(total_ep_n, 1)),
        }
        return merged, stats

    def _all_gather_merged(
        self, local_merged: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """All-gather a per-rank merged dict across DDP ranks.

        Per-rank buffer lengths may differ (rollouts terminate at different
        times via the verified-lock terminal). Handles this with pad-and-trim:
          1) all-gather per-rank N (timestep count)
          2) for each key: pad to max(N) → NCCL all_gather → trim & concat

        Tensors are temporarily moved to GPU for NCCL, then back to CPU to
        match the rest of the trainer (which expects merged on CPU and moves
        minibatches to GPU per-iteration).
        """
        import torch.distributed as dist

        # 1. Discover per-rank lengths
        local_n = local_merged["observations"].shape[0]
        n_t = torch.tensor([local_n], device=self.device, dtype=torch.int64)
        all_ns = [torch.zeros_like(n_t) for _ in range(self.world_size)]
        dist.all_gather(all_ns, n_t)
        lengths = [int(n.item()) for n in all_ns]
        max_n = max(lengths)

        # 2. For each tensor key, pad-on-GPU → all_gather → trim → cat → back to CPU
        global_merged: dict[str, torch.Tensor] = {}
        for key, t in local_merged.items():
            gpu_t = t.to(self.device, non_blocking=True)
            if gpu_t.shape[0] < max_n:
                pad_shape = (max_n - gpu_t.shape[0],) + tuple(gpu_t.shape[1:])
                pad = torch.zeros(pad_shape, dtype=gpu_t.dtype, device=self.device)
                gpu_t = torch.cat([gpu_t, pad], dim=0)
            gathered = [torch.zeros_like(gpu_t) for _ in range(self.world_size)]
            dist.all_gather(gathered, gpu_t)
            global_merged[key] = torch.cat(
                [g[: lengths[r]] for r, g in enumerate(gathered)], dim=0
            ).cpu()
        return global_merged

    def _all_reduce_stats(
        self,
        reward_sum: float,
        reward_count: int,
        terminal_count: int,
        ep_len_sum: int,
        ep_len_count: int,
    ) -> tuple[float, int, int, int, int]:
        """SUM these five scalars across ranks via NCCL all-reduce."""
        import torch.distributed as dist

        t = torch.tensor(
            [reward_sum, reward_count, terminal_count, ep_len_sum, ep_len_count],
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        vals = t.tolist()
        return (
            float(vals[0]),
            int(vals[1]),
            int(vals[2]),
            int(vals[3]),
            int(vals[4]),
        )

    def _maybe_eval_mkv(self) -> None:
        """Periodically run an eval rollout, render mkv, upload to wandb.

        Disabled unless ``eval_mkv_config["every_n_updates"] > 0`` AND the
        reward fn exposes a VAE (pokemon_za:make_reward_fn attaches one to
        the returned closure). Only runs on the main node.
        """
        cfg = self.eval_mkv_config
        every_n = int(cfg.get("every_n_updates", 0))
        if every_n <= 0 or not self.is_main_node:
            return
        if self._update % every_n != 0:
            return
        vae = getattr(self.reward_fn, "vae", None)
        if vae is None:
            return  # Reward fn doesn't expose a VAE; can't decode latents.

        from nxrl.algorithms.ppo.rollout import collect_rollout
        from nxrl.cli_impl.rollout_debug import render_rollout_frames

        seed_indices = cfg.get("seed_indices") or [0]
        fps = int(cfg.get("fps", 30))
        out_dir = self.run_dir / "eval_mkvs"
        out_dir.mkdir(parents=True, exist_ok=True)
        bc_seq_len = self.policy_module.sequence_length
        self.policy.eval()

        for s_idx in seed_indices:
            if s_idx < 0 or s_idx >= len(self.seeds):
                continue
            seed = self.seeds[s_idx]
            seed_latents, seed_actions = self._seed_data(seed)

            # Fix RNG per (seed_idx) so eval mkvs across updates differ only
            # because the policy weights changed, not because action sampling
            # diverged. Makes wandb side-by-side comparisons meaningful.
            torch.manual_seed(s_idx + 1)

            preds: list[torch.Tensor] = []
            buf = collect_rollout(
                self.policy_module,
                self.world_model,
                self.sampler,
                seed_latents,
                seed_actions,
                config=self.config,
                rollout=self.rollout_spec,
                reward_fn=self.reward_fn,
                device=self.device,
                start_frame=seed.start_frame - bc_seq_len,
                predicted_latents_out=preds,
            )
            if not preds:
                continue
            from pathlib import Path as _P
            episode_stem = _P(seed.npz_path).stem
            seed_label = f"seed{s_idx:02d} {episode_stem}@{seed.start_frame} u{self._update:04d}"
            try:
                # Render frames as an ndarray. We pass these directly to
                # wandb.Video — wandb re-encodes via moviepy/ffmpeg into a
                # browser-friendly H.264 mp4 (cv2 in this env can't write
                # H.264, only mp4v which won't play inline in wandb's UI).
                frames_bgr = render_rollout_frames(
                    predicted_latents=preds,
                    reward_components=buf.reward_components,
                    vae=vae,
                    seed_label=seed_label,
                    terminal=buf.terminal,
                    device=self.device,
                )
            except Exception as e:
                print(f"[ppo eval-mkv] render failed for seed{s_idx}: {e}")
                continue

            try:
                import wandb  # type: ignore[import-not-found]

                # wandb.Video(ndarray) expects (T, C, H, W) in RGB.
                # frames_bgr is (T, H, W, 3) BGR — convert.
                import cv2

                rgb = np.stack(
                    [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr],
                    axis=0,
                )
                video_tchw = rgb.transpose(0, 3, 1, 2)  # (T, 3, H, W)
                self._wandb_log(
                    {
                        f"eval/seed{s_idx:02d}_video": wandb.Video(
                            video_tchw, fps=fps, format="mp4"
                        ),
                        f"eval/seed{s_idx:02d}_reward_sum": float(sum(buf.rewards)),
                        f"eval/seed{s_idx:02d}_terminal": int(buf.terminal),
                        f"eval/seed{s_idx:02d}_frames": len(preds),
                    }
                )
            except ImportError:
                pass
            except Exception as e:
                # wandb upload failed (rate-limit, network, bad format, etc.) —
                # log and continue. An eval-mkv hiccup must not kill training.
                print(f"[ppo eval-mkv] wandb upload failed for seed{s_idx}: {e}")

    def update_step(self) -> dict[str, float]:
        import time

        t0 = time.perf_counter()
        merged, rollout_stats = self.collect_update_buffers()
        metrics = self._ppo_update(merged)
        metrics.update(rollout_stats)
        self._update += 1
        dt = time.perf_counter() - t0
        metrics["update_seconds"] = dt
        # Track 5-update smoothed reward; on new high, save best.pt (main_node only).
        self._reward_window.append(float(rollout_stats["rollout/mean_reward"]))
        if len(self._reward_window) > 5:
            self._reward_window.pop(0)
        smoothed = sum(self._reward_window) / len(self._reward_window)
        metrics["smoothed_reward_5"] = smoothed
        if self.is_main_node and smoothed > self._best_smoothed_reward:
            self._best_smoothed_reward = smoothed
            self._best_update = self._update
            self.save_checkpoint(
                name="best.pt",
                metadata={
                    "smoothed_reward_5": smoothed,
                    "raw_reward": rollout_stats["rollout/mean_reward"],
                    "terminal_frac": rollout_stats["rollout/terminal_frac"],
                    "best_update": self._update,
                },
            )
            print(
                f"  [ppo] new best (smoothed_reward_5={smoothed:+.4f} "
                f"at update {self._update}) → best.pt"
            )
        if self.is_main_node:
            self._wandb_log({"update": self._update, **metrics})
        self._maybe_eval_mkv()
        if self.is_main_node:
            print(
                f"  update {self._update} "
                f"t={dt:5.1f}s "
                f"reward={rollout_stats['rollout/mean_reward']:+.3f} "
                f"policy_loss={metrics['policy_loss']:+.4f} "
                f"value_loss={metrics['value_loss']:.4f} "
                f"bc_loss={metrics['bc_loss']:.4f}"
            )
        return metrics

    def _ppo_update(self, rollout_data: dict[str, torch.Tensor]) -> dict[str, float]:
        self.policy.train()
        obs = rollout_data["observations"]
        actions = rollout_data["actions"]
        old_log_probs = rollout_data["log_probs"]
        advantages = rollout_data["advantages"]
        returns = rollout_data["returns"]

        n = obs.shape[0]
        cfg = self.config

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_bc_loss = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for _epoch in range(cfg.ppo_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, cfg.minibatch_size):
                end = min(start + cfg.minibatch_size, n)
                idx = perm[start:end]
                mb_obs = obs[idx].to(self.device)
                mb_actions = actions[idx].to(self.device)
                mb_old_lp = old_log_probs[idx].to(self.device)
                mb_adv = advantages[idx].to(self.device)
                mb_returns = returns[idx].to(self.device)

                # Single forward through DDP-wrapped policy. Both the PPO
                # surrogate (needs action_out + value + log_std + button_bias)
                # and the BC anchor (needs raw action_out) come from this one
                # call — avoids a 2x-forward pattern that breaks DDP gradient
                # sync (DDP expects one forward per backward) and halves the
                # per-minibatch wall-clock.
                from torch.distributions import Bernoulli, Independent, Normal

                action_out, new_value = self.policy(mb_obs)
                stick_mean = action_out[:, :4]
                # `button_bias` lives on the inner module; via DDP's __getattr__
                # buffer-fallback it's also reachable as `self.policy.button_bias`
                # but going through `policy_module` is explicit.
                button_logits = action_out[:, 4:] + self.policy_module.button_bias
                stick_std = self.policy_module.log_std.exp().expand_as(stick_mean)
                stick_dist = Independent(Normal(stick_mean, stick_std), 1)
                button_dist = Independent(Bernoulli(logits=button_logits), 1)
                new_lp = (
                    stick_dist.log_prob(mb_actions[:, :4])
                    + button_dist.log_prob(mb_actions[:, 4:])
                )
                entropy = stick_dist.entropy() + button_dist.entropy()

                ratio = (new_lp - mb_old_lp).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_value, mb_returns)

                with torch.no_grad():
                    frozen_out = self.frozen_policy(mb_obs)
                # BC anchor uses the same forward's action_out. Crucially this
                # is the BIAS-FREE action_out (button_bias is added separately
                # above for the PPO loss only) so the BC reference compares
                # against the true underlying logits.
                current_out = action_out

                bc_left_stick_loss = F.mse_loss(current_out[:, :2], frozen_out[:, :2])
                bc_right_stick_loss = F.mse_loss(current_out[:, 2:4], frozen_out[:, 2:4])
                bc_button_loss = F.binary_cross_entropy_with_logits(
                    current_out[:, 4:], torch.sigmoid(frozen_out[:, 4:])
                )

                right_stick_coef = (
                    cfg.bc_reg_right_stick_coef
                    if cfg.bc_reg_right_stick_coef is not None
                    else cfg.bc_reg_coef
                )
                bc_loss = bc_left_stick_loss + bc_right_stick_loss + bc_button_loss

                loss = (
                    policy_loss
                    + cfg.vf_coef * value_loss
                    - cfg.ent_coef * entropy.mean()
                    + cfg.bc_reg_coef * (bc_left_stick_loss + bc_button_loss)
                    + right_stick_coef * bc_right_stick_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_lp - new_lp).mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_bc_loss += bc_loss.item()
                total_approx_kl += approx_kl.item()
                n_updates += 1

        n_div = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n_div,
            "value_loss": total_value_loss / n_div,
            "entropy": total_entropy / n_div,
            "bc_loss": total_bc_loss / n_div,
            "approx_kl": total_approx_kl / n_div,
            "log_std_mean": self.policy_module.log_std.data.mean().item(),
        }

    def save_checkpoint(self, *, name: str = "latest.pt", metadata: dict | None = None) -> None:
        if not self.is_main_node:
            return
        meta = {"update": self._update}
        meta.update(metadata or {})
        # Save the inner module's state_dict (DDP's wrapper prefixes keys with
        # `module.`, which existing loaders/checkpoints don't expect).
        save_checkpoint(
            architecture=self.policy_name,
            config=self.policy_config,
            state_dict=self.policy_module.state_dict(),
            path=self.run_dir / name,
            extra={
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_metadata": meta,
                "algorithm": "ppo",
            },
        )
