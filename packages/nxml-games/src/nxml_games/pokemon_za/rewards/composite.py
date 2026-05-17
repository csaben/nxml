"""Reward shaping for RL training — v2 with trajectory-aware rewards.

Components:
  - Action matching (v1): reward stick positions near targets
  - Progress vector: reward stick delta toward a goal direction
  - Stagnation penalty: penalize staying near start frame in latent space
  - Goal proximity: reward approaching goal frame in latent space
  - Start divergence: penalize remaining too similar to start frame
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from nxml_games.pokemon_za.config import PNGSimilarityPenaltyConfig, RewardConfig, StickTarget
from nxml_games.pokemon_za.target_ui_detector import TargetUIDetector

LATENT_SCALE = 0.18215


class RewardShaper:
    """Computes per-step rewards using action-based and observation-based signals.

    Requires start_latent and goal_latent to be set before computing rewards
    for observation-based components (stagnation, goal proximity, start divergence).
    """

    def __init__(self, config: RewardConfig, vae=None, device: torch.device | None = None):
        self.config = config
        self.vae = vae
        self.targets = config.rewardable_action_state
        self.reward_values = config.reward_values

        # Target UI detector (optional, requires VAE for latent decoding)
        self.detector: TargetUIDetector | None = None
        if config.target_ui_detection is not None:
            tui = config.target_ui_detection
            self.detector = TargetUIDetector(
                tui.template_path,
                threshold=tui.score_threshold,
                sat_threshold=tui.sat_threshold,
                min_consecutive_hits=tui.min_consecutive_hits,
            )

        # v1: action target specs
        self._target_specs: list[tuple[list[int], StickTarget, float]] = []
        for name, target in self.targets.items():
            reward_key = f"{name}-reward"
            penalty_key = f"{name}-penalty"
            reward_val = self.reward_values.get(
                reward_key, self.reward_values.get(penalty_key, 0.0)
            )
            indices = self._name_to_indices(name)
            if indices is not None:
                self._target_specs.append((indices, target, reward_val))

        # v2: progress vector setup
        self._prev_stick: torch.Tensor | None = None
        if config.progress_vector:
            pv = config.progress_vector
            self._pv_indices = [0, 1] if pv.stick == "left" else [2, 3]
            self._pv_target_dir = torch.tensor(pv.target_direction)
            norm = self._pv_target_dir.norm()
            if norm > 1e-6:
                self._pv_target_dir = self._pv_target_dir / norm

        # Reference latents (set by rollout before stepping)
        self.start_latent: torch.Tensor | None = None
        self.goal_latent: torch.Tensor | None = None

        # Per-rollout: counter for stuck-turn reward (frames pushing forward
        # without seeing a target). Resets in reset() and on payout.
        self._stuck_forward_counter: int = 0
        # Per-rollout: consecutive frames of sustained turn input (for stuck_turn).
        self._turn_frame_counter: int = 0

        # Per-rollout state for unique_acquisition reward.
        self._prev_locked: bool = False
        self._unique_lock_count: int = 0
        # Latent at the frame of the most recent lock BEGIN (or rollout start).
        # Used to gate new locks on having actually moved through the world.
        self._last_lock_begin_latent: torch.Tensor | None = None
        # Frames spent unlocked since the last release (anti-flicker guard).
        self._unlocked_frame_count: int = 0
        # Deferred-payout mode (require_attack_press): track the current lock.
        self._cur_lock_begin_latent: torch.Tensor | None = None
        self._cur_lock_attack_pressed: bool = False

        # Escape reward state.
        self._escape_paid: bool = False
        self._escape_initial_dist_sum: float = 0.0
        self._escape_initial_frame_count: int = 0

        # Sliding-window stagnation reward state. Mirrors the wm-player
        # stuck-detector overlay (sliding mode): rolling history of recent
        # latents + a streak counter for "barely moved vs window frames ago".
        self._slide_stag_history: list[torch.Tensor] = []
        self._slide_stag_streak: int = 0

        # Unstuck reward state. Independent rolling window from
        # sliding_stagnation so the two can run with different parameters.
        self._unstuck_history: list[torch.Tensor] = []
        self._unstuck_streak: int = 0
        self._unstuck_has_been_stuck: bool = False

        # Stuck_turn stagnation-trigger state:
        # Ring buffer of recent latents (appended each step), compared against
        # the one from `stagnation_window` frames ago.
        self._latent_history: list[torch.Tensor] = []
        # Once stagnation has been detected for `stuck_threshold_frames`,
        # this latches to True until the sweep completes or times out.
        self._pending_sweep: bool = False
        self._pending_sweep_age: int = 0
        # Count of consecutive stagnant frames (toward the latch threshold).
        self._stagnant_frame_counter: int = 0

        # PNG similarity penalty: VAE-encode reference PNGs to latents at init.
        self._penalty_ref_latents: list[torch.Tensor] = []
        self._penalty_streaks: list[int] = []
        if config.png_similarity_penalty and config.png_similarity_penalty.paths and vae is not None:
            self._init_penalty_latents(config.png_similarity_penalty, vae, device)

    @staticmethod
    def _name_to_indices(name: str) -> list[int] | None:
        mapping = {
            "left-stick": [0, 1],
            "right-stick": [2, 3],
            "left-stick-forward": [0, 1],
            "right-stick-forward": [2, 3],
        }
        return mapping.get(name)

    @torch.no_grad()
    def _init_penalty_latents(
        self,
        cfg: PNGSimilarityPenaltyConfig,
        vae,
        device: torch.device | None,
    ) -> None:
        """Load penalty PNGs, VAE-encode to reference latents."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for path in cfg.paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"WARNING: penalty PNG not found, skipping: {path}")
                continue
            img = cv2.resize(img, (256, 128))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            tensor = (tensor - 0.5) / 0.5  # normalize to [-1, 1]
            latent = vae.encode(
                tensor.unsqueeze(0).half().to(device)
            ).latent_dist.mode()
            latent = latent.squeeze(0).float() * LATENT_SCALE
            self._penalty_ref_latents.append(latent)
        self._penalty_streaks = [0] * len(self._penalty_ref_latents)
        print(f"PNG similarity penalty: loaded {len(self._penalty_ref_latents)} reference latents")

    def reset(self):
        """Reset per-rollout state."""
        self._prev_stick = None
        self._stuck_forward_counter = 0
        self._turn_frame_counter = 0
        self._prev_locked = False
        self._unique_lock_count = 0
        self._last_lock_begin_latent = None
        self._unlocked_frame_count = 0
        self._cur_lock_begin_latent = None
        self._cur_lock_attack_pressed = False
        self._latent_history = []
        self._pending_sweep = False
        self._pending_sweep_age = 0
        self._stagnant_frame_counter = 0
        self._escape_paid = False
        self._escape_initial_dist_sum = 0.0
        self._escape_initial_frame_count = 0
        self._slide_stag_history = []
        self._slide_stag_streak = 0
        self._unstuck_history = []
        self._unstuck_streak = 0
        self._unstuck_has_been_stuck = False
        self._penalty_streaks = [0] * len(self._penalty_ref_latents)
        if self.detector is not None:
            self.detector.reset()

    def compute_reward(
        self,
        action: torch.Tensor,
        current_latent: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute reward for a single step.

        Args:
            action: (1, 26) — sticks[:4] in [-1,1], buttons[4:] in [0,1]
            current_latent: (1, 4, 16, 32) — current generated frame latent
            step: current step index (0-based)
            total_steps: total rollout length

        Returns:
            (reward, components) — reward is (1,) scalar, components is per-signal breakdown
        """
        device = action.device
        components: dict[str, float] = {}

        # --- v1: Action matching ---
        r_action = self._action_reward(action)
        components["action"] = r_action.item()

        # --- v2: Progress vector ---
        r_progress = torch.zeros(1, device=device)
        if self.config.progress_vector:
            r_progress = self._progress_reward(action)
        components["progress"] = r_progress.item()

        # --- v2: Stagnation penalty ---
        r_stagnation = torch.zeros(1, device=device)
        if self.config.stagnation_penalty and self.start_latent is not None:
            r_stagnation = self._stagnation_reward(current_latent, step, total_steps)
        components["stagnation"] = r_stagnation.item()

        # --- v2: Goal proximity ---
        r_goal = torch.zeros(1, device=device)
        if self.config.goal_proximity and self.goal_latent is not None:
            r_goal = self._goal_proximity_reward(current_latent)
        components["goal_prox"] = r_goal.item()

        # --- v2: Start divergence ---
        r_diverge = torch.zeros(1, device=device)
        if self.config.start_divergence and self.start_latent is not None:
            r_diverge = self._start_divergence_reward(current_latent, step, total_steps)
        components["start_div"] = r_diverge.item()

        # --- Stuck-turn reward (must run AFTER target_ui to read streak) ---
        r_stuck_turn = torch.zeros(1, device=device)
        if self.config.stuck_turn is not None:
            r_stuck_turn = self._stuck_turn_reward(action, current_latent)
        components["stuck_turn"] = r_stuck_turn.item()

        # --- Locked-attack reward (must run AFTER target_ui to read streak) ---
        r_locked_attack = torch.zeros(1, device=device)
        if (
            self.config.locked_attack is not None
            and self.detector is not None
        ):
            r_locked_attack = self._locked_attack_reward(action)
        components["lock_atk"] = r_locked_attack.item()

        # --- Movement rewards (positive pressure for stick-driven motion) ---
        r_movement = torch.zeros(1, device=device)
        if self.config.movement_rewards:
            r_movement = self._movement_reward(action)
        components["movement"] = r_movement.item()

        # --- Stick penalties ---
        r_stick_pen = torch.zeros(1, device=device)
        if self.config.stick_penalties:
            r_stick_pen = self._stick_penalty(action)
        components["stick_pen"] = r_stick_pen.item()

        # --- Target UI detection (decode latent → match popup template) ---
        r_target_ui = torch.zeros(1, device=device)
        if self.detector is not None and self.vae is not None:
            r_target_ui = self._target_ui_reward(current_latent)
        components["target_ui"] = r_target_ui.item()

        # --- Escape reward (one-shot bonus for breaking out of stuck-corner seed) ---
        r_escape = torch.zeros(1, device=device)
        if self.config.escape is not None and self.start_latent is not None:
            r_escape = self._escape_reward(current_latent, step)
        components["escape"] = r_escape.item()

        # --- Sliding-window stagnation (per-step penalty while pinned) ---
        r_slide_stag = torch.zeros(1, device=device)
        if self.config.sliding_stagnation is not None:
            r_slide_stag = self._sliding_stagnation_reward(current_latent, step)
        components["slide_stag"] = r_slide_stag.item()

        # --- Unstuck reward (paid only after first stuck event) ---
        r_unstuck = torch.zeros(1, device=device)
        if self.config.unstuck is not None:
            r_unstuck = self._unstuck_reward(current_latent, step)
        components["unstuck"] = r_unstuck.item()

        # --- Unique acquisition bonus (must run AFTER target_ui so streak is fresh) ---
        r_unique = torch.zeros(1, device=device)
        if (
            self.config.unique_acquisition is not None
            and self.detector is not None
        ):
            r_unique = self._unique_acquisition_reward(action, current_latent)
        components["unique_acq"] = r_unique.item()

        # --- PNG similarity penalty (latent-space cosine sim to bad-state refs) ---
        r_png_pen = torch.zeros(1, device=device)
        if self.config.png_similarity_penalty is not None and self._penalty_ref_latents:
            r_png_pen = self._png_similarity_penalty(current_latent)
        components["png_pen"] = r_png_pen.item()

        total = (
            r_action + r_progress + r_stagnation + r_goal + r_diverge
            + r_stick_pen + r_target_ui + r_movement + r_locked_attack
            + r_stuck_turn + r_unique + r_escape + r_slide_stag + r_unstuck
            + r_png_pen
        )
        components["total"] = total.item()
        return total, components

    def _action_reward(self, action: torch.Tensor) -> torch.Tensor:
        B = action.shape[0]
        reward = torch.zeros(B, device=action.device)

        for indices, target, reward_val in self._target_specs:
            actual = action[:, indices]
            target_vals = torch.tensor(
                target.values, device=action.device, dtype=action.dtype
            )
            thresholds = torch.tensor(
                target.thresholds, device=action.device, dtype=action.dtype
            )
            diff = actual - target_vals.unsqueeze(0)
            match_mask = torch.ones(B, dtype=torch.bool, device=action.device)
            for dim in range(len(indices)):
                thresh = thresholds[dim].item()
                if thresh >= 0:
                    match_mask &= diff[:, dim] >= -abs(thresh)
                else:
                    match_mask &= diff[:, dim] <= abs(thresh)
            reward += match_mask.float() * reward_val

        return reward

    def _progress_reward(self, action: torch.Tensor) -> torch.Tensor:
        """Reward stick delta in the target direction."""
        pv = self.config.progress_vector
        device = action.device
        current_stick = action[:, self._pv_indices]  # (1, 2)
        target_dir = self._pv_target_dir.to(device)

        if self._prev_stick is None:
            self._prev_stick = current_stick.clone()
            return torch.zeros(1, device=device)

        delta = current_stick - self._prev_stick  # (1, 2)
        self._prev_stick = current_stick.clone()

        mag = delta.norm(dim=1)  # (1,)
        if mag.item() < pv.min_magnitude:
            return torch.zeros(1, device=device)

        # dot product of normalized delta with target direction
        dot = (delta * target_dir.unsqueeze(0)).sum(dim=1)  # (1,)
        return dot.clamp(min=0) * pv.reward_per_step

    def _stagnation_reward(
        self, current_latent: torch.Tensor, step: int, total_steps: int
    ) -> torch.Tensor:
        """Penalize if current frame is too similar to start frame."""
        sp = self.config.stagnation_penalty
        if step < sp.grace_period:
            return torch.zeros(1, device=current_latent.device)

        dist = self._frame_distance(
            current_latent, self.start_latent, sp.metric
        )

        if sp.metric == "cosine":
            # High similarity = stuck. Penalize if similarity > threshold
            too_similar = dist > sp.threshold
        else:
            # Low L2 distance = stuck. Penalize if distance < threshold
            too_similar = dist < sp.threshold

        if too_similar:
            # Scale penalty by progress through rollout
            scale = step / max(total_steps, 1)
            return torch.tensor([sp.penalty_per_step * scale], device=current_latent.device)
        return torch.zeros(1, device=current_latent.device)

    def _goal_proximity_reward(self, current_latent: torch.Tensor) -> torch.Tensor:
        """Reward approaching the goal frame."""
        gp = self.config.goal_proximity
        dist = self._frame_distance(current_latent, self.goal_latent, gp.metric)

        reward = torch.zeros(1, device=current_latent.device)

        if gp.continuous:
            if gp.metric == "cosine":
                reward += dist * gp.continuous_scale  # similarity * scale
            else:
                # For L2: reward inversely proportional to distance
                reward += (1.0 / (1.0 + dist)) * gp.continuous_scale

        # Threshold bonus
        if gp.metric == "cosine":
            if dist > gp.threshold:
                reward += torch.tensor([gp.reward], device=current_latent.device)
        else:
            if dist < gp.threshold:
                reward += torch.tensor([gp.reward], device=current_latent.device)

        return reward

    def _start_divergence_reward(
        self, current_latent: torch.Tensor, step: int, total_steps: int
    ) -> torch.Tensor:
        """Penalize staying too similar to start frame."""
        sd = self.config.start_divergence
        dist = self._frame_distance(current_latent, self.start_latent, sd.metric)

        if sd.metric == "cosine":
            too_similar = dist > sd.threshold
        else:
            too_similar = dist < sd.threshold

        if too_similar:
            penalty = sd.penalty
            if sd.step_scaling:
                penalty *= step / max(total_steps, 1)
            return torch.tensor([penalty], device=current_latent.device)
        return torch.zeros(1, device=current_latent.device)

    @torch.no_grad()
    def _target_ui_reward(self, current_latent: torch.Tensor) -> torch.Tensor:
        """Decode latent → BGR uint8 → run detector → return reward."""
        device = current_latent.device
        decoded = self.vae.decode((current_latent / LATENT_SCALE).half()).sample
        img = ((decoded + 1.0) / 2.0).clamp(0, 1).squeeze(0).permute(1, 2, 0)
        img_np = img.float().cpu().numpy()
        img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        detected, _score, _sat = self.detector.detect(img_bgr)

        tui = self.config.target_ui_detection
        r = 0.0
        if detected:
            r += tui.reward_per_step
        if tui.streak_ramp_scale > 0 and tui.min_consecutive_hits > 0:
            ramp = min(self.detector.streak / tui.min_consecutive_hits, 1.0)
            r += ramp * tui.streak_ramp_scale
        return torch.tensor([r], device=device)

    def _escape_reward(
        self, current_latent: torch.Tensor, step: int
    ) -> torch.Tensor:
        """One-shot bonus for breaking out of a stuck-corner seed position.

        Pays `bonus` the first time the current latent diverges from the
        rollout start latent by `escape_distance` (cosine), gated on:
          - `step >= grace_period` (no instant-circle hacks)
          - `require_initial_stagnation`: average start-distance during the
            first `initial_stagnation_window` frames must be < threshold
            (the agent didn't immediately bolt off, suggesting it really
            was pinned by the WM collision and recognized it).

        After the one-shot fires, `sustain_reward_per_step` keeps paying
        a small per-step amount as long as the agent stays past the
        escape distance — discourages immediately bouncing back.
        """
        esc = self.config.escape
        assert esc is not None and self.start_latent is not None
        device = current_latent.device

        # Track initial-window stagnation regardless of payment state.
        if step < esc.initial_stagnation_window:
            d = self._frame_distance(
                current_latent, self.start_latent, "cosine"
            )
            d_val = d.item() if torch.is_tensor(d) else float(d)
            self._escape_initial_dist_sum += d_val
            self._escape_initial_frame_count += 1

        # Current cosine distance from start.
        cur_d = self._frame_distance(
            current_latent, self.start_latent, "cosine"
        )
        cur_d_val = cur_d.item() if torch.is_tensor(cur_d) else float(cur_d)

        if not self._escape_paid:
            if step < esc.grace_period:
                return torch.zeros(1, device=device)
            if cur_d_val < esc.escape_distance:
                return torch.zeros(1, device=device)

            # Optional initial-stagnation gate.
            if esc.require_initial_stagnation:
                if self._escape_initial_frame_count == 0:
                    return torch.zeros(1, device=device)
                avg_d = (
                    self._escape_initial_dist_sum
                    / self._escape_initial_frame_count
                )
                if avg_d > esc.initial_stagnation_threshold:
                    # Agent moved too much during the early window — not a
                    # legitimate "stuck → realize → escape" trajectory.
                    return torch.zeros(1, device=device)

            self._escape_paid = True
            return torch.tensor([esc.bonus], device=device)

        # Already paid: sustain reward while still beyond escape_distance.
        if cur_d_val >= esc.escape_distance:
            return torch.tensor([esc.sustain_reward_per_step], device=device)
        return torch.zeros(1, device=device)

    def _sliding_stagnation_reward(
        self, current_latent: torch.Tensor, step: int
    ) -> torch.Tensor:
        """Per-step penalty while the agent is pinned by sliding-window check.

        Mirrors the wm-player stuck-detector overlay (sliding mode):
          dist = 1 - cos_sim(current_latent, latent_from_window_frames_ago)
          if dist < cos_threshold for >= min_consecutive_hits frames:
              pay reward_per_step (typically negative)

        Anchor slides along the trajectory, so the agent can move legitimately
        across the level and only get penalized when it gets re-pinned.
        """
        ss = self.config.sliding_stagnation
        device = current_latent.device

        # Maintain bounded history of recent latents.
        self._slide_stag_history.append(current_latent.detach())
        if len(self._slide_stag_history) > ss.window + 1:
            self._slide_stag_history.pop(0)

        if step < ss.grace_period or len(self._slide_stag_history) <= ss.window:
            self._slide_stag_streak = 0
            return torch.zeros(1, device=device)

        past = self._slide_stag_history[-ss.window - 1]
        cos_sim = self._frame_distance(current_latent, past, "cosine")
        cos_dist = 1.0 - cos_sim

        if cos_dist < ss.cos_threshold:
            self._slide_stag_streak += 1
        else:
            self._slide_stag_streak = 0

        if self._slide_stag_streak >= ss.min_consecutive_hits:
            return torch.tensor([ss.reward_per_step], device=device)
        return torch.zeros(1, device=device)

    def _unstuck_reward(
        self, current_latent: torch.Tensor, step: int
    ) -> torch.Tensor:
        """Per-step reward for time spent free, gated on having been stuck once.

        Mirrors the wm-player overlay's sliding-mode detector. The reward is
        zero until the detector fires for the first time in this rollout
        (`has_been_stuck` latches True). After that, every frame where the
        agent is NOT currently stuck pays `reward_per_step`. While stuck
        again, payment pauses (no reward, no penalty); the latch persists,
        so the next escape immediately resumes payment.
        """
        us = self.config.unstuck
        device = current_latent.device

        self._unstuck_history.append(current_latent.detach())
        if len(self._unstuck_history) > us.window + 1:
            self._unstuck_history.pop(0)

        if step < us.grace_period or len(self._unstuck_history) <= us.window:
            return torch.zeros(1, device=device)

        past = self._unstuck_history[-us.window - 1]
        cos_sim = self._frame_distance(current_latent, past, "cosine")
        cos_dist = 1.0 - cos_sim

        if cos_dist < us.cos_threshold:
            self._unstuck_streak += 1
        else:
            self._unstuck_streak = 0

        currently_stuck = self._unstuck_streak >= us.min_consecutive_hits
        if currently_stuck:
            self._unstuck_has_been_stuck = True
            return torch.zeros(1, device=device)

        if self._unstuck_has_been_stuck:
            return torch.tensor([us.reward_per_step], device=device)
        return torch.zeros(1, device=device)

    def _unique_acquisition_reward(
        self, action: torch.Tensor, current_latent: torch.Tensor
    ) -> torch.Tensor:
        """Sparse bonus for each new target acquired during a rollout.

        Two payout modes:

        1) Default (`require_attack_press=False`) — pay on lock-BEGIN.
           Fires on unlocked→locked transitions that pass flicker + movement
           gates. Credit lands on the frame the lock-on is achieved.

        2) `require_attack_press=True` — pay on lock-RELEASE, gated on at
           least one attack button being pressed during the lock. Credit
           lands on the release frame; GAE carries it back to the attack-
           press frames, giving them positive advantage and forcing PPO to
           learn to press attack while locked.
        """
        ua = self.config.unique_acquisition
        tui = self.config.target_ui_detection
        assert ua is not None and self.detector is not None
        device = current_latent.device

        min_hits = tui.min_consecutive_hits if tui is not None else 1
        currently_locked = self.detector.streak >= min_hits

        reward = torch.zeros(1, device=device)

        anchor = (
            self._last_lock_begin_latent
            if self._last_lock_begin_latent is not None
            else self.start_latent
        )

        def _flicker_and_move_ok(lock_begin_latent: torch.Tensor) -> bool:
            flicker_ok = (
                self._unique_lock_count == 0
                or self._unlocked_frame_count >= ua.min_unlocked_frames
            )
            moved_enough = True
            if anchor is not None:
                cos_dist = self._frame_distance(
                    lock_begin_latent, anchor, "cosine"
                )
                cos_val = (
                    cos_dist.item()
                    if torch.is_tensor(cos_dist)
                    else float(cos_dist)
                )
                moved_enough = cos_val >= ua.min_latent_move
            return flicker_ok and moved_enough

        if ua.require_attack_press:
            # --- Deferred payout: pay on lock-release, require attack press ---

            # unlocked → locked: capture candidate begin latent, reset flag
            if currently_locked and not self._prev_locked:
                self._cur_lock_begin_latent = current_latent.detach().clone()
                self._cur_lock_attack_pressed = False

            # While locked: track any attack press
            if currently_locked:
                for idx in ua.attack_indices:
                    if action[0, idx].item() > 0.5:
                        self._cur_lock_attack_pressed = True
                        break

            # locked → unlocked: evaluate and maybe pay
            if self._prev_locked and not currently_locked:
                if (
                    self._cur_lock_attack_pressed
                    and self._cur_lock_begin_latent is not None
                    and _flicker_and_move_ok(self._cur_lock_begin_latent)
                ):
                    self._unique_lock_count += 1
                    scale = 1.0 + ua.chain_multiplier * (self._unique_lock_count - 1)
                    reward = reward + ua.bonus_per_target * scale
                    self._last_lock_begin_latent = self._cur_lock_begin_latent
                self._cur_lock_begin_latent = None
                self._cur_lock_attack_pressed = False
        else:
            # --- Immediate payout: pay on lock-begin ---
            if currently_locked and not self._prev_locked:
                if _flicker_and_move_ok(current_latent):
                    self._unique_lock_count += 1
                    scale = 1.0 + ua.chain_multiplier * (self._unique_lock_count - 1)
                    reward = reward + ua.bonus_per_target * scale
                    self._last_lock_begin_latent = current_latent.detach().clone()

        if currently_locked:
            self._unlocked_frame_count = 0
        else:
            self._unlocked_frame_count += 1

        self._prev_locked = currently_locked
        return reward

    def _stuck_turn_reward(
        self, action: torch.Tensor, current_latent: torch.Tensor
    ) -> torch.Tensor:
        """One-shot reward for committing to a camera turn after being stuck.

        Two trigger modes:

        - "forward": counts frames of pushing-forward-without-target.
          Good for running-into-a-wall detection.

        - "stagnation": counts frames where the latent has barely moved vs the
          latent `stagnation_window` frames ago. Uses a LATCH: once stagnation
          threshold is met, `pending_sweep=True` and the policy has a fixed
          window to complete a sustained camera sweep — even as latent changes
          during the rotation — before the latch expires. This is the only
          way to make "look around when stuck" work without the sweep itself
          resetting the trigger that enabled it.

        Payout requires both a stuck trigger AND `min_turn_frames` of
        consecutive right-stick turn input past `turn_threshold`. Payout
        consumes the trigger (anti-farming).
        """
        st = self.config.stuck_turn
        device = action.device

        seeing_target = (
            self.detector is not None and self.detector.streak > 0
        )

        # --- Turn input (shared across modes) ---
        turn_idx = 2 + st.turn_axis  # right stick → indices 2,3
        turn_val = action[0, turn_idx].item()
        is_turning = abs(turn_val) > st.turn_threshold

        trigger_mode = getattr(st, "trigger", "forward")

        if trigger_mode == "stagnation":
            # Maintain a bounded latent history.
            window = st.stagnation_window
            self._latent_history.append(current_latent.detach())
            if len(self._latent_history) > window + 1:
                self._latent_history.pop(0)

            is_stagnant = False
            if not seeing_target and len(self._latent_history) > window:
                past = self._latent_history[-window - 1]
                cos_dist = self._frame_distance(current_latent, past, "cosine")
                cos_val = (
                    cos_dist.item() if torch.is_tensor(cos_dist) else float(cos_dist)
                )
                is_stagnant = cos_val < st.stagnation_cos_threshold

            # Advance stagnation counter; resets if a target appears.
            if seeing_target:
                self._stagnant_frame_counter = 0
                self._pending_sweep = False
                self._pending_sweep_age = 0
            elif is_stagnant:
                self._stagnant_frame_counter += 1
            else:
                # Only reset the raw counter, NOT the latch — the latch is
                # supposed to survive the latent changing during the sweep.
                self._stagnant_frame_counter = 0

            # Latch when counter crosses threshold.
            if (
                not self._pending_sweep
                and self._stagnant_frame_counter >= st.stuck_threshold_frames
            ):
                self._pending_sweep = True
                self._pending_sweep_age = 0

            # Age the latch and expire it if the sweep never completes.
            if self._pending_sweep:
                self._pending_sweep_age += 1
                if self._pending_sweep_age > st.pending_sweep_timeout:
                    self._pending_sweep = False
                    self._pending_sweep_age = 0
                    self._stagnant_frame_counter = 0
                    self._turn_frame_counter = 0

            # Sustained-turn counter — only accumulates while latched.
            if self._pending_sweep and is_turning:
                self._turn_frame_counter += 1
            elif not is_turning:
                self._turn_frame_counter = 0

            if (
                self._pending_sweep
                and self._turn_frame_counter >= st.min_turn_frames
            ):
                self._pending_sweep = False
                self._pending_sweep_age = 0
                self._stagnant_frame_counter = 0
                self._turn_frame_counter = 0
                return torch.tensor([st.reward_per_event], device=device)

            return torch.zeros(1, device=device)

        # --- Legacy forward-trigger mode ---
        forward_idx = st.forward_axis  # left stick → indices 0,1
        forward_val = action[0, forward_idx].item()
        is_pushing_forward = forward_val > st.forward_threshold

        if seeing_target:
            self._stuck_forward_counter = 0
        elif is_pushing_forward:
            self._stuck_forward_counter += 1
        else:
            self._stuck_forward_counter = 0

        if (
            self._stuck_forward_counter >= st.stuck_threshold_frames
            and is_turning
        ):
            self._turn_frame_counter += 1
        else:
            self._turn_frame_counter = 0

        if self._turn_frame_counter >= st.min_turn_frames:
            self._stuck_forward_counter = 0
            self._turn_frame_counter = 0
            return torch.tensor([st.reward_per_event], device=device)

        return torch.zeros(1, device=device)

    def _locked_attack_reward(self, action: torch.Tensor) -> torch.Tensor:
        """Reward attack-button presses while target_ui detector is locked on.

        Scheme C: scales with streak as min(streak / min_consecutive_hits, 1.0),
        symmetric with the tui streak ramp. Gates on streak >= require_min_streak
        so partial flickers don't trigger. Reads `self.detector.streak` which
        was updated by `_target_ui_reward` earlier in this same step.
        """
        la = self.config.locked_attack
        device = action.device

        streak = self.detector.streak
        if streak < la.require_min_streak:
            return torch.zeros(1, device=device)

        min_hits = max(self.config.target_ui_detection.min_consecutive_hits, 1)
        ramp = min(streak / min_hits, 1.0)

        # Any of the attack buttons pressed (post-Bernoulli sample, so already 0/1)
        any_attack = (action[:, la.button_indices] > 0.5).any(dim=1).float()

        reward = any_attack * la.reward_per_step * ramp
        # Penalize being locked on but not pressing any attack button.
        if la.idle_penalty_per_step != 0.0 and any_attack.item() < 0.5:
            reward = reward + la.idle_penalty_per_step * ramp
        return reward

    def _movement_reward(self, action: torch.Tensor) -> torch.Tensor:
        """Reward stick-driven movement to fight the 'stand still and lock on' hack."""
        device = action.device
        reward = torch.zeros(1, device=device)

        for mr in self.config.movement_rewards:
            base_idx = 0 if mr.stick == "left" else 2
            idx = base_idx + mr.axis
            value = action[:, idx].item()

            if mr.direction == "positive":
                fires = value > mr.threshold
                overshoot = value - mr.threshold
            elif mr.direction == "negative":
                fires = value < -mr.threshold
                overshoot = -mr.threshold - value
            else:  # magnitude
                fires = abs(value) > mr.threshold
                overshoot = abs(value) - mr.threshold

            if fires:
                r = mr.reward_per_step
                if mr.scale_by_magnitude:
                    r *= overshoot / (1.0 - mr.threshold + 1e-6)
                reward += r

        return reward

    def _stick_penalty(self, action: torch.Tensor) -> torch.Tensor:
        """Penalize stick axis values beyond a threshold magnitude."""
        device = action.device
        penalty = torch.zeros(1, device=device)

        for sp in self.config.stick_penalties:
            # Map stick name + axis to action index
            base_idx = 0 if sp.stick == "left" else 2
            idx = base_idx + sp.axis
            value = action[:, idx].item()

            direction = getattr(sp, "direction", "magnitude")
            if direction == "positive":
                fired = value > sp.threshold
                excess = value - sp.threshold
            elif direction == "negative":
                fired = value < -sp.threshold
                excess = -sp.threshold - value
            else:  # magnitude
                fired = abs(value) > sp.threshold
                excess = abs(value) - sp.threshold

            if fired:
                p = sp.penalty_per_step
                if sp.scale_by_magnitude:
                    overshoot = excess / (1.0 - sp.threshold + 1e-6)
                    p *= overshoot
                penalty += p

        return penalty

    def _png_similarity_penalty(self, current_latent: torch.Tensor) -> torch.Tensor:
        """Penalize when current frame is too similar to any bad-state reference."""
        cfg = self.config.png_similarity_penalty
        device = current_latent.device
        cur_flat = current_latent.view(1, -1).float()
        penalty = 0.0
        for i, ref_lat in enumerate(self._penalty_ref_latents):
            ref_flat = ref_lat.view(1, -1).float().to(device)
            sim = F.cosine_similarity(cur_flat, ref_flat, dim=1).item()
            if sim >= cfg.cosine_threshold:
                self._penalty_streaks[i] += 1
            else:
                self._penalty_streaks[i] = 0
            if self._penalty_streaks[i] >= cfg.min_consecutive_hits:
                penalty += cfg.penalty_per_step
        return torch.tensor([penalty], device=device)

    @staticmethod
    def _frame_distance(a: torch.Tensor, b: torch.Tensor, metric: str) -> float:
        """Compute distance/similarity between two latent frames.

        Args:
            a, b: (1, 4, 16, 32) latent tensors
            metric: "l2" or "cosine"

        Returns:
            For "cosine": similarity in [-1, 1] (higher = more similar)
            For "l2": distance >= 0 (lower = more similar)
        """
        a_flat = a.view(1, -1).float()
        b_flat = b.view(1, -1).float()
        if metric == "cosine":
            return F.cosine_similarity(a_flat, b_flat, dim=1).item()
        else:
            return (a_flat - b_flat).norm(dim=1).item()
