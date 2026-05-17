"""Reward + termination configuration dataclasses for Pokémon ZA.

Only the dataclasses referenced by ``RewardShaper`` land here — PPO /
curriculum / seed dataclasses live in ``nxrl/configs`` since they're
game-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StickTarget:
    values: list[float] = field(default_factory=list)
    thresholds: list[float] = field(default_factory=list)


@dataclass
class ProgressVectorConfig:
    target_direction: list[float] = field(default_factory=lambda: [0.0, 0.0])
    stick: str = "left"
    reward_per_step: float = 0.5
    min_magnitude: float = 0.05


@dataclass
class StagnationPenaltyConfig:
    metric: str = "l2"
    threshold: float = 5.0
    penalty_per_step: float = -0.1
    grace_period: int = 10


@dataclass
class GoalProximityConfig:
    metric: str = "cosine"
    threshold: float = 0.85
    reward: float = 2.0
    continuous: bool = True
    continuous_scale: float = 0.5


@dataclass
class StartDivergenceConfig:
    metric: str = "cosine"
    threshold: float = 0.95
    penalty: float = -0.3
    step_scaling: bool = True


@dataclass
class MovementRewardConfig:
    stick: str = "left"
    axis: int = 1
    direction: str = "positive"
    threshold: float = 0.1
    reward_per_step: float = 0.3
    scale_by_magnitude: bool = True


@dataclass
class TargetUIRewardConfig:
    template_path: str = "reward-signals/bottom-right-ui.PNG"
    score_threshold: float = 0.15
    sat_threshold: float = 68.0
    min_consecutive_hits: int = 5
    reward_per_step: float = 8.0
    streak_ramp_scale: float = 1.0


@dataclass
class LockedAttackRewardConfig:
    button_indices: list[int] = field(default_factory=lambda: [22, 23, 24, 25])
    reward_per_step: float = 0.6
    require_min_streak: int = 1
    idle_penalty_per_step: float = 0.0


@dataclass
class EscapeRewardConfig:
    grace_period: int = 30
    escape_distance: float = 0.4
    bonus: float = 10.0
    require_initial_stagnation: bool = True
    initial_stagnation_window: int = 30
    initial_stagnation_threshold: float = 0.15
    sustain_reward_per_step: float = 0.05


@dataclass
class SlidingStagnationConfig:
    window: int = 15
    cos_threshold: float = 0.408
    min_consecutive_hits: int = 16
    reward_per_step: float = -0.1
    grace_period: int = 0


@dataclass
class UnstuckRewardConfig:
    window: int = 30
    cos_threshold: float = 0.25
    min_consecutive_hits: int = 5
    reward_per_step: float = 0.2
    grace_period: int = 0


@dataclass
class UniqueAcquisitionRewardConfig:
    bonus_per_target: float = 3.0
    chain_multiplier: float = 1.0
    min_latent_move: float = 0.20
    min_unlocked_frames: int = 15
    require_attack_press: bool = False
    attack_indices: list[int] = field(default_factory=lambda: [22, 23, 24, 25])


@dataclass
class StuckTurnRewardConfig:
    forward_axis: int = 1
    forward_threshold: float = 0.3
    turn_axis: int = 0
    turn_threshold: float = 0.4
    stuck_threshold_frames: int = 60
    min_turn_frames: int = 10
    reward_per_event: float = 2.0
    trigger: str = "forward"
    stagnation_window: int = 30
    stagnation_cos_threshold: float = 0.02
    pending_sweep_timeout: int = 45


@dataclass
class StickPenaltyConfig:
    stick: str = "right"
    axis: int = 1
    direction: str = "magnitude"
    threshold: float = 0.5
    penalty_per_step: float = -1.0
    scale_by_magnitude: bool = True


@dataclass
class PNGSimilarityPenaltyConfig:
    paths: list[str] = field(default_factory=list)
    cosine_threshold: float = 0.85
    min_consecutive_hits: int = 10
    penalty_per_step: float = -0.5


@dataclass
class VerifiedLockTerminalConfig:
    """Two-gate terminal condition harder to exploit than target_ui-streak alone:

      1) detector.streak >= min_detector_streak — popup visible N frames
      2) >= min_attack_presses attack-button presses (in the action vector
         sampled by the policy) DURING frames where the popup was visible,
         counted over the last attack_window steps — agent is actively
         engaging the visible target

    Both must fire simultaneously. Gate 2 is purely action-based + gated on
    detector visibility, so the WM cannot fake a terminal by hallucinating
    a popup alone — the agent must also have been pressing attack while the
    popup was up.

    On terminal, ``terminal_bonus`` is added to the step's reward.
    Compatible with ``unique_acquisition`` being enabled — overrides the
    streak-alone terminal when configured."""

    enabled: bool = True
    min_detector_streak: int = 5

    # Attack-engagement gate
    attack_window: int = 10
    min_attack_presses: int = 3
    button_indices: list[int] = field(default_factory=lambda: [22, 23, 24, 25])

    terminal_bonus: float = 20.0


@dataclass
class RewardConfig:
    """Top-level RewardShaper config. All shaping components are optional;
    leave a field at ``None`` / empty list to disable that component."""

    reward_type: str = "escape-stuck-position"
    start_frame_idx: int = 0
    goal_frame_idx: int = -1

    rewardable_action_state: dict[str, StickTarget] = field(default_factory=dict)
    reward_values: dict[str, float] = field(default_factory=dict)

    progress_vector: ProgressVectorConfig | None = None
    stagnation_penalty: StagnationPenaltyConfig | None = None
    goal_proximity: GoalProximityConfig | None = None
    start_divergence: StartDivergenceConfig | None = None
    stick_penalties: list[StickPenaltyConfig] = field(default_factory=list)
    movement_rewards: list[MovementRewardConfig] = field(default_factory=list)
    target_ui_detection: TargetUIRewardConfig | None = None
    locked_attack: LockedAttackRewardConfig | None = None
    stuck_turn: StuckTurnRewardConfig | None = None
    unique_acquisition: UniqueAcquisitionRewardConfig | None = None
    escape: EscapeRewardConfig | None = None
    sliding_stagnation: SlidingStagnationConfig | None = None
    unstuck: UnstuckRewardConfig | None = None
    png_similarity_penalty: PNGSimilarityPenaltyConfig | None = None
    verified_lock_terminal: VerifiedLockTerminalConfig | None = None
