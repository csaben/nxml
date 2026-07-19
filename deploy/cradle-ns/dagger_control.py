"""Deterministic local DAgger action arbitration; no network or cluster I/O."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

DIM = 26
SPEC = "switch_packets.v1"
L_STICK, R_STICK = 4, 5
STICK_DEADZONE = 0.15


class Mode(StrEnum):
    HUMAN = "human"
    PURE_AI = "pure_ai"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class MuteMask:
    values: tuple[bool, ...] = (False,) * DIM
    version: str = "switch_packets.v1/mute.v1"

    def __post_init__(self):
        if len(self.values) != DIM:
            raise ValueError("mute mask must have 26 dimensions")


@dataclass(frozen=True)
class Proposal:
    action: np.ndarray
    monotonic_ns: int
    revision: str | None = None
    observation_monotonic_ns: int | None = None
    sequence: int | None = None


@dataclass(frozen=True)
class Applied:
    action: np.ndarray
    human: np.ndarray
    policy: np.ndarray
    muted_policy: np.ndarray
    ownership: np.ndarray
    source: str
    revision: str | None
    mute: MuteMask
    monotonic_ns: int
    disarmed: bool = False
    boundary: str | None = None
    takeover: bool = False
    takeover_reason: str | None = None
    takeover_release_remaining_ns: int = 0
    policy_monotonic_ns: int | None = None
    policy_observation_monotonic_ns: int | None = None
    gap_state: str = "none"
    gap_reason: str | None = None
    gap_duration_ns: int = 0
    valid: bool = True


class Arbitrator:
    def __init__(
        self,
        *,
        stale_ns: int = 55_000_000,
        hard_stall_ns: int = 250_000_000,
        max_gaps_per_window: int = 16,
        takeover_release_grace_ns: int = 200_000_000,
    ):
        self.mode = Mode.HUMAN
        self.mute = MuteMask()
        self.stale_ns = stale_ns
        self.hard_stall_ns = hard_stall_ns
        self.max_gaps_per_window = max_gaps_per_window
        self.takeover_release_grace_ns = takeover_release_grace_ns
        self._takeover = False
        self._takeover_reason: str | None = None
        self._last_human_activity_ns: int | None = None
        self._neutral_boundary: str | None = None
        self._gap_started_ns: int | None = None
        self._recent_gaps: list[int] = []
        self.gap_count = 0
        self.gap_recoveries = 0
        self.gap_neutral_ticks = 0
        self.gap_hard_disarms = 0
        self.last_gap_duration_ns = 0
        self.max_gap_duration_ns = 0
        self._gap_hard_reported = False

    def transition(self, *, mode: Mode | None = None, mute: MuteMask | None = None) -> Applied:
        if mode is not None:
            self.mode = mode
        if mute is not None:
            self.mute = mute
        self._takeover = False
        self._takeover_reason = None
        self._last_human_activity_ns = None
        self._gap_started_ns = None
        return self._neutral(0, "configuration_changed")

    def apply(
        self, now_ns: int, human: Proposal | None, policy: Proposal | None, *, eject=False
    ) -> Applied:
        if eject:
            self.mode = Mode.HUMAN
            self._takeover = False
            self._takeover_reason = None
            self._last_human_activity_ns = None
            return self._neutral(now_ns, "emergency_eject", disarmed=True)
        if self._neutral_boundary:
            reason, self._neutral_boundary = self._neutral_boundary, None
            return self._neutral(now_ns, reason)
        fresh_h = human is not None and 0 <= now_ns - human.monotonic_ns <= self.stale_ns
        fresh_p = policy is not None and 0 <= now_ns - policy.monotonic_ns <= self.stale_ns
        if self.mode is Mode.HUMAN:
            return (
                self._owned(now_ns, human, "human", 1)
                if fresh_h
                else self._neutral(now_ns, "stale_human")
            )
        activity_reason = self._human_activity_reason(human) if fresh_h else None
        if self.mode is Mode.HYBRID and activity_reason is not None:
            if not self._takeover:
                self._takeover_reason = activity_reason
            self._takeover = True
            self._last_human_activity_ns = now_ns
        elif self.mode is Mode.HYBRID and self._takeover and fresh_h:
            assert self._last_human_activity_ns is not None
            quiet_ns = now_ns - self._last_human_activity_ns
            if quiet_ns >= self.takeover_release_grace_ns:
                self._takeover = False
                self._takeover_reason = None
                self._last_human_activity_ns = None
                self._neutral_boundary = None
                return self._neutral(now_ns, "takeover_released")

        # Explicit full-packet human takeover is independent of policy health.
        # Eject was handled above and remains the highest-priority transition.
        if self.mode is Mode.HYBRID and self._takeover and fresh_h:
            assert human is not None
            if fresh_p:
                assert policy is not None
                original_policy = policy
                muted = policy.action.copy()
                muted[np.asarray(self.mute.values)] = 0
                policy = Proposal(
                    muted,
                    policy.monotonic_ns,
                    policy.revision,
                    policy.observation_monotonic_ns,
                    policy.sequence,
                )
                return self._owned(
                    now_ns,
                    human,
                    "human",
                    1,
                    policy=policy,
                    original_policy=original_policy,
                    takeover=True,
                    takeover_reason=self._takeover_reason,
                    takeover_release_remaining_ns=self._takeover_remaining(now_ns),
                )
            return self._owned(
                now_ns,
                human,
                "human",
                1,
                takeover=True,
                takeover_reason=self._takeover_reason,
                takeover_release_remaining_ns=self._takeover_remaining(now_ns),
            )

        if not fresh_p:
            state, reason, duration, hard = self.observe_policy(now_ns, policy)
            return self._neutral(
                now_ns,
                reason,
                disarmed=hard,
                gap_state=state,
                gap_duration_ns=duration,
            )
        assert policy is not None
        state, _reason, gap_duration, _hard = self.observe_policy(now_ns, policy)
        recovered = state == "recovered"
        original_policy = policy
        muted = policy.action.copy()
        muted[np.asarray(self.mute.values)] = 0
        policy = Proposal(
            muted,
            policy.monotonic_ns,
            policy.revision,
            policy.observation_monotonic_ns,
            policy.sequence,
        )
        result = self._owned(
            now_ns, policy, "policy", 2, human=human, original_policy=original_policy
        )
        if recovered:
            return replace(
                result,
                boundary="policy_gap_recovered",
                gap_state="recovered",
                gap_reason="policy_transient_gap",
                gap_duration_ns=gap_duration,
            )
        return result

    def observe_policy(
        self, now_ns: int, policy: Proposal | None
    ) -> tuple[str, str | None, int, bool]:
        """Track proposal advancement independently of which source owns the packet."""
        fresh = policy is not None and 0 <= now_ns - policy.monotonic_ns <= self.stale_ns
        if fresh:
            if self._gap_started_ns is None:
                return "none", None, 0, False
            duration = max(0, now_ns - self._gap_started_ns)
            self._gap_started_ns = None
            self.gap_recoveries += 1
            self.last_gap_duration_ns = duration
            self.max_gap_duration_ns = max(self.max_gap_duration_ns, duration)
            self._gap_hard_reported = False
            return "recovered", "policy_transient_gap", duration, False
        if self._gap_started_ns is None:
            self._gap_started_ns = (
                policy.monotonic_ns + self.stale_ns
                if policy is not None and policy.monotonic_ns <= now_ns
                else now_ns
            )
            self._recent_gaps = [x for x in self._recent_gaps if now_ns - x <= 10_000_000_000]
            self._recent_gaps.append(now_ns)
            self.gap_count += 1
            self._gap_hard_reported = False
        duration = max(0, now_ns - self._gap_started_ns)
        self.gap_neutral_ticks += 1
        hard = (
            duration >= self.hard_stall_ns
            or len(self._recent_gaps) >= self.max_gaps_per_window
        )
        if hard and not self._gap_hard_reported:
            self.gap_hard_disarms += 1
            self._gap_hard_reported = True
        return (
            "disarmed" if hard else "transient_gap",
            "policy_stall" if hard else "policy_transient_gap",
            duration,
            hard,
        )

    def gap_status(self) -> dict[str, int | str | None]:
        return {
            "gap_count": self.gap_count,
            "gap_recoveries": self.gap_recoveries,
            "gap_neutral_ticks": self.gap_neutral_ticks,
            "gap_hard_disarms": self.gap_hard_disarms,
            "last_gap_duration_ns": self.last_gap_duration_ns,
            "max_gap_duration_ns": self.max_gap_duration_ns,
            "gap_density_count": len(self._recent_gaps),
            "gap_density_window_ns": 10_000_000_000,
            "gap_density_limit": self.max_gaps_per_window,
        }

    def _owned(
        self,
        now,
        proposal,
        source,
        owner,
        *,
        human=None,
        policy=None,
        original_policy=None,
        takeover=False,
        takeover_reason=None,
        takeover_release_remaining_ns=0,
    ):
        assert proposal is not None
        if source == "human":
            human = proposal
        else:
            policy = proposal
        z = np.zeros(DIM, np.float32)
        original_policy = original_policy or policy
        return Applied(
            proposal.action.copy(),
            human.action.copy() if human else z.copy(),
            original_policy.action.copy() if original_policy else z.copy(),
            policy.action.copy() if policy else z.copy(),
            np.full(DIM, owner, np.uint8),
            source,
            policy.revision if policy else None,
            self.mute,
            now,
            takeover=takeover,
            takeover_reason=takeover_reason,
            takeover_release_remaining_ns=takeover_release_remaining_ns,
            policy_monotonic_ns=original_policy.monotonic_ns if original_policy else None,
            policy_observation_monotonic_ns=(
                original_policy.observation_monotonic_ns if original_policy else None
            ),
        )

    @staticmethod
    def _human_activity_reason(human: Proposal | None) -> str | None:
        if human is None:
            return None
        action = human.action
        if np.any(np.abs(action[:4]) > STICK_DEADZONE):
            return "stick_motion"
        pressed = np.flatnonzero(action[4:] > 0.5)
        if pressed.size:
            dimension = int(pressed[0] + 4)
            return "trigger_press" if dimension in {11, 13} else "button_press"
        return None

    def _takeover_remaining(self, now_ns: int) -> int:
        if self._last_human_activity_ns is None:
            return 0
        quiet_ns = max(0, now_ns - self._last_human_activity_ns)
        return max(0, self.takeover_release_grace_ns - quiet_ns)

    def _neutral(self, now, reason, disarmed=False, gap_state="none", gap_duration_ns=0):
        z = np.zeros(DIM, np.float32)
        return Applied(
            z,
            z.copy(),
            z.copy(),
            z.copy(),
            np.zeros(DIM, np.uint8),
            "none",
            None,
            self.mute,
            now,
            disarmed,
            reason,
            gap_state=gap_state,
            gap_reason=reason if gap_state != "none" else None,
            gap_duration_ns=gap_duration_ns,
            valid=gap_state == "none",
        )
