"""Deterministic local DAgger action arbitration; no network or cluster I/O."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

DIM = 26
SPEC = "switch_packets.v1"
L_STICK, R_STICK = 4, 5


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
        max_gaps_per_window: int = 8,
    ):
        self.mode = Mode.HUMAN
        self.mute = MuteMask()
        self.stale_ns = stale_ns
        self.hard_stall_ns = hard_stall_ns
        self.max_gaps_per_window = max_gaps_per_window
        self._takeover = False
        self._neutral_boundary: str | None = None
        self._gap_started_ns: int | None = None
        self._recent_gaps: list[int] = []

    def transition(self, *, mode: Mode | None = None, mute: MuteMask | None = None) -> Applied:
        if mode is not None:
            self.mode = mode
        if mute is not None:
            self.mute = mute
        self._takeover = False
        self._gap_started_ns = None
        return self._neutral(0, "configuration_changed")

    def apply(
        self, now_ns: int, human: Proposal | None, policy: Proposal | None, *, eject=False
    ) -> Applied:
        if eject:
            self.mode = Mode.HUMAN
            self._takeover = False
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
        gesture = (
            self.mode is Mode.HYBRID
            and fresh_h
            and human is not None
            and human.action[L_STICK] > 0.5
            and human.action[R_STICK] > 0.5
        )
        if gesture:
            self._takeover = True
        elif self.mode is Mode.HYBRID and self._takeover and fresh_h:
            self._takeover = False
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
                )
            return self._owned(now_ns, human, "human", 1, takeover=True)

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
            return "recovered", "policy_transient_gap", duration, False
        if self._gap_started_ns is None:
            self._gap_started_ns = (
                policy.monotonic_ns + self.stale_ns
                if policy is not None and policy.monotonic_ns <= now_ns
                else now_ns
            )
            self._recent_gaps = [x for x in self._recent_gaps if now_ns - x <= 10_000_000_000]
            self._recent_gaps.append(now_ns)
        duration = max(0, now_ns - self._gap_started_ns)
        hard = (
            duration >= self.hard_stall_ns
            or len(self._recent_gaps) >= self.max_gaps_per_window
        )
        return (
            "disarmed" if hard else "transient_gap",
            "policy_stall" if hard else "policy_transient_gap",
            duration,
            hard,
        )

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
            policy_monotonic_ns=original_policy.monotonic_ns if original_policy else None,
            policy_observation_monotonic_ns=(
                original_policy.observation_monotonic_ns if original_policy else None
            ),
        )

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
