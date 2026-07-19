"""Deterministic local DAgger action arbitration; no network or cluster I/O."""

from __future__ import annotations

from dataclasses import dataclass
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


class Arbitrator:
    def __init__(self, *, stale_ns: int = 250_000_000):
        self.mode = Mode.HUMAN
        self.mute = MuteMask()
        self.stale_ns = stale_ns
        self._takeover = False
        self._neutral_boundary: str | None = None

    def transition(self, *, mode: Mode | None = None, mute: MuteMask | None = None) -> Applied:
        if mode is not None:
            self.mode = mode
        if mute is not None:
            self.mute = mute
        self._takeover = False
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
        if not fresh_p:
            return self._neutral(now_ns, "stale_policy", disarmed=True)
        assert policy is not None
        original_policy = policy
        muted = policy.action.copy()
        muted[np.asarray(self.mute.values)] = 0
        policy = Proposal(muted, policy.monotonic_ns, policy.revision)
        if self.mode is Mode.PURE_AI:
            return self._owned(
                now_ns, policy, "policy", 2, human=human, original_policy=original_policy
            )
        gesture = (
            fresh_h
            and human is not None
            and human.action[L_STICK] > 0.5
            and human.action[R_STICK] > 0.5
        )
        if gesture:
            self._takeover = True
        elif self._takeover:
            self._takeover = False
            self._neutral_boundary = None
            return self._neutral(now_ns, "takeover_released")
        return (
            self._owned(
                now_ns,
                human,
                "human",
                1,
                policy=policy,
                original_policy=original_policy,
                takeover=True,
            )
            if self._takeover and fresh_h
            else self._owned(
                now_ns, policy, "policy", 2, human=human, original_policy=original_policy
            )
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

    def _neutral(self, now, reason, disarmed=False):
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
        )
