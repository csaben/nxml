"""Bounded causal history of complete local arbitration decisions."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
from dagger_control import Applied, Mode, Proposal
from nx_packets import ACTION_DIM, neutral_action
from nxml_capture.synchronizer import SyncedFrame


@dataclass(frozen=True)
class ArbitrationRecord:
    applied: Applied
    human: Proposal | None
    policy: Proposal | None
    policy_digest: str | None
    mode: Mode
    takeover: bool
    proposal_valid: bool
    proposal_fresh: bool


class ArbitratorHistory:
    def __init__(self, *, maxlen: int = 4096, max_age_ns: int = 500_000_000):
        self.max_age_ns = max_age_ns
        self._records: deque[ArbitrationRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._first = threading.Event()
        # Single-writer integer handoff: recorder writes acknowledgments and
        # the 60 Hz action owner only reads. No lock or wait enters the fast path.
        self.recording_active = False
        self.boundary_ack_claim_sequence = 0
        self.boundary_ack_sequence = 0

    @property
    def is_connected(self) -> bool:
        return True

    def start(self):
        pass

    def stop(self):
        pass

    def begin_recording(self) -> None:
        self.recording_active = True

    def end_recording(self) -> None:
        self.recording_active = False

    def acknowledge_boundary(self, sequence: int) -> None:
        if sequence > self.boundary_ack_sequence:
            self.boundary_ack_sequence = sequence

    def claim_boundary_ack(self, sequence: int) -> bool:
        """Atomically claim a sequence once; recorder is the sole writer."""
        if sequence <= self.boundary_ack_claim_sequence:
            return False
        self.boundary_ack_claim_sequence = sequence
        return True

    def wait_for_first(self, timeout: float = 5.0) -> bool:
        return self._first.wait(timeout)

    def append(self, record: ArbitrationRecord) -> None:
        with self._lock:
            if (
                self._records
                and record.applied.monotonic_ns <= self._records[-1].applied.monotonic_ns
            ):
                raise ValueError("arbitration timestamps must increase strictly")
            self._records.append(record)
            self._first.set()

    def latest_at_ns(self, frame_ns: int) -> ArbitrationRecord | None:
        with self._lock:
            for record in reversed(self._records):
                if record.applied.monotonic_ns <= frame_ns:
                    return record
        return None

    def pair(self, frame) -> SyncedFrame:
        record = self.latest_at_ns(frame.monotonic_ns)
        if record is None:
            return self._invalid(frame, "no_prior_arbitration_record")
        age_ns = frame.monotonic_ns - record.applied.monotonic_ns
        if age_ns < 0:
            return self._invalid(frame, "future_arbitration_record")
        if age_ns > self.max_age_ns:
            return self._invalid(frame, "stale_arbitration_record")
        applied, human, policy = record.applied, record.human, record.policy
        human_action = human.action.copy() if human is not None else neutral_action()
        policy_action = policy.action.copy() if policy is not None else neutral_action()
        muted = policy_action.copy()
        muted[np.asarray(applied.mute.values)] = 0
        human_present = human is not None and human.monotonic_ns <= frame.monotonic_ns
        return SyncedFrame(
            timestamp=frame.timestamp,
            frame=frame.image,
            action=applied.action.copy(),
            action_age=age_ns / 1e9,
            frame_monotonic_ns=frame.monotonic_ns,
            action_timestamp=None,
            action_monotonic_ns=applied.monotonic_ns,
            human_action=human_action,
            human_mask=np.ones(ACTION_DIM, bool) if human_present else np.zeros(ACTION_DIM, bool),
            policy_action=policy_action,
            ownership=applied.ownership.copy(),
            controller_id="dagger-arbitrator:switch_packets.v1",
            active_driver=applied.source,
            policy_id="bc_transformer_v1" if policy is not None else None,
            policy_revision=policy.revision if policy is not None else None,
            policy_digest=record.policy_digest,
            human_monotonic_ns=human.monotonic_ns if human is not None else None,
            policy_monotonic_ns=applied.policy_monotonic_ns,
            policy_observation_monotonic_ns=applied.policy_observation_monotonic_ns,
            muted_policy_action=muted,
            mute_mask=np.asarray(applied.mute.values, bool),
            mute_mask_version=applied.mute.version,
            ownership_source=applied.source,
            mode=record.mode.value,
            takeover=applied.takeover,
            takeover_reason=applied.takeover_reason,
            takeover_release_remaining_ns=applied.takeover_release_remaining_ns,
            proposal_valid=record.proposal_valid,
            proposal_fresh=record.proposal_fresh,
            proposal_sequence=policy.sequence if policy is not None else None,
            proposal_age_ns=(
                applied.monotonic_ns - policy.monotonic_ns if policy is not None else None
            ),
            gap_state=applied.gap_state,
            gap_reason=applied.gap_reason,
            gap_duration_ns=applied.gap_duration_ns,
            boundary_sequence=applied.boundary_sequence,
            boundary_acknowledged=applied.boundary_acknowledged,
            valid=applied.valid,
            invalid_reasons=(applied.gap_reason,) if not applied.valid and applied.gap_reason else (),
        )

    @staticmethod
    def _invalid(frame, reason):
        zero = neutral_action()
        return SyncedFrame(
            timestamp=frame.timestamp,
            frame=frame.image,
            action=zero.copy(),
            action_age=0,
            frame_monotonic_ns=frame.monotonic_ns,
            human_action=zero.copy(),
            human_mask=np.zeros(ACTION_DIM, bool),
            policy_action=zero.copy(),
            ownership=np.zeros(ACTION_DIM, np.uint8),
            active_driver="none",
            valid=False,
            invalid_reasons=(reason,),
            muted_policy_action=zero.copy(),
            mute_mask=np.zeros(ACTION_DIM, bool),
            proposal_valid=False,
            proposal_fresh=False,
        )


class ArbitrationSynchronizer:
    def __init__(self, source, history: ArbitratorHistory):
        self.source, self.history = source, history
        self.invalid_samples = 0

    def frames(self):
        if not self.history.wait_for_first():
            raise TimeoutError("no arbitration record")
        for frame in self.source.frames():
            synced = self.history.pair(frame)
            self.invalid_samples += int(not synced.valid)
            yield synced
