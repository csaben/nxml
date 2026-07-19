"""Local fail-closed DAgger action plane and complete arbitration history."""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import replace

import numpy as np
from dagger_control import Arbitrator, Mode, MuteMask, Proposal
from dagger_history import ArbitrationRecord, ArbitratorHistory


class ActionPlane:
    def __init__(self, orchestrator, inference, *, hz: float = 60.0, stale_ns: int = 33_000_000):
        self.orchestrator = orchestrator
        self.inference = inference
        self.period = 1.0 / hz
        self.arbitrator = Arbitrator(stale_ns=stale_ns)
        self.history = ArbitratorHistory(max_age_ns=500_000_000)
        self._lock = threading.RLock()
        self._armed = False
        self._human: Proposal | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_reason: str | None = None
        self._revision = inference.client.revision["revision_id"] if inference else None
        self._digest = inference.client.revision["checkpoint_sha256"] if inference else None

    def start(self):
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="dagger-action-plane"
            )
            self._thread.start()

    def stop(self):
        self.disarm("service_stop")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def submit_human(self, vector: list[float]):
        now = time.monotonic_ns()
        proposal = Proposal(np.asarray(vector, np.float32), now)
        with self._lock:
            self._human = proposal
            if not self._armed:
                self.orchestrator.post_action(vector)
                applied = self.arbitrator.apply(now, proposal, None)
                self._append(applied, proposal, None)

    def arm(self):
        with self._lock:
            if self._armed:
                raise RuntimeError("action plane is already armed")
            boundary = replace(
                self.arbitrator.transition(mode=Mode.HUMAN), monotonic_ns=time.monotonic_ns()
            )
            self.orchestrator.post_action([0.0] * 26)
            self._append(boundary, self._human, None)
            self._armed = True
            self._last_reason = None

    def disarm(self, reason="operator_disarm"):
        now = time.monotonic_ns()
        with self._lock:
            self._armed = False
            boundary = replace(self.arbitrator.transition(mode=Mode.HUMAN), monotonic_ns=now)
            self._last_reason = reason
            human = self._human
            self.orchestrator.post_action([0.0] * 26)
            self._append(boundary, human, None)

    def inference_failure(self, reason: str):
        """Neutral-disarm only if inference owned an armed action plane."""
        with self._lock:
            if not self._armed:
                self._last_reason = reason
                return
        self.disarm(reason)

    def eject(self):
        now = time.monotonic_ns()
        with self._lock:
            human = self._human
            applied = self.arbitrator.apply(now, human, None, eject=True)
            self._armed = False
            self._last_reason = "emergency_eject"
            self.orchestrator.post_action(applied.action.tolist())
            self._append(applied, human, None)

    def set_mode(self, mode: Mode):
        with self._lock:
            if mode is not Mode.HUMAN and not self._armed:
                raise RuntimeError("AI modes require an armed healthy action plane")
            boundary = replace(
                self.arbitrator.transition(mode=mode), monotonic_ns=time.monotonic_ns()
            )
            self.orchestrator.post_action(boundary.action.tolist())
            self._append(
                boundary,
                self._human,
                self.inference.latest_proposal() if self.inference else None,
            )

    def set_mute(self, mute: MuteMask):
        with self._lock:
            boundary = replace(
                self.arbitrator.transition(mute=mute), monotonic_ns=time.monotonic_ns()
            )
            self.orchestrator.post_action(boundary.action.tolist())
            self._append(
                boundary,
                self._human,
                self.inference.latest_proposal() if self.inference else None,
            )

    def status(self):
        with self._lock:
            return {
                "schema_version": "nxml.dagger-action-plane-status.v1",
                "armed": self._armed,
                "mode": self.arbitrator.mode.value,
                "revision": self._revision,
                "checkpoint_sha256": self._digest,
                "mute_mask": list(self.arbitrator.mute.values),
                "mute_mask_version": self.arbitrator.mute.version,
                "last_disarm_reason": self._last_reason,
            }

    def recording_state(self):
        status = self.status()
        return {
            key: status[key]
            for key in (
                "mode",
                "revision",
                "checkpoint_sha256",
                "mute_mask",
                "mute_mask_version",
                "armed",
            )
        }

    def _append(self, applied, human, policy):
        self.history.append(
            ArbitrationRecord(
                applied=applied,
                human=human,
                policy=policy,
                policy_digest=self._digest if policy is not None else None,
                mode=self.arbitrator.mode,
                takeover=applied.takeover,
                proposal_valid=policy is not None and np.isfinite(policy.action).all(),
                proposal_fresh=policy is not None
                and 0 <= applied.monotonic_ns - policy.monotonic_ns <= self.arbitrator.stale_ns,
            )
        )

    def _run(self):
        deadline = time.monotonic()
        while not self._stop.is_set():
            deadline += self.period
            with self._lock:
                try:
                    if self._armed:
                        now = time.monotonic_ns()
                        human = self._human
                        policy = self.inference.latest_proposal() if self.inference else None
                        applied = self.arbitrator.apply(now, human, policy)
                        self.orchestrator.post_action(applied.action.tolist())
                        self._append(applied, human, policy)
                        if applied.disarmed:
                            self.disarm(applied.boundary or "action_plane_failure")
                except Exception as error:
                    self._armed = False
                    self.arbitrator.transition(mode=Mode.HUMAN)
                    self._last_reason = f"action_output_failure: {error}"
                    with contextlib.suppress(Exception):
                        self.orchestrator.post_action([0.0] * 26)
            self._stop.wait(max(0.0, deadline - time.monotonic()))
