"""Local frame-to-action inference isolated from capture and control loops."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from dagger_control import DIM, Proposal


@dataclass(frozen=True)
class InferenceStatus:
    schema_version: str = "nxml.dagger-inference-status.v1"
    health: str = "unloaded"
    armed: bool = False
    revision: str | None = None
    sequence_length: int | None = None
    history_frames: int = 0
    frames_seen: int = 0
    proposals: int = 0
    observation_age_ms: float | None = None
    proposal_age_ms: float | None = None
    inference_latency_ms: float | None = None
    error: str | None = None

    def wire(self) -> dict[str, Any]:
        return asdict(self)


class LocalInferenceWorker:
    """Consumes fanout's latest MJPEG on a private thread and caches proposals.

    The policy handle implements ``predict_frame(jpeg)`` and owns the exact
    checkpoint-defined VAE/history. No caller in the capture or action path
    waits for decode, GPU work, model loading, or network I/O.
    """

    def __init__(
        self,
        *,
        source,
        runtime,
        stale_ns: int = 250_000_000,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        on_disarm: Callable[[str], None] | None = None,
    ) -> None:
        self.source = source
        self.runtime = runtime
        self.stale_ns = stale_ns
        self.clock_ns = clock_ns
        self.on_disarm = on_disarm or runtime.neutral_disarm
        self._lock = threading.Lock()
        self._status = InferenceStatus()
        self._proposal: Proposal | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="dagger-inference")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def status(self) -> dict[str, Any]:
        now = self.clock_ns()
        with self._lock:
            status = self._status
            proposal = self._proposal
        wire = status.wire()
        if proposal is not None:
            wire["proposal_age_ms"] = max(0, now - proposal.monotonic_ns) / 1e6
            if now - proposal.monotonic_ns > self.stale_ns:
                wire["health"], wire["armed"] = "stale", False
        return wire

    def latest_proposal(self) -> Proposal | None:
        now = self.clock_ns()
        with self._lock:
            proposal = self._proposal
        if proposal is None or not 0 <= now - proposal.monotonic_ns <= self.stale_ns:
            return None
        return proposal

    def _fail(self, reason: str, *, health: str = "failed") -> None:
        self.on_disarm(reason)
        with self._lock:
            old = self._status
            self._proposal = None
            self._status = InferenceStatus(
                health=health,
                revision=old.revision,
                sequence_length=old.sequence_length,
                history_frames=old.history_frames,
                frames_seen=old.frames_seen,
                proposals=old.proposals,
                error=reason,
            )

    def _run(self) -> None:
        sequence = -1
        loaded_revision: str | None = None
        last_observation_ns: int | None = None
        while not self._stop.is_set():
            try:
                revision, policy = self.runtime.active_handle()
                if policy is None or revision is None:
                    with self._lock:
                        self._status = InferenceStatus()
                        self._proposal = None
                    self._stop.wait(0.05)
                    continue
                if revision != loaded_revision:
                    if hasattr(policy, "reset_frame_window"):
                        policy.reset_frame_window()
                    loaded_revision = revision
                frame = self.source.latest_mjpeg(after_sequence=sequence, timeout=0.1)
                if frame is None:
                    if (
                        last_observation_ns is not None
                        and self.clock_ns() - last_observation_ns > self.stale_ns
                    ):
                        self._fail("observation stream lag", health="stale")
                        last_observation_ns = None
                    continue
                sequence = frame.sequence
                now = self.clock_ns()
                age = now - frame.monotonic_ns
                if age < 0 or age > self.stale_ns:
                    self._fail("stale observation", health="stale")
                    continue
                last_observation_ns = frame.monotonic_ns
                started = self.clock_ns()
                action = policy.predict_frame(frame.jpeg)
                finished = self.clock_ns()
                seq_len = int(policy.sequence_length)
                with self._lock:
                    old = self._status
                    seen = old.frames_seen + 1
                    history = min(seen, seq_len)
                    proposals = old.proposals
                    if action is not None:
                        action = np.asarray(action, dtype=np.float32)
                        if action.shape != (DIM,) or not np.isfinite(action).all():
                            raise ValueError("policy returned invalid switch_packets.v1 action")
                        self._proposal = Proposal(
                            action.copy(),
                            finished,
                            revision,
                            observation_monotonic_ns=frame.monotonic_ns,
                        )
                        proposals += 1
                    self._status = InferenceStatus(
                        health="healthy" if action is not None else "warming",
                        armed=False,
                        revision=revision,
                        sequence_length=seq_len,
                        history_frames=history,
                        frames_seen=seen,
                        proposals=proposals,
                        observation_age_ms=age / 1e6,
                        inference_latency_ms=(finished - started) / 1e6,
                    )
            except Exception as error:
                self._fail(f"inference error: {error}")
                self._stop.wait(0.05)


def load_bc_frame_policy(path, *, device: str = "cuda", vae_path: str | None = None):
    """Load the locally cached checkpoint; imports GPU dependencies lazily."""
    from nxrl.serve.server import PolicyServer

    policy = PolicyServer(model_path=path, device=device, enable_frame_mode=True, vae_path=vae_path)
    info = policy.info()
    if info.architecture != "bc_transformer_v1" or info.action_dim != DIM:
        raise ValueError("loaded policy is not bc_transformer_v1 switch_packets.v1/26-D")
    return policy
