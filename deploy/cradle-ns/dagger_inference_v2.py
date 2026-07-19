"""Fail-closed edge client for immutable policy inference ZMQ v2."""

from __future__ import annotations

import contextlib
import json
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from dagger_control import DIM, Proposal
from dagger_models import check_compatibility

MODE_INFO = 0x04
MODE_FRAME_V2 = 0x11
MODE_RESET_V2 = 0x12
MAX_MESSAGE_BYTES = 2 * 1024 * 1024
INFERENCE_WIDTH = 256
INFERENCE_HEIGHT = 128
INFERENCE_JPEG_QUALITY = 85


class InferenceV2Error(RuntimeError):
    pass


def prepare_inference_jpeg(jpeg: bytes) -> bytes:
    """Downscale off-path to the server's canonical VAE input resolution."""
    import cv2

    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InferenceV2Error("capture JPEG could not be decoded")
    resized = cv2.resize(image, (INFERENCE_WIDTH, INFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, INFERENCE_JPEG_QUALITY])
    if not ok:
        raise InferenceV2Error("inference JPEG could not be encoded")
    return encoded.tobytes()


@dataclass(frozen=True)
class RemoteResult:
    action: np.ndarray | None
    state: str
    frame_timestamp_ns: int
    received_monotonic_ns: int
    transport_latency_ns: int
    processing_latency_ns: int | None
    proposal_timestamp_ns: int | None


class InferenceV2Client:
    """Single-thread/single-flight ZMQ REQ client with reconnect epochs."""

    def __init__(
        self,
        endpoint: str,
        revision: dict[str, Any],
        *,
        timeout_ms: int = 100,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        socket_factory: Callable[[], Any] | None = None,
    ) -> None:
        if revision.get("state") not in {"validated", "active"}:
            raise ValueError("inference v2 requires a validated or active selected revision")
        check_compatibility(revision)
        if timeout_ms < 1 or timeout_ms > 250:
            raise ValueError("inference timeout must be between 1 and 250 ms")
        if not endpoint.startswith("tcp://"):
            raise ValueError("inference endpoint must use tcp://")
        self.endpoint = endpoint
        self.revision = dict(revision)
        self.timeout_ms = timeout_ms
        self.clock_ns = clock_ns
        self.socket_factory = socket_factory
        self._socket = None
        self._context = None
        self._owner_thread: int | None = None
        self.info: dict[str, Any] | None = None

    def _new_socket(self):
        if self.socket_factory is not None:
            return self.socket_factory()
        import zmq

        self._context = zmq.Context()
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.MAXMSGSIZE, MAX_MESSAGE_BYTES)
        socket.connect(self.endpoint)
        return socket

    def connect(self) -> dict[str, Any]:
        self.close()
        self._owner_thread = threading.get_ident()
        self._socket = self._new_socket()
        try:
            info = self._decode(self._request(bytes([MODE_INFO])))
            self._validate_info(info)
            reset = self._decode(self._request(bytes([MODE_RESET_V2])))
            self._validate_info(reset)
            if reset.get("reset") is not True:
                raise InferenceV2Error("inference RESET was not acknowledged")
        except Exception:
            self.invalidate()
            raise
        self.info = info
        return info

    def predict_frame(self, frame_timestamp_ns: int, jpeg: bytes) -> RemoteResult:
        if self._socket is None:
            self.connect()
        if len(jpeg) + 9 > MAX_MESSAGE_BYTES:
            raise InferenceV2Error("frame exceeds inference v2 message limit")
        started = self.clock_ns()
        try:
            response = self._decode(
                self._request(bytes([MODE_FRAME_V2]) + struct.pack("<Q", frame_timestamp_ns) + jpeg)
            )
        except Exception:
            self.invalidate()
            raise
        received = self.clock_ns()
        transport = received - started
        if transport < 0 or transport > self.timeout_ms * 1_000_000:
            self.invalidate()
            raise InferenceV2Error("inference request exceeded edge timeout")
        state = response.get("state")
        if state not in {"warming", "proposal", "error"}:
            self.invalidate()
            raise InferenceV2Error("invalid inference proposal state")
        if state == "error":
            self.invalidate()
            raise InferenceV2Error(str(response.get("error") or "remote inference error"))
        if response.get("schema_id") != "nxml.policy-proposal.v2":
            self.invalidate()
            raise InferenceV2Error("invalid inference proposal schema")
        self._validate_identity(response)
        if response.get("frame_timestamp_ns") != frame_timestamp_ns:
            self.invalidate()
            raise InferenceV2Error("proposal frame timestamp mismatch")
        action = np.asarray(response.get("action"), dtype=np.float32)
        if action.shape != (DIM,) or not np.isfinite(action).all():
            self.invalidate()
            raise InferenceV2Error("remote proposal is not finite switch_packets.v1/26-D")
        if state == "warming" and np.any(action != 0):
            self.invalidate()
            raise InferenceV2Error("warming response was not neutral")
        processing = response.get("processing_latency_ns")
        proposal_ns = response.get("proposal_timestamp_ns")
        if not isinstance(processing, int) or processing < 0:
            self.invalidate()
            raise InferenceV2Error("invalid server processing latency")
        return RemoteResult(
            action if state == "proposal" else None,
            state,
            frame_timestamp_ns,
            received,
            transport,
            processing,
            proposal_ns if isinstance(proposal_ns, int) else None,
        )

    def _request(self, payload: bytes) -> bytes:
        if threading.get_ident() != self._owner_thread or self._socket is None:
            raise InferenceV2Error("inference socket used outside its single owner thread")
        try:
            self._socket.send(payload)
            return self._socket.recv()
        except Exception as error:
            raise InferenceV2Error(f"inference transport failure: {error}") from error

    @staticmethod
    def _decode(payload: bytes) -> dict[str, Any]:
        try:
            value = json.loads(payload)
        except (TypeError, ValueError) as error:
            raise InferenceV2Error("inference response is not JSON") from error
        if not isinstance(value, dict):
            raise InferenceV2Error("inference response root is not an object")
        return value

    def _validate_identity(self, value: dict[str, Any]) -> None:
        expected = self.revision
        if (
            value.get("revision_id") != expected["revision_id"]
            or value.get("checkpoint_sha256") != expected["checkpoint_sha256"]
            or value.get("action_spec_id") != "switch_packets.v1"
        ):
            raise InferenceV2Error("inference identity differs from selected revision")

    def _validate_info(self, info: dict[str, Any]) -> None:
        self._validate_identity(info)
        compatibility = self.revision["compatibility"]
        expected = {
            "schema_id": "nxml.policy-inference-info.v2",
            "ready": True,
            "action_dim": DIM,
            "sequence_length": compatibility["sequence_length"],
            "latent_shape": compatibility["latent_shape"],
            "max_message_bytes": MAX_MESSAGE_BYTES,
        }
        if any(info.get(key) != value for key, value in expected.items()):
            raise InferenceV2Error("inference INFO compatibility/readiness mismatch")

    def invalidate(self) -> None:
        self.close()

    def close(self) -> None:
        socket, context = self._socket, self._context
        self._socket = None
        self._context = None
        self._owner_thread = None
        self.info = None
        if socket is not None:
            socket.close()
        if context is not None:
            context.term()


@dataclass(frozen=True)
class RemoteInferenceStatus:
    schema_version: str = "nxml.dagger-remote-inference-status.v1"
    enabled: bool = False
    ready: bool = False
    armed: bool = False
    health: str = "disabled"
    revision: str | None = None
    checkpoint_sha256: str | None = None
    sequence_length: int | None = None
    warmup_frames: int = 0
    frames_sent: int = 0
    proposals: int = 0
    warming: int = 0
    reconnects: int = 0
    observation_age_ms: float | None = None
    proposal_age_ms: float | None = None
    proposal_sequence: int = 0
    freshness_state: str = "unavailable"
    fresh_budget_ms: float = 33.0
    hold_horizon_ms: float = 55.0
    transport_latency_ms: float | None = None
    processing_latency_ms: float | None = None
    error: str | None = None
    gap_started_monotonic_ns: int | None = None
    gap_duration_ms: float = 0.0
    gap_reason: str | None = None
    transient_gaps: int = 0

    def wire(self) -> dict[str, Any]:
        return asdict(self)


class RemoteInferenceWorker:
    """Latest-frame async transport; never queues requests or blocks producers."""

    def __init__(
        self,
        *,
        source,
        client: InferenceV2Client,
        stale_ns: int = 100_000_000,
        fresh_ns: int | None = None,
        reconnect_seconds: float = 0.25,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        on_disarm: Callable[[str], None] | None = None,
    ) -> None:
        self.source = source
        self.client = client
        self.stale_ns = stale_ns
        fresh_ns = min(33_000_000, stale_ns - 1) if fresh_ns is None else fresh_ns
        if fresh_ns >= stale_ns:
            raise ValueError("fresh budget must be smaller than hold horizon")
        self.fresh_ns = fresh_ns
        self.reconnect_seconds = reconnect_seconds
        self.clock_ns = clock_ns
        self.on_disarm = on_disarm or (lambda _reason: None)
        self._lock = threading.Lock()
        self._enabled = threading.Event()
        self._disconnect = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proposal: Proposal | None = None
        self._status = RemoteInferenceStatus(
            revision=client.revision["revision_id"],
            checkpoint_sha256=client.revision["checkpoint_sha256"],
            sequence_length=client.revision["compatibility"]["sequence_length"],
            fresh_budget_ms=self.fresh_ns / 1e6,
            hold_horizon_ms=self.stale_ns / 1e6,
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="dagger-inference-v2")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._enabled.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self.client.close()

    def enable(self) -> None:
        """Explicitly allow proposals; readiness still happens on the worker thread."""
        self._enabled.set()

    def disable(self, reason: str = "disabled") -> None:
        self._enabled.clear()
        self._disconnect.set()
        self._clear(reason, health="disabled")

    def latest_proposal(self) -> Proposal | None:
        if not self._enabled.is_set():
            return None
        now = self.clock_ns()
        with self._lock:
            proposal = self._proposal
        if proposal is None or not 0 <= now - proposal.monotonic_ns <= self.stale_ns:
            return None
        return proposal

    def proposal_for_arbitration(self) -> Proposal | None:
        """Return the last observed proposal so the arbitrator can measure a gap causally."""
        if not self._enabled.is_set():
            return None
        with self._lock:
            return self._proposal

    def wait_until_fresh(self, timeout: float = 2.0) -> dict[str, Any] | None:
        """Require a new proposal, unless the cached one is under one 60 Hz tick old."""
        deadline = time.monotonic() + timeout
        initial = self.status()
        initial_sequence = initial["proposal_sequence"]
        while time.monotonic() < deadline:
            status = self.status()
            proposal = self.latest_proposal()
            age_ns = self.clock_ns() - proposal.monotonic_ns if proposal is not None else None
            recent_cached = age_ns is not None and 0 <= age_ns <= 16_667_000
            sequence_advanced = status["proposal_sequence"] > initial_sequence
            if (
                status["ready"]
                and status["health"] == "healthy"
                and status["warmup_frames"] >= status["sequence_length"] - 1
                and proposal is not None
                and (recent_cached or sequence_advanced)
            ):
                return status
            time.sleep(0.005)
        return None

    def status(self) -> dict[str, Any]:
        now = self.clock_ns()
        with self._lock:
            value = self._status.wire()
            proposal = self._proposal
        if proposal is not None:
            age_ns = now - proposal.monotonic_ns
            value["proposal_age_ms"] = max(0, age_ns) / 1e6
            if age_ns > self.stale_ns:
                value["health"] = "degraded"
                value["ready"] = False
                value["freshness_state"] = "transient_gap"
                value["gap_reason"] = value["gap_reason"] or "proposal_hold_exceeded"
                started = value["gap_started_monotonic_ns"] or proposal.monotonic_ns + self.stale_ns
                value["gap_started_monotonic_ns"] = started
                value["gap_duration_ms"] = max(0, now - started) / 1e6
            elif age_ns > self.fresh_ns:
                value["freshness_state"] = "cadence_hold"
            else:
                value["freshness_state"] = "fresh"
        return value

    def _mark_gap(self, reason: str) -> None:
        """Expose a cadence gap without converting it into a transport failure."""
        now = self.clock_ns()
        with self._lock:
            old = self._status
            started = old.gap_started_monotonic_ns or now
            self._status = replace(
                old,
                health="healthy",
                ready=True,
                freshness_state="transient_gap",
                gap_started_monotonic_ns=started,
                gap_duration_ms=max(0, now - started) / 1e6,
                gap_reason=reason,
                transient_gaps=old.transient_gaps + int(old.gap_started_monotonic_ns is None),
                error=None,
            )

    def _clear(self, reason: str, *, health: str) -> None:
        # Safety notification must never be able to terminate the transport
        # owner thread. The action plane independently remains fail-closed.
        with contextlib.suppress(Exception):
            self.on_disarm(reason)
        with self._lock:
            old = self._status
            self._proposal = None
            self._status = RemoteInferenceStatus(
                enabled=self._enabled.is_set(),
                health=health,
                revision=self.client.revision["revision_id"],
                checkpoint_sha256=self.client.revision["checkpoint_sha256"],
                sequence_length=self.client.revision["compatibility"]["sequence_length"],
                warmup_frames=old.warmup_frames,
                frames_sent=old.frames_sent,
                proposals=old.proposals,
                warming=old.warming,
                reconnects=old.reconnects,
                proposal_sequence=old.proposal_sequence,
                freshness_state="stale_stall" if health == "stale" else "unavailable",
                fresh_budget_ms=self.fresh_ns / 1e6,
                hold_horizon_ms=self.stale_ns / 1e6,
                error=reason,
            )

    def _run(self) -> None:
        sequence = -1
        while not self._stop.is_set():
            if self._disconnect.is_set():
                self.client.invalidate()
                self._disconnect.clear()
            if not self._enabled.wait(0.05) or self._stop.is_set():
                continue
            try:
                if self.client.info is None:
                    self.client.connect()
                    with self._lock:
                        old = self._status
                        self._status = RemoteInferenceStatus(
                            enabled=True,
                            ready=True,
                            health="warming",
                            revision=self.client.revision["revision_id"],
                            checkpoint_sha256=self.client.revision["checkpoint_sha256"],
                            sequence_length=self.client.revision["compatibility"][
                                "sequence_length"
                            ],
                            frames_sent=old.frames_sent,
                            proposals=old.proposals,
                            warming=old.warming,
                            reconnects=old.reconnects + 1,
                            proposal_sequence=old.proposal_sequence,
                            fresh_budget_ms=self.fresh_ns / 1e6,
                            hold_horizon_ms=self.stale_ns / 1e6,
                        )
                frame = self.source.latest_mjpeg(after_sequence=sequence, timeout=0.1)
                if frame is None:
                    continue
                sequence = frame.sequence
                now = self.clock_ns()
                observation_age = now - frame.monotonic_ns
                if observation_age < 0 or observation_age > self.stale_ns:
                    self._mark_gap("stale observation")
                    continue
                inference_jpeg = prepare_inference_jpeg(frame.jpeg)
                result = self.client.predict_frame(frame.monotonic_ns, inference_jpeg)
                if result.received_monotonic_ns - frame.monotonic_ns > self.stale_ns:
                    if result.action is None:
                        with self._lock:
                            old = self._status
                            warming = old.warming + 1
                            self._status = RemoteInferenceStatus(
                                enabled=True,
                                ready=True,
                                health="warming",
                                revision=self.client.revision["revision_id"],
                                checkpoint_sha256=self.client.revision["checkpoint_sha256"],
                                sequence_length=self.client.revision["compatibility"][
                                    "sequence_length"
                                ],
                                warmup_frames=min(
                                    warming,
                                    self.client.revision["compatibility"]["sequence_length"] - 1,
                                ),
                                frames_sent=old.frames_sent + 1,
                                proposals=old.proposals,
                                warming=warming,
                                reconnects=old.reconnects,
                                proposal_sequence=old.proposal_sequence,
                                freshness_state="warming",
                                fresh_budget_ms=self.fresh_ns / 1e6,
                                hold_horizon_ms=self.stale_ns / 1e6,
                                observation_age_ms=observation_age / 1e6,
                                transport_latency_ms=result.transport_latency_ns / 1e6,
                                processing_latency_ms=(result.processing_latency_ns or 0) / 1e6,
                            )
                    self._mark_gap("stale remote proposal")
                    continue
                with self._lock:
                    old = self._status
                    proposal_count = old.proposals
                    warming = old.warming
                    self._proposal = None
                    if result.action is None:
                        warming += 1
                    else:
                        self._proposal = Proposal(
                            result.action.copy(),
                            result.received_monotonic_ns,
                            self.client.revision["revision_id"],
                            observation_monotonic_ns=frame.monotonic_ns,
                            sequence=proposal_count + 1,
                        )
                        proposal_count += 1
                    self._status = RemoteInferenceStatus(
                        enabled=True,
                        ready=True,
                        health="healthy" if result.action is not None else "warming",
                        revision=self.client.revision["revision_id"],
                        checkpoint_sha256=self.client.revision["checkpoint_sha256"],
                        sequence_length=self.client.revision["compatibility"]["sequence_length"],
                        warmup_frames=min(
                            warming,
                            self.client.revision["compatibility"]["sequence_length"] - 1,
                        ),
                        frames_sent=old.frames_sent + 1,
                        proposals=proposal_count,
                        warming=warming,
                        reconnects=old.reconnects,
                        proposal_sequence=proposal_count,
                        freshness_state="fresh" if result.action is not None else "warming",
                        fresh_budget_ms=self.fresh_ns / 1e6,
                        hold_horizon_ms=self.stale_ns / 1e6,
                        observation_age_ms=observation_age / 1e6,
                        transport_latency_ms=result.transport_latency_ns / 1e6,
                        processing_latency_ms=(
                            result.processing_latency_ns / 1e6
                            if result.processing_latency_ns is not None
                            else None
                        ),
                        gap_started_monotonic_ns=None,
                        gap_duration_ms=0.0,
                        gap_reason=None,
                        transient_gaps=old.transient_gaps,
                    )
            except Exception as error:
                self.client.invalidate()
                self._clear(str(error), health="error")
                self._stop.wait(self.reconnect_seconds)
