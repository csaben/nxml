from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Protocol

from nx_packets import ACTION_DIM, Packet, neutral_action, packet_to_action
from pydantic import BaseModel, Field, field_validator


class HumanActionRequest(BaseModel):
    session_id: str
    sequence: int = Field(ge=0)
    client_timestamp_ms: float = Field(ge=0)
    vector: list[float]

    @field_validator("vector")
    @classmethod
    def valid_switch_action(cls, value: list[float]) -> list[float]:
        if len(value) != ACTION_DIM:
            raise ValueError(f"switch_packets.v1 requires {ACTION_DIM} dimensions")
        if any(not math.isfinite(item) or item < -1 or item > 1 for item in value):
            raise ValueError("action values must be finite and within [-1, 1]")
        return value


class HumanControlError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class ActionTransport(Protocol):
    def post_human(self, vector: list[float]) -> dict[str, object]: ...


class NxbtActionClient:
    """Narrow loopback-only client: vector input, hard-coded human provenance."""

    def __init__(self, base_url: str, *, timeout: float = 1.0) -> None:
        self.url = base_url.rstrip("/") + "/action"
        self.timeout = timeout

    def post_human(self, vector: list[float]) -> dict[str, object]:
        payload = json.dumps({"vector": vector, "source": "human"}).encode()
        request = urllib.request.Request(
            self.url,
            method="POST",
            headers={"Content-Type": "application/json"},
            data=payload,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except (OSError, ValueError, urllib.error.URLError) as error:
            raise RuntimeError(f"NXBT action failed: {error}") from error
        if not isinstance(result, dict) or result.get("applied") is not True:
            raise RuntimeError(f"NXBT rejected human action: {result}")
        receipt: dict[str, object] = {"status": 200, "applied": True}
        if any(abs(value) > 1e-6 for value in vector):
            try:
                with urllib.request.urlopen(
                    self.url.removesuffix("/action") + "/state", timeout=self.timeout
                ) as response:
                    state = json.loads(response.read())
                applied = packet_to_action(Packet.model_validate(state)).tolist()
                receipt["applied_summary"] = _vector_summary(applied)
            except (OSError, ValueError, urllib.error.URLError) as error:
                receipt["state_error"] = str(error)
        return receipt


class HumanControlBridge:
    def __init__(
        self,
        transport: ActionTransport,
        *,
        stale_after: float = 0.25,
        max_hz: float = 45.0,
        clock=time.monotonic,
        start_watchdog: bool = True,
    ) -> None:
        self.transport = transport
        self.stale_after = stale_after
        self.min_interval = 1.0 / max_hz
        self.clock = clock
        self._lock = threading.RLock()
        self._enabled = False
        self._session_id: str | None = None
        self._last_sequence = -1
        self._last_seen: float | None = None
        self._last_applied: float | None = None
        self._last_client_timestamp_ms: float | None = None
        self._last_error: str | None = None
        self._neutral_reason: str | None = None
        self._enable_count = 0
        self._action_requests = 0
        self._actions_accepted = 0
        self._actions_rejected = 0
        self._neutral_posts = 0
        self._last_vector_summary: dict[str, object] = _vector_summary(neutral_action().tolist())
        self._last_nxbt_receipt: dict[str, object] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if start_watchdog:
            self._thread = threading.Thread(target=self._watchdog, daemon=True)
            self._thread.start()

    def enable(self) -> dict[str, object]:
        with self._lock:
            if self._enabled:
                self._neutral_locked("superseded")
            self._post_neutral_locked()
            self._enabled = True
            self._enable_count += 1
            self._session_id = str(uuid.uuid4())
            self._last_sequence = -1
            self._last_seen = self.clock()
            self._last_applied = None
            self._last_error = None
            self._neutral_reason = "enabled"
            return self.status()

    def apply(self, request: HumanActionRequest) -> dict[str, object]:
        with self._lock:
            now = self.clock()
            self._action_requests += 1
            if not self._enabled or request.session_id != self._session_id:
                self._actions_rejected += 1
                raise HumanControlError(409, "human control session is not enabled")
            if request.sequence <= self._last_sequence:
                self._actions_rejected += 1
                raise HumanControlError(409, "action sequence is stale or duplicated")
            if self._last_applied is not None and now - self._last_applied < self.min_interval:
                self._actions_rejected += 1
                raise HumanControlError(429, "human action rate exceeds server limit")
            self._last_seen = now
            self._last_sequence = request.sequence
            self._last_client_timestamp_ms = request.client_timestamp_ms
            self._last_vector_summary = _vector_summary(request.vector)
            try:
                self._last_nxbt_receipt = self.transport.post_human(request.vector)
            except RuntimeError as error:
                self._actions_rejected += 1
                self._last_error = str(error)
                self._neutral_locked("server_error")
                raise HumanControlError(502, str(error)) from error
            self._last_applied = now
            self._actions_accepted += 1
            self._neutral_reason = None
            return self.status()

    def disable(self, reason: str = "disabled") -> dict[str, object]:
        with self._lock:
            if self._enabled:
                self._neutral_locked(reason)
            return self.status()

    def expire_if_stale(self, *, now: float | None = None) -> bool:
        with self._lock:
            current = self.clock() if now is None else now
            if (
                self._enabled
                and self._last_seen is not None
                and current - self._last_seen > self.stale_after
            ):
                self._neutral_locked("stale_heartbeat")
                return True
            return False

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "session_id": self._session_id,
                "last_sequence": self._last_sequence,
                "last_client_timestamp_ms": self._last_client_timestamp_ms,
                "last_error": self._last_error,
                "neutral_reason": self._neutral_reason,
                "stale_after_ms": int(self.stale_after * 1000),
                "action_spec": "switch_packets.v1",
                "source": "human",
                "enable_count": self._enable_count,
                "action_requests": self._action_requests,
                "actions_accepted": self._actions_accepted,
                "actions_rejected": self._actions_rejected,
                "neutral_posts": self._neutral_posts,
                "last_vector_summary": self._last_vector_summary,
                "last_nxbt_receipt": self._last_nxbt_receipt,
            }

    def close(self) -> None:
        self.disable("server_shutdown")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _neutral_locked(self, reason: str) -> None:
        try:
            self._post_neutral_locked()
        except RuntimeError as error:
            self._last_error = str(error)
        self._enabled = False
        self._neutral_reason = reason

    def _post_neutral_locked(self) -> None:
        self._neutral_posts += 1
        self._last_nxbt_receipt = self.transport.post_human(neutral_action().tolist())

    def _watchdog(self) -> None:
        interval = min(0.05, self.stale_after / 2)
        while not self._stop.wait(interval):
            self.expire_if_stale()


def _vector_summary(vector: list[float]) -> dict[str, object]:
    non_neutral = [
        {"index": index, "value": round(float(value), 3)}
        for index, value in enumerate(vector)
        if abs(value) > 1e-6
    ]
    return {"non_neutral_count": len(non_neutral), "non_neutral": non_neutral}
