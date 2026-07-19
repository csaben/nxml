from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Protocol

from nx_packets import ACTION_DIM, neutral_action
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
    def post_human(self, vector: list[float]) -> None: ...


class NxbtActionClient:
    """Narrow loopback-only client: vector input, hard-coded human provenance."""

    def __init__(self, base_url: str, *, timeout: float = 1.0) -> None:
        self.url = base_url.rstrip("/") + "/action"
        self.timeout = timeout

    def post_human(self, vector: list[float]) -> None:
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
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if start_watchdog:
            self._thread = threading.Thread(target=self._watchdog, daemon=True)
            self._thread.start()

    def enable(self) -> dict[str, object]:
        with self._lock:
            if self._enabled:
                self._neutral_locked("superseded")
            self.transport.post_human(neutral_action().tolist())
            self._enabled = True
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
            if not self._enabled or request.session_id != self._session_id:
                raise HumanControlError(409, "human control session is not enabled")
            if request.sequence <= self._last_sequence:
                raise HumanControlError(409, "action sequence is stale or duplicated")
            if self._last_applied is not None and now - self._last_applied < self.min_interval:
                raise HumanControlError(429, "human action rate exceeds server limit")
            self._last_seen = now
            self._last_sequence = request.sequence
            self._last_client_timestamp_ms = request.client_timestamp_ms
            try:
                self.transport.post_human(request.vector)
            except RuntimeError as error:
                self._last_error = str(error)
                self._neutral_locked("server_error")
                raise HumanControlError(502, str(error)) from error
            self._last_applied = now
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
            }

    def close(self) -> None:
        self.disable("server_shutdown")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _neutral_locked(self, reason: str) -> None:
        try:
            self.transport.post_human(neutral_action().tolist())
        except RuntimeError as error:
            self._last_error = str(error)
        self._enabled = False
        self._neutral_reason = reason

    def _watchdog(self) -> None:
        interval = min(0.05, self.stale_after / 2)
        while not self._stop.wait(interval):
            self.expire_if_stale()
