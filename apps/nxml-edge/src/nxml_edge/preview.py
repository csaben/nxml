from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Protocol

from nxml_capture.backends.ffmpeg_v4l2 import capture_preview_jpeg, inspect_jpeg


class PreviewSource(Protocol):
    def frames(self) -> Iterator[bytes]: ...

    def status(self) -> dict[str, object]: ...


class CapturePreview:
    """Bounded read-only preview which releases the V4L2 device after every frame."""

    boundary = b"frame"

    def __init__(self, device: str, *, fps: float = 1.0, width: int = 960) -> None:
        self.device = device
        self.period = 1.0 / fps
        self.width = width
        self._status: dict[str, object] = {"ok": False, "error": "waiting for first frame"}
        self._lock = threading.Lock()

    def status(self) -> dict[str, object]:
        with self._lock:
            return dict(self._status)

    def frames(self) -> Iterator[bytes]:
        while True:
            started = time.monotonic()
            try:
                payload = capture_preview_jpeg(self.device, width=self.width)
                sanity = inspect_jpeg(payload)
                with self._lock:
                    self._status = {
                        "ok": sanity.sane,
                        "error": sanity.reason,
                        "capture": "mjpeg 1920x1080@30 via ffmpeg/v4l2",
                        "channel_means": sanity.channel_means,
                        "channel_stds": sanity.channel_stds,
                        "green_ratio": sanity.green_ratio,
                        "edge_variance": sanity.edge_variance,
                    }
                if sanity.sane:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(payload)).encode()
                        + b"\r\n\r\n"
                        + payload
                        + b"\r\n"
                    )
            except (OSError, RuntimeError, TimeoutError) as error:
                with self._lock:
                    self._status = {"ok": False, "error": f"bad capture format: {error}"}
            time.sleep(max(0.0, self.period - (time.monotonic() - started)))


class NxbtStateClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7777", *, timeout: float = 1.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def status(self) -> dict[str, object]:
        try:
            health = self._get("/health")
            state = self._get("/state") if health.get("connected") else None
            return {"reachable": True, "health": health, "state": state}
        except (OSError, ValueError, urllib.error.URLError) as error:
            return {"reachable": False, "error": str(error), "health": {}, "state": None}

    def _get(self, path: str) -> dict[str, object]:
        with urllib.request.urlopen(self.base_url + path, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise ValueError("NXBT returned a non-object response")
        return payload
