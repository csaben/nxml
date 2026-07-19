from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Protocol

import cv2


class PreviewSource(Protocol):
    def frames(self) -> Iterator[bytes]: ...


class CapturePreview:
    """Bounded read-only preview which releases the V4L2 device after every frame."""

    boundary = b"frame"

    def __init__(self, device: str, *, fps: float = 2.0, width: int = 640) -> None:
        self.device = device
        self.period = 1.0 / fps
        self.width = width

    def frames(self) -> Iterator[bytes]:
        while True:
            started = time.monotonic()
            capture = cv2.VideoCapture(self.device)
            try:
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    ok, frame = capture.read()
                    if ok:
                        encoded, jpeg = cv2.imencode(
                            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72]
                        )
                        if encoded:
                            payload = jpeg.tobytes()
                            yield (
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(payload)).encode()
                                + b"\r\n\r\n"
                                + payload
                                + b"\r\n"
                            )
            finally:
                capture.release()
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
