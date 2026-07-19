"""V4L2-backed :class:`CaptureSource` using explicit FFmpeg negotiation.

The capture loop runs on a background thread so callers can poll the latest
frame at their own cadence without coupling consumer rate to camera FPS.
``frames()`` is a blocking generator for "consume every frame" workloads
(e.g. recording); :meth:`latest` is the right choice for "best effort"
consumers (e.g. inference clients).
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from collections.abc import Iterator
from queue import Empty, Full, Queue

import numpy as np

from nxml_capture.backends.ffmpeg_v4l2 import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    v4l2_input_args,
)
from nxml_capture.source import Frame


class V4L2Source:
    def __init__(
        self,
        camera_id: int = 0,
        *,
        queue_size: int = 4,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.camera_id = camera_id
        self._width = width or CAPTURE_WIDTH
        self._height = height or CAPTURE_HEIGHT
        self._queue: Queue[Frame] = Queue(maxsize=queue_size)
        self._latest: Frame | None = None
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_open = False
        self._process: subprocess.Popen[bytes] | None = None
        self._process_lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._is_open

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"v4l2-capture-{self.camera_id}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._process_lock:
            process = self._process
        if process is not None:
            process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive() and process is not None:
                process.kill()
                self._thread.join(timeout=1.0)
            self._thread = None

    def latest(self) -> Frame | None:
        with self._latest_lock:
            return self._latest

    def frames(self) -> Iterator[Frame]:
        while not self._stop_event.is_set():
            try:
                yield self._queue.get(timeout=0.5)
            except Empty:
                continue

    def _capture_loop(self) -> None:
        device = f"/dev/video{self.camera_id}"
        frame_bytes = self._width * self._height * 3
        while not self._stop_event.is_set():
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                *v4l2_input_args(device),
                "-vf",
                f"scale={self._width}:{self._height}",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=frame_bytes * 2,
            )
            with self._process_lock:
                self._process = process
            self._is_open = True
            try:
                while not self._stop_event.is_set():
                    payload = _read_exact(process, frame_bytes)
                    if payload is None:
                        break
                    image = np.frombuffer(payload, dtype=np.uint8).reshape(
                        self._height, self._width, 3
                    )
                    frame = Frame(
                        timestamp=time.time(),
                        image=np.ascontiguousarray(image),
                        monotonic_ns=time.monotonic_ns(),
                    )
                    with self._latest_lock:
                        self._latest = frame
                    try:
                        self._queue.put_nowait(frame)
                    except Full:
                        with contextlib.suppress(Empty):
                            self._queue.get_nowait()
                        with contextlib.suppress(Full):
                            self._queue.put_nowait(frame)
            finally:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=1.0)
                if process.poll() is None:
                    process.kill()
                with self._process_lock:
                    self._process = None
                self._is_open = False
            if not self._stop_event.is_set():
                time.sleep(0.25)


def _read_exact(process: subprocess.Popen[bytes], size: int) -> bytes | None:
    assert process.stdout is not None
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = process.stdout.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
