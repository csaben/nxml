"""V4L2-backed :class:`CaptureSource` using OpenCV's ``cv2.VideoCapture``.

The capture loop runs on a background thread so callers can poll the latest
frame at their own cadence without coupling consumer rate to camera FPS.
``frames()`` is a blocking generator for "consume every frame" workloads
(e.g. recording); :meth:`latest` is the right choice for "best effort"
consumers (e.g. inference clients).
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from queue import Empty, Full, Queue

import cv2
import numpy as np

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
        self._width = width
        self._height = height
        self._queue: Queue[Frame] = Queue(maxsize=queue_size)
        self._latest: Frame | None = None
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_open = False

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
        if self._thread is not None:
            self._thread.join(timeout=5.0)
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
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self._is_open = False
            return
        if self._width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._is_open = True
        try:
            while not self._stop_event.is_set():
                ok, image = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                frame = Frame(timestamp=time.time(), image=np.ascontiguousarray(image))
                with self._latest_lock:
                    self._latest = frame
                try:
                    self._queue.put_nowait(frame)
                except Full:
                    # Drop oldest to keep up with producer.
                    with contextlib.suppress(Empty):
                        self._queue.get_nowait()
                    with contextlib.suppress(Full):
                        self._queue.put_nowait(frame)
        finally:
            cap.release()
            self._is_open = False
