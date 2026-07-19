"""Single-owner native-MJPEG capture fan-out for preview and recording."""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Empty, Full, Queue

import cv2
import numpy as np

from nxml_capture.backends.ffmpeg_v4l2 import v4l2_mjpeg_frames_command
from nxml_capture.source import Frame


class CaptureFrameLossError(RuntimeError):
    """A strict recording consumer could not keep up with source capture."""


@dataclass(frozen=True, slots=True)
class MjpegFrame:
    sequence: int
    timestamp: float
    monotonic_ns: int
    jpeg: bytes
    image: np.ndarray | None = None

    def decoded(self) -> Frame:
        image = self.image
        if image is None:
            image = _decode_jpeg(self.jpeg)
        return Frame(self.timestamp, image, self.monotonic_ns)


class MjpegFanoutSource:
    """Own one FFmpeg/V4L2 process and distribute its native MJPEG frames.

    Preview reads only the latest compressed frame and may drop freely. The
    recording iterator has a bounded queue; overflow is latched and raised to
    the recorder rather than hidden.
    """

    def __init__(self, device: str, *, recording_queue_size: int = 120) -> None:
        self.device = device
        self._recording: Queue[MjpegFrame] = Queue(maxsize=recording_queue_size)
        self._latest: MjpegFrame | None = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._process_lock = threading.Lock()
        self._is_open = False
        self._recording_active = False
        self._recording_loss: tuple[int, int] | None = None
        self._sequence = 0
        self._error: str | None = None

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="mjpeg-fanout")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._process_lock:
            process = self._process
        if process is not None:
            process.terminate()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive() and process is not None:
                process.kill()
                self._thread.join(timeout=1)
            self._thread = None

    def latest(self) -> Frame | None:
        with self._condition:
            item = self._latest
        return item.decoded() if item is not None else None

    def latest_mjpeg(self, *, after_sequence: int = -1, timeout: float = 2.0) -> MjpegFrame | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._stop.is_set():
                if self._latest is not None and self._latest.sequence > after_sequence:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
        return None

    def frames(self) -> Iterator[Frame]:
        """Strict single recording subscription; raises on any queue overflow."""
        with self._condition:
            if self._recording_active:
                raise RuntimeError("a recording consumer is already active")
            self._recording_active = True
            self._recording_loss = None
            while not self._recording.empty():
                with contextlib.suppress(Empty):
                    self._recording.get_nowait()
        expected: int | None = None
        try:
            while not self._stop.is_set():
                if self._recording_loss is not None:
                    first, latest = self._recording_loss
                    raise CaptureFrameLossError(
                        f"recording queue overflow; source frames {first}..{latest} were not recorded"
                    )
                try:
                    item = self._recording.get(timeout=0.5)
                except Empty:
                    continue
                if expected is not None and item.sequence != expected:
                    raise CaptureFrameLossError(
                        f"recording sequence gap: expected {expected}, received {item.sequence}"
                    )
                expected = item.sequence + 1
                yield item.decoded()
        finally:
            with self._condition:
                self._recording_active = False

    def _publish(self, jpeg: bytes, *, timestamp: float, monotonic_ns: int) -> None:
        with self._condition:
            recording_active = self._recording_active
        image = _decode_jpeg(jpeg) if recording_active else None
        item = MjpegFrame(self._sequence, timestamp, monotonic_ns, jpeg, image)
        self._sequence += 1
        with self._condition:
            self._latest = item
            if self._recording_active:
                try:
                    self._recording.put_nowait(item)
                except Full:
                    if self._recording_loss is None:
                        self._recording_loss = (item.sequence, item.sequence)
                    else:
                        self._recording_loss = (self._recording_loss[0], item.sequence)
            self._condition.notify_all()

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            process = subprocess.Popen(
                v4l2_mjpeg_frames_command(self.device),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            with self._process_lock:
                self._process = process
            self._is_open = True
            try:
                for jpeg in _jpeg_frames(process):
                    if self._stop.is_set():
                        break
                    self._publish(jpeg, timestamp=time.time(), monotonic_ns=time.monotonic_ns())
            finally:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=1)
                if process.poll() is None:
                    process.kill()
                if process.returncode not in {0, -15} and process.stderr is not None:
                    self._error = process.stderr.read(4096).decode(errors="replace").strip()
                with self._process_lock:
                    self._process = None
                self._is_open = False
            if not self._stop.is_set():
                time.sleep(0.25)


def _jpeg_frames(process: subprocess.Popen[bytes]) -> Iterator[bytes]:
    """Split concatenated JPEGs without assuming FFmpeg read boundaries."""
    assert process.stdout is not None
    buffer = bytearray()
    while True:
        chunk = process.stdout.read(65536)
        if not chunk:
            return
        buffer.extend(chunk)
        while True:
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                if len(buffer) > 1:
                    del buffer[:-1]
                break
            end = buffer.find(b"\xff\xd9", start + 2)
            if end < 0:
                if start:
                    del buffer[:start]
                break
            end += 2
            yield bytes(buffer[start:end])
            del buffer[:end]


def _decode_jpeg(jpeg: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("capture produced an invalid JPEG frame")
    return image
