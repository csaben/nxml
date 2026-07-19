from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest
from nxml_capture.backends.mjpeg_fanout import (
    CaptureFrameLossError,
    MjpegFanoutSource,
    _jpeg_frames,
)


def jpeg(value: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((8, 12, 3), value, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class Stdout:
    def __init__(self, chunks): self.chunks = iter(chunks)
    def read(self, _size): return next(self.chunks, b"")


def test_parser_handles_arbitrary_pipe_boundaries():
    one, two = jpeg(1), jpeg(2)
    process = type("P", (), {"stdout": Stdout([one[:7], one[7:] + two[:3], two[3:]])})()
    assert list(_jpeg_frames(process)) == [one, two]


def test_preview_may_skip_while_recording_receives_same_timestamped_source():
    source = MjpegFanoutSource("/dev/null", recording_queue_size=3)
    source._recording_active = True
    source._publish(jpeg(10), timestamp=12.5, monotonic_ns=99)
    recorded = source._recording.get_nowait()
    preview = source.latest_mjpeg()
    assert preview is not None and preview.timestamp == 12.5 and preview.monotonic_ns == 99
    assert recorded.timestamp == preview.timestamp
    assert recorded.jpeg == preview.jpeg


def test_preview_path_does_not_decode_native_mjpeg(monkeypatch):
    source = MjpegFanoutSource("/dev/null")
    monkeypatch.setattr(cv2, "imdecode", lambda *_args: pytest.fail("preview decoded JPEG"))
    source._publish(jpeg(10), timestamp=12.5, monotonic_ns=99)
    assert source.latest_mjpeg().image is None


def test_recording_overflow_is_latched_and_never_silent():
    source = MjpegFanoutSource("/dev/null", recording_queue_size=1)
    iterator = source.frames()
    consumed = []
    thread = threading.Thread(target=lambda: consumed.append(next(iterator)))
    thread.start()
    while not source._recording_active:
        time.sleep(0.001)
    source._publish(jpeg(1), timestamp=1, monotonic_ns=1)
    thread.join(timeout=1)
    source._publish(jpeg(2), timestamp=2, monotonic_ns=2)
    source._publish(jpeg(3), timestamp=3, monotonic_ns=3)
    with pytest.raises(CaptureFrameLossError, match="overflow"):
        next(iterator)
    iterator.close()
