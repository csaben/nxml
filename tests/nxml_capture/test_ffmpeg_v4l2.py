from __future__ import annotations

import cv2
import numpy as np
from nxml_capture.backends.ffmpeg_v4l2 import inspect_jpeg, v4l2_input_args


def _jpeg(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def test_capture_contract_is_explicit_1080p_mjpeg_at_30fps() -> None:
    assert v4l2_input_args("/dev/video0") == [
        "-f",
        "v4l2",
        "-input_format",
        "mjpeg",
        "-video_size",
        "1920x1080",
        "-framerate",
        "30",
        "-i",
        "/dev/video0",
    ]


def test_capture_contract_can_select_native_720p60_mjpeg() -> None:
    args = v4l2_input_args("/dev/video0", width=1280, height=720, fps=60)
    assert args[args.index("-video_size") + 1] == "1280x720"
    assert args[args.index("-framerate") + 1] == "60"


def test_uniform_green_corruption_is_rejected() -> None:
    green = np.zeros((180, 320, 3), dtype=np.uint8)
    green[:, :, 1] = 154

    sanity = inspect_jpeg(_jpeg(green))

    assert sanity.sane is False
    assert sanity.reason == "near_uniform_frame"
    assert sanity.green_ratio > 100
    assert sanity.edge_variance == 0


def test_realistic_varied_frame_is_accepted() -> None:
    x = np.linspace(0, 255, 320, dtype=np.uint8)
    gradient = np.tile(x, (180, 1))
    frame = np.stack((gradient, np.flip(gradient, axis=1), gradient // 2), axis=2)
    cv2.rectangle(frame, (30, 30), (180, 120), (255, 255, 255), 4)

    sanity = inspect_jpeg(_jpeg(frame))

    assert sanity.sane is True
    assert sanity.reason is None
    assert max(sanity.channel_stds) > 20
    assert sanity.edge_variance > 1
