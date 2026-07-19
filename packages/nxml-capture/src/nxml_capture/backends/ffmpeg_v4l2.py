from __future__ import annotations

import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 30
CAPTURE_INPUT_FORMAT = "mjpeg"


def v4l2_input_args(device: str) -> list[str]:
    """Authoritative Hagibis/Switch capture negotiation shared by edge and recorder."""
    return [
        "-f",
        "v4l2",
        "-input_format",
        CAPTURE_INPUT_FORMAT,
        "-video_size",
        f"{CAPTURE_WIDTH}x{CAPTURE_HEIGHT}",
        "-framerate",
        str(CAPTURE_FPS),
        "-i",
        device,
    ]


def capture_preview_jpeg(device: str, *, width: int = 960, timeout: float = 5.0) -> bytes:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        *v4l2_input_args(device),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-2",
        "-q:v",
        "5",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"capture timed out after {timeout}s") from error
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"capture unavailable: {detail or result.returncode}")
    return result.stdout


@dataclass(frozen=True, slots=True)
class FrameSanity:
    sane: bool
    reason: str | None
    channel_means: tuple[float, float, float]
    channel_stds: tuple[float, float, float]
    green_ratio: float
    edge_variance: float


def inspect_jpeg(payload: bytes) -> FrameSanity:
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return FrameSanity(False, "jpeg_decode_failed", (0, 0, 0), (0, 0, 0), 0, 0)
    means_array = frame.mean(axis=(0, 1))
    stds_array = frame.std(axis=(0, 1))
    means = tuple(float(value) for value in means_array)
    stds = tuple(float(value) for value in stds_array)
    green_ratio = means[1] / max(1.0, (means[0] + means[2]) / 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if max(stds) < 1.0:
        return FrameSanity(False, "near_uniform_frame", means, stds, green_ratio, edge_variance)
    if green_ratio > 2.5 and means[1] > 40:
        return FrameSanity(False, "green_corrupt_frame", means, stds, green_ratio, edge_variance)
    return FrameSanity(True, None, means, stds, green_ratio, edge_variance)
