"""Video + parquet sidecar episode writer.

This is the canonical capture format for nxml episodes — replaces the
``NpzEpisodeWriter`` for any non-debug recording. Three files per episode::

    {name}.mkv            # ffv1 lossless RGB (or h264 if --codec h264)
    {name}.parquet        # frame_idx, timestamp, action (fixed_size_list<f32,26>)
    {name}.manifest.json  # schema_version, format, action_spec, video block, ...

Why this shape:
  - Watchable in VLC / ffmpeg / mpv with no tooling.
  - h264-compressed gameplay is ~50-100x smaller than uncompressed; ffv1
    lossless is ~3-5x smaller than uncompressed and bit-exact.
  - Parquet is column-store, so dataloaders can read just the action column.
  - mkv (Matroska) is the natural container for ffv1; .mp4 doesn't support
    ffv1 reliably. h264 will use .mp4 if you swap codecs.

The class accepts ``SyncedFrame`` (BGR HWC uint8 per the synchronizer
contract) and converts BGR→RGB internally before encoding so the on-disk
file plays correctly in any tool.

Frames are streamed straight into the encoder in ``append()``; the parquet
rows are buffered (small) and flushed on ``close()``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Literal

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from nx_packets import ACTION_DIM

from nxml_capture.synchronizer import SyncedFrame

SCHEMA_VERSION = 1
FORMAT_TAG = "video_parquet"
ACTION_SPEC_NAME = "switch_packets.v1"

Codec = Literal["ffv1", "h264"]


@dataclass(frozen=True)
class _CodecProfile:
    codec: str
    container_ext: str
    pixel_format: str
    lossless: bool
    private_options: dict[str, str]
    gop_size: int


_PROFILES: dict[Codec, _CodecProfile] = {
    "ffv1": _CodecProfile(
        codec="ffv1",
        container_ext=".mkv",
        # bgr0 is the only RGB-native ffv1 pix_fmt available in the PyAV-bundled
        # ffmpeg here (no gbrp). It's 4 bytes/pixel but bit-exact for RGB
        # round-trip (no colorspace math, just byte reorder + alpha padding).
        # yuv444p would also be lossless but only "lossless within YUV" — the
        # RGB→YUV→RGB roundtrip introduces matrix rounding.
        pixel_format="bgr0",
        lossless=True,
        # ffv1 is intra-only by design; defaults are fine. Older ffmpeg versions
        # reject `level` as an avoption, so we leave the dict empty.
        private_options={},
        gop_size=1,
    ),
    "h264": _CodecProfile(
        codec="libx264",
        container_ext=".mp4",
        pixel_format="yuv420p",
        lossless=False,
        # CRF 18 is visually near-lossless for naturalistic content.
        private_options={"crf": "18", "preset": "medium"},
        # 16-frame GOP keeps random-window seek cheap inside a chunk.
        gop_size=16,
    ),
}


def _action_array_type() -> pa.DataType:
    return pa.list_(pa.float32(), ACTION_DIM)


_PARQUET_SCHEMA = pa.schema(
    [
        ("frame_idx", pa.int64()),
        ("timestamp", pa.float64()),
        ("action", _action_array_type()),
    ]
)


class VideoParquetEpisodeWriter:
    """Stream-encode frames to ffv1/h264; flush parquet sidecar on close."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        episode_name: str | None = None,
        codec: Codec = "ffv1",
        fps: float = 30.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_name = episode_name or _default_episode_name()
        self._profile = _PROFILES[codec]
        self.codec = codec
        self.fps = fps

        self._video_path = self.output_dir / f"{self.episode_name}{self._profile.container_ext}"
        self._parquet_path = self.output_dir / f"{self.episode_name}.parquet"
        self._manifest_path = self.output_dir / f"{self.episode_name}.manifest.json"

        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._frame_idxs: list[int] = []
        self._timestamps: list[float] = []
        self._actions: list[np.ndarray] = []
        self._first_timestamp: float | None = None
        self._closed = False

    def _ensure_open(self, height: int, width: int) -> None:
        if self._container is not None:
            return
        container = av.open(str(self._video_path), mode="w")
        stream = container.add_stream(self._profile.codec, rate=Fraction(int(self.fps), 1))
        stream.width = width
        stream.height = height
        stream.pix_fmt = self._profile.pixel_format
        stream.time_base = Fraction(1, int(self.fps))
        stream.codec_context.gop_size = self._profile.gop_size
        if self._profile.private_options:
            stream.options = dict(self._profile.private_options)
        self._container = container
        self._stream = stream

    def append(self, synced: SyncedFrame) -> None:
        if self._closed:
            raise RuntimeError("writer already closed")
        if synced.action.shape != (ACTION_DIM,):
            raise ValueError(f"action shape {synced.action.shape} != ({ACTION_DIM},)")

        bgr = synced.frame
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"frame must be (H, W, 3) uint8; got {bgr.shape} {bgr.dtype}")

        h, w = bgr.shape[:2]
        self._ensure_open(h, w)
        assert self._container is not None and self._stream is not None

        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        idx = len(self._frame_idxs)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        # Assign pts/time_base in 1/fps units. Containers (mkv = 1/1000) will
        # rebase as needed; what matters is that the source time_base matches
        # the source pts so PyAV can compute the right packet pts.
        frame.pts = idx
        frame.time_base = Fraction(1, int(self.fps))
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

        if self._first_timestamp is None:
            self._first_timestamp = synced.timestamp
        self._frame_idxs.append(idx)
        self._timestamps.append(synced.timestamp)
        self._actions.append(synced.action.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self._frame_idxs)

    def close(self) -> Path | None:
        if self._closed:
            return None
        self._closed = True
        if self._container is None or self._stream is None or not self._frame_idxs:
            if self._container is not None:
                self._container.close()
            return None

        for packet in self._stream.encode(None):
            self._container.mux(packet)
        self._container.close()
        self._container = None
        self._stream = None

        action_arr = np.stack(self._actions, axis=0)
        action_storage = pa.array(action_arr.reshape(-1), type=pa.float32())
        action_col = pa.FixedSizeListArray.from_arrays(action_storage, ACTION_DIM)
        table = pa.Table.from_arrays(
            [
                pa.array(self._frame_idxs, type=pa.int64()),
                pa.array(self._timestamps, type=pa.float64()),
                action_col,
            ],
            schema=_PARQUET_SCHEMA,
        )
        pq.write_table(table, self._parquet_path, compression="zstd")

        self._manifest_path.write_text(
            json.dumps(self._manifest(action_arr.shape[0]), indent=2)
        )
        return self._video_path

    def _manifest(self, frame_count: int) -> dict[str, object]:
        if frame_count > 1 and self._first_timestamp is not None:
            duration = self._timestamps[-1] - self._timestamps[0]
            fps_est = (frame_count - 1) / duration if duration > 0 else 0.0
        else:
            fps_est = 0.0
        # frame_shape is set lazily once we've seen at least one frame.
        return {
            "schema_version": SCHEMA_VERSION,
            "format": FORMAT_TAG,
            "action_spec": ACTION_SPEC_NAME,
            "action_dim": ACTION_DIM,
            "frame_count": int(frame_count),
            "fps_estimate": float(fps_est),
            "fps_nominal": float(self.fps),
            "video": {
                "codec": self.codec,
                "container": self._profile.container_ext.lstrip("."),
                "pixel_format": self._profile.pixel_format,
                "lossless": self._profile.lossless,
                "gop_size": self._profile.gop_size,
                "options": dict(self._profile.private_options),
            },
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
        }

    def __enter__(self) -> VideoParquetEpisodeWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _default_episode_name() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
