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

import hashlib
import json
import platform
import time
import uuid
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

SCHEMA_VERSION = 2
SCHEMA_ID = "nxml.episode.v2"
ACTION_SCHEMA_ID = "nxml.dagger-actions.v2"
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
        ("frame_monotonic_ns", pa.int64()),
        ("frame_timestamp_ns", pa.int64()),
        ("action_timestamp", pa.float64()),
        ("action_monotonic_ns", pa.int64()),
        ("action_timestamp_ns", pa.int64()),
        ("action_age_ns", pa.int64()),
        ("action_age", pa.float64()),
        ("valid", pa.bool_()),
        ("invalid_reasons", pa.list_(pa.string())),
        # Kept as an alias so schema-v1 readers continue to work.
        ("action", _action_array_type()),
        ("applied_action", _action_array_type()),
        ("human_action", _action_array_type()),
        ("human_mask", pa.list_(pa.bool_(), ACTION_DIM)),
        ("policy_action", _action_array_type()),
        ("ownership", pa.list_(pa.uint8(), ACTION_DIM)),
        ("controller_id", pa.string()),
        ("active_driver", pa.string()),
        ("controller", pa.string()),
        ("policy_id", pa.string()),
        ("policy_revision", pa.string()),
        ("policy_digest", pa.string()),
        ("human_monotonic_ns", pa.int64()),
        ("policy_monotonic_ns", pa.int64()),
        ("policy_observation_monotonic_ns", pa.int64()),
        ("muted_policy_action", _action_array_type()),
        ("mute_mask", pa.list_(pa.bool_(), ACTION_DIM)),
        ("mute_mask_version", pa.string()),
        ("ownership_source", pa.string()),
        ("mode", pa.string()),
        ("takeover", pa.bool_()),
        ("takeover_reason", pa.string()),
        ("takeover_release_remaining_ns", pa.int64()),
        ("proposal_valid", pa.bool_()),
        ("proposal_fresh", pa.bool_()),
        ("proposal_sequence", pa.int64()),
        ("proposal_age_ns", pa.int64()),
        ("gap_state", pa.string()),
        ("gap_reason", pa.string()),
        ("gap_duration_ns", pa.int64()),
        ("bc_training_eligible", pa.bool_()),
    ]
)

_EVENTS_SCHEMA = pa.schema(
    [
        ("event_idx", pa.int64()),
        ("timestamp", pa.float64()),
        ("monotonic_ns", pa.int64()),
        ("kind", pa.string()),
        ("source", pa.string()),
        ("payload_json", pa.string()),
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
        game: str | None = None,
        config: dict[str, object] | None = None,
        build: dict[str, object] | None = None,
        lineage: dict[str, object] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_name = episode_name or _default_episode_name()
        self._profile = _PROFILES[codec]
        self.codec = codec
        self.fps = fps
        self.episode_id = str(uuid.uuid4())
        self.game = game
        self.config = dict(config or {})
        self.build = dict(build or {})
        self.lineage = dict(lineage or {})
        self._clock_monotonic_origin_ns = time.monotonic_ns()
        self._clock_utc_origin = datetime.now(tz=UTC).isoformat()

        self._video_path = self.output_dir / f"{self.episode_name}{self._profile.container_ext}"
        self._parquet_path = self.output_dir / f"{self.episode_name}.parquet"
        self._manifest_path = self.output_dir / f"{self.episode_name}.manifest.json"
        self._events_path = self.output_dir / f"{self.episode_name}.events.parquet"

        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._frame_idxs: list[int] = []
        self._timestamps: list[float] = []
        self._actions: list[np.ndarray] = []
        self._records: list[SyncedFrame] = []
        self._events: list[dict[str, object]] = []
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
        for name in ("action", "human_action", "policy_action", "muted_policy_action"):
            value = getattr(synced, name)
            if value is not None and (value.shape != (ACTION_DIM,) or not np.isfinite(value).all()):
                raise ValueError(f"{name} must be finite switch_packets.v1/{ACTION_DIM}-D")
        for name in ("human_mask", "mute_mask", "ownership"):
            value = getattr(synced, name)
            if value is not None and value.shape != (ACTION_DIM,):
                raise ValueError(f"{name} shape {value.shape} != ({ACTION_DIM},)")
        if synced.ownership is not None and not np.isin(synced.ownership, (0, 1, 2)).all():
            raise ValueError("ownership values must be 0=unowned, 1=human, or 2=policy")
        if synced.valid:
            if synced.frame_monotonic_ns is None or synced.action_monotonic_ns is None:
                raise ValueError("valid DAgger rows require frame/action monotonic timestamps")
            if synced.action_monotonic_ns > synced.frame_monotonic_ns:
                raise ValueError("valid DAgger action timestamp cannot follow its frame")
            expected_age = (synced.frame_monotonic_ns - synced.action_monotonic_ns) / 1e9
            if abs(synced.action_age - expected_age) > 1e-9:
                raise ValueError("action_age must exactly match causal monotonic timestamps")
        elif not synced.invalid_reasons:
            raise ValueError("invalid DAgger rows require a sparse invalid reason")

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
        self._records.append(synced)

    def append_event(
        self,
        kind: str,
        *,
        timestamp: float,
        monotonic_ns: int | None = None,
        source: str = "edge",
        payload: dict[str, object] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("writer already closed")
        self._events.append(
            {
                "event_idx": len(self._events),
                "timestamp": timestamp,
                "monotonic_ns": monotonic_ns,
                "kind": kind,
                "source": source,
                "payload_json": json.dumps(payload or {}, sort_keys=True),
            }
        )

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
        zero_action = np.zeros(ACTION_DIM, dtype=np.float32)
        zero_mask = np.zeros(ACTION_DIM, dtype=bool)
        zero_owner = np.zeros(ACTION_DIM, dtype=np.uint8)
        rows = []
        for idx, synced in enumerate(self._records):
            rows.append(
                {
                    "frame_idx": idx,
                    "timestamp": synced.timestamp,
                    "frame_monotonic_ns": synced.frame_monotonic_ns,
                    "frame_timestamp_ns": synced.frame_monotonic_ns,
                    "action_timestamp": synced.action_timestamp,
                    "action_monotonic_ns": synced.action_monotonic_ns,
                    "action_timestamp_ns": synced.action_monotonic_ns,
                    "action_age_ns": (
                        synced.frame_monotonic_ns - synced.action_monotonic_ns
                        if synced.frame_monotonic_ns is not None
                        and synced.action_monotonic_ns is not None
                        else None
                    ),
                    "action_age": synced.action_age,
                    "valid": synced.valid,
                    "invalid_reasons": list(synced.invalid_reasons),
                    "action": synced.action.tolist(),
                    "applied_action": synced.applied_action.tolist(),
                    "human_action": (
                        synced.human_action if synced.human_action is not None else zero_action
                    ).tolist(),
                    "human_mask": (
                        synced.human_mask if synced.human_mask is not None else zero_mask
                    ).tolist(),
                    "policy_action": (
                        synced.policy_action if synced.policy_action is not None else zero_action
                    ).tolist(),
                    "ownership": (
                        synced.ownership if synced.ownership is not None else zero_owner
                    ).tolist(),
                    "controller_id": synced.controller_id,
                    "active_driver": synced.active_driver,
                    "controller": synced.ownership_source or synced.active_driver,
                    "policy_id": synced.policy_id,
                    "policy_revision": synced.policy_revision,
                    "policy_digest": synced.policy_digest,
                    "human_monotonic_ns": synced.human_monotonic_ns,
                    "policy_monotonic_ns": synced.policy_monotonic_ns,
                    "policy_observation_monotonic_ns": synced.policy_observation_monotonic_ns,
                    "muted_policy_action": (
                        synced.muted_policy_action
                        if synced.muted_policy_action is not None
                        else zero_action
                    ).tolist(),
                    "mute_mask": (
                        synced.mute_mask if synced.mute_mask is not None else zero_mask
                    ).tolist(),
                    "mute_mask_version": synced.mute_mask_version,
                    "ownership_source": synced.ownership_source,
                    "mode": synced.mode,
                    "takeover": synced.takeover,
                    "takeover_reason": synced.takeover_reason,
                    "takeover_release_remaining_ns": synced.takeover_release_remaining_ns,
                    "proposal_valid": synced.proposal_valid,
                    "proposal_fresh": synced.proposal_fresh,
                    "proposal_sequence": synced.proposal_sequence,
                    "proposal_age_ns": synced.proposal_age_ns,
                    "gap_state": synced.gap_state,
                    "gap_reason": synced.gap_reason,
                    "gap_duration_ns": synced.gap_duration_ns,
                    "bc_training_eligible": bool(
                        synced.valid
                        and np.isfinite(synced.applied_action).all()
                        and synced.ownership is not None
                        and np.all(synced.ownership == 1)
                    ),
                }
            )
        table = pa.Table.from_pylist(rows, schema=_PARQUET_SCHEMA)
        pq.write_table(table, self._parquet_path, compression="zstd")

        events = pa.Table.from_pylist(self._events, schema=_EVENTS_SCHEMA)
        pq.write_table(events, self._events_path, compression="zstd")

        self._manifest_path.write_text(
            json.dumps(self._manifest(action_arr.shape[0]), indent=2, sort_keys=True)
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
            "schema_id": SCHEMA_ID,
            "episode_id": self.episode_id,
            "format": FORMAT_TAG,
            "action_spec": ACTION_SPEC_NAME,
            "action_spec_id": ACTION_SPEC_NAME,
            "action_schema_id": ACTION_SCHEMA_ID,
            "action_schema_version": 2,
            "action_rows_schema_id": ACTION_SCHEMA_ID,
            "action_dim": ACTION_DIM,
            "frame_count": int(frame_count),
            "fps_estimate": float(fps_est),
            "fps_nominal": float(self.fps),
            "game": self.game,
            "config": self.config,
            "build": {
                "hostname": platform.node(),
                **self.build,
            },
            "clock_mapping": {
                "clock_id": "linux-monotonic",
                "monotonic_origin_ns": self._clock_monotonic_origin_ns,
                "utc_origin": self._clock_utc_origin,
                "uncertainty_ns": 0,
            },
            "first_frame_timestamp_ns": min(
                record.frame_monotonic_ns
                for record in self._records
                if record.frame_monotonic_ns is not None
            )
            if any(record.frame_monotonic_ns is not None for record in self._records)
            else 0,
            "last_frame_timestamp_ns": max(
                record.frame_monotonic_ns
                for record in self._records
                if record.frame_monotonic_ns is not None
            )
            if any(record.frame_monotonic_ns is not None for record in self._records)
            else 0,
            "capture": {
                "timestamps": "frame arrival",
                "frame_color": "bgr24",
            },
            "lineage": self.lineage,
            "video": {
                "codec": self.codec,
                "container": self._profile.container_ext.lstrip("."),
                "pixel_format": self._profile.pixel_format,
                "lossless": self._profile.lossless,
                "gop_size": self._profile.gop_size,
                "options": dict(self._profile.private_options),
            },
            "files": {
                self._video_path.name: _file_metadata(self._video_path),
                self._parquet_path.name: _file_metadata(self._parquet_path),
                self._events_path.name: _file_metadata(self._events_path),
            },
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
        }

    def __enter__(self) -> VideoParquetEpisodeWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _default_episode_name() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")


def _file_metadata(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
