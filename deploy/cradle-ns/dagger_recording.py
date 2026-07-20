"""Human DAgger recording sessions over the shared capture fan-out."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from dagger_history import ArbitrationSynchronizer, ArbitratorHistory
from dagger_segments import SegmentDeliveryWorker, SegmentSource
from nxml_capture import VideoParquetEpisodeWriter
from nxml_capture.backends.mjpeg_fanout import CaptureFrameLossError, MjpegFanoutSource


@dataclass(frozen=True)
class RecordingStatus:
    schema_version: str = "nxml.dagger-recording-status.v1"
    state: str = "idle"
    episode_id: str | None = None
    episode_name: str | None = None
    started_monotonic_ns: int | None = None
    duration_seconds: float = 0.0
    frames: int = 0
    invalid_actions: int = 0
    error: str | None = None
    finalized_manifest: str | None = None
    rolling: bool = False
    current_segment: int | None = None
    segment_count: int = 0


class HumanRecordingSession:
    def __init__(
        self,
        source: MjpegFanoutSource,
        *,
        output_dir: Path,
        orchestrator_ws: str = "ws://127.0.0.1:7777/ws/state",
        codec: str = "ffv1",
        fps: float = 30.0,
        game: str = "pokemon-za",
        history: ArbitratorHistory | None = None,
        state_provider=None,
        segment_worker: SegmentDeliveryWorker | None = None,
        segment_staging_dir: Path | None = None,
        segment_dataset_id: str = "nxml-pokemon-za-v2",
        segment_duration_seconds: float = 30.0,
        segment_max_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.source = source
        self.output_dir = output_dir
        self.orchestrator_ws = orchestrator_ws
        self.codec = codec
        self.fps = fps
        self.game = game
        self.history = history
        self.state_provider = state_provider or (lambda: {"mode": "human"})
        self.segment_worker = segment_worker
        self.segment_staging_dir = segment_staging_dir
        self.segment_dataset_id = segment_dataset_id
        self.segment_duration_ns = int(segment_duration_seconds * 1e9)
        self.segment_max_bytes = segment_max_bytes
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: VideoParquetEpisodeWriter | None = None
        self._status = RecordingStatus()

    def status(self) -> dict:
        with self._lock:
            value = self._status
        wire = asdict(value)
        if value.started_monotonic_ns is not None and value.state in {"recording", "stopping"}:
            wire["duration_seconds"] = max(
                0.0, (time.monotonic_ns() - value.started_monotonic_ns) / 1e9
            )
        return wire

    def start(self) -> dict:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("a recording session is already active")
            self._stop.clear()
            started = time.monotonic_ns()
            state = self.state_provider()
            mode = state.get("mode", "human")
            name = time.strftime(f"dagger-{mode}-%Y%m%d-%H%M%S", time.gmtime())
            episode_id = str(uuid.uuid4())
            if self.segment_worker is not None and not self.segment_worker.reserve(
                episode_id, 0, self.segment_max_bytes
            ):
                raise RuntimeError("rolling segment byte admission is closed")
            writer_name = f"{episode_id}.000000" if self.segment_worker else name
            writer = VideoParquetEpisodeWriter(
                self.output_dir,
                episode_name=writer_name,
                episode_id=episode_id,
                codec=self.codec,
                fps=self.fps,
                game=self.game,
                config={
                    **state,
                    "actions_schema_id": "nxml.dagger-actions.v2",
                    "capture_source": "native-mjpeg-fanout.v1",
                },
            )
            self._status = RecordingStatus(
                state="recording",
                episode_id=episode_id,
                episode_name=name,
                started_monotonic_ns=started,
                rolling=self.segment_worker is not None,
                current_segment=0 if self.segment_worker else None,
            )
            self._thread = threading.Thread(
                target=self._record, args=(writer,), daemon=True, name="dagger-human-recording"
            )
            self._writer = writer
            if hasattr(self.history, "begin_recording"):
                self.history.begin_recording()
            self._thread.start()
        return self.status()

    def append_boundary(self, kind: str, payload: dict | None = None) -> None:
        with self._lock:
            writer = self._writer
        if writer is not None:
            writer.append_event(
                kind,
                timestamp=time.time(),
                monotonic_ns=time.monotonic_ns(),
                source="dagger-ui",
                payload=payload,
            )

    def stop(self, *, timeout: float = 15.0) -> dict:
        with self._lock:
            thread = self._thread
            if thread is None:
                raise RuntimeError("no recording session is active")
            self._stop.set()
            self._status = RecordingStatus(**{**asdict(self._status), "state": "stopping"})
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError("recording did not finalize within timeout")
        return self.status()

    def _record(self, writer: VideoParquetEpisodeWriter) -> None:
        if self.history is None:
            raise RuntimeError("DAgger recording requires arbitration history")
        synchronizer = ArbitrationSynchronizer(self.source, self.history)
        error: str | None = None
        video_path: Path | None = None
        last_invalid_reasons: tuple[str, ...] | None = None
        last_takeover = False
        last_gap_state = "none"
        segment_index = 0
        segment_start_ns: int | None = None
        last_frame_ns: int | None = None
        total_frames = 0
        try:
            for synced in synchronizer.frames():
                if self._stop.is_set():
                    break
                frame_ns = getattr(synced, "frame_monotonic_ns", None)
                if self.segment_worker is not None and segment_start_ns is None:
                    segment_start_ns = frame_ns
                video_path_now = getattr(writer, "_video_path", None)
                video_bytes = (
                    video_path_now.stat().st_size
                    if video_path_now is not None and video_path_now.exists()
                    else 0
                )
                cut = (
                    self.segment_worker is not None
                    and len(writer) > 0
                    and frame_ns is not None
                    and segment_start_ns is not None
                    and (
                        frame_ns - segment_start_ns >= self.segment_duration_ns
                        or video_bytes >= self.segment_max_bytes
                    )
                )
                if cut:
                    assert frame_ns is not None and segment_start_ns is not None
                    self._finalize_segment(writer, segment_index, segment_start_ns, frame_ns)
                    segment_index += 1
                    if not self.segment_worker.reserve(
                        writer.episode_id, segment_index, self.segment_max_bytes
                    ):
                        self._close_rolling_episode(writer.episode_id, segment_index, None)
                        raise RuntimeError("rolling segment byte admission is closed")
                    writer = self._new_segment_writer(writer, segment_index)
                    with self._lock:
                        self._writer = writer
                        self._status = RecordingStatus(
                            **{
                                **asdict(self._status),
                                "current_segment": segment_index,
                                "segment_count": segment_index,
                            }
                        )
                    segment_start_ns = frame_ns
                boundary_sequence = getattr(synced, "boundary_sequence", None)
                claim_ack = boundary_sequence is not None and self.history.claim_boundary_ack(
                    boundary_sequence
                )
                if claim_ack:
                    synced = replace(synced, boundary_acknowledged=True)
                writer.append(synced)
                total_frames += 1
                if self.segment_worker is not None:
                    last_frame_ns = frame_ns
                if claim_ack:
                    self.history.acknowledge_boundary(boundary_sequence)
                    writer.append_event(
                        "neutral_boundary_acknowledged",
                        timestamp=synced.timestamp,
                        monotonic_ns=synced.action_monotonic_ns,
                        source="dagger-recorder",
                        payload={"boundary_sequence": boundary_sequence},
                    )
                if not synced.valid and synced.invalid_reasons != last_invalid_reasons:
                    writer.append_event(
                        "controller_sample_invalid",
                        timestamp=synced.timestamp,
                        monotonic_ns=synced.frame_monotonic_ns,
                        source="dagger-ui",
                        payload={"reasons": list(synced.invalid_reasons)},
                    )
                elif synced.valid and last_invalid_reasons is not None:
                    writer.append_event(
                        "controller_sample_recovered",
                        timestamp=synced.timestamp,
                        monotonic_ns=synced.frame_monotonic_ns,
                        source="dagger-ui",
                    )
                last_invalid_reasons = None if synced.valid else synced.invalid_reasons
                if synced.takeover != last_takeover:
                    writer.append_event(
                        "takeover_started" if synced.takeover else "takeover_released",
                        timestamp=synced.timestamp,
                        monotonic_ns=synced.action_monotonic_ns,
                        source="dagger-arbitrator",
                        payload={
                            "mode": synced.mode,
                            "reason": getattr(synced, "takeover_reason", None),
                            "release_remaining_ns": getattr(
                                synced, "takeover_release_remaining_ns", 0
                            ),
                        },
                    )
                last_takeover = synced.takeover
                gap_state = getattr(synced, "gap_state", "none")
                if gap_state != last_gap_state:
                    kind = {
                        "transient_gap": "policy_gap_started",
                        "recovered": "policy_gap_recovered",
                        "disarmed": "policy_gap_disarmed",
                    }.get(gap_state, "policy_gap_ended")
                    writer.append_event(
                        kind,
                        timestamp=synced.timestamp,
                        monotonic_ns=synced.action_monotonic_ns,
                        source="dagger-arbitrator",
                        payload={
                            "proposal_sequence": getattr(synced, "proposal_sequence", None),
                            "proposal_age_ns": getattr(synced, "proposal_age_ns", None),
                            "reason": getattr(synced, "gap_reason", None),
                            "duration_ns": getattr(synced, "gap_duration_ns", 0),
                        },
                    )
                    last_gap_state = gap_state
                with self._lock:
                    self._status = RecordingStatus(
                        **{
                            **asdict(self._status),
                            "frames": total_frames,
                            "invalid_actions": synchronizer.invalid_samples,
                        }
                    )
        except (CaptureFrameLossError, TimeoutError, OSError, RuntimeError) as caught:
            error = str(caught)
            writer.append_event(
                "recording_failed",
                timestamp=time.time(),
                monotonic_ns=time.monotonic_ns(),
                source="dagger-ui",
                payload={"error": error},
            )
        finally:
            if hasattr(self.history, "end_recording"):
                self.history.end_recording()
            writer.config["capture_integrity"] = {
                "status": "failed" if error else "complete",
                "error": error,
            }
            if (
                self.segment_worker is not None
                and len(writer) > 0
                and not getattr(writer, "_closed", False)
            ):
                assert segment_start_ns is not None and last_frame_ns is not None
                self._finalize_segment(
                    writer,
                    segment_index,
                    segment_start_ns,
                    last_frame_ns + max(1, int(1e9 / self.fps)),
                )
                self._close_rolling_episode(writer.episode_id, segment_index + 1, error)
                video_path = None
            else:
                video_path = writer.close()
            manifest = None
            if video_path is not None:
                manifest_path = video_path.with_name(f"{writer.episode_name}.manifest.json")
                manifest = str(manifest_path)
            with self._lock:
                prior = self._status
                self._status = RecordingStatus(
                    **{
                        **asdict(prior),
                        "state": "failed" if error else "finalized",
                        "duration_seconds": (
                            (time.monotonic_ns() - prior.started_monotonic_ns) / 1e9
                            if prior.started_monotonic_ns is not None
                            else 0.0
                        ),
                        "error": error,
                        "finalized_manifest": manifest,
                    }
                )
                self._thread = None
                self._writer = None

    def _new_segment_writer(
        self, prior: VideoParquetEpisodeWriter, index: int
    ) -> VideoParquetEpisodeWriter:
        return VideoParquetEpisodeWriter(
            self.output_dir,
            episode_name=f"{prior.episode_id}.{index:06d}",
            episode_id=prior.episode_id,
            codec=self.codec,
            fps=self.fps,
            game=self.game,
            config=dict(prior.config),
        )

    def _finalize_segment(
        self,
        writer: VideoParquetEpisodeWriter,
        index: int,
        start_ns: int,
        end_ns: int,
    ) -> None:
        video = writer.close()
        if video is None or self.segment_worker is None or self.segment_staging_dir is None:
            raise RuntimeError("rolling segment did not produce a complete triplet")
        base = self.output_dir / f"{writer.episode_id}.{index:06d}"
        source = SegmentSource(
            episode_id=writer.episode_id,
            sequence_index=index,
            timeline_start_ns=start_ns,
            timeline_end_ns=end_ns,
            video=video,
            actions=Path(f"{base}.parquet"),
            events=Path(f"{base}.events.parquet"),
            manifest=Path(f"{base}.manifest.json"),
        )
        if not self.segment_worker.submit_source(source):
            raise RuntimeError("rolling segment delivery queue reached bounded capacity")

    def _close_rolling_episode(self, episode_id: str, count: int, error: str | None) -> None:
        if error or self.segment_worker is None:
            return

        def close() -> None:
            try:
                self.segment_worker.close_episode(episode_id, count)
            except TimeoutError:
                # The durable close intent remains in the journal and startup
                # recovery will finish it. A slow receipt is delivery lag, not
                # corruption of the already-finalized local recording.
                return
            except Exception as caught:
                with self._lock:
                    self._status = RecordingStatus(
                        **{**asdict(self._status), "state": "failed", "error": str(caught)}
                    )

        threading.Thread(target=close, daemon=True, name="segment-episode-close").start()
