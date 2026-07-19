"""Human DAgger recording sessions over the shared capture fan-out."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from nxml_capture import ControllerSubscription, Synchronizer, VideoParquetEpisodeWriter
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
    ) -> None:
        self.source = source
        self.output_dir = output_dir
        self.orchestrator_ws = orchestrator_ws
        self.codec = codec
        self.fps = fps
        self.game = game
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
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
            name = time.strftime("dagger-human-%Y%m%d-%H%M%S", time.gmtime())
            writer = VideoParquetEpisodeWriter(
                self.output_dir,
                episode_name=name,
                codec=self.codec,
                fps=self.fps,
                game=self.game,
                config={"mode": "human", "capture_source": "native-mjpeg-fanout.v1"},
            )
            self._status = RecordingStatus(
                state="recording",
                episode_id=writer.episode_id,
                episode_name=name,
                started_monotonic_ns=started,
            )
            self._thread = threading.Thread(
                target=self._record, args=(writer,), daemon=True, name="dagger-human-recording"
            )
            self._thread.start()
        return self.status()

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
        controller = ControllerSubscription(url=self.orchestrator_ws)
        synchronizer = Synchronizer(self.source, controller, driver="human")
        error: str | None = None
        video_path: Path | None = None
        controller.start()
        last_invalid_reasons: tuple[str, ...] | None = None
        try:
            for synced in synchronizer.frames():
                if self._stop.is_set():
                    break
                writer.append(synced)
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
                with self._lock:
                    self._status = RecordingStatus(
                        **{
                            **asdict(self._status),
                            "frames": len(writer),
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
            controller.stop()
            writer.config["capture_integrity"] = {
                "status": "failed" if error else "complete",
                "error": error,
            }
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
