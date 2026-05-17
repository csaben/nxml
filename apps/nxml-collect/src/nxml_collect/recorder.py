"""Episode recorder: capture + controller subscribe + sync + write.

One process invocation == one episode. Ctrl-C flushes and closes cleanly. A
``--max-frames`` cap is provided as a safety stop for unattended runs.
"""

from __future__ import annotations

import json
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType

from nxml_capture import (
    ControllerSubscription,
    NpzEpisodeWriter,
    Synchronizer,
    VideoParquetEpisodeWriter,
)
from nxml_capture.backends.v4l2 import V4L2Source


@dataclass
class RecorderConfig:
    output_dir: Path
    game: str
    camera_id: int = 0
    orchestrator_url: str = "ws://127.0.0.1:7777/ws/state"
    max_frames: int | None = None
    max_action_age: float = 0.5
    initial_timeout: float = 10.0
    progress_every: int = 60
    ui_port: int | None = None
    ui_host: str = "0.0.0.0"  # noqa: S104
    writer: str = "video_parquet"  # "video_parquet" | "npz"
    codec: str = "ffv1"  # only used when writer == "video_parquet"
    fps: float = 30.0
    extra_metadata: dict[str, object] = field(default_factory=dict)


def run_recorder(config: RecorderConfig) -> Path | None:
    source = V4L2Source(camera_id=config.camera_id)
    controller = ControllerSubscription(url=config.orchestrator_url)
    sync = Synchronizer(
        source,
        controller,
        max_action_age=config.max_action_age,
        initial_timeout=config.initial_timeout,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.writer == "video_parquet":
        writer: NpzEpisodeWriter | VideoParquetEpisodeWriter = VideoParquetEpisodeWriter(
            config.output_dir,
            codec=config.codec,
            fps=config.fps,
        )
    elif config.writer == "npz":
        writer = NpzEpisodeWriter(config.output_dir)
    else:
        raise ValueError(f"unknown writer: {config.writer!r}")

    stop_flag = {"stop": False}

    def _on_signal(signum: int, _frame: FrameType | None) -> None:
        stop_flag["stop"] = True
        print(f"\n[nxml-collect] received signal {signum}, flushing…", flush=True)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"[nxml-collect] camera={config.camera_id} orchestrator={config.orchestrator_url}")
    print(f"[nxml-collect] output_dir={config.output_dir} episode={writer.episode_name}")

    ui_server = None
    if config.ui_port is not None:
        # Eagerly start capture so MJPEG has frames immediately. Synchronizer
        # __enter__ is idempotent on already-started sources.
        source.start()
        from nxml_collect.ui import UiServer, derive_orchestrator_http

        ui_server = UiServer(
            source,
            host=config.ui_host,
            port=config.ui_port,
            orchestrator_http_url=derive_orchestrator_http(config.orchestrator_url),
        )
        ui_server.start()
        print(f"[nxml-collect] teleop UI on http://{config.ui_host}:{config.ui_port}")

    print("[nxml-collect] waiting for first controller snapshot…")

    out_path: Path | None = None
    try:
        with sync:
            t_start = time.time()
            for synced in sync.frames():
                writer.append(synced)
                count = len(writer)
                if count % config.progress_every == 0:
                    elapsed = time.time() - t_start
                    fps = count / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[nxml-collect] frames={count} elapsed={elapsed:5.1f}s fps={fps:5.2f}",
                        flush=True,
                    )
                if stop_flag["stop"]:
                    break
                if config.max_frames is not None and count >= config.max_frames:
                    print(f"[nxml-collect] reached --max-frames={config.max_frames}")
                    break
    finally:
        if ui_server is not None:
            ui_server.stop()
        out_path = writer.close()
        if out_path is not None:
            _stamp_metadata(out_path, config)
            print(f"[nxml-collect] wrote {out_path} ({len(writer)} frames)")
        else:
            print("[nxml-collect] no frames captured; nothing written")

    return out_path


def _stamp_metadata(episode_path: Path, config: RecorderConfig) -> None:
    """Augment the writer's manifest with collector-level metadata."""
    manifest_path = episode_path.with_name(f"{episode_path.stem}.manifest.json")
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest["game"] = config.game
    manifest["orchestrator_url"] = config.orchestrator_url
    manifest["camera_id"] = config.camera_id
    if config.extra_metadata:
        manifest["extra"] = config.extra_metadata
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main_with_exit(config: RecorderConfig) -> int:
    try:
        run_recorder(config)
    except TimeoutError as e:
        print(f"[nxml-collect] {e}", file=sys.stderr)
        return 1
    return 0
