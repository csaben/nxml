"""nxml-collect CLI — passive episode recording.

Usage:

    nxml-collect --game pokemon-za --output ./data/$(date +%Y%m%d)/

Pre-requisite: ``nxbt-orchestrator serve`` running and connected to the
target controller. The CLI is deliberately torch-free; ``nxml-collect --help``
should be near-instant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from nxml_collect import __version__
from nxml_collect.recorder import RecorderConfig, main_with_exit


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__)
@click.option(
    "--game",
    required=True,
    help="Game identifier — stamped into manifest.json (e.g. pokemon-za).",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where the episode (mkv/parquet or npz, per --writer) + manifest.json are written.",
)
@click.option(
    "--camera",
    "camera_id",
    default=0,
    type=int,
    show_default=True,
    help="v4l2 device index for cv2.VideoCapture.",
)
@click.option(
    "--orchestrator",
    "orchestrator_url",
    default="ws://127.0.0.1:7777/ws/state",
    show_default=True,
    help="WebSocket URL of nxbt-orchestrator's state stream.",
)
@click.option(
    "--max-frames",
    default=None,
    type=int,
    help="Stop after this many synced frames (default: run until Ctrl-C).",
)
@click.option(
    "--max-action-age",
    default=0.5,
    type=float,
    show_default=True,
    help="Drop frames whose nearest controller snapshot is older than this many seconds.",
)
@click.option(
    "--initial-timeout",
    default=10.0,
    type=float,
    show_default=True,
    help="Seconds to wait for the first controller snapshot before bailing.",
)
@click.option(
    "--ui",
    "ui",
    is_flag=True,
    default=False,
    help="Serve a browser teleop UI (MJPEG live feed + browser-Gamepad → /action).",
)
@click.option(
    "--ui-host",
    default="0.0.0.0",  # noqa: S104
    show_default=True,
    help="Bind address for the teleop UI server.",
)
@click.option(
    "--ui-port",
    default=8080,
    type=int,
    show_default=True,
    help="Port for the teleop UI server (only used when --ui is set).",
)
@click.option(
    "--writer",
    type=click.Choice(["video_parquet", "npz"]),
    default="video_parquet",
    show_default=True,
    help="Episode format: video_parquet (mkv+parquet, canonical) or npz (debug).",
)
@click.option(
    "--codec",
    type=click.Choice(["ffv1", "h264"]),
    default="ffv1",
    show_default=True,
    help="Video codec when --writer=video_parquet (ffv1=lossless mkv, h264=CRF18 mp4).",
)
@click.option(
    "--fps",
    default=30.0,
    type=float,
    show_default=True,
    help="Nominal capture fps; controls video time_base. Real per-frame timestamps are stored in parquet.",
)
def main(
    game: str,
    output_dir: Path,
    camera_id: int,
    orchestrator_url: str,
    max_frames: int | None,
    max_action_age: float,
    initial_timeout: float,
    ui: bool,
    ui_host: str,
    ui_port: int,
    writer: str,
    codec: str,
    fps: float,
) -> None:
    """Record one episode of paired (frame, action) data."""
    config = RecorderConfig(
        output_dir=output_dir,
        game=game,
        camera_id=camera_id,
        orchestrator_url=orchestrator_url,
        max_frames=max_frames,
        max_action_age=max_action_age,
        initial_timeout=initial_timeout,
        ui_port=ui_port if ui else None,
        ui_host=ui_host,
        writer=writer,
        codec=codec,
        fps=fps,
    )
    sys.exit(main_with_exit(config))


if __name__ == "__main__":
    main(prog_name="nxml-collect")
