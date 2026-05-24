"""nxml-autopilot CLI — hybrid human/AI play.

Usage:

    nxml-autopilot --game pokemon-za \\
                --policy hf:csaben/za-ppo-v1.pt \\
                --controller http://localhost:7777 \\
                --controller-input auto \\
                --record ./data/autopilot/

Pre-requisite: ``nxbt-orchestrator`` running and connected to the target
controller; the human's PC controller plugged in (xbox / dualshock / etc.,
auto-detected by mapper name).
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from types import FrameType

import click

from nxml_autopilot import __version__


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__)
@click.option("--game", required=True, help="Game identifier — stamped into recorded manifest.")
@click.option("--policy", "policy_uri", required=True, help="Policy URI: hf:..., zmq://host:port, or local path.")
@click.option(
    "--controller",
    "controller_url",
    required=True,
    help="HTTP URL of nxbt-orchestrator (e.g. http://localhost:7777). Actions POSTed to /action.",
)
@click.option(
    "--controller-input",
    default="auto",
    show_default=True,
    help="Mapper id ('xbox_one', 'switch_pro', ...) or 'auto' to detect by device name.",
)
@click.option(
    "--device-path",
    default=None,
    type=click.Path(),
    help="evdev device path (e.g. /dev/input/event7). If omitted with 'auto', enumerates devices.",
)
@click.option("--camera", "camera_id", default=0, type=int, show_default=True, help="v4l2 device index.")
@click.option("--capture-width", default=1920, type=int, show_default=True, help="Requested capture width (px).")
@click.option("--capture-height", default=1080, type=int, show_default=True, help="Requested capture height (px).")
@click.option(
    "--input-source",
    default="evdev",
    type=click.Choice(["evdev", "web"]),
    show_default=True,
    help="Where the human input comes from. 'web' serves a browser teleop UI with gamepad + MJPEG.",
)
@click.option("--web-host", default="0.0.0.0", show_default=True, help="Bind host for --input-source=web.")
@click.option("--web-port", default=8080, type=int, show_default=True, help="Bind port for --input-source=web.")
@click.option(
    "--web-token",
    envvar="AUTOPILOT_WEB_TOKEN",
    default=None,
    help="Shared secret required on the WS UI. Empty = no auth (LAN-only). Env: AUTOPILOT_WEB_TOKEN.",
)
@click.option(
    "--web-stick-deadzone",
    default=0.15,
    type=float,
    show_default=True,
    help="Deadzone for browser-side stick input. Bump if your controller's resting drift exceeds this.",
)
@click.option(
    "--mode",
    default="human-priority",
    type=click.Choice(["human-priority", "human-takeover"]),
    show_default=True,
    help=(
        "Mux strategy. 'human-priority' merges per-index (AI fills indices the human isn't touching). "
        "'human-takeover' fully suppresses AI while any human input is active (stick or button)."
    ),
)
@click.option(
    "--record",
    "record_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="If set, record session as a video+parquet episode under this directory.",
)
@click.option(
    "--macro-dir",
    "macro_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for macro JSON files. Default: ./data/macros/<game>/.",
)
@click.option(
    "--trigger-dir",
    "trigger_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help=(
        "Directory for visual-trigger JSON specs + reference images. "
        "Default: ./data/triggers/<game>/. When set (or defaulted) the web UI "
        "exposes a Triggers panel for arming a 'watch for image → run macro' loop."
    ),
)
@click.option(
    "--vae-path",
    envvar="VAE_PATH",
    default=None,
    help="Local VAE dir or HF model id (default: stabilityai/sd-vae-ft-mse).",
)
@click.option("--device", default=None, help="Inference device (default: cuda if available).")
@click.option("--tick-hz", default=30.0, type=float, show_default=True, help="Mux/POST tick rate.")
def main(
    game: str,
    policy_uri: str,
    controller_url: str,
    controller_input: str,
    device_path: str | None,
    camera_id: int,
    capture_width: int,
    capture_height: int,
    input_source: str,
    web_host: str,
    web_port: int,
    web_token: str | None,
    web_stick_deadzone: float,
    mode: str,
    record_dir: Path | None,
    macro_dir: Path | None,
    trigger_dir: Path | None,
    vae_path: str | None,
    device: str | None,
    tick_hz: float,
) -> None:
    """Run a hybrid human/AI autopilot session."""
    from nxml_autopilot.runner import AutopilotConfig, AutopilotRunner

    if macro_dir is None:
        macro_dir = Path("./data/macros") / game
    if trigger_dir is None:
        trigger_dir = Path("./data/triggers") / game

    config = AutopilotConfig(
        game=game,
        policy_uri=policy_uri,
        controller_url=controller_url,
        controller_input=controller_input,
        device_path=device_path,
        camera_id=camera_id,
        capture_width=capture_width,
        capture_height=capture_height,
        input_source=input_source,
        web_host=web_host,
        web_port=web_port,
        web_token=web_token or "",
        web_stick_deadzone=web_stick_deadzone,
        mode=mode,
        record_dir=record_dir,
        macro_dir=macro_dir,
        trigger_dir=trigger_dir,
        vae_path=vae_path,
        device=device,
        tick_hz=tick_hz,
    )

    runner = AutopilotRunner(config)

    def _on_signal(signum: int, _frame: FrameType | None) -> None:
        print(f"\n[autopilot] received signal {signum}, shutting down…", flush=True)
        runner.request_stop()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        runner.run()
    except KeyboardInterrupt:
        runner.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main(prog_name="nxml-autopilot")
