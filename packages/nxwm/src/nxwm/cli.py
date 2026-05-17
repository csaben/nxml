"""nxwm CLI — world model tooling for Nintendo Switch games.

Subcommands:
  train     — train a world model from YAML config (DDP via launcher)
  rollout   — generate frames offline from a model + seed episode
  serve     — run an inference server for remote clients
  ui        — interactive web UI for inspecting/labeling

Important: this module must stay torch-free. ``nxwm --help`` is expected to be
near-instant. All heavy imports (torch, diffusers, the model code) live inside
the subcommand functions or in :mod:`nxwm.cli_impl.<command>`.
"""

from __future__ import annotations

import click

from nxwm import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__)
def main() -> None:
    """nxwm: Nintendo Switch world model tooling."""


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--resume",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Resume from this checkpoint (overrides config's resume_from)",
)
@click.option(
    "--world-size",
    type=int,
    default=None,
    help="Number of GPUs (default: torch.cuda.device_count())",
)
def train(config_path: str, resume: str | None, world_size: int | None) -> None:
    """Train a world model from a YAML config."""
    from nxwm.cli_impl.train import run_train

    run_train(config_path=config_path, resume=resume, world_size=world_size)


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", required=True, help="Model URI (path, hf:..., zmq://...)")
@click.option(
    "--seed-episode",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help=".npz episode file to seed from",
)
@click.option("--start-frame", default=100, type=int, show_default=True)
@click.option("--steps", default=60, type=int, show_default=True, help="Frames to generate")
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output path (extension picked by --format)",
)
@click.option(
    "--format",
    "out_format",
    default="gif",
    type=click.Choice(["gif", "npz"]),
    show_default=True,
    help="Output format. 'npz' requires an in-process model (saves raw latents).",
)
@click.option("--device", default=None, help="(in-process only)")
@click.option("--flow-steps", default=5, type=int, show_default=True)
@click.option("--cfg-scale", default=1.0, type=float, show_default=True)
@click.option(
    "--actions",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=".npy file of (steps, action_dim) actions; defaults to all-zero (no input)",
)
def rollout(**kwargs: object) -> None:
    """Generate frames offline from a model + seed episode."""
    from nxwm.cli_impl.rollout import run_rollout

    run_rollout(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Episode video (.mkv/.mp4) or directory of them; each must have a sibling .parquet",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Output directory for chunked latent .npz files",
)
@click.option(
    "--vae-path",
    envvar="VAE_PATH",
    default=None,
    help="Local VAE directory or HF model id (default: stabilityai/sd-vae-ft-mse)",
)
@click.option("--device", default=None, help="Encoding device (default: cuda if available)")
@click.option("--batch-size", default=32, type=int, show_default=True)
@click.option(
    "--frames-per-chunk",
    default=30 * 60,
    type=int,
    show_default=True,
    help="Frames per output .npz part (default: 60s @ 30fps)",
)
@click.option("--overwrite", is_flag=True, help="Re-encode files whose parts already exist")
def encode(**kwargs: object) -> None:
    """Encode raw frame episodes into chunked VAE-latent .npz files."""
    from nxwm.cli_impl.encode import run_encode

    run_encode(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", required=True, help="Model URI (path or hf:...)")
@click.option("--port", default=5556, type=int, show_default=True)
@click.option(
    "--host",
    default="*",
    show_default=True,
    help='Bind address (use "*" for all interfaces)',
)
@click.option(
    "--transport",
    default="zmq",
    type=click.Choice(["zmq", "http"]),
    show_default=True,
    help="Transport protocol: zmq (binary, REQ/REP) or http (FastAPI + WebSocket session)",
)
@click.option("--device", default=None, help="Inference device (default: cuda if available)")
@click.option(
    "--ssl-certfile",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="(http transport only) TLS cert file — enables https:// / wss://",
)
@click.option(
    "--ssl-keyfile",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="(http transport only) TLS key file paired with --ssl-certfile",
)
@click.option(
    "--root-path",
    default="",
    show_default=True,
    help="(http transport only) ASGI root_path when behind a reverse proxy",
)
@click.option(
    "--data-path",
    envvar="WM_DATA_PATH",
    default=None,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--checkpoint-dir",
    envvar="WM_CHECKPOINT_DIR",
    default=None,
    type=click.Path(file_okay=False),
)
@click.option(
    "--vae-path",
    envvar="VAE_PATH",
    default=None,
    help="Local VAE directory or HF model id (default: stabilityai/sd-vae-ft-mse)",
)
@click.option("--flow-steps", default=5, type=int, show_default=True)
@click.option("--cfg-scale", default=1.0, type=float, show_default=True)
def serve(**kwargs: object) -> None:
    """Run the world model inference server."""
    from nxwm.cli_impl.serve import run_serve

    run_serve(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ui
# ---------------------------------------------------------------------------


@main.command()
@click.argument("game")
@click.option("--model", required=True, help="Model URI (path, hf:..., zmq://...)")
@click.option("--port", default=8080, type=int, show_default=True, help="UI server port")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="UI bind address (use 0.0.0.0 to expose on LAN)",
)
@click.option("--no-browser", is_flag=True, help="Don't auto-open browser")
@click.option("--device", default=None, help="(in-process only)")
@click.option("--flow-steps", default=5, type=int, show_default=True, help="(in-process only)")
@click.option("--cfg-scale", default=1.0, type=float, show_default=True, help="(in-process only)")
@click.option(
    "--data-path",
    envvar="WM_DATA_PATH",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="(in-process only) directory of seed episodes",
)
@click.option(
    "--vae-path",
    envvar="VAE_PATH",
    default=None,
    help="(in-process only) local VAE dir or HF model id (default: stabilityai/sd-vae-ft-mse)",
)
@click.option(
    "--detector",
    "detector_name",
    default=None,
    help=(
        "(in-process only) attach a CV detector for live threshold tuning, "
        "e.g. 'pokemon_za:target_ui'. Use --detector-arg key=value for "
        "ctor args (template_path, threshold, sat_threshold, ...)."
    ),
)
@click.option(
    "--detector-arg",
    "detector_args",
    multiple=True,
    metavar="KEY=VALUE",
    help="Per-detector ctor argument; pass multiple. Numeric values auto-cast.",
)
def ui(**kwargs: object) -> None:
    """Launch the interactive UI for inspecting / labeling.

    Examples:

    \b
        # In-process (single-line, model loads in this process)
        nxwm ui my-game --model hf:owner/repo/file.pt
        nxwm ui my-game --model ./checkpoints/best_val.pt

    \b
        # Remote (model runs in a separate `nxwm serve` process)
        nxwm ui my-game --model zmq://gpu-box.local:5556
    """
    from nxwm.cli_impl.ui import run_ui

    run_ui(**kwargs)  # type: ignore[arg-type]


if __name__ == "__main__":
    main(prog_name="nxwm")
