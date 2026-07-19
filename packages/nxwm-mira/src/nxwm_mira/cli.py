"""nxwm-mira CLI — RAEv2 codec training and tooling.

Subcommands:
  train     — train the codec from a YAML config
  watch     — serve a live web viewer for a training run directory
  encode    — encode episodes to codec latents with a trained checkpoint

Important: this module must stay torch-free. ``nxwm-mira --help`` is expected
to be near-instant. All heavy imports (torch, the model code) live inside
:mod:`nxwm_mira.cli_impl.<command>`.
"""

from __future__ import annotations

import click

from nxwm_mira import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__)
def main() -> None:
    """nxwm-mira: RAEv2 representation-autoencoder codec tooling."""


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--resume",
    type=click.Path(exists=True),
    default=None,
    help="Checkpoint (.pth or checkpoint-N dir) to continue from (overrides run.continue_from)",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Run directory (overrides run.output_dir; required under torchrun so ranks agree)",
)
def train(config_path: str, resume: str | None, output_dir: str | None) -> None:
    """Train the codec from a YAML config."""
    from nxwm_mira.cli_impl.train import run_train

    run_train(config_path=config_path, resume=resume, output_dir=output_dir)


@main.command(name="train-wm")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--resume",
    type=click.Path(exists=True),
    default=None,
    help="Checkpoint (.pth or checkpoint-N dir) to continue from (overrides run.continue_from)",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Run directory (overrides run.output_dir; required under torchrun so ranks agree)",
)
def train_wm(config_path: str, resume: str | None, output_dir: str | None) -> None:
    """Train the latent world model from a YAML config (needs a codec checkpoint)."""
    from nxwm_mira.cli_impl.train import run_train_wm

    run_train_wm(config_path=config_path, resume=resume, output_dir=output_dir)


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8800, type=int, show_default=True)
def watch(run_dir: str, host: str, port: int) -> None:
    """Serve a live web viewer for a training run directory (no GPU needed)."""
    from nxwm_mira.cli_impl.watch import run_watch

    run_watch(run_dir=run_dir, host=host, port=port)


@main.command()
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(exists=True),
    help="Codec checkpoint for roundtrip mode (.pth, checkpoint-N dir, or run dir -> latest)",
)
@click.option(
    "--world-model",
    default=None,
    type=click.Path(exists=True),
    help="World-model checkpoint: actions drive latent dynamics (the real playable mode)",
)
@click.option(
    "--data-root",
    default="data/za-mp4",
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("--host", default="127.0.0.1", show_default=True, help="0.0.0.0 for tailnet access")
@click.option("--port", default=8801, type=int, show_default=True)
@click.option("--device", default=None, help="Default: cuda if available")
@click.option("--flow-steps", default=10, type=int, show_default=True, help="(world-model only)")
def play(
    checkpoint: str | None,
    world_model: str | None,
    data_root: str,
    host: str,
    port: int,
    device: str | None,
    flow_steps: int,
) -> None:
    """Gamepad browser session (nxwm UI).

    With --world-model, your controller drives the model's latent dynamics (kv-cached
    streaming rollout). With only --checkpoint (a codec), runs in codec-roundtrip mode:
    streams decoded episode frames while capturing your input. The UI's reload box
    hot-swaps to newer checkpoints mid-training in either mode.
    """
    from nxwm_mira.cli_impl.play import run_play

    run_play(
        checkpoint=checkpoint,
        data_root=data_root,
        host=host,
        port=port,
        device=device,
        world_model=world_model,
        flow_steps=flow_steps,
    )


@main.command()
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True),
    help="Codec checkpoint (.pth or checkpoint-N dir)",
)
@click.option(
    "--episodes",
    default=None,
    help='Comma-separated episode indices (e.g. "0,5,17"); default: all',
)
@click.option(
    "--data-root",
    default="data/za-mp4",
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Output directory for per-episode latent .npz files",
)
@click.option("--device", default=None, help="Default: cuda if available")
@click.option("--batch-frames", default=64, type=int, show_default=True)
def encode(**kwargs: object) -> None:
    """Encode raw episodes into codec-latent .npz files."""
    from nxwm_mira.cli_impl.encode import run_encode

    run_encode(**kwargs)  # type: ignore[arg-type]


if __name__ == "__main__":
    main(prog_name="nxwm-mira")
