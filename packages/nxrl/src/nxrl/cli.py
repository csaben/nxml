"""nxrl CLI — behavior cloning + reinforcement learning.

Subcommands: ``train``, ``serve``, ``eval``, ``rollout-debug``.

Important: this module must stay torch-free. ``nxrl --help`` should be
near-instant. Heavy imports (torch, the model code) live inside the
subcommand functions or in ``nxrl.cli_impl.<command>``.
"""

from __future__ import annotations

import click

from nxrl import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__)
def main() -> None:
    """nxrl: behavior cloning + reinforcement learning."""


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--resume",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Resume from this checkpoint (overrides config's run.resume_from)",
)
@click.option(
    "--world-size",
    type=int,
    default=None,
    help="Number of GPUs (default: torch.cuda.device_count())",
)
def train(config_path: str, resume: str | None, world_size: int | None) -> None:
    """Train a policy from a YAML config (DDP across visible GPUs)."""
    from nxrl.cli_impl.train import run_train

    run_train(config_path=config_path, resume=resume, world_size=world_size)


@main.command()
@click.option("--policy", required=True, help="Policy URI (path or hf:owner/repo/file.pt)")
@click.option("--port", default=5557, type=int, show_default=True)
@click.option(
    "--host",
    default="*",
    show_default=True,
    help='Bind address (use "*" for all interfaces)',
)
@click.option(
    "--transport",
    default="zmq",
    type=click.Choice(["zmq"]),
    show_default=True,
    help="Transport protocol (only zmq supported in v1)",
)
@click.option("--device", default=None, help="Inference device (default: cuda if available)")
@click.option(
    "--checkpoint-dir",
    envvar="NXRL_CHECKPOINT_DIR",
    default=None,
    type=click.Path(file_okay=False),
)
@click.option(
    "--enable-frame-mode",
    is_flag=True,
    default=False,
    help="Accept MODE_PREDICT_FRAME requests: server runs the VAE and a sliding latent window itself. Single-client.",
)
@click.option(
    "--vae-path",
    envvar="VAE_PATH",
    default=None,
    help="VAE for frame mode (default: stabilityai/sd-vae-ft-mse). Ignored unless --enable-frame-mode.",
)
def serve(**kwargs: object) -> None:
    """Run the policy inference server (ZMQ REQ/REP)."""
    from nxrl.cli_impl.serve import run_serve

    run_serve(**kwargs)  # type: ignore[arg-type]


@main.command("rollout-debug")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="PPO YAML config (same one used by `nxrl train`)",
)
@click.option(
    "--policy",
    "policy_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Policy checkpoint to evaluate (PPO or BC)",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory for per-rollout .mkv files",
)
@click.option(
    "--seed-indices",
    default=None,
    help="Comma-separated seed indices to run (default: all seeds in config)",
)
@click.option(
    "--rollouts-per-seed",
    default=1,
    type=int,
    show_default=True,
    help="Trajectories per seed (stochastic policy = different each run)",
)
@click.option("--fps", default=30, type=int, show_default=True)
@click.option("--device", default=None, help="(default: cuda if available)")
@click.option(
    "--load-as",
    default="auto",
    type=click.Choice(["auto", "ppo", "bc"]),
    show_default=True,
    help="Treat --policy as a PPO or BC checkpoint (auto-detects by default)",
)
def rollout_debug(
    config_path: str,
    policy_path: str,
    output_dir: str,
    seed_indices: str | None,
    rollouts_per_seed: int,
    fps: int,
    device: str | None,
    load_as: str,
) -> None:
    """Roll out a policy and write .mkv files with per-step reward components burned in.

    Reuses the trainer's rollout + reward path, then decodes predicted latents
    via the VAE and overlays the 16-component reward breakdown on the side of
    each frame. One file per (seed, rollout) under --output-dir.
    """
    from nxrl.cli_impl.rollout_debug import run_rollout_debug

    parsed_seeds: tuple[int, ...] | None = None
    if seed_indices:
        parsed_seeds = tuple(int(s.strip()) for s in seed_indices.split(",") if s.strip())

    run_rollout_debug(
        config_path=config_path,
        policy_path=policy_path,
        output_dir=output_dir,
        seed_indices=parsed_seeds,
        rollouts_per_seed=rollouts_per_seed,
        fps=fps,
        device=device,
        load_as=load_as,
    )


@main.command()
@click.option("--policy", required=True, help="Policy URI (path, hf:..., zmq://...)")
@click.option("--world-model", "wm_path", required=True, help="World model checkpoint path")
@click.option(
    "--seed-episode",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Latent .npz episode to seed the rollout from",
)
@click.option("--start-frame", default=0, type=int, show_default=True)
@click.option("--frames", default=150, type=int, show_default=True, help="Frames to roll out")
@click.option("--goal-offset", default=30, type=int, show_default=True)
@click.option("--flow-steps", default=5, type=int, show_default=True)
@click.option("--cfg-scale", default=1.0, type=float, show_default=True)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output GIF path",
)
@click.option(
    "--vae-path",
    envvar="VAE_PATH",
    default=None,
    help="Local VAE directory or HF model id (default: stabilityai/sd-vae-ft-mse)",
)
@click.option("--device", default=None, help="Inference device (default: cuda if available)")
def eval(**kwargs: object) -> None:
    """Roll out a policy in a frozen world model and save an eval GIF."""
    from nxrl.cli_impl.eval import run_eval

    run_eval(**kwargs)  # type: ignore[arg-type]


if __name__ == "__main__":
    main(prog_name="nxrl")
