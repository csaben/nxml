"""nxml-spool CLI.

  nxml-spool run   --watch DIR [--watch DIR2] --repo owner/name [options]
  nxml-spool status --state-dir DIR
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from nxml_spool import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__)
def main() -> None:
    """nxml-spool: episodes -> WebDataset shards -> HuggingFace -> verified local delete."""


@main.command()
@click.option(
    "--watch",
    "watch_dirs",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Episode output dir to watch (repeatable; collect flat dirs and autopilot roots both work)",
)
@click.option("--repo", "repo_id", required=True, help="HF dataset repo, e.g. arelius/nxml-pokemon-legends-za-v2")
@click.option(
    "--storage-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Mounted object-storage root. Uses immutable filesystem commits instead of HF.",
)
@click.option(
    "--state-dir",
    default=Path("~/.local/state/nxml-spool").expanduser(),
    type=click.Path(file_okay=False, path_type=Path),
    show_default=True,
    help="Journal + staging + status.json location",
)
@click.option("--shard-size-mb", default=1024, show_default=True, type=int)
@click.option("--settle-seconds", default=30.0, show_default=True, type=float,
              help="Quiet time before an episode counts as complete")
@click.option("--poll-seconds", default=15.0, show_default=True, type=float)
@click.option("--flush-partial-after", default=900.0, show_default=True, type=float,
              help="Seal a below-size shard once its oldest episode is this old (seconds)")
@click.option("--no-delete", is_flag=True, help="Keep local episode files after verified upload")
@click.option("--public", is_flag=True, help="Create the repo public (default private/gated)")
@click.option("--once", is_flag=True, help="One pass (pack+upload whatever is ready) then exit")
@click.option("--disk-high-watermark", default=0.85, show_default=True, type=float)
@click.option("--disk-low-watermark", default=0.75, show_default=True, type=float)
def run(
    watch_dirs: tuple[Path, ...],
    repo_id: str,
    storage_root: Path | None,
    state_dir: Path,
    shard_size_mb: int,
    settle_seconds: float,
    poll_seconds: float,
    flush_partial_after: float,
    no_delete: bool,
    public: bool,
    once: bool,
    disk_high_watermark: float,
    disk_low_watermark: float,
) -> None:
    """Run the spool loop."""
    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%X]", level=logging.INFO)
    from nxml_spool.spooler import run_spooler
    from nxml_spool.storage import FilesystemStorageBackend, HFDatasetStorageBackend

    storage = (
        FilesystemStorageBackend(storage_root)
        if storage_root is not None
        else HFDatasetStorageBackend(repo_id, private=not public)
    )

    run_spooler(
        list(watch_dirs),
        repo_id=repo_id,
        state_dir=state_dir,
        shard_size_mb=shard_size_mb,
        settle_seconds=settle_seconds,
        poll_seconds=poll_seconds,
        delete_after_upload=not no_delete,
        flush_partial_after_s=flush_partial_after,
        storage=storage,
        once=once,
        disk_high_watermark=disk_high_watermark,
        disk_low_watermark=disk_low_watermark,
    )


@main.command()
@click.option(
    "--state-dir",
    default=Path("~/.local/state/nxml-spool").expanduser(),
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    show_default=True,
)
def status(state_dir: Path) -> None:
    """Print the spooler's current status.json."""
    status_path = state_dir / "status.json"
    if not status_path.is_file():
        raise click.ClickException(f"No status.json in {state_dir} — is the spooler running?")
    click.echo(json.dumps(json.loads(status_path.read_text()), indent=2))


if __name__ == "__main__":
    main(prog_name="nxml-spool")
