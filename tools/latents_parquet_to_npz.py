"""Split the latents-repo chunk parquet back into per-episode ``.npz`` files.

Input (downloaded from ``arelius/nxml-pokemon-legends-za-latents``)::

    data/chunk-000.parquet     cols: episode_index, frame_idx, timestamp,
                                     action (list<f32,26>), latent (list<f16,2048>)
    meta/episodes.parquet      cols: episode_index, episode_id, ...

Output (consumable by ``nxwm.training.dataset.LatentEpisodeDataset``)::

    {episode_id}.npz with {"latents": (T, 4, 16, 32) f16, "actions": (T, 26) f32}

Usage::

    uv run --with click --with pandas --with pyarrow python \\
        tools/latents_parquet_to_npz.py \\
        --parquet ./za-latents/data/chunk-000.parquet \\
        --meta    ./za-latents/meta/episodes.parquet \\
        --out     ./data/latents
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pyarrow.parquet as pq


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--parquet",
    "parquet_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="data/chunk-XXX.parquet from the HF dataset.",
)
@click.option(
    "--meta",
    "meta_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="meta/episodes.parquet from the HF dataset.",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write per-episode .npz files into.",
)
@click.option(
    "--latent-shape",
    default="4,16,32",
    show_default=True,
    help="(C,H,W) shape of each latent.",
)
def main(parquet_path: Path, meta_path: Path, out_dir: Path, latent_shape: str) -> None:
    """Split the latents-repo chunk parquet into per-episode .npz files."""
    c, h, w = (int(x) for x in latent_shape.split(","))
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_meta = pq.read_table(meta_path).to_pandas()
    id_by_idx = dict(zip(ep_meta.episode_index, ep_meta.episode_id, strict=False))

    frames = pq.read_table(parquet_path).to_pandas()
    frames = frames.sort_values(["episode_index", "frame_idx"])

    n_eps = frames.episode_index.nunique()
    click.echo(f"splitting {len(frames):,} rows -> {n_eps} episodes -> {out_dir}")

    for ep_idx, sub in frames.groupby("episode_index", sort=False):
        ep_id = id_by_idx.get(int(ep_idx), f"episode_{int(ep_idx):06d}")
        latents = np.stack(sub.latent.values).astype(np.float16).reshape(-1, c, h, w)
        actions = np.stack(sub.action.values).astype(np.float32)
        out_path = out_dir / f"{ep_id}.npz"
        np.savez(out_path, latents=latents, actions=actions)
        click.echo(f"  wrote {out_path.name} ({latents.shape[0]} frames)")

    click.echo("done")


if __name__ == "__main__":
    main()
