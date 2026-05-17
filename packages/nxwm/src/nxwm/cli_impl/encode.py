"""Implementation of ``nxwm encode``: video+parquet episode -> VAE latent ``.npz``.

Input: directory (or single file) of episodes in the canonical
``nxml-capture`` format (one ``{name}.mkv``/``.mp4`` + ``{name}.parquet``
+ ``{name}.manifest.json`` per episode).

Output: chunked latent files of the shape consumed by
:class:`nxwm.training.dataset.LatentEpisodeDataset`::

    {name}_part{i}.npz
        latents: (T, C, H, W) float16   # 4x16x32 for sd-vae-ft-mse @ 128x256
        actions: (T, A)        float16

Decoding uses ``torchcodec`` (NVDEC on CUDA when available). Pre-VAE
processing: bilinear resize to 128x256, scale to [0, 1], normalize to
[-1, 1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"
DEFAULT_FRAMES_PER_CHUNK = 30 * 60  # 60s @ 30fps
DEFAULT_BATCH = 32
DEFAULT_RESIZE = (128, 256)  # (H, W)
LATENT_SCALE = 0.18215  # sd-vae-ft-mse convention
VIDEO_EXTS = (".mkv", ".mp4")


def _find_episode_videos(src: Path) -> list[Path]:
    if src.is_file():
        if src.suffix in VIDEO_EXTS:
            return [src]
        raise ValueError(f"{src} is not a .mkv/.mp4 file")
    if src.is_dir():
        return sorted(p for p in src.iterdir() if p.suffix in VIDEO_EXTS)
    raise FileNotFoundError(f"{src} is neither a file nor a directory")


def _load_actions(parquet_path: Path) -> np.ndarray:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    # action: fixed_size_list<f32, 26> -> (T, 26) float32
    flat = table.column("action").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    n_frames = len(table)
    return flat.reshape(n_frames, -1).astype(np.float32, copy=False)


def run_encode(
    *,
    input_path: str,
    output_dir: str,
    vae_path: str | None,
    device: str | None,
    batch_size: int,
    frames_per_chunk: int,
    overwrite: bool,
) -> None:
    import torch
    import torchvision.transforms.v2.functional as F
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    from torchcodec.decoders import VideoDecoder

    src = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    videos = _find_episode_videos(src)
    if not videos:
        raise FileNotFoundError(f"no episode videos in {src}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vae_id = vae_path or DEFAULT_VAE
    print(f"loading VAE {vae_id} on {dev}")
    vae = AutoencoderKL.from_pretrained(vae_id).to(dev, dtype=torch.float16)  # type: ignore[arg-type]
    vae.eval()

    decoder_device = "cuda" if dev.type == "cuda" else "cpu"

    for video_path in videos:
        base = video_path.stem
        parquet_path = video_path.with_suffix(".parquet")
        if not parquet_path.exists():
            print(f"skip  {video_path.name}: missing {parquet_path.name}")
            continue
        existing = list(out.glob(f"{base}_part*.npz"))
        if existing and not overwrite:
            print(f"skip  {video_path.name} ({len(existing)} parts already present)")
            continue

        print(f"encoding {video_path.name}")
        # decoder = VideoDecoder(str(video_path), device=decoder_device)
        decoder = VideoDecoder(str(video_path), device="cpu")
        # hack: force cpu since 3.14 too new for torchcodec
        actions = _load_actions(parquet_path)
        n_frames = decoder.metadata.num_frames
        if n_frames != actions.shape[0]:
            raise ValueError(
                f"{video_path.name}: {n_frames} video frames but {actions.shape[0]} parquet rows"
            )

        latents_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, n_frames, batch_size):
                stop = min(start + batch_size, n_frames)
                # FrameBatch.data: (B, C, H, W) uint8 RGB on `decoder_device`.
                # batch = decoder.get_frames_in_range(start=start, stop=stop).data
                # batch = batch.to(dev)
                # FrameBatch.data: (B, C, H, W) uint8 RGB on CPU
                batch = decoder.get_frames_in_range(start=start, stop=stop).data

                # non_blocking=True allows the CPU to keep working while the GPU receives data
                batch = batch.pin_memory().to(dev, non_blocking=True)
                # Resize bilinear to (128, 256) preserving uint8, then normalize.
                resized = F.resize(batch, list(DEFAULT_RESIZE), antialias=True)
                normed = resized.to(torch.float16) / 255.0
                normed = (normed - 0.5) / 0.5
                z = vae.encode(normed).latent_dist.mode()  # type: ignore[union-attr]
                z = z * LATENT_SCALE
                latents_chunks.append(z.cpu())

        latents = torch.cat(latents_chunks, dim=0).numpy().astype(np.float16)
        actions_f16 = actions.astype(np.float16, copy=False)

        total = latents.shape[0]
        n_parts = 0
        for part_idx, start in enumerate(range(0, total, frames_per_chunk)):
            end = start + frames_per_chunk
            chunk_l = latents[start:end]
            chunk_a = actions_f16[start:end]
            out_path = out / f"{base}_part{part_idx}.npz"
            np.savez(out_path, latents=chunk_l, actions=chunk_a)
            n_parts += 1
            print(
                f"  wrote {out_path.name}  latents={tuple(chunk_l.shape)} actions={tuple(chunk_a.shape)}"
            )
        print(f"  {total} frames -> {n_parts} part(s)")
