"""`nxwm-mira encode`: encode raw episodes into codec-latent .npz files.

Output per episode: ``{output_dir}/episode_{index:06d}.npz`` with
  latents  (T_latent, C, h, w) float16 — unnormalized codec latents
  actions  (T_frames, 26) float32 — pooled onto the sampled frames (sticks mean, buttons OR)
plus a ``meta.json`` recording the codec checkpoint, ``latent_mean_std`` (what a world
model divides by), frame stride, and fps. Chunked encoding at even frame boundaries is
exact: the encoder is per-frame DINO + a stride-2 temporal conv, so no cross-chunk state.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def run_encode(
    checkpoint: str,
    episodes: str | None,
    data_root: str,
    output_dir: str,
    device: str | None,
    batch_frames: int,
) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    import numpy as np
    import torch

    from nxwm_mira.codec.codec_model import VideoCodec
    from nxwm_mira.data.za_dataset import (
        _decode_clip,
        load_actions,
        load_episodes,
        pool_actions,
    )
    from nxwm_mira.training.checkpoints import resolve_checkpoint

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = resolve_checkpoint(checkpoint)
    codec = VideoCodec.load_from_checkpoint(ckpt, device=device)
    codec.eval()

    td = codec.temporal_downsampling
    fps = codec.config.encoder.video.fps
    stride = max(1, round(30 / fps))
    # Chunk length in sampled frames; must be a multiple of the temporal stride.
    chunk = max(td, (batch_frames // td) * td)

    all_episodes = load_episodes(data_root)
    actions_by_episode = load_actions(data_root)
    if episodes is not None:
        wanted = {int(x) for x in episodes.split(",")}
        all_episodes = [ep for ep in all_episodes if ep.episode_index in wanted]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = codec.info_from_checkpoint or {}
    (out / "meta.json").write_text(
        json.dumps(
            {
                "codec_checkpoint": str(ckpt),
                "latent_mean_std": info.get("latent_mean_std"),
                "frame_stride": stride,
                "latent_fps": fps / td,
                "video_fps": fps,
                "aspect_mode": codec.config.encoder.aspect_mode,
                "height": codec.config.encoder.video.height,
                "width": codec.config.encoder.video.width,
            },
            indent=2,
        )
    )

    def autocast():
        if device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        import contextlib

        return contextlib.nullcontext()

    from nxwm_mira.codec.codec_model import preprocess_video

    n_done, t0 = 0, time.time()
    for ep in all_episodes:
        n_sampled = ep.frame_count // stride
        n_sampled -= n_sampled % td  # even multiple of the temporal stride
        if n_sampled < td:
            print(f"skip episode {ep.episode_index} ({ep.frame_count} frames, too short)")
            continue
        dest = out / f"episode_{ep.episode_index:06d}.npz"
        if dest.exists():
            n_done += 1
            continue

        indices = list(range(0, n_sampled * stride, stride))
        latent_chunks = []
        with torch.no_grad(), autocast():
            for c0 in range(0, n_sampled, chunk):
                chunk_indices = indices[c0 : c0 + chunk]
                video = _decode_clip(ep.video_path, chunk_indices)[None].to(device)
                video = preprocess_video(
                    video,
                    target_h=codec.config.encoder.video.height,
                    target_w=codec.config.encoder.video.width,
                    aspect_mode=codec.config.encoder.aspect_mode,
                )
                _, enc = codec.encode(video, trim_video=False)
                latent_chunks.append(enc.z[0].float().cpu())

        latents = torch.cat(latent_chunks, dim=0).to(torch.float16).numpy()
        actions = pool_actions(actions_by_episode[ep.episode_index], indices, stride).numpy()
        np.savez_compressed(dest, latents=latents, actions=actions)
        n_done += 1
        print(
            f"[{n_done}/{len(all_episodes)}] episode {ep.episode_index}: "
            f"{latents.shape} latents, {actions.shape} actions "
            f"({time.time() - t0:.0f}s elapsed)"
        )

    print(f"Done: {n_done} episodes -> {out}")
