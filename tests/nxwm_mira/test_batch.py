"""VideoBatch container behavior."""

from __future__ import annotations

import torch

from nxwm_mira.data.batch import ClipMeta, VideoBatch

from .conftest import random_video


def test_len_and_video_shape() -> None:
    batch = VideoBatch(video=random_video(batch=3))
    assert len(batch) == 3
    assert batch.video.shape == (3, 4, 3, 64, 64)


def test_to_dtype_returns_new_batch() -> None:
    batch = VideoBatch(video=random_video())
    moved = batch.to(torch.float32)
    assert moved.video.dtype == torch.float32
    assert batch.video.dtype == torch.uint8  # original untouched


def test_clone_is_deep() -> None:
    batch = VideoBatch(video=random_video())
    cloned = batch.clone()
    cloned.video[:] = 0
    assert batch.video.abs().sum() > 0


def test_clip_meta_caption() -> None:
    meta = ClipMeta(episode_index=7, episode_id="episode_000007", start_frame=120, frame_stride=2)
    assert meta.caption == "ep7@120"
