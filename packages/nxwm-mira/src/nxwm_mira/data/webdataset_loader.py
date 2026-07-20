"""Training DataLoader construction for pinned immutable WebDataset publications."""

from __future__ import annotations

from pathlib import Path

import torch
from nxml_core.contracts.webdataset_v1 import WebDatasetSnapshotV1
from torch.utils.data import DataLoader

from nxwm_mira.data.batch import VideoBatch
from nxwm_mira.data.webdataset import BoundedShardCache, hub_fetcher
from nxwm_mira.data.webdataset_clips import WebDatasetClipDataset
from nxwm_mira.data.za_dataset import collate_action_clips


def _codec_collate(items):
    videos = torch.stack([video for video, _actions, _meta in items])
    metas = [meta for _video, _actions, meta in items]
    return VideoBatch(video=videos), metas


def create_webdataset_loader(
    cfg,
    *,
    split: str,
    clip_len: int,
    batch_size: int,
    with_actions: bool,
    shuffle_shards: bool,
    seed: int,
):
    required = {
        "webdataset_repo_id": cfg.data.webdataset_repo_id,
        "webdataset_revision": cfg.data.webdataset_revision,
        "webdataset_manifest": cfg.data.webdataset_manifest,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"webdataset source requires pinned fields: {missing}")
    from huggingface_hub import hf_hub_download

    manifest_path = Path(
        hf_hub_download(
            required["webdataset_repo_id"],
            required["webdataset_manifest"],
            repo_type="dataset",
            revision=required["webdataset_revision"],
        )
    )
    manifest = WebDatasetSnapshotV1.model_validate_json(manifest_path.read_text())
    from nxwm_mira.training.distributed import get_distributed_settings

    distributed = get_distributed_settings()
    dataset = WebDatasetClipDataset(
        manifest,
        cache=BoundedShardCache(
            cfg.data.webdataset_cache, max_bytes=cfg.data.webdataset_cache_bytes
        ),
        fetch=hub_fetcher(
            repo_id=required["webdataset_repo_id"],
            revision=required["webdataset_revision"],
            xet_cache=cfg.data.hf_xet_cache,
        ),
        split=split,
        clip_len=clip_len,
        frame_stride=cfg.frame_stride,
        rank=distributed.rank,
        world_size=distributed.world_size,
        seed=seed,
        shuffle_shards=shuffle_shards,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_action_clips if with_actions else _codec_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=4 if cfg.data.num_workers > 0 else None,
        drop_last=shuffle_shards,
    )
