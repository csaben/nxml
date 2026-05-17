"""DDP launcher for nxwm training.

Reads a typed ``TrainingConfig`` and architecture config from YAML; saves
self-describing checkpoints via ``nxml_core.save_checkpoint``. Uses
``mp.spawn`` for single-box multi-GPU; single-GPU fallback runs without
DDP for development.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nxwm.architectures import dit_v1, fa_dit  # noqa: F401  (registers architectures)
from nxwm.core import architecture_registry
from nxwm.training.config import TrainingConfig
from nxwm.training.data_resolver import resolve_data_paths
from nxwm.training.dataset import LatentEpisodeDataset
from nxwm.training.trainer import WorldModelTrainer


def next_run_dir(base: Path = Path("checkpoints/wm")) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("run_*"))
    next_num = 1
    if existing:
        try:
            next_num = int(existing[-1].name.split("_")[1]) + 1
        except (IndexError, ValueError):
            next_num = len(existing) + 1
    run_dir = base / f"run_{next_num:03d}"
    run_dir.mkdir()
    return run_dir


def _setup_ddp(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _filter_to_dataclass(cls, raw: dict[str, Any]) -> dict[str, Any]:
    keys = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in keys}


def _build_dataloaders(
    args: dict[str, Any],
    rank: int,
    world_size: int,
    *,
    future_anchor_len: int = 0,
    plan_horizon: int = 0,
):
    train_files, val_files = resolve_data_paths(
        args["data"]["data_paths"], args["data"].get("val_files")
    )
    sequence_length = args["data"].get("sequence_length", 10)
    goal_offset = args["data"].get("goal_offset", 30)
    unroll_steps = args["training"].get("unroll_steps", 1)

    ds_kwargs = dict(
        sequence_length=sequence_length,
        unroll_steps=unroll_steps,
        goal_offset=goal_offset,
        future_anchor_len=future_anchor_len,
        plan_horizon=plan_horizon,
    )
    train_ds = LatentEpisodeDataset(train_files, **ds_kwargs)
    val_ds = LatentEpisodeDataset(val_files or train_files, **ds_kwargs)

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=42
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    bs = args["data"].get("batch_size_per_gpu", 32)
    num_workers = args["data"].get("num_workers", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


def main_worker(rank: int, world_size: int, raw_config: dict[str, Any], run_dir: Path) -> None:
    if world_size > 1:
        _setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main_node = rank == 0

    arch_name = raw_config["architecture"]["name"]
    arch_raw_cfg = raw_config["architecture"].get("config", {})
    config_cls = architecture_registry.get_config(arch_name)
    model_cls = architecture_registry.get(arch_name)
    arch_cfg = config_cls(**_filter_to_dataclass(config_cls, arch_raw_cfg))
    raw_model = model_cls(arch_cfg).to(device)

    checkpoint = None
    start_epoch = 0
    resume_from = raw_config["training"].get("resume_from")
    if resume_from:
        ckpt_path = Path(resume_from)
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = checkpoint.get("state_dict") or checkpoint["model_state_dict"]
            raw_model.load_state_dict(sd)
            start_epoch = (
                checkpoint.get("training_metadata", {}).get("epoch", checkpoint.get("epoch", 0))
                + 1
            )
            if is_main_node:
                print(f"Loaded weights from {ckpt_path} (resuming at epoch {start_epoch})")
        elif is_main_node:
            print(f"Warning: checkpoint {ckpt_path} not found, training from scratch")

    if world_size > 1:
        model = DDP(raw_model, device_ids=[rank])
    else:
        model = raw_model

    train_loader, val_loader, train_sampler = _build_dataloaders(
        raw_config,
        rank,
        world_size,
        future_anchor_len=getattr(arch_cfg, "future_anchor_len", 0),
        plan_horizon=getattr(arch_cfg, "plan_horizon", 0),
    )

    tcfg_kwargs = _filter_to_dataclass(TrainingConfig, raw_config["training"])
    tcfg_kwargs["run_dir"] = run_dir
    tcfg = TrainingConfig(**tcfg_kwargs)

    vae = None
    lpips_fn = None
    needs_vae = tcfg.lpips_weight > 0 or tcfg.eval_gif_every_n_epochs > 0
    needs_lpips = tcfg.lpips_weight > 0
    if needs_vae:
        import warnings

        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

        vae_path = os.getenv("vae_path", "stabilityai/sd-vae-ft-mse")
        # See nxwm/inference/vae.py for the rationale on this filter.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*local_dir_use_symlinks.*",
                category=UserWarning,
            )
            vae = AutoencoderKL.from_pretrained(vae_path).to(device)  # type: ignore[no-untyped-call]
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
    if needs_lpips:
        import lpips

        lpips_fn = lpips.LPIPS(net="alex").to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    trainer = WorldModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=tcfg,
        architecture=arch_name,
        is_main_node=is_main_node,
        checkpoint=checkpoint,
        vae=vae,
        lpips_fn=lpips_fn,
    )

    epochs = raw_config["data"].get("epochs", raw_config["training"].get("epochs", 1))
    for epoch in range(start_epoch, start_epoch + epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        avg_loss, last_train_batch = trainer.train_epoch(epoch, start_epoch=start_epoch)
        trainer.val_epoch(epoch, train_batch=last_train_batch)
        if is_main_node:
            trainer.save_checkpoint(epoch, avg_loss)

    if world_size > 1:
        dist.destroy_process_group()


def launch(
    config_path: Path,
    world_size: int | None = None,
    *,
    resume_override: Path | str | None = None,
) -> None:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if resume_override is not None:
        raw.setdefault("training", {})["resume_from"] = str(resume_override)

    run_dir_override = raw["training"].get("run_dir")
    run_dir = Path(run_dir_override) if run_dir_override else next_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_copy = run_dir / config_path.name
    if config_path.resolve() != config_copy.resolve():
        shutil.copy2(config_path, config_copy)
    print(f"Saving checkpoints to: {run_dir}")

    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size <= 1:
        main_worker(0, 1, raw, run_dir)
    else:
        mp.spawn(  # type: ignore[attr-defined]
            main_worker,
            args=(world_size, raw, run_dir),
            nprocs=world_size,
            join=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="nxwm DDP training launcher")
    ap.add_argument("config", type=Path, help="YAML training config")
    ap.add_argument("--world-size", type=int, default=None, help="override world_size")
    args = ap.parse_args()
    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    launch(args.config, world_size=args.world_size)


if __name__ == "__main__":
    main()
