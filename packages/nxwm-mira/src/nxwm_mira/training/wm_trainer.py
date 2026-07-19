"""Train the latent world model — a port of MIRA's ``scripts/train_world_model.py``.

Same skeleton as :mod:`nxwm_mira.training.trainer` (the codec trainer): distributed setup,
bf16 autocast, AdamW + warmup/cosine, model EMA via CheckpointManager, resume precedence,
live status files. Differences: the model is a :class:`LatentWorldModel` (frozen codec
inside), the loss is the model's own flow-matching forward, and the viz cadence renders
autoregressive replay rollouts (recorded actions, generated frames) instead of codec
reconstructions.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import tqdm
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from nxwm_mira.data.actions import SwitchActions
from nxwm_mira.data.batch import ClipMeta, VideoActionBatch
from nxwm_mira.data.fake_dataset import FakeClipDataset
from nxwm_mira.data.za_dataset import (
    ZAClipDataset,
    build_loader,
    collate_action_clips,
    load_actions,
    load_episodes,
    split_episodes,
)
from nxwm_mira.training.checkpoint_manager import CheckpointManager
from nxwm_mira.training.checkpoints import resume_wandb_run_id
from nxwm_mira.training.distributed import get_distributed_settings, set_up_distributed
from nxwm_mira.training.live_status import LiveStatusWriter, prune_recons
from nxwm_mira.training.lr_schedule import WarmupConstantCosineDecayLR
from nxwm_mira.training.metrics import DistributedMetric
from nxwm_mira.training.tracker import TrainingTracker, display_execution_time, periodic_event
from nxwm_mira.training.wandb_media import save_gif
from nxwm_mira.training.wm_config import WMTrainConfig
from nxwm_mira.world_model.config import WorldModelInferenceConfig
from nxwm_mira.world_model.latent_world_model import LatentWorldModel

logger = logging.getLogger(__name__)


def _autocast(device: int | str | torch.device):
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _infinite(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def run_training(cfg: WMTrainConfig) -> None:
    assert cfg.run.output_dir is not None, "run.output_dir must be resolved by the caller"
    output_dir = Path(cfg.run.output_dir)

    distributed_settings = set_up_distributed()
    is_main_process = distributed_settings.is_main_process
    device = distributed_settings.device

    torch.manual_seed(cfg.run.seed + distributed_settings.rank)
    logging.basicConfig(format="%(message)s", datefmt="[%X]", level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO if is_main_process else logging.ERROR)

    logger.info("=" * 60)
    logger.info("World-model training configuration:")
    logger.info(yaml.safe_dump(cfg.model_dump(), sort_keys=False))

    run_name = f"{output_dir.name}-{datetime.now().strftime('%y%m%d-%H%M')}"
    status = LiveStatusWriter(output_dir, total_steps=cfg.run.steps, run_name=run_name)

    wandb_run = None
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / LatentWorldModel.CONFIG_FILENAME).write_text(
            yaml.safe_dump({"model": {"config": cfg.model.config.model_dump()}})
        )
        wandb_run = _init_wandb(cfg, output_dir, run_name)
        if wandb_run is not None:
            status.wandb_url = wandb_run.url

    raw_model = LatentWorldModel(cfg.model.config)
    raw_model.train().to(device)
    is_distributed = dist.is_available() and dist.is_initialized()
    model = DistributedDataParallel(raw_model, device_ids=[device]) if is_distributed else raw_model

    if cfg.run.compile:
        model.compile()

    if is_main_process:
        n_trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad) / 1e6
        n_frozen = sum(p.numel() for p in raw_model.parameters() if not p.requires_grad) / 1e6
        logger.info(
            f"Initialized world model: {n_trainable:.1f}M trainable + {n_frozen:.1f}M frozen codec"
        )

    train_loader, val_loader, val_episode_indices = _create_dataloaders(cfg)
    status.val_episodes = val_episode_indices

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=torch.tensor(cfg.optim.optimizer.lr),
        betas=cfg.optim.optimizer.betas,
        weight_decay=cfg.optim.optimizer.weight_decay,
    )
    lr_scheduler = WarmupConstantCosineDecayLR(
        optimizer,
        warmup_steps=cfg.optim.scheduler.warmup_steps,
        constant_steps=cfg.optim.scheduler.constant_steps,
        decay_steps=cfg.optim.scheduler.decay_steps,
        min_lr=cfg.optim.scheduler.min_lr,
    )

    @torch.compile(disable=not cfg.run.compile)
    def optimizer_step() -> None:
        optimizer.step()
        lr_scheduler.step()

    iter_train_loader = _infinite(train_loader)
    iter_val_loader = _infinite(val_loader)
    with display_execution_time("Warming up dataloader", print_output=is_main_process):
        next(iter_train_loader)
        viz_batch, viz_meta = next(iter_val_loader)

    training_tracker = TrainingTracker(
        world_size=distributed_settings.world_size, device=device, total_steps=cfg.run.steps
    )
    checkpoint_manager = CheckpointManager(
        raw_model,
        checkpoint_dir=str(output_dir),
        save_every=cfg.run.checkpoint_every,
        keep_recent=cfg.run.checkpoint_keep_recent,
        keep_permanent_every=cfg.run.checkpoint_keep_permanent_every,
        total_steps=cfg.run.steps,
        model_ema_decay=cfg.optim.model_ema_decay,
    )
    checkpoint_manager.register({"optimizer": optimizer, "lr_scheduler": lr_scheduler})

    start_step = _resume(cfg, checkpoint_manager)

    losses: dict[str, torch.Tensor] = {}
    iter_num = start_step - 1
    for iter_num in range(start_step, cfg.run.steps):
        step_start_time = time.monotonic()

        batch, _ = next(iter_train_loader)
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            losses = model(batch)

        losses["loss_total"].backward()
        optimizer_step()
        checkpoint_manager.model_ema.step()

        training_tracker.on_batch_processed(batch, losses)

        early_logging_steps = 10
        if periodic_event(iter_num, cfg.run.log_every, cfg.run.steps) or iter_num < early_logging_steps:
            stats = training_tracker.get_stats(step=iter_num)
            stats["train/learning_rate"] = float(optimizer.param_groups[0]["lr"])
            if is_main_process:
                stats["System/step_ms"] = (time.monotonic() - step_start_time) * 1000
                logger.info(f"Step {iter_num}: total loss {stats['train/loss_total']:.4f}")
                _wandb_log(wandb_run, stats, step=iter_num)
                status.log(iter_num, stats)

        if is_main_process and periodic_event(iter_num, cfg.run.viz_every, cfg.run.steps):
            with checkpoint_manager.model_ema.average_parameters():
                _visualize(cfg, raw_model, viz_batch, viz_meta, device, iter_num, wandb_run, status)

        if periodic_event(
            iter_num, cfg.validation.val_every, cfg.run.steps, include_0=cfg.validation.val_first
        ):
            with checkpoint_manager.model_ema.average_parameters():
                run_validation(cfg, device, raw_model, iter_val_loader, iter_num, wandb_run, status)

        if periodic_event(iter_num, cfg.run.checkpoint_every, cfg.run.steps, include_0=False):
            if is_main_process:
                checkpoint_manager.maybe_save_checkpoint(
                    iter_num, extra_data=_extra_data(iter_num, losses)
                )
            if is_distributed:
                dist.barrier()

        if is_distributed:
            dist.barrier()

    if is_main_process and iter_num >= start_step:
        checkpoint_manager.maybe_save_checkpoint(
            iter_num, extra_data=_extra_data(iter_num, losses), final=True
        )
    if wandb_run is not None:
        wandb_run.finish()
    logger.info("Done training")


def _extra_data(iter_num: int, losses: dict[str, torch.Tensor]) -> dict:
    extra_data = {k: v.item() for k, v in losses.items()}
    extra_data["iter_num"] = iter_num
    return extra_data


def _init_wandb(cfg: WMTrainConfig, output_dir: Path, run_name: str):
    if cfg.wandb.mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed (uv sync --extra training); skipping wandb logging")
        return None

    run_id = resume_wandb_run_id(cfg.run.continue_from, output_dir)
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name or run_name,
        group=cfg.wandb.group,
        config=cfg.model_dump(),
        dir=str(output_dir),
        mode=cfg.wandb.mode,
        id=run_id,
        resume="allow" if run_id else None,
    )
    (output_dir / "wandb_run_id.txt").write_text(run.id)
    return run


def _wandb_log(wandb_run, stats: dict, step: int) -> None:
    if wandb_run is not None:
        wandb_run.log(stats, step=step)


def _resume(cfg: WMTrainConfig, checkpoint_manager: CheckpointManager) -> int:
    if cfg.run.continue_from and cfg.run.finetune_from:
        raise ValueError("Set at most one of run.continue_from and run.finetune_from")
    if cfg.run.continue_from:
        return checkpoint_manager.continue_from(cfg.run.continue_from)
    if cfg.run.finetune_from:
        checkpoint_manager.finetune_from(cfg.run.finetune_from)
        return 0
    if checkpoint_manager.latest_checkpoint is not None:
        logger.info(f"Auto-resuming from existing checkpoint in {cfg.run.output_dir}")
        return checkpoint_manager.continue_from(checkpoint_manager.latest_checkpoint)
    return 0


def _create_dataloaders(cfg: WMTrainConfig) -> tuple[DataLoader, DataLoader, list[int] | None]:
    clip_len = cfg.model.config.video.timesteps
    val_batch_size = cfg.validation.batch_size or cfg.run.batch_size

    if cfg.data.source == "fake":
        height = cfg.model.config.video.height
        width = cfg.model.config.video.width
        train_ds = FakeClipDataset(
            n_clips=cfg.data.fake_n_clips, clip_len=clip_len, height=height, width=width,
            with_actions=True,
        )
        val_ds = FakeClipDataset(
            n_clips=max(4, cfg.data.fake_n_clips // 4), clip_len=clip_len, height=height,
            width=width, seed=1337, with_actions=True,
        )
        val_indices = None
    else:
        stride = cfg.frame_stride
        episodes = load_episodes(cfg.data.root)
        actions = load_actions(cfg.data.root)
        train_eps, val_eps = split_episodes(
            episodes,
            min_frames=clip_len * stride + 1,
            holdout_per_folder=cfg.data.holdout_per_folder,
            val_episodes=cfg.data.val_episodes,
        )
        train_ds = ZAClipDataset(
            train_eps, clip_len=clip_len, frame_stride=stride,
            clip_spacing=cfg.data.clip_spacing, actions_by_episode=actions,
        )
        # Val windows are longer than training windows so the viz rollout has room to
        # generate many latent steps past the context. Validation loss is unaffected:
        # the training forward slices batches down to video.timesteps.
        val_clip_len = clip_len + cfg.viz.extra_rollout_frames
        val_ds = ZAClipDataset(
            val_eps, clip_len=val_clip_len, frame_stride=stride, actions_by_episode=actions
        )
        val_indices = sorted(ep.episode_index for ep in val_eps)
        logger.info(f"Dataset: {len(train_ds)} train clips / {len(val_ds)} val clips")

    train_loader = build_loader(
        train_ds, batch_size=cfg.run.batch_size, num_workers=cfg.data.num_workers,
        shuffle=True, seed=cfg.run.seed + get_distributed_settings().rank,
        collate_fn=collate_action_clips,
    )
    val_loader = build_loader(
        val_ds, batch_size=val_batch_size, num_workers=min(2, cfg.data.num_workers),
        shuffle=False, seed=37, collate_fn=collate_action_clips,
    )
    return train_loader, val_loader, val_indices


def run_validation(
    cfg: WMTrainConfig,
    device: torch.device | int | str,
    model: LatentWorldModel,
    iter_val_loader: Iterator[tuple],
    iter_num: int,
    wandb_run,
    status: LiveStatusWriter,
) -> None:
    t1 = time.time()
    distributed_settings = get_distributed_settings()
    is_distributed = dist.is_available() and dist.is_initialized()
    model.eval()
    world_size = distributed_settings.world_size
    val_batch_size = cfg.validation.batch_size or cfg.run.batch_size
    n_batches = cfg.validation.val_n_samples // (val_batch_size * world_size)

    metric_trackers: dict[str, DistributedMetric] = defaultdict(
        lambda: DistributedMetric(device=device)
    )
    for _ in tqdm.trange(
        max(1, n_batches), disable=not distributed_settings.is_main_process, desc="Running validation"
    ):
        batch, _ = next(iter_val_loader)
        batch = batch.to(device)
        with torch.no_grad(), _autocast(device):
            for k, v in model(batch).items():
                metric_trackers[k].update(v)

    metrics = {k: tracker.compute_and_reset().item() for k, tracker in metric_trackers.items()}
    if distributed_settings.is_main_process:
        logger.info(f"Validation took {time.time() - t1:.2f}s")
        logger.info(
            f"Validation at step {iter_num}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        test_stats = {f"test/{k}": v for k, v in metrics.items()}
        _wandb_log(wandb_run, test_stats, step=iter_num)
        status.log(iter_num, test_stats)

    model.train()
    if is_distributed:
        dist.barrier()


def _visualize(
    cfg: WMTrainConfig,
    model: LatentWorldModel,
    viz_batch: VideoActionBatch,
    viz_meta: list[ClipMeta],
    device: torch.device | int | str,
    iter_num: int,
    wandb_run,
    status: LiveStatusWriter,
) -> None:
    """Autoregressive replay rollout on a fixed val batch: prediction over ground truth."""
    model.eval()
    n = min(cfg.viz.n_samples, len(viz_batch))
    batch = VideoActionBatch(
        video=viz_batch.video[:n].clone(),
        actions=SwitchActions(viz_batch.actions.actions[:n].clone()),
    ).to(device)
    with torch.no_grad(), _autocast(device):
        outputs = model.inference(
            batch,
            config=WorldModelInferenceConfig(n_diffusion_steps=cfg.viz.n_diffusion_steps),
            progress_bar=False,
        )
        viz = model.visualize(outputs)
    model.train()

    combined = torch.cat(list(viz["viz_video"].cpu()), dim=-2)  # stack samples vertically

    output_dir = Path(cfg.run.output_dir or ".")
    recons_dir = output_dir / "recons"
    fps = cfg.model.config.video.fps
    gif_path = save_gif(combined, recons_dir / f"step_{iter_num:07d}.gif", fps=fps)
    latest_tmp = recons_dir / "latest.gif.tmp"
    latest_tmp.write_bytes(gif_path.read_bytes())
    latest_tmp.replace(recons_dir / "latest.gif")
    prune_recons(recons_dir, keep_every=cfg.run.viz_keep_every)

    if wandb_run is not None:
        import wandb

        wandb_run.log({"videos/rollout": wandb.Video(str(gif_path), format="gif")}, step=iter_num)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
