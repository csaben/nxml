"""DDP launcher for nxrl training.

Reads a YAML with ``policy``, ``algorithm``, optional ``run`` plus the
algorithm-specific blocks (``data`` for BC; ``world_model`` + ``bc_init`` +
``seeds`` + ``rollout`` + ``reward`` for PPO). Dispatches on
``algorithm.name`` to the matching worker.

BC reuses ``nxwm``-style DDP (``mp.spawn`` across visible GPUs).
PPO is single-process: DDP'ing the rollout collector is out of scope.
"""

from __future__ import annotations

import importlib
import os
import shutil
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from nxml_core.registry import Registry
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nxrl import algorithms as _algorithms  # noqa: F401  (registers BC + PPO)
from nxrl import policies as _policies  # noqa: F401  (registers BC + PPO archs)
from nxrl.core.registry import algorithm_registry, policy_registry
from nxrl.training.data_resolver import resolve_data_paths
from nxrl.training.dataset import LatentBCDataset


def next_run_dir(base: Path = Path("checkpoints/bc")) -> Path:
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
    os.environ.setdefault("MASTER_PORT", "12356")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _filter_to_dataclass(cls, raw: dict[str, Any]) -> dict[str, Any]:
    keys = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in keys}


def _instantiate_from_registry(reg: Registry, raw: dict[str, Any], registry_label: str):
    name = raw.get("name")
    if not name:
        raise ValueError(f"missing '{registry_label}.name' in config")
    cfg_cls = reg.get_config(name)
    cfg_kwargs = _filter_to_dataclass(cfg_cls, raw.get("config") or {})
    return name, cfg_cls(**cfg_kwargs), reg.get(name)


def _load_callable(spec: dict[str, Any] | None, default_factory: Callable | None = None):
    """Resolve a ``{callable: "module:fn", kwargs: {...}}`` config spec.

    The named function should be a *factory* that returns the callable used at
    runtime. This indirection lets configs pass kwargs at load-time without
    every callable having to support ``functools.partial``.
    """
    if not spec or not spec.get("callable"):
        if default_factory is None:
            raise ValueError("no callable specified and no default factory")
        return default_factory(**(spec.get("kwargs") if spec else {}) or {})
    module_name, fn_name = spec["callable"].split(":")
    mod = importlib.import_module(module_name)
    factory = getattr(mod, fn_name)
    return factory(**(spec.get("kwargs") or {}))


# ----------------------------------------------------------------------------
# BC worker (DDP-friendly)
# ----------------------------------------------------------------------------


def _build_dataloaders(
    raw: dict[str, Any], sequence_length: int, rank: int, world_size: int
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    data = raw["data"]
    train_files, val_files = resolve_data_paths(data["data_paths"], data.get("val_files"))
    align_starts = bool(data.get("align_starts", False))

    train_ds = LatentBCDataset(train_files, sequence_length=sequence_length, align_starts=align_starts)
    val_ds = LatentBCDataset(
        val_files or train_files, sequence_length=sequence_length, align_starts=align_starts
    )

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=42
        )
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    bs = data.get("batch_size_per_gpu", 32)
    num_workers = data.get("num_workers", 4)
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


def _bc_main_worker(rank: int, world_size: int, raw: dict[str, Any], run_dir: Path) -> None:
    if world_size > 1:
        _setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main_node = rank == 0

    policy_name, policy_cfg, policy_cls = _instantiate_from_registry(
        policy_registry, raw["policy"], "policy"
    )
    raw_model = policy_cls(policy_cfg).to(device)

    checkpoint = None
    start_epoch = 0
    resume_from = (raw.get("run") or {}).get("resume_from") or raw["policy"].get("resume_from")
    if resume_from:
        ckpt_path = Path(resume_from)
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
            if sd is None:
                raise ValueError(f"{ckpt_path} has no state_dict / model_state_dict")
            raw_model.load_state_dict(sd)
            start_epoch = (
                checkpoint.get("training_metadata", {}).get("epoch", checkpoint.get("epoch", 0)) + 1
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
        raw, policy_cfg.sequence_length, rank, world_size
    )
    if is_main_node and len(val_loader.dataset) == 0:
        print(
            f"[nxrl] WARNING: val dataset is empty "
            f"(sequence_length={policy_cfg.sequence_length}, "
            f"align_starts={bool(raw['data'].get('align_starts', False))}). "
            f"Likely the val .npz has fewer than sequence_length frames. "
            f"Val loss will be NaN every epoch."
        )

    algo_name, algo_cfg, algo_cls = _instantiate_from_registry(
        algorithm_registry, raw["algorithm"], "algorithm"
    )

    trainer = algo_cls(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=algo_cfg,
        policy_name=policy_name,
        policy_config=policy_cfg,
        is_main_node=is_main_node,
        run_dir=run_dir,
        checkpoint=checkpoint,
    )

    epochs = algo_cfg.epochs
    for epoch in range(start_epoch, start_epoch + epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        avg_loss = trainer.train_epoch(epoch)
        trainer.val_epoch(epoch)
        if is_main_node:
            trainer.save_checkpoint(epoch, avg_loss)

    if world_size > 1:
        dist.destroy_process_group()
    if is_main_node:
        print(f"[nxrl] training complete ({algo_name} on {policy_name})")


# ----------------------------------------------------------------------------
# PPO worker (single-process)
# ----------------------------------------------------------------------------


def _zero_reward_fn(*args: Any, **_kwargs: Any):
    """Default reward: always zero. Useful for smoke-testing the PPO pipeline
    before plugging in real rewards (e.g. ``nxml-games/pokemon_za``).
    """
    return 0.0, {}


def _make_zero_reward(**_kwargs: Any):
    return _zero_reward_fn


def _load_world_model(spec: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Load any registered nxwm world model from a self-describing checkpoint."""
    import nxwm.architectures  # noqa: F401  (registers DiT)
    from nxml_core.checkpoint import load_checkpoint
    from nxwm.core.registry import architecture_registry

    ckpt_path = Path(spec["ckpt_path"])
    model, _cfg, _ckpt = load_checkpoint(ckpt_path, architecture_registry, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _ppo_main_worker(
    rank: int, world_size: int, raw: dict[str, Any], run_dir: Path
) -> None:
    """PPO training worker.

    Signature mirrors ``_bc_main_worker`` so the top-level launcher can dispatch
    both algorithms through the same ``mp.spawn`` shape.

    DDP wiring lands across multiple steps of docs/ddp-ppo-design.md:
      - step 2 (here): process-group setup/teardown + per-rank device.
      - step 3: DDP-wrap the policy.
      - step 4: thread rank/world_size into the trainer.
      - step 5: all-gather rollout buffers across ranks.
      - step 6: launcher routes world_size>1 through mp.spawn.
    Until step 6 lands, the launcher still pins (0, 1, ...) so the
    ``world_size > 1`` branch below is dead code in production paths.
    """
    if world_size > 1:
        _setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main_node = rank == 0
    tag = f"PPO rank={rank}/{world_size}" if world_size > 1 else "PPO single-process"
    print(f"[nxrl] {tag} on {device}")

    # 1. Build PPO policy from registry.
    policy_name, policy_cfg, policy_cls = _instantiate_from_registry(
        policy_registry, raw["policy"], "policy"
    )
    policy = policy_cls(policy_cfg).to(device)

    # 2. Either resume full PPO checkpoint, or seed BC weights from a BC checkpoint.
    resume_from = (raw.get("run") or {}).get("resume_from")
    if resume_from:
        ckpt = torch.load(Path(resume_from), map_location=device, weights_only=False)
        sd = ckpt.get("state_dict") or ckpt.get("model_state_dict")
        if sd is None:
            raise ValueError(f"{resume_from} has no state_dict")
        policy.load_state_dict(sd)
        print(f"[nxrl] resumed full PPO state from {resume_from}")
    elif raw.get("bc_init"):
        bc_path = Path(raw["bc_init"]["ckpt_path"])
        bc_ckpt = torch.load(bc_path, map_location=device, weights_only=False)
        bc_sd = bc_ckpt.get("state_dict") or bc_ckpt.get("model_state_dict")
        if bc_sd is None:
            raise ValueError(f"{bc_path} has no state_dict")
        policy.load_bc_state_dict(bc_sd)
        print(f"[nxrl] seeded BC weights from {bc_path}")

    # 3. Frozen BC reference (for the BC anchor in PPO loss). Build an
    # independent BC policy of the same arch and weight, freeze it.
    base_cls = policy_registry.get(policy_cfg.base_policy_name)
    base_cfg_cls = policy_registry.get_config(policy_cfg.base_policy_name)
    frozen_base = base_cls(base_cfg_cls(**policy_cfg.base_policy_config)).to(device)
    if raw.get("bc_init"):
        bc_path = Path(raw["bc_init"]["ckpt_path"])
        bc_ckpt = torch.load(bc_path, map_location=device, weights_only=False)
        bc_sd = bc_ckpt.get("state_dict") or bc_ckpt.get("model_state_dict")
        if bc_sd is not None:
            frozen_base.load_state_dict(bc_sd)
    frozen_base.eval()
    for p in frozen_base.parameters():
        p.requires_grad_(False)

    # DDP-wrap the policy so backward syncs gradients across ranks. Frozen
    # base stays bare (no_grad path; no gradient sync needed). The trainer
    # uses `policy` for forward calls during `_ppo_update` (DDP-aware) and
    # `policy_module` for custom-method/attribute access (which DDP doesn't
    # delegate). When world_size == 1, `policy` and `policy_module` are the
    # same object — single-GPU behavior is unchanged.
    policy_module = policy
    if world_size > 1:
        policy = DDP(policy, device_ids=[rank])

    # 4. World model.
    world_model = _load_world_model(raw["world_model"], device)

    # 5. Algorithm config.
    algo_name, algo_cfg, algo_cls = _instantiate_from_registry(
        algorithm_registry, raw["algorithm"], "algorithm"
    )
    if algo_name != "ppo":
        raise ValueError(f"_ppo_main_worker called with non-PPO algorithm {algo_name!r}")

    # DDP arithmetic: rollouts_per_update is the GLOBAL count across all ranks.
    # Reject configs where it isn't evenly divisible so we don't silently
    # under/over-train. Single-GPU (world_size=1) trivially passes.
    if algo_cfg.rollouts_per_update % world_size != 0:
        raise ValueError(
            f"rollouts_per_update ({algo_cfg.rollouts_per_update}) must be "
            f"divisible by world_size ({world_size}). Either bump it to a "
            f"multiple or change --world-size."
        )
    local_rollouts_per_update = algo_cfg.rollouts_per_update // world_size

    # 6. Seeds + rollout spec.
    from nxrl.algorithms.ppo import RolloutSpec, SeedSpec

    seed_specs = [SeedSpec(**s) for s in (raw.get("seeds") or [])]
    if not seed_specs:
        raise ValueError("PPO config requires at least one seed in 'seeds'")
    rollout_spec = RolloutSpec(**(raw.get("rollout") or {}))

    # 7. Reward function (default: zero reward — useful for smoke tests).
    reward_fn = _load_callable(raw.get("reward"), default_factory=_make_zero_reward)

    # Eval-mkv block: every N updates, render one rollout per listed seed
    # to mkv + upload to wandb. Disabled when every_n_updates is 0/missing.
    run_block = raw.get("run") or {}
    eval_mkv_config = {
        "every_n_updates": int(run_block.get("eval_mkv_every_n_updates", 0)),
        "seed_indices": list(run_block.get("eval_mkv_seed_indices") or [0]),
        "fps": int(run_block.get("eval_mkv_fps", 30)),
    }

    trainer = algo_cls(
        policy=policy,
        policy_module=policy_module,
        frozen_policy=frozen_base,
        world_model=world_model,
        config=algo_cfg,
        rollout_spec=rollout_spec,
        seeds=seed_specs,
        reward_fn=reward_fn,
        device=device,
        policy_name=policy_name,
        policy_config=policy_cfg,
        is_main_node=is_main_node,
        run_dir=run_dir,
        eval_mkv_config=eval_mkv_config,
        rank=rank,
        world_size=world_size,
        local_rollouts_per_update=local_rollouts_per_update,
    )

    checkpoint_every = (raw.get("run") or {}).get("checkpoint_frequency", 50)
    snapshot_every = (raw.get("run") or {}).get("snapshot_frequency", 0)
    for update in range(algo_cfg.total_updates):
        metrics = trainer.update_step()
        if (update + 1) % checkpoint_every == 0:
            trainer.save_checkpoint(name="latest.pt", metadata={"reward": metrics.get("rollout/mean_reward")})
        if snapshot_every > 0 and (update + 1) % snapshot_every == 0:
            trainer.save_checkpoint(
                name=f"update_{update + 1:04d}.pt",
                metadata={"reward": metrics.get("rollout/mean_reward")},
            )
    trainer.save_checkpoint(name="final.pt")
    if world_size > 1:
        dist.destroy_process_group()
    if is_main_node:
        print(f"[nxrl] PPO training complete ({policy_name})")


# ----------------------------------------------------------------------------
# Top-level launch
# ----------------------------------------------------------------------------


def launch(
    config_path: Path,
    *,
    world_size: int | None = None,
    resume_override: Path | str | None = None,
) -> None:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if resume_override is not None:
        raw.setdefault("run", {})["resume_from"] = str(resume_override)

    algo_name = (raw.get("algorithm") or {}).get("name", "")
    default_base = Path("checkpoints/ppo") if algo_name == "ppo" else Path("checkpoints/bc")

    run_dir_override = (raw.get("run") or {}).get("run_dir")
    run_dir = Path(run_dir_override) if run_dir_override else next_run_dir(default_base)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_copy = run_dir / config_path.name
    if config_path.resolve() != config_copy.resolve():
        shutil.copy2(config_path, config_copy)
    print(f"Saving checkpoints to: {run_dir}")

    if algo_name == "ppo":
        if world_size is None:
            world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if world_size <= 1:
            _ppo_main_worker(0, 1, raw, run_dir)
        else:
            mp.spawn(  # type: ignore[attr-defined]
                _ppo_main_worker,
                args=(world_size, raw, run_dir),
                nprocs=world_size,
                join=True,
            )
        return

    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size <= 1:
        _bc_main_worker(0, 1, raw, run_dir)
    else:
        mp.spawn(  # type: ignore[attr-defined]
            _bc_main_worker, args=(world_size, raw, run_dir), nprocs=world_size, join=True
        )
