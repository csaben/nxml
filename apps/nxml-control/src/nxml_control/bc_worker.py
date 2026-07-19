"""Concrete Pokemon ZA raw WebDataset -> latent BC worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tarfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from nxml_core.contracts import DaggerActionRecordV2
from pydantic import BaseModel, ConfigDict, Field

from nxml_control.catalog import Catalog
from nxml_control.storage import LocalObjectStorage

ACTION_SPEC_ID = "switch_packets.v1"
ACTION_DIM = 26
WORKER_SCHEMA = "nxml.pokemon-za-bc-worker.v1"
VAE_ID = "stabilityai/sd-vae-ft-mse"
VAE_PROFILE = "sd-vae-ft-mse.rgb-bilinear-128x256.mode.scale-0.18215.v1"


class BootstrapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: str = Field(default="pokemon-za-bootstrap-v1", pattern="^pokemon-za-bootstrap-v1$")
    epochs: int = Field(default=25, ge=1, le=500)
    sequence_length: int = Field(default=32, ge=8, le=300)
    batch_size: int = Field(default=8, ge=1, le=64)
    num_workers: int = Field(default=4, ge=0, le=16)
    learning_rate: float = Field(default=1.0e-4, gt=0, le=0.01)
    validation_fraction: float = Field(default=0.1, ge=0.05, le=0.25)
    vae_path: str = VAE_ID
    encode_batch_size: int = Field(default=24, ge=1, le=128)


@dataclass(frozen=True)
class PreparedData:
    train_files: list[str]
    val_files: list[str]
    train_frames: int
    val_frames: int
    episodes: list[str]
    source_members: list[dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_name(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe tar member path: {name}")


def _episode_members(shard_manifest: dict, episode_id: str) -> tuple[dict, dict]:
    action = None
    video = None
    for member in shard_manifest["members"]:
        _safe_member_name(member["path"])
        name = Path(member["path"]).name
        if member["kind"] == "actions" and name == f"{episode_id}.parquet":
            action = member
        if member["kind"] == "video" and name in {
            f"{episode_id}.mkv",
            f"{episode_id}.mp4",
        }:
            video = member
    if action is None or video is None:
        raise ValueError(f"episode {episode_id} lacks exact video/actions members")
    return video, action


def _extract_verified(archive: tarfile.TarFile, member: dict, destination: Path) -> Path:
    info = archive.getmember(member["path"])
    if not info.isfile() or info.size != member["size_bytes"]:
        raise ValueError(f"member size mismatch: {member['path']}")
    source = archive.extractfile(info)
    if source is None:
        raise ValueError(f"member unreadable: {member['path']}")
    output = destination / Path(member["path"]).name
    digest = hashlib.sha256()
    with output.open("xb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != member["sha256"]:
        output.unlink(missing_ok=True)
        raise ValueError(f"member checksum mismatch: {member['path']}")
    return output


def decode_action_rows(
    parquet_path: Path, *, control_source: str
) -> tuple[list[int], np.ndarray, int]:
    """Validate the emitted edge-v2 row sequence, then select causal applied actions."""
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet_path).to_pylist()
    frame_indices: list[int] = []
    actions: list[list[float]] = []
    previous_frame_ns = -1
    for expected_index, row in enumerate(rows):
        is_dagger = row.get("row_schema_id") == "nxml.dagger-actions.v2" or "policy_digest" in row
        if is_dagger:
            parsed = DaggerActionRecordV2.model_validate(row)
            frame_index = parsed.effective_frame_index
            frame_ns = parsed.effective_frame_ns
            action_ns = parsed.effective_action_ns
            action_age = (
                parsed.action_age_ns / 1_000_000_000
                if parsed.action_age_ns is not None
                else -1
            )
            ownership = [int(value) for value in parsed.ownership]
            human_mask = parsed.human_mask or parsed.human_action_mask or []
            row_valid = parsed.valid and parsed.applied_action_valid is not False
            row_eligible = parsed.bc_training_eligible
            applied_values = parsed.applied_action
        else:
            frame_index = int(row.get("frame_idx", -1))
            frame_ns = int(row.get("frame_monotonic_ns", -1))
            action_ns = int(row.get("action_monotonic_ns", -1))
            action_age = float(row.get("action_age", -1))
            ownership = [int(value) for value in row.get("ownership", ())]
            human_mask = [bool(value) for value in row.get("human_mask", ())]
            row_valid = bool(row.get("valid", False))
            row_eligible = True
            applied_values = row.get("applied_action", ())
        required = (
            "frame_idx",
            "frame_monotonic_ns",
            "action_monotonic_ns",
            "action_age",
            "applied_action",
            "human_mask",
            "ownership",
            "valid",
        )
        missing = [field for field in required if field not in row]
        if not is_dagger and missing:
            raise ValueError(f"edge-v2 action row lacks required fields: {missing}")
        if frame_index != expected_index:
            raise ValueError(
                "frame_idx must be a complete, ordered, duplicate-free sequence from zero"
            )
        if frame_ns is None or frame_ns <= previous_frame_ns:
            raise ValueError("frame_monotonic_ns must be strictly increasing")
        previous_frame_ns = frame_ns
        if is_dagger and (not row_valid or not row_eligible):
            continue
        if action_ns is None or action_ns > frame_ns:
            raise ValueError("noncausal action alignment")
        expected_age = (frame_ns - action_ns) / 1_000_000_000
        if not math.isclose(
            action_age, expected_age, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("action_age does not match monotonic timestamps")
        if not row_valid or not row_eligible:
            continue
        if len(ownership) != ACTION_DIM or len(human_mask) != ACTION_DIM:
            raise ValueError("ownership and human_mask must have 26 dimensions")
        if control_source == "human" and not (any(human_mask) or 1 in ownership):
            continue
        if control_source == "policy" and 2 not in ownership:
            continue
        if is_dagger and control_source == "policy":
            continue
        applied = [float(value) for value in applied_values]
        if len(applied) != ACTION_DIM:
            raise ValueError(f"applied_action must have {ACTION_DIM} dimensions")
        frame_indices.append(frame_index)
        actions.append(applied)
    if not actions:
        raise ValueError("episode has no eligible action rows")
    return frame_indices, np.asarray(actions, dtype=np.float32), len(rows)


def encode_selected_frames(
    video_path: Path,
    frame_indices: list[int],
    *,
    vae_path: str,
    device: str,
    batch_size: int,
    expected_frame_count: int,
) -> np.ndarray:
    """Decode RGB frames and reuse nxwm's canonical SD-VAE preprocessing."""
    import torch
    import torchvision.transforms.v2.functional as F
    from nxwm.inference.vae import LATENT_SCALE, load_vae
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video_path), device="cpu")
    if decoder.metadata.num_frames != expected_frame_count:
        raise ValueError(
            "media frame count does not match the complete edge-v2 action row sequence"
        )
    if not frame_indices or frame_indices[-1] >= decoder.metadata.num_frames:
        raise ValueError("action frame index exceeds decoded video")
    wanted = set(frame_indices)
    encoded: dict[int, np.ndarray] = {}
    dev = torch.device(device)
    vae = load_vae(vae_path, device=dev)
    with torch.no_grad():
        for start in range(0, decoder.metadata.num_frames, batch_size):
            stop = min(start + batch_size, decoder.metadata.num_frames)
            selected = [index for index in range(start, stop) if index in wanted]
            if not selected:
                continue
            frames = decoder.get_frames_at(selected).data
            frames = frames.pin_memory().to(dev, non_blocking=True)
            resized = F.resize(frames, [128, 256], antialias=True)
            normalized = (resized.to(torch.float16) / 255.0 - 0.5) / 0.5
            latents = vae.encode(normalized).latent_dist.mode() * LATENT_SCALE
            for index, latent in zip(selected, latents.cpu().numpy(), strict=True):
                encoded[index] = latent.astype(np.float16, copy=False)
    return np.stack([encoded[index] for index in frame_indices])


def _split_episode(
    latents: np.ndarray, actions: np.ndarray, *, sequence_length: int, val_fraction: float
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    minimum = sequence_length + 1
    if len(latents) < minimum * 2:
        raise ValueError(
            f"episode needs at least {minimum * 2} eligible frames for leakage-free split"
        )
    val_count = max(minimum, int(len(latents) * val_fraction))
    split = len(latents) - val_count
    if split < minimum:
        raise ValueError("training split is shorter than sequence length")
    return (latents[:split], actions[:split]), (latents[split:], actions[split:])


def prepare_snapshot(
    *,
    state_dir: Path,
    snapshot_id: str,
    workspace: Path,
    config: BootstrapConfig,
    encode_fn: Callable[..., np.ndarray] = encode_selected_frames,
    device: str = "cuda:0",
) -> PreparedData:
    catalog = Catalog(state_dir / "catalog.sqlite3")
    snapshot = catalog.get_snapshot(snapshot_id)
    if snapshot.control_source != "human":
        raise ValueError("pokemon-za-bootstrap-v1 requires a human-filtered snapshot")
    storage = LocalObjectStorage(state_dir / "objects")
    raw_dir = workspace / "raw"
    latent_dir = workspace / "latents"
    raw_dir.mkdir(parents=True)
    latent_dir.mkdir(parents=True)
    train_files: list[str] = []
    val_files: list[str] = []
    source_members: list[dict[str, Any]] = []
    train_frames = 0
    val_frames = 0
    episodes: list[str] = []
    for shard in snapshot.manifest["shards"]:
        info = storage.inspect(shard["object_key"])
        if info is None or info.sha256 != shard["sha256"] or info.size_bytes != shard["size_bytes"]:
            raise ValueError(f"cluster object verification failed: {shard['id']}")
        with (
            storage.open(shard["object_key"]) as stream,
            tarfile.open(fileobj=stream, mode="r:*") as archive,
        ):
            shard_record = next(
                item for item in catalog.list_shards(snapshot.dataset_id) if item.id == shard["id"]
            )
            for episode_id in shard["episode_ids"]:
                video_member, action_member = _episode_members(shard_record.manifest, episode_id)
                episode_dir = raw_dir / hashlib.sha256(episode_id.encode()).hexdigest()[:16]
                episode_dir.mkdir()
                video_path = _extract_verified(archive, video_member, episode_dir)
                action_path = _extract_verified(archive, action_member, episode_dir)
                indices, actions, row_count = decode_action_rows(
                    action_path, control_source="human"
                )
                latents = encode_fn(
                    video_path,
                    indices,
                    vae_path=config.vae_path,
                    device=device,
                    batch_size=config.encode_batch_size,
                    expected_frame_count=row_count,
                )
                if len(latents) != len(actions) or tuple(latents.shape[1:]) != (4, 16, 32):
                    raise ValueError(
                        "encoder output must align actions with latent shape (4,16,32)"
                    )
                train, val = _split_episode(
                    latents,
                    actions,
                    sequence_length=config.sequence_length,
                    val_fraction=config.validation_fraction,
                )
                train_path = latent_dir / f"{episode_id}.train.npz"
                val_path = latent_dir / f"{episode_id}.val.npz"
                np.savez(train_path, latents=train[0], actions=train[1].astype(np.float16))
                np.savez(val_path, latents=val[0], actions=val[1].astype(np.float16))
                train_files.append(str(train_path))
                val_files.append(str(val_path))
                train_frames += len(train[0])
                val_frames += len(val[0])
                episodes.append(episode_id)
                source_members.extend([video_member, action_member])
    if not train_files:
        raise ValueError("snapshot has no materializable eligible episodes")
    return PreparedData(
        train_files,
        val_files,
        train_frames,
        val_frames,
        episodes,
        source_members,
    )


def _training_config(prepared: PreparedData, config: BootstrapConfig, run_dir: Path) -> dict:
    return {
        "policy": {
            "name": "bc_transformer_v1",
            "config": {
                "sequence_length": config.sequence_length,
                "hidden_size": 256,
                "num_layers": 3,
                "num_heads": 8,
                "dropout": 0.2,
            },
        },
        "algorithm": {
            "name": "bc",
            "config": {
                "lr": config.learning_rate,
                "weight_decay": 0.05,
                "button_loss_weight": 1.0,
                "action_weight": 5.0,
                "stick_deadzone": 0.1,
                "grad_clip": 1.0,
                "epochs": config.epochs,
                "project_name": "nxrl-pokemon-za-bootstrap",
            },
        },
        "data": {
            "data_paths": prepared.train_files,
            "val_files": prepared.val_files,
            "batch_size_per_gpu": config.batch_size,
            "num_workers": config.num_workers,
            "align_starts": False,
        },
        "run": {"run_dir": str(run_dir)},
    }


def run_worker(
    request_path: Path, result_path: Path, state_dir: Path, checkpoint_dir: Path
) -> None:
    import torch
    import yaml

    request = json.loads(request_path.read_text())
    if request.get("schema_id") != "nxml.bc-job.v1":
        raise ValueError("request schema must be nxml.bc-job.v1")
    config = BootstrapConfig.model_validate(request.get("config") or {})
    job_id = request["job_id"]
    snapshot_id = request["snapshot_id"]
    workspace = request_path.parent / "worker"
    workspace.mkdir()
    prepared = prepare_snapshot(
        state_dir=state_dir,
        snapshot_id=snapshot_id,
        workspace=workspace,
        config=config,
    )
    run_dir = checkpoint_dir / job_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = workspace / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(_training_config(prepared, config, run_dir), sort_keys=True)
    )
    os.environ.setdefault("WANDB_MODE", "disabled")
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    from nxrl.training.launcher import launch

    launch(config_path, world_size=1)
    checkpoint = run_dir / "best_val.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "best.pt"
    if not checkpoint.is_file():
        raise RuntimeError("nxrl did not produce a policy checkpoint")
    import nxrl  # noqa: F401
    from nxml_core.checkpoint import load_checkpoint
    from nxrl.core.registry import policy_registry

    model, _policy_config, saved = load_checkpoint(checkpoint, policy_registry, device="cpu")
    smoke = model(torch.zeros(1, config.sequence_length, 4, 16, 32))
    if tuple(smoke.shape) != (1, ACTION_DIM) or not torch.isfinite(smoke).all():
        raise RuntimeError("edge compatibility smoke inference failed")
    digest = _sha256(checkpoint)
    loss = float((saved.get("training_metadata") or {}).get("loss", float("nan")))
    result = {
        "schema_id": "nxml.bc-result.v1",
        "worker_schema_id": WORKER_SCHEMA,
        "job_id": job_id,
        "snapshot_id": snapshot_id,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest,
        "metrics": {
            "loss": loss,
            "train_frames": float(prepared.train_frames),
            "val_frames": float(prepared.val_frames),
        },
        "logs": [f"episodes={len(prepared.episodes)}", f"vae_profile={VAE_PROFILE}"],
        "artifact": {
            "architecture": "bc_transformer_v1",
            "action_spec_id": ACTION_SPEC_ID,
            "action_dim": ACTION_DIM,
            "sequence_length": config.sequence_length,
            "latent_shape": [4, 16, 32],
            "vae_path": config.vae_path,
            "vae_profile": VAE_PROFILE,
            "source_snapshot_id": snapshot_id,
            "source_members": prepared.source_members,
            "worker_config": config.model_dump(),
            "materialization": asdict(prepared),
        },
    }
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True))
    temporary.replace(result_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pokemon ZA raw WebDataset to nxrl BC worker")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--state-dir", default="/var/lib/nxml-control", type=Path)
    parser.add_argument("--checkpoint-dir", default="/var/lib/nxml-control/checkpoints", type=Path)
    args = parser.parse_args()
    run_worker(args.request, args.result, args.state_dir, args.checkpoint_dir)


if __name__ == "__main__":
    main()
