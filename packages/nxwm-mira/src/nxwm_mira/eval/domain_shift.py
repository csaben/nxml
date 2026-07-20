"""Measured, immutable codec domain-shift audit gate."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from nxml_core.contracts.webdataset_v2 import CompactCodecLineageV2, WebDatasetSnapshotV2
from nxwm_mira.data.webdataset_episode import EpisodeWindow, iter_episode_windows
from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_ID = r"^sha256:[0-9a-f]{64}$"


class EligibleEpisodeSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    episode_id: str = Field(min_length=1)
    split: Literal["train", "validation"]
    disposition_id: str = Field(min_length=1)
    validator: str = Field(min_length=1)
    validator_version: str = Field(min_length=1)
    training_eligible: Literal[True]
    human_owned_rows: int = Field(gt=0)
    dynamic_frame_fraction: float = Field(gt=0, le=1)


class CodecCheckpointIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    checkpoint_id: Literal["checkpoint-99999"]
    checkpoint_digest: str = Field(pattern=SHA256_ID)
    config_digest: str = Field(pattern=SHA256_ID)


class DomainShiftThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    l1_relative_increase: float = Field(ge=0)
    lpips_relative_increase: float = Field(ge=0)
    dino_relative_increase: float = Field(ge=0)
    motion_relative_increase: float = Field(ge=0)
    action_conditioned_relative_increase: float = Field(ge=0)
    confidence_level: float = Field(gt=0.5, lt=1)
    minimum_windows_per_domain: int = Field(gt=1)


class DomainShiftAuditInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["nxml.codec-domain-shift-input.v1"]
    source_snapshot_id: str = Field(pattern=SHA256_ID)
    hub_repo_id: str = Field(min_length=1)
    hub_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    compact_profile: CompactCodecLineageV2
    checkpoint: CodecCheckpointIdentity
    legacy_episodes: list[EligibleEpisodeSelection] = Field(min_length=1)
    compact_episodes: list[EligibleEpisodeSelection] = Field(min_length=1)
    seed: int = Field(ge=0)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    audit_config_digest: str = Field(pattern=SHA256_ID)
    thresholds: DomainShiftThresholds

    @model_validator(mode="after")
    def validate_selections(self):
        legacy = {item.episode_id for item in self.legacy_episodes}
        compact = {item.episode_id for item in self.compact_episodes}
        if legacy & compact:
            raise ValueError("legacy and compact episode selections must be disjoint")
        return self


class WindowMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str = Field(pattern=SHA256_ID)
    domain: Literal["legacy", "compact"]
    episode_id: str
    frame_indices: tuple[int, ...]
    l1: float = Field(ge=0)
    lpips: float = Field(ge=0)
    dino: float = Field(ge=0)
    motion: float = Field(ge=0)
    action_conditioned: float = Field(ge=0)
    artifact_sha256: str = Field(pattern=SHA256_ID)


class MetricDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    legacy_mean: float = Field(ge=0)
    compact_mean: float = Field(ge=0)
    relative_increase: float
    bootstrap_low: float
    bootstrap_high: float
    threshold: float = Field(ge=0)
    exceeded: bool


class DomainShiftDecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["nxml.codec-domain-shift-decision.v1"]
    input_digest: str = Field(pattern=SHA256_ID)
    decision: Literal["finetune_codec", "retain_codec"]
    metrics: dict[str, MetricDecision]
    sample_ids: list[str] = Field(min_length=1)
    result_digest: str = Field(pattern=SHA256_ID)


class CodecJobManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["nxml.codec-job.v1"]
    job_id: str = Field(min_length=1)
    decision_digest: str = Field(pattern=SHA256_ID)
    source_snapshot_id: str = Field(pattern=SHA256_ID)
    codec_input_digest: str = Field(pattern=SHA256_ID)
    mode: Literal["finetune_from", "continue_from"]
    checkpoint_id: str = Field(min_length=1)
    resume_job_id: str | None = None
    max_steps: int = Field(gt=0, le=50_000)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_digest: str = Field(pattern=SHA256_ID)

    @model_validator(mode="after")
    def validate_resume_lineage(self):
        if self.mode == "finetune_from" and self.resume_job_id is not None:
            raise ValueError("finetune_from cannot resume optimizer/job state")
        if self.mode == "continue_from" and self.resume_job_id is None:
            raise ValueError("continue_from requires exact prior job id")
        return self


def canonical_digest(model: BaseModel) -> str:
    encoded = json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def pad_to_frame_size(video: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Resize isotropically and symmetrically pad TCHW; never stretch or crop."""
    if video.ndim != 4:
        raise ValueError("video must be TCHW")
    source_h, source_w = video.shape[-2:]
    target_h, target_w = target_hw
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("frame dimensions must be positive")
    scale = min(target_h / source_h, target_w / source_w)
    resized_h = max(1, min(target_h, round(source_h * scale)))
    resized_w = max(1, min(target_w, round(source_w * scale)))
    resized = F.interpolate(
        video.float(), size=(resized_h, resized_w), mode="bilinear", antialias=True
    )
    top = (target_h - resized_h) // 2
    left = (target_w - resized_w) // 2
    return F.pad(resized, (left, target_w - resized_w - left, top, target_h - resized_h - top))


def deterministic_windows(
    manifest: WebDatasetSnapshotV2,
    selections: list[EligibleEpisodeSelection],
    *,
    seed: int,
    limit: int,
) -> tuple[EpisodeWindow, ...]:
    if limit <= 0:
        raise ValueError("window limit must be positive")
    selected = {item.episode_id: item for item in selections}
    manifest_episodes = {item.episode_id: item for item in manifest.episodes}
    for episode_id, selection in selected.items():
        episode = manifest_episodes.get(episode_id)
        if episode is None or episode.split != selection.split or episode.split == "canary":
            raise ValueError(f"selected episode absent or split mismatch: {episode_id}")
    windows = [
        window
        for split in ("train", "validation")
        for window in iter_episode_windows(manifest, split=split)
        if window.episode_id in selected
    ]
    ranked = sorted(
        windows,
        key=lambda window: hashlib.sha256(
            f"{seed}:{window.episode_id}:{','.join(map(str, window.global_frame_indices))}".encode()
        ).digest(),
    )
    if len(ranked) < limit:
        raise ValueError("not enough deterministic eligible windows")
    return tuple(ranked[:limit])


def measure_window(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    actions: torch.Tensor,
    *,
    feature_fn,
    lpips_fn,
) -> dict[str, float]:
    if target.shape != reconstruction.shape or target.ndim != 4:
        raise ValueError("target and reconstruction must have equal TCHW shapes")
    if actions.ndim != 2 or actions.shape[0] != target.shape[0]:
        raise ValueError("actions must align one-to-one with frames")
    target = target.float()
    reconstruction = reconstruction.float()
    l1 = (target - reconstruction).abs().mean()
    lpips = torch.as_tensor(lpips_fn(reconstruction, target)).float().mean()
    target_features = feature_fn(target)
    reconstruction_features = feature_fn(reconstruction)
    dino = (1 - F.cosine_similarity(target_features, reconstruction_features, dim=-1)).mean()
    target_motion = target[1:] - target[:-1]
    reconstruction_motion = reconstruction[1:] - reconstruction[:-1]
    motion = (target_motion - reconstruction_motion).abs().mean()
    weights = actions[1:].abs().mean(dim=-1) + 1.0
    per_transition = (target_motion - reconstruction_motion).abs().flatten(1).mean(dim=1)
    action_conditioned = (per_transition * weights).sum() / weights.sum()
    return {
        "l1": float(l1),
        "lpips": float(lpips),
        "dino": float(dino),
        "motion": float(motion),
        "action_conditioned": float(action_conditioned),
    }


def write_side_by_side(target: torch.Tensor, reconstruction: torch.Tensor, path: Path) -> str:
    """Write a bounded representative first/middle/last RGB contact sheet."""
    from PIL import Image

    if target.shape != reconstruction.shape or target.ndim != 4 or target.shape[1] != 3:
        raise ValueError("side-by-side inputs must be equal RGB TCHW")
    indices = sorted({0, target.shape[0] // 2, target.shape[0] - 1})
    rows = []
    for index in indices:
        pair = torch.cat((target[index], reconstruction[index]), dim=-1)
        rows.append(pair)
    image = torch.cat(rows, dim=-2).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="PNG", optimize=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:" + digest


def _bootstrap_relative(
    legacy: list[float], compact: list[float], *, seed: int, confidence: float
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(1000):
        old = sum(rng.choice(legacy) for _ in legacy) / len(legacy)
        new = sum(rng.choice(compact) for _ in compact) / len(compact)
        estimates.append((new - old) / max(old, 1e-12))
    estimates.sort()
    tail = (1 - confidence) / 2
    return estimates[int(tail * 999)], estimates[int((1 - tail) * 999)]


def decide_domain_shift(
    audit_input: DomainShiftAuditInput, results: list[WindowMetrics]
) -> DomainShiftDecisionRecord:
    legacy = [item for item in results if item.domain == "legacy"]
    compact = [item for item in results if item.domain == "compact"]
    minimum = audit_input.thresholds.minimum_windows_per_domain
    if len(legacy) < minimum or len(compact) < minimum:
        raise ValueError("insufficient measured windows for domain-shift decision")
    decisions = {}
    for offset, name in enumerate(("l1", "lpips", "dino", "motion", "action_conditioned")):
        old = [getattr(item, name) for item in legacy]
        new = [getattr(item, name) for item in compact]
        old_mean = sum(old) / len(old)
        new_mean = sum(new) / len(new)
        relative = (new_mean - old_mean) / max(old_mean, 1e-12)
        low, high = _bootstrap_relative(
            old,
            new,
            seed=audit_input.seed + offset,
            confidence=audit_input.thresholds.confidence_level,
        )
        threshold = getattr(audit_input.thresholds, f"{name}_relative_increase")
        decisions[name] = MetricDecision(
            legacy_mean=old_mean,
            compact_mean=new_mean,
            relative_increase=relative,
            bootstrap_low=low,
            bootstrap_high=high,
            threshold=threshold,
            exceeded=low > threshold,
        )
    decision = (
        "finetune_codec" if any(item.exceeded for item in decisions.values()) else "retain_codec"
    )
    provisional = {
        "input_digest": canonical_digest(audit_input),
        "decision": decision,
        "metrics": {key: value.model_dump(mode="json") for key, value in decisions.items()},
        "sample_ids": sorted(item.sample_id for item in results),
    }
    result_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(provisional, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return DomainShiftDecisionRecord(
        schema_id="nxml.codec-domain-shift-decision.v1",
        result_digest=result_digest,
        **provisional,
    )


@dataclass(frozen=True)
class PublicationDryRun:
    episode_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]


def publication_dry_run(
    snapshot: dict, requested_episode_ids: set[str], inspections: dict[str, dict]
):
    """Refuse excluded/vetoed/static episodes and segments lacking exact inspection evidence."""
    excluded = {item["episode_id"]: item for item in snapshot.get("excluded_episodes", [])}
    blocked = requested_episode_ids & excluded.keys()
    if blocked:
        reasons = {
            episode_id: (excluded[episode_id].get("episode_disposition") or {}).get("reason")
            for episode_id in sorted(blocked)
        }
        raise ValueError(f"requested episodes are excluded or vetoed: {reasons}")
    episodes = {item["episode_id"]: item for item in snapshot.get("episodes", [])}
    missing = requested_episode_ids - episodes.keys()
    if missing:
        raise ValueError(f"requested episodes missing from eligible snapshot: {sorted(missing)}")
    segment_ids = tuple(
        segment_id
        for episode_id in sorted(requested_episode_ids)
        for segment_id in episodes[episode_id]["segment_ids"]
    )
    for segment_id in segment_ids:
        evidence = inspections.get(segment_id)
        if evidence is None or evidence.get("decoded_frames_equal_action_rows") is not True:
            raise ValueError(f"missing or failed artifact inspection: {segment_id}")
        CompactCodecLineageV2.model_validate(evidence.get("codec_compatibility"))
    return PublicationDryRun(tuple(sorted(requested_episode_ids)), segment_ids)
