"""Explicit edge-v2 to strict cluster-v2 compatibility mapping.

The first edge producer shipped before the strict ``nxml-core`` contract.
This module is deliberately explicit: it never converts Unix seconds to
monotonic nanoseconds. Missing or inconsistent monotonic data yields a
strict-shaped but invalid record with migration provenance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class MigrationResult:
    value: dict[str, Any]
    compatible: bool
    provenance: list[str]


_CONTROLLER_MAP = {
    "human": "human",
    "policy": "policy",
    "human+policy": "blended",
    "blended": "blended",
    "macro": "safety",
    "mash_a": "safety",
    "safety": "safety",
    "ejected": "safety",
    "neutral": "none",
    "none": "none",
}


def migrate_edge_action_row(row: dict[str, Any]) -> MigrationResult:
    """Map an early edge-v2 parquet row to ``ActionRecordV2`` field names."""
    provenance: list[str] = []
    invalid_reasons = list(row.get("invalid_reasons") or [])
    frame_ns = _optional_int(row.get("frame_timestamp_ns", row.get("frame_monotonic_ns")))
    action_ns = _optional_int(
        row.get("action_timestamp_ns", row.get("action_monotonic_ns"))
    )
    if frame_ns is None:
        frame_ns = 0
        invalid_reasons.append("missing_monotonic_frame_timestamp")
        provenance.append("wall timestamp was not converted to monotonic nanoseconds")
    if action_ns is None:
        action_ns = 0
        invalid_reasons.append("missing_monotonic_action_timestamp")
        provenance.append("wall action timestamp was not converted to monotonic nanoseconds")
    if action_ns > frame_ns:
        provenance.append(f"original action_monotonic_ns={action_ns} was after frame={frame_ns}")
        invalid_reasons.append("action_timestamp_after_frame")
        # Strict records require a non-negative exact age. Preserve the bad
        # original in provenance and normalize only the invalid projection.
        action_ns = frame_ns

    applied = row.get("applied_action", row.get("action"))
    if not isinstance(applied, list):
        applied = list(applied) if applied is not None else []
        invalid_reasons.append("missing_applied_action")
    human_action = _optional_list(row.get("human_action"))
    policy_action = _optional_list(row.get("policy_action"))
    human_mask = row.get("human_action_mask", row.get("human_mask"))
    if human_mask is None:
        human_mask = [False] * len(applied)
        invalid_reasons.append("missing_human_action_mask")
    ownership = row.get("ownership")
    if ownership is None:
        ownership = [0] * len(applied)
        invalid_reasons.append("missing_ownership")
    raw_controller = str(row.get("controller", row.get("active_driver") or "none"))
    controller = _CONTROLLER_MAP.get(raw_controller)
    if controller is None:
        controller = "none"
        invalid_reasons.append("unknown_controller")
        provenance.append(f"unmapped controller={raw_controller!r}")

    valid = bool(row.get("valid", True)) and not invalid_reasons
    value = {
        "frame_index": int(row.get("frame_index", row.get("frame_idx", 0))),
        "frame_timestamp_ns": frame_ns,
        "action_timestamp_ns": action_ns,
        "action_age_ns": frame_ns - action_ns,
        "applied_action": applied,
        "human_action": human_action,
        "human_action_mask": list(human_mask),
        "policy_action": policy_action,
        "controller": controller,
        "ownership": list(ownership),
        "policy_id": row.get("policy_id"),
        "policy_revision": row.get("policy_revision"),
        "valid": valid,
        "invalid_reasons": sorted(set(invalid_reasons)),
    }
    return MigrationResult(value, valid, provenance)


def migrate_edge_manifest(manifest: dict[str, Any]) -> MigrationResult:
    """Project an early edge manifest into strict ``EpisodeManifestV2`` shape."""
    provenance: list[str] = []
    missing: list[str] = []
    frame_times = manifest.get("frame_monotonic_bounds_ns") or {}
    first_ns = _optional_int(
        manifest.get("first_frame_timestamp_ns", frame_times.get("first"))
    )
    last_ns = _optional_int(
        manifest.get("last_frame_timestamp_ns", frame_times.get("last"))
    )
    if first_ns is None or last_ns is None:
        first_ns = first_ns or 0
        last_ns = last_ns or first_ns
        missing.append("monotonic_frame_bounds")
        provenance.append("created_at_utc/floating timestamps were not used as monotonic bounds")

    clock = manifest.get("clock_mapping") or {}
    strict_clock = {
        "clock_id": clock.get("clock_id") or "unknown-monotonic",
        "monotonic_origin_ns": _optional_int(clock.get("monotonic_origin_ns")) or 0,
        "utc_origin": clock.get("utc_origin") or manifest.get("created_at_utc"),
        "uncertainty_ns": _optional_int(clock.get("uncertainty_ns")) or 0,
    }
    if not clock.get("monotonic_origin_ns") or not clock.get("utc_origin"):
        missing.append("strict_clock_mapping")

    files = manifest.get("files") or {}
    strict_files = []
    if isinstance(files, dict):
        for path, metadata in files.items():
            strict_files.append(
                {
                    "path": path,
                    "size_bytes": metadata.get("bytes", 0),
                    "algorithm": "sha256",
                    "digest": metadata.get("sha256", ""),
                }
            )
    elif isinstance(files, list):
        strict_files = list(files)

    config = manifest.get("config") or {}
    config_id = manifest.get("config_id") or _stable_id("config", config)
    build = manifest.get("build") or {}
    build_id = manifest.get("build_id") or build.get("git_revision") or "unknown-build"
    game_id = manifest.get("game_id", manifest.get("game"))
    if not game_id:
        game_id = "unknown-game"
        missing.append("game_id")
    lineage = manifest.get("lineage") or {}
    source_model_id = lineage.get("source_model_id", lineage.get("policy_id"))
    source_revision = lineage.get(
        "source_model_revision", lineage.get("policy_revision")
    )
    strict_lineage = {
        "capture_session_id": lineage.get("capture_session_id")
        or manifest.get("episode_id", "unknown-session"),
        "parent_episode_ids": lineage.get("parent_episode_ids", []),
        "source_dataset_ids": lineage.get("source_dataset_ids", []),
        "source_model_id": source_model_id,
        "source_model_revision": source_revision if source_model_id else None,
    }
    capture = manifest.get("capture") or {}
    video = manifest.get("video") or {}
    strict_capture = {
        "host_id": capture.get("host_id") or build.get("hostname") or "unknown-host",
        "capture_device": capture.get("capture_device") or str(config.get("camera_id", "unknown")),
        "video_codec": capture.get("video_codec") or video.get("codec") or "unknown",
        "container": capture.get("container") or video.get("container") or "unknown",
        "pixel_format": capture.get("pixel_format") or video.get("pixel_format") or "unknown",
        "width": capture.get("width") or config.get("capture_width") or 1,
        "height": capture.get("height") or config.get("capture_height") or 1,
        "nominal_fps": capture.get("nominal_fps") or manifest.get("fps_nominal") or 1,
        "lossless": capture.get("lossless", video.get("lossless", False)),
        "extra": {"migration_missing": missing},
    }
    value = {
        "schema_id": "nxml.episode.v2",
        "episode_id": manifest.get("episode_id", "unknown-episode"),
        "game_id": game_id,
        "config_id": config_id,
        "build_id": build_id,
        "action_spec_id": manifest.get("action_spec_id", manifest.get("action_spec", "")),
        "action_dim": manifest.get("action_dim", 0),
        "created_at_utc": manifest.get("created_at_utc") or datetime.now(UTC).isoformat(),
        "frame_count": manifest.get("frame_count", 0),
        "first_frame_timestamp_ns": first_ns,
        "last_frame_timestamp_ns": last_ns,
        "clock_mapping": strict_clock,
        "capture": strict_capture,
        "files": strict_files,
        "lineage": strict_lineage,
    }
    return MigrationResult(value, not missing, provenance + [f"missing:{item}" for item in missing])


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    return list(value)


def _stable_id(prefix: str, value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}:sha256:{hashlib.sha256(payload).hexdigest()}"
