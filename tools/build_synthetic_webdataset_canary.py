#!/usr/bin/env python3
"""Build a tiny non-gameplay H.264 WebDataset canary and strict publication manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from nxml_core.contracts.webdataset_v1 import WebDatasetSnapshotV1

EPISODE_ID = "00000000-0000-4000-8000-000000000001"


def digest(path: Path) -> tuple[int, str]:
    value = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            value.update(chunk)
    return size, value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    prefix = f"{EPISODE_ID}.000000"
    video = root / f"{prefix}.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=60:duration=2",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-level:v",
            "4.2",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "60",
            "-bf",
            "0",
            "-keyint_min",
            "60",
            "-sc_threshold",
            "0",
            "-an",
            str(video),
        ],
        check=True,
    )
    actions = root / f"{prefix}.parquet"
    rows = []
    for index in range(120):
        frame_ns = 1_000_000_000 + index * 16_666_667
        action_ns = frame_ns - 1_000_000
        neutral = [0.0] * 26
        rows.append(
            {
                "row_schema_id": "nxml.dagger-actions.v2",
                "action_spec_id": "switch_packets.v1",
                "frame_idx": index,
                "timestamp": frame_ns / 1e9,
                "frame_monotonic_ns": frame_ns,
                "frame_timestamp_ns": frame_ns,
                "action_timestamp": action_ns / 1e9,
                "action_monotonic_ns": action_ns,
                "action_timestamp_ns": action_ns,
                "action_age_ns": 1_000_000,
                "action_age": 0.001,
                "valid": True,
                "invalid_reasons": [],
                "action": neutral,
                "applied_action": neutral,
                "human_action": neutral,
                "human_mask": [False] * 26,
                "policy_action": neutral,
                "muted_policy_action": neutral,
                "mute_mask": [False] * 26,
                "mute_mask_version": "switch_packets.v1/mute.v1",
                "ownership": [0] * 26,
                "controller_id": "synthetic-canary",
                "active_driver": "none",
                "controller": "none",
                "policy_id": None,
                "policy_revision": None,
                "policy_digest": None,
                "human_monotonic_ns": None,
                "policy_monotonic_ns": None,
                "policy_observation_monotonic_ns": None,
                "ownership_source": "none",
                "mode": "human",
                "takeover": False,
                "takeover_reason": None,
                "takeover_release_remaining_ns": 0,
                "proposal_valid": False,
                "proposal_fresh": False,
                "proposal_sequence": None,
                "proposal_age_ns": None,
                "gap_state": "none",
                "gap_reason": None,
                "gap_duration_ns": 0,
                "boundary_sequence": None,
                "boundary_acknowledged": False,
                "applied_action_valid": True,
                "bc_training_eligible": False,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), actions, compression="zstd")
    events = root / f"{prefix}.events.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"event_schema_id": "nxml.synthetic-canary.v1", "monotonic_ns": 1_000_000_000}]
        ),
        events,
        compression="zstd",
    )
    tar_path = root / "segment.tar"
    members = [("video", video), ("actions", actions), ("events", events)]
    with tarfile.open(tar_path, "w") as archive:
        for _role, path in members:
            archive.add(path, arcname=path.name, recursive=False)
    size, sha = digest(tar_path)
    remote = f"shards/{sha[:2]}/{sha}.tar"
    destination = root / remote
    destination.parent.mkdir(parents=True)
    tar_path.replace(destination)
    manifest = WebDatasetSnapshotV1.model_validate(
        {
            "schema_id": "nxml.hf-webdataset-snapshot.v1",
            "dataset_id": "nxml-pokemon-za-v2",
            "source_snapshot_id": "sha256:" + sha,
            "action_spec_id": "switch_packets.v1",
            "publication_kind": "canary",
            "shards": [
                {
                    "path": remote,
                    "size_bytes": size,
                    "sha256": sha,
                    "episode_id": EPISODE_ID,
                    "segment_id": "sha256:" + sha,
                    "sequence_index": 0,
                    "split": "canary",
                    "codec": {
                        "codec": "h264",
                        "container": "matroska",
                        "profile": "High",
                        "level": 42,
                        "pixel_format": "yuv420p",
                        "width": 1280,
                        "height": 720,
                        "nominal_fps": 60.0,
                        "time_base": "1/1000",
                        "gop_size": 60,
                        "aspect_mode": "pad",
                    },
                    "members": [
                        {
                            "role": role,
                            "path": path.name,
                            "size_bytes": digest(path)[0],
                            "sha256": digest(path)[1],
                        }
                        for role, path in members
                    ],
                }
            ],
        }
    )
    manifest_path = root / "canaries" / sha / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    print(json.dumps({"root": str(root), "sha256": sha, "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
