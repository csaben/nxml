#!/usr/bin/env python3
"""Convert a strict synthetic v1 canary into an artifact-probed v2 manifest."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from nxml_core.contracts.webdataset_v1 import WebDatasetSnapshotV1
from nxml_core.contracts.webdataset_v2 import WebDatasetSnapshotV2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--publisher-git-commit", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    v1_path = next(root.glob("canaries/*/manifest.json"))
    v1 = WebDatasetSnapshotV1.model_validate_json(v1_path.read_text())
    if len(v1.shards) != 1 or v1.publication_kind != "canary":
        raise ValueError("v2 synthetic conversion requires exactly one v1 canary shard")
    shard = v1.shards[0]
    with tempfile.TemporaryDirectory(prefix="nxml-v2-canary-") as raw_tmp:
        temp = Path(raw_tmp)
        with tarfile.open(root / shard.path, "r:*") as archive:
            declared = {member.path: member for member in shard.members}
            if {info.name for info in archive.getmembers()} != set(declared):
                raise ValueError("tar member set mismatch")
            extracted = {}
            for info in archive.getmembers():
                if (
                    not info.isfile()
                    or Path(info.name).is_absolute()
                    or ".." in Path(info.name).parts
                ):
                    raise ValueError("unsafe tar member")
                source = archive.extractfile(info)
                if source is None:
                    raise ValueError("unreadable tar member")
                target = temp / Path(info.name).name
                target.write_bytes(source.read())
                extracted[declared[info.name].role] = target
        probe = json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,profile,level,pix_fmt,width,height,r_frame_rate,avg_frame_rate,time_base,has_b_frames:format=format_name,bit_rate:frame=key_frame",
                    "-of",
                    "json",
                    str(extracted["video"]),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        stream = probe["streams"][0]
        keyframes = [
            index for index, frame in enumerate(probe["frames"]) if frame["key_frame"] == 1
        ]
        intervals = [right - left for left, right in itertools.pairwise(keyframes)]
        actions = pq.read_table(extracted["actions"])
        rows = actions.to_pylist()
    if not intervals or set(intervals) != {60}:
        raise ValueError(f"artifact GOP is not exactly 60: {intervals}")
    if [row["frame_idx"] for row in rows] != list(range(len(rows))):
        raise ValueError("synthetic frame ordinals are not contiguous")
    start_ns = rows[0]["frame_monotonic_ns"]
    end_ns = rows[-1]["frame_monotonic_ns"] + 16_666_667
    close_body = {
        "episode_id": shard.episode_id,
        "segment_ids": [shard.segment_id],
        "timeline_start_ns": start_ns,
        "timeline_end_ns": end_ns,
    }
    close_id = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(close_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    manifest = WebDatasetSnapshotV2.model_validate(
        {
            "schema_id": "nxml.hf-webdataset-snapshot.v2",
            "dataset_id": v1.dataset_id,
            "source_snapshot_id": v1.source_snapshot_id,
            "publication_kind": "canary",
            "episodes": [
                {
                    "episode_id": shard.episode_id,
                    "close_id": close_id,
                    "split": "canary",
                    "clock_id": "linux-monotonic",
                    "timeline_start_ns": start_ns,
                    "timeline_end_ns": end_ns,
                    "segment_ids": [shard.segment_id],
                }
            ],
            "segments": [
                {
                    "path": shard.path,
                    "size_bytes": shard.size_bytes,
                    "sha256": shard.sha256,
                    "episode_id": shard.episode_id,
                    "segment_id": shard.segment_id,
                    "sequence_index": 0,
                    "split": "canary",
                    "temporal": {
                        "clock_id": "linux-monotonic",
                        "timeline_start_ns": start_ns,
                        "timeline_end_ns": end_ns,
                        "frame_index_origin": 0,
                        "frame_count": len(rows),
                        "video_frame_mapping": "decoded_ordinal_equals_frame_idx",
                        "frame_timestamp_column": "frame_monotonic_ns",
                        "action_timestamp_column": "action_monotonic_ns",
                    },
                    "codec": {
                        "compatibility_id": "nxml.compact-h264-720p60.v1",
                        "codec": stream["codec_name"],
                        "container": "matroska"
                        if probe["format"]["format_name"] == "matroska,webm"
                        else probe["format"]["format_name"],
                        "profile": stream["profile"],
                        "level": stream["level"],
                        "pixel_format": stream["pix_fmt"],
                        "width": stream["width"],
                        "height": stream["height"],
                        "r_frame_rate": stream["r_frame_rate"],
                        "avg_frame_rate": stream["avg_frame_rate"],
                        "time_base": stream["time_base"],
                        "gop_size": max(intervals),
                        "max_b_frames": stream["has_b_frames"],
                        "measured_bit_rate": int(probe["format"]["bit_rate"]),
                        "aspect_mode": "pad",
                        "artifact_probe": "ffprobe",
                    },
                    "control": {
                        "mode": "human",
                        "action_schema_id": "nxml.dagger-actions.v2",
                        "action_spec_id": "switch_packets.v1",
                        "mute_mask_version": "switch_packets.v1/mute.v1",
                    },
                    "members": [member.model_dump(mode="json") for member in shard.members],
                }
            ],
            "training": {
                "publisher_git_commit": args.publisher_git_commit,
                "source_snapshot_id": v1.source_snapshot_id,
                "hub_repo_id": "arelius/nxml-pokemon-za-gameplay",
                "window": {
                    "source_fps": 60,
                    "target_fps": 15,
                    "frame_stride": 4,
                    "clip_frames": 16,
                    "action_pooling": "sticks_mean_buttons_or",
                    "aspect_mode": "pad",
                    "frame_size_hw": [288, 512],
                },
                "checkpoint_mode": "finetune_from",
                "checkpoint_id": "checkpoint-99999",
            },
        }
    )
    destination = root / "canaries" / shard.sha256 / "manifest.v2.json"
    destination.write_text(manifest.model_dump_json(indent=2) + "\n")
    print(json.dumps({"manifest": str(destination), "sha256": shard.sha256}))


if __name__ == "__main__":
    main()
