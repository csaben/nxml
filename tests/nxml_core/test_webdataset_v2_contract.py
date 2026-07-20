import pytest
from nxml_core.contracts.webdataset_v2 import WebDatasetSnapshotV2
from pydantic import ValidationError


def body():
    digest = "a" * 64

    def member(role, suffix):
        return {
            "role": role,
            "path": "episode.000000" + suffix,
            "size_bytes": 1,
            "sha256": "b" * 64,
        }

    return {
        "schema_id": "nxml.hf-webdataset-snapshot.v2",
        "dataset_id": "nxml-pokemon-za-v2",
        "source_snapshot_id": "sha256:" + "c" * 64,
        "publication_kind": "snapshot",
        "episodes": [
            {
                "episode_id": "episode",
                "close_id": "sha256:" + "d" * 64,
                "split": "train",
                "clock_id": "linux-monotonic",
                "timeline_start_ns": 100,
                "timeline_end_ns": 200,
                "segment_ids": ["sha256:" + digest],
            }
        ],
        "segments": [
            {
                "path": f"shards/aa/{digest}.tar",
                "size_bytes": 3,
                "sha256": digest,
                "episode_id": "episode",
                "segment_id": "sha256:" + digest,
                "sequence_index": 0,
                "split": "train",
                "temporal": {
                    "clock_id": "linux-monotonic",
                    "timeline_start_ns": 100,
                    "timeline_end_ns": 200,
                    "frame_index_origin": 0,
                    "frame_count": 60,
                    "video_frame_mapping": "decoded_ordinal_equals_frame_idx",
                    "frame_timestamp_column": "frame_monotonic_ns",
                    "action_timestamp_column": "action_monotonic_ns",
                    "interval_semantics": "half_open",
                },
                "codec": {
                    "compatibility_id": "nxml.compact-h264-main32-720p60.v1",
                    "codec": "h264",
                    "container": "matroska",
                    "profile": "Main",
                    "level": 32,
                    "pixel_format": "yuv420p",
                    "width": 1280,
                    "height": 720,
                    "r_frame_rate": "60/1",
                    "avg_frame_rate": "60/1",
                    "time_base": "1/1000",
                    "gop_size": 60,
                    "max_b_frames": 0,
                    "measured_bit_rate": 16000000,
                    "artifact_probe": "ffprobe",
                    "aspect_mode": "pad",
                },
                "control": {
                    "mode": "human",
                    "action_schema_id": "nxml.dagger-actions.v2",
                    "action_spec_id": "switch_packets.v1",
                    "mute_mask_version": "switch_packets.v1/mute.v1",
                    "policy_id": None,
                    "policy_revision": None,
                    "policy_digest": None,
                },
                "members": [
                    member("video", ".mkv"),
                    member("actions", ".parquet"),
                    member("events", ".events.parquet"),
                ],
            }
        ],
        "training": {
            "publisher_git_commit": "e" * 40,
            "source_snapshot_id": "sha256:" + "c" * 64,
            "hub_repo_id": "arelius/nxml-pokemon-za-gameplay",
            "hub_revision": None,
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


def test_complete_v2_contract_and_fail_closed_lineage():
    parsed = WebDatasetSnapshotV2.model_validate(body())
    assert parsed.training.window.aspect_mode == "pad"
    bad = body()
    bad["training"]["window"]["aspect_mode"] = "stretch"
    with pytest.raises(ValidationError, match="aspect_mode"):
        WebDatasetSnapshotV2.model_validate(bad)
    bad = body()
    bad["segments"][0]["control"]["policy_id"] = "unexpected"
    with pytest.raises(ValidationError, match="human lineage"):
        WebDatasetSnapshotV2.model_validate(bad)
    bad = body()
    bad["training"]["checkpoint_mode"] = "continue_from"
    bad["training"]["checkpoint_id"] = None
    with pytest.raises(ValidationError, match="checkpoint mode"):
        WebDatasetSnapshotV2.model_validate(bad)
