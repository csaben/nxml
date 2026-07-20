from copy import deepcopy

from nxml_core.contracts.webdataset_v2 import WebDatasetSnapshotV2
from nxwm_mira.data.webdataset_episode import assigned_episode_ids, iter_episode_windows


def _manifest():
    digest = "a" * 64
    segments = []
    episodes = []
    for episode_number in range(2):
        episode_id = f"episode-{episode_number}"
        segment_ids = []
        for sequence_index in range(2):
            shard = f"{episode_number + 1}{sequence_index + 1}" * 32
            segment_id = "sha256:" + shard
            segment_ids.append(segment_id)
            origin = sequence_index * 8
            start_ns = episode_number * 1_000 + sequence_index * 100
            segments.append(
                {
                    "path": f"shards/{shard[:2]}/{shard}.tar",
                    "size_bytes": 3,
                    "sha256": shard,
                    "episode_id": episode_id,
                    "segment_id": segment_id,
                    "sequence_index": sequence_index,
                    "split": "train",
                    "temporal": {
                        "clock_id": "linux-monotonic",
                        "timeline_start_ns": start_ns,
                        "timeline_end_ns": start_ns + 100,
                        "frame_index_origin": origin,
                        "frame_count": 8,
                        "video_frame_mapping": "decoded_ordinal_equals_frame_idx",
                        "frame_timestamp_column": "frame_monotonic_ns",
                        "action_timestamp_column": "action_monotonic_ns",
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
                    },
                    "members": [
                        {"role": role, "path": f"x{suffix}", "size_bytes": 1, "sha256": digest}
                        for role, suffix in (
                            ("video", ".mkv"),
                            ("actions", ".parquet"),
                            ("events", ".events.parquet"),
                        )
                    ],
                }
            )
        episodes.append(
            {
                "episode_id": episode_id,
                "close_id": "sha256:" + ("d" * 64),
                "split": "train",
                "clock_id": "linux-monotonic",
                "timeline_start_ns": episode_number * 1_000,
                "timeline_end_ns": episode_number * 1_000 + 200,
                "segment_ids": segment_ids,
            }
        )
    return WebDatasetSnapshotV2.model_validate(
        {
            "schema_id": "nxml.hf-webdataset-snapshot.v2",
            "dataset_id": "test",
            "source_snapshot_id": "sha256:" + "c" * 64,
            "publication_kind": "snapshot",
            "episodes": episodes,
            "segments": segments,
            "training": {
                "publisher_git_commit": "e" * 40,
                "source_snapshot_id": "sha256:" + "c" * 64,
                "hub_repo_id": "arelius/nxml-pokemon-za-gameplay",
                "window": {
                    "source_fps": 60,
                    "target_fps": 15,
                    "frame_stride": 4,
                    "clip_frames": 3,
                    "action_pooling": "sticks_mean_buttons_or",
                    "aspect_mode": "pad",
                    "frame_size_hw": [288, 512],
                },
                "checkpoint_mode": "finetune_from",
                "checkpoint_id": "checkpoint-99999",
            },
        }
    )


def test_episode_assignment_is_deterministic_and_has_no_duplicates():
    manifest = _manifest()
    assignments = [assigned_episode_ids(manifest, rank=rank, world_size=2) for rank in range(2)]
    assert assignments == [("episode-0",), ("episode-1",)]
    assert set(assignments[0]).isdisjoint(assignments[1])


def test_window_can_cross_segment_boundary_without_stretching():
    windows = iter_episode_windows(_manifest(), split="train")
    crossing = next(window for window in windows if len(window.spans) == 2)
    assert crossing.global_frame_indices == (0, 4, 8)
    assert [span.local_frame_indices for span in crossing.spans] == [(0, 4), (0,)]


def test_manifest_rejects_frame_mapping_gap():
    body = _manifest().model_dump(mode="json")
    broken = deepcopy(body)
    broken["segments"][1]["temporal"]["frame_index_origin"] = 9
    try:
        WebDatasetSnapshotV2.model_validate(broken)
    except ValueError as exc:
        assert "global frame mapping gap or overlap" in str(exc)
    else:
        raise AssertionError("frame mapping gap accepted")
