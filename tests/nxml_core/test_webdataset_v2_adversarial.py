import pytest
from nxml_core.contracts.webdataset_v2 import (
    CompactCodecLineageV2,
    ImmutableTrainingLineageV2,
    MiraWindowContractV2,
    TemporalMappingV2,
)
from pydantic import ValidationError


def codec():
    return {
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
        "measured_bit_rate": 16_000_000,
        "aspect_mode": "pad",
        "artifact_probe": "ffprobe",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("codec", "hevc"),
        ("profile", "High"),
        ("level", 41),
        ("pixel_format", "yuv444p"),
        ("width", 1920),
        ("r_frame_rate", "30000/1001"),
        ("avg_frame_rate", "30/1"),
        ("gop_size", 120),
        ("max_b_frames", 2),
        ("aspect_mode", "stretch"),
    ],
)
def test_codec_compatibility_rejects_artifact_mismatch(field, value):
    broken = {**codec(), field: value}
    with pytest.raises(ValidationError):
        CompactCodecLineageV2.model_validate(broken)


def test_codec_requires_measured_bitrate_and_probe_lineage():
    missing = codec()
    missing.pop("measured_bit_rate")
    with pytest.raises(ValidationError, match="measured_bit_rate"):
        CompactCodecLineageV2.model_validate(missing)
    with pytest.raises(ValidationError, match="artifact_probe"):
        CompactCodecLineageV2.model_validate({**codec(), "artifact_probe": "configured"})


def test_temporal_mapping_rejects_missing_or_empty_interval():
    mapping = {
        "clock_id": "linux-monotonic",
        "timeline_start_ns": 100,
        "timeline_end_ns": 200,
        "frame_index_origin": 0,
        "frame_count": 60,
        "video_frame_mapping": "decoded_ordinal_equals_frame_idx",
        "frame_timestamp_column": "frame_monotonic_ns",
        "action_timestamp_column": "action_monotonic_ns",
    }
    missing = dict(mapping)
    missing.pop("clock_id")
    with pytest.raises(ValidationError, match="clock_id"):
        TemporalMappingV2.model_validate(missing)
    with pytest.raises(ValidationError, match="non-empty"):
        TemporalMappingV2.model_validate({**mapping, "timeline_end_ns": 100})


def test_checkpoint_modes_are_mutually_exclusive_contracts():
    common = {
        "publisher_git_commit": "a" * 40,
        "source_snapshot_id": "sha256:" + "b" * 64,
        "hub_repo_id": "arelius/nxml-pokemon-za-gameplay",
        "window": MiraWindowContractV2(
            source_fps=60,
            target_fps=15,
            frame_stride=4,
            clip_frames=16,
            action_pooling="sticks_mean_buttons_or",
            aspect_mode="pad",
            frame_size_hw=(288, 512),
        ),
    }
    with pytest.raises(ValidationError, match="checkpoint mode"):
        ImmutableTrainingLineageV2.model_validate(
            {**common, "checkpoint_mode": "none", "checkpoint_id": "checkpoint-99999"}
        )
    with pytest.raises(ValidationError, match="checkpoint mode"):
        ImmutableTrainingLineageV2.model_validate(
            {**common, "checkpoint_mode": "continue_from", "checkpoint_id": None}
        )
