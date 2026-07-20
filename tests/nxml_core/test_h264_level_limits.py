import pytest
from nxml_core.contracts.webdataset_v2 import CompactCodecLineageV2
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
        "measured_bit_rate": 16_125_611,
        "aspect_mode": "pad",
        "artifact_probe": "ffprobe",
    }


def test_main_level_32_boundary_is_exactly_valid():
    parsed = CompactCodecLineageV2.model_validate(codec())
    macroblocks_per_frame = ((parsed.width + 15) // 16) * ((parsed.height + 15) // 16)
    assert macroblocks_per_frame == 3_600
    assert macroblocks_per_frame * 60 == 216_000
    assert parsed.measured_bit_rate < 20_000_000


def test_main_level_32_rejects_above_60fps_and_max_bitrate():
    with pytest.raises(ValidationError, match="MaxMBPS"):
        CompactCodecLineageV2.model_validate(
            {**codec(), "r_frame_rate": "61/1", "avg_frame_rate": "61/1"}
        )
    with pytest.raises(ValidationError, match="MaxBR"):
        CompactCodecLineageV2.model_validate({**codec(), "measured_bit_rate": 20_000_001})


def test_main_level_32_rejects_wrong_dimensions_and_profile():
    with pytest.raises(ValidationError, match="width"):
        CompactCodecLineageV2.model_validate({**codec(), "width": 1920})
    with pytest.raises(ValidationError, match="profile"):
        CompactCodecLineageV2.model_validate({**codec(), "profile": "High"})
