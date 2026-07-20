#!/usr/bin/env python3
"""Fetch the fixed six compact-canary inspections without exposing credentials."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from nxml_core.contracts.webdataset_v2 import CompactCodecLineageV2
from pydantic import ValidationError

SEGMENT_IDS = (
    "sha256:effde146c4f68ac9774e17c150e866d75d6071b71e104f01561f047363683f94",
    "sha256:f44b33c9adf8068c43c6226f245beeeccc3f827bd66b86ce2fe5084692c468da",
    "sha256:d3e67d229dbe2c3ee48c3a6fdf88cc6545564cd1623c05defb3e79fed4809fcc",
    "sha256:9b5d80a46e6b08ae57fba2ed8ee05bca4c46a7751e809deec665a0877fa50464",
    "sha256:05e9cae7af059f733d9a5369eaf0e1375d14f80faa6504726e26dd1e43808082",
    "sha256:ea7f44b2a63544174241039609daf4b3af2aa1b223f574a8d3f23b94278d9727",
)


def codec_compatibility(result: dict) -> dict:
    video = result["video"]
    container = "matroska" if video["container"] == "matroska,webm" else video["container"]
    parsed = CompactCodecLineageV2.model_validate(
        {
            "compatibility_id": "nxml.compact-h264-main32-720p60.v1",
            "codec": video["codec"],
            "container": container,
            "profile": video["profile"],
            "level": video["level"],
            "pixel_format": video["pixel_format"],
            "width": video["width"],
            "height": video["height"],
            "r_frame_rate": video["r_frame_rate"],
            "avg_frame_rate": video["avg_frame_rate"],
            "time_base": video["time_base"],
            "gop_size": video["gop_size"],
            "max_b_frames": video["max_b_frames"],
            "measured_bit_rate": video["measured_bit_rate"],
            "aspect_mode": "pad",
            "artifact_probe": "ffprobe",
        }
    )
    return parsed.model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default="http://100.80.98.4:8787")
    parser.add_argument("--token-file", type=Path, default=Path("/etc/nxml-control/token"))
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    if not token:
        raise ValueError("empty control token")
    results = []
    for segment_id in SEGMENT_IDS:
        request = urllib.request.Request(
            f"{args.base_url}/v1/segments/{segment_id}/artifact-inspection",
            headers={"Authorization": "Bearer " + token},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        if (
            result.get("schema_id") != "nxml.segment-artifact-inspection.v1"
            or result.get("segment_id") != segment_id
            or result.get("decoded_frames_equal_action_rows") is not True
        ):
            raise ValueError(f"failed inspection response: {segment_id}")
        try:
            result["codec_compatibility"] = codec_compatibility(result)
            result["codec_compatible"] = True
        except ValidationError as error:
            result["codec_compatible"] = False
            result["codec_compatibility_errors"] = error.errors(
                include_url=False, include_input=False
            )
        results.append(result)
    payload = {"schema_id": "nxml.compact-canary-inspection-set.v1", "segments": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "verified_segments": len(results),
                "compatible_segments": sum(item["codec_compatible"] for item in results),
            }
        )
    )


if __name__ == "__main__":
    main()
