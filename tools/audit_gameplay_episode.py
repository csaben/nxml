#!/usr/bin/env python3
"""Authenticated, read-only aggregate audit for one closed rolling episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from inspect_compact_canary_segments import codec_compatibility


def get_json(base_url: str, token: str, path: str) -> dict:
    request = urllib.request.Request(base_url + path, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--close-id", required=True)
    parser.add_argument("--segment-id", action="append", required=True, dest="segment_ids")
    parser.add_argument("--base-url", default="http://100.80.98.4:8787")
    parser.add_argument("--token-file", type=Path, default=Path("/etc/nxml-control/token"))
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    close = get_json(args.base_url, token, f"/v1/episode-closes/{args.close_id}")
    manifest = close["manifest"]
    if close["episode_id"] != args.episode_id or manifest["episode_id"] != args.episode_id:
        raise ValueError("close episode identity mismatch")
    expected = [item["segment_id"] for item in manifest["segments"]]
    if expected != args.segment_ids or len(expected) != len(set(expected)):
        raise ValueError("provided segments do not exactly match ordered close")
    for index, item in enumerate(manifest["segments"]):
        if item["sequence_index"] != index or item["object_sha256"] != item[
            "segment_id"
        ].removeprefix("sha256:"):
            raise ValueError("close sequence/content identity mismatch")
        if (
            index
            and manifest["segments"][index - 1]["timeline_end_ns"] != item["timeline_start_ns"]
        ):
            raise ValueError("cross-segment timeline gap or overlap")
    results = []
    for segment_id in expected:
        result = get_json(args.base_url, token, f"/v1/segments/{segment_id}/artifact-inspection")
        if (
            result["segment_id"] != segment_id
            or result["decoded_frames_equal_action_rows"] is not True
            or "action_quality" not in result
            or "visual_quality" not in result
        ):
            raise ValueError(f"incomplete segment audit: {segment_id}")
        result["codec_compatibility"] = codec_compatibility(result)
        results.append(result)
    action = [item["action_quality"] for item in results]
    visual = [item["visual_quality"] for item in results]
    rows = sum(item["row_count"] for item in action)
    pairs = sum(item["sampled_pair_count"] for item in visual)
    distinct_hashes = sorted(
        {value for item in action for value in item["distinct_applied_action_hashes"]}
    )
    truncated = any(item["distinct_hashes_truncated"] for item in action)
    aggregate = {
        "rows": rows,
        "decoded_frames": sum(item["video"]["decoded_frames"] for item in results),
        "invalid_rows": sum(item["invalid_rows"] for item in action),
        "bc_training_eligible_rows": sum(item["bc_training_eligible_rows"] for item in action),
        "human_owned_rows": sum(item["human_owned_rows"] for item in action),
        "non_neutral_rows": sum(item["non_neutral_rows"] for item in action),
        "non_neutral_fraction": sum(item["non_neutral_rows"] for item in action) / rows,
        "button_active_rows": sum(item["button_active_rows"] for item in action),
        "trigger_active_rows": sum(item["trigger_active_rows"] for item in action),
        "action_transitions": sum(item["action_transitions"] for item in action),
        "distinct_applied_action_count": len(distinct_hashes),
        "distinct_applied_action_set_sha256": hashlib.sha256(
            "".join(distinct_hashes).encode()
        ).hexdigest(),
        "distinct_hashes_truncated": truncated,
        "sampled_visual_pairs": pairs,
        "motion_active_fraction": sum(
            item["motion_active_fraction"] * item["sampled_pair_count"] for item in visual
        )
        / pairs,
        "repeated_frame_fraction": sum(
            item["repeated_frame_fraction"] * item["sampled_pair_count"] for item in visual
        )
        / pairs,
    }
    thresholds = {
        "invalid_rows": 0,
        "human_and_bc_fraction_min": 0.99,
        "non_neutral_fraction_min": 0.05,
        "distinct_actions_min": 8,
        "action_transitions_min": 8,
        "motion_active_fraction_min": 0.10,
        "repeated_frame_fraction_max": 0.90,
    }
    passed = (
        aggregate["invalid_rows"] == 0
        and aggregate["human_owned_rows"] / rows >= thresholds["human_and_bc_fraction_min"]
        and aggregate["bc_training_eligible_rows"] / rows >= thresholds["human_and_bc_fraction_min"]
        and aggregate["non_neutral_fraction"] >= thresholds["non_neutral_fraction_min"]
        and aggregate["distinct_applied_action_count"] >= thresholds["distinct_actions_min"]
        and aggregate["action_transitions"] >= thresholds["action_transitions_min"]
        and aggregate["motion_active_fraction"] >= thresholds["motion_active_fraction_min"]
        and aggregate["repeated_frame_fraction"] <= thresholds["repeated_frame_fraction_max"]
        and not truncated
    )
    payload = {
        "schema_id": "nxml.episode-artifact-audit.v1",
        "episode_id": args.episode_id,
        "close_id": args.close_id,
        "clock_id": manifest["clock_id"],
        "timeline_start_ns": manifest["timeline_start_ns"],
        "timeline_end_ns": manifest["timeline_end_ns"],
        "segments": results,
        "aggregate": aggregate,
        "thresholds": thresholds,
        "quality_gate_passed": passed,
        "mutation_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "quality_gate_passed": passed, **aggregate}))


if __name__ == "__main__":
    main()
