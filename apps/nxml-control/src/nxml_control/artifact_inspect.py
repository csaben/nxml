"""Read-only inspection of catalog-resolved immutable segment artifacts."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from torchcodec.decoders import VideoDecoder

from nxml_control.quality_inspect import inspect_actions, inspect_visual_dynamics

CHUNK = 1024 * 1024
ROLES = {"video", "actions", "events"}


def _copy_and_digest(source, target: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with target.open("xb") as output:
        while chunk := source.read(CHUNK):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def inspect_segment_artifact(segments, segment_id: str) -> dict:
    """Inspect one cataloged object without accepting paths or command arguments."""
    record = segments.get_segment(segment_id)
    manifest = record["manifest"]
    receipt = record["receipt"]
    if manifest["segment_id"] != segment_id or receipt["segment_id"] != segment_id:
        raise ValueError("catalog segment identity mismatch")
    if manifest["object_sha256"] != segment_id.removeprefix("sha256:"):
        raise ValueError("content address does not match manifest")
    declared = {member["path"]: member for member in manifest["members"]}
    if len(declared) != 3 or {member["role"] for member in declared.values()} != ROLES:
        raise ValueError("segment must declare exact video/actions/events members")
    with tempfile.TemporaryDirectory(prefix="nxml-artifact-inspect-") as raw_tmp:
        root = Path(raw_tmp)
        outer = root / "segment.tar"
        with segments.storage.open(receipt["storage_key"]) as source:
            outer_size, outer_sha = _copy_and_digest(source, outer)
        if (outer_size, outer_sha) != (receipt["size_bytes"], receipt["sha256"]):
            raise ValueError("outer object checksum mismatch")
        extracted: dict[str, Path] = {}
        member_evidence = []
        with tarfile.open(outer, "r:*") as archive:
            infos = archive.getmembers()
            if len(infos) != 3 or {info.name for info in infos} != set(declared):
                raise ValueError("tar member set mismatch")
            for index, info in enumerate(infos):
                if (
                    not info.isfile()
                    or Path(info.name).is_absolute()
                    or ".." in Path(info.name).parts
                ):
                    raise ValueError("unsafe tar member")
                source = archive.extractfile(info)
                if source is None:
                    raise ValueError("unreadable tar member")
                target = root / f"member-{index}"
                size, sha256 = _copy_and_digest(source, target)
                expected = declared[info.name]
                if (size, sha256) != (expected["size_bytes"], expected["sha256"]):
                    raise ValueError("member checksum mismatch")
                extracted[expected["role"]] = target
                member_evidence.append(
                    {"role": expected["role"], "size_bytes": size, "sha256": sha256}
                )
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,profile,level,pix_fmt,width,height,r_frame_rate,avg_frame_rate,time_base,has_b_frames,bit_rate:format=format_name,bit_rate:frame=key_frame",
                "-of",
                "json",
                str(extracted["video"]),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        parsed = json.loads(probe.stdout)
        streams = parsed.get("streams", [])
        if len(streams) != 1:
            raise ValueError("video must have exactly one selected stream")
        stream = streams[0]
        keyframes = [
            index
            for index, frame in enumerate(parsed.get("frames", []))
            if frame.get("key_frame") == 1
        ]
        intervals = [right - left for left, right in itertools.pairwise(keyframes)]
        action_table = pq.read_table(extracted["actions"])
        action_rows = action_table.num_rows
        action_quality = inspect_actions(action_table.to_pylist())
        event_rows = pq.ParquetFile(extracted["events"]).metadata.num_rows
        decoder = VideoDecoder(str(extracted["video"]), device="cpu")
        decoded_frames = decoder.metadata.num_frames
        visual_quality = inspect_visual_dynamics(decoder, decoded_frames, segment_id)
        if decoded_frames != action_rows:
            raise ValueError(f"video/action frame mismatch: {decoded_frames}!={action_rows}")
        return {
            "schema_id": "nxml.segment-artifact-inspection.v1",
            "segment_id": segment_id,
            "sequence_index": manifest["sequence_index"],
            "outer": {"size_bytes": outer_size, "sha256": outer_sha},
            "members": sorted(member_evidence, key=lambda item: item["role"]),
            "video": {
                "container": parsed.get("format", {}).get("format_name"),
                "codec": stream.get("codec_name"),
                "profile": stream.get("profile"),
                "level": stream.get("level"),
                "pixel_format": stream.get("pix_fmt"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "r_frame_rate": stream.get("r_frame_rate"),
                "avg_frame_rate": stream.get("avg_frame_rate"),
                "time_base": stream.get("time_base"),
                "max_b_frames": stream.get("has_b_frames"),
                "stream_bit_rate": int(stream["bit_rate"]) if stream.get("bit_rate") else None,
                "measured_bit_rate": int(parsed["format"]["bit_rate"])
                if parsed.get("format", {}).get("bit_rate")
                else None,
                "keyframe_indices": keyframes,
                "keyframe_intervals": intervals,
                "gop_size": max(intervals) if intervals else None,
                "decoded_frames": decoded_frames,
            },
            "action_rows": action_rows,
            "event_rows": event_rows,
            "action_quality": action_quality,
            "visual_quality": visual_quality,
            "decoded_frames_equal_action_rows": True,
        }
