"""Strict semantic validation for published NXML WebDataset segment shards."""

from __future__ import annotations

import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from nxml_core.contracts import DaggerActionRecordV2
from nxml_core.contracts.webdataset_v1 import PublishedShardV1
from torchcodec.decoders import VideoDecoder

from nxwm_mira.data.webdataset import CHUNK_BYTES, verify_shard


def validate_segment_semantics(path: Path, shard: PublishedShardV1) -> dict[str, int | str]:
    """Fail closed on action schema/order, decoded alignment, and codec lineage."""
    verify_shard(path, shard)
    by_role = {member.role: member for member in shard.members}
    with tempfile.TemporaryDirectory(prefix="nxml-wds-validate-") as raw_tmp:
        root = Path(raw_tmp)
        extracted: dict[str, Path] = {}
        with tarfile.open(path, "r:*") as archive:
            for role, member in by_role.items():
                source = archive.extractfile(member.path)
                if source is None:
                    raise ValueError("unreadable tar member")
                target = root / Path(member.path).name
                with target.open("xb") as output:
                    while chunk := source.read(CHUNK_BYTES):
                        output.write(chunk)
                extracted[role] = target
        rows = pq.read_table(extracted["actions"]).to_pylist()
        if not rows:
            raise ValueError("empty action parquet")
        prior_ns = -1
        for expected_index, raw in enumerate(rows):
            parsed = DaggerActionRecordV2.model_validate(raw)
            if parsed.effective_frame_index != expected_index:
                raise ValueError("action frame sequence is missing, duplicate, or out of order")
            frame_ns = parsed.effective_frame_ns
            if frame_ns is None or frame_ns <= prior_ns:
                raise ValueError("action frame timeline is not strictly monotonic")
            prior_ns = frame_ns
        frame_count = VideoDecoder(str(extracted["video"]), device="cpu").metadata.num_frames
        if frame_count != len(rows):
            raise ValueError(f"video/action frame mismatch: {frame_count}!={len(rows)}")
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=codec_name,profile,level,pix_fmt,width,height,r_frame_rate,time_base",
                "-of", "json", str(extracted["video"]),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        fps_num, fps_den = (int(value) for value in stream["r_frame_rate"].split("/"))
        actual = {
            "codec": stream["codec_name"], "profile": stream["profile"],
            "level": stream.get("level"), "pixel_format": stream["pix_fmt"],
            "width": stream["width"], "height": stream["height"],
            "nominal_fps": fps_num / fps_den, "time_base": stream["time_base"],
        }
        expected = shard.codec.model_dump(exclude={"container", "gop_size", "aspect_mode"})
        if actual != expected:
            raise ValueError(f"codec lineage mismatch: {actual} != {expected}")
        return {"frames": frame_count, "action_rows": len(rows), "codec": stream["codec_name"]}
