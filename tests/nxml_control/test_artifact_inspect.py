import hashlib
import io
import json
import tarfile
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nxml_control.artifact_inspect import inspect_segment_artifact


class Storage:
    def __init__(self, payload):
        self.payload = payload

    def open(self, key):
        assert key == "catalog/resolved.tar"
        return io.BytesIO(self.payload)


def parquet_bytes(rows):
    sink = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), sink)
    return sink.getvalue()


def fixture(tmp_path, *, unsafe=False):
    members = {
        "episode.000000.mkv": b"video",
        "episode.000000.parquet": parquet_bytes([{"frame_idx": 0}, {"frame_idx": 1}]),
        "episode.000000.events.parquet": parquet_bytes([]),
    }
    roles = ("video", "actions", "events")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for index, (name, payload) in enumerate(members.items()):
            tar_name = "../escape" if unsafe and index == 0 else name
            info = tarfile.TarInfo(tar_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    payload = buffer.getvalue()
    outer_sha = hashlib.sha256(payload).hexdigest()
    declarations = []
    for role, (name, value) in zip(roles, members.items(), strict=True):
        declarations.append(
            {
                "role": role,
                "path": "../escape" if unsafe and role == "video" else name,
                "size_bytes": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        )
    segment_id = "sha256:" + outer_sha
    record = {
        "manifest": {
            "segment_id": segment_id,
            "sequence_index": 0,
            "object_sha256": outer_sha,
            "members": declarations,
        },
        "receipt": {
            "segment_id": segment_id,
            "storage_key": "catalog/resolved.tar",
            "size_bytes": len(payload),
            "sha256": outer_sha,
        },
    }
    return SimpleNamespace(storage=Storage(payload), get_segment=lambda _: record), segment_id


def test_inspector_is_catalog_resolved_and_returns_exact_probe(monkeypatch, tmp_path):
    segments, segment_id = fixture(tmp_path)
    probe = {
        "streams": [
            {
                "codec_name": "h264",
                "profile": "High",
                "level": 42,
                "pix_fmt": "yuv420p",
                "width": 1280,
                "height": 720,
                "r_frame_rate": "60/1",
                "avg_frame_rate": "60/1",
                "time_base": "1/1000",
                "has_b_frames": 0,
            }
        ],
        "format": {"format_name": "matroska,webm", "bit_rate": "16000000"},
        "frames": [{"key_frame": 1}, {"key_frame": 0}],
    }
    monkeypatch.setattr(
        "nxml_control.artifact_inspect.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(probe)),
    )
    monkeypatch.setattr(
        "nxml_control.artifact_inspect.VideoDecoder",
        lambda *args, **kwargs: SimpleNamespace(metadata=SimpleNamespace(num_frames=2)),
    )
    result = inspect_segment_artifact(segments, segment_id)
    assert result["outer"]["sha256"] == segment_id.removeprefix("sha256:")
    assert result["video"]["max_b_frames"] == 0
    assert result["decoded_frames_equal_action_rows"] is True


def test_inspector_rejects_unsafe_tar_member(tmp_path):
    segments, segment_id = fixture(tmp_path, unsafe=True)
    with pytest.raises(ValueError, match="unsafe tar member"):
        inspect_segment_artifact(segments, segment_id)
