#!/usr/bin/env python3
"""Read-only strict validator for closed rolling episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tarfile
import tempfile
from itertools import pairwise
from pathlib import Path

import pyarrow.parquet as pq
from nxml_core.contracts import DaggerActionRecordV2
from torchcodec.decoders import VideoDecoder

CHUNK = 1024 * 1024
ROLES = {"video", "actions", "events"}
ACTION_SCHEMA_ID = "nxml.dagger-actions.v2"
ACTION_SCHEMA_MARKERS = {"row_schema_id", "action_schema_id", "action_rows_schema_id"}
EDGE_ACTION_COLUMNS = {
    "frame_idx", "timestamp", "frame_monotonic_ns", "frame_timestamp_ns",
    "action_timestamp", "action_monotonic_ns", "action_timestamp_ns", "action_age_ns",
    "action_age", "valid", "invalid_reasons", "action", "applied_action",
    "human_action", "human_mask", "policy_action", "ownership", "controller_id",
    "active_driver", "controller", "policy_id", "policy_revision", "policy_digest",
    "human_monotonic_ns", "policy_monotonic_ns", "policy_observation_monotonic_ns",
    "muted_policy_action", "mute_mask", "mute_mask_version", "ownership_source",
    "mode", "takeover", "takeover_reason", "takeover_release_remaining_ns",
    "proposal_valid", "proposal_fresh", "proposal_sequence", "proposal_age_ns",
    "gap_state", "gap_reason", "gap_duration_ns", "boundary_sequence",
    "boundary_acknowledged", "bc_training_eligible",
}


def validate_action_schema(column_names: list[str], rows: list[dict]) -> str:
    columns = set(column_names)
    marker_columns = columns & ACTION_SCHEMA_MARKERS
    physical = columns - marker_columns
    if physical != EDGE_ACTION_COLUMNS:
        missing = sorted(EDGE_ACTION_COLUMNS - physical)
        extra = sorted(physical - EDGE_ACTION_COLUMNS)
        raise ValueError(f"wrong action schema columns: missing={missing}, extra={extra}")
    encountered = {
        row.get(marker)
        for row in rows
        for marker in marker_columns
        if row.get(marker) is not None
    }
    if encountered and encountered != {ACTION_SCHEMA_ID}:
        raise ValueError(f"wrong action schema marker: encountered={sorted(map(str, encountered))}")
    return ACTION_SCHEMA_ID


def digest(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(CHUNK):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def copy_member(source, target: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with target.open("xb") as output:
        while chunk := source.read(CHUNK):
            output.write(chunk)
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def validate_episode(db, objects: Path, episode_id: str) -> dict:
    close = db.execute("SELECT * FROM episode_closes WHERE episode_id=?", (episode_id,)).fetchone()
    if close is None:
        raise ValueError(f"episode not closed: {episode_id}")
    refs = db.execute(
        "SELECT b.* FROM episode_close_segments x JOIN segment_bundles b ON b.segment_id=x.segment_id WHERE x.close_id=? ORDER BY x.ordinal",
        (close["close_id"],),
    ).fetchall()
    manifest = json.loads(close["manifest_json"])
    expected = manifest["segments"]
    if len(refs) != len(expected):
        raise ValueError("close/catalog segment count mismatch")
    reports = []
    prior_end = None
    for ordinal, (row, ref) in enumerate(zip(refs, expected, strict=True)):
        bundle = json.loads(row["manifest_json"])
        sequence = bundle["sequence_index"]
        if (
            ordinal != sequence
            or ref["sequence_index"] != sequence
            or ref["segment_id"] != row["segment_id"]
        ):
            raise ValueError("close ordering/identity mismatch")
        if prior_end is not None and bundle["timeline_start_ns"] != prior_end:
            raise ValueError("timeline gap or overlap")
        prior_end = bundle["timeline_end_ns"]
        path = objects / row["storage_key"]
        outer_size, outer_sha = digest(path)
        if (outer_size, outer_sha) != (row["size_bytes"], row["sha256"]):
            raise ValueError("outer object digest mismatch")
        members = bundle["members"]
        if len(members) != 3 or {m["role"] for m in members} != ROLES:
            raise ValueError("bundle is not an exact triplet")
        expected_members = {m["path"]: m for m in members}
        extracted = {}
        with tempfile.TemporaryDirectory(prefix="nxml-segment-validate-") as tmp:
            root = Path(tmp)
            with tarfile.open(path, "r:*") as archive:
                infos = archive.getmembers()
                if len(infos) != 3 or {i.name for i in infos} != set(expected_members):
                    raise ValueError("tar member set mismatch")
                for info in infos:
                    if (
                        not info.isfile()
                        or Path(info.name).is_absolute()
                        or ".." in Path(info.name).parts
                    ):
                        raise ValueError("unsafe tar member")
                    source = archive.extractfile(info)
                    if source is None:
                        raise ValueError("unreadable tar member")
                    target = root / Path(info.name).name
                    size, sha = copy_member(source, target)
                    expected_member = expected_members[info.name]
                    if (size, sha) != (expected_member["size_bytes"], expected_member["sha256"]):
                        raise ValueError("member digest mismatch")
                    extracted[expected_member["role"]] = target
            action_table = pq.read_table(extracted["actions"])
            rows = action_table.to_pylist()
            if not rows:
                raise ValueError("empty actions parquet")
            validate_action_schema(action_table.column_names, rows)
            eligible = 0
            prior_frame = None
            frame_ids = []
            for raw in rows:
                normalized = {key: value for key, value in raw.items() if key not in ACTION_SCHEMA_MARKERS}
                parsed = DaggerActionRecordV2.model_validate(normalized)
                frame_ns = parsed.effective_frame_ns
                frame_id = parsed.effective_frame_index
                if frame_ns is None or (prior_frame is not None and frame_ns <= prior_frame):
                    raise ValueError("non-monotonic frame timeline")
                prior_frame = frame_ns
                frame_ids.append(frame_id)
                valid = parsed.valid and parsed.applied_action_valid is not False
                if valid:
                    if parsed.effective_action_ns is None or parsed.effective_action_ns > frame_ns:
                        raise ValueError("noncausal valid action")
                    if parsed.bc_training_eligible is not True:
                        raise ValueError("valid human row lacks explicit BC eligibility")
                    eligible += 1
                elif parsed.bc_training_eligible:
                    raise ValueError("invalid row marked BC eligible")
            if any(b <= a for a, b in pairwise(frame_ids)):
                raise ValueError("duplicate/out-of-order frame sequence")
            event_rows = pq.ParquetFile(extracted["events"]).metadata.num_rows
            video_frames = VideoDecoder(str(extracted["video"]), device="cpu").metadata.num_frames
            if video_frames != len(rows):
                raise ValueError(f"video/action frame mismatch: {video_frames}!={len(rows)}")
        reports.append(
            {
                "sequence_index": sequence,
                "segment_id": row["segment_id"],
                "receipt_id": db.execute(
                    "SELECT receipt_id FROM segment_receipts WHERE segment_id=?",
                    (row["segment_id"],),
                ).fetchone()[0],
                "size_bytes": outer_size,
                "rows": len(rows),
                "eligible_rows": eligible,
                "video_frames": video_frames,
                "event_rows": event_rows,
            }
        )
    if (
        manifest["timeline_start_ns"] != refs[0]["timeline_start_ns"]
        or manifest["timeline_end_ns"] != refs[-1]["timeline_end_ns"]
    ):
        raise ValueError("close timeline mismatch")
    return {
        "episode_id": episode_id,
        "close_id": close["close_id"],
        "segments": reports,
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_ids", nargs="+")
    parser.add_argument("--state-dir", default="/var/lib/nxml-control")
    args = parser.parse_args()
    state = Path(args.state_dir)
    db = sqlite3.connect(f"file:{state / 'catalog.sqlite3'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    print(
        json.dumps(
            {
                "schema_id": "nxml.rolling-validation.v1",
                "episodes": [
                    validate_episode(db, state / "objects", episode) for episode in args.episode_ids
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
