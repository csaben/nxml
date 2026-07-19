import hashlib
import io
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq


def fixture_shard(episode_id="e1"):
    def parquet(table):
        sink = io.BytesIO()
        pq.write_table(pa.table(table), sink)
        return sink.getvalue()

    members = {
        f"{episode_id}.parquet": parquet(
            {
                "frame_index": [0, 1, 2],
                "valid": [True, True, False],
                "human_action_mask": [[False], [True], [True]],
                "ownership": [[0], [1], [1]],
            }
        ),
        f"{episode_id}.events.parquet": parquet({"event_id": ["start"], "timestamp_ns": [1]}),
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    manifest = {
        "schema_id": "nxml.episode.v2",
        "action_spec_id": "switch_packets.v1",
        "episodes": [{"episode_id": episode_id}],
        "members": [],
    }
    for name, data in sorted(members.items()):
        manifest["members"].append(
            {
                "path": name,
                "kind": "events" if name.endswith(".events.parquet") else "actions",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return stream.getvalue(), manifest
