# Continuous segment contract v1 (inactive)

This document coordinates the future edge/control contract for continuous recording. The
models and catalog implementation exist in `nxml_control.segments`, but **no segment route
is registered in the current FastAPI application**. Deploying the current control-plane
release therefore does not activate or migrate this contract.

## Producer sequence and cleanup gate

1. Before segment 0, the edge generates one canonical UUID for `episode_id` and
   reuses it in every object key, bundle, member basename, receipt, quality
   disposition, and close manifest. Display names are separate metadata.
2. The edge creates an upload through the existing checksum-declared upload lifecycle.
   The compatible staged key is
   `uploads/segments/{episode_uuid}/{sequence_index:06d}-{object_sha256}.tar`;
   it satisfies the existing `/v1/uploads` `uploads/*.tar` contract. Commit does
   not rename or copy this object.
3. The edge commits the segment with `nxml.segment-bundle.v1`. The cluster checks the
   outer object size/SHA-256, all tar member sizes/SHA-256 values, and atomically registers
   the immutable segment and receipt.
4. An exact retry returns the original receipt. A changed request using the same upload or
   identity is a conflict. Invalid shape, checksum, member set, or timeline is invalid data.
5. Only after receiving and independently querying a `state=committed` receipt may the
   edge delete both its source segment and staged tar. The receipt is the durability gate;
   episode close, quality indexing, snapshots, training, and replication are downstream.
6. Episode close sends only an ordered list of committed segment identities. It never
   reuploads media. The cluster rejects missing segments, nonzero/duplicate sequence
   starts, gaps, overlaps, clock changes, or metadata/digest mismatches.

The eventual versioned routes should be:

- `POST /v1/segment-bundles/{upload_id}/commit`
- `GET /v1/segment-receipts/{receipt_id}`
- `GET /v1/segments/{segment_id}` (external committed manifest plus receipt)
- `POST /v1/datasets/{dataset_id}/episodes/{episode_id}/close`
- `GET /v1/episode-closes/{close_id}`
- `POST /v1/segments/{segment_id}/quality-dispositions`
- `GET /v1/segments/{segment_id}/quality-dispositions`
- `POST /v1/datasets/{dataset_id}/segment-snapshots`
- `GET /v1/segment-snapshots/{snapshot_id}`
- `GET /v1/datasets/{dataset_id}/segment-status`

Mutations require `Idempotency-Key` and cluster bearer authentication. The typed
mapping is 409 for identity/idempotency conflicts; 422 for schema, checksum, member,
order, gap, overlap, or metadata validation; and 404 for a missing upload, receipt,
segment, or close. These routes remain intentionally absent until activation is
coordinated with dagger-ui.

## Segment bundle

`segment_id` is the content address `sha256:<object_sha256>`. The timeline is half-open
`[timeline_start_ns, timeline_end_ns)` on one named monotonic `clock_id`. Each tar has
exactly one independently checksummed video/action/event triplet.

```json
{
  "schema_id": "nxml.segment-bundle.v1",
  "dataset_id": "nxml-pokemon-za-v2",
  "episode_id": "5e2d0f44-b532-46ff-b433-4082b12945ab",
  "segment_id": "sha256:<64 lowercase hex>",
  "sequence_index": 0,
  "clock_id": "linux-monotonic",
  "timeline_start_ns": 849000000000,
  "timeline_end_ns": 879000000000,
  "object_size_bytes": 1073741824,
  "object_sha256": "<64 lowercase hex>",
  "members": [
    {
      "role": "video",
      "path": "episode.000000.mkv",
      "size_bytes": 1000000000,
      "sha256": "<64 lowercase hex>"
    },
    {
      "role": "actions",
      "path": "episode.000000.parquet",
      "size_bytes": 1000000,
      "sha256": "<64 lowercase hex>"
    },
    {
      "role": "events",
      "path": "episode.000000.events.parquet",
      "size_bytes": 10000,
      "sha256": "<64 lowercase hex>"
    }
  ]
}
```

Member paths are safe relative paths. Action rows retain the strict
`nxml.episode.v2`/`switch_packets.v1` semantics, including policy proposal, human
proposal/mask, applied action, per-dimension ownership `0=unowned, 1=human, 2=policy`,
selected immutable policy revision, mute mask/version, monotonic timestamps, causal age,
validity, and controller/mode. A model, mode, or mute-mask change closes the logical
episode; it must not silently alter metadata inside a segment chain.

Cut points are defined only by the shared monotonic timeline. A segment owns
`[timeline_start_ns, timeline_end_ns)`; a frame/action/event belongs to it exactly
when `start <= timestamp < end`. Producers must not assume that the first timestamp
equals `start` or the last timestamp equals `end`.

The tar contains exactly the three declared members. The independently stored external
bundle manifest is returned by `GET /v1/segments/{segment_id}` with its immutable
receipt, allowing the cluster to export or reconstruct the complete bundle without an
embedded fourth manifest member.

## Receipt

```json
{
  "receipt_id": "<uuid>",
  "segment_id": "sha256:<object digest>",
  "upload_id": "<uuid>",
  "dataset_id": "nxml-pokemon-za-v2",
  "episode_id": "5e2d0f44-b532-46ff-b433-4082b12945ab",
  "sequence_index": 0,
  "storage_key": "uploads/segments/5e2d0f44-b532-46ff-b433-4082b12945ab/000000-<digest>.tar",
  "size_bytes": 1073741824,
  "sha256": "<64 lowercase hex>",
  "timeline_start_ns": 849000000000,
  "timeline_end_ns": 879000000000,
  "state": "committed",
  "committed_at": "<RFC3339 UTC>"
}
```

The local authoritative backend flushes the object and its containing directory before
catalog commit. The receipt, quality metadata, and episode close remain server-side after
edge cleanup.

## Episode close

```json
{
  "schema_id": "nxml.episode-close.v1",
  "dataset_id": "nxml-pokemon-za-v2",
  "episode_id": "5e2d0f44-b532-46ff-b433-4082b12945ab",
  "clock_id": "linux-monotonic",
  "timeline_start_ns": 849000000000,
  "timeline_end_ns": 939000000000,
  "segments": [
    {
      "segment_id": "sha256:<digest-0>",
      "sequence_index": 0,
      "timeline_start_ns": 849000000000,
      "timeline_end_ns": 879000000000,
      "object_sha256": "<digest-0>"
    },
    {
      "segment_id": "sha256:<digest-1>",
      "sequence_index": 1,
      "timeline_start_ns": 879000000000,
      "timeline_end_ns": 939000000000,
      "object_sha256": "<digest-1>"
    }
  ]
}
```

`close_id` is the SHA-256 content identity of the canonical close manifest. Close
validates only immutable identities, ordering, clocks, and timeline continuity; it never
claims that media/action contents passed training-quality validation. The reconstruction
iterator reads segments in `sequence_index` order from cluster object storage, rechecks
object size and SHA-256, and yields chunks bounded to at most 1 MiB. It never calls an
unbounded `read()` on a segment.

## Quality, snapshot, and compact status

Segment quality is append-only and idempotent:

```json
{
  "training_eligible": false,
  "reason": "noncausal_action_alignment",
  "validator": "edge-action-alignment",
  "validator_version": "1",
  "validator_state": "failed"
}
```

Snapshot creation fails closed and reads the latest disposition for every segment. A
segment is eligible only when that explicit latest disposition has both
`training_eligible=true` and `validator_state=passed`; a missing disposition is
recorded as `missing_quality_disposition` and excludes the episode. Every exclusion
records segment ID, reason, validator, validator version/state, and disposition ID. The canonical
`nxml.segment-snapshot.v1` body is content-addressed and stored append-only; later quality
changes create a different snapshot and cannot alter an existing one.

Compact status is bounded and contains only:

```json
{
  "schema_id": "nxml.segment-status.v1",
  "dataset_id": "nxml-pokemon-za-v2",
  "committed_segments": 12,
  "committed_bytes": 12884901888,
  "closed_episodes": 2,
  "eligible_episodes": 1,
  "excluded_episodes": 1,
  "latest_segment_committed_at": "<RFC3339 UTC>"
}
```

Search/indexing remains asynchronous and cannot block segment receipts, edge deletion,
episode close, snapshots, inference, training, or deployment.
