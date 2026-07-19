# Continuous segment contract v1 (inactive)

This document coordinates the future edge/control contract for continuous recording. The
models and catalog implementation exist in `nxml_control.segments`, but **no segment route
is registered in the current FastAPI application**. Deploying the current control-plane
release therefore does not activate or migrate this contract.

## Producer sequence and cleanup gate

1. The edge creates an upload through the existing checksum-declared upload lifecycle,
   uploads one immutable tar bundle, and asks the cluster to inspect it.
2. The edge commits the segment with `nxml.segment-bundle.v1`. The cluster checks the
   outer object size/SHA-256, all tar member sizes/SHA-256 values, and atomically registers
   the immutable segment and receipt.
3. An exact retry returns the original receipt. A changed request using the same upload or
   identity is a conflict. Invalid shape, checksum, member set, or timeline is invalid data.
4. Only after receiving and independently querying a `state=committed` receipt may the
   edge delete both its source segment and staged tar. The receipt is the durability gate;
   episode close, quality indexing, snapshots, training, and replication are downstream.
5. Episode close sends only an ordered list of committed segment identities. It never
   reuploads media. The cluster rejects missing segments, nonzero/duplicate sequence
   starts, gaps, overlaps, clock changes, or metadata/digest mismatches.

The eventual versioned routes should be:

- `POST /v1/segment-bundles/{upload_id}/commit`
- `GET /v1/segment-receipts/{receipt_id}`
- `POST /v1/datasets/{dataset_id}/episodes/{episode_id}/close`
- `GET /v1/episode-closes/{close_id}`
- `POST /v1/segments/{segment_id}/quality-dispositions`
- `GET /v1/datasets/{dataset_id}/segment-status`

Mutations require `Idempotency-Key` and cluster bearer authentication. Suggested API
mapping is HTTP 409 for identity/idempotency conflict and 422 for invalid schema,
checksum, members, or timeline. These routes remain intentionally absent until activation
is coordinated with dagger-ui.

## Segment bundle

`segment_id` is the content address `sha256:<object_sha256>`. The timeline is half-open
`[timeline_start_ns, timeline_end_ns)` on one named monotonic `clock_id`. Each tar has
exactly one independently checksummed video/action/event triplet.

```json
{
  "schema_id": "nxml.segment-bundle.v1",
  "dataset_id": "nxml-pokemon-za-v2",
  "episode_id": "dagger-human-20260719-180350",
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

## Receipt

```json
{
  "receipt_id": "<uuid>",
  "segment_id": "sha256:<object digest>",
  "upload_id": "<uuid>",
  "dataset_id": "nxml-pokemon-za-v2",
  "episode_id": "dagger-human-20260719-180350",
  "sequence_index": 0,
  "storage_key": "segments/<digest>.tar",
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
  "episode_id": "dagger-human-20260719-180350",
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

`close_id` is the SHA-256 content identity of the canonical close manifest. The BC
reconstruction iterator reads segments in `sequence_index` order from cluster object
storage and rechecks object size and SHA-256 before yielding bytes.

## Quality, snapshot, and compact status

Segment quality is append-only and idempotent:

```json
{
  "training_eligible": false,
  "reason": "noncausal_action_alignment",
  "validator": "edge-action-alignment",
  "validator_version": "1"
}
```

Snapshot creation reads the latest disposition for every segment. A single ineligible
segment excludes its closed episode by default and records segment ID, reason, validator,
version, and disposition ID in `excluded_episodes`. The canonical
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
