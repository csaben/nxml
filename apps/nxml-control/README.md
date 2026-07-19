# nxml-control

Durable ingest and ML control plane. Default bind is `127.0.0.1:8787`; on cradle bind to its Tailnet IP, never `0.0.0.0`, and rely on Tailnet identity at the reverse proxy until service tokens are added.

Producer contract:
1. `POST /v1/uploads` with required `Idempotency-Key` and `{object_key,size_bytes,sha256}`.
2. `PUT` raw immutable tar bytes to returned `upload_url`.
3. Optionally `POST .../inspect`; checksum mismatch is `422`.
4. `POST .../commit` with `{dataset_id,shard_id,manifest}`. Commit before verification is `409`.

An exact create retry returns the same upload. Reusing an idempotency key with changed fields is `409`. Object keys and shard IDs are globally unique and immutable. Successful commit retries return the existing registration. UI may use `/healthz`; generated OpenAPI is at `/openapi.json`.

Storage is behind `ObjectStorage`. `LocalObjectStorage` supports credential-free tests; S3/HF Storage Buckets implement the same `put_if_absent`, `inspect`, and `open` operations without coupling catalog state to Git history.

Training job and worker details are versioned in [`TRAINING_WORKER.md`](TRAINING_WORKER.md). Production startup requires a configured real worker; the fake executor is opt-in development behavior.

Episode v2 Parquet ownership is integer encoded: `0` unowned, `1` human, `2` policy. `human_action_mask` separately indicates explicit human inputs.

## Strict commit manifest

Commit accepts only this field mapping; unknown fields are rejected:

- `schema_id`: exactly `nxml.episode.v2`
- `action_spec_id`: exactly `switch_packets.v1`
- `episodes`: non-empty objects with unique `episode_id`
- `members`: unique `path`, `kind`, `size_bytes`, lowercase SHA-256 `sha256`
- member kinds: `video`, `actions`, `events`, `episode_manifest`, `other`
- at least one `actions` `.parquet` and one `events` `.events.parquet` member

Parquet action rows use `frame_index`, `frame_timestamp_ns`, `action_timestamp_ns`, `action_age_ns`, `applied_action`, `human_action`, `human_action_mask`, `policy_action`, `controller`, integer `ownership`, `policy_id`, `policy_revision`, `valid`, and `invalid_reasons`. Nanosecond timestamps use the manifest monotonic clock mapping; never substitute wall-clock seconds. Ownership is 0 unowned, 1 human, 2 policy. Human demonstration selection is `any(human_action_mask) OR 1 in ownership`; invalid rows are always excluded.

A successful commit returns and persists `commit_id`, `upload_id`, `checksum`, `size_bytes`, `storage_key`, `state`, `committed_at`, `dataset_id`, and `shard_id`. Read it later at `GET /v1/commits/{commit_id}`. Exact retries return that original receipt. Schema failures are 422; immutable identity conflicts are 409.

## Search metadata foundation

Search indexing is strictly asynchronous and downstream of committed episodes. `index_versions`, `derived_artifacts`, and `annotations` are never consulted by upload commit/receipt acknowledgement, edge source deletion, inference, snapshot training, model promotion, or rollback. No embedding compute or vector database is included.

Canonical clip references are `{dataset_id, episode_id, window_start_ns, window_end_ns}` with a half-open `[start_ns, end_ns)` window. Artifact episode-level references omit both window fields; annotations require both. Indexes and artifacts use explicit `pending -> running -> complete|failed` transitions and exact idempotency keys. Reserved artifact metadata includes `artifact_type`, `embedding_model`, `embedding_version`, `labels`, `derived_features`, `source_snapshot_id`, `index_version`, and `status`.

## Training eligibility and edge cleanup

Episode quality is append-only server-side metadata under `nxml.episode-quality.v1`. Set it with authenticated `POST /v1/datasets/{dataset_id}/episodes/{episode_id}/quality-dispositions` and an `Idempotency-Key`; audit it with the matching GET. Snapshot creation uses the latest disposition, excludes `training_eligible=false` episodes by default, records exclusions in the immutable `nxml.dataset-snapshot.v1` manifest, and returns 422 when nothing remains eligible. Raw shard and receipt bytes are never mutated.

A verified commit receipt is sufficient for edge cleanup. The authoritative bytes are under the control-plane object root (`/var/lib/nxml-control/objects/<storage_key>` in the cradle deployment), while receipt, episode, quality, and snapshot evidence persist in `/var/lib/nxml-control/catalog.sqlite3`. Training consumes these cluster objects only. See [`HUGGING_FACE_ARCHITECTURE.md`](HUGGING_FACE_ARCHITECTURE.md) for asynchronous replication design.
