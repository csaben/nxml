# nxml-control

Durable ingest and ML control plane. Default bind is `127.0.0.1:8787`; on cradle bind to its Tailnet IP, never `0.0.0.0`, and rely on Tailnet identity at the reverse proxy until service tokens are added.

Producer contract:
1. `POST /v1/uploads` with required `Idempotency-Key` and `{object_key,size_bytes,sha256}`.
2. `PUT` raw immutable tar bytes to returned `upload_url`.
3. Optionally `POST .../inspect`; checksum mismatch is `422`.
4. `POST .../commit` with `{dataset_id,shard_id,manifest}`. Commit before verification is `409`.

An exact create retry returns the same upload. Reusing an idempotency key with changed fields is `409`. Object keys and shard IDs are globally unique and immutable. Successful commit retries return the existing registration. UI may use `/healthz`; generated OpenAPI is at `/openapi.json`.

Storage is behind `ObjectStorage`. `LocalObjectStorage` supports credential-free tests; S3/HF Storage Buckets implement the same `put_if_absent`, `inspect`, and `open` operations without coupling catalog state to Git history.

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
