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
