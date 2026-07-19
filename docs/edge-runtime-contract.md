# Edge runtime contract

`nxml-autopilot` exposes the runtime contract consumed by the separate
`nxml-edge` supervisor. All endpoints except `/health` require the configured
`X-Autopilot-Token` header or `?token=` query value.

## Endpoints

- `GET /runtime/status` — complete status snapshot.
- `POST /runtime/ai {"enabled": bool}` — enable/disable policy; enabling while
  ejected returns HTTP 409.
- `POST /runtime/mode {"mode": "human-priority"|"human-takeover"}`.
- `POST /runtime/eject` — idempotently latch neutral output, disable policy,
  stop macro/mash, persist the latch, and post neutral immediately.
- `POST /runtime/rearm` — idempotently clear the persistent latch. Policy stays
  disabled until explicitly enabled.

## Status model

- `active_driver`: schema-v2 aggregate enum: `human`, `policy`, `blended`,
  `safety`, or `none`.
- `driver_detail`: operational detail such as `human+policy`, `macro`,
  `mash_a`, `ejected`, or `neutral`.
- `policy`: `id`, `revision`, `previous_revision`, `endpoint`, `state`,
  `ready`, inference count/timestamp/age/latency, and last error. Capture
  staleness or a latched eject suppresses readiness.
- `capture`: V4L2 device, open state, last frame monotonic timestamp, age in
  milliseconds, and stale flag.
- `controller`: source ID, transport/sample age and freshness, plus separate
  meaningful-input age.
- `orchestrator`: reachability/connection payload, checked age, cumulative
  success count, cumulative error count, and latest error.
- `spool`: atomically read `status.json`, including thresholds, disk pressure,
  pending/staging state, age, and `admission_open`.
- `ejected`: persistent safety latch.

For the REST ingest backend, `spool` additionally exposes:

- `backend="control-plane"`, `cluster_connected`, and `cluster_error`;
- `cluster_upload_counts` (`created`, `uploaded`, `committed`);
- `dataset_count`, `dataset_shard_count`, and `dataset_episode_count`;
- `snapshot_count` (currently `null`, because the control API has lookup but no
  snapshot-list endpoint);
- `active_policy_revision`, `previous_policy_revision`, and
  `deployment_generation`;
- `episodes_blocked`, per-episode durable conflict details,
  `backend_state.last_error_kind`, and receipt-bearing uploaded-shard journal
  entries.

`policy.cluster_active_revision`, `policy.previous_revision`, and
`policy.deployment_generation` mirror those deployment fields for supervisor
consumers. `policy.active_revision` remains the revision actually loaded by
the local autopilot process; the distinction prevents catalog intent from
being mistaken for completed edge activation.

## Edge-v2 to strict cluster-v2 mapping

The explicit mapper is `nxml_capture.schema_v2_compat`. It applies:

| Early edge field | Strict field | Rule |
| --- | --- | --- |
| `frame_idx` | `frame_index` | Integer rename. |
| `frame_monotonic_ns` | `frame_timestamp_ns` | Direct monotonic-ns copy only. |
| `action_monotonic_ns` | `action_timestamp_ns` | Direct monotonic-ns copy only. |
| derived difference | `action_age_ns` | Exact `frame - action`; after-frame actions are invalid projections with original value in provenance. |
| `action` / `applied_action` | `applied_action` | Applied/transmitted vector. |
| `human_mask` | `human_action_mask` | Explicit contribution mask. |
| `active_driver` | `controller` | `human`, `policy`, `human+policy→blended`, macro/mash/eject→`safety`, neutral→`none`. |
| `action_spec` | `action_spec_id` | Identifier preserved; `switch_packets.v1` is never mutated. |
| files object | files list | `bytes→size_bytes`, `sha256→digest`, algorithm=`sha256`. |

Unix wall-clock `timestamp` and `action_timestamp` values are never multiplied
or otherwise reinterpreted as monotonic nanoseconds. Records lacking monotonic
timestamps are emitted as invalid strict-shaped projections with
`invalid_reasons` and migration provenance. Early manifests lacking strict
clock origins or monotonic frame bounds are likewise marked incompatible; UTC
creation time is not substituted for monotonic bounds.
