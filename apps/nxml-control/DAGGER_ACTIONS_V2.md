# DAgger action and provenance contract v2

This is the producer/consumer contract for new DAgger recordings. It is
additive to `nxml.episode.v2` and does not change `switch_packets.v1` or the
legacy human-only Parquet profile already stored on cradle.

## Version negotiation

Each new DAgger episode entry in the committed shard manifest must contain:

```json
{
  "episode_id": "stable-uuid-created-before-frame-0",
  "action_rows_schema_id": "nxml.dagger-actions.v2"
}
```

Every action row for that episode must have
`row_schema_id="nxml.dagger-actions.v2"` and
`action_spec_id="switch_packets.v1"`. Do not add fields conditionally under
the same version: nullable fields must exist as typed nullable Parquet columns.
A model, mode, or mute-mask change closes the episode. Existing episodes with
no `action_rows_schema_id` remain the accepted legacy edge-v2 profile and keep
their current snapshot/training behavior.

## Exact Parquet fields

All timestamps ending in `_monotonic_ns` are unsigned/int64 nanoseconds.
Packets are fixed-size/list float32 `[26]`; masks are fixed-size/list boolean
`[26]`; ownership is fixed-size/list int8 `[26]`.

| Field | Type | Meaning |
| --- | --- | --- |
| `row_schema_id` | string, exact | `nxml.dagger-actions.v2` |
| `action_spec_id` | string, exact | `switch_packets.v1` |
| `frame_idx` | int64, nonnegative | Complete zero-based media-frame sequence |
| `frame_monotonic_ns` | int64 | Edge capture clock; strictly increasing |
| `policy_action` | float32[26], nullable | Unmuted cluster proposal |
| `policy_source_frame_monotonic_ns` | int64, nullable | Edge timestamp sent with the proposal's source frame |
| `policy_cluster_proposal_monotonic_ns` | int64, nullable | Opaque cluster-clock timestamp returned by inference; never subtract from edge time |
| `policy_action_monotonic_ns` | int64, nullable | Edge-clock time the proposal became available to arbitration |
| `policy_action_age_ns` | int64, nullable | `frame_monotonic_ns - policy_action_monotonic_ns` |
| `policy_action_valid`, `policy_action_fresh` | bool | Decode/identity validity and producer freshness decision |
| `human_action` | float32[26], nullable | Full human packet, not only explicitly moved dimensions |
| `human_action_monotonic_ns` | int64, nullable | Edge-clock human sample time |
| `human_action_age_ns` | int64, nullable | Frame minus human sample time |
| `human_action_valid`, `human_action_fresh` | bool | Human source validity/freshness |
| `muted_policy_action` | float32[26], nullable | Policy proposal after forcing every `mute_mask=true` dimension to neutral zero |
| `muted_policy_action_monotonic_ns` | int64, nullable | Same edge availability timestamp as `policy_action_monotonic_ns` |
| `applied_action` | float32[26] | Exact packet sent to the controller after arbitration |
| `applied_action_monotonic_ns` | int64 | Edge-clock application/sample time, not cluster time |
| `applied_action_age_ns` | int64 | Frame minus applied time; must be causal |
| `applied_action_valid` | bool | Controller-bound packet was valid |
| `human_mask` | bool[26] | Explicit physical human inputs only; independent of ownership |
| `mute_mask` | bool[26] | `true` means neutralize that AI dimension before arbitration |
| `mute_mask_version` | string, exact | `switch_packets.v1` |
| `ownership` | int8[26] | `0=unowned/neutral`, `1=human`, `2=policy` |
| `ownership_source` | enum string | `neutral`, `human_takeover`, `per_dimension`, or `policy` |
| `mode` | enum string | `human_only`, `policy_only`, or `hybrid` |
| `takeover_active` | bool | Full HumanTakeover state |
| `policy_revision_id` | string, nullable | Immutable control-plane revision ID |
| `policy_checkpoint_sha256` | lowercase hex[64], nullable | Exact INFO/artifact digest |
| `bc_training_eligible` | bool | Explicit producer row eligibility |
| `valid` | bool | Overall row validity |
| `invalid_reasons` | list<string> | Empty only for valid rows |

Policy fields, immutable revision, and digest are all null in `human_only`.
They are all required in `policy_only` and `hybrid`. The edge must verify INFO
revision+digest before setting `policy_action_valid=true`. It decides freshness
against its configured maximum using edge-clock age, records both the age and
boolean, and fails closed to neutral/human behavior when stale.

For `ownership_source=human_takeover`, `takeover_active=true` and all 26
ownership values must be `1`, even though `human_mask` records only explicit
inputs. Per dimension, `applied_action[i]` must equal full `human_action[i]`
when owner is `1`, `muted_policy_action[i]` when owner is `2`, and neutral zero
when owner is `0`. These equalities are validated exactly.

## Ingest, snapshot, and BC rules

Raw shard receipt remains independent of training quality: valid immutable
bytes commit even before semantic validation. For an episode declaring the
DAgger row schema, snapshot creation fails closed until the latest append-only
`nxml.episode-quality.v1` disposition explicitly has
`training_eligible=true`. Missing or false dispositions are excluded and the
snapshot records the exclusion; raw data and receipts are never changed.

The DAgger BC reader accepts a row only when all of these hold:

1. strict schema/provenance validation succeeds;
2. `valid`, `applied_action_valid`, and `bc_training_eligible` are true;
3. at least one dimension is human-owned (`ownership` contains `1`);
4. the immutable snapshot requests `control_source=human` or `all`.

The target is always the complete `applied_action[26]`. The reader never trains
against `policy_action`, `muted_policy_action`, or `human_action` directly.
Pure-policy DAgger rows are retained as provenance but excluded from this BC
bootstrap path. Episode-level explicit quality and row-level eligibility are
both required.
