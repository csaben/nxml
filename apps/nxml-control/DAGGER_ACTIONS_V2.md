# DAgger action and provenance contract v2

This contract matches the physical Parquet writer deployed by dagger-ui at
`9b570c2`. It is additive to `nxml.episode.v2`; existing human-only rows remain
accepted unchanged. Unknown Parquet fields are rejected by the shared strict
validator.

## Version marker

The episode JSON manifest uses:

```json
{"action_schema_id":"nxml.dagger-actions.v2","action_schema_version":2}
```

The shard API manifest must copy `action_schema_id` into its corresponding
`episodes[]` entry. `action_rows_schema_id` is accepted as a compatibility
alias. Until that marker is present, the cluster cannot enforce explicit
episode-quality gating before snapshot creation.

## Exact physical columns

All action packets are fixed-size float32 lists of length 26. `human_mask` and
`mute_mask` are fixed-size boolean lists of length 26. `ownership` is a
fixed-size uint8 list using `0=unowned/neutral`, `1=human`, `2=policy`.

| Column | Parquet type | Contract |
| --- | --- | --- |
| `frame_idx` | int64 | Complete zero-based media-frame index |
| `timestamp` | float64 | Compatibility wall timestamp |
| `frame_monotonic_ns` | nullable int64 | Edge frame clock |
| `frame_timestamp_ns` | nullable int64 | Must equal `frame_monotonic_ns` |
| `action_timestamp` | nullable float64 | Compatibility wall timestamp |
| `action_monotonic_ns` | nullable int64 | Applied arbitration time on edge clock |
| `action_timestamp_ns` | nullable int64 | Must equal `action_monotonic_ns` |
| `action_age_ns` | nullable int64 | Frame minus applied timestamp |
| `action_age` | float64 | Seconds; must equal `action_age_ns / 1e9` for valid rows |
| `valid` | bool | Overall causal/action validity |
| `invalid_reasons` | list<string> | Empty for valid rows; non-empty for invalid rows |
| `action` | float32[26] | Compatibility alias; exactly equals `applied_action` |
| `applied_action` | float32[26] | Exact controller-bound arbitration output |
| `human_action` | float32[26] | Full human proposal or neutral packet |
| `human_mask` | bool[26] | Explicit human-input mask, independent of ownership |
| `policy_action` | float32[26] | Unmuted policy proposal or neutral packet |
| `ownership` | uint8[26] | Applied per-dimension ownership |
| `controller_id` | nullable string | Producer/controller identity |
| `active_driver` | nullable string | `human`, `policy`, or `none` |
| `controller` | nullable string | Compatibility controller/source label |
| `policy_id` | nullable string | Policy architecture/logical identity |
| `policy_revision` | nullable string | Immutable registry revision |
| `policy_digest` | nullable string | Exact immutable checkpoint digest |
| `human_monotonic_ns` | nullable int64 | Human proposal timestamp on edge clock |
| `policy_monotonic_ns` | nullable int64 | Policy proposal availability on edge clock |
| `policy_observation_monotonic_ns` | nullable int64 | Edge source-frame timestamp sent to inference |
| `muted_policy_action` | float32[26] | Proposal after `mute_mask=true` dimensions become zero |
| `mute_mask` | bool[26] | Per-action AI mute mask |
| `mute_mask_version` | nullable string | `switch_packets.v1/mute.v1` |
| `ownership_source` | nullable string | `human`, `policy`, or `none` |
| `mode` | nullable string | `human`, `pure_ai`, or `hybrid` |
| `takeover` | bool | Full human takeover state |
| `takeover_reason` | nullable string | `stick_motion`, `trigger_press`, or `button_press`; required only during takeover |
| `takeover_release_remaining_ns` | int64 | Nonnegative grace time; zero outside takeover |
| `proposal_valid` | bool | Policy response identity/shape/finite validation |
| `proposal_fresh` | bool | Edge-clock freshness decision |
| `proposal_sequence` | nullable int64 | Nonnegative immutable proposal sequence; paired with age |
| `proposal_age_ns` | nullable int64 | Nonnegative applied time minus policy time; paired with sequence |
| `gap_state` | string | `none`, `transient_gap`, `recovered`, or `disarmed` |
| `gap_reason` | nullable string | State-bound `policy_transient_gap` or `policy_stall` |
| `gap_duration_ns` | int64 | Nonnegative; zero when `gap_state=none` |
| `boundary_sequence` | nullable int64 | Positive neutral-boundary persistence sequence |
| `boundary_acknowledged` | bool | True only after recorder claims that boundary sequence |
| `applied_action_valid` | nullable bool | Optional explicit applied-packet validity; absent means the value of `valid` |
| `bc_training_eligible` | bool | Explicit row-level training permission |

`bc_training_eligible` is the one required addition not present in edge commit
`19db93f`. The cluster treats an absent column as `false`; this is intentional
fail-closed compatibility. The producer must emit it explicitly before a new
DAgger episode can contribute BC rows.

## Valid and invalid rows

A valid row requires non-null frame/action monotonic timestamps, matching alias
columns, causal action time, exact nanosecond/second age, finite 26-D packets,
and an empty `invalid_reasons` list.

An invalid no-prior-action row may use null action timestamps and null
`action_age_ns`. It must have:

- a neutral zero `action` and `applied_action`;
- all ownership values `0`;
- at least one explicit `invalid_reasons` value;
- `bc_training_eligible=false`.

Invalid rows are retained for frame/media alignment and provenance but never
train. Transient and disarmed gaps are invalid neutral rows with state-matched
reasons; recovered gaps require a valid, fresh policy proposal. Proposal sequence
and age are nullable as a pair and, when present, age exactly matches applied time
minus policy time. Acknowledged boundaries require a positive sequence, and every
boundary row is neutral, unowned, and training-ineligible. Takeover requires hybrid
mode, full human ownership, a typed activity reason, and a nonnegative release
grace; inactive takeover carries neither reason nor grace. Unknown fields remain
forbidden.

## Snapshot and BC rules

Receipt/commit remains independent of semantic training quality. A DAgger
episode is snapshot-eligible only after its latest append-only
`nxml.episode-quality.v1` disposition explicitly sets
`training_eligible=true`. Missing or false dispositions are excluded once the
episode marker is carried in the shard API manifest.

The BC reader additionally requires `valid=true`,
`bc_training_eligible=true`, and at least one human-owned dimension. It trains
only the full `applied_action`; proposals and muted proposals are provenance,
never alternate targets. Pure-policy rows are retained but excluded from this
human-correction bootstrap path.
