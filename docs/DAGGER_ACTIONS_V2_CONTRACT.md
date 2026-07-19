# `nxml.dagger-actions.v2` edge contract

The Parquet schema is identified by `action_rows_schema_id =
"nxml.dagger-actions.v2"`. All timestamps below are Linux monotonic nanoseconds.
Parquet fields are nullable unless an invariant explicitly requires a value.

## Arbitration extension fields

| Field | Arrow type | Semantics |
|---|---|---|
| `takeover_reason` | `string` | Why Hybrid entered full-packet human ownership. Null unless `takeover=true`. Current producer values are `stick_motion`, `button_press`, and `trigger_press`. |
| `takeover_release_remaining_ns` | `int64` | Remaining quiet-time grace before Hybrid may leave human ownership. Non-negative. It is zero outside takeover, may reset upward on new activity, and reaches zero before the neutral return boundary. |
| `proposal_sequence` | `int64` | Sequence of the exact policy proposal observed by this arbitration record. Null when no policy proposal was observed. Within one loaded inference identity it is non-decreasing across rows; repeats are normal because 30 Hz frames sample a 60 Hz action history, and gaps are allowed. Do not treat it as a row index. |
| `proposal_age_ns` | `int64` | `action_timestamp_ns - policy_monotonic_ns` for the observed proposal. Null with no proposal and non-negative when present. This is proposal age at arbitration, not `frame_timestamp_ns - policy_monotonic_ns`. It may exceed the application hold horizon on neutral gap rows. |
| `gap_state` | `string` | Policy-stream state for the applied arbitration record: `none`, `transient_gap`, `recovered`, or `disarmed`. |
| `gap_reason` | `string` | Null for `gap_state=none`; currently `policy_transient_gap` for transient/recovered gaps and `policy_stall` for hard disarm. Consumers must preserve unknown future reason strings. |
| `gap_duration_ns` | `int64` | Non-negative duration measured from the first missed hold horizon. Zero for `none`; increasing on `transient_gap`; final measured duration on `recovered` or `disarmed`. |
| `boundary_sequence` | `int64` | Episode/process-local, strictly increasing identifier for a recorder-observable takeover-release neutral boundary. Null on ordinary rows. The same value may appear on several held-neutral rows until acknowledgment. |
| `boundary_acknowledged` | `bool` | True on exactly one causally sampled neutral/unowned row for a given `boundary_sequence`. False on all ordinary rows and on other held rows with that sequence. Policy may resume only on a later action tick/frame after this acknowledgment. |

## Cross-field invariants

- `takeover=true` implies `mode=hybrid`, full ownership is human (`ownership` is
  26 ones), and `takeover_reason` is non-null.
- A takeover-release boundary has a finite neutral `applied_action`, 26 zero
  ownership values, `ownership_source=none`, a non-null `boundary_sequence`,
  and `boundary_acknowledged=true` on exactly one row per sequence.
- A policy transient-gap row is neutral/unowned, `valid=false`,
  `proposal_fresh=false`, and includes its sparse `gap_reason` in
  `invalid_reasons`. A recovered row may be valid and policy-owned.
- `proposal_sequence` and `proposal_age_ns` describe the observed policy
  proposal independently of which source owns `applied_action`; therefore they
  may be present during Human takeover or armed Human observation.
- With no observed proposal, `proposal_sequence`, `proposal_age_ns`,
  `policy_revision`, `policy_digest`, and policy timestamps are null. The
  vector compatibility columns remain finite neutral 26-D vectors.
- Every valid row remains causal:
  `action_timestamp_ns <= frame_timestamp_ns` and
  `action_age_ns = frame_timestamp_ns - action_timestamp_ns`.

## Recovered Human rolling episodes

The two receipt-recovered episodes were created with `mode=human` and
`armed=false` in every segment manifest:

- `4d5e3d4b-139c-44bd-b510-12f388035fb7`: 5 contiguous segments,
  `[24199709736535, 24243071675632)`.
- `9b940398-9db9-4fdb-adfd-c454b8ebbc7e`: 9 contiguous segments,
  `[24309064092917, 24406151227156)`.

For unarmed Human input the action plane records no policy proposal. The
producer-enforced pattern for these fields is consequently:

```text
takeover_reason=null
takeover_release_remaining_ns=0
proposal_sequence=null
proposal_age_ns=null
gap_state="none"
gap_reason=null
gap_duration_ns=0
boundary_sequence=null
boundary_acknowledged=false
```

The local payloads were deleted after authoritative receipt fsync, so row-level
frequency claims beyond this producer-enforced Human profile require a cluster
export/read route. Receipt and close metadata alone cannot prove Parquet value
distributions.

## Parser compatibility notes

The complete v2 table currently contains 44 columns. In addition to the nine
fields above, validators must accept `frame_timestamp_ns`,
`action_timestamp_ns`, `action_age_ns`, `applied_action`, `human_action`,
`human_mask`, `policy_action`, `ownership`, `policy_revision`, `policy_digest`,
`human_monotonic_ns`, `policy_monotonic_ns`,
`policy_observation_monotonic_ns`, `muted_policy_action`, `mute_mask`,
`mute_mask_version`, `ownership_source`, `mode`, `takeover`, `proposal_valid`,
`proposal_fresh`, and `bc_training_eligible`. The older `action`,
`action_monotonic_ns`, `action_age`, and `controller` columns remain aliases for
backward-compatible readers; they are not replacements for the causal v2
fields.

## Action-schema marker layers and current rolling drift

These identifiers are not interchangeable:

- An episode writer manifest has `schema_id = "nxml.episode.v2"`. Its canonical
  action-table marker is `action_schema_id = "nxml.dagger-actions.v2"`; the
  writer also emits `action_rows_schema_id = "nxml.dagger-actions.v2"` as the
  compatibility/API alias.
- A rolling bundle has `schema_id = "nxml.segment-bundle.v1"`, and a rolling
  close has `schema_id = "nxml.episode-close.v1"`. Those `schema_id` values
  identify their JSON envelopes, not the action table.
- `action_rows_schema_id` is copied into legacy spool/shard `episodes[]`
  entries. It is not a physical Parquet row field.
- `DaggerActionRecordV2.row_schema_id` is the optional logical per-row parser
  field. The deployed edge Parquet writer does **not** emit `row_schema_id`,
  `action_schema_id`, or `action_rows_schema_id` as a column.

Consequently, the recovered rolling Parquet files have the exact 44 columns
listed by the producer schema, with no physical action-schema marker column.
For each recovered row:

```text
DaggerActionRecordV2.model_validate(raw).row_schema_id == None
```

This explains the validator sequence at ml-stream `f0bc93d`:

1. `DaggerActionRecordV2.model_validate(raw)` succeeds because
   `row_schema_id` defaults to null.
2. The following helper comparison against `"nxml.dagger-actions.v2"` raises
   `wrong action schema`.

The episode manifest that carried both action markers was local-only for the
rolling path. A `nxml.segment-bundle.v1` tar contains exactly three members
(`video`, `actions`, `events`), and the strict live bundle/close envelopes have
no `action_schema_id` or `action_rows_schema_id` property. Therefore the marker
is absent from both the physical rows and the authoritative rolling envelope;
it cannot be recovered by renaming `schema_id`.

Existing immutable recovered segments must not be rewritten. A validator can
support them by assigning the logical row schema in memory only after matching
the exact known 44-column producer contract and trusted rolling producer
version. For future data, the contract should choose one explicit route:

1. add physical `row_schema_id = "nxml.dagger-actions.v2"` to every Parquet
   row; or
2. version `SegmentBundleV1` to carry an authoritative
   `action_rows_schema_id`, and validate rows under that trusted envelope.

The current helper must not require a field that neither accepted rolling
contract transmits. Adding a trusted envelope marker is preferable to silently
inferring the action schema from `schema_id`, which names a different object.
