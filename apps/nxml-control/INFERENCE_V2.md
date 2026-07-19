# Immutable revision inference data plane v2

`nxml-inference` is a separate ZMQ `REP` service bound only to cradle's
Tailnet address at `tcp://100.80.98.4:5557`. It is outside the browser and the
60 Hz input/preview path. The edge server may poll it at about 30 fps; timeout,
warming, or error always means a neutral proposal and never changes the
locally applied human action.

At startup the service resolves `NXML_INFERENCE_REVISION_ID` through the local
model registry. It accepts only a `validated`, `active`, or `retired` immutable
revision, confines its artifact to the managed checkpoint directory, verifies
SHA-256, checks `bc_transformer_v1` compatibility, and smoke-infers before
binding the socket. Edge requests never contain filesystem paths.

## Wire protocol

Every request is one opcode byte followed by a payload. Messages larger than
2 MiB are rejected. The production socket has receive/send high-water marks of
one and a 250 ms processing budget.

| Opcode | Request | Response |
| --- | --- | --- |
| `0x00` | legacy latent tensor | legacy 26 little-endian float32 values |
| `0x01` | legacy JPEG | legacy empty marker while warming, otherwise 26 float32 values |
| `0x03` | legacy path reload | rejected with a neutral v2 error |
| `0x04` | empty INFO | `nxml.policy-inference-info.v2` JSON |
| `0x11` | uint64 little-endian edge monotonic timestamp + JPEG | `nxml.policy-proposal.v2` JSON |
| `0x12` | empty RESET | info JSON with `reset=true` |
| `0x13` | JSON registry reload | info JSON after atomic switch |
| `0x14` | empty HEALTH/readiness | info JSON |
| `0x15` | JSON rollback | info JSON after atomic switch |

Reload JSON is
`{"revision_id":"...","expected_revision_id":"..."}`. Rollback JSON is
`{"expected_revision_id":"..."}`. The expected ID is mandatory CAS state;
stale requests do not change the loaded revision. Candidate loading, digest
verification, compatibility checks, and smoke inference happen before the
atomic switch. Rollback uses the previous successfully loaded revision.

V2 proposal responses include `state` (`warming`, `proposal`, or `error`), a
finite 26-D `action`, `frame_timestamp_ns`, cluster
`proposal_timestamp_ns`, `processing_latency_ns`, immutable `revision_id`,
`checkpoint_sha256`, and `action_spec_id=switch_packets.v1`. Warming and error
responses contain 26 zeros. Edge frame timestamps must increase strictly per
reset/reload epoch; duplicate or out-of-order frames fail closed.

ZMQ provides no application bearer authentication here. Exposure is limited
to the Tailnet bind/firewall and the edge server; the browser must never access
this socket. The authenticated REST control plane and inference data plane are
independent. Promotion/arming is not implied by starting this service.

## GPU ownership

The production unit exposes only physical GPU 0 (`CUDA_VISIBLE_DEVICES=0`) and
uses logical `cuda:0`. GPU 1 is never visible to it. Before starting inference,
drain the managed GPU-0 workload according to `/home/arelius/deploy`, record
whether it was running, and restore it automatically if inference readiness
fails. Do not run BC training concurrently on GPU 0.
