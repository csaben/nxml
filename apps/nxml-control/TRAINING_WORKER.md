# BC worker contract (`nxml.bc-job.v1`)

Production `nxml-control` starts training asynchronously through a configured subprocess. Without a configured worker it starts with training unavailable and returns HTTP 503 on submission; it never silently runs fake training. The deterministic fake executor requires explicit `--allow-fake-training` and is for tests and local development only.

Configure the real worker with `NXML_BC_WORKER_COMMAND` or `--bc-worker-command`. The command is invoked as:

```text
COMMAND --request /var/lib/nxml-control/jobs/JOB_ID/request.json --result /var/lib/nxml-control/jobs/JOB_ID/result.json
```

The immutable, mode `0440` request is canonical JSON:

```json
{
  "schema_id": "nxml.bc-job.v1",
  "created_at": "RFC3339 UTC",
  "job_id": "UUID",
  "snapshot_id": "sha256:...",
  "config": {}
}
```

The worker must materialize only the named immutable snapshot, train, write its checkpoint, then atomically rename a complete result manifest into the supplied `--result` path:

```json
{
  "schema_id": "nxml.bc-result.v1",
  "job_id": "UUID matching request",
  "snapshot_id": "sha256 matching request",
  "checkpoint_path": "/persistent/path/policy.pt",
  "checkpoint_sha256": "lowercase SHA-256",
  "metrics": {"loss": 0.125},
  "logs": ["optional structured summary"]
}
```

Stdout and stderr are captured as persistent job logs. On exit zero, the controller validates schema and lineage, verifies the checkpoint exists, computes its SHA-256, compares it in constant time, and changes the result manifest to mode `0440`. A mismatch fails the job and publishes no artifact. `GET .../artifacts` exposes only a successfully verified checkpoint.

Cancellation transitions a queued job directly to `cancelled`. A running job enters the internal `cancelling` state before the controller sends `SIGTERM`; this prevents termination from racing into `failed`. Worker processes should handle `SIGTERM`, stop children, and leave incomplete temporary result files unrenamed.

Model registration can include `training_job_id`. When present, the controller requires a succeeded job and exact snapshot, checkpoint URI, and digest equality before creating the candidate revision. This is the stable boundary from verified training output to model registry.

## Raw video to latent decision

The generic executor intentionally does not pretend raw H.264 frames are already nxrl latent tensors. Recommended first-production default: freeze one explicitly versioned VAE encoder profile, materialize a content-addressed latent artifact from each immutable human-filtered snapshot, and train only against that artifact. Record the encoder model digest, preprocessing/crop/normalization, latent shape/dtype, materializer commit, source snapshot ID, and artifact digest in the job config/result lineage.

Recommended scheduling default on cradle: one job at a time, pinned to GPU 1 only after an operator drain/availability check, with no automatic preemption of VLLM or inference. Keep GPU selection in the worker/scheduler config, never in the HTTP request supplied by the browser. No training starts merely by installing or restarting the control plane.

## Pokemon ZA bootstrap worker

The repository now provides `nxml-bc-worker`, implementing `nxml.pokemon-za-bc-worker.v1`. Production command, pinned to physical GPU 0 while leaving GPU 1 outside the process:

```text
/usr/bin/env CUDA_VISIBLE_DEVICES=0 HF_HOME=/var/lib/nxml-control/hf /opt/nxml/.venv/bin/nxml-bc-worker --state-dir /var/lib/nxml-control --checkpoint-dir /var/lib/nxml-control/checkpoints
```

Set that complete string as `NXML_BC_WORKER_COMMAND`. Under CUDA visibility remapping, encoding and single-process nxrl training both use logical `cuda:0`; no second GPU is visible. The controller still does not start a job automatically.

Request config is strict. The practical default is:

```json
{
  "profile": "pokemon-za-bootstrap-v1",
  "epochs": 25,
  "sequence_length": 32,
  "batch_size": 8,
  "num_workers": 4,
  "learning_rate": 0.0001,
  "validation_fraction": 0.1,
  "vae_path": "stabilityai/sd-vae-ft-mse",
  "encode_batch_size": 24
}
```

The worker resolves only the immutable snapshot from `/var/lib/nxml-control/catalog.sqlite3`, re-verifies each cluster object and selected tar member, and never reads an edge path. It requires a human-filtered snapshot and uses its exact per-shard `episode_ids`; quarantined exclusions never reach decoding.

For every eligible row it requires `valid=true`, strictly increasing `frame_index`, 26-dimensional ownership/mask/action fields, human ownership or mask, and `action_timestamp_ns <= frame_timestamp_ns` with exact `action_age_ns`. Targets are `applied_action`, never the policy proposal. Frames are RGB bilinear-resized to 128×256, normalized to [-1,1], deterministically encoded with the mode of `stabilityai/sd-vae-ft-mse`, and scaled by 0.18215 to `(4,16,32)` float16 latents.

Each episode is split temporally: the final 10% (at least one complete sequence plus target) is validation and the prefix is training, with no overlapping window across the boundary. Fixed seed 42 controls initialization and loader shuffle. The bootstrap policy is `bc_transformer_v1`, sequence 32, hidden size 256, three layers, eight heads, dropout 0.2, and 26 outputs. The self-describing nxrl checkpoint is CPU smoke-inferred before publication. Artifact metadata exposes the action spec, latent/VAE profile, sequence length, source snapshot, and source member digests through the training artifact endpoint.
