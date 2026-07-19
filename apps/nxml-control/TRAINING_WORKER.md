# BC worker contract (`nxml.bc-job.v1`)

Production `nxml-control` starts training asynchronously through a configured subprocess. It refuses the deterministic fake executor unless an operator explicitly passes `--allow-fake-training`; that switch is for tests and local development only.

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
