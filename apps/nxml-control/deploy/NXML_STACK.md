# NXML DAgger stack entry

`nxml-stack` is an operational wrapper for the existing root-managed
`nxml-control.service` and `nxml-inference.service`. It does not install code,
change the active model revision, submit training, promote a model, or arm edge AI.

```bash
~/deploy/nxml/nxml-stack status
~/deploy/nxml/nxml-stack dry-run
sudo ~/deploy/nxml/nxml-stack start
sudo ~/deploy/nxml/nxml-stack verify
sudo ~/deploy/nxml/nxml-stack restart
sudo ~/deploy/nxml/nxml-stack stop
```

`start` reports every existing GPU 0 compute process but never stops one. It admits
the immutable inference endpoint only when `nvidia-smi` reports at least 4096 MiB
free on physical GPU 0. Override that conservative model-load-plus-margin floor with
the positive integer `NXML_INFERENCE_MIN_FREE_MIB`; missing, malformed, or insufficient
telemetry fails closed. GPU 1 is never selected or modified.

Successful verification prints the Tailnet control and inference endpoints, storage
admission, BC worker capability, and immutable revision/checkpoint identity. It never
prints the bearer token or submits a frame/action request.

After starting the units, the wrapper polls control health and digest-bound ZMQ INFO
for up to 90 seconds by default, reporting warm-up progress. Configure the positive
integer attempt and interval values with `NXML_INFERENCE_READINESS_ATTEMPTS` and
`NXML_INFERENCE_READINESS_INTERVAL_SECONDS`. A readiness timeout leaves already-active
services running for diagnosis rather than misleadingly rolling them back.

Inference admission does not imply training capacity. Status and verification report
only that the real BC worker is configured; they do not claim, reserve, or occupy a
GPU for training.
