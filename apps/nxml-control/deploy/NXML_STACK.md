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

`start` fails closed if GPU 0 contains any process other than an already-running
NXML inference process. To stop the specifically managed `docker/vllm-qwen`
container, pass `--stop-managed-vllm`; no other process is ever stopped. The wrapper
records that action and restores the container if startup fails or when `stop` is
run. GPU 1 is never selected or modified.

Successful verification prints the Tailnet control and inference endpoints, storage
admission, BC worker capability, and immutable revision/checkpoint identity. It never
prints the bearer token or submits a frame/action request.
