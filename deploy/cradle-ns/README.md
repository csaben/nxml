# cradle-ns operations runbook

This directory is the operator-facing deployment layer for the NXML edge host.
It does not install services, change system configuration, pair Bluetooth, or
contain credentials. Copy the examples into place only after reviewing the
physical-interaction and privileged steps below.

## Service and data flow

```text
Nintendo Switch <-Bluetooth-> nxbt-orchestrator (127.0.0.1:7777)
       | HDMI
       v
Hagibis /dev/v4l/by-id/... -> nxml-autopilot (0.0.0.0:8080)
                                      | completed schema-v2 episodes
                                      v
                              nxml-spool -> authoritative object storage
                                      | durable commit receipt
                                      v
                                  local deletion
```

The orchestrator stays loopback-only. Only the token-protected autopilot UI is
bound to the Tailnet. `switch_packets.v1` remains the action-space identifier.

## One-time unprivileged setup

```bash
cd /home/arelius/Code/nxml
uv sync --all-packages --all-extras
install -d -m 700 ~/.config/nxml
install -d ~/.config/systemd/user
cp deploy/cradle-ns/cradle.env.example ~/.config/nxml/cradle.env
chmod 600 ~/.config/nxml/cradle.env
```

Edit the environment file. Generate `AUTOPILOT_WEB_TOKEN` locally with
`openssl rand -hex 32`; never commit it. Prefer a stable
`/dev/v4l/by-id/...` path, but the current autopilot CLI accepts a numeric
camera index, so `NXML_CAMERA_ID=0` remains the compatibility setting.

Install the user units only after configuration review:

```bash
cp deploy/cradle-ns/systemd/user/*.service ~/.config/systemd/user/
cp deploy/cradle-ns/systemd/user/*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable nxml-spool.service nxml-edge-health.timer
```

Autopilot is intentionally not enabled until the Switch/capture preflight
passes. Start it with `systemctl --user start nxml-autopilot.service`.

## Privileged/physical checklist (never automate silently)

1. Confirm the Switch dock HDMI output is connected through the Hagibis card.
2. Wake the Switch with a real controller and open **Controllers -> Change
   Grip / Order**.
3. Confirm `bluetooth.service` is active and the Linux adapter is powered.
4. First pairing: omit `--reconnect-address`; follow nxbt's pairing prompt.
5. Save the address logged by `paired Switch address(es)` in the root service
   configuration. Current known address on cradle-ns: `58:2F:40:23:3C:CA`.
6. Reconnection command (requires Clark's sudo approval/password):

   ```bash
   sudo -E env "PYTHONDONTWRITEBYTECODE=1" PATH="$PATH" \
     nxbt-orchestrator serve --host 127.0.0.1 --port 7777 \
     --reconnect-address "58:2F:40:23:3C:CA"
   ```

7. Wait for `Switch connected`, then require
   `GET http://127.0.0.1:7777/health` to report both `running` and `connected`.
8. If reconnection repeatedly fails, stop all nxbt processes cleanly and fully
   power the Switch off before retrying. Do not delete BlueZ pairing state as a
   routine recovery step.

The current account depends on login-session ACLs for video capture. For
boot-without-login operation, an administrator must explicitly grant the
service account persistent video access, normally by adding it to `video`:

```bash
sudo usermod -aG video arelius
```

A full logout/login or reboot is required. Verify with `id` and a one-frame
V4L2 read before enabling services. This is a privileged system change and is
not performed by repository tooling.

If multiple consumers need the card, loading `v4l2loopback` is also a separate
administrator-approved change. Autopilot alone does not need loopback because
its UI and inference share one capture source.

## Start sequence

1. `curl -fsS http://127.0.0.1:7777/health` -> connected.
2. Verify `/dev/video0` returns a frame and is not held by another process.
3. Verify the policy endpoint or local policy artifact before starting control.
4. `systemctl --user start nxml-autopilot.service`.
5. `python deploy/cradle-ns/nxml_edge_health.py --env ~/.config/nxml/cradle.env`.
6. Open `http://cradle-ns:8080/?token=...` on the Tailnet.
7. Keep initial mode `human-takeover`; prove human control before enabling AI.
8. Start `nxml-spool.service` only after its backend credentials and destination
   are configured.

## Emergency recovery

In order, preferring the least destructive action:

```bash
# Disable AI through the existing REST control plane.
curl -fsS -X POST -H "X-Autopilot-Token: $AUTOPILOT_WEB_TOKEN" \
  -H 'Content-Type: application/json' -d '{"enabled":false}' \
  http://127.0.0.1:8080/runtime/ai

# Stop action production while leaving Bluetooth observable.
systemctl --user stop nxml-autopilot.service

# If controller state remains unsafe, stop the root orchestrator in its own
# terminal/service. This requires explicit sudo approval.
```

Do not delete capture episodes or staging shards during recovery. A shard is
deletable only after checksum verification and an authoritative durable commit.

## Failure drills

- **Tailnet outage:** local control path remains loopback; UI becomes remote-
  unreachable. Disable AI locally if operator visibility is required.
- **Autopilot restart:** start in `human-takeover`; verify policy revision and
  capture before enabling AI.
- **Stale capture/controller:** health must become degraded. Current REST
  surfaces do not publish freshness, so production enablement is blocked until
  the edge runtime adds these fields.
- **Disk high watermark:** stop new recording, continue shipping committed
  data, and never delete uncommitted episodes. Current spooler does not yet
  implement high/low watermarks.
- **Duplicate upload/restart:** repeat the same immutable shard checksum and
  idempotency key; authoritative ingest must return the existing commit.
- **Candidate activation failure:** keep the active revision unchanged and
  report failure through REST. Roll back to the last known-good revision; do
  not hot-swap by replacing files in place.

Run the fixture-backed operator smoke test with:

```bash
python deploy/cradle-ns/nxml_edge_smoke.py
python deploy/cradle-ns/nxml_edge_smoke.py --inject autopilot-down
python deploy/cradle-ns/nxml_edge_smoke.py --inject stale-spool
```

