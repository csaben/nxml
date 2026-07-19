# cradle-ns operations runbook

This directory is the operator-facing deployment layer for the NXML edge host.
It does not install services, change system configuration, pair Bluetooth, or
contain credentials. Copy the examples into place only after reviewing the
physical-interaction and privileged steps below.

`apps/nxml-edge` is the always-on supervisor. In production its backend listens
only on loopback port 8090 behind Tailscale Serve, discovers capture through the
configured `/dev/v4l/by-id` identity, and consumes autopilot's authenticated
U4 health/eject/re-arm contract. It never accepts a command name or unit name
from the browser: only the predefined `nxml-bt.service` and
`nxml-autopilot.service` operations exist.

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

The orchestrator and edge backend stay loopback-only. Tailscale Serve is the
sole edge UI entry point. `switch_packets.v1` remains the action-space identifier.

## Tailnet-only access

Clark is the sole user of these Tailscale peers. User-facing services on this
host therefore do not require application tokens, browser sign-ins, or a
Tailscale identity gate unless Clark explicitly requests one. Keep the edge
backend loopback-only behind Tailscale Serve, and bind minui only to the
machine's Tailnet IP. Never enable Funnel or bind either UI to a public or
wider interface. This preference does not remove private machine-to-machine
credentials such as the ml-stream cluster token.

```bash
tailscale serve --bg --yes http://127.0.0.1:8090
tailscale serve status
```

Open `https://<edge-node>.<tailnet>.ts.net/`. Serve removes spoofed identity
headers before proxying to the loopback backend. Funnel must remain disabled.

For the minimal low-latency play UI, bind directly to the Tailnet address:

```bash
uv run python deploy/cradle-ns/minui.py \
  --host "$(tailscale ip -4)" --port 8091
```

Open `http://cradle-ns:8091/` from a Tailnet peer.

## One-time unprivileged setup

```bash
cd /home/arelius/Code/nxml
uv sync --all-packages --all-extras
install -d -m 700 ~/.config/nxml
install -d ~/.config/systemd/user
cp deploy/cradle-ns/cradle.env.example ~/.config/nxml/cradle.env
cp deploy/cradle-ns/edge.json.example ~/.config/nxml/edge.json
chmod 600 ~/.config/nxml/cradle.env
chmod 600 ~/.config/nxml/edge.json
```

Edit the environment file. Do not put the web token in it: the supervisor
provisions `~/.config/nxml/autopilot.token` with mode `0600`, and the static
autopilot launcher reads that file so the two services cannot drift. Prefer a stable
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

The root Bluetooth unit and optional PolicyKit rule are review artifacts, not
automatic installers. Enabling edge control of Bluetooth requires Clark to
approve copying them to `/etc/systemd/system/nxml-bt.service` and
`/etc/polkit-1/rules.d/49-nxml-edge.rules`, followed by daemon reload. The
PolicyKit rule grants only start/stop/restart of `nxml-bt.service` to
`arelius`; it does not grant arbitrary systemd or shell access.

Boot persistence without a graphical login additionally requires the one-time
administrator-approved command `sudo loginctl enable-linger arelius`.

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

## Single entrypoint: nsctl

`deploy/cradle-ns/nsctl` (also `just ns-up` / `ns-down` / `ns-status` /
`ns-logs`) is the day-to-day entrypoint for the human-capture stack. It audits
for stray orchestrator processes before starting anything, starts
`nxml-bt.service` plus the `nxml-edge` user service, and reports the
authoritative readiness signals. `nsctl manual` stops the service and runs the
orchestrator in the foreground for debugging.

The orchestrator now starts its API immediately and connects to the Switch in
the background, and sd_notifies systemd when the API is serving:

- `systemctl is-active nxml-bt.service` == the API on 127.0.0.1:7777 is up
  (`Type=notify`, no more false "active" during startup).
- `curl http://127.0.0.1:7777/health` reports `switch_state`
  (`connecting` / `connected` / `crashed`) — this is the Switch truth.
- A second orchestrator instance refuses to start (port preflight) before it
  touches BlueZ, so duplicate-instance adapter contention cannot recur.
- `systemctl stop` completes within `TimeoutStopSec=20` even if an nxbt
  worker wedges (`KillMode=mixed` escalation), instead of 90 s in
  final-sigterm.

After changing `packages/nxbt-orchestrator`, reinstall the root tool so the
service picks it up:

```bash
# The SYSTEM interpreter is required: uv's standalone CPython lacks
# socket.AF_BLUETOOTH (controller crashes at reconnect), and 3.14 can't
# build dbus-python. Stop the service first — install briefly removes the
# executable and a restart-looping unit would hit 203/EXEC.
sudo systemctl stop nxml-bt.service
uv tool install --force --python /usr/bin/python3.11 \
  --from packages/nxbt-orchestrator nxbt-orchestrator
sudo systemctl start nxml-bt.service
```

Unit file changes additionally need
`sudo cp deploy/cradle-ns/systemd/system/nxml-bt.service /etc/systemd/system/`
and `sudo systemctl daemon-reload`.

## Start sequence

1. Run `just ns-up` to audit duplicate processes and start the orchestrator
   plus edge dashboard.
2. Require `curl -fsS http://127.0.0.1:7777/health` to report
   `switch_state: "connected"`.
3. Verify the stable Hagibis by-id device exists and is not held by a recorder.
4. Start minui on the Tailnet address and open `http://cradle-ns:8091/`.
5. Connect a standard-mapped gamepad and press a button; minui should report
   `input: streaming 60 Hz`.
6. Keep autopilot disabled. For a human-only episode, close live preview users
   so the single-consumer capture device is released, then start
   `nxml-collect --driver human` explicitly.
7. Keep `nxml-spool.service` running only after its private backend credentials
   and destination are configured; local deletion remains receipt-gated.

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
