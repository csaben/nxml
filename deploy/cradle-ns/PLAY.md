# Playing the Switch through the browser (minui)

The fast path: your gamepad in a browser tab → 60 Hz WebSocket → the nxbt
orchestrator on cradle-ns → Bluetooth → Switch, with a ~30 fps zero-transcode
capture preview coming back. Input and video are both fast enough to play on.

Two scenarios below. Everything runs on **cradle-ns**; you play from any
tailnet device with a standard-mapping gamepad (Xbox pad, most USB pads).

---

## 1. First-time setup (never set this machine up before)

Prerequisites: Switch dock HDMI → Hagibis capture card → cradle-ns USB;
a working Bluetooth adapter; the machine joined to the tailnet; repo at
`/home/arelius/Code/nxml`.

```bash
cd /home/arelius/Code/nxml

# 1. Workspace deps (edge UI, capture, minui)
uv sync --all-packages --all-extras

# 2. The Bluetooth orchestrator tool — MUST use the system interpreter
#    (uv's standalone CPython lacks socket.AF_BLUETOOTH; 3.14 can't build
#    dbus-python)
uv tool install --force --python /usr/bin/python3.11 \
  --from packages/nxbt-orchestrator nxbt-orchestrator

# 3. Install + enable the root Bluetooth service
sudo cp deploy/cradle-ns/systemd/system/nxml-bt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nxml-bt.service
```

First pairing (only needed once per Switch console):

1. If pairing a **new** Switch (not the known one), edit the unit's
   `--reconnect-address` out first: the service ships preset to
   `58:2F:40:23:3C:CA` (the current console).
2. On the Switch: **Controllers → Change Grip/Order**, leave that screen open.
3. `sudo systemctl start nxml-bt.service`, then watch
   `sudo journalctl -u nxml-bt.service -f` until you see
   `Switch connected` and `paired Switch address(es): ...`.
4. Put that address into the unit's `--reconnect-address`, then
   `sudo systemctl daemon-reload && sudo systemctl restart nxml-bt.service`.
   Future runs reconnect without the Change Grip screen.

Then continue with scenario 2 below from step 3.

---

## 2. Turning the computer back on (already set up)

```bash
cd /home/arelius/Code/nxml

# 1. Start the Bluetooth orchestrator (+ edge dashboard) and audit health
just ns-up          # == deploy/cradle-ns/nsctl up

# 2. Wake the Switch (tap it, or a paired physical controller). Watch until
#    switch_state says "connected":
curl http://127.0.0.1:7777/health
#    If it sits in "connecting" for >30 s, open Controllers → Change
#    Grip/Order on the Switch once.

# 3. Start the play page, bound to the tailnet only (no auth on this page)
uv run python deploy/cradle-ns/minui.py \
  --host "$(tailscale ip -4)" --port 8091
#    (keep it in a tmux/second terminal, it runs in the foreground)
```

Now from any tailnet device:

1. Open **http://cradle-ns:8091/** (or `http://<tailscale-ip>:8091/`).
2. Plug in / connect your gamepad and **press any button** — browsers only
   expose gamepads after the first press. The status bar goes green:
   `input: streaming 60 Hz`.
3. Play. Input auto-pauses (and the controller goes neutral) when the tab
   loses focus, and auto-reconnects when you come back.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `nsctl up` says orchestrator API didn't come up | `just ns-logs`; check for "port already in use" → `pgrep -af nxbt-orchestrator` and stop the stray instance |
| `/health` shows `switch_state: "crashed"` | `just ns-logs` for the traceback; `sudo systemctl restart nxml-bt.service`. If it's `AF_BLUETOOTH` — the tool was rebuilt on the wrong Python, redo setup step 2 |
| Preview is black | Check the capture card: `ls -l /dev/v4l/by-id/` should show the Hagibis; Switch must be docked and awake |
| Preview stutters / stops | Another viewer took the device (one stream at a time — minui preempts, the edge dashboard falls back to 1 fps) |
| Gamepad never detected | Must be standard mapping (Xbox pads are); press a button first; try Chrome/Firefox |
| Inputs feel laggy | Play watching this page or the TV — not the edge dashboard's 1 fps preview. See `MEMORY.md` for the latency investigation |

The full dashboard (session/cluster controls, telemetry) stays at the usual
Tailscale Serve URL and is unaffected by minui.
