# MEMORY — Switch stack overhaul (2026-07-19)

Working notes from the bt/capture/io cleanup session (Claude Code). Branch:
`feature/nxml-ui-ops`, uncommitted at time of writing.

## Clark's standing Tailnet authentication preference

Clark is the sole user of his Tailscale peers. Tailnet-only user-facing
services should not require application tokens, browser sign-ins, Tailscale
identity gates, or similar authentication unless Clark explicitly asks for
it. Bind them only to the Tailnet (never Funnel/public/wider interfaces) and
prioritize momentum. This does not automatically remove existing private
machine-to-machine credentials such as the ml-stream cluster token.

For E2E development, strict filesystem ownership of that machine token must
not block operator tooling: it is acceptable for the deployed nxml-control
token to be readable by the operator account and development processes. The
token must still stay server-side and must never be echoed into user-facing
browser responses or logs. This does not change the Tailnet UI no-auth
preference above.

## What was wrong, and what changed

### 1. CPU burn / loud fans — root-caused

- Upstream nxbt 0.1.4 `wait_for_connection()` is a **busy-spin with no
  sleep** (`while not connected: pass`), each iteration doing two pickled
  round trips through the multiprocessing manager. Any time the Switch was
  disconnected, one core sat at 100 %. The orchestrator no longer calls it:
  `controller.py` polls nxbt's shared state at 4 Hz in a background thread.
- The orchestrator's 120 Hz update loop pushed the full controller dict
  through the manager proxy every tick even when idle. It now writes **only
  on state change** (1 Hz keepalive refresh); nxbt re-applies the last
  direct-input packet itself at 132 Hz, so this is behavior-neutral.

### 2. UI gamepad latency (browser path) — fixed

- Old path: one HTTPS POST per input sample through Tailscale Serve,
  serialized on the response (`humanSending` guard) → effective rate =
  1/RTT, every sample stale by the round trip. The edge also made an extra
  blocking `GET /state` to the orchestrator per non-neutral input.
- New path: browser streams 60 Hz over WebSocket `/api/human/ws`,
  fire-and-forget with a `bufferedAmount` backpressure guard. Closing the
  socket forces neutral (server-side `finally` is deliberately synchronous —
  an `await` there gets skipped on task cancellation). Edge rate cap raised
  45 → 125 Hz; per-input readback removed. Old HTTP endpoints remain as
  automatic fallback. Human control **auto-enables** on `gamepadconnected`
  and on window focus with a pad present.
- Verified: orchestrator `/action` ingest is 0.38 ms p50 (probe `rtt` mode);
  inari ↔ cradle-ns is a direct LAN tailscale link, not DERP-relayed.

### 3. systemd — now trustworthy

- Orchestrator starts its API immediately; the Switch connects in the
  background. `/health` reports `switch_state: connecting|connected|crashed`
  and is the authoritative readiness signal (edge dashboard already keys on
  it).
- `nxml-bt.service` is `Type=notify` (orchestrator sd_notifies READY=1), so
  `active` == API serving. `TimeoutStopSec=20` + `KillMode=mixed` bounds the
  old 90 s final-sigterm hangs. The CLI SIGTERM handler no longer SIGKILLs
  the process group on the first signal, so nxbt's BlueZ cleanup actually
  runs.
- The CLI refuses to start if port 7777 is taken **before touching BlueZ**
  — the duplicate-instance adapter-contention failure mode is gone.

### 4. Entrypoint

- `deploy/cradle-ns/nsctl` (`just ns-up / ns-down / ns-status / ns-logs`):
  audits for stray orchestrator processes, starts bt + edge, waits for
  health, reports Switch state. `nsctl manual` for the foreground workflow.

## Deployment traps (learned the hard way)

- The orchestrator uv tool MUST be built on the **system** interpreter:
  `uv tool install --force --python /usr/bin/python3.11 --from
  packages/nxbt-orchestrator nxbt-orchestrator`. uv's standalone CPython
  lacks `socket.AF_BLUETOOTH` → nxbt crashes at reconnect with
  `AttributeError`. Python 3.14 can't build dbus-python at all.
- Stop `nxml-bt.service` before reinstalling the tool: the executable
  vanishes mid-install and a restart-looping unit spams 203/EXEC.
- After edge changes: `systemctl --user restart nxml-edge.service` AND
  hard-refresh the browser tab (the HTML/JS is served by the backend).

## Remaining issue: input latency to the console (NOT the UI)

`deploy/cradle-ns/latency_probe.py` bisects the chain (`rtt`, `pulse`,
`evdev`, `measure` modes). Findings so far:

- `measure` (closed loop: POST dpad tap → nxbt → BT → Switch → HDMI →
  Hagibis → frame diff): **median 230 ms, min 198, max 233** over 6 trials.
  This is with zero browser/network/UI involvement, so the web UI is
  **ruled out** as the dominant source.
- `--dither` (keeps the HID report stream continuously changing to prevent
  BT sniff-mode idling): **no improvement** (216 ms median) — rules out
  nxbt's report-caching/idle-link theory.
- The number includes the Hagibis MJPEG capture pipeline, which on these
  MacroSilicon dongles is plausibly 100–150 ms by itself. True BT input
  latency is therefore bounded at ≲200 ms but not yet separated.

**Resolved 2026-07-19:** the `pulse` TV test confirmed the input path is
effectively instant — the felt lag was the *video feedback loop*: gameplay
was judged through the dashboard preview, which was **1 fps by design**
(`CapturePreview` re-spawns ffmpeg per frame). Fix shipped:

- `v4l2_mjpeg_stream_command()` in `nxml_capture.backends.ffmpeg_v4l2`:
  persistent zero-transcode remux (Hagibis emits MJPEG natively → `-c copy`
  → `mpjpeg`), ~30 fps at 1-2 frames of latency, near-zero CPU.
- `CapturePreview.stream()` + `GET /api/preview/stream.mjpeg` (single
  client, 409 when busy; `status()` reports the live stream instead of
  probing the held device). Dashboard `<img>` uses the stream and falls
  back to the 1 fps endpoint on error.
- Caveat: the stream holds the V4L2 device while a tab is watching. Episode
  recording (nxml-collect/autopilot capture) can't share the device with a
  live stream viewer — mediate with v4l2loopback or a fan-out service if
  both are ever needed at once.

If input lag is ever felt again while watching the real TV: suspect the BT
adapter/driver (`hciconfig`, link quality, 2.4 GHz Wi-Fi coexistence), not
the software stack — everything above the radio measures <10 ms.

## minui — the minimal play page (validated, feels right)

`deploy/cradle-ns/minui.py`: a standalone one-page play UI (gamepad → 60 Hz
WebSocket → orchestrator, plus the zero-transcode 30 fps preview). No auth —
bind it to the tailscale IP only:

```bash
uv run python deploy/cradle-ns/minui.py --host "$(tailscale ip -4)" --port 8091
```

Open http://cradle-ns:8091/ from a tailnet device and press any gamepad
button. Clark confirmed this feels as fast as the direct pulse test
("perfection"). Its preview endpoint preempts any other ffmpeg holding the
capture device (edge dashboard falls back to 1 fps). Player-facing runbook:
`deploy/cradle-ns/PLAY.md`.
