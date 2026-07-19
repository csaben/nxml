# HANDOFF — cradle-ns Switch stack (2026-07-19, Claude Code session)

Audience: Codex (or any harness) picking this up. **This document supersedes
the previous "NXML Bluetooth / NXBT Orchestrator handoff" runbook.** The
Bluetooth, systemd, input, and preview work described here was implemented,
measured on the live rig, and confirmed by Clark ("feels as fast as the
direct pulse test"). Treat it as canonical and build from it — do not revert
to the older manual-first workflow or its assumptions.

Companion docs (same session, same status):

- `MEMORY.md` (repo root) — full narrative: root causes, measurements,
  deployment traps.
- `deploy/cradle-ns/PLAY.md` — player-facing runbook (first-time + cold boot).
- `deploy/cradle-ns/README.md` — updated ops runbook (`nsctl` section).

Branch state: all of this is **uncommitted on `feature/nxml-ui-ops`**.
First task for the next session, once Clark approves: commit it (suggested
split: orchestrator+systemd / edge input+preview / probes+minui / docs).

## What is now canonical (supersedes the old handoff)

| Old handoff said | Now |
| --- | --- |
| Prefer the manual `sudo nxbt-orchestrator serve ...` terminal workflow | `sudo systemctl start nxml-bt.service` (or `just ns-up`) is the preferred owner. `nsctl manual` exists for foreground debugging and stops the service first. |
| `Type=simple`; "active" is meaningless; check ss/curl by hand | `Type=notify`: the orchestrator sd_notifies READY=1 when the API is serving. `systemctl is-active nxml-bt` == port 7777 is up. |
| Startup blocks until the Switch connects; port 7777 appears late | API comes up immediately; the Switch connects in a background thread. `GET /health` → `switch_state: connecting\|connected\|crashed` is THE readiness signal for the console. |
| Duplicate instances caused adapter contention, 80 % CPU, wedged stops | CLI refuses to start if the port is busy **before touching BlueZ**. Stop is bounded: `TimeoutStopSec=20`, `KillMode=mixed`; SIGTERM now runs graceful nxbt/BlueZ cleanup (old code SIGKILLed the process group on first TERM). |
| High CPU / loud fans blamed on duplicate processes | Root-caused deeper: upstream nxbt `wait_for_connection()` is a **no-sleep busy-spin** through the multiprocessing manager, and the orchestrator wrote full state at 120 Hz through the same manager. Fixed in `nxbt_orchestrator/controller.py`: 4 Hz connection polling + change-only writes (1 Hz keepalive). Idle CPU is now near zero. |
| Browser input: 30 Hz HTTP POST per sample, serialized on RTT | 60 Hz fire-and-forget WebSocket (`/api/human/ws` on the edge). Old POST endpoints remain as automatic fallback. Human control auto-enables on `gamepadconnected`/focus. |
| Dashboard preview: 1 fps ffmpeg-per-frame | That 1 fps preview was the "terrible latency" Clark felt (video loop, not input). New: zero-transcode 30 fps MJPEG passthrough — `/api/preview/stream.mjpeg` on the edge, and `minui`. |

## Latency: measured, closed

`deploy/cradle-ns/latency_probe.py` (modes `rtt`, `pulse`, `evdev`,
`measure`, `--dither`):

- Orchestrator `/action` ingest: **0.38 ms p50**.
- Closed loop dpad-tap → capture frame change: ~230 ms, of which ~100–150 ms
  is the Hagibis capture pipeline itself.
- `pulse` mode watched on the real TV: effectively instant → BT input path
  is fine. Report-stream dither changed nothing → BT sniff-mode theory dead.
- Conclusion: input path <10 ms above the radio. If lag is ever reported
  again while watching the TV, investigate the radio (link quality, 2.4 GHz
  Wi-Fi coex), not the software.

## The stack, as deployed right now

- `nxml-bt.service` (root, Type=notify) → nxbt-orchestrator on
  `127.0.0.1:7777`. Installed from `deploy/cradle-ns/systemd/system/`.
- The orchestrator uv tool is built from `packages/nxbt-orchestrator` on
  **`/usr/bin/python3.11`** (system interpreter — uv's standalone CPython
  lacks `socket.AF_BLUETOOTH`; 3.14 can't build dbus-python). Reinstall:
  stop the service first (executable vanishes mid-install → 203/EXEC loop),
  then `uv tool install --force --python /usr/bin/python3.11 --from
  packages/nxbt-orchestrator nxbt-orchestrator`.
- `nxml-edge.service` (user) on `127.0.0.1:8090` behind Tailscale Serve —
  full dashboard, WS input, streaming preview.
- `minui` — `deploy/cradle-ns/minui.py`, the minimal play page Clark is
  using: `uv run python deploy/cradle-ns/minui.py --host "$(tailscale ip
  -4)" --port 8091` → http://cradle-ns:8091/. **No auth, tailnet-bind only —
  never bind it wider and never put it behind Funnel.** Currently running
  as a foreground/background process, not a unit.
- Entrypoint: `deploy/cradle-ns/nsctl` (`just ns-up / ns-down / ns-status /
  ns-logs`). It audits for stray orchestrator processes before starting.

## Invariants to preserve when building on this

1. **Exactly one orchestrator instance.** It owns the BT adapter and port
   7777. The port preflight enforces this — don't weaken it.
2. **`/health.switch_state` is the only truthful Switch signal.** Not
   `bluetoothctl`, not the adapter alias, not systemd active-ness.
3. **Human input stays fire-and-forget.** Do not reintroduce per-sample
   request/response round trips anywhere in the input path. Closing the
   input socket must force neutral (the edge WS handler's `finally` is
   deliberately synchronous — an await there is skipped on cancellation).
4. **Capture is single-consumer.** A live preview stream holds the V4L2
   device; minui's stream deliberately preempts other ffmpeg holders and
   the edge dashboard falls back to 1 fps. If recording (collect/autopilot)
   must coexist with a live viewer, that's a v4l2loopback / fan-out-service
   project — don't try to share the raw device.
5. **Paused work stays paused**: PolicyKit rules, Tailscale identity
   architecture, autopilot. Same as before. Autopilot remains disabled.
6. Don't print/modify `~/.config/nxml/*.token`.

## Verification commands

```bash
just ns-status                          # processes, ports, services, health
curl http://127.0.0.1:7777/health       # switch_state is the truth
uv run pytest tests/nxml_edge tests/integration -q   # 131 passed (64 edge + 67 integration)
uv run pytest tests/nxml_capture tests/nxml_spool -q # 31 passed touched-suite gate
uv run python deploy/cradle-ns/latency_probe.py rtt  # ~0.4 ms if healthy
```

`tests/nxwm_mira` fails collection (`No module named 'nxwm_mira.data'`) —
**pre-existing**, unrelated, not introduced by this work.

## Open threads (in rough priority order)

1. Commit `feature/nxml-ui-ops` once Clark approves.
2. Decide minui's future: keep as throwaway, or fold its full-width
   low-latency layout into the edge dashboard's human-capture mode (which
   already has the same WS input + streaming preview server-side).
3. Optional: a user systemd unit for minui if it becomes a daily driver.
4. Capture→spool→cluster ingest validation (the original project focus)
   is still the next milestone — the stack under it is now solid.
5. Longer-term: v4l2loopback or a capture fan-out service if play-by-preview
   and episode recording need to run simultaneously.
