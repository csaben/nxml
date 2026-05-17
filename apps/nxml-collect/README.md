# nxml-collect

Passive episode recorder. Runs alongside `nxbt-orchestrator`, opens a v4l2
capture device, subscribes to the orchestrator's `/ws/state` WebSocket,
time-aligns each frame with the most recent controller snapshot, and writes
the buffered episode to disk on Ctrl-C.

"Passive" = doesn't drive the Switch itself. The human plays through a real
controller wired to the orchestrator (or through the optional `--ui`
browser teleop); this tool just observes and pairs.

## Install

From a workspace checkout, just `uv sync` and use `uv run nxml-collect …`.

Standalone (no clone), as a `uv tool`:

```bash
uv tool install \
    "git+https://github.com/csaben/nxml.git#subdirectory=apps/nxml-collect"

# Pin to a commit or tag for reproducibility:
#   "git+https://github.com/csaben/nxml.git@<sha>#subdirectory=apps/nxml-collect"
```

Once installed, drop the `uv run` prefix from the invocations below.

## Quick start

Start the orchestrator first (see [`packages/nxbt-orchestrator/README.md`](../../packages/nxbt-orchestrator/README.md)
for the install + run command), then:

```bash
nxml-collect --game pokemon-za --output ./data/$(date +%Y%m%d)/
```

One process invocation = one episode. Ctrl-C flushes and closes cleanly.

## Usage

### Override defaults

```bash
nxml-collect \
    --game pokemon-za \
    --camera 1 \
    --orchestrator ws://localhost:7777/ws/state \
    --output ./data/$(date +%Y%m%d)/ \
    --max-frames 18000
```

### Browser teleop (`--ui`)

Adds a tiny FastAPI server that serves the live capture as MJPEG and
forwards browser-Gamepad input to the orchestrator. Useful when the host
running `nxml-collect` doesn't have your controller plugged in directly:

```bash
nxml-collect --game pokemon-za --output ./data/$(date +%Y%m%d)/ \
    --ui --ui-port 8080
```

Open `http://<host>:8080/` in a browser, plug a USB/BT pad into that
machine, and press any button to wake the Gamepad API. The page POSTs the
26-dim action to `/action`, which proxies to the orchestrator (no CORS to
configure). The page also streams the live capture via `/mjpeg`.

### Writer / codec choice

| `--writer`               | Files per episode                                                  | When to pick                                                       |
| ------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `video_parquet` (default) | `{name}.mkv` (ffv1) + `{name}.parquet` + `{name}.manifest.json`     | Canonical; watchable in VLC/mpv; ~50–100× smaller than raw frames. |
| `npz`                    | `{name}.npz` + `{name}.manifest.json`                              | Debug only; whole episode buffered in RAM, no PyAV dependency.     |

`--codec` only applies to `video_parquet`:
- `ffv1` (default) — lossless RGB, mkv container, bit-exact round-trip.
- `h264` — CRF 18, mp4 container, ~3–5× smaller than ffv1, near-visually-lossless.

## On-disk format

### `video_parquet` (default)

```
{name}.mkv             # ffv1 lossless (or .mp4 for h264)
{name}.parquet         # frame_idx (i64), timestamp (f64), action (fixed_size_list<f32, 26>)
{name}.manifest.json   # schema_version, format, action_spec, action_dim,
                       # frame_count, fps_estimate, fps_nominal, video{...}, created_at_utc
```

### `npz` (debug fallback)

```
{name}.npz             # frames     (T, H, W, 3) uint8, BGR (cv2-native)
                       # actions    (T, 26)      float32
                       # timestamps (T,)         float64
{name}.manifest.json   # schema_version, action_spec, action_dim,
                       # frame_count, frame_shape, fps_estimate, created_at_utc
```

Both share `action_spec = "switch_packets.v1"` — the 26-dim contract from
`nx-packets`.

## CLI

```
nxml-collect --game GAME --output DIR
             [--camera N]                    # v4l2 device index (default: 0)
             [--orchestrator WS_URL]         # default: ws://127.0.0.1:7777/ws/state
             [--max-frames N]                # safety cap; default: run until Ctrl-C
             [--max-action-age SECS]         # drop frames whose latest snapshot is older (default: 0.5)
             [--initial-timeout SECS]        # wait for first snapshot before bailing (default: 10)
             [--ui]                          # serve browser teleop UI (MJPEG + gamepad)
             [--ui-host HOST] [--ui-port N]  # ui bind (default: 0.0.0.0:8080)
             [--writer video_parquet|npz]    # default: video_parquet
             [--codec ffv1|h264]             # only for video_parquet; default: ffv1
             [--fps N]                       # nominal capture fps; default: 30.0
```
