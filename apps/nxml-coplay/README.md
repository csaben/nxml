# nxml-coplay

Hybrid human/AI play. A human drives a PC controller (xbox / dualshock /
etc.) via evdev while a trained `nxrl` policy proposes actions on its own
thread; `nxml-mux`'s `HumanPriority` strategy merges the streams — human
wins per-index, AI fills everything else — and the merged 26-dim action is
POSTed to `nxbt-orchestrator`'s `/action` endpoint.

Optionally records the session as a video+parquet episode (same format as
`nxml-collect`) so coplay sessions can feed offline training.

## Install

From a workspace checkout, just `uv sync` and use `uv run nxml-coplay …`.

Standalone (no clone), as a `uv tool`:

```bash
uv tool install \
    "git+https://github.com/csaben/nxml.git#subdirectory=apps/nxml-coplay"

# Pin to a commit or tag for reproducibility:
#   "git+https://github.com/csaben/nxml.git@<sha>#subdirectory=apps/nxml-coplay"
```

uv clones the whole repo so the workspace siblings (`nxml-mux`, `nxrl`,
`nxml-capture`, etc.) resolve from the same tree. Once installed, drop the
`uv run` prefix from the invocations below.

## Quick start (Ubuntu)

Set up a virtual camera from your capture card and start the orchestrator:

```bash
# terminal 1: virtual camera (so multiple consumers can read the capture)
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="Virtual-Loopback" exclusive_caps=0
ffmpeg -f v4l2 -i /dev/video0 -f v4l2 /dev/video10

# terminal 2: local nxbt-orchestrator
sudo -E env "PYTHONDONTWRITEBYTECODE=1" PATH=$PATH \
    nxbt-orchestrator serve --reconnect-address "AA:BB:CC:DD:EE:FF"
```

Verify the camera in OBS/VLC on `/dev/video10`, then run coplay:

```bash
nxml-coplay --game pokemon-za \
    --policy hf:csaben/za-ppo-v1.pt \
    --controller http://localhost:7777 \
    --controller-input auto \
    --camera 10
```

If auto-detection of your gamepad fails, pass `--device-path /dev/input/eventNN`.

## Loading policies

`--policy` accepts:

- `hf:<user>/<repo>` or `hf:<user>/<repo>/<path>.pt` — HuggingFace Hub
- `./path/to/checkpoint.pt` — local file
- `zmq://host:port` — remote policy server (client holds VAE; GPU on play box)
- `zmq+frames://host:port` — remote policy **and** VAE (play box ships frames only, no GPU needed)

## Usage

### Remote browser play (`--input-source=web`)

Run coplay on the box with the capture card and orchestrator, but drive it
from a controller plugged into a phone or laptop on your network:

```bash
nxml-coplay --game pokemon-za \
    --policy hf:csaben/za-ppo-v1.pt \
    --controller http://localhost:7777 \
    --camera 10 \
    --input-source web \
    --web-port 8080 \
    --web-token "$(openssl rand -hex 16)"
```

Open `http://<host>:8080/?token=…` in a browser, plug a USB/BT controller
into that machine, and press any button to wake the Gamepad API. The page
polls the pad at 60 Hz, POSTs the 26-dim action to `/action`, which feeds
a `WebGamepadReader` into the same `HumanPriority` mux — so the AI still
fills in indices you aren't touching, exactly like the local evdev path.
The page also shows the live capture via `/mjpeg`, so OBS/VLC isn't
needed when playing remotely.

Tunnel over Tailscale/WireGuard for play outside your LAN; do not expose
the port publicly without setting `--web-token` (anyone hitting it can
drive your Switch).

### Remote policy: `zmq://` (client holds VAE)

Server holds the policy, client holds the VAE. Each tick the client
encodes the frame to a latent locally and ships the sliding window. Needs
a GPU on the play box for the VAE.

```bash
# on the GPU box
uv run nxrl serve \
    --policy checkpoints/legacy/ppo_run_062_update_0002.pt \
    --port 5557

# on the play box
nxml-coplay --game pokemon-za \
    --policy zmq://gpu-box:5557 \
    --controller http://localhost:7777 \
    --camera 0 --input-source web --web-port 8081 \
    --mode human-takeover
```

### Remote policy: `zmq+frames://` (server holds VAE)

Server holds **both** the VAE and the sliding window; client just
JPEG-encodes the latest frame and ships ~5 KB per tick. No torch/VAE
loaded on the play box at all. Single-client by construction (one window
per server).

```bash
# on the GPU box
uv run nxrl serve \
    --policy checkpoints/legacy/ppo_run_062_update_0002.pt \
    --port 5557 \
    --enable-frame-mode

# on the play box (no GPU required)
nxml-coplay --game pokemon-za \
    --policy zmq+frames://gpu-box:5557 \
    --controller http://localhost:7777 \
    --camera 0 --input-source web --web-port 8081 \
    --mode human-takeover
```

Use a hostname or LAN IP your client box can resolve. Don't expose the
port on the public internet — there's no auth on the ZMQ socket; tunnel
via Tailscale/WireGuard.

### Live runtime controls (web UI)

When `--input-source=web`, the page exposes two strips of buttons that
operate on the *running* coplay process — no restart needed:

- **AI: on / off** — flips `CachedAiSource.enabled`. Off → only your
  human input drives the Switch (the inference thread keeps running, so
  flipping it back on resumes from the cached action immediately).
- **human-priority / human-takeover** — swaps the mux strategy live.
  `--mode` only sets the *initial* value; either button can be clicked
  while playing.

### Macros (record / replay / loop)

Pass `--macro-dir DIR` to override where macros live. The default is
`./data/macros/<game>/`, derived from `--game`, so the **Macros** panel
in the web UI is enabled automatically.

```bash
nxml-coplay --game pokemon-za \
    --policy hf:csaben/za-ppo-v1.pt \
    --controller http://localhost:7777 \
    --camera 0 --input-source web --web-port 8081 \
    --macro-dir ./data/macros/pokemon-za
```

The panel:

- **Record macro** — prompts for a name, then captures every merged 26-dim
  action that goes to the orchestrator until you click **Stop & save**.
- **Play / loop** — replays a saved macro through the same `/action` POST
  path. Tick the **loop** checkbox to keep cycling until **Stop**.
- Macros are JSON files under `--macro-dir` (one per macro), portable across
  hosts. The format is `nx_macros.Macro` — see `packages/nx-macros/README.md`.

Recording is web-UI only for now; in `--input-source=evdev` mode the buttons
are reserved for the mux, so there's no idle "command" channel for a
record-toggle. Run with `--input-source=web` to use macros, or use the
`nx_macros` library directly from a separate process.

### Triggers (visual-detect → run macro)

The **Triggers** panel turns coplay into a "watch for a screen state, then
synthesize an A-mash for N seconds" loop. For Pokémon ZA the two defaults
(`end_screen` and `connection_lost`) are seeded on first boot, both bound
to the in-memory `MashController` — autobattling is just "tick both
checkboxes".

```bash
nxml-coplay --game pokemon-za \
    --policy zmq+frames://gpu-box:5557 \
    --controller http://localhost:7777 \
    --camera 0 --input-source web --web-port 8081 \
    --macro-dir ./data/macros/pokemon-za \
    --trigger-dir ./data/triggers/pokemon-za
```

`--trigger-dir` defaults to `./data/triggers/<game>/`, so you usually don't
need to pass it. For Pokémon ZA, the `mash_a` macro is auto-seeded on
first boot (40s of alternating-A at 5 Hz), and seeded triggers can also
use `action_kind: "mash_a"` to drive the in-memory `MashController`
directly without any Macro file — so you can just tick the checkbox in
the **Triggers** panel and go.

The watcher polls capture at ~10 Hz, runs a mean-abs-diff template match
(64×32 grayscale — cheap), and after a match persists for `debounce_sec`
it fires either the named macro (`action_kind: "macro"`) through
`MacroPlayer` or the `MashController` (`action_kind: "mash_a"`) for
`mash_duration_sec`. While the macro/mash plays, the AI/human mux yields
the `/action` POST to it. After a fire, that trigger ignores further
matches for `cooldown_sec` (each trigger has its own cooldown).

Multiple triggers can be active at once; if two match in the same tick the
macro player's single-shot guard keeps them from clobbering each other.

Tuning notes:

- The seeded `similarity_threshold` (0.85 for `end_screen`, 0.75 for
  `connection_lost`) is a starting point; the score shown next to each
  trigger (e.g. `0.91/0.85`) is the live margin.
- `debounce_sec` defends against one-frame false positives — bump it if a
  fade-in transitions briefly through a state that looks like your template.
- `cooldown_sec` (60s by default) prevents re-firing while a lingering
  overlay is still on screen.
- Unticking a checkbox does **not** stop a currently-playing macro (use
  **Stop** in the Macros panel for that).

Triggers are JSON specs in `<trigger-dir>/<name>.json`; reference images
live in `<trigger-dir>/images/`. Portable across hosts. Edit the JSON to
retune thresholds — the seeder is idempotent and won't clobber your edits.

## CLI

```
nxml-coplay --game GAME --policy URI --controller URL
            [--controller-input MAPPER] [--device-path PATH]   # local pad selection
            [--camera N] [--capture-width PX] [--capture-height PX]
            [--input-source evdev|web]                         # default: evdev
            [--web-host HOST] [--web-port N]                   # web bind (default: 0.0.0.0:8080)
            [--web-token TOKEN] [--web-stick-deadzone F]       # web mode
            [--mode human-priority|human-takeover]             # default: human-priority
            [--record DIR]                                     # optional recording root
            [--macro-dir DIR] [--trigger-dir DIR]              # default: ./data/{macros,triggers}/<game>/
            [--vae-path URI]                                   # default: stabilityai/sd-vae-ft-mse
            [--device cuda|cpu] [--tick-hz N]                  # inference device, mux/POST rate
```

---

## Development

### Architecture

```
                    +-------------------+
                    | V4L2Source (cam)  |
                    +---------+---------+
                              |
                              v
              +---------------+---------------+
              |  background AI inference loop |
              |  - VAE encode latest frame    |
              |  - slide latent window (T)    |
              |  - policy_client.predict(...) |
              |  - cache action atomically    |
              +---------------+---------------+
                              |
                              v
   +-------+   +--------------+--------------+
   | evdev |   |  CachedAiSource (mux side)  |
   +---+---+   +--------------+--------------+
       |                      |
       v                      v
       +-----------+----------+
                   v
        ControllerMux + HumanPriority
                   |
                   v
       merged 26-dim action (numpy)
                   |
            +------+------+
            v             v
   POST /action     VideoParquetEpisodeWriter (--record)
   to orchestrator
```

The AI thread runs at the camera's rate (e.g., 30 Hz), independent from the
mux/POST tick rate (default 30 Hz, overridable). When AI inference is
slower than the tick rate, the AI source returns a slightly stale action;
the human source is always live. This is what "non-blocking" means here —
the human never has to wait for the policy.

### Current scope

- **Mux strategies**: `human-priority` (per-index merge, AI fills what the
  human isn't touching) and `human-takeover` (any human input fully
  suppresses the AI). Swap live via the web UI or set the initial value
  with `--mode`.
- **Human input**: local evdev or browser-fed `WebGamepadReader`. Mappers
  auto-detect from device name; pass an explicit id via `--controller-input`
  (e.g., `xbox_one`).
- **Transport out**: HTTP `POST /action` to `nxbt-orchestrator`.
- **Recording**: optional via `--record`, written by `nxml-capture`'s
  `VideoParquetEpisodeWriter` (ffv1 + parquet by default). Frames pair
  with the *merged* action (what the Switch saw), not the human's raw
  evdev state — so a 1 s pause replays as a 1 s pause.
- **No overlay UI** in evdev mode; the web-mode page is the only HUD.
- **Linux-only**, via the `nxml-mux` evdev dep.
