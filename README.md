# nxml

Workspace for training and deploying ML policies for Nintendo Switch games —
from data collection (real Switch over Bluetooth) through world-model and
RL training to live hybrid human/AI play.

## Tools

CLIs you run directly. Each ships its own README with install + usage.

| Tool                 | What it does                                                                 | README |
| -------------------- | ---------------------------------------------------------------------------- | ------ |
| `nxbt-orchestrator`  | HTTP/WS server that drives a real Switch as a Pro Controller over BlueZ.     | [`packages/nxbt-orchestrator`](packages/nxbt-orchestrator/README.md) |
| `nxml-collect`       | Records paired `(frame, action)` episodes from a live session.               | [`apps/nxml-collect`](apps/nxml-collect/README.md) |
| `nxml-coplay`        | Hybrid play: human drives a PC controller while a policy fills in alongside. | [`apps/nxml-coplay`](apps/nxml-coplay/README.md) |
| `nxwm`               | World-model training, encoding, inference, ZMQ/HTTP serve, web UI.           | [`packages/nxwm`](packages/nxwm/README.md) |
| `nxrl`               | BC + PPO training, policy serve (ZMQ), eval GIF.                             | [`packages/nxrl`](packages/nxrl/README.md) |

## Libraries

Workspace packages with no CLI — consumed by the tools above.

| Package         | What it provides                                                             |
| --------------- | ---------------------------------------------------------------------------- |
| `nx-packets`    | Switch input types and the canonical **26-dim action vector contract**. Single source of truth for action layout. |
| `nxml-core`     | Protocols, registry, self-describing checkpoint format, URI resolver. Torch-free. |
| `nx-macros`     | Macro schema, recorder, player, on-disk store.                               |
| `nxml-capture`  | Frame capture, controller-state subscription, frame↔action synchronizer, episode writers. |
| `nxml-mux`      | `ControllerMux` + strategies (`HumanPriority`, `HumanTakeover`); evdev / web gamepad readers. |
| `nxml-games`    | Game-specific reward shaping, terminations, seeds, detectors. Currently: Pokémon ZA. |

## Live flows

**nxml-collect** — record paired `(frame, action)` episodes:

```
  human pad  ──▶  nxbt-orchestrator  ──▶  Switch
                          │
                     /ws/state
                          ▼
  capture card  ──▶  nxml-collect  ──▶  .mkv + .parquet
```

**nxml-coplay** — hybrid human/AI play:

```
                       ┌──▶  VAE  ──▶  policy (local or zmq://)  ─┐
  capture card  ──────┤                                            ├──▶  mux  ──▶  POST /action  ──▶  nxbt-orchestrator  ──▶  Switch
                       │      human pad (evdev or web)  ──────────┘      │
                       └──▶  episode recording (optional, paired with the merged action)
```

**nxwm ui** — drive a trained world model directly, no Switch needed:

```
  seed episode (.npz)  ──▶  initial context window
                                       │
  human (browser)  ──▶  action  ──▶  world model  ──▶  latent  ──▶  VAE decode  ──▶  JPEG  ──▶  browser
                                       (in-process, or remote via zmq:// / http(s)://)
```

## Develop in the workspace

```bash
uv sync --all-packages --all-extras   # editable install of every member (incl. wandb)
just train-tiny              # smoke train (~60 s) for the dit_v1 world model
just test                    # pytest across all packages
```

The workspace is Python 3.14, except `nxbt-orchestrator` which is pinned to
3.11 (because of `nxbt` + `pybluez`) and installed separately via `uv tool`.

## Standalone install (no clone)

Each tool can be installed without cloning the repo via:

```
uv tool install "git+https://github.com/csaben/nxml.git#subdirectory=<path>"
```

See the per-tool README for the exact command and any system prerequisites
(e.g., BlueZ for `nxbt-orchestrator`, v4l2 for the capture path).
