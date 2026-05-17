# nxbt-orchestrator

Bluetooth controller server for the nxml ecosystem. Wraps
[nxbt](https://github.com/Brikwerk/nxbt) (Pro Controller emulation over BlueZ)
and exposes the controller as an HTTP/WebSocket service — accepts action
frames over `POST /action`, streams live controller state for time-aligned
capture, and gates inference actions when a human is also driving (see
[Human override](#human-override)).

## Install

This package depends on `nxbt` and `pybluez`, which are pinned to Python
3.11 and need `sudo` at runtime to bind the BlueZ HCI socket. It does
**not** install into the workspace 3.14 venv — install it standalone with
`uv tool`.

### System libraries (Ubuntu/Debian)

The C extensions `dbus-python` and `pybluez` compile against system
headers. Install these first:

```bash
sudo apt-get install \
    bluetooth bluez libbluetooth-dev \
    libdbus-1-dev libdbus-glib-1-dev \
    pkg-config
```

### Python 3.11

**Use a system Python 3.11, not uv's managed one.** `nxbt` opens an L2CAP
socket via `socket.AF_BLUETOOTH`, which is only present in a CPython built
against `libbluetooth-dev`. uv-managed Pythons (from
[`python-build-standalone`](https://github.com/astral-sh/python-build-standalone))
deliberately strip optional modules including `AF_BLUETOOTH`, so they will
install fine but crash on first connect with:

```
AttributeError: module 'socket' has no attribute 'AF_BLUETOOTH'
```

The fix is to install Ubuntu's Python 3.11 from the deadsnakes PPA
(its build includes bluetooth support) and point uv at it explicitly:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.11 python3.11-dev
```

### Install from a local checkout

```bash
uv tool install --python /usr/bin/python3.11 ./packages/nxbt-orchestrator
# Or from inside the package directory:
#   cd packages/nxbt-orchestrator && uv tool install --python /usr/bin/python3.11 .
```

### Install from git (no local checkout)

```bash
uv tool install --python /usr/bin/python3.11 \
    "git+https://github.com/csaben/nxml.git#subdirectory=packages/nxbt-orchestrator"

# Pin to a tag or commit:
#   "git+https://github.com/csaben/nxml.git@v0.1.0#subdirectory=packages/nxbt-orchestrator"
```

uv clones the whole repo so the `../nx-packets` path source resolves.

Passing the absolute path (`/usr/bin/python3.11`) is what forces uv to use
the system interpreter; bare `--python 3.11` may match either a system or
a previously-downloaded managed Python depending on what uv discovers
first.

## Run

`sudo` is required for BlueZ HCI socket access. The `-E env … PATH=$PATH`
prefix preserves the calling user's `PATH` so `nxbt-orchestrator` resolves
out of `~/.local/share/uv/tools/...`:

```bash
sudo -E env "PYTHONDONTWRITEBYTECODE=1" PATH=$PATH \
    nxbt-orchestrator serve --reconnect-address "AA:BB:CC:DD:EE:FF"
```

## Pairing

- On first run, hold the L+R button on a real controller next to the Switch's
  *Change Grip / Order* screen so the Switch broadcasts. nxbt will pair and
  the script will stay connected.
- If the connection is forcibly broken, the Switch may need to be **fully
  powered off** before Linux can re-pair. This is a Switch / BlueZ quirk, not
  a bug in this package.
- `--reconnect-address <BT-MAC>` makes nxbt try to re-establish a previously
  paired connection without re-doing the *Change Grip / Order* dance. After
  the first successful pair the server logs the Switch's MAC; pass it back on
  subsequent runs.

## CLI

```
nxbt-orchestrator serve [--host 127.0.0.1] [--port 7777]
                        [--update-rate 120]
                        [--override-window 0.3]
                        [--reconnect-address AA:BB:CC:DD:EE:FF]
                        [--debug]
```

---

## Endpoints

- **`POST /action`** — apply one frame of controller state (full
  `Packet` JSON or 26-dim float action vector from `nx-packets`). Accepts
  an optional `source` field (`"human"` | `"inference"`, default
  `"inference"`).
- **`POST /buttons`** — set of held button names; treated as a human-source
  update.
- **`POST /stick`** — analog stick update for one stick (`LEFT_STICK` /
  `RIGHT_STICK`); treated as a human-source update.
- **`POST /macro`** / **`POST /control`** — fire nxbt macros, toggle
  recording, set the recording output path.
- **`GET /state`** — one-shot snapshot of the current controller state.
- **`GET /health`** — liveness + connection probe.
- **`WebSocket /ws/state`** — live state stream at the update rate, used
  by `nxml-capture` for frame ↔ action time alignment during data
  collection.

### Human override

When actions arrive both from a human source (e.g., a separate
`nxml-mux` connection) and an inference source, inference is ignored
for `--override-window` seconds after any human input. Callers opt in by
setting `"source": "human"` on the relevant `/action` POSTs.
