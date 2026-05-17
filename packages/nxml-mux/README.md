# nxml-mux

Action arbitration: multiplex multiple controller / policy / macro sources
into a single `nx_packets` action stream.

## Surface

- **`ActionSource` protocol** + **`CallableActionSource`** for synchronous
  in-process callers (macros, mocked AIs, in-process policies).
- **`ControllerMux`** — polls all sources, applies a strategy, exposes
  the merged latest action.
- **Strategies**:
  - **`HumanPriority`** — per-index merge: the human's actively-contributed
    indices win, AI/macro fills the rest non-blockingly.
  - **`HumanTakeover`** — all-or-nothing: any human input fully suppresses
    the AI for that tick.
- **Readers**:
  - **`EvdevReader`** — Linux gamepads via `/dev/input/event*`. Mappers
    are YAML in `input_devices/mappers/`; `auto_detect` matches a device
    to its mapper by name. Bundled mappers include `xbox_one`.
  - **`WebGamepadReader`** — push-driven source for browser-side gamepads;
    the page POSTs 26-dim vectors and the reader exposes them on the same
    `ActionSource` contract.

## Usage

```python
from nxml_mux import ControllerMux, HumanPriority
from nxml_mux.input_devices.auto_detect import detect_mapper_for_device
from nxml_mux.input_devices.readers import EvdevReader

human = EvdevReader(
    "/dev/input/event7",
    detect_mapper_for_device("/dev/input/event7"),
)
mux = ControllerMux(
    sources=[human, my_ai_source],
    strategy=HumanPriority({human.source_id}),
)

with mux:
    while running:
        action = mux.tick()          # merged 26-dim vector
        post_to_orchestrator(action)
```

`nxml-coplay`'s runner does exactly this with a `CachedAiSource` (background
inference loop) in place of `my_ai_source`.

## Platform support

Linux only. `EvdevReader` is a hard dep on `python-evdev`, which is a thin
wrapper over Linux's `/dev/input/event*` interface and does not run on
macOS or Windows. This matches the rest of the stack — `nxbt` (the
controller-emulator that `nxbt-orchestrator` wraps) needs raw L2CAP via
BlueZ, which also locks the ecosystem to Linux.

A macOS/Windows port would drop in a sibling reader (`HidReader` on top of
`hidapi`, say) without touching anything else in this package. The pieces
that already isolate the OS-specific code:

- `ActionSource` is a Protocol — any class with `source_id`, `latest()`,
  `start()`, `stop()` works regardless of how it gets events.
- `Mapper` YAMLs name buttons/axes by evdev code strings today, but the
  schema has no evdev-specific structure. A HID-flavored mapper is just a
  new YAML in the same shape with HID usage names instead of `BTN_*` /
  `ABS_*`.
- `ControllerMux`, the strategies, `CallableActionSource`,
  `WebGamepadReader`, the registry, and `detect_mapper_for_name` are all
  OS-agnostic.

Only `detect_mapper_for_device` and `EvdevReader` import `evdev`. Everything
else is portable as-is. (The bigger blocker for non-Linux is
`nxbt-orchestrator` — that's a separate port and a much larger one.)
