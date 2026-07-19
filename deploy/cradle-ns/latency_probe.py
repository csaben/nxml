"""Minimal latency bisection probe for the Switch control stack.

Isolates each hop of: gamepad -> browser -> tailnet -> edge -> orchestrator
-> nxbt -> Bluetooth -> Switch. Run on cradle-ns.

Modes:

  rtt     POST neutral /action frames as fast as possible and report RTT
          percentiles. Measures orchestrator ingest cost only.

  pulse   Press A for 150 ms every 2 s. Watch the Switch screen and compare
          against the printed timestamps. Measures the orchestrator ->
          nxbt -> Bluetooth -> Switch half with no UI, browser, or network.

  evdev   Stream a locally connected controller (auto-detected via the
          nxml-mux mapper registry) straight to the orchestrator at 60 Hz.
          The full input path minus browser/tailnet/edge. If this feels
          snappy while the web UI feels laggy, the problem is upstream of
          the edge host (browser sampling or tailnet RTT/DERP relay).

  measure Closed-loop, no human needed: taps DPAD RIGHT then LEFT via the
          orchestrator and times how long the HDMI capture takes to show a
          screen change. Measures POST -> nxbt -> Bluetooth -> Switch ->
          HDMI -> capture; includes ~1-3 frames (33-100 ms) of camera
          pipeline latency, so treat values as an upper bound. Requires a
          screen where the d-pad visibly moves a cursor (e.g. HOME menu)
          and the capture device to be free (close the dashboard preview).

Usage:

  uv run python deploy/cradle-ns/latency_probe.py rtt
  uv run python deploy/cradle-ns/latency_probe.py pulse
  uv run python deploy/cradle-ns/latency_probe.py evdev

Reading /dev/input/event* may require membership in the `input` group.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import sys
import time

ACTION_DIM = 26
A_BUTTON_INDEX = 25  # switch_packets.v1: index 25 is A
DPAD_LEFT_INDEX = 7
DPAD_RIGHT_INDEX = 8
CAPTURE_DEVICE = "/dev/v4l/by-id/usb-MACROSILICON_Hagibis_20210623-video-index0"


class OrchestratorClient:
    """Persistent-connection client so probe overhead stays out of the numbers."""

    def __init__(self, host: str, port: int, timeout: float = 1.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._conn: http.client.HTTPConnection | None = None

    def _connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return self._conn

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        try:
            conn = self._connection()
            conn.request(method, path, body=payload, headers=headers)
            response = conn.getresponse()
            data = response.read()
        except OSError, http.client.HTTPException:
            self._conn = None  # reconnect once on a dropped keep-alive
            conn = self._connection()
            conn.request(method, path, body=payload, headers=headers)
            response = conn.getresponse()
            data = response.read()
        if response.status >= 400:
            raise RuntimeError(f"{method} {path} -> {response.status}: {data[:200]!r}")
        return json.loads(data)

    def post_action(self, vector: list[float]) -> float:
        """Post one human action frame; returns the RTT in milliseconds."""
        started = time.perf_counter()
        self.request("POST", "/action", {"vector": vector, "source": "human"})
        return (time.perf_counter() - started) * 1000.0


def neutral() -> list[float]:
    return [0.0] * ACTION_DIM


def print_health(client: OrchestratorClient) -> None:
    health = client.request("GET", "/health")
    print(f"orchestrator /health: {health}")
    if health.get("switch_state") != "connected":
        print("WARNING: Switch is not connected — probe results are meaningless.")


def cmd_rtt(client: OrchestratorClient, args: argparse.Namespace) -> None:
    print_health(client)
    rtts = [client.post_action(neutral()) for _ in range(args.count)]
    rtts.sort()
    print(
        f"orchestrator /action RTT over {args.count} frames: "
        f"p50={statistics.median(rtts):.2f}ms "
        f"p95={rtts[int(len(rtts) * 0.95)]:.2f}ms max={rtts[-1]:.2f}ms"
    )
    print("(> a few ms here would implicate the orchestrator itself)")


def cmd_pulse(client: OrchestratorClient, args: argparse.Namespace) -> None:
    print_health(client)
    press = neutral()
    press[A_BUTTON_INDEX] = 1.0
    print("Pressing A for 150 ms every 2 s — watch the Switch screen for the delay.")
    print("Ctrl-C to stop (a neutral frame is always sent last).")
    try:
        while True:
            rtt = client.post_action(press)
            print(f"{time.strftime('%H:%M:%S')} A DOWN  (post rtt {rtt:.2f}ms)", flush=True)
            time.sleep(0.15)
            client.post_action(neutral())
            time.sleep(1.85)
    except KeyboardInterrupt:
        pass
    finally:
        client.post_action(neutral())


def cmd_evdev(client: OrchestratorClient, args: argparse.Namespace) -> None:
    import evdev
    from nxml_mux.input_devices.auto_detect import detect_mapper_for_name
    from nxml_mux.input_devices.readers.evdev_reader import EvdevReader

    print_health(client)
    device_path = args.device
    mapper = None
    if device_path is None:
        for candidate in evdev.list_devices():
            name = evdev.InputDevice(candidate).name
            mapper = detect_mapper_for_name(name)
            if mapper is not None:
                device_path = candidate
                print(f"using {candidate} ({name}) via mapper {mapper.name}")
                break
        if device_path is None:
            sys.exit("no evdev device matched a registered mapper (try --device /dev/input/eventN)")
    else:
        mapper = detect_mapper_for_name(evdev.InputDevice(device_path).name)
        if mapper is None:
            sys.exit(f"no mapper matches {device_path}")

    reader = EvdevReader(device_path, mapper)
    reader.start()
    period = 1.0 / args.hz
    rtts: list[float] = []
    window_start = time.monotonic()
    frames = 0
    print(f"streaming to orchestrator at {args.hz} Hz — Ctrl-C to stop.")
    try:
        while True:
            tick = time.monotonic()
            snapshot = reader.latest()
            vector = snapshot.action.tolist() if snapshot is not None else neutral()
            rtts.append(client.post_action(vector))
            frames += 1
            if tick - window_start >= 1.0:
                print(
                    f"{frames} f/s, post rtt p50={statistics.median(rtts):.2f}ms "
                    f"max={max(rtts):.2f}ms",
                    flush=True,
                )
                rtts.clear()
                frames = 0
                window_start = tick
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        client.post_action(neutral())


def cmd_measure(client: OrchestratorClient, args: argparse.Namespace) -> None:
    import cv2
    import numpy as np

    print_health(client)
    # cv2's V4L2 backend rejects by-id symlinks; hand it the real node.
    device = os.path.realpath(args.capture)
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        sys.exit(
            "could not open capture device — close anything holding it "
            "(e.g. the dashboard preview tab), then retry"
        )

    def grab_gray() -> tuple[float, np.ndarray]:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("capture read failed mid-run")
        small = cv2.resize(frame, (320, 180))
        return time.perf_counter(), cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)

    def frame_delta(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).mean())

    # Warm up and learn the idle screen's animation noise so pulsing
    # cursors or clocks don't count as the response.
    for _ in range(10):
        grab_gray()
    _, reference = grab_gray()
    noise = []
    for _ in range(30):
        _, frame = grab_gray()
        noise.append(frame_delta(frame, reference))
        reference = frame
    threshold = max(3.0 * max(noise), 2.0)
    print(f"idle frame delta max={max(noise):.2f}, change threshold={threshold:.2f}")

    # --dither keeps the HID report stream continuously changing (an
    # imperceptible +-1/100 stick wiggle) so nxbt's report cache never goes
    # quiet. If latency collapses with dither on, the culprit is the idle
    # report stream letting the Bluetooth link sink into sniff mode.
    dither_phase = 0.0

    def post(vector: list[float]) -> None:
        nonlocal dither_phase
        frame = list(vector)
        if args.dither:
            dither_phase = 0.01 if dither_phase <= 0.0 else -0.01
            frame[0] = frame[0] + dither_phase
        client.post_action(frame)

    directions = [DPAD_RIGHT_INDEX, DPAD_LEFT_INDEX] * (args.trials // 2 + 1)
    results: list[float] = []
    for trial, index in enumerate(directions[: args.trials]):
        _, before = grab_gray()
        press = neutral()
        press[index] = 1.0
        pressed_at = time.perf_counter()
        post(press)
        latency = None
        while time.perf_counter() - pressed_at < args.window:
            post(press)
            stamp, frame = grab_gray()
            if frame_delta(frame, before) > threshold:
                latency = (stamp - pressed_at) * 1000.0
                break
        post(neutral())
        name = "RIGHT" if index == DPAD_RIGHT_INDEX else "LEFT"
        if latency is None:
            print(f"trial {trial + 1} (dpad {name}): no screen change within {args.window:.1f}s")
        else:
            results.append(latency)
            print(f"trial {trial + 1} (dpad {name}): {latency:.0f} ms")
        # Let the screen settle before the next tap.
        settle = time.perf_counter()
        while time.perf_counter() - settle < 1.0:
            post(neutral())
            grab_gray()
    capture.release()
    if results:
        print(
            f"\npress -> screen-change latency over {len(results)} trials: "
            f"median={statistics.median(results):.0f} ms  "
            f"min={min(results):.0f}  max={max(results):.0f}"
        )
        print("(includes ~33-100 ms of capture pipeline; an official pad feels like <=50 ms)")
    else:
        print("\nno trials registered a screen change — is the screen d-pad responsive?")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["rtt", "pulse", "evdev", "measure"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--count", type=int, default=300, help="rtt mode: frames to send")
    parser.add_argument("--hz", type=float, default=60.0, help="evdev mode: stream rate")
    parser.add_argument("--device", default=None, help="evdev mode: explicit /dev/input/eventN")
    parser.add_argument("--capture", default=CAPTURE_DEVICE, help="measure mode: V4L2 device")
    parser.add_argument("--trials", type=int, default=6, help="measure mode: taps to time")
    parser.add_argument(
        "--window", type=float, default=2.0, help="measure mode: seconds to wait per tap"
    )
    parser.add_argument(
        "--dither",
        action="store_true",
        help="measure mode: keep the report stream busy with a tiny stick wiggle",
    )
    args = parser.parse_args()

    client = OrchestratorClient(args.host, args.port)
    {"rtt": cmd_rtt, "pulse": cmd_pulse, "evdev": cmd_evdev, "measure": cmd_measure}[args.mode](
        client, args
    )


if __name__ == "__main__":
    main()
