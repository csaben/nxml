"""Benchmark-only inference v2 client; never selects, promotes, or arms a model."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).parent))
from dagger_inference_v2 import InferenceV2Client, InferenceV2Error


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


class LatestPreview:
    def __init__(self, url: str):
        self.url = url
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.latest: tuple[int, int, bytes] | None = None
        self.frames = 0
        self.started = 0.0
        self.ended = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True, name="v2-bench-preview")

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=3)

    def after(self, sequence: int, timeout: float = 2.0) -> tuple[int, int, bytes]:
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.stop.is_set():
                if self.latest is not None and self.latest[0] > sequence:
                    return self.latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("preview source stopped producing frames")
                self.condition.wait(remaining)
        raise RuntimeError("preview source stopped")

    def _run(self):
        buffer = b""
        self.started = time.monotonic()
        with urllib.request.urlopen(self.url, timeout=3) as stream:
            while not self.stop.is_set():
                buffer += stream.read(65536)
                while True:
                    begin = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9", begin + 2) if begin >= 0 else -1
                    if begin < 0 or end < 0:
                        break
                    jpeg = buffer[begin : end + 2]
                    buffer = buffer[end + 2 :]
                    with self.condition:
                        self.frames += 1
                        self.latest = (self.frames, time.monotonic_ns(), jpeg)
                        self.condition.notify_all()
        self.ended = time.monotonic()


class NeutralInput:
    def __init__(self, url: str):
        self.url = url
        self.stop = threading.Event()
        self.send_ms: list[float] = []
        self.intervals_ms: list[float] = []
        self.thread = threading.Thread(target=self._run, daemon=True, name="v2-bench-input")

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)

    def _run(self):
        asyncio.run(self._async_run())

    async def _async_run(self):
        neutral = json.dumps({"vector": [0.0] * 26})
        async with websockets.connect(self.url, open_timeout=3) as socket:
            deadline = time.perf_counter()
            last = None
            while not self.stop.is_set():
                now = time.perf_counter()
                if last is not None:
                    self.intervals_ms.append((now - last) * 1000)
                last = now
                started = time.perf_counter()
                await socket.send(neutral)
                self.send_ms.append((time.perf_counter() - started) * 1000)
                deadline += 1 / 60
                await asyncio.sleep(max(0, deadline - time.perf_counter()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://100.80.98.4:5557")
    parser.add_argument("--control-url", default="http://100.80.98.4:8787")
    parser.add_argument("--token", type=Path, required=True)
    parser.add_argument("--revision-id", required=True)
    parser.add_argument("--preview-url", default="http://100.73.109.68:8091/stream.mjpeg")
    parser.add_argument("--input-url", default="ws://100.73.109.68:8091/ws")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--timeout-ms", type=int, default=100)
    args = parser.parse_args()

    token = args.token.read_text().strip()
    request = urllib.request.Request(
        f"{args.control_url.rstrip('/')}/v1/models/revisions/{args.revision_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        revision = json.load(response)

    preview = LatestPreview(args.preview_url)
    neutral = NeutralInput(args.input_url)
    preview.start()
    neutral.start()
    client = InferenceV2Client(args.endpoint, revision, timeout_ms=args.timeout_ms)
    info = client.connect()
    transport_ms: list[float] = []
    processing_ms: list[float] = []
    network_overhead_ms: list[float] = []
    total_ms: list[float] = []
    warming_total_ms: list[float] = []
    source_drops = 0
    warming = 0
    proposals = 0
    sequence = -1
    sent = 0
    started = time.monotonic()
    first_proposal_at = None
    try:
        while proposals < args.samples:
            next_sequence, frame_ns, jpeg = preview.after(sequence)
            if sequence >= 0:
                source_drops += max(0, next_sequence - sequence - 1)
            sequence = next_sequence
            result = client.predict_frame(frame_ns, jpeg)
            sent += 1
            if result.action is None:
                warming += 1
                warming_total_ms.append((result.received_monotonic_ns - frame_ns) / 1e6)
            else:
                if first_proposal_at is None:
                    first_proposal_at = time.monotonic()
                transport_ms.append(result.transport_latency_ns / 1e6)
                processing_ms.append((result.processing_latency_ns or 0) / 1e6)
                network_overhead_ms.append(
                    (result.transport_latency_ns - (result.processing_latency_ns or 0)) / 1e6
                )
                total_ms.append((result.received_monotonic_ns - frame_ns) / 1e6)
                proposals += 1

        measurement_ended = time.monotonic()
        measurement_elapsed = measurement_ended - started
        steady_elapsed = measurement_ended - first_proposal_at

        # Force a frame request to time out, discard that socket, then prove a
        # new INFO+RESET epoch warms from neutral and never returns the old reply.
        client.close()
        timeout_client = InferenceV2Client(args.endpoint, revision, timeout_ms=1)
        timeout_observed = False
        try:
            timeout_client.connect()
            sequence, frame_ns, jpeg = preview.after(sequence)
            timeout_client.predict_frame(frame_ns, jpeg)
        except InferenceV2Error:
            timeout_observed = True
        finally:
            timeout_client.close()
        recovery = InferenceV2Client(args.endpoint, revision, timeout_ms=args.timeout_ms)
        recovery_info = recovery.connect()
        recovery_warming = 0
        recovery_first_proposal_frame = None
        while recovery_first_proposal_frame is None:
            next_sequence, frame_ns, jpeg = preview.after(sequence)
            sequence = next_sequence
            result = recovery.predict_frame(frame_ns, jpeg)
            if result.action is None:
                recovery_warming += 1
            else:
                if result.frame_timestamp_ns != frame_ns:
                    raise RuntimeError("recovery reused a stale proposal")
                recovery_first_proposal_frame = frame_ns
        recovery.close()
    finally:
        client.close()
        neutral.close()
        preview.close()
    elapsed = time.monotonic() - started
    preview_elapsed = (preview.ended or time.monotonic()) - preview.started
    print(
        json.dumps(
            {
                "benchmark_only": True,
                "armed": False,
                "info": info,
                "frames_sent": sent,
                "warming": warming,
                "proposals": proposals,
                "source_frames_skipped_latest_wins": source_drops,
                "total_harness_seconds": elapsed,
                "measurement_seconds": measurement_elapsed,
                "proposal_hz_including_warmup": proposals / measurement_elapsed,
                "steady_proposal_hz": (proposals - 1) / steady_elapsed,
                "warming_observation_to_response_ms": summary(warming_total_ms),
                "network_roundtrip_ms": summary(transport_ms),
                "network_and_edge_overhead_ms": summary(network_overhead_ms),
                "cluster_processing_ms": summary(processing_ms),
                "observation_to_proposal_ms": summary(total_ms),
                "freshest_observation_to_proposal_ms": min(total_ms),
                "preview_frames": preview.frames,
                "preview_fps": preview.frames / preview_elapsed,
                "input_neutral_frames": len(neutral.send_ms),
                "input_send_ms": summary(neutral.send_ms),
                "input_cadence_ms": summary(neutral.intervals_ms),
                "timeout_observed": timeout_observed,
                "reconnect_info_identity_match": recovery_info == info,
                "recovery_warming_frames": recovery_warming,
                "stale_proposal_reused": False,
                "recovery_first_proposal_frame_timestamp_ns": recovery_first_proposal_frame,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
