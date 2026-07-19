"""Benchmark-only inference v2 client; never selects, promotes, or arms a model."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dagger_inference_v2 import InferenceV2Client


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://100.80.98.4:5557")
    parser.add_argument("--control-url", default="http://100.80.98.4:8787")
    parser.add_argument("--token", type=Path, required=True)
    parser.add_argument("--revision-id", required=True)
    parser.add_argument("--preview-url", default="http://100.73.109.68:8091/stream.mjpeg")
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
    client = InferenceV2Client(args.endpoint, revision, timeout_ms=args.timeout_ms)
    info = client.connect()
    latencies: list[float] = []
    processing: list[float] = []
    warming = 0
    proposals = 0
    sequence = 0
    buffer = b""
    started = time.monotonic()
    try:
        with urllib.request.urlopen(args.preview_url, timeout=3) as stream:
            while proposals < args.samples:
                buffer += stream.read(65536)
                begin = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", begin + 2) if begin >= 0 else -1
                if begin < 0 or end < 0:
                    continue
                jpeg = buffer[begin : end + 2]
                buffer = buffer[end + 2 :]
                result = client.predict_frame(time.monotonic_ns(), jpeg)
                sequence += 1
                latencies.append(result.transport_latency_ns / 1e6)
                processing.append((result.processing_latency_ns or 0) / 1e6)
                if result.action is None:
                    warming += 1
                else:
                    proposals += 1
    finally:
        client.close()
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "benchmark_only": True,
                "armed": False,
                "info": info,
                "frames_sent": sequence,
                "warming": warming,
                "proposals": proposals,
                "elapsed_seconds": elapsed,
                "proposal_hz": proposals / elapsed,
                "transport_ms": {
                    "p50": statistics.median(latencies),
                    "p95": percentile(latencies, 0.95),
                    "max": max(latencies),
                },
                "processing_ms": {
                    "p50": statistics.median(processing),
                    "p95": percentile(processing, 0.95),
                    "max": max(processing),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
