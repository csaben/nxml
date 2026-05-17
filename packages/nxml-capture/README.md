# nxml-capture

Frame capture, controller-state subscription, and frame↔action synchronization
for the nxml ecosystem.

## Components

- `source.CaptureSource` — protocol for any frame source (raw uint8 HWC).
- `backends.v4l2.V4L2Source` — cv2.VideoCapture-backed source for capture cards
  and webcams (Linux v4l2 path).
- `controller_subscribe.ControllerSubscription` — async client to
  `nxbt-orchestrator`'s `/ws/state` WebSocket, decoding each tick into the
  canonical 26-dim action vector.
- `synchronizer.Synchronizer` — pairs each frame with the most recent
  controller-state snapshot, producing `(timestamp, frame, action)` triples.
- `writers.video_parquet.VideoParquetEpisodeWriter` — canonical writer:
  ffv1/h264 video + parquet sidecar + manifest.
- `writers.npz.NpzEpisodeWriter` — debug writer: single compressed `.npz` +
  manifest.

## Usage

```python
from nxml_capture import (
    ControllerSubscription, Synchronizer, VideoParquetEpisodeWriter,
)
from nxml_capture.backends.v4l2 import V4L2Source

source = V4L2Source(camera_id=0)
controller = ControllerSubscription(url="ws://127.0.0.1:7777/ws/state")
sync = Synchronizer(source, controller)

writer = VideoParquetEpisodeWriter("./out/", codec="ffv1", fps=30.0)
with sync:
    for synced in sync.frames():
        writer.append(synced)
writer.close()
```

The above is what `nxml-collect` does under the hood. End-to-end recording
workflows should use that CLI rather than wiring the pieces up by hand —
this snippet is for library consumers who want to build their own collector.
