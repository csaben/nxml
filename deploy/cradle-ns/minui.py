"""Minimal tailnet play page: browser gamepad -> orchestrator + live preview.

A deliberately tiny validation UI with none of the edge dashboard around it:
one page, one WebSocket for 60 Hz fire-and-forget gamepad frames, one
zero-transcode MJPEG stream for the capture. No auth — bind it to the
tailscale interface IP only.

  uv run python deploy/cradle-ns/minui.py --host 100.73.109.68 --port 8091

Then open http://cradle-ns:8091/ from any tailnet device and press a button
on the gamepad. Starting the preview preempts any other ffmpeg holding the
capture device (e.g. the edge dashboard's preview).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http.client
import ipaddress
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from nxml_capture.backends.ffmpeg_v4l2 import v4l2_mjpeg_stream_command

sys.path.insert(0, str(Path(__file__).parent))
from dagger_status import OperationsReader

ACTION_DIM = 26
CAPTURE_DEVICE = "/dev/v4l/by-id/usb-MACROSILICON_Hagibis_20210623-video-index0"


def validate_tailnet_bind(host: str) -> str:
    """Accept only a direct Tailscale IPv4 interface address."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("bind host must be a literal Tailscale IPv4 address") from error
    if address.version != 4 or address not in ipaddress.ip_network("100.64.0.0/10"):
        raise ValueError("bind host must be a Tailscale IPv4 address in 100.64.0.0/10")
    return host

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NXML DAgger</title>
<style>
  body { margin:0; background:#0b0d10; color:#e8edf3; font:14px system-ui,sans-serif; }
  img { display:block; width:100vw; max-height:88vh; object-fit:contain; background:#000; }
  #bar { display:flex; gap:1.5rem; padding:.5rem 1rem; align-items:center; }
  #ops { display:grid; grid-template-columns:repeat(4,minmax(10rem,1fr)); gap:.5rem; padding:0 1rem 1rem; }
  .card { background:#171b21; border:1px solid #29313a; border-radius:.4rem; padding:.65rem; }
  .label { color:#9aa7b5; font-size:.75rem; text-transform:uppercase; }
  .value { margin-top:.25rem; }
  .on { color:#51cf66 } .off { color:#ff6b6b }
</style></head><body>
<img id="preview" src="/stream.mjpeg" alt="Switch capture">
<div id="bar">
  <span id="ws" class="off">input: disconnected</span>
  <span id="pad">pad: press any gamepad button</span>
  <span id="seq"></span>
</div>
<section id="ops">
 <div class="card"><div class="label">Session</div><div class="value" id="session">human · idle</div></div>
 <div class="card"><div class="label">Local spool</div><div class="value" id="spool">loading…</div></div>
 <div class="card"><div class="label">Cluster</div><div class="value" id="cluster">loading…</div></div>
 <div class="card"><div class="label">Model</div><div class="value" id="model">loading…</div></div>
</section>
<script>
  const HZ=60, DEADZONE=0.15, DIM=26;
  const MAP={10:4,11:5,12:6,14:7,15:8,13:9,4:10,6:11,5:12,7:13,9:18,8:19,16:20,2:22,3:23,0:24,1:25};
  let ws=null, seq=0, timer=null;
  const $=id=>document.getElementById(id);
  function pad(){const l=navigator.getGamepads?navigator.getGamepads():[];for(const g of l)if(g&&g.mapping==='standard')return g;return null}
  function dz(v){v=Number(v)||0;return Math.abs(v)<DEADZONE?0:Math.max(-1,Math.min(1,v))}
  function vec(g){const a=new Array(DIM).fill(0);a[0]=dz(g.axes[0]);a[1]=-dz(g.axes[1]);a[2]=dz(g.axes[2]);a[3]=-dz(g.axes[3]);for(const[b,i]of Object.entries(MAP)){const x=g.buttons[Number(b)];a[i]=x&&x.pressed?1:0}return a}
  function connect(){
    if(ws)return; const g=pad(); if(!g)return;
    ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);
    ws.onopen=()=>{ $('ws').textContent='input: streaming '+HZ+' Hz'; $('ws').className='on'; $('pad').textContent='pad: '+g.id; seq=0; loop(); };
    ws.onclose=()=>{ teardown('disconnected'); };
    ws.onerror=()=>{ teardown('error'); };
  }
  function loop(){
    if(!ws||ws.readyState!==1)return;
    const g=pad(); if(!g){teardown('gamepad lost');return}
    if(ws.bufferedAmount<4096){ ws.send(JSON.stringify({vector:vec(g)})); seq++; }
    $('seq').textContent='seq '+seq;
    timer=setTimeout(loop,1000/HZ);
  }
  function teardown(msg){ if(timer){clearTimeout(timer);timer=null} if(ws){try{ws.close()}catch(e){}ws=null} $('ws').textContent='input: '+msg; $('ws').className='off'; }
  window.addEventListener('gamepadconnected',connect);
  window.addEventListener('focus',connect);
  window.addEventListener('blur',()=>teardown('paused (window blur)'));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)teardown('paused (hidden)')});
  setInterval(()=>{if(!ws)connect()},1000);
  async function pollOps(){
    try { const s=await fetch('/api/ops/status',{cache:'no-store'}).then(r=>r.json());
      $('session').textContent=`${s.session.mode} · ${s.session.recording_state}`;
      const p=s.spool; $('spool').textContent=p?`${p.pending_episodes||0} pending · ${p.episodes_uploaded||0} shipped`:`unavailable: ${s.errors.spool||'unknown'}`;
      const c=s.cluster; $('cluster').textContent=c?`${(c.datasets.datasets||[]).length} datasets · ${(c.snapshots.snapshots||[]).length} snapshots`:`unavailable: ${s.errors.cluster||'unknown'}`;
      const d=c&&c.deployment; $('model').textContent=d&&d.active_revision?`active ${d.active_revision.slice(0,8)} · gen ${d.generation}`:'none active';
    } catch(e) { $('cluster').textContent='operations status unavailable'; }
  }
  pollOps(); setInterval(pollOps,5000);
</script></body></html>"""


class OrchestratorClient:
    def __init__(self, host: str, port: int, timeout: float = 1.0) -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self._conn: http.client.HTTPConnection | None = None
        self._lock = threading.Lock()

    def post_action(self, vector: list[float]) -> None:
        payload = json.dumps({"vector": vector, "source": "human"}).encode()
        headers = {"Content-Type": "application/json"}
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._conn is None:
                        self._conn = http.client.HTTPConnection(
                            self.host, self.port, timeout=self.timeout
                        )
                    self._conn.request("POST", "/action", body=payload, headers=headers)
                    self._conn.getresponse().read()
                    return
                except OSError, http.client.HTTPException:
                    self._conn = None
                    if attempt == 2:
                        raise


def create_app(
    orchestrator: OrchestratorClient,
    capture_device: str,
    operations: OperationsReader | None = None,
) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        if operations is not None:
            operations.start()
        try:
            yield
        finally:
            if operations is not None:
                operations.stop()

    app = FastAPI(title="nxml-minui", lifespan=lifespan)
    stream_state: dict[str, subprocess.Popen | None] = {"proc": None}
    stream_lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(PAGE)

    @app.get("/api/ops/status")
    def ops_status() -> dict:
        if operations is None:
            return {"schema_version": "nxml.dagger-operations-status.v1", "observed_at": time.time(), "session": {"schema_version": "nxml.dagger-session-status.v1", "mode": "human", "recording_state": "idle", "recording_available": False, "recording_blocked_reason": "Operations polling is not configured"}, "spool": None, "cluster": None, "errors": {"operations": "not configured"}}
        return operations.snapshot().wire()

    @app.websocket("/ws")
    async def ws_input(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                message = await ws.receive_text()
                try:
                    vector = json.loads(message).get("vector")
                except ValueError:
                    break
                if (
                    not isinstance(vector, list)
                    or len(vector) != ACTION_DIM
                    or not all(isinstance(v, (int, float)) and -1 <= v <= 1 for v in vector)
                ):
                    break
                await asyncio.to_thread(orchestrator.post_action, [float(v) for v in vector])
        except WebSocketDisconnect:
            pass
        finally:
            # Synchronous on purpose: must run even under task cancellation.
            with contextlib.suppress(Exception):
                orchestrator.post_action([0.0] * ACTION_DIM)

    @app.get("/stream.mjpeg")
    def stream() -> StreamingResponse:
        def frames():
            with stream_lock:
                old = stream_state["proc"]
                if old is not None:
                    old.kill()
                # Preempt any other ffmpeg holding the capture device
                # (e.g. the edge dashboard preview in another tab).
                subprocess.run(
                    ["pkill", "-f", f"ffmpeg.*{capture_device.rsplit('/', 1)[-1]}"],
                    check=False,
                )
                time.sleep(0.2)
                proc = subprocess.Popen(
                    v4l2_mjpeg_stream_command(capture_device),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
                stream_state["proc"] = proc
            try:
                assert proc.stdout is not None
                while True:
                    chunk = proc.stdout.read(65536)
                    if not chunk:
                        break
                    yield chunk
            finally:
                proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
                with stream_lock:
                    if stream_state["proc"] is proc:
                        stream_state["proc"] = None

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=ffmpeg",
            headers={"Cache-Control": "no-store, private"},
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="bind address (use the tailscale IP)")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--orchestrator-host", default="127.0.0.1")
    parser.add_argument("--orchestrator-port", type=int, default=7777)
    parser.add_argument("--capture", default=CAPTURE_DEVICE)
    parser.add_argument("--spool-status", type=Path, default=Path("~/.local/state/nxml-spool/status.json").expanduser())
    parser.add_argument("--cluster-url", default="http://100.80.98.4:8787")
    parser.add_argument("--cluster-token", type=Path, default=Path("~/.config/nxml/cluster.token").expanduser())
    args = parser.parse_args()
    try:
        validate_tailnet_bind(args.host)
    except ValueError as error:
        parser.error(str(error))
    client = OrchestratorClient(args.orchestrator_host, args.orchestrator_port)
    operations = OperationsReader(
        spool_status_path=args.spool_status,
        cluster_url=args.cluster_url,
        cluster_token_path=args.cluster_token,
    )
    uvicorn.run(create_app(client, args.capture, operations), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
