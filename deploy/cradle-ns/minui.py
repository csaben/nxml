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
import urllib.request
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from nxml_capture.backends.ffmpeg_v4l2 import v4l2_mjpeg_stream_command

sys.path.insert(0, str(Path(__file__).parent))
from dagger_action_plane import ActionPlane
from dagger_control import Mode, MuteMask
from dagger_inference import InferenceStatus
from dagger_inference_v2 import InferenceV2Client, RemoteInferenceWorker
from dagger_models import AtomicModelRuntime
from dagger_recording import HumanRecordingSession
from dagger_segments import SegmentClient, SegmentDeliveryWorker, SegmentJournal
from dagger_status import OperationsReader
from nxml_capture.backends.mjpeg_fanout import MjpegFanoutSource

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
  button { background:#2b3440; color:#eef3f8; border:1px solid #465363; border-radius:.35rem; padding:.4rem .7rem; }
  progress { width:100%; accent-color:#51cf66; } progress.warn { accent-color:#ffd43b } progress.stop { accent-color:#ff6b6b }
</style></head><body>
<img id="preview" src="/stream.mjpeg" alt="Switch capture">
<div id="bar">
  <span id="ws" class="off">input: disconnected</span>
  <span id="pad">pad: press any gamepad button</span>
  <span id="gamepad-telemetry">L3 up · R3 up · override neutral</span>
  <span id="seq"></span>
</div>
<section id="ops">
 <div class="card"><div class="label">Session</div><div class="value" id="session">human · idle</div><div class="value"><select id="mode"><option value="human">Human</option><option value="pure_ai" disabled>Pure AI</option><option value="hybrid" disabled>Hybrid</option></select> <button id="record">Start episode</button></div><div class="value"><button id="arm">Arm AI</button> <button id="disarm" disabled>Disarm</button> <button id="eject">Emergency eject</button></div><div class="value"><label>Muted dimensions <input id="mute" size="12" placeholder="e.g. 0,4,5"></label> <button id="mute-set" disabled>Apply mute</button></div><div class="value label">switch_packets.v1/mute.v1 · AI proposal pre-arbitration</div></div>
 <div class="card"><div class="label">Cradle storage / spool</div><progress id="local-disk" max="1" value="0"></progress><div class="value" id="spool">loading…</div><div class="value" id="local-rate"></div><div class="value" id="segment-backlog"></div></div>
 <div class="card"><div class="label">Cluster</div><progress id="cluster-disk" max="1" value="0"></progress><div class="value" id="cluster">loading…</div><div class="value" id="cluster-storage"></div></div>
 <div class="card"><div class="label">Models / training</div><div class="value" id="model">loading…</div><div class="value" id="readiness">local model: unloaded · unarmed</div><div class="value" id="inference">inference: unloaded · unarmed</div><select id="model-select"><option value="">No validated revisions</option></select> <button id="model-load" disabled>Load verified revision</button><div class="value" id="jobs"></div></div>
</section>
<script>
  const HZ=60, DEADZONE=0.15, DIM=26;
  const MAP={10:4,11:5,12:6,14:7,15:8,13:9,4:10,6:11,5:12,7:13,9:18,8:19,16:20,2:22,3:23,0:24,1:25};
  let ws=null, seq=0, timer=null, controlsBusy=false;
  const $=id=>document.getElementById(id);
  const bytes=n=>n==null?'?':(n/1e9).toFixed(2)+' GB', rate=n=>n==null?'?':(n/1e6).toFixed(1)+' MB/s';
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
    const v=vec(g),l3=!!(g.buttons[10]&&g.buttons[10].pressed),r3=!!(g.buttons[11]&&g.buttons[11].pressed),active=v.some((x,i)=>i<4?Math.abs(x)>DEADZONE:x>0.5);
    $('gamepad-telemetry').textContent=`L3 ${l3?'DOWN':'up'} · R3 ${r3?'DOWN':'up'} · ${active?'takeover trigger READY':'override neutral'}`;
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
      const a=s.action_plane||{}; const r=s.recording||{}; $('session').textContent=`${a.mode||'human'} · ${a.armed?'armed':'unarmed'} · ${r.state||'idle'} · ${r.frames||0} frames · ${(r.duration_seconds||0).toFixed(1)}s`;
      $('gamepad-telemetry').className=a.takeover?'on':''; if(a.mode==='hybrid') $('gamepad-telemetry').textContent+=a.takeover?` · HUMAN OVERRIDE (${a.takeover_reason||'activity'})${a.takeover_release_remaining_ms?` · release in ${a.takeover_release_remaining_ms.toFixed(0)}ms`:''}`:' · AI authority';
      $('record').textContent=['recording','stopping'].includes(r.state)?'Stop episode':'Start episode';
      $('mode').value=a.mode||'human'; [...$('mode').options].forEach(o=>o.disabled=controlsBusy||(o.value!=='human'&&!a.armed)); $('arm').disabled=controlsBusy||!!a.armed; $('disarm').disabled=controlsBusy||!a.armed; $('mute-set').disabled=controlsBusy||!a.armed; $('record').disabled=controlsBusy;
      const p=s.spool,ls=p&&p.local_storage,vol=ls&&ls.capture_filesystem; $('spool').textContent=p&&ls&&ls.state==='ready'?`${bytes(vol.available_bytes)} available / ${bytes(vol.total_bytes)} · ${bytes(ls.source_buffered_bytes)} source · ${bytes(ls.staged_bytes)} staged · ${p.pending_episodes||0} pending · ${p.receipt_state||'unknown'}${ls.recording_blocked_reason?' · BLOCKED: '+ls.recording_blocked_reason:ls.warning?' · low-space warning':''}`:`storage unavailable: ${(ls&&ls.error)||(s.errors&&s.errors.spool)||'unknown'}`;
      $('local-disk').value=vol?vol.used_fraction:0;$('local-disk').className=ls&&!ls.recording_admission_open?'stop':ls&&ls.warning?'warn':'';$('local-rate').textContent=ls&&ls.state==='ready'?`write ${rate(ls.capture_write_rate_bytes_per_second)} · upload ${ls.upload_rate_bytes_per_second==null?'unavailable':rate(ls.upload_rate_bytes_per_second)} · receipts ${rate(ls.receipt_rate_bytes_per_second)} · remaining ${ls.estimated_recording_seconds_remaining==null?'measuring':(ls.estimated_recording_seconds_remaining/60).toFixed(1)+' min'} · ${bytes(ls.receipted_bytes)} receipted`:'';
      const seg=p&&p.rolling_segments;$('segment-backlog').textContent=seg?`segments: ${seg.state}${seg.segment_backlog_count==null?'':` · ${seg.segment_backlog_count} queued · ${bytes(seg.segment_backlog_bytes)}`}${seg.backpressure?' · BACKPRESSURE':''}${seg.blocked_reason?' · '+seg.blocked_reason:''}`:'segments: unavailable';
      const c=s.cluster,cs=c&&c.storage; $('cluster').textContent=c?`${(c.datasets.datasets||[]).length} datasets · ${(c.snapshots.snapshots||[]).length} snapshots`:`unavailable: ${s.errors.cluster||'unknown'}`;$('cluster-storage').textContent=cs&&cs.state==='ready'?`${bytes(cs.available_bytes)} available / ${bytes(cs.total_bytes)}`:`storage unavailable${cs&&cs.reason?' · '+cs.reason:''}`;$('cluster-disk').value=cs&&cs.used_fraction||0;$('cluster-disk').className=cs&&cs.warning?'warn':'';
      const d=c&&c.deployment; $('model').textContent=d&&d.active_revision?`active ${d.active_revision.slice(0,8)} · gen ${d.generation}`:'none active';
      const m=s.model_readiness||{}; $('readiness').textContent=`model readiness: ${m.phase||'unloaded'} · ${m.armed?'armed':'unarmed'}${m.active?' · revision '+m.active.slice(0,8):''}${m.checkpoint_sha256?' · sha '+m.checkpoint_sha256.slice(0,8):''}${m.warmup_frames!=null&&m.sequence_length?' · warmup '+m.warmup_frames+'/'+m.sequence_length:''}${m.blocked_reason?' · '+m.blocked_reason:''}${m.error?' · '+m.error:''}`;
      const i=s.inference||{}; $('inference').textContent=`inference: ${i.health||'unloaded'} · ${i.freshness_state||'unavailable'} · ${i.armed?'armed':'unarmed'}${i.revision?' · '+i.revision.slice(0,8):''}${i.processing_latency_ms!=null?' · cluster '+i.processing_latency_ms.toFixed(1)+'ms':''}${i.transport_latency_ms!=null?' · RTT '+i.transport_latency_ms.toFixed(1)+'ms':''}${i.proposal_age_ms!=null?' · proposal '+i.proposal_age_ms.toFixed(0)+'ms old':''}${i.proposal_sequence!=null?' · seq '+i.proposal_sequence:''}${i.gap_duration_ms?' · gap '+i.gap_duration_ms.toFixed(0)+'ms':''}${i.gap_reason?' · '+i.gap_reason:''}${i.error?' · '+i.error:''}`;
      const revisions=c&&c.revisions&&c.revisions.revisions||[]; const validated=revisions.filter(r=>['validated','active'].includes(r.state)); $('model-select').innerHTML=validated.length?'<option value="">Select validated revision</option>'+validated.map(r=>`<option value="${r.revision_id}">${r.model_id} · ${r.revision_id.slice(0,8)} · ${r.state}</option>`).join(''):'<option value="">No validated revisions</option>'; $('model-load').disabled=!validated.length||m.loading||!m.load_available;
      const jobs=c&&c.jobs&&c.jobs.jobs||[]; $('jobs').textContent=jobs.length?jobs.map(j=>`${j.state} ${j.job_id.slice(0,8)}`).join(' · '):'no training jobs';
    } catch(e) { $('cluster').textContent='operations status unavailable'; }
  }
  pollOps(); setInterval(pollOps,5000);
  $('record').onclick=async()=>{const stop=$('record').textContent.startsWith('Stop');controlsBusy=true;await pollOps();try{await fetch(stop?'/api/recording/stop':'/api/recording/start',{method:'POST'});await pollOps()}finally{controlsBusy=false;await pollOps()}};
  $('model-load').onclick=async()=>{const revision=$('model-select').value;if(!revision)return;$('model-load').disabled=true;try{const response=await fetch('/api/models/load/'+encodeURIComponent(revision),{method:'POST'});if(!response.ok)throw new Error((await response.json()).detail||'load rejected');await pollOps()}catch(e){$('readiness').textContent='local model: rejected · '+e.message}finally{setTimeout(pollOps,500)}};
  async function control(path,body){controlsBusy=true;await pollOps();try{const response=await fetch(path,{method:'POST',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):null});if(!response.ok)throw new Error((await response.json()).detail||'control rejected');await pollOps()}finally{controlsBusy=false;await pollOps()}}
  $('arm').onclick=()=>control('/api/control/arm').catch(e=>alert(e.message));
  $('disarm').onclick=()=>control('/api/control/disarm').catch(e=>alert(e.message));
  $('eject').onclick=()=>control('/api/control/eject').catch(e=>alert(e.message));
  $('mode').onchange=()=>control('/api/control/mode/'+$('mode').value).catch(e=>{alert(e.message);pollOps()});
  $('mute-set').onclick=()=>{const mask=new Array(DIM).fill(false);for(const raw of $('mute').value.split(',')){if(!raw.trim())continue;const i=Number(raw);if(!Number.isInteger(i)||i<0||i>=DIM){alert('Mute dimensions must be 0-25');return}mask[i]=true}control('/api/control/mute',mask).catch(e=>alert(e.message))};
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

    def health(self) -> dict:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            payload = json.loads(response.read())
            if response.status != 200 or not isinstance(payload, dict):
                raise RuntimeError("orchestrator health unavailable")
            return payload
        finally:
            connection.close()


def disabled_model_readiness() -> dict:
    """Explicit fail-closed contract until local vs cluster inference is selected."""
    return {
        "schema_version": "nxml.dagger-model-readiness.v1",
        "phase": "disabled",
        "armed": False,
        "ready": False,
        "load_available": False,
        "active": None,
        "previous": None,
        "loading": None,
        "blocked_reason": "inference runtime boundary is not configured",
    }


def remote_model_readiness(inference: RemoteInferenceWorker) -> dict:
    status = inference.status()
    return {
        "schema_version": "nxml.dagger-model-readiness.v1",
        "phase": "ready" if status["ready"] else status["health"],
        "armed": False,
        "ready": status["ready"],
        "load_available": True,
        "active": status["revision"] if status["enabled"] else None,
        "previous": None,
        "loading": status["revision"] if status["enabled"] and not status["ready"] else None,
        "checkpoint_sha256": status["checkpoint_sha256"],
        "sequence_length": status["sequence_length"],
        "warmup_frames": status["warmup_frames"],
        "blocked_reason": None if status["ready"] else status["error"],
    }


def create_app(
    orchestrator: OrchestratorClient,
    capture_device: str,
    operations: OperationsReader | None = None,
    capture_source: MjpegFanoutSource | None = None,
    recorder: HumanRecordingSession | None = None,
    inference=None,
    model_runtime: AtomicModelRuntime | None = None,
    remote_inference: RemoteInferenceWorker | None = None,
    action_plane: ActionPlane | None = None,
    segment_worker: SegmentDeliveryWorker | None = None,
) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        if operations is not None:
            operations.start()
        if segment_worker is not None:
            segment_worker.start()
        if capture_source is not None:
            capture_source.start()
        if inference is not None:
            inference.start()
        if action_plane is not None:
            action_plane.start()
        try:
            yield
        finally:
            if operations is not None:
                operations.stop()
            if segment_worker is not None:
                segment_worker.stop()
            if action_plane is not None:
                action_plane.stop()
            if inference is not None:
                inference.stop()
            if capture_source is not None:
                capture_source.stop()

    app = FastAPI(title="nxml-minui", lifespan=lifespan)
    stream_state: dict[str, subprocess.Popen | None] = {"proc": None}
    stream_lock = threading.Lock()
    recording_transition_lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(PAGE)

    @app.get("/api/ops/status")
    def ops_status() -> dict:
        if operations is None:
            return {
                "schema_version": "nxml.dagger-operations-status.v1",
                "observed_at": time.time(),
                "session": {
                    "schema_version": "nxml.dagger-session-status.v1",
                    "mode": "human",
                    "recording_state": "idle",
                    "recording_available": False,
                    "recording_blocked_reason": "Operations polling is not configured",
                },
                "spool": None,
                "cluster": None,
                "inference": (
                    inference.status() if inference is not None else InferenceStatus().wire()
                ),
                "model_readiness": (
                    model_runtime.state().wire()
                    if model_runtime is not None
                    else remote_model_readiness(remote_inference)
                    if remote_inference is not None
                    else disabled_model_readiness()
                ),
                "errors": {"operations": "not configured"},
            }
        wire = operations.snapshot().wire()
        wire["recording"] = recorder.status() if recorder is not None else {"state": "unavailable"}
        wire["inference"] = (
            inference.status() if inference is not None else InferenceStatus().wire()
        )
        wire["model_readiness"] = (
            model_runtime.state().wire()
            if model_runtime is not None
            else remote_model_readiness(remote_inference)
            if remote_inference is not None
            else disabled_model_readiness()
        )
        wire["action_plane"] = action_plane.status() if action_plane else None
        if segment_worker is not None and wire.get("spool") is not None:
            wire["spool"]["rolling_segments"] = segment_worker.status()
        return wire

    def recording_boundary(operation, *, kind="configuration_changed", payload=None):
        with recording_transition_lock:
            active = recorder is not None and recorder.status()["state"] in {
                "recording",
                "stopping",
            }
            if active:
                recorder.append_boundary(kind, payload)
                recorder.stop()
            result = operation()
            if active:
                recorder.start()
            return result

    def arm_readiness(*, wait: bool) -> dict:
        if action_plane is None or remote_inference is None or operations is None:
            raise HTTPException(503, "remote action plane is not configured")
        status = (
            remote_inference.wait_until_fresh(timeout=2.0) if wait else remote_inference.status()
        )
        if status is None:
            raise HTTPException(409, "inference did not produce a fresh proposal within 2 seconds")
        deployment = (operations.snapshot().cluster or {}).get("deployment") or {}
        expected = remote_inference.client.revision
        if deployment.get("active_revision") != expected["revision_id"]:
            raise HTTPException(409, "control-plane active revision differs from selected revision")
        if status.get("checkpoint_sha256") != expected["checkpoint_sha256"]:
            raise HTTPException(409, "inference digest differs from selected revision")
        if not status.get("ready") or status.get("health") != "healthy":
            raise HTTPException(409, "inference is not healthy and warm")
        if status.get("warmup_frames", 0) < status.get("sequence_length", 1) - 1:
            raise HTTPException(409, "inference warmup is incomplete")
        age = status.get("proposal_age_ms")
        if age is None or age > status.get("hold_horizon_ms", 55):
            raise HTTPException(409, "policy proposal stream exceeded its hold horizon")
        health = orchestrator.health()
        if health.get("switch_state") != "connected":
            raise HTTPException(409, "Switch is not connected")
        return {
            "schema_version": "nxml.dagger-arm-readiness.v1",
            "ready": True,
            "revision": expected["revision_id"],
            "checkpoint_sha256": expected["checkpoint_sha256"],
            "proposal_age_ms": age,
            "warmup_frames": status["warmup_frames"],
            "sequence_length": status["sequence_length"],
            "switch_state": health["switch_state"],
        }

    @app.get("/api/control/readiness")
    def control_readiness() -> dict:
        return arm_readiness(wait=True)

    @app.post("/api/control/arm")
    def control_arm() -> dict:
        arm_readiness(wait=True)
        recording_boundary(action_plane.arm, kind="armed")
        return action_plane.status()

    @app.post("/api/control/disarm")
    def control_disarm() -> dict:
        if action_plane is None:
            raise HTTPException(503, "action plane is not configured")
        recording_boundary(action_plane.disarm, kind="disarmed")
        return action_plane.status()

    @app.post("/api/control/eject")
    def control_eject() -> dict:
        if action_plane is None:
            raise HTTPException(503, "action plane is not configured")
        recording_boundary(action_plane.eject, kind="emergency_eject")
        return action_plane.status()

    @app.post("/api/control/mode/{mode}")
    def control_mode(mode: str) -> dict:
        if action_plane is None:
            raise HTTPException(503, "action plane is not configured")
        try:
            selected = Mode(mode)
            recording_boundary(
                lambda: action_plane.set_mode(selected),
                kind="mode_changed",
                payload={"mode": selected.value},
            )
        except ValueError as error:
            raise HTTPException(400, "invalid DAgger mode") from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        return action_plane.status()

    @app.post("/api/control/mute")
    def control_mute(mask: list[bool]) -> dict:
        if action_plane is None:
            raise HTTPException(503, "action plane is not configured")
        try:
            mute = MuteMask(tuple(mask))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        recording_boundary(
            lambda: action_plane.set_mute(mute),
            kind="mute_changed",
            payload={"version": mute.version, "mask": list(mute.values)},
        )
        return action_plane.status()

    @app.post("/api/models/load/{revision_id}", status_code=202)
    def model_load(revision_id: str) -> dict:
        if operations is None:
            raise HTTPException(503, "operations catalog is not configured")
        cluster = operations.snapshot().cluster or {}
        revisions = (cluster.get("revisions") or {}).get("revisions") or []
        revision = next(
            (item for item in revisions if item.get("revision_id") == revision_id), None
        )
        if revision is None:
            raise HTTPException(404, "revision is not present in the polled cluster catalog")
        if revision.get("state") not in {"validated", "active"}:
            raise HTTPException(409, "revision has not passed authoritative cluster validation")
        if remote_inference is not None:
            configured = remote_inference.client.revision
            if (
                revision_id != configured["revision_id"]
                or revision.get("checkpoint_sha256") != configured["checkpoint_sha256"]
            ):
                raise HTTPException(409, "revision does not match configured inference endpoint")
            recording_boundary(
                remote_inference.enable,
                kind="model_changed",
                payload={
                    "revision": revision_id,
                    "checkpoint_sha256": revision.get("checkpoint_sha256"),
                },
            )
            return remote_model_readiness(remote_inference)
        if model_runtime is None:
            raise HTTPException(503, "model runtime boundary is not configured")
        try:
            model_runtime.load_async(revision)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(409, str(error)) from error
        return model_runtime.state().wire()

    @app.post("/api/recording/start")
    def recording_start() -> dict:
        if recorder is None:
            raise HTTPException(503, "recording is not configured")
        if operations is not None:
            spool = operations.snapshot().spool or {}
            if spool.get("admission_open") is False:
                reason = spool.get("blocked_reason") or "local spool admission is closed"
                raise HTTPException(409, str(reason))
        try:
            with recording_transition_lock:
                return recorder.start()
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/recording/stop")
    def recording_stop() -> dict:
        if recorder is None:
            raise HTTPException(503, "recording is not configured")
        try:
            with recording_transition_lock:
                return recorder.stop()
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

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
                target = action_plane.submit_human if action_plane else orchestrator.post_action
                await asyncio.to_thread(target, [float(v) for v in vector])
        except WebSocketDisconnect:
            pass
        finally:
            # Synchronous on purpose: must run even under task cancellation.
            with contextlib.suppress(Exception):
                if action_plane is not None:
                    action_plane.disarm("browser_disconnect")
                else:
                    orchestrator.post_action([0.0] * ACTION_DIM)

    @app.get("/stream.mjpeg")
    def stream() -> StreamingResponse:
        if capture_source is not None:

            def shared_frames():
                sequence = -1
                while True:
                    item = capture_source.latest_mjpeg(after_sequence=sequence)
                    if item is None:
                        continue
                    sequence = item.sequence
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(item.jpeg)).encode()
                        + b"\r\n\r\n"
                        + item.jpeg
                        + b"\r\n"
                    )

            return StreamingResponse(
                shared_frames(),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-store, private"},
            )

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
    parser.add_argument(
        "--spool-status",
        type=Path,
        default=Path("~/.local/state/nxml-spool/status.json").expanduser(),
    )
    parser.add_argument("--cluster-url", default="http://100.80.98.4:8787")
    parser.add_argument(
        "--cluster-storage-path",
        default="/v1/datasets/nxml-pokemon-za-v2/segment-status",
        help="published cluster storage telemetry path; omitted means explicitly unavailable",
    )
    parser.add_argument(
        "--cluster-token", type=Path, default=Path("~/.config/nxml/cluster.token").expanduser()
    )
    parser.add_argument(
        "--capture-output", type=Path, default=Path("~/captures/pokemon-za").expanduser()
    )
    parser.add_argument("--inference-endpoint")
    parser.add_argument("--inference-revision")
    parser.add_argument("--inference-digest")
    parser.add_argument("--inference-timeout-ms", type=int, default=100)
    parser.add_argument("--rolling-segments", action="store_true")
    parser.add_argument("--segment-duration-seconds", type=float, default=30.0)
    parser.add_argument("--segment-max-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--segment-max-pending", type=int, default=3)
    args = parser.parse_args()
    try:
        validate_tailnet_bind(args.host)
    except ValueError as error:
        parser.error(str(error))
    client = OrchestratorClient(args.orchestrator_host, args.orchestrator_port)
    rolling_root = Path("~/.local/state/nxml-segments").expanduser()
    capture_output = rolling_root / "source" if args.rolling_segments else args.capture_output
    operations = OperationsReader(
        spool_status_path=args.spool_status,
        cluster_url=args.cluster_url,
        cluster_token_path=args.cluster_token,
        capture_dir=capture_output,
        cluster_storage_path=args.cluster_storage_path,
    )
    fanout = MjpegFanoutSource(args.capture)
    remote_inference = None
    if any((args.inference_endpoint, args.inference_revision, args.inference_digest)):
        if not all((args.inference_endpoint, args.inference_revision, args.inference_digest)):
            parser.error("inference endpoint, revision, and digest must be configured together")
        token = args.cluster_token.read_text().strip()
        request = urllib.request.Request(
            f"{args.cluster_url.rstrip('/')}/v1/models/revisions/{args.inference_revision}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            revision = json.load(response)
        if revision.get("checkpoint_sha256") != args.inference_digest:
            parser.error("configured inference digest differs from validated revision")
        inference_client = InferenceV2Client(
            args.inference_endpoint, revision, timeout_ms=args.inference_timeout_ms
        )
        remote_inference = RemoteInferenceWorker(
            source=fanout,
            client=inference_client,
            fresh_ns=33_000_000,
            stale_ns=55_000_000,
        )
    action_plane = ActionPlane(client, remote_inference)
    if remote_inference is not None:
        remote_inference.on_disarm = action_plane.inference_failure
    segment_worker = None
    if args.rolling_segments:
        token = args.cluster_token.read_text().strip()
        segment_worker = SegmentDeliveryWorker(
            SegmentClient(args.cluster_url, token, "nxml-pokemon-za-v2"),
            SegmentJournal(rolling_root / "journal.json"),
            staging_dir=rolling_root / "staging",
            max_pending=args.segment_max_pending,
        )
    recorder = HumanRecordingSession(
        fanout,
        output_dir=capture_output,
        history=action_plane.history,
        state_provider=action_plane.recording_state,
        segment_worker=segment_worker,
        segment_staging_dir=rolling_root / "staging" if segment_worker else None,
        segment_duration_seconds=args.segment_duration_seconds,
        segment_max_bytes=args.segment_max_bytes,
    )
    uvicorn.run(
        create_app(
            client,
            args.capture,
            operations,
            fanout,
            recorder,
            inference=remote_inference,
            remote_inference=remote_inference,
            action_plane=action_plane,
            segment_worker=segment_worker,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
