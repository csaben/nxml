// nxwm-ui — gamepad-driven world-model viewport.
//
// The 26-dim action layout is fixed by nx_packets.action_spec:
//   indices 0..3   — L stick X/Y, R stick X/Y, in [-1, 1]
//   indices 4..25  — 22 buttons in [0, 1] (binary at this layer)
//
// Browser Gamepad API mapping ("Standard Gamepad") → 26-dim action:
//   axes[0..3]                            → action[0..3]   (LX, LY, RX, RY)
//   button 0  (bottom face physically)    → BUTTON_INDEX["B"]   (Switch B is bottom)
//   button 1  (right face physically)     → BUTTON_INDEX["A"]   (Switch A is right)
//   button 2  (left face physically)      → BUTTON_INDEX["Y"]
//   button 3  (top face physically)       → BUTTON_INDEX["X"]
//   button 4  (L1)                        → BUTTON_INDEX["L"]
//   button 5  (R1)                        → BUTTON_INDEX["R"]
//   button 6  (L2)                        → BUTTON_INDEX["ZL"]
//   button 7  (R2)                        → BUTTON_INDEX["ZR"]
//   button 8  (back / -)                  → BUTTON_INDEX["MINUS"]
//   button 9  (start / +)                 → BUTTON_INDEX["PLUS"]
//   button 10 (L3)                        → BUTTON_INDEX["L_STICK_PRESSED"]
//   button 11 (R3)                        → BUTTON_INDEX["R_STICK_PRESSED"]
//   button 12 (dpad up)                   → BUTTON_INDEX["DPAD_UP"]
//   button 13 (dpad down)                 → BUTTON_INDEX["DPAD_DOWN"]
//   button 14 (dpad left)                 → BUTTON_INDEX["DPAD_LEFT"]
//   button 15 (dpad right)                → BUTTON_INDEX["DPAD_RIGHT"]
//   button 16 (home)                      → BUTTON_INDEX["HOME"]
//
// We map by *physical position* so a Switch Pro and an Xbox controller produce the
// same world-model input — the model was trained on Switch hardware where A is the
// right face button.

const BUTTON_INDEX = {
  L_STICK_PRESSED: 4,  R_STICK_PRESSED: 5,
  DPAD_UP: 6, DPAD_LEFT: 7, DPAD_RIGHT: 8, DPAD_DOWN: 9,
  L: 10, ZL: 11, R: 12, ZR: 13,
  JCL_SR: 14, JCL_SL: 15, JCR_SR: 16, JCR_SL: 17,
  PLUS: 18, MINUS: 19, HOME: 20, CAPTURE: 21,
  Y: 22, X: 23, B: 24, A: 25,
};

// Browser gamepad button index → action vector index.
const GAMEPAD_BUTTON_MAP = {
  0:  BUTTON_INDEX.B,                 // bottom face
  1:  BUTTON_INDEX.A,                 // right face
  2:  BUTTON_INDEX.Y,                 // left face
  3:  BUTTON_INDEX.X,                 // top face
  4:  BUTTON_INDEX.L,                 // L1
  5:  BUTTON_INDEX.R,                 // R1
  6:  BUTTON_INDEX.ZL,                // L2
  7:  BUTTON_INDEX.ZR,                // R2
  8:  BUTTON_INDEX.MINUS,             // back / -
  9:  BUTTON_INDEX.PLUS,              // start / +
  10: BUTTON_INDEX.L_STICK_PRESSED,   // L3
  11: BUTTON_INDEX.R_STICK_PRESSED,   // R3
  12: BUTTON_INDEX.DPAD_UP,
  13: BUTTON_INDEX.DPAD_DOWN,
  14: BUTTON_INDEX.DPAD_LEFT,
  15: BUTTON_INDEX.DPAD_RIGHT,
  16: BUTTON_INDEX.HOME,
};

const STICK_DEADZONE = 0.05;
const action = new Float32Array(26);
let activeGamepadIndex = null;
let prevPlusPressed = false;          // edge-detect Plus to toggle play
let playInterval = null;
let detectorDebugInterval = null;
let frameUrl = null;
let inFlightStep = false;

const $ = (id) => document.getElementById(id);
const status = (msg, kind) => {
  const el = $("status");
  el.textContent = msg;
  el.className = "status" + (kind ? " " + kind : "");
};

// ---------- gamepad polling ----------
function pollGamepad() {
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  let active = null;
  for (const p of pads) {
    if (p && p.connected) { active = p; break; }
  }
  if (!active) {
    activeGamepadIndex = null;
    const gs = $("gamepad-status");
    gs.textContent = "no gamepad — press any button on a connected controller";
    gs.className = "gamepad-status";
  } else {
    activeGamepadIndex = active.index;
    const gs = $("gamepad-status");
    gs.textContent = `gamepad: ${active.id} (mapping: ${active.mapping || "non-standard"})`;
    gs.className = "gamepad-status connected";

    // Sticks
    const invertY = $("invert-y").checked ? -1 : 1;
    action[0] = applyDeadzone(active.axes[0] || 0);
    action[1] = invertY * applyDeadzone(active.axes[1] || 0);
    action[2] = applyDeadzone(active.axes[2] || 0);
    action[3] = invertY * applyDeadzone(active.axes[3] || 0);

    // Buttons — clear all first, then set from gamepad
    for (let i = 4; i < 26; i++) action[i] = 0;
    for (let i = 0; i < active.buttons.length; i++) {
      const dst = GAMEPAD_BUTTON_MAP[i];
      if (dst === undefined) continue;
      const pressed = typeof active.buttons[i] === "object" ? active.buttons[i].pressed : !!active.buttons[i];
      if (pressed) action[dst] = 1.0;
    }

    // Plus button toggles play (edge-triggered)
    const plusNow = !!(active.buttons[9] && (typeof active.buttons[9] === "object" ? active.buttons[9].pressed : active.buttons[9]));
    if (plusNow && !prevPlusPressed) {
      togglePlay();
    }
    prevPlusPressed = plusNow;
  }
  requestAnimationFrame(pollGamepad);
}

function applyDeadzone(v) {
  return Math.abs(v) < STICK_DEADZONE ? 0 : v;
}

window.addEventListener("gamepadconnected", (ev) => {
  status(`gamepad connected: ${ev.gamepad.id}`, "ok");
});
window.addEventListener("gamepaddisconnected", () => {
  status("gamepad disconnected", "error");
});

// ---------- API ----------
async function apiInfo() {
  const r = await fetch("/api/info");
  if (!r.ok) throw new Error("info: " + r.status);
  return r.json();
}

async function apiReseed(file, startFrame) {
  const r = await fetch("/api/reseed", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ file, start_frame: startFrame }),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(detail.detail || r.statusText);
  }
}

async function apiReload(modelPath) {
  const r = await fetch("/api/reload", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model_path: modelPath }),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(detail.detail || r.statusText);
  }
}

// ---------- detector ----------
// State held client-side so slider edits and live telemetry can feed the
// same readout. `schema` comes from the server on init; `params` is the
// authoritative current value (we update it after every successful POST so
// the UI can reflect e.g. clamped values).
const detectorState = { name: null, schema: null, params: null, debugFresh: false };

function fmtNum(v, schemaEntry) {
  if (schemaEntry && schemaEntry.type === "int") return String(Math.round(v));
  return Number(v).toFixed(2);
}

async function fetchDetectorConfig() {
  const r = await fetch("/api/detector/config");
  if (!r.ok) return null;
  const cfg = await r.json();
  if (!cfg || !cfg.name) return null;
  detectorState.name = cfg.name;
  detectorState.schema = cfg.schema || {};
  detectorState.params = { ...(cfg.params || {}) };
  return cfg;
}

function buildDetectorPanel() {
  if (!detectorState.name) return;
  $("detector-panel").hidden = false;
  $("detector-name").textContent = detectorState.name;
  const grid = $("detector-sliders");
  grid.innerHTML = "";
  for (const [key, spec] of Object.entries(detectorState.schema)) {
    const cur = detectorState.params[key];
    const label = document.createElement("label");
    label.htmlFor = `det-${key}`;
    label.textContent = spec.label || key;
    const input = document.createElement("input");
    input.type = "range";
    input.id = `det-${key}`;
    input.min = spec.min;
    input.max = spec.max;
    input.step = spec.step;
    input.value = cur;
    const value = document.createElement("span");
    value.className = "value-cell";
    value.id = `det-val-${key}`;
    value.textContent = fmtNum(cur, spec);
    input.addEventListener("input", () => {
      // Visual feedback during drag, persist after debounce.
      value.textContent = fmtNum(input.value, spec);
      scheduleDetectorApply(key, parseFloat(input.value));
    });
    grid.appendChild(label);
    grid.appendChild(input);
    grid.appendChild(value);
  }
}

let detectorApplyTimer = null;
const detectorPendingPatch = {};
function scheduleDetectorApply(key, val) {
  detectorPendingPatch[key] = val;
  if (detectorApplyTimer) clearTimeout(detectorApplyTimer);
  detectorApplyTimer = setTimeout(flushDetectorApply, 200);
}

async function flushDetectorApply() {
  detectorApplyTimer = null;
  const patch = { ...detectorPendingPatch };
  for (const k of Object.keys(detectorPendingPatch)) delete detectorPendingPatch[k];
  if (Object.keys(patch).length === 0) return;
  try {
    const r = await fetch("/api/detector/config", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ params: patch }),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(detail.detail || r.statusText);
    }
    const out = await r.json();
    if (out.params) {
      detectorState.params = { ...detectorState.params, ...out.params };
      // Reflect server-side clamping back to the sliders.
      for (const [k, v] of Object.entries(out.params)) {
        const inp = $(`det-${k}`);
        const cell = $(`det-val-${k}`);
        if (inp && Math.abs(parseFloat(inp.value) - v) > 1e-9) inp.value = v;
        if (cell) cell.textContent = fmtNum(v, detectorState.schema[k]);
      }
    }
    if (out.state) renderDetectorState(out.state, /*signals*/ null);
  } catch (e) {
    status("detector: " + e.message, "error");
  }
}

function renderDetectorState(state, signals) {
  const lamp = $("detector-lamp");
  lamp.dataset.detected = state.detected ? "true" : "false";
  const parts = [];
  if (signals) {
    for (const [k, v] of Object.entries(signals)) {
      parts.push(`${k}: ${typeof v === "number" ? v.toFixed(3) : v}`);
    }
  }
  parts.push(`streak: ${state.streak ?? 0}`);
  parts.push(state.detected ? "DETECTED" : "—");
  $("detector-readout-line").textContent = parts.join(" · ");
}

function applyTelemetry(tel) {
  if (!tel || !tel.state) return;
  renderDetectorState(tel.state, tel.signals || null);
  if (tel.params) detectorState.params = { ...detectorState.params, ...tel.params };
  detectorState.debugFresh = true;
}

async function refreshDetectorDebug() {
  if (!detectorState.name || !detectorState.debugFresh) return;
  detectorState.debugFresh = false;
  try {
    const r = await fetch(`/api/detector/debug.png?ts=${Date.now()}`);
    if (!r.ok) return;
    const blob = await r.blob();
    const img = $("detector-debug");
    if (img.src) URL.revokeObjectURL(img.src);
    img.src = URL.createObjectURL(blob);
  } catch (_) { /* ignore */ }
}

function copyDetectorYaml() {
  if (!detectorState.params) return;
  // Match the existing pokemon_za.config.TargetUIRewardConfig field names so
  // the user can paste straight into a training config.
  const lines = ["target_ui_detection:"];
  const map = {
    threshold: "score_threshold",
    sat_threshold: "sat_threshold",
    min_consecutive_hits: "min_consecutive_hits",
  };
  for (const [k, v] of Object.entries(detectorState.params)) {
    const yamlKey = map[k] || k;
    lines.push(`  ${yamlKey}: ${v}`);
  }
  const text = lines.join("\n") + "\n";
  navigator.clipboard.writeText(text).then(
    () => status("copied yaml", "ok"),
    (e) => status("clipboard: " + e.message, "error"),
  );
}

async function resetDetectorStreak() {
  try {
    await fetch("/api/detector/reset", { method: "POST" });
    renderDetectorState({ detected: false, streak: 0 }, null);
    status("streak reset", "ok");
  } catch (e) {
    status("detector reset: " + e.message, "error");
  }
}

// ---------- core actions ----------
async function doStep() {
  if (inFlightStep) return;  // drop frames if model can't keep up with fps
  inFlightStep = true;
  try {
    const r = await fetch("/api/step", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ action: Array.from(action) }),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(detail.detail || r.statusText);
    }
    const blob = await r.blob();
    if (frameUrl) URL.revokeObjectURL(frameUrl);
    frameUrl = URL.createObjectURL(blob);
    $("frame").src = frameUrl;
    const telHeader = r.headers.get("X-Detector-Telemetry");
    if (telHeader) {
      try { applyTelemetry(JSON.parse(telHeader)); } catch (_) { /* ignore */ }
    }
    status("ok", "ok");
  } catch (e) {
    status(e.message, "error");
    stopPlay();
  } finally {
    inFlightStep = false;
  }
}

function startPlay() {
  if (playInterval) return;
  const fps = Math.max(1, Math.min(60, parseInt($("fps").value, 10) || 10));
  $("play-btn").classList.add("playing");
  $("play-btn").textContent = "■ stop";
  playInterval = setInterval(doStep, Math.round(1000 / fps));
  // Debug image is expensive to encode and only useful while open. Refresh
  // it on a slower clock than the step rate.
  if (!detectorDebugInterval) {
    detectorDebugInterval = setInterval(refreshDetectorDebug, 500);
  }
}

function stopPlay() {
  if (playInterval) clearInterval(playInterval);
  playInterval = null;
  if (detectorDebugInterval) clearInterval(detectorDebugInterval);
  detectorDebugInterval = null;
  $("play-btn").classList.remove("playing");
  $("play-btn").textContent = "▶ play";
}

function togglePlay() { playInterval ? stopPlay() : startPlay(); }

async function refreshInfo() {
  const info = await apiInfo();
  $("info-json").textContent = JSON.stringify(info, null, 2);
  const sel = $("episode-select");
  const prev = sel.value;
  sel.innerHTML = "";
  for (const ep of info.episodes || []) {
    const opt = document.createElement("option");
    opt.value = ep; opt.textContent = ep;
    sel.appendChild(opt);
  }
  if (info.current_episode_file && [...sel.options].some(o => o.value === info.current_episode_file)) {
    sel.value = info.current_episode_file;
  } else if (prev && [...sel.options].some(o => o.value === prev)) {
    sel.value = prev;
  }
  return info;
}

// ---------- fullscreen ----------
function toggleFullscreen() {
  const pane = $("frame-pane");
  const isFs = document.fullscreenElement || document.webkitFullscreenElement;
  if (isFs) {
    (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  } else {
    (pane.requestFullscreen || pane.webkitRequestFullscreen).call(pane);
  }
}

function syncFullscreenIcon() {
  const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
  $("fullscreen-icon-enter").style.display = isFs ? "none" : "";
  $("fullscreen-icon-exit").style.display = isFs ? "" : "none";
}

// ---------- wiring ----------
window.addEventListener("DOMContentLoaded", async () => {
  $("step-btn").addEventListener("click", doStep);
  $("step-btn-empty").addEventListener("click", doStep);
  $("play-btn").addEventListener("click", togglePlay);
  $("fullscreen-btn").addEventListener("click", toggleFullscreen);
  document.addEventListener("fullscreenchange", syncFullscreenIcon);
  document.addEventListener("webkitfullscreenchange", syncFullscreenIcon);

  $("reseed-btn").addEventListener("click", async () => {
    const file = $("episode-select").value;
    const startFrame = parseInt($("reseed-start").value, 10) || 0;
    if (!file) { status("no episode selected", "error"); return; }
    try {
      stopPlay();
      await apiReseed(file, startFrame);
      $("frame").removeAttribute("src");
      await refreshInfo();
      status(`reseeded ${file}@${startFrame}`, "ok");
    } catch (e) { status(e.message, "error"); }
  });

  $("reload-btn").addEventListener("click", async () => {
    const path = $("reload-path").value.trim();
    if (!path) { status("no model path given", "error"); return; }
    try {
      stopPlay();
      await apiReload(path);
      $("frame").removeAttribute("src");
      await refreshInfo();
      status(`reloaded ${path}`, "ok");
    } catch (e) { status(e.message, "error"); }
  });

  // Keyboard shortcuts: space = step, p = play/stop, f = fullscreen.
  window.addEventListener("keydown", (ev) => {
    if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLSelectElement) return;
    if (ev.code === "Space") { ev.preventDefault(); doStep(); }
    if (ev.code === "KeyP")  { ev.preventDefault(); togglePlay(); }
    if (ev.code === "KeyF")  { ev.preventDefault(); toggleFullscreen(); }
  });

  $("detector-reset-btn").addEventListener("click", resetDetectorStreak);
  $("detector-copy-btn").addEventListener("click", copyDetectorYaml);

  try {
    await refreshInfo();
    const cfg = await fetchDetectorConfig();
    if (cfg) buildDetectorPanel();
    status("ready — press play (or +/start on gamepad)", "ok");
  } catch (e) {
    status("info: " + e.message, "error");
  }

  requestAnimationFrame(pollGamepad);
});
