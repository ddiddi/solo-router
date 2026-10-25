// SO-101 browser demo: one task, routed perceive -> reason -> act by the real
// browser router (router.js), then executed by a simulated arm in three.js.
// Every routing decision is real. In "Run in your browser" mode the router is
// told it can reach browser compute, picks the ONNX models in the config, and
// they run for real on the rendered camera frame (see onnx.js). With an API key
// (set in the playground), a hosted planner streams the plan for real.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { complete, enabledProviders, loadKeys } from "./providers.js";

const TASK = "put the red cube in the bowl";
const REQUEST = { instruction: TASK, embodiment: "so101", observations: ["rgb", "wrist_rgb", "proprio"], control_hz: 30 };

// SO-101 geometry in meters (approximate).
const H0 = 0.105; // shoulder axis height
const L1 = 0.116; // shoulder -> elbow
const L2 = 0.135; // elbow -> wrist
const L3 = 0.095; // wrist -> grasp point
const CUBE = 0.025;
const GRIP_OPEN = 0.046;
const GRIP_CLOSED = CUBE + 0.002;

const C = {
  base: 0x060d1a, surface: 0x0d1526, border: 0x1e2a3e, lavender: 0xad94f9,
  purple: 0x8369e4, teal: 0x3c9cc4, servo: 0x121a2b, red: 0xf87171, blue: 0x60a5fa, green: 0x34d399,
};
const JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

class Aborted extends Error {}

function ik(tip) {
  const pan = Math.atan2(-tip.z, tip.x);
  const r = Math.hypot(tip.x, tip.z);
  const wy = tip.y + L3 - H0; // wrist sits L3 above the grasp point (gripper points down)
  const d = clamp(Math.hypot(r, wy), Math.abs(L1 - L2) + 1e-3, L1 + L2 - 1e-4);
  const elbow = -Math.acos(clamp((d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1));
  const lift = Math.atan2(wy, r) - Math.atan2(L2 * Math.sin(elbow), L1 + L2 * Math.cos(elbow));
  return { pan, lift, elbow, wrist: -Math.PI / 2 - lift - elbow };
}

function hudTemplate() {
  return `
  <div class="demo-stage">
    <canvas></canvas>
    <div class="demo-labels"></div>
    <div class="demo-topbar"><span class="demo-live"><i></i><span>SO-101 · simulated</span></span><span class="demo-status">loading…</span></div>
    <div class="demo-controls">
      <button type="button" class="demo-btn estop" data-act="estop">E-STOP</button>
      <button type="button" class="demo-btn" data-act="replay">Replay</button>
      <button type="button" class="demo-btn onnx" data-act="onnx" aria-pressed="false" title="Downloads Florence-2 and Qwen2.5-0.5B as ONNX (about 850 MB on WebGPU, cached after the first run) and runs them on WebGPU, or WASM without it">Run in your browser</button>
    </div>
    <div class="demo-banner" hidden></div>
  </div>
  <aside class="demo-hud">
    <div class="demo-block">
      <small>instruction · so101 · rgb + wrist_rgb · 30 Hz</small>
      <div class="demo-instr"><span class="typed"></span><span class="caret"></span></div>
    </div>
    <ol class="demo-stages">
      ${["perceive", "reason", "act"].map((s, i) => `
      <li data-stage="${s}">
        <span class="demo-dot"></span>
        <div><small>0${i + 1} · ${s}</small><b class="model">—</b><span class="meta"></span></div>
      </li>`).join("")}
    </ol>
    <div class="demo-block">
      <small>plan</small>
      <ol class="demo-plan"></ol>
    </div>
    <div class="demo-block">
      <small>joints</small>
      <div class="demo-joints">${JOINTS.map((j) => `
        <div><span>${j}</span><b data-j="${j}">0°</b><i><em data-jb="${j}"></em></i></div>`).join("")}
      </div>
    </div>
    <p class="demo-note">Simulated arm and scene; every perceive / reason / act decision comes from the router config the server ships with. <b>Run in your browser</b> routes to ONNX models (Florence-2, Qwen2.5-0.5B) that run on your GPU via WebGPU. Add an OpenAI, Claude or Nebius key in the playground and the planner runs for real.</p>
  </aside>`;
}

export function mountDemo(root, router) {
  root.classList.add("demo");
  root.innerHTML = hudTemplate();
  const $ = (s) => root.querySelector(s);
  const stageEl = $(".demo-stage");
  const canvas = $("canvas");

  // ---------------- three.js scene ----------------
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(C.base);
  scene.fog = new THREE.Fog(C.base, 1.2, 2.6);

  const camera = new THREE.PerspectiveCamera(36, 1, 0.01, 10);
  camera.position.set(0.66, 0.5, 0.6);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0.08, 0.1, 0.04);
  controls.enableDamping = true;
  controls.minDistance = 0.3;
  controls.maxDistance = 1.3;
  controls.maxPolarAngle = 1.42;
  controls.enablePan = false;

  scene.add(new THREE.HemisphereLight(0xc9bcff, C.base, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 2.6);
  sun.position.set(0.45, 0.9, 0.35);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -0.5, right: 0.5, top: 0.5, bottom: -0.5, near: 0.1, far: 2 });
  sun.shadow.bias = -0.0005;
  scene.add(sun);
  const rim = new THREE.DirectionalLight(C.teal, 1.6);
  rim.position.set(-0.6, 0.4, -0.5);
  scene.add(rim);

  const table = new THREE.Mesh(new RoundedBoxGeometry(0.9, 0.03, 0.7, 4, 0.01),
    new THREE.MeshStandardMaterial({ color: C.surface, roughness: 0.85 }));
  table.position.set(0.12, -0.015, 0.02);
  table.receiveShadow = true;
  scene.add(table);
  const grid = new THREE.GridHelper(0.8, 32, C.teal, C.border);
  grid.position.set(0.12, 0.0006, 0.02);
  grid.material.transparent = true;
  grid.material.opacity = 0.35;
  scene.add(grid);

  // ---------------- SO-101 ----------------
  const printed = new THREE.MeshStandardMaterial({ color: C.lavender, roughness: 0.5, metalness: 0.05 });
  const servoMat = new THREE.MeshStandardMaterial({ color: C.servo, roughness: 0.4, metalness: 0.3 });
  const hornMat = new THREE.MeshStandardMaterial({ color: C.teal, roughness: 0.3, metalness: 0.6 });
  const part = (w, h, d, mat, r = 0.006) => {
    const m = new THREE.Mesh(new RoundedBoxGeometry(w, h, d, 3, Math.min(r, w / 2, h / 2, d / 2)), mat);
    m.castShadow = true; m.receiveShadow = true;
    return m;
  };
  const horn = () => {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, 0.058, 24), hornMat);
    m.rotation.x = Math.PI / 2;
    return m;
  };

  const arm = { pan: new THREE.Group(), shoulder: new THREE.Group(), elbow: new THREE.Group(), wrist: new THREE.Group(), roll: new THREE.Group() };
  const plate = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.055, 0.012, 40), servoMat);
  plate.position.y = 0.006; plate.receiveShadow = true; plate.castShadow = true;
  scene.add(plate);
  scene.add(arm.pan);
  const baseBody = part(0.07, 0.07, 0.07, printed, 0.012);
  baseBody.position.y = 0.047;
  arm.pan.add(baseBody);
  const panServo = part(0.04, 0.03, 0.05, servoMat, 0.004);
  panServo.position.set(-0.012, 0.085, 0);
  arm.pan.add(panServo);

  arm.shoulder.position.y = H0;
  arm.pan.add(arm.shoulder);
  arm.shoulder.add(part(0.05, 0.04, 0.05, servoMat, 0.005), horn());
  const upper = part(L1, 0.03, 0.042, printed, 0.01);
  upper.position.x = L1 / 2;
  arm.shoulder.add(upper);

  arm.elbow.position.x = L1;
  arm.shoulder.add(arm.elbow);
  arm.elbow.add(part(0.045, 0.036, 0.05, servoMat, 0.005), horn());
  const fore = part(L2, 0.026, 0.036, printed, 0.009);
  fore.position.x = L2 / 2;
  arm.elbow.add(fore);

  arm.wrist.position.x = L2;
  arm.elbow.add(arm.wrist);
  arm.wrist.add(part(0.04, 0.032, 0.046, servoMat, 0.005), horn());
  arm.wrist.add(arm.roll);
  const rollServo = part(0.032, 0.03, 0.036, servoMat, 0.004);
  rollServo.position.x = 0.026;
  arm.roll.add(rollServo);
  const palm = part(0.026, 0.034, 0.056, printed, 0.006);
  palm.position.x = 0.052;
  arm.roll.add(palm);
  const cam = part(0.016, 0.014, 0.024, servoMat, 0.003);
  cam.position.set(0.05, 0.024, 0);
  arm.roll.add(cam);
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.004, 16),
    new THREE.MeshStandardMaterial({ color: C.teal, emissive: C.teal, emissiveIntensity: 1.5 }));
  lens.rotation.z = Math.PI / 2;
  lens.position.set(0.059, 0.024, 0);
  arm.roll.add(lens);
  const jawA = part(0.05, 0.022, 0.006, printed, 0.002);
  const jawB = part(0.05, 0.022, 0.006, printed, 0.002);
  jawA.position.x = jawB.position.x = 0.088;
  arm.roll.add(jawA, jawB);
  const tipAnchor = new THREE.Object3D();
  tipAnchor.position.x = L3;
  arm.roll.add(tipAnchor);

  const pose = { pan: 0, lift: 0, elbow: 0, wrist: 0, roll: 0, grip: GRIP_OPEN };
  const tip = new THREE.Vector3();
  function applyPose() {
    arm.pan.rotation.y = pose.pan;
    arm.shoulder.rotation.z = pose.lift;
    arm.elbow.rotation.z = pose.elbow;
    arm.wrist.rotation.z = pose.wrist;
    arm.roll.rotation.x = pose.roll;
    jawA.position.z = -pose.grip / 2 - 0.003;
    jawB.position.z = pose.grip / 2 + 0.003;
  }
  function setTip(p) {
    tip.copy(p);
    Object.assign(pose, ik(p));
    applyPose();
  }

  // ---------------- objects ----------------
  const cubeMat = (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.45 });
  function makeCube(color, x, z) {
    const m = part(CUBE, CUBE, CUBE, cubeMat(color), 0.003);
    m.userData.home = new THREE.Vector3(x, CUBE / 2, z);
    scene.add(m);
    return m;
  }
  const red = makeCube(C.red, 0.2, 0.065);
  const blue = makeCube(C.blue, 0.17, -0.11);
  const green = makeCube(C.green, 0.255, -0.03);
  const bowlProfile = [[0.0, 0.0], [0.034, 0.0], [0.046, 0.008], [0.054, 0.024], [0.058, 0.04], [0.054, 0.041], [0.05, 0.026], [0.042, 0.012], [0.03, 0.006], [0.0, 0.006]]
    .map(([x, y]) => new THREE.Vector2(x, y));
  const bowl = new THREE.Mesh(new THREE.LatheGeometry(bowlProfile, 48),
    new THREE.MeshStandardMaterial({ color: C.teal, roughness: 0.3, metalness: 0.1, side: THREE.DoubleSide }));
  bowl.position.set(0.035, 0, 0.2);
  bowl.castShadow = bowl.receiveShadow = true;
  scene.add(bowl);

  // perception overlays: wireframe boxes attached to objects + HTML labels
  const labelsEl = $(".demo-labels");
  const detections = [
    [red, "red cube", 0.97, [CUBE, CUBE, CUBE]],
    [bowl, "bowl", 0.94, [0.12, 0.045, 0.12]],
    [green, "green cube", 0.93, [CUBE, CUBE, CUBE]],
    [blue, "blue cube", 0.95, [CUBE, CUBE, CUBE]],
  ].map(([obj, name, conf, size]) => {
    const pad = 1.25;
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(size[0] * pad, size[1] * pad, size[2] * pad)),
      new THREE.LineBasicMaterial({ color: obj === red || obj === bowl ? C.lavender : C.teal, transparent: true, opacity: 0.95 }));
    if (obj === bowl) box.position.y = size[1] / 2;
    box.visible = false;
    obj.add(box);
    const el = document.createElement("div");
    el.className = "demo-label" + (obj === red || obj === bowl ? " target" : "");
    el.innerHTML = `${esc(name)} <em>${conf.toFixed(2)}</em>`;
    labelsEl.appendChild(el);
    return { obj, box, el, height: size[1] * pad + (obj === bowl ? 0.045 : 0.012) };
  });

  // radar sweep for the perception stage
  const sweep = new THREE.Mesh(new THREE.CircleGeometry(0.36, 48, -0.25, 0.5),
    new THREE.MeshBasicMaterial({ color: C.teal, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
  sweep.rotation.x = -Math.PI / 2;
  sweep.position.y = 0.002;
  sweep.visible = false;
  scene.add(sweep);

  // ---------------- clock + tweens (pause when off-screen) ----------------
  let now = 0;
  let gen = 0;
  let tweens = [];
  function tween(ms, update) {
    const g = gen;
    return new Promise((resolve, reject) => {
      tweens.push({ g, start: now, dur: ms / 1000, update, resolve, reject });
    });
  }
  const sleep = (ms) => tween(ms);
  function abortAll() {
    gen++;
    const pending = tweens;
    tweens = [];
    pending.forEach((t) => t.reject(new Aborted()));
  }
  function stepTweens() {
    const live = [];
    for (const t of tweens) {
      const k = t.dur ? clamp((now - t.start) / t.dur, 0, 1) : 1;
      t.update?.(k);
      if (k >= 1) t.resolve(); else live.push(t);
    }
    // Promise continuations run as microtasks after this frame, so nothing is pushed mid-loop.
    tweens = live;
  }

  const moveTip = (to, ms) => {
    const from = tip.clone();
    const target = to.clone();
    return tween(ms, (k) => setTip(from.clone().lerp(target, ease(k))));
  };
  const grip = (to, ms) => {
    const from = pose.grip;
    return tween(ms, (k) => { pose.grip = from + (to - from) * ease(k); applyPose(); });
  };

  // ---------------- HUD ----------------
  const status = (t, cls = "") => { const el = $(".demo-status"); el.textContent = t; el.className = "demo-status " + cls; };
  const stageLi = (s) => root.querySelector(`[data-stage="${s}"]`);
  function setStage(s, route, state, note = "") {
    const li = stageLi(s);
    li.dataset.state = state;
    if (!route) { li.querySelector(".model").textContent = "—"; li.querySelector(".meta").textContent = ""; return; }
    li.querySelector(".model").textContent = route.model;
    li.querySelector(".meta").innerHTML =
      `<span class="loc ${route.location}">${route.location}</span> ~${route.est_latency_ms} ms · ${esc(route.decision)}` +
      (note ? `<br><span class="out">${esc(note)}</span>`
        : route.fallbacks.length ? `<br><span class="fb">fallback → ${esc(route.fallbacks[0])}</span>` : "");
  }
  const planEl = $(".demo-plan");
  const jointEls = Object.fromEntries(JOINTS.map((j) => [j, [root.querySelector(`[data-j="${j}"]`), root.querySelector(`[data-jb="${j}"]`)]]));
  function updateJoints() {
    const deg = (r) => (r * 180) / Math.PI;
    const vals = {
      shoulder_pan: deg(pose.pan), shoulder_lift: deg(pose.lift), elbow_flex: deg(pose.elbow),
      wrist_flex: deg(pose.wrist), wrist_roll: deg(pose.roll), gripper: ((pose.grip - GRIP_CLOSED + 0.002) / (GRIP_OPEN - GRIP_CLOSED + 0.002)) * 100,
    };
    for (const j of JOINTS) {
      const [txt, bar] = jointEls[j];
      const v = vals[j];
      txt.textContent = j === "gripper" ? `${v.toFixed(0)}%` : `${v.toFixed(1)}°`;
      bar.style.width = `${j === "gripper" ? clamp(v, 0, 100) : clamp((Math.abs(v) / 180) * 100, 2, 100)}%`;
    }
  }
  async function type(text) {
    const el = $(".typed");
    el.textContent = "";
    for (const ch of text) { el.textContent += ch; await sleep(32); }
  }

  // ---------------- the task ----------------
  const HOME = new THREE.Vector3(0.14, 0.17, 0.0);
  const OBJECTS = { red_cube: red, blue_cube: blue, green_cube: green, bowl };
  const PHRASES = { red_cube: "red cube", blue_cube: "blue cube", green_cube: "green cube", bowl: "bowl" };
  const SKILLS = ["move_above", "grasp", "lift", "release", "go_home"];
  const DEFAULT_PLAN = [["move_above", "red_cube"], ["grasp", "red_cube"], ["lift", ""], ["move_above", "bowl"], ["release", "bowl"], ["go_home", ""]];
  const detsEl = document.createElement("div");
  detsEl.className = "demo-dets";
  stageEl.insertBefore(detsEl, labelsEl);

  let mode = "sim"; // "sim" | "browser"
  let runtimes = [];
  let models = null; // BrowserModels once loaded
  let gate = Promise.resolve();
  let resume = null;

  const request = () => mode === "browser"
    ? { ...REQUEST, locations: ["browser"], client_runtimes: runtimes }
    : { ...REQUEST, providers: enabledProviders(loadKeys().keys) };
  // Plans from hosted models, keyed by model: the demo loops, so call each provider once.
  const cloudPlans = new Map();
  const isBrowserModel = (route) => mode === "browser" && route.location === "browser" && router.models[route.model]?.onnx;
  // Resolve a real inference call, but drop its result if the run was aborted meanwhile.
  async function guarded(promise) {
    const g = gen;
    const out = await promise;
    if (g !== gen) throw new Aborted();
    return out;
  }

  function reset() {
    abortAll();
    for (const o of [red, blue, green]) {
      scene.attach(o);
      o.position.copy(o.userData.home);
      o.rotation.set(0, Math.atan2(-o.position.z, o.position.x), 0);
    }
    detections.forEach((d) => { d.box.visible = false; d.el.classList.remove("on"); });
    detsEl.innerHTML = "";
    sweep.visible = false;
    pose.grip = GRIP_OPEN;
    pose.roll = 0;
    setTip(HOME);
    ["perceive", "reason", "act"].forEach((s) => setStage(s, null, "idle"));
    planEl.innerHTML = "";
    $(".typed").textContent = "";
    $(".demo-banner").hidden = true;
    root.classList.remove("stopped");
  }

  function showDetection(name) {
    const d = detections.find((x) => x.obj === OBJECTS[name]);
    if (d) { d.box.visible = true; d.el.classList.add("on"); }
  }

  // Render one frame and grab it as the robot's camera image.
  function captureFrame() {
    renderer.render(scene, camera);
    return { url: canvas.toDataURL("image/jpeg", 0.92), cam: camera.clone(), w: canvas.width, h: canvas.height };
  }

  const ray = new THREE.Raycaster();
  function groundPoint(box, frame, height) {
    const cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2;
    ray.setFromCamera(new THREE.Vector2((cx / frame.w) * 2 - 1, -(cy / frame.h) * 2 + 1), frame.cam);
    const hit = new THREE.Vector3();
    return ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), -height), hit) ? hit : null;
  }

  async function perceiveStage(route) {
    const targets = {};
    if (!isBrowserModel(route)) {
      sweep.visible = true;
      await tween(1500, (k) => { sweep.rotation.z = -1.7 + k * 3.4; sweep.material.opacity = 0.22 * Math.sin(k * Math.PI) + 0.04; });
      sweep.visible = false;
      for (const name of ["red_cube", "bowl", "green_cube", "blue_cube"]) {
        showDetection(name);
        targets[name] = OBJECTS[name].position.clone();
        await sleep(220);
      }
      return { targets, note: "" };
    }
    const spec = router.models[route.model];
    const frame = captureFrame();
    const t0 = performance.now();
    const found = await guarded(models.ground(spec, frame.url, Object.values(PHRASES)));
    const ms = Math.round(performance.now() - t0);
    const scale = stageEl.clientWidth / frame.w;
    for (const { label, box } of found) {
      const name = Object.keys(PHRASES).find((k) => PHRASES[k] === label);
      const el = document.createElement("div");
      Object.assign(el.style, { left: `${box[0] * scale}px`, top: `${box[1] * scale}px`, width: `${(box[2] - box[0]) * scale}px`, height: `${(box[3] - box[1]) * scale}px` });
      el.innerHTML = `<span>${esc(label)}</span>`;
      detsEl.appendChild(el);
      const p = groundPoint(box, frame, name === "bowl" ? 0.02 : CUBE / 2);
      if (p) targets[name] = p;
    }
    // Grounding names may miss; fall back to nothing rather than guessing a position.
    return { targets, note: `${found.length}/4 objects grounded · ${ms} ms on ${models.device}` };
  }

  function parsePlan(text) {
    const plan = [];
    for (const m of text.matchAll(/([a-z_]+)\s*\(\s*([a-z_ ]*)\s*\)/gi)) {
      const skill = m[1].toLowerCase();
      const arg = m[2].trim().toLowerCase().replace(/\s+/g, "_");
      if (SKILLS.includes(skill) && (!arg || OBJECTS[arg])) plan.push([skill, arg]);
    }
    return plan;
  }

  function planMessages(targets) {
    const seen = Object.keys(targets).join(", ") || "none";
    // One worked example is enough for a 0.5B model to follow the skill format.
    return [
      { role: "system", content: "You plan motions for an SO-101 robot arm. Skills: move_above(obj), grasp(obj), lift(), release(obj), go_home(). Reply with only skill calls, one per line." },
      { role: "user", content: "Objects: red_cube, blue_cube, green_cube, bowl.\nTask: put the green cube in the bowl" },
      { role: "assistant", content: "move_above(green_cube)\ngrasp(green_cube)\nlift()\nmove_above(bowl)\nrelease(bowl)\ngo_home()" },
      { role: "user", content: `Objects: ${seen}.\nTask: ${TASK}` },
    ];
  }

  // Show a planner's last two lines under the reason stage while it streams. Tokens that
  // arrive after the run was aborted or restarted are dropped.
  function liveOutput() {
    const meta = stageLi("reason").querySelector(".meta");
    const base = meta.innerHTML;
    const g = gen;
    let streamed = "";
    return (tok) => {
      if (g !== gen) return;
      streamed += tok;
      meta.innerHTML = `${base}<br><span class="out">${esc(streamed.trim().split("\n").slice(-2).join(" · "))}</span>`;
    };
  }

  async function cloudPlan(route, targets) {
    const spec = router.models[route.model];
    const key = loadKeys().keys[spec.provider];
    const label = router.config.providers[spec.provider].label;
    const cacheKey = spec.name + ":" + key;
    if (!cloudPlans.has(cacheKey)) {
      const t0 = performance.now();
      cloudPlans.set(cacheKey, complete(router.config, spec, planMessages(targets), key, 200, liveOutput())
        .then((text) => ({ plan: parsePlan(text), ms: Math.round(performance.now() - t0) })));
    }
    try {
      const { plan, ms } = await guarded(cloudPlans.get(cacheKey));
      return plan.length
        ? { plan, note: `${ms} ms · ${label} · your key` }
        : { plan: DEFAULT_PLAN, note: `${label} output did not parse, using default plan` };
    } catch (e) {
      if (e instanceof Aborted) throw e;
      return { plan: DEFAULT_PLAN, note: e.message };
    }
  }

  async function reasonStage(route, targets) {
    const spec = router.models[route.model];
    if (mode === "sim" && spec?.provider && loadKeys().keys[spec.provider]) return cloudPlan(route, targets);
    if (!isBrowserModel(route)) {
      await sleep(Math.min(1800, 500 + route.est_latency_ms * 0.5));
      return { plan: DEFAULT_PLAN, note: "" };
    }
    const t0 = performance.now();
    const text = await guarded(models.chat(spec, planMessages(targets), liveOutput()));
    const ms = Math.round(performance.now() - t0);
    const plan = parsePlan(text);
    return plan.length
      ? { plan, note: `${ms} ms on ${models.device}` }
      : { plan: DEFAULT_PLAN, note: `${ms} ms · output did not parse, using default plan` };
  }

  async function actStage(plan, targets) {
    const steps = planEl.querySelectorAll("li");
    const mark = (i) => steps.forEach((li, j) => (li.className = j < i ? "done" : j === i ? "now" : ""));
    const go = (p) => moveTip(p, clamp(tip.distanceTo(p) * 5500, 350, 1500));
    let holding = null;
    for (let i = 0; i < plan.length; i++) {
      mark(i);
      const [skill, arg] = plan[i];
      const p = targets[arg];
      if (skill === "move_above" && p) {
        await go(new THREE.Vector3(p.x, holding ? 0.12 : 0.085, p.z));
      } else if (skill === "grasp" && p && !holding) {
        if (Math.hypot(tip.x - p.x, tip.z - p.z) > 0.01) await go(new THREE.Vector3(p.x, 0.085, p.z));
        await go(new THREE.Vector3(p.x, CUBE / 2, p.z));
        await grip(GRIP_CLOSED, 450);
        const tipWorld = tipAnchor.getWorldPosition(new THREE.Vector3());
        const cube = [red, blue, green].find((c) => Math.hypot(c.position.x - tipWorld.x, c.position.z - tipWorld.z) < 0.02);
        if (cube) { tipAnchor.attach(cube); holding = cube; } else { await grip(GRIP_OPEN, 300); status("grasp missed: nothing between the jaws"); }
      } else if (skill === "lift") {
        await go(new THREE.Vector3(tip.x, 0.12, tip.z));
      } else if (skill === "release") {
        if (p && Math.hypot(tip.x - p.x, tip.z - p.z) > 0.01) await go(new THREE.Vector3(p.x, 0.12, p.z));
        const overBowl = Math.hypot(tip.x - bowl.position.x, tip.z - bowl.position.z) < 0.04;
        await go(new THREE.Vector3(tip.x, overBowl ? 0.075 : CUBE / 2 + 0.004, tip.z));
        await grip(GRIP_OPEN, 300);
        if (holding) {
          const cube = holding;
          holding = null;
          scene.attach(cube);
          const from = cube.position.clone();
          const to = overBowl
            ? new THREE.Vector3(bowl.position.x, 0.006 + CUBE / 2, bowl.position.z)
            : new THREE.Vector3(from.x, CUBE / 2, from.z);
          const spin = cube.rotation.y;
          await tween(380, (k) => {
            cube.position.lerpVectors(from, to, k);
            cube.position.y = from.y + (to.y - from.y) * k * k;
            cube.rotation.y = spin + k * 0.5;
          });
        }
      } else if (skill === "go_home") {
        await go(HOME.clone());
      }
    }
    mark(plan.length);
    return Math.hypot(red.position.x - bowl.position.x, red.position.z - bowl.position.z) < 0.04;
  }

  async function runTask() {
    const [perceive, reason, act] = router.routeTask(request());
    status("instruction received");
    await sleep(500);
    await type(TASK);
    await sleep(350);

    setStage("perceive", perceive, "running");
    status(`perceiving · ${perceive.model}`);
    const seen = await perceiveStage(perceive);
    setStage("perceive", perceive, "done", seen.note);

    setStage("reason", reason, "running");
    status(`planning · ${reason.model} @ ${reason.location}`);
    const { plan, note } = await reasonStage(reason, seen.targets);
    for (const [skill, arg] of plan) {
      planEl.insertAdjacentHTML("beforeend", `<li><code>${skill}(${arg})</code></li>`);
      await sleep(140);
    }
    setStage("reason", reason, "done", note);

    setStage("act", act, "running");
    status(`acting · ${act.model} @ ${act.location} · 30 Hz`);
    detsEl.innerHTML = "";
    const ok = await actStage(plan, seen.targets);
    setStage("act", act, "done");
    status(ok ? "task complete ✓" : "plan finished · red cube not in bowl", ok ? "ok" : "");
    await sleep(3200);
  }

  async function loop() {
    for (;;) {
      await gate;
      reset();
      try {
        await runTask();
      } catch (e) {
        if (e instanceof Aborted) continue;
        console.error(e);
        if (mode === "browser") setMode("sim", `In-browser run failed (${e.message}). Back to simulated routing.`);
        await new Promise((r) => setTimeout(r, 1500));
      }
    }
  }

  // ---------------- controls ----------------
  const onnxBtn = $('[data-act="onnx"]');
  const estopBtn = $('[data-act="estop"]');
  function flash(msg) {
    const b = $(".demo-banner");
    b.hidden = false;
    b.innerHTML = esc(msg);
    setTimeout(() => (b.hidden = true), 4000);
  }
  function setMode(next, message) {
    mode = next;
    onnxBtn.setAttribute("aria-pressed", String(next === "browser"));
    onnxBtn.textContent = next === "browser" ? `In browser · ${models.device}` : "Run in your browser";
    $(".demo-live").lastChild.textContent = next === "browser" ? `SO-101 · ONNX on ${models.device}` : "SO-101 · simulated";
    if (message) flash(message);
  }

  async function enableBrowser() {
    runtimes = await detectRuntimes();
    const device = runtimes[0];
    const routes = router.routeTask({ ...REQUEST, locations: ["browser"], client_runtimes: runtimes });
    const specs = routes.map((r) => router.models[r.model]).filter((m) => m?.onnx);
    if (!specs.length) throw new Error("the router selected no browser models");
    const prog = document.createElement("div");
    prog.className = "demo-progress";
    prog.innerHTML = `<span>Downloading ${specs.map((s) => esc(s.name)).join(" + ")} for ${device}${device === "wasm" ? " (no WebGPU: slower)" : ""}…</span><i><em></em></i>`;
    stageEl.appendChild(prog);
    models = new BrowserModels(device, ({ loaded, total }) => {
      if (!total) return;
      prog.querySelector("em").style.width = `${(loaded / total) * 100}%`;
      prog.querySelector("span").textContent = `Downloading models for ${device} · ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`;
    });
    try {
      for (const spec of specs) await models.get(spec);
    } finally {
      prog.remove();
    }
    setMode("browser");
  }

  let detectRuntimes, BrowserModels;
  onnxBtn.onclick = () => {
    if (mode === "browser") { setMode("sim"); abortAll(); return; }
    // Download in the background while the simulated run keeps looping, then
    // restart the task in browser mode once the weights are ready.
    onnxBtn.disabled = true;
    (async () => {
      try {
        ({ detectRuntimes, BrowserModels } = await import("./onnx.js"));
        await enableBrowser();
        abortAll();
      } catch (e) {
        console.error(e);
        flash(`Could not run models in this browser: ${e.message}`);
      } finally {
        onnxBtn.disabled = false;
      }
    })();
  };

  function estop() {
    const route = router.route({ instruction: "stop", estop: true, embodiment: "so101" });
    gate = new Promise((r) => (resume = r));
    abortAll(); // freeze the arm exactly where it is
    root.classList.add("stopped");
    const b = $(".demo-banner");
    b.hidden = false;
    b.innerHTML = `<b>E-STOP</b> ${esc(route.decision)} → ${esc(route.model)} @ ${route.location} · ~${route.est_latency_ms} ms · hold position`;
    status("stopped · holding position", "stop");
    root.querySelectorAll("[data-stage]").forEach((li) => { if (li.dataset.state === "running") li.dataset.state = "halted"; });
    estopBtn.textContent = "Resume";
  }
  function unstop() {
    estopBtn.textContent = "E-STOP";
    root.classList.remove("stopped");
    resume?.();
  }
  estopBtn.onclick = () => (root.classList.contains("stopped") ? unstop() : estop());
  $('[data-act="replay"]').onclick = () => (root.classList.contains("stopped") ? unstop() : abortAll());

  // ---------------- render loop ----------------
  let visible = false;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }, { threshold: 0.05 }).observe(root);
  function resize() {
    const { clientWidth: w, clientHeight: h } = stageEl;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(stageEl);
  resize();

  const v = new THREE.Vector3();
  function placeLabels() {
    const w = stageEl.clientWidth, h = stageEl.clientHeight;
    for (const d of detections) {
      if (!d.box.visible) continue;
      d.obj.getWorldPosition(v);
      v.y += d.height;
      v.project(camera);
      d.el.style.transform = `translate(-50%, -100%) translate(${((v.x + 1) / 2) * w}px, ${((1 - v.y) / 2) * h}px)`;
    }
  }
  const clock = new THREE.Clock();
  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.05);
    if (!visible || document.hidden) return;
    now += dt;
    stepTweens();
    controls.update();
    updateJoints();
    renderer.render(scene, camera);
    placeLabels();
  });

  loop();
}
