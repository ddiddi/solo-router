import { Router } from "./router.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// ---------- chrome ----------
const nav = $("#nav");
addEventListener("scroll", () => nav.classList.toggle("scrolled", scrollY > 8), { passive: true });
$("#burger").onclick = () => $("#menu").classList.add("open");
$("#menu-close").onclick = () => $("#menu").classList.remove("open");
$$("#menu a").forEach((a) => (a.onclick = () => $("#menu").classList.remove("open")));

async function copy(text, flash) {
  try { await navigator.clipboard.writeText(text); } catch { return; }
  if (flash) { flash.hidden = false; setTimeout(() => (flash.hidden = true), 1400); }
}
$$("[data-copy]").forEach((b) => (b.onclick = () => copy(b.dataset.copy, b.querySelector(".copied"))));
$$("[data-copy-target]").forEach((b) => (b.onclick = () => {
  const lines = $("#" + b.dataset.copyTarget).innerText.split("\n").filter((l) => l.startsWith("$ ") || l.startsWith("    ") || l.startsWith("  }"));
  copy(lines.map((l) => l.replace(/^\$ /, "")).join("\n"));
  b.textContent = "copied"; setTimeout(() => (b.textContent = "copy"), 1400);
}));

$$(".tab").forEach((t) => (t.onclick = () => {
  $$(".tab").forEach((x) => x.setAttribute("aria-selected", String(x === t)));
  $$("[data-panel]").forEach((p) => (p.hidden = p.dataset.panel !== t.dataset.tab));
}));

const io = new IntersectionObserver((entries) => entries.forEach((e) => {
  if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
}), { threshold: 0.12 });
$$(".reveal").forEach((el) => io.observe(el));

// ---------- marquee ----------
const families = [
  ["π0", "#AD94F9"], ["GR00T N1.5", "#3C9CC4"], ["SmolVLA", "#AD94F9"], ["OpenVLA", "#3C9CC4"], ["Octo", "#AD94F9"],
  ["RT-2", "#3C9CC4"], ["Qwen2.5-VL", "#AD94F9"], ["SmolLM2", "#3C9CC4"], ["Qwen3", "#AD94F9"], ["Claude", "#3C9CC4"],
  ["Gemini Robotics", "#AD94F9"], ["LeRobot ACT", "#3C9CC4"], ["Diffusion Policy", "#AD94F9"], ["Llama", "#3C9CC4"],
];
const chips = families.map(([n, c]) => `<span class="chip"><i style="background:${c}"></i>${esc(n)}</span>`).join("");
$("#marquee").innerHTML = chips + chips.replace(/<span class="chip">/g, '<span class="chip" aria-hidden="true">');

// ---------- playground ----------
const HZ = [0, 2, 5, 10, 20, 30, 50, 100, 200];
const PRESETS = [
  ["pick & place", "pick up the red cup and place it on the tray", { embodiment: "arm" }],
  ["stop!", "stop!", {}],
  ["hand me scissors", "hand me the scissors", { embodiment: "arm", humans: true }],
  ["humanoid fold", "fold the towel", { embodiment: "humanoid" }],
  ["patrol", "patrol the warehouse aisle", { embodiment: "quadruped", obs: ["lidar"] }],
  ["chore", "clean the kitchen, then load the dishwasher and finally wipe the counters", { embodiment: "mobile_manipulator", hz: 0 }],
  ["what's on the shelf", "what do you see on the shelf", { hz: 0 }],
  ["joint fault", "why is joint 3 overheating", { hz: 0 }],
];

let router;
const state = { network: "online" };

function pressed(el, v) { el.setAttribute("aria-pressed", String(v)); }
function isPressed(el) { return el.getAttribute("aria-pressed") === "true"; }

function readRequest() {
  const hz = HZ[+$("#hz").value];
  return {
    instruction: $("#instruction").value,
    embodiment: $("#embodiment").value,
    observations: $$("#obs .toggle").filter(isPressed).map((b) => b.dataset.v),
    control_hz: hz || null,
    network: state.network,
    battery_pct: +$("#battery").value,
    humans_nearby: isPressed($("#humans")),
    estop: isPressed($("#estop")),
    preference: $("#preference").value,
  };
}

function flash(id, text) {
  const el = $(id);
  if (el.textContent !== text) {
    el.textContent = text;
    const stage = el.parentElement;
    stage.classList.remove("on"); void stage.offsetWidth; stage.classList.add("on");
  }
}

function render() {
  $("#hz-val").textContent = HZ[+$("#hz").value] ? `${HZ[+$("#hz").value]} Hz · ${(1000 / HZ[+$("#hz").value]).toFixed(0)} ms budget` : "no control loop";
  $("#battery-val").textContent = `${$("#battery").value}%`;
  if (!router) return;
  const req = readRequest();
  const r = router.route(req);
  const s = r.signals;

  flash("#st-signal", `${s.intent} · ${s.horizon}`);
  flash("#st-decision", r.decision);
  flash("#st-algo", r.algorithm);
  flash("#st-backend", r.location);

  $("#r-model").textContent = r.model;
  $("#r-desc").textContent = r.description || "";
  $("#r-badges").innerHTML = [
    `<span class="badge ${r.location}">${r.location}</span>`,
    `<span class="badge">~${r.est_latency_ms} ms</span>`,
    `<span class="badge ${s.safety}">safety: ${s.safety}</span>`,
    `<span class="badge">${s.latency_tier}</span>`,
  ].join("");

  const g = r.guardrails;
  $("#r-guard").innerHTML = g.action === "stop"
    ? `<div class="guard stop">STOP · hold position · ${esc(s.safety_reasons.join("; "))}</div>`
    : Object.keys(g).length
      ? `<div class="guard">guardrails · max ${g.max_speed_mps} m/s · max ${g.max_force_n} N · ${esc(s.safety_reasons.join("; "))}</div>`
      : "";

  const decision = router.decisions.find((d) => d.name === r.decision);
  const names = decision ? decision.candidates : [r.model];
  const scores = Object.fromEntries(r.ranked.map((c) => [c.model, c.score]));
  const vals = r.ranked.map((c) => c.score);
  const lo = Math.min(0, ...vals), hi = Math.max(...vals, 0.0001);
  $("#r-bars").innerHTML = names.map((n) => {
    const rej = r.rejected[n];
    const win = n === r.model;
    const pct = rej ? 0 : Math.max(6, ((scores[n] - lo) / (hi - lo || 1)) * 100);
    const val = rej ? esc(rej) : (r.algorithm === "static" ? "fixed" : scores[n].toFixed(3));
    return `<div class="bar ${win ? "win" : ""} ${rej ? "no" : ""}" title="${esc(rej || router.models[n].description || "")}">
      <span class="name">${esc(n)}</span><span class="track"><span class="fill" style="width:${pct}%"></span></span><span class="val">${val}</span></div>`;
  }).join("") + (r.model === "safety-controller" && !names.includes(r.model)
    ? `<div class="bar win"><span class="name">safety-controller</span><span class="track"><span class="fill" style="width:100%"></span></span><span class="val">fail-safe</span></div>` : "");

  $("#r-trace").innerHTML = r.trace.map((line) => {
    const cls = line.startsWith("select") || line.startsWith("no feasible") ? "t-sel" : line.startsWith("reject") ? "t-rej" : line.startsWith("decision") ? "t-dec" : "";
    return `<span class="${cls}">› ${esc(line)}</span>`;
  }).join("\n") + (r.fallbacks.length ? `\n<span class="t-rej">› fallbacks: ${esc(r.fallbacks.join(" → "))}</span>` : "");
}

function applyPreset([, text, opts]) {
  $("#instruction").value = text;
  if (opts.embodiment) $("#embodiment").value = opts.embodiment;
  pressed($("#humans"), !!opts.humans);
  pressed($("#estop"), false);
  const obs = opts.obs || ["rgb", "proprio"];
  $$("#obs .toggle").forEach((b) => pressed(b, obs.includes(b.dataset.v)));
  $("#hz").value = opts.hz === 0 ? 0 : 4;
  render();
}

$("#presets").innerHTML = PRESETS.map(([label], i) => `<button type="button" class="preset" data-i="${i}">${esc(label)}</button>`).join("");
$$(".preset").forEach((b) => (b.onclick = () => applyPreset(PRESETS[+b.dataset.i])));
$$("#obs .toggle, #humans, #estop").forEach((b) => (b.onclick = () => { pressed(b, !isPressed(b)); render(); }));
$$("#network button").forEach((b) => (b.onclick = () => {
  $$("#network button").forEach((x) => pressed(x, x === b)); state.network = b.dataset.v; render();
}));
["#instruction", "#hz", "#battery"].forEach((id) => $(id).addEventListener("input", render));
["#embodiment", "#preference"].forEach((id) => $(id).addEventListener("change", render));

render();
fetch("assets/router-config.json")
  .then((r) => r.json())
  .then((cfg) => {
    router = new Router(cfg);
    const set = (k, v) => { const el = document.querySelector(`[data-stat="${k}"]`); if (el) el.textContent = v; };
    set("decisions", cfg.decisions.length);
    set("models", cfg.models.length);
    render();
  })
  .catch((err) => { $("#r-trace").textContent = "could not load router config: " + err; });
