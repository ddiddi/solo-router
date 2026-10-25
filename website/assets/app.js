import { Router } from "./router.js";
import { complete, enabledProviders, loadKeys, saveKeys } from "./providers.js";

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
  ["π0", "#AD94F9"], ["GR00T N1.5", "#3C9CC4"], ["SmolVLA", "#AD94F9"], ["OpenVLA", "#3C9CC4"], ["Florence-2", "#AD94F9"],
  ["SmolVLM2", "#3C9CC4"], ["Octo", "#AD94F9"],
  ["RT-2", "#3C9CC4"], ["Qwen2.5-VL", "#AD94F9"], ["SmolLM2", "#3C9CC4"], ["Qwen3", "#AD94F9"], ["Claude", "#3C9CC4"], ["GPT-5.5", "#AD94F9"], ["Nebius Token Factory", "#3C9CC4"], ["gpt-oss", "#AD94F9"],
  ["Gemini Robotics", "#AD94F9"], ["LeRobot ACT", "#3C9CC4"], ["Diffusion Policy", "#AD94F9"], ["ONNX · WebGPU", "#3C9CC4"],
];
const chips = families.map(([n, c]) => `<span class="chip"><i style="background:${c}"></i>${esc(n)}</span>`).join("");
$("#marquee").innerHTML = chips + chips.replace(/<span class="chip">/g, '<span class="chip" aria-hidden="true">');

// ---------- playground ----------
const HZ = [0, 2, 5, 10, 20, 30, 50, 100, 200];
const PRESETS = [
  ["pick & place", "put the red cube in the bowl", { embodiment: "so101" }],
  ["stop!", "stop!", {}],
  ["hand me scissors", "hand me the scissors", { embodiment: "so101", humans: true }],
  ["humanoid fold", "fold the towel", { embodiment: "humanoid" }],
  ["patrol", "patrol the warehouse aisle", { embodiment: "quadruped", obs: ["lidar"] }],
  ["plan a chore", "clean the kitchen, then load the dishwasher and finally wipe the counters", { embodiment: "mobile_manipulator", hz: 0 }],
  ["what's on the shelf", "what do you see on the shelf", { hz: 0 }],
  ["joint fault", "why is joint 3 overheating", { hz: 0 }],
];

let router;
const state = { network: "online", ...loadKeys() };

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
    providers: enabledProviders(state.keys),
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

function render(keepAnswer = false) {
  if (!keepAnswer) $("#r-answer").hidden = true;
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
    ...(r.provider ? [`<span class="badge provider">${esc(router.config.providers[r.provider].label)}${state.keys[r.provider] ? " · your key" : ""}</span>`] : []),
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

// ---------- one prompt, one click ----------
function answer(html, cls = "") {
  const el = $("#r-answer");
  el.className = "answer " + cls;
  el.innerHTML = html;
  el.hidden = false;
}

async function run() {
  if (!router || !$("#instruction").value.trim()) return;
  render(true);
  const req = readRequest();
  let r = router.route(req);
  if (r.signals.safety === "critical") return answer("STOP · the safety controller holds position. No model is called.", "stop");
  let note = "";
  if (!router.models[r.model].provider) {
    // A browser can't reach the robot's onboard or edge models, so route among hosted ones,
    // as `solo-router run` does.
    const robotSide = r;
    r = router.route({ ...req, locations: ["cloud"] });
    if (!router.models[r.model].provider) {
      return answer(`Routed to <b>${esc(robotSide.model)}</b> @ ${robotSide.location}, a ${esc(router.models[robotSide.model].kind)} that runs on the robot. No hosted model can do this step. On the robot, <code>solo-router serve --proxy</code> forwards it to the policy server.`);
    }
    note = ` · on a robot: ${robotSide.model} @ ${robotSide.location}`;
  }
  const spec = router.models[r.model];
  const key = state.keys[spec.provider];
  const label = router.config.providers[spec.provider].label;
  if (!key) {
    openKeys(true);
    return answer(`The router picked <b>${esc(r.model)}</b>. Add your ${esc(label)} key to run it, or any other key to route to that provider instead.`);
  }
  const btn = $("#run");
  btn.disabled = true;
  answer(`<span class="answer-head">${esc(r.model)} · ${esc(label)}</span><span class="answer-wait">thinking…</span>`);
  const t0 = performance.now();
  try {
    const messages = [
      { role: "system", content: "You are the reasoning model behind a robot router. Answer briefly. For tasks, reply with a short numbered plan of robot skills." },
      { role: "user", content: $("#instruction").value },
    ];
    const text = await complete(router.config, spec, messages, key);
    const s = ((performance.now() - t0) / 1000).toFixed(1);
    answer(`<span class="answer-head">${esc(r.model)} · ${esc(label)} · ${s}s${esc(note)}</span>${esc(text.trim() || "(empty reply)")}`);
  } catch (e) {
    answer(esc(e.message), "error");
  } finally {
    btn.disabled = false;
  }
}

function openKeys(open) {
  $("#keys").hidden = !open;
  $("#keys-toggle").setAttribute("aria-expanded", String(open));
}

function renderKeys(cfg) {
  const providers = Object.entries(cfg.providers || {});
  $("#keys-fields").innerHTML = providers.map(([id, p]) => `
    <div class="key-field">
      <label for="key-${id}">${esc(p.label)}</label>
      <input id="key-${id}" data-provider="${id}" type="password" autocomplete="off" spellcheck="false"
        placeholder="${esc(p.api_key_env)}" value="${esc(state.keys[id] || "")}">
      ${p.console_url ? `<a href="${esc(p.console_url)}" target="_blank" rel="noopener">get key ↗</a>` : ""}
    </div>`).join("");
  $("#keys-remember").checked = state.remember;
  const sync = () => {
    const keys = Object.fromEntries($$("#keys-fields input").map((i) => [i.dataset.provider, i.value.trim()]));
    state.remember = $("#keys-remember").checked;
    state.keys = saveKeys(keys, state.remember);
    const n = Object.keys(state.keys).length;
    $("#keys-count").textContent = n ? `· ${n} set` : "";
    render();
  };
  $$("#keys-fields input").forEach((i) => i.addEventListener("input", sync));
  $("#keys-remember").onchange = sync;
  $("#keys-toggle").onclick = () => openKeys($("#keys").hidden);
  $("#keys-count").textContent = Object.keys(state.keys).length ? `· ${Object.keys(state.keys).length} set` : "";
}

$("#run").onclick = run;
$("#instruction").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); run(); }
});

function applyPreset([, text, opts]) {
  $("#instruction").value = text;
  if (opts.embodiment) $("#embodiment").value = opts.embodiment;
  pressed($("#humans"), !!opts.humans);
  pressed($("#estop"), false);
  const obs = opts.obs || ["rgb", "proprio"];
  $$("#obs .toggle").forEach((b) => pressed(b, obs.includes(b.dataset.v)));
  $("#hz").value = opts.hz === 0 ? 0 : 5;
  render();
}

$("#presets").innerHTML = PRESETS.map(([label], i) => `<button type="button" class="preset" data-i="${i}">${esc(label)}</button>`).join("");
$$(".preset").forEach((b) => (b.onclick = () => applyPreset(PRESETS[+b.dataset.i])));
$$("#obs .toggle, #humans, #estop").forEach((b) => (b.onclick = () => { pressed(b, !isPressed(b)); render(); }));
$$("#network button").forEach((b) => (b.onclick = () => {
  $$("#network button").forEach((x) => pressed(x, x === b)); state.network = b.dataset.v; render();
}));
["#instruction", "#hz", "#battery"].forEach((id) => $(id).addEventListener("input", () => render()));
["#embodiment", "#preference"].forEach((id) => $(id).addEventListener("change", () => render()));

render();
fetch("assets/router-config.json")
  .then((r) => r.json())
  .then((cfg) => {
    router = new Router(cfg);
    renderKeys(cfg);
    const set = (k, v) => { const el = document.querySelector(`[data-stat="${k}"]`); if (el) el.textContent = v; };
    set("decisions", cfg.decisions.length);
    set("models", cfg.models.length);
    render();
    renderRoles(cfg);
    mountDemoWhenNear();
  })
  .catch((err) => { $("#r-trace").textContent = "could not load router config: " + err; });

// ---------- default model set, grouped by role ----------
const ROLE_INFO = {
  perception: ["01", "Perception", "See", "Find, box and describe what is in front of the robot."],
  reasoning: ["02", "Reasoning", "Think", "Plan multi-step tasks, explain, diagnose."],
  action: ["03", "Action", "Move", "Policies that output joint and base commands."],
};
function renderRoles(cfg) {
  const card = (m) => `
    <div class="model-card">
      <div class="model-top"><b>${esc(m.name)}</b><span class="badge ${m.location}">${m.location === "browser" ? "browser · " + esc((m.runtimes || []).join("/")) : m.provider ? "cloud · " + esc(cfg.providers[m.provider].label) : m.location}</span></div>
      <p>${esc(m.description || "")}</p>
      <div class="model-meta"><span>${esc(m.kind)}</span><span>~${m.p50_latency_ms} ms</span>${m.action_chunk > 1 ? `<span>chunk ${m.action_chunk}</span>` : ""}${m.onnx ? `<span class="onnx">ONNX</span>` : ""}</div>
    </div>`;
  $("#roles").innerHTML = Object.entries(ROLE_INFO).map(([role, [n, title, verb, blurb]]) => `
    <div class="role reveal in">
      <div class="role-head"><span class="role-num">${n}</span><div><h3>${title} <em>· ${verb}</em></h3><p>${blurb}</p></div></div>
      ${cfg.models.filter((m) => m.role === role).map(card).join("")}
    </div>`).join("");
  const fs = cfg.models.find((m) => m.role === "safety");
  if (fs) $("#failsafe").innerHTML = `<span class="badge critical">safety</span> <b>${esc(fs.name)}</b> · ${esc(fs.description)} Every fallback chain ends here.`;
}

// ---------- SO-101 demo (three.js loads only when the section is near) ----------
function mountDemoWhenNear() {
  const el = $("#demo-root");
  if (!el) return;
  const obs = new IntersectionObserver(async ([e]) => {
    if (!e.isIntersecting) return;
    obs.disconnect();
    try {
      const { mountDemo } = await import("./demo.js");
      mountDemo(el, router);
    } catch (err) {
      el.innerHTML = `<div class="demo-loading">Could not start the 3D demo (${esc(err.message)}). It needs WebGL and access to cdn.jsdelivr.net.</div>`;
    }
  }, { rootMargin: "400px" });
  obs.observe(el);
}
