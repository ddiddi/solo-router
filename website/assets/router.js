// Browser port of src/solo_router/{signals,router}.py. Reads the same config
// (router-config.json, exported by `solo-router export`) so the playground
// makes exactly the decisions the Python router makes.

const FAILSAFE = "safety-controller";
const PREFERENCE_WEIGHTS = {
  balanced: [1.0, 0.3, 0.05],
  quality: [1.0, 0.05, 0.02],
  latency: [0.6, 1.0, 0.05],
  cost: [0.6, 0.1, 1.0],
};
const DEGRADED_CLOUD_PENALTY = 0.3;
const LOW_BATTERY_PENALTY_PER_W = 0.01;
const DEFAULT_LATENCY_SCALE_MS = 3000;
const RGB_STREAMS = ["rgb", "wrist_rgb"];

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const phrase = (p) => new RegExp("(?<![a-z0-9])" + escapeRe(p.toLowerCase()) + "(?![a-z0-9])", "g");
const countMatches = (rx, text) => (text.match(rx) || []).length;

export class Router {
  constructor(config) {
    this.config = config;
    const s = config.signals || {};
    this.intents = Object.entries(s.intent || {}).map(([name, words]) => [name, words.map(phrase)]);
    this.horizon = { long_markers: 2, long_words: 24, ...(s.horizon || {}) };
    this.markers = (this.horizon.markers || []).map(phrase);
    this.critical = (s.safety?.critical_patterns || []).map((p) => new RegExp(p, "i"));
    this.hazards = (s.safety?.hazard_words || []).map((w) => [w, phrase(w)]);
    this.realtimeMs = s.latency?.realtime_ms ?? 50;
    this.interactiveMs = s.latency?.interactive_ms ?? 500;
    this.lowBattery = s.energy?.low_battery_pct ?? 20;
    this.models = Object.fromEntries(config.models.map((m) => [m.name, {
      max_hz: null, cost_per_1k: 0, power_w: 0, embodiments: ["*"], requires: [], safety_tier: 1, ...m,
    }]));
    this.decisions = [...config.decisions]
      .map((d) => ({ priority: 0, when: {}, algorithm: "static", prefer: null, ...d }))
      .sort((a, b) => b.priority - a.priority);
    this.minElevatedTier = config.guardrails?.elevated?.min_safety_tier ?? 0;
  }

  signals(req) {
    const text = (req.instruction || "").toLowerCase().trim();
    const scores = {};
    for (const [name, rxs] of this.intents) scores[name] = rxs.filter((rx) => countMatches(rx, text) > 0).length;
    const best = Math.max(0, ...Object.values(scores));
    const intent = best ? Object.keys(scores).find((k) => scores[k] === best) : "conversation";

    const markerHits = this.markers.reduce((n, rx) => n + countMatches(rx, text), 0) + (text.split(",").length - 1);
    const words = text.split(/\s+/).filter(Boolean).length;
    const horizon = markerHits >= this.horizon.long_markers || words >= this.horizon.long_words ? "long" : "short";

    let safety = "normal";
    let reasons = [];
    if (req.estop) {
      safety = "critical"; reasons = ["estop flag set"];
    } else if (this.critical.some((rx) => rx.test(text))) {
      safety = "critical"; reasons = ["stop command in instruction"];
    } else {
      const hazards = this.hazards.filter(([, rx]) => countMatches(rx, text) > 0).map(([w]) => w);
      if (hazards.length) reasons.push("hazard words: " + hazards.join(", "));
      if (req.humans_nearby && (intent === "manipulation" || intent === "navigation")) reasons.push(`humans nearby during ${intent}`);
      if (reasons.length) safety = "elevated";
    }

    let budget = req.latency_budget_ms ?? null;
    if (budget == null && req.control_hz) budget = 1000 / req.control_hz;
    const tier = budget != null && budget < this.realtimeMs ? "realtime"
      : budget != null && budget < this.interactiveMs ? "interactive" : "deliberative";
    const energy = req.battery_pct != null && req.battery_pct < this.lowBattery ? "low" : "normal";
    const obs = req.observations || [];

    return {
      intent, intent_scores: scores, horizon, safety, safety_reasons: reasons,
      latency_budget_ms: budget, latency_tier: tier,
      modalities: text ? ["text", ...obs] : [...obs],
      embodiment: req.embodiment || null, network: req.network || "online", energy,
      preference: req.preference || "balanced",
    };
  }

  matches(when, s) {
    return Object.entries(when).every(([key, expected]) => {
      if (key === "any") return expected.some((sub) => this.matches(sub, s));
      if (key === "all") return expected.every((sub) => this.matches(sub, s));
      return (Array.isArray(expected) ? expected : [expected]).includes(s[key]);
    });
  }

  reject(m, s, req) {
    if (m.name === FAILSAFE) return null;
    if (s.embodiment && !m.embodiments.includes("*") && !m.embodiments.includes(s.embodiment)) return `does not support embodiment ${s.embodiment}`;
    for (const need of m.requires) {
      const have = need === "rgb" ? RGB_STREAMS : [need];
      if (!have.some((h) => s.modalities.includes(h))) return `needs ${need} observation`;
    }
    if (s.network === "offline" && m.location === "cloud") return "robot is offline";
    if (s.latency_budget_ms != null && m.p50_latency_ms > s.latency_budget_ms) return `p50 ${fmt(m.p50_latency_ms)}ms exceeds budget ${fmt(s.latency_budget_ms)}ms`;
    if (req.control_hz && m.max_hz != null && m.max_hz < req.control_hz) return `max ${fmt(m.max_hz)}Hz below control rate ${fmt(req.control_hz)}Hz`;
    if (s.safety === "elevated" && m.safety_tier < this.minElevatedTier) return `safety tier ${m.safety_tier} below required ${this.minElevatedTier}`;
    return null;
  }

  rank(decision, feasible, s) {
    if (!feasible.length) return [];
    const cand = (m, score) => ({ model: m.name, location: m.location, score, p50_latency_ms: m.p50_latency_ms });
    if (decision.algorithm === "static") return feasible.map((m) => cand(m, 1));
    if (decision.algorithm === "latency_aware") {
      return [...feasible].sort((a, b) => b.quality - a.quality || a.p50_latency_ms - b.p50_latency_ms).map((m) => cand(m, m.quality));
    }
    let pref = s.preference !== "balanced" ? s.preference : decision.prefer || "balanced";
    if (!PREFERENCE_WEIGHTS[pref]) pref = "balanced";
    const [wq, wl, wc] = PREFERENCE_WEIGHTS[pref];
    const scale = s.latency_budget_ms || DEFAULT_LATENCY_SCALE_MS;
    const maxCost = Math.max(0, ...feasible.map((m) => m.cost_per_1k)) || 1;
    return feasible.map((m) => {
      let score = wq * m.quality - wl * Math.min(m.p50_latency_ms / scale, 1) - wc * m.cost_per_1k / maxCost;
      if (s.network === "degraded" && m.location === "cloud") score -= DEGRADED_CLOUD_PENALTY;
      if (s.energy === "low") score -= LOW_BATTERY_PENALTY_PER_W * m.power_w;
      return cand(m, Math.round(score * 1e4) / 1e4);
    }).sort((a, b) => b.score - a.score);
  }

  route(req) {
    const s = this.signals(req);
    const trace = [`signals: intent=${s.intent} horizon=${s.horizon} safety=${s.safety} latency=${s.latency_tier} network=${s.network}`];
    const decision = this.decisions.find((d) => this.matches(d.when, s))
      || { name: "failsafe", priority: 0, when: {}, candidates: [FAILSAFE], algorithm: "static" };
    trace.push(`decision: ${decision.name} (priority ${decision.priority}, ${decision.algorithm})`);

    const feasible = [];
    const rejected = {};
    for (const name of decision.candidates) {
      const m = this.models[name];
      const why = this.reject(m, s, req);
      if (why) { rejected[name] = why; trace.push(`reject ${name}: ${why}`); } else feasible.push(m);
    }
    const ranked = this.rank(decision, feasible, s);
    const chosen = ranked.length ? this.models[ranked[0].model] : this.models[FAILSAFE];
    trace.push(ranked.length
      ? `select ${chosen.name} @ ${chosen.location} (score ${ranked[0].score.toFixed(3)})`
      : "no feasible candidate: falling back to safety-controller (hold position)");

    const fallbacks = ranked.slice(1).map((c) => c.model);
    if (chosen.name !== FAILSAFE) fallbacks.push(FAILSAFE);
    let guardrails = {};
    if (s.safety === "elevated") {
      const { min_safety_tier, ...rest } = this.config.guardrails?.elevated || {};
      guardrails = rest;
    } else if (s.safety === "critical") guardrails = { action: "stop", hold_position: true };

    return {
      decision: decision.name, algorithm: decision.algorithm, model: chosen.name, location: chosen.location,
      est_latency_ms: chosen.p50_latency_ms, description: chosen.description, kind: chosen.kind,
      signals: s, guardrails, ranked, fallbacks, rejected, trace,
    };
  }
}

// Match Python's `:g` formatting for the numbers that appear in reasons.
function fmt(n) {
  return Number.isInteger(n) ? String(n) : String(Number(n.toPrecision(6)));
}
