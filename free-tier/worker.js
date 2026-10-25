// Free runs for website visitors: a Cloudflare Worker that holds provider keys as
// secrets (OPENAI_API_KEY, ANTHROPIC_API_KEY, NEBIUS_API_KEY) so they never ship in
// the static site. Each visitor gets RUNS_PER_DAY runs, all visitors together get
// DAILY_CAP, and only models in the router config can be called. After that,
// visitors bring their own keys.
//
//   GET  /status  -> { providers: [...], remaining: n }
//   POST /run     -> { model, messages } streamed back as the provider's own SSE,
//                    with x-solo-format: openai | anthropic

import config from "../website/assets/router-config.json" with { type: "json" };

const MAX_MESSAGES = 8;
const MAX_CHARS = 4000;

const secretFor = (env, provider) => env[config.providers[provider]?.api_key_env];
const freeProviders = (env) => Object.keys(config.providers).filter((p) => secretFor(env, p));
const today = () => new Date().toISOString().slice(0, 10);

async function visitorId(request) {
  // Hash the address so the store never holds raw IPs.
  const ip = request.headers.get("cf-connecting-ip") || "unknown";
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`${today()}:${ip}`));
  return [...new Uint8Array(digest).slice(0, 12)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function counts(env, request) {
  const visitor = `v:${today()}:${await visitorId(request)}`;
  const total = `t:${today()}`;
  const [used, all] = await Promise.all([env.RUNS.get(visitor), env.RUNS.get(total)]);
  return { visitor, total, used: Number(used || 0), all: Number(all || 0) };
}

function remaining(env, c) {
  const perVisitor = Number(env.RUNS_PER_DAY || 1) - c.used;
  const overall = Number(env.DAILY_CAP || 200) - c.all;
  return Math.max(0, Math.min(perVisitor, overall));
}

function cors(env, request) {
  const origin = request.headers.get("origin") || "";
  const allowed = (env.ALLOWED_ORIGINS || "").split(",").map((s) => s.trim()).filter(Boolean);
  return {
    ok: allowed.includes(origin),
    headers: {
      "access-control-allow-origin": allowed.includes(origin) ? origin : allowed[0] || "",
      "access-control-allow-methods": "GET, POST, OPTIONS",
      "access-control-allow-headers": "content-type",
      "access-control-expose-headers": "x-solo-format, x-solo-remaining",
      vary: "origin",
    },
  };
}

const json = (body, status, headers) =>
  new Response(JSON.stringify(body), { status, headers: { ...headers, "content-type": "application/json" } });

function validMessages(messages) {
  if (!Array.isArray(messages) || !messages.length || messages.length > MAX_MESSAGES) return false;
  let chars = 0;
  for (const m of messages) {
    if (!m || !["system", "user", "assistant"].includes(m.role) || typeof m.content !== "string") return false;
    chars += m.content.length;
  }
  return chars <= MAX_CHARS;
}

function upstream(env, spec, messages) {
  const p = config.providers[spec.provider];
  const key = secretFor(env, spec.provider);
  const model = spec.upstream_model || spec.name.slice(spec.provider.length + 1);
  const maxTokens = Number(env.MAX_TOKENS || 600);
  const base = p.base_url.replace(/\/$/, "");
  if (p.api_format === "anthropic") {
    const system = messages.filter((m) => m.role === "system").map((m) => m.content).join("\n\n");
    return fetch(`${base}/messages`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01" },
      body: JSON.stringify({
        model, max_tokens: maxTokens, stream: true, ...(system ? { system } : {}),
        messages: messages.filter((m) => m.role !== "system"),
      }),
    });
  }
  return fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
    body: JSON.stringify({ model, messages, stream: true, max_completion_tokens: maxTokens }),
  });
}

export default {
  async fetch(request, env) {
    const { ok, headers } = cors(env, request);
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers });
    if (!ok) return json({ error: { message: "free runs are only available on the Solo Router website" } }, 403, headers);
    const path = new URL(request.url).pathname;

    if (request.method === "GET" && path === "/status") {
      const c = await counts(env, request);
      return json({ providers: freeProviders(env), remaining: remaining(env, c) }, 200, headers);
    }
    if (request.method !== "POST" || path !== "/run") return json({ error: { message: "not found" } }, 404, headers);

    const body = await request.json().catch(() => null);
    const spec = config.models.find((m) => m.name === body?.model);
    if (!spec?.provider || !secretFor(env, spec.provider)) {
      return json({ error: { message: `no free runs for ${body?.model}` } }, 400, headers);
    }
    if (!validMessages(body.messages)) {
      return json({ error: { message: `up to ${MAX_MESSAGES} text messages, ${MAX_CHARS} characters in all` } }, 400, headers);
    }
    const c = await counts(env, request);
    const left = remaining(env, c);
    if (left <= 0) {
      return json({ error: { message: "You've used today's free run. Add your own API key for more." } }, 429, headers);
    }
    // Count the run before calling out. KV is not atomic, so a burst of parallel requests
    // can slip a few extra runs through; DAILY_CAP bounds the total either way.
    await Promise.all([
      env.RUNS.put(c.visitor, String(c.used + 1), { expirationTtl: 172800 }),
      env.RUNS.put(c.total, String(c.all + 1), { expirationTtl: 172800 }),
    ]);

    const res = await upstream(env, spec, body.messages);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      return json({ error: { message: `${config.providers[spec.provider].label} ${res.status}: ${err.error?.message || res.statusText}` } },
        502, headers);
    }
    return new Response(res.body, {
      status: 200,
      headers: {
        ...headers, "content-type": "text/event-stream", "cache-control": "no-cache",
        "x-solo-format": config.providers[spec.provider].api_format, "x-solo-remaining": String(left - 1),
      },
    });
  },
};
