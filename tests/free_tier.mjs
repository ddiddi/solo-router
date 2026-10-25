// Runs free-tier/worker.js with an in-memory store and a stubbed provider, prints results as JSON.
const worker = (await import("../free-tier/worker.js")).default;
const store = new Map();
const env = {
  RUNS: { get: async (k) => store.get(k) ?? null, put: async (k, v) => void store.set(k, v) },
  ALLOWED_ORIGINS: "https://site.example", RUNS_PER_DAY: "1", DAILY_CAP: "2", MAX_TOKENS: "100",
  NEBIUS_API_KEY: "nb-secret", ANTHROPIC_API_KEY: "ak-secret",
};
const upstream = [];
globalThis.fetch = async (url, opts) => {
  upstream.push({ url, headers: opts.headers, body: JSON.parse(opts.body) });
  return new Response("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n", { status: 200 });
};
const msgs = [{ role: "user", content: "plan the chore" }];
const call = async (path, { method = "GET", body, ip = "1.1.1.1", origin = "https://site.example" } = {}) => {
  const res = await worker.fetch(new Request("https://free.example" + path, {
    method, body: body && JSON.stringify(body),
    headers: { "content-type": "application/json", "cf-connecting-ip": ip, ...(origin ? { origin } : {}) },
  }), env);
  return { status: res.status, format: res.headers.get("x-solo-format"), remaining: res.headers.get("x-solo-remaining"),
           body: await res.text() };
};
const out = {
  status: await call("/status"),
  wrongOrigin: await call("/status", { origin: "https://elsewhere.example" }),
  selfHosted: await call("/run", { method: "POST", body: { model: "qwen3-8b", messages: msgs } }),
  noSecret: await call("/run", { method: "POST", body: { model: "openai/gpt-5.5", messages: msgs } }),
  tooLong: await call("/run", { method: "POST", body: { model: "nebius/gpt-oss-120b", messages: [{ role: "user", content: "x".repeat(5000) }] } }),
  first: await call("/run", { method: "POST", body: { model: "nebius/gpt-oss-120b", messages: msgs } }),
  second: await call("/run", { method: "POST", body: { model: "nebius/gpt-oss-120b", messages: msgs } }),
  otherVisitor: await call("/run", { method: "POST", ip: "2.2.2.2", body: { model: "anthropic/claude-sonnet-5-5", messages: msgs } }),
  capped: await call("/run", { method: "POST", ip: "3.3.3.3", body: { model: "nebius/gpt-oss-120b", messages: msgs } }),
  upstream,
  storedKeys: [...store.keys()],
};
console.log(JSON.stringify(out));
