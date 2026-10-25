// Bring your own key: the hosted providers in router-config.json (OpenAI, Anthropic,
// Nebius Token Factory). Keys stay in this browser and go straight to the provider;
// nothing passes through a Solo Router server.

const STORE = "solo-router-keys";

function storage(remember) {
  try { return remember ? localStorage : sessionStorage; } catch { return null; }
}

export function loadKeys() {
  for (const s of [storage(true), storage(false)]) {
    try {
      const raw = s?.getItem(STORE);
      if (raw) return { keys: JSON.parse(raw), remember: s === storage(true) };
    } catch { /* blocked or corrupt storage */ }
  }
  return { keys: {}, remember: false };
}

export function saveKeys(keys, remember) {
  const clean = Object.fromEntries(Object.entries(keys).filter(([, v]) => v));
  for (const s of [storage(true), storage(false)]) {
    try { s?.removeItem(STORE); } catch { /* ignore */ }
  }
  try { if (Object.keys(clean).length) storage(remember)?.setItem(STORE, JSON.stringify(clean)); } catch { /* ignore */ }
  return clean;
}

// Providers to route among: the ones with a key, or every provider when none is set
// (so the playground still shows what the router would pick).
export const enabledProviders = (keys) => (Object.keys(keys).length ? Object.keys(keys) : null);

const upstreamId = (spec) => spec.upstream_model
  || (spec.name.startsWith(spec.provider + "/") ? spec.name.slice(spec.provider.length + 1) : spec.name);

/** One chat completion against the model's provider. Returns the reply text. */
export async function complete(config, spec, messages, key, maxTokens = 512) {
  const p = config.providers[spec.provider];
  const base = p.base_url.replace(/\/$/, "");
  let res;
  try {
    if (p.api_format === "anthropic") {
      const system = messages.filter((m) => m.role === "system").map((m) => m.content).join("\n\n");
      res = await fetch(`${base}/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01",
          "anthropic-dangerous-direct-browser-access": "true",
        },
        body: JSON.stringify({
          model: upstreamId(spec), max_tokens: maxTokens, ...(system ? { system } : {}),
          messages: messages.filter((m) => m.role !== "system"),
        }),
      });
    } else {
      res = await fetch(`${base}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
        body: JSON.stringify({ model: upstreamId(spec), messages }),
      });
    }
  } catch {
    throw new Error(`${p.label} blocked the browser request. Use \`solo-router run\` from a terminal instead.`);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`${p.label} ${res.status}: ${data.error?.message || res.statusText}`);
  return p.api_format === "anthropic"
    ? (data.content || []).filter((b) => b.type === "text").map((b) => b.text).join("")
    : data.choices?.[0]?.message?.content ?? "";
}
