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

// Yield the JSON payload of each `data:` line in a Server-Sent Events response.
async function* sseData(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") return;
      if (data) yield JSON.parse(data);
    }
  }
}

/**
 * One chat completion against the model's provider. Returns the reply text.
 * With onToken, the reply streams and onToken gets each piece of text as it arrives.
 */
export async function complete(config, spec, messages, key, maxTokens = 512, onToken = null) {
  const stream = typeof onToken === "function";
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
          messages: messages.filter((m) => m.role !== "system"), stream,
        }),
      });
    } else {
      res = await fetch(`${base}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
        body: JSON.stringify({ model: upstreamId(spec), messages, stream }),
      });
    }
  } catch {
    throw new Error(`${p.label} blocked the browser request. Use \`solo-router run\` from a terminal instead.`);
  }
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(`${p.label} ${res.status}: ${data.error?.message || res.statusText}`);
  }
  if (stream) {
    let text = "";
    for await (const ev of sseData(res)) {
      if (ev.error || ev.type === "error") throw new Error(`${p.label}: ${ev.error?.message || "stream error"}`);
      const piece = p.api_format === "anthropic"
        ? (ev.type === "content_block_delta" && ev.delta?.type === "text_delta" ? ev.delta.text : "")
        : ev.choices?.[0]?.delta?.content || "";
      if (piece) { text += piece; onToken(piece); }
    }
    return text;
  }
  const data = await res.json().catch(() => ({}));
  return p.api_format === "anthropic"
    ? (data.content || []).filter((b) => b.type === "text").map((b) => b.text).join("")
    : data.choices?.[0]?.message?.content ?? "";
}
