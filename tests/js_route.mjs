// Reads routing requests as JSON lines on stdin, prints the JS router's decisions.
import { readFileSync } from "node:fs";
import { Router } from "../website/assets/router.js";

const cfg = JSON.parse(readFileSync(new URL("../website/assets/router-config.json", import.meta.url)));
const router = new Router(cfg);
const reqs = JSON.parse(readFileSync(0, "utf8"));
console.log(JSON.stringify(reqs.map((r) => {
  const out = router.route(r);
  return { decision: out.decision, model: out.model, rejected: out.rejected, fallbacks: out.fallbacks,
           guardrails: out.guardrails, trace: out.trace, signals: out.signals };
})));
