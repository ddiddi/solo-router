// Reads routing requests as JSON lines on stdin, prints the JS router's decisions.
import { readFileSync } from "node:fs";
import { Router } from "../website/assets/router.js";

const cfg = JSON.parse(readFileSync(new URL("../website/assets/router-config.json", import.meta.url)));
const router = new Router(cfg);
const { routes: reqs, tasks } = JSON.parse(readFileSync(0, "utf8"));
const pick = (out) => ({ decision: out.decision, model: out.model, rejected: out.rejected, fallbacks: out.fallbacks,
                         guardrails: out.guardrails, trace: out.trace, signals: out.signals });
console.log(JSON.stringify({
  tasks: tasks.map((r) => router.routeTask(r).map((s) => ({ stage: s.stage, ...pick(s) }))),
  routes: reqs.map((r) => {
    return pick(router.route(r));
  }),
}));
