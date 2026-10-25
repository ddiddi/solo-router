<div align="center">

# solo router

**Every action. The right policy.**

An open, programmable decision layer for robots and physical AI.

[Website](https://ddiddi.github.io/solo-router/) ·
[SO-101 demo](https://ddiddi.github.io/solo-router/demo.html) ·
[Quick start](#quick-start) ·
[Providers & keys](#providers-and-api-keys) ·
[Configuration](#configuration) ·
[HTTP API](#http-api)

</div>

---

```bash
pip install git+https://github.com/ddiddi/solo-router
export ANTHROPIC_API_KEY=sk-ant-...      # or OPENAI_API_KEY, or NEBIUS_API_KEY
solo-router run "plan how to clear the table, then wipe it"
```

One prompt, one command. The router picks the right model among the providers you have keys for, calls it, and falls back if it fails. On the [website](https://ddiddi.github.io/solo-router/#playground) it is one prompt and one click: paste a key, type, press **Run**.

## About

A robot does not run one model. A fold needs a VLA at 30 Hz, a hallway needs a navigation policy, "what's on the shelf?" needs a VLM, a ten-step chore needs a planner, and "stop!" needs no model at all.

Solo Router sits between your robot stack and those models. For every instruction it extracts **signals**, matches a **decision**, filters candidates against hard limits, and scores the survivors with an **algorithm**. The result is a model, a location (onboard, edge or cloud), a fallback chain that always ends in a fail-safe, and a trace explaining every step.

It applies the signal → decision → algorithm architecture of [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) to robotics.

| Dimension | Fragmented today | With Solo Router |
| --- | --- | --- |
| **Policies** | Hard-coded model per skill | Pick the best VLA / VLM / planner per instruction |
| **Compute** | Onboard, edge GPU and cloud wired separately | One pool, chosen per request |
| **Connectivity** | Robots stall when Wi-Fi drops | Offline never routes to cloud; degraded prefers edge |
| **Control loop** | Slow models silently miss deadlines | Control rate → latency budget → hard filter |
| **Safety** | Spread across the stack | Stop preempts everything; elevated risk tightens limits |

## Default model set: perceive, reason, act

The bundled pool is grouped by **role**. A task runs through all three, and each stage is routed on its own.

| Role | Onboard | Edge | Cloud | Browser (ONNX) |
| --- | --- | --- | --- | --- |
| **Perception** · see | Florence-2-base, SmolVLM2-2.2B | Qwen2.5-VL-7B | `anthropic/claude-sonnet-5-5`, `openai/gpt-5.5` | Florence-2-base on WebGPU / WASM |
| **Reasoning** · think | SmolLM2-1.7B | Qwen3-8B | `anthropic/claude-sonnet-5-5`, `openai/gpt-5.5`, `nebius/gpt-oss-120b` | Qwen2.5-0.5B on WebGPU / WASM |
| **Action** · move | ACT (SO-101), SmolVLA, nav policy | π0, GR00T N1.5, OpenVLA | — | scripted IK controller (SO-101 sim) |
| **Safety** · stop | `safety-controller` (deterministic) | | | |

```bash
solo-router models                     # the pool, grouped by role
solo-router task "put the red cube in the bowl" -e so101 -o rgb,proprio --hz 30
# [perceive] scene_understanding -> qwen2.5-vl-7b @ edge (~380ms)
# [reason]   long_horizon_task   -> nebius/gpt-oss-120b @ cloud (~1500ms)
# [act]      realtime_manipulation -> pi0-3b @ edge (~45ms)
```

Specific robots inherit their family's policies (`so101 → arm`, `aloha → bimanual`, `unitree_g1 → humanoid`, …) through `embodiment_families`. Action policies declare `action_chunk`: a chunked policy only has to answer once per chunk, so π0 at 50 steps per call fits a 30 Hz loop that a one-step policy could not.

## Providers and API keys

Cloud models come from hosted providers you bring your own key for, in the spirit of [OpenRouter](https://openrouter.ai): one OpenAI-compatible endpoint, `provider/model` IDs, and automatic fallback across providers.

| Provider | Key | Models | Endpoint |
| --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-5.5` | `https://api.openai.com/v1` |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-5-5` | `https://api.anthropic.com/v1` (Messages API, translated for you) |
| Nebius Token Factory | `NEBIUS_API_KEY` | `nebius/gpt-oss-120b` (→ `openai/gpt-oss-120b`) | `https://api.tokenfactory.nebius.com/v1` |

**Keys decide routing.** A provider's models are only routable when you have its key. With one key, the router uses that provider for cloud work and the robot's own models for everything else. With none, it stays self-hosted. Each rejected model shows up in the trace as `no <provider> API key`.

```bash
solo-router providers                                   # which keys are set, which models they unlock
solo-router route "plan the chore, then tidy up" --providers openai   # route as if you only had an OpenAI key
solo-router run "why is joint 3 overheating"            # route among hosted providers and call the model
solo-router serve --proxy                               # OpenAI-compatible router on :8899 using your env keys
```

Where keys come from:

- **Environment**: the `api_key_env` of each provider (above).
- **Per request**: an `x-openai-api-key`, `x-anthropic-api-key` or `x-nebius-api-key` header on any `/v1/*` call, so one router can serve several users with their own keys.
- **Website**: the playground's *API keys* panel. Keys stay in your browser (session storage, or local storage if you tick *Remember*) and go straight to the provider. Nothing passes through a Solo Router server.

Add a provider or a model in `router.yaml`. Models inherit `base_url`, `api_format` and `api_key_env` from their provider; `upstream_model` sets the provider's ID when it differs from the name:

```yaml
providers:
  nebius:
    label: Nebius Token Factory
    base_url: https://api.tokenfactory.nebius.com/v1
    api_format: openai          # openai | anthropic
    api_key_env: NEBIUS_API_KEY

models:
  - name: nebius/gpt-oss-120b
    provider: nebius
    upstream_model: openai/gpt-oss-120b
    role: reasoning
    kind: planner
    location: cloud
    p50_latency_ms: 1500
    quality: 0.83
    cost_per_1k: 0.4
```

Model IDs and numbers are examples. Check each provider's catalog for the models your account can use, and measure latency yourself.

## Run models in the browser (ONNX + WebGPU)

`browser` is a fourth location. Browser models carry an `onnx` block (Hugging Face repo, task and per-device dtype) and the `runtimes` they need. The router only considers them when the request says the client can run them:

```json
{ "instruction": "put the red cube in the bowl", "embodiment": "so101", "observations": ["rgb"],
  "locations": ["browser"], "client_runtimes": ["webgpu", "wasm"] }
```

[`website/assets/onnx.js`](website/assets/onnx.js) loads those models with [transformers.js](https://github.com/huggingface/transformers.js) (ONNX Runtime Web), uses WebGPU when `navigator.gpu` has an adapter and WASM otherwise, and lets the browser cache the weights. Without a usable runtime, the browser models are rejected and the chain falls back as usual.

## SO-101 demo

[`demo.html`](https://ddiddi.github.io/solo-router/demo.html) (also embedded on the homepage) runs one task, *put the red cube in the bowl*, on a three.js SO-101:

1. **Perceive**: the router picks a perception model; detections appear on the scene.
2. **Reason**: the router picks a planner; the plan appears as skill calls.
3. **Act**: the router picks a policy; the arm executes the plan with inverse kinematics.

**E-STOP** routes `stop` to the safety controller and freezes the arm mid-motion.

**With an API key** (set in the playground), the reason stage calls the routed hosted planner for real and the arm executes its plan. The demo loops, so each model is called once and its plan reused.

**Run in your browser** re-routes the same task with `locations: ["browser"]`. The router then picks Florence-2 (grounding on the rendered camera frame), Qwen2.5-0.5B (writing the plan) and the scripted IK controller. The first two run for real on your GPU. Detected boxes are ray-cast onto the table to set grasp targets, and a grasp only succeeds if a cube is actually between the jaws. The first run downloads about 850 MB of weights for WebGPU while the simulated demo keeps looping; later runs load from the browser cache. The arm, scene and the default-mode models are simulated.

## Quick start

```bash
pip install git+https://github.com/ddiddi/solo-router

export OPENAI_API_KEY=sk-...                 # any one provider key is enough
solo-router run "clean the kitchen, then load the dishwasher"
# [long_horizon_task -> openai/gpt-5.5 @ cloud]
# 1. ...

solo-router route "pick up the red cup" -e arm -o rgb,proprio --hz 20
# realtime_manipulation -> pi0-3b @ edge (~45ms)
# fallbacks: smolvla-450m -> safety-controller

solo-router route "stop!"
# emergency_stop -> safety-controller @ onboard (~2ms)

solo-router route "clean the kitchen, then load the dishwasher" --network offline
# long_horizon_task -> qwen3-8b @ edge (~900ms)

solo-router serve            # dry-run server on :8899
solo-router serve --proxy    # forward to the configured backends with your keys
```

`run` and `route`/`task` take the same robot flags (`-e`, `-o`, `--hz`, `--network`, `--humans`, `--prefer`, `--providers`). `run` routes among hosted providers by default; add `--local` to include self-hosted onboard and edge endpoints.

Python 3.10+, with PyYAML as the only dependency. The server uses only the standard library, so it runs on the robot.

## How it decides

```
instruction + robot state
        │
        ▼
 ┌─────────────┐   intent · horizon · safety · latency tier · modalities
 │   Signals   │   embodiment · network · energy · preference
 └─────┬───────┘
       ▼
 ┌─────────────┐   first match by priority, e.g.
 │  Decisions  │   emergency_stop(1000) > realtime_manipulation(300) > navigation(290)
 └─────┬───────┘   > long_horizon_task(200) > scene_understanding(150) > diagnostics(120) > conversation(0)
       ▼
 ┌─────────────┐   embodiment support · required cameras · offline → no cloud
 │   Filters   │   p50 latency ≤ budget · max Hz ≥ control rate · safety tier when elevated
 │             │   provider key present (openai · anthropic · nebius)
 └─────┬───────┘
       ▼
 ┌─────────────┐   static         – first feasible
 │ Algorithms  │   latency_aware  – highest quality that meets the deadline
 └─────┬───────┘   hybrid         – quality / latency / cost weights, cloud penalty on weak links
       ▼
 model @ onboard | edge | cloud | browser  +  guardrails  +  fallbacks → safety-controller
```

### Signals

| Signal | Source | Values |
| --- | --- | --- |
| `intent` | keyword lexicon | manipulation, navigation, perception, planning, diagnostics, conversation |
| `horizon` | sequencing markers, length | short, long |
| `safety` | stop patterns, hazard words, `humans_nearby`, `estop` | normal, elevated, critical |
| `latency_tier` | `latency_budget_ms` or `1000 / control_hz` | realtime (<50 ms), interactive (<500 ms), deliberative |
| `modalities` | `observations` / image parts in chat | rgb, wrist_rgb, depth, proprio, lidar, … |
| `embodiment` | request | arm, bimanual, mobile_manipulator, humanoid, quadruped, amr, drone, or a robot such as so101 |
| `network` | request | online, degraded, offline |
| `energy` | `battery_pct` | normal, low (offloads away from power-hungry onboard models) |
| `preference` | request or decision default | balanced, quality, latency, cost |

### Safety

- **Critical** (`stop`, `halt`, `emergency`, `e-stop`, or `estop: true`) routes to `safety-controller` at priority 1000 and returns `{"action": "stop", "hold_position": true}`.
- **Elevated** (hazard words such as knife or stove, or humans near a moving robot) attaches guardrails (`max_speed_mps`, `max_force_n`) and drops policies below `min_safety_tier`.
- If no candidate survives the filters, the robot holds position through `safety-controller` instead of guessing.
- Stop matching deliberately errs toward stopping: a bare "stop" always halts the robot.

## Configuration

Everything lives in one YAML file: [`src/solo_router/router.yaml`](src/solo_router/router.yaml). Copy it, edit it, and pass it with `-c`:

```bash
solo-router -c my-fleet.yaml serve --proxy
```

```yaml
models:
  - name: pi0-3b
    kind: vla               # vla | nav | vlm | llm | planner | safety
    location: edge          # onboard | edge | cloud | browser
    p50_latency_ms: 45
    max_hz: 50
    quality: 0.82
    embodiments: [arm, bimanual, mobile_manipulator]
    requires: [rgb]
    safety_tier: 2
    base_url: http://edge.local:8102

decisions:
  - name: realtime_manipulation
    priority: 300
    when: { intent: manipulation, horizon: short }
    candidates: [pi0-3b, gr00t-n1.5-3b, smolvla-450m, openvla-7b]
    algorithm: latency_aware
```

`when` matches any signal by value or list of values, and accepts `any:` / `all:` for composition. An empty `when` matches everything. A model named `safety-controller` is required.

Models also declare a `role` (perception, reasoning, action, safety). Optional fields include `provider`, `upstream_model`, `action_chunk`, `runtimes` and `onnx`. The `pipeline` block defines the stages for `/v1/task`: which signals each stage fixes (`intent`, `horizon`; safety can never be overridden) and whether the control-loop deadline applies.

The bundled pool is an example. Latency and quality numbers are illustrative; measure your own and replace them.

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | OpenAI-compatible. Use `"model": "solo/auto"` and an optional `robot` object |
| `POST` | `/v1/actions` | Robot-native. Routes, then forwards `{instruction, observation, guardrails}` to the policy's `/act` |
| `POST` | `/v1/route` | Decision only, no backend call |
| `POST` | `/v1/task` | Route every pipeline stage (perceive → reason → act); headers `x-solo-router-{stage}` |
| `GET` | `/v1/models` | `solo/auto` plus the configured pool, with `provider` and whether its key is `configured` |
| `GET` | `/v1/providers` | Providers, their key status and the models each unlocks |
| `GET` | `/health` | Liveness |

```bash
curl localhost:8899/v1/chat/completions -d @examples/chat_hand_me_scissors.json -i
# x-solo-router-decision: realtime_manipulation
# x-solo-router-model: pi0-3b
# x-solo-router-location: edge
# x-solo-router-safety: elevated
```

Every response includes a `solo_router` object (or `route` for `/v1/actions`) with signals, ranked candidates, rejected models with reasons, guardrails, fallbacks and the trace.

In dry-run mode (the default) chat returns a summary of the decision instead of calling a model. With `--proxy`, chat goes to the selected model (OpenAI-compatible or Anthropic Messages, translated both ways) and actions go to VLA/nav policy servers. Requests may set `robot.providers` to narrow the providers; in proxy mode the router only routes to providers it holds a key for.

Like OpenRouter, a failed backend falls through to the next model in the chain. The response says which model answered (`x-solo-router-model`, `solo_router.served_by`) and lists failed `attempts`. If every backend fails you get `502` with all attempts; unreachable policies return a `hold` action. Streaming is not supported yet; `stream` is ignored.

## Website

[`website/`](website/) is a static site deployed to GitHub Pages. Its playground runs [`website/assets/router.js`](website/assets/router.js), a browser port of the router, on [`router-config.json`](website/assets/router-config.json). Type a prompt and press **Run**: the router picks a model, and with a key in the *API keys* panel, [`providers.js`](website/assets/providers.js) calls OpenAI, Anthropic or Nebius Token Factory straight from your browser. If a provider blocks browser (CORS) requests, the answer box says so; use `solo-router run` instead. Robot-side picks (VLAs, edge models) are re-routed to the hosted models a browser can reach, the same way `solo-router run` does. After changing the YAML, regenerate it:

```bash
solo-router export > website/assets/router-config.json
```

`tests/test_js_parity.py` routes several hundred requests and task pipelines through both implementations and fails if they disagree.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
python -m http.server -d website 8765   # preview the site and demo
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
