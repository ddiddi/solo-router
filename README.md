<div align="center">

# solo router

**Every action. The right policy.**

An open, programmable decision layer for robots and physical AI.

[Website & Playground](https://ddiddi.github.io/solo-router/) ·
[Quick start](#quick-start) ·
[Configuration](#configuration) ·
[HTTP API](#http-api)

</div>

---

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

## Quick start

```bash
pip install git+https://github.com/ddiddi/solo-router

solo-router route "pick up the red cup" -e arm -o rgb,proprio --hz 20
# realtime_manipulation -> pi0-3b @ edge (~45ms)
# fallbacks: smolvla-450m -> safety-controller

solo-router route "stop!"
# emergency_stop -> safety-controller @ onboard (~2ms)

solo-router route "clean the kitchen, then load the dishwasher" --network offline
# long_horizon_task -> qwen3-8b @ edge (~900ms)

solo-router serve            # dry-run server on :8899
solo-router serve --proxy    # forward to the configured backends
```

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
 └─────┬───────┘
       ▼
 ┌─────────────┐   static         – first feasible
 │ Algorithms  │   latency_aware  – highest quality that meets the deadline
 └─────┬───────┘   hybrid         – quality / latency / cost weights, cloud penalty on weak links
       ▼
 model @ onboard | edge | cloud  +  guardrails  +  fallbacks → safety-controller
```

### Signals

| Signal | Source | Values |
| --- | --- | --- |
| `intent` | keyword lexicon | manipulation, navigation, perception, planning, diagnostics, conversation |
| `horizon` | sequencing markers, length | short, long |
| `safety` | stop patterns, hazard words, `humans_nearby`, `estop` | normal, elevated, critical |
| `latency_tier` | `latency_budget_ms` or `1000 / control_hz` | realtime (<50 ms), interactive (<500 ms), deliberative |
| `modalities` | `observations` / image parts in chat | rgb, wrist_rgb, depth, proprio, lidar, … |
| `embodiment` | request | arm, bimanual, mobile_manipulator, humanoid, quadruped, amr, drone |
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
    location: edge          # onboard | edge | cloud
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

The bundled pool (SmolVLA, π0, OpenVLA, GR00T N1.5, a navigation policy, Qwen2.5-VL, SmolLM2, Qwen3 and a Claude cloud planner) is an example. Latency and quality numbers are illustrative; measure your own and replace them.

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | OpenAI-compatible. Use `"model": "solo/auto"` and an optional `robot` object |
| `POST` | `/v1/actions` | Robot-native. Routes, then forwards `{instruction, observation, guardrails}` to the policy's `/act` |
| `POST` | `/v1/route` | Decision only, no backend call |
| `GET` | `/v1/models` | `solo/auto` plus the configured pool |
| `GET` | `/health` | Liveness |

```bash
curl localhost:8899/v1/chat/completions -d @examples/chat_hand_me_scissors.json -i
# x-solo-router-decision: realtime_manipulation
# x-solo-router-model: pi0-3b
# x-solo-router-location: edge
# x-solo-router-safety: elevated
```

Every response includes a `solo_router` object (or `route` for `/v1/actions`) with signals, ranked candidates, rejected models with reasons, guardrails, fallbacks and the trace.

In dry-run mode (the default) chat returns a summary of the decision instead of calling a model. With `--proxy`, chat is forwarded to `api_format: openai` backends and actions to VLA/nav policy servers. Unreachable backends return `502` with the fallback chain; unreachable policies return a `hold` action.

## Website

[`website/`](website/) is a static site deployed to GitHub Pages. Its playground runs [`website/assets/router.js`](website/assets/router.js), a browser port of the router, on [`router-config.json`](website/assets/router-config.json). After changing the YAML, regenerate it:

```bash
solo-router export > website/assets/router-config.json
```

`tests/test_js_parity.py` routes 360 requests through both implementations and fails if they disagree.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
python -m http.server -d website 8765   # preview the site
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
