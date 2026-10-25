import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from solo_router import RobotRequest, Router, load_config
from solo_router.config import ConfigError, RouterConfig, default_config_path
from solo_router.server import App, make_handler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def router():
    return Router(load_config(default_config_path()))


def route(router, instruction, **kw):
    return router.route(RobotRequest(instruction=instruction, **kw))


@pytest.mark.parametrize("text", ["stop", "STOP!", "emergency, the arm is stuck", "e-stop", "stop moving"])
def test_stop_commands_preempt_everything(router, text):
    r = route(router, text, embodiment="arm", observations=["rgb"])
    assert r.decision == "emergency_stop"
    assert r.model == "safety-controller"
    assert r.guardrails == {"action": "stop", "hold_position": True}


def test_estop_flag_overrides_instruction(router):
    r = route(router, "pick up the cup", estop=True, embodiment="arm", observations=["rgb"])
    assert r.model == "safety-controller"


def test_manipulation_picks_best_vla_for_embodiment(router):
    assert route(router, "pick up the red cup", embodiment="arm", observations=["rgb"], control_hz=20).model == "pi0-3b"
    assert route(router, "fold the towel", embodiment="humanoid", observations=["rgb"], control_hz=20).model == "gr00t-n1.5-3b"


def test_action_chunk_extends_latency_budget(router):
    # 30 Hz is a 33 ms step, but GR00T predicts 16 steps per call.
    r = route(router, "fold the towel", embodiment="humanoid", observations=["rgb"], control_hz=30)
    assert r.model == "gr00t-n1.5-3b"
    r = route(router, "fold the towel", embodiment="humanoid", observations=["rgb"], control_hz=500)
    assert r.rejected["gr00t-n1.5-3b"].startswith("p50 40ms exceeds 16-step chunk budget")


def test_specific_robot_inherits_family_policies(router):
    r = route(router, "put the red cube in the bowl", embodiment="so101", observations=["rgb"], control_hz=30)
    assert r.model == "pi0-3b"
    assert "act-so101" in r.fallbacks and "gr00t-n1.5-3b" in r.rejected
    r = route(router, "pick up the cup", embodiment="koch", observations=["rgb"], control_hz=30)
    assert "act-so101" in r.rejected


def test_overrides_cannot_lower_safety(router):
    r = router.route(RobotRequest(instruction="stop", overrides={"intent": "manipulation", "safety": "normal"}))
    assert r.model == "safety-controller"


def test_task_pipeline_routes_each_role(router):
    task = router.route_task(RobotRequest(instruction="put the red cube in the bowl", embodiment="so101",
                                          observations=["rgb"], control_hz=30))
    roles = {stage: router.config.models[r.model].role for stage, r in task.stages}
    assert roles == {"perceive": "perception", "reason": "reasoning", "act": "action"}
    # The control-rate deadline only binds the stage that moves the robot.
    assert task.stages[0][1].signals["latency_budget_ms"] is None


def test_task_pipeline_offline_stays_local(router):
    task = router.route_task(RobotRequest(instruction="put the red cube in the bowl", embodiment="so101",
                                          observations=["rgb"], control_hz=30, network="offline"))
    assert all(r.location != "cloud" for _, r in task.stages)


def test_browser_models_need_browser_location_and_runtime(router):
    base = dict(instruction="put the red cube in the bowl", embodiment="so101", observations=["rgb"], control_hz=30)
    # By default the browser is not a reachable location.
    r = router.route(RobotRequest(**base, overrides={"intent": "perception"}))
    assert r.rejected["florence-2-base-web"] == "browser not reachable"
    # A browser client with WebGPU gets the ONNX models for every stage.
    task = router.route_task(RobotRequest(**base, locations=["browser"], client_runtimes=["webgpu", "wasm"]))
    assert [r.model for _, r in task.stages] == ["florence-2-base-web", "qwen2.5-0.5b-web", "so101-ik-web"]
    # Without a runtime the ONNX models are rejected and the stage fails safe.
    task = router.route_task(RobotRequest(**base, locations=["browser"], client_runtimes=[]))
    perceive = task.stages[0][1]
    assert perceive.model == "safety-controller"
    assert perceive.rejected["florence-2-base-web"] == "needs runtime webgpu or wasm"


def test_browser_models_declare_how_to_load(router):
    for m in router.config.models.values():
        if m.location == "browser" and m.kind != "controller":
            assert m.onnx and m.onnx["hf_id"] and m.runtimes


def test_elevated_safety_only_restricts_action_models(router):
    r = route(router, "is there a knife on the table", observations=["rgb"])
    assert r.decision == "scene_understanding" and r.signals["safety"] == "elevated"
    assert r.location != "cloud"


def test_control_rate_filters_slow_policies(router):
    r = route(router, "pick up the cup", embodiment="arm", observations=["rgb"], control_hz=10, latency_budget_ms=300)
    assert "openvla-7b" in r.rejected
    assert r.rejected["openvla-7b"] == "max 6Hz below control rate 10Hz"


def test_vla_without_camera_is_rejected(router):
    r = route(router, "pick up the cup", embodiment="arm")
    assert r.model == "safety-controller"
    assert r.rejected["pi0-3b"] == "needs rgb observation"


def test_wrist_camera_satisfies_rgb(router):
    assert route(router, "grasp the block", embodiment="arm", observations=["wrist_rgb"]).model == "pi0-3b"


def test_humans_nearby_elevates_and_restricts_tier(router):
    r = route(router, "hand me the cup", embodiment="arm", observations=["rgb"], humans_nearby=True)
    assert r.signals["safety"] == "elevated"
    assert r.guardrails["max_speed_mps"] == 0.25
    assert "smolvla-450m" in r.rejected


def test_navigation_stays_onboard(router):
    r = route(router, "go to the kitchen", embodiment="quadruped", control_hz=20)
    assert (r.decision, r.location) == ("navigation", "onboard")


def test_long_horizon_goes_to_cloud_planner_when_online(router):
    r = route(router, "clean the kitchen, then load the dishwasher and finally wipe the counters")
    assert r.decision == "long_horizon_task"
    assert r.location == "cloud"
    r = route(router, "clean the kitchen, then load the dishwasher", preference="quality")
    assert r.model == "anthropic/claude-sonnet-5-5"


def test_offline_never_routes_to_cloud(router):
    r = route(router, "clean the kitchen, then load the dishwasher", network="offline")
    assert r.location != "cloud"
    assert r.rejected["anthropic/claude-sonnet-5-5"] == "robot is offline"


def test_degraded_network_prefers_edge(router):
    r = route(router, "set the table for four, then serve water", network="degraded")
    assert r.model == "qwen3-8b"


def test_tight_budget_excludes_cloud(router):
    r = route(router, "what do you see on the shelf", observations=["rgb"], latency_budget_ms=400)
    assert r.location != "cloud"
    assert "anthropic/claude-sonnet-5-5" in r.rejected


def test_conversation_and_diagnostics_stay_local(router):
    assert route(router, "hello, who are you").model == "smollm2-1.7b"
    assert route(router, "why is joint 3 overheating").decision == "diagnostics"


def test_fallback_chain_ends_in_safety_controller(router):
    r = route(router, "pick up the red cup", embodiment="arm", observations=["rgb"])
    assert r.fallbacks[-1] == "safety-controller"


def test_config_rejects_unknown_candidates():
    with pytest.raises(ConfigError):
        RouterConfig.from_dict({
            "models": [{"name": "safety-controller", "kind": "safety", "role": "safety", "location": "onboard",
                        "p50_latency_ms": 1, "quality": 1}],
            "decisions": [{"name": "x", "candidates": ["nope"]}],
        })


def test_config_reports_missing_model_fields():
    with pytest.raises(ConfigError, match="role"):
        RouterConfig.from_dict({"models": [{"name": "x", "kind": "vla", "location": "edge",
                                            "p50_latency_ms": 1, "quality": 1}]})


def test_openai_body_with_image_adds_rgb():
    req = RobotRequest.from_openai({
        "model": "solo/auto",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "pick up the cup"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
        "robot": {"embodiment": "arm"},
    })
    assert req.instruction == "pick up the cup"
    assert req.observations == ["rgb"] and req.embodiment == "arm"


def test_website_playground_config_in_sync():
    exported = json.loads((ROOT / "website" / "assets" / "router-config.json").read_text())
    assert exported == load_config(default_config_path()).raw, \
        "run: solo-router export > website/assets/router-config.json"


def test_http_server_end_to_end(router):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App(router, dry_run=True)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        body = json.dumps({"model": "solo/auto", "messages": [{"role": "user", "content": "stop"}]}).encode()
        req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.headers["x-solo-router-model"] == "safety-controller"
            payload = json.loads(resp.read())
        assert payload["solo_router"]["decision"] == "emergency_stop"
        with urllib.request.urlopen(base + "/v1/models") as resp:
            assert json.loads(resp.read())["data"][0]["id"] == "solo/auto"
        body = json.dumps({"instruction": "put the red cube in the bowl", "embodiment": "so101",
                           "observations": ["rgb"], "control_hz": 30}).encode()
        with urllib.request.urlopen(urllib.request.Request(base + "/v1/task", data=body)) as resp:
            assert resp.headers["x-solo-router-act"] == "pi0-3b"
            assert [s["stage"] for s in json.loads(resp.read())["stages"]] == ["perceive", "reason", "act"]
    finally:
        httpd.shutdown()


CHORE = "clean the kitchen, then load the dishwasher and finally wipe the counters"


@pytest.mark.parametrize("provider, model", [
    ("openai", "openai/gpt-5.5"), ("anthropic", "anthropic/claude-sonnet-5-5"), ("nebius", "nebius/gpt-oss-120b"),
])
def test_one_provider_key_routes_to_that_provider(router, provider, model):
    r = route(router, CHORE, providers=[provider])
    assert r.model == model and r.provider == provider
    others = {m.name for m in router.config.models.values() if m.provider and m.provider != provider}
    assert all(r.rejected[o] == f"no {router.config.models[o].provider} API key" for o in others)


def test_no_provider_keys_stays_self_hosted(router):
    r = route(router, CHORE, providers=[])
    assert r.model == "qwen3-8b" and r.provider is None


def test_providers_fill_model_endpoints(router):
    m = router.config.models["nebius/gpt-oss-120b"]
    assert m.base_url == "https://api.tokenfactory.nebius.com/v1" and m.api_format == "openai"
    assert m.api_key_env == "NEBIUS_API_KEY" and m.upstream_id == "openai/gpt-oss-120b"
    assert router.config.models["anthropic/claude-sonnet-5-5"].upstream_id == "claude-sonnet-5-5"


def test_config_rejects_unknown_provider():
    with pytest.raises(ConfigError, match="unknown provider"):
        RouterConfig.from_dict({"models": [{"name": "x", "kind": "llm", "role": "reasoning", "location": "cloud",
                                            "p50_latency_ms": 1, "quality": 1, "provider": "nope"}]})


def test_openai_body_translates_to_anthropic():
    from solo_router.server import _from_anthropic, _to_anthropic

    body = _to_anthropic({"messages": [
        {"role": "system", "content": "plan for a robot"},
        {"role": "user", "content": [{"type": "text", "text": "what is this"},
                                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
    ], "temperature": 0}, "claude-sonnet-5-5")
    assert body["system"] == "plan for a robot" and body["max_tokens"] == 1024 and body["temperature"] == 0
    assert body["messages"][0]["content"][1]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}
    out = _from_anthropic({"id": "msg_1", "content": [{"type": "text", "text": "a cube"}], "stop_reason": "end_turn",
                           "usage": {"input_tokens": 3, "output_tokens": 2}})
    assert out["choices"][0]["message"]["content"] == "a cube" and out["usage"]["total_tokens"] == 5


def _serve(handler_cls):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_proxy_uses_provider_key_and_falls_back(monkeypatch):
    from http.server import BaseHTTPRequestHandler

    seen = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append((self.path, body.get("model"), self.headers.get("authorization")))
            if body.get("model") == "gpt-5.5":
                self.send_response(500)
                self.end_headers()
                return
            data = json.dumps({"id": "x", "object": "chat.completion", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "plan"}}]}).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    up, up_url = _serve(Upstream)
    data = load_config(default_config_path()).raw
    data = {**data, "providers": {k: {**v, "base_url": up_url} for k, v in data["providers"].items()}}
    monkeypatch.setenv("SOLO_ROUTER_QUIET", "1")
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NEBIUS_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    app = App(Router(RouterConfig.from_dict(data)), dry_run=False, keys={"openai": "sk-test"})
    httpd, base = _serve(make_handler(app))
    try:
        body = json.dumps({"model": "solo/auto", "messages": [{"role": "user", "content": CHORE}]}).encode()
        req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                     headers={"x-nebius-api-key": "nb-test"})
        with urllib.request.urlopen(req) as resp:
            payload = json.loads(resp.read())
            served = resp.headers["x-solo-router-model"]
        # Only keyed providers are routable; Nebius ranks first but its request comes from the header.
        assert payload["solo_router"]["model"] == "nebius/gpt-oss-120b"
        assert served == "nebius/gpt-oss-120b"
        assert seen[0] == ("/chat/completions", "openai/gpt-oss-120b", "Bearer nb-test")
        assert "anthropic/claude-sonnet-5-5" in payload["solo_router"]["rejected"]

        # Without the Nebius header, OpenAI is chosen; it fails, so the chain falls back through
        # the self-hosted models (unreachable here) and reports every attempt.
        seen.clear()
        req = urllib.request.Request(base + "/v1/chat/completions", data=body)
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req)
        payload = json.loads(err.value.read())
        assert seen[0] == ("/chat/completions", "gpt-5.5", "Bearer sk-test")
        assert [a["model"] for a in payload["error"]["attempts"]][:2] == ["openai/gpt-5.5", "qwen3-8b"]

        with urllib.request.urlopen(base + "/v1/providers") as resp:
            provs = {p["id"]: p["configured"] for p in json.loads(resp.read())["data"]}
        assert provs == {"openai": True, "anthropic": False, "nebius": False}
    finally:
        httpd.shutdown()
        up.shutdown()


def _read_sse(resp):
    events = [line[6:] for line in resp.read().decode().splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


def _text(chunks):
    return "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))


class _StreamingUpstream:
    """Fake OpenAI and Anthropic endpoints that stream, or fail for models in `fail`."""

    def __init__(self, fail=()):
        from http.server import BaseHTTPRequestHandler

        calls = self.calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                calls.append((self.path, body["model"], body.get("stream")))
                if body["model"] in fail:
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                if self.path == "/messages":
                    events = [{"type": "message_start"},
                              {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "pick "}},
                              {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "up"}},
                              {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
                              {"type": "message_stop"}]
                    out = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
                else:
                    deltas = [{"role": "assistant", "content": ""}, {"content": "1. "}, {"content": "wipe"}]
                    out = "".join(f"data: {json.dumps({'choices': [{'index': 0, 'delta': d}]})}\n\n" for d in deltas)
                    out += "data: [DONE]\n\n"
                self.wfile.write(out.encode())

            def log_message(self, *a):
                pass

        self.httpd, self.url = _serve(Handler)


@pytest.fixture
def stream_app(monkeypatch):
    def make(fail=(), keys=None, dry_run=False):
        upstream = _StreamingUpstream(fail)
        data = load_config(default_config_path()).raw
        data = {**data, "providers": {k: {**v, "base_url": upstream.url} for k, v in data["providers"].items()}}
        monkeypatch.setenv("SOLO_ROUTER_QUIET", "1")
        for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NEBIUS_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        httpd, base = _serve(make_handler(App(Router(RouterConfig.from_dict(data)), dry_run=dry_run, keys=keys)))
        servers.extend([httpd, upstream.httpd])
        return upstream, base

    servers = []
    yield make
    for s in servers:
        s.shutdown()


def _stream_chat(base, text, **extra):
    body = json.dumps({"model": "solo/auto", "stream": True, "messages": [{"role": "user", "content": text}],
                       **extra}).encode()
    return urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=body))


def test_proxy_streams_openai_compatible(stream_app):
    upstream, base = stream_app(keys={"nebius": "nb"})
    with _stream_chat(base, CHORE) as resp:
        assert resp.headers["content-type"] == "text/event-stream"
        assert resp.headers["x-solo-router-model"] == "nebius/gpt-oss-120b"
        chunks = _read_sse(resp)
    assert upstream.calls == [("/chat/completions", "openai/gpt-oss-120b", True)]
    assert _text(chunks) == "1. wipe"
    assert all(c["model"] == "nebius/gpt-oss-120b" for c in chunks)
    assert chunks[-1]["solo_router"]["served_by"] == "nebius/gpt-oss-120b"


def test_proxy_translates_anthropic_stream(stream_app):
    _, base = stream_app(keys={"anthropic": "ak"})
    with _stream_chat(base, CHORE) as resp:
        chunks = _read_sse(resp)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert _text(chunks) == "pick up"
    assert [c["choices"][0]["finish_reason"] for c in chunks if c.get("choices")][-1] == "stop"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_stream_falls_back_before_first_byte(stream_app):
    upstream, base = stream_app(fail={"openai/gpt-oss-120b"}, keys={"nebius": "nb", "anthropic": "ak"})
    with _stream_chat(base, CHORE) as resp:
        assert resp.headers["x-solo-router-model"] == "anthropic/claude-sonnet-5-5"
        chunks = _read_sse(resp)
    assert [c[1] for c in upstream.calls] == ["openai/gpt-oss-120b", "claude-sonnet-5-5"]
    assert _text(chunks) == "pick up"
    assert chunks[-1]["solo_router"]["attempts"][0]["model"] == "nebius/gpt-oss-120b"


def test_dry_run_streams_the_decision(stream_app):
    upstream, base = stream_app(dry_run=True)
    with _stream_chat(base, "stop") as resp:
        chunks = _read_sse(resp)
    assert upstream.calls == []
    assert "emergency_stop -> safety-controller" in _text(chunks)


def test_stream_error_midway_becomes_error_event():
    from solo_router.server import _sse

    def broken():
        yield {"choices": [{"index": 0, "delta": {"content": "par"}}]}
        raise ValueError("connection reset")

    events = [e.decode() for e in _sse(broken(), {"model": "m"})]
    assert "stream interrupted: connection reset" in events[-2] and events[-1] == "data: [DONE]\n\n"


def test_openai_gets_max_completion_tokens(router):
    app = App(router, dry_run=True)
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50, "robot": {}}
    _, payload, _ = app._upstream(router.config.models["openai/gpt-5.5"], body, "k", stream=True)
    assert payload["max_completion_tokens"] == 50 and "max_tokens" not in payload and payload["stream"] is True
    _, payload, _ = app._upstream(router.config.models["nebius/gpt-oss-120b"], body, "k", stream=False)
    assert payload["max_tokens"] == 50 and "robot" not in payload
